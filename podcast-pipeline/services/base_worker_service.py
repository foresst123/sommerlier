import collections
import json
import os
import subprocess
import threading
import time
from typing import Callable, List, Optional, Sequence, Union


def foreign_library_paths_removed(ld_library_path, python_bin) -> str:
    """LD_LIBRARY_PATH without entries that live in another environment.

    The launcher points LD_LIBRARY_PATH at the main environment's CUDA libraries
    (cuDNN, cuBLAS, torch/lib), and every worker inherits it. A worker running a
    different torch/CUDA build then loads the main environment's libraries first
    and fails on import. Entries that are not inside some site-packages
    directory (a system CUDA, say) and entries inside the worker's own
    environment are kept.
    """
    if not ld_library_path:
        return ""
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(str(python_bin))))
    kept = []
    for entry in str(ld_library_path).split(":"):
        if not entry:
            continue
        foreign = "site-packages" in entry and not (
            entry == prefix or entry.startswith(prefix + os.sep))
        if not foreign:
            kept.append(entry)
    return ":".join(kept)


class WorkerProcessService:
    """Lifecycle manager for a worker subprocess that speaks line JSON over stdio.

    Workers write diagnostics to stderr continuously. Nothing reads that pipe
    during normal operation, so once the OS buffer (~64KB) fills, the worker
    blocks in ``write`` while the parent blocks reading stdout: a hard deadlock.
    A daemon thread drains stderr for the process's whole life and keeps only the
    tail for error reporting.
    """

    STDERR_TAIL_LINES = 50

    def __init__(
        self,
        name: str,
        python_bin: str,
        worker_script: str,
        extra_args: Optional[List[str]] = None,
        device_id: Optional[Union[int, Sequence[int]]] = None,
        ready_timeout: float = 900.0,
        logger=None,
        isolate_library_path: bool = False,
    ):
        self.name = name
        # True for workers whose interpreter has its own torch/CUDA (vLLM).
        self.isolate_library_path = bool(isolate_library_path)
        self.python_bin = python_bin
        self.worker_script = worker_script
        self.extra_args = list(extra_args or [])
        self.device_id = device_id
        self.ready_timeout = ready_timeout
        self.logger = logger

        self.process = None
        self._stderr_tail = collections.deque(maxlen=self.STDERR_TAIL_LINES)
        self._stderr_thread = None
        self._io_lock = threading.Lock()

    @property
    def cuda_visible_devices(self) -> Optional[str]:
        """Physical CUDA devices exposed to the child, in local-index order."""
        if self.device_id is None:
            return None
        if isinstance(self.device_id, (list, tuple)):
            return ",".join(str(value) for value in self.device_id)
        return str(self.device_id)

    # ------------------------------------------------------------------
    # stderr draining
    # ------------------------------------------------------------------
    def _drain_stderr(self, stream):
        try:
            for line in iter(stream.readline, ""):
                line = line.rstrip("\n")
                if line:
                    self._stderr_tail.append(line)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def stderr_tail(self, lines: int = 10) -> str:
        return " | ".join(list(self._stderr_tail)[-lines:])

    # ------------------------------------------------------------------
    # readiness
    # ------------------------------------------------------------------
    def is_ready_line(self, line: str) -> bool:
        """Whether ``line`` is the worker's ready handshake.

        Handles both protocols in use: a bare marker and a JSON status object.
        Subclasses override for anything stricter.
        """
        stripped = line.strip()
        if not stripped:
            return False
        try:
            msg = json.loads(stripped)
        except Exception:
            return "ready" in stripped.lower()
        if isinstance(msg, dict):
            status = str(msg.get("status", "")).lower()
            if status == "ready":
                return True
            if status == "error":
                raise RuntimeError(f"{self.name} worker reported error: {msg.get('message')}")
        return False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def build_command(self) -> List[str]:
        return [self.python_bin, self.worker_script] + self.extra_args

    def start(self):
        """Spawn and block until ready. Kept for callers that want the old flow."""
        self.spawn()
        self.wait_ready()

    def spawn(self):
        """Launch the subprocess without waiting for its ready handshake.

        Split out of start() so several workers can be launched at once and
        joined afterwards; starting them one at a time serialised ~100s of model
        loading. stderr is drained by a daemon thread from this point on, but
        nothing reads stdout until wait_ready(), so a worker that writes more
        than the pipe buffer (~64KB) before signalling ready would block. All
        current workers stay well under that.
        """
        if self.process is not None:
            return

        # The interpreter may be given as a callable, resolved here rather than
        # at construction. Finding it can fail -- a missing venv raises -- and
        # that failure belongs to the stage that needs the worker, not to the
        # start of a run that may never reach it.
        if callable(self.python_bin):
            self.python_bin = self.python_bin()

        if not os.path.exists(self.python_bin):
            raise FileNotFoundError(
                f"{self.name} worker interpreter not found: {self.python_bin}"
            )

        env = os.environ.copy()
        if self.isolate_library_path and env.get("LD_LIBRARY_PATH"):
            cleaned = foreign_library_paths_removed(env["LD_LIBRARY_PATH"], self.python_bin)
            if cleaned:
                env["LD_LIBRARY_PATH"] = cleaned
            else:
                env.pop("LD_LIBRARY_PATH", None)
        visible_devices = self.cuda_visible_devices
        if visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible_devices
        # tqdm progress from hf_hub_download would otherwise flood stderr before
        # the ready handshake, on top of being useless in a captured pipe.
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        cmd = self.build_command()
        if self.logger:
            target = (f" on CUDA_VISIBLE_DEVICES={visible_devices}"
                      if visible_devices is not None else "")
            self.logger.info(f"Starting {self.name} worker subprocess{target}")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self.process.stderr,), daemon=True
        )
        self._stderr_thread.start()

    def wait_ready(self):
        """Block until the worker signals ready. Safe to call once per spawn."""
        if self.process is None:
            raise RuntimeError(f"{self.name} worker was never spawned")
        self._wait_for_ready()

    def _wait_for_ready(self):
        deadline = time.monotonic() + self.ready_timeout
        last_stdout = collections.deque(maxlen=5)

        while True:
            if time.monotonic() > deadline:
                self._fail_start(
                    f"timed out after {self.ready_timeout:.0f}s waiting for ready",
                    last_stdout,
                )

            if self.process.poll() is not None:
                self._fail_start(
                    f"exited with code {self.process.returncode} before signalling ready",
                    last_stdout,
                )

            line = self.process.stdout.readline()
            if not line:
                self._fail_start("closed stdout before signalling ready", last_stdout)

            stripped = line.strip()
            if not stripped:
                continue
            last_stdout.append(stripped)

            try:
                if self.is_ready_line(stripped):
                    if self.logger:
                        self.logger.info(f"{self.name} worker is ready.")
                    return
            except RuntimeError as e:
                self._fail_start(str(e), last_stdout)

            if self.logger:
                self.logger.debug(f"[{self.name} worker] {stripped}")

    def _fail_start(self, reason: str, last_stdout):
        stdout_log = " | ".join(last_stdout)
        stderr_log = self.stderr_tail(self.STDERR_TAIL_LINES)
        detail = f"{self.name} worker did not start: {reason}."
        if stdout_log:
            detail += f" stdout: {stdout_log}."
        if stderr_log:
            detail += f" stderr: {stderr_log}"
        if self.logger:
            self.logger.error(detail)
        self.stop()
        raise RuntimeError(detail)

    def stop(self):
        if not self.process:
            return

        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
        except Exception:
            pass

        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if self.logger:
                    self.logger.warning(f"{self.name} worker ignored terminate; killing.")
                self.process.kill()
                self.process.wait(timeout=15)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error stopping {self.name} worker: {e}")

        if self._stderr_thread:
            self._stderr_thread.join(timeout=5)
            self._stderr_thread = None

        try:
            if self.process.stdout and not self.process.stdout.closed:
                self.process.stdout.close()
        except Exception:
            pass

        self.process = None
        if self.logger:
            self.logger.info(f"{self.name} worker terminated.")

    def request(self, payload: dict, *, response_id=None) -> dict:
        """Send one JSON request without allowing concurrent pipe corruption.

        A worker owns one stdin/stdout pair.  Multiple scheduler threads may
        share the service, but exactly one request is in flight on that pair.
        A pool obtains parallelism by owning several services/processes.
        """
        with self._io_lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError(f"{self.name} worker is not running")
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError(f"{self.name} worker closed stdout")
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if response_id is None or response.get("id") == response_id:
                    return response
