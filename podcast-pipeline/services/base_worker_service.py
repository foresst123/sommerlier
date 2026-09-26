import atexit
import collections
import json
import os
import signal
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


# Process groups of workers that are running. A worker gets a group of its own so that
# everything it starts (vLLM's engine process, which holds the GPU memory) can be
# stopped with it; this set lets the exit hook do that for any still alive.
_LIVE_GROUPS = set()
_LIVE_GROUPS_LOCK = threading.Lock()


def _signal_group(group, sig):
    """Send `sig` to every process of `group`; False when none is left."""
    try:
        os.killpg(group, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _group_alive(group):
    return _signal_group(group, 0)


def stop_group(process, grace: float = 15.0, children_grace: float = 10.0):
    """Stop a worker and everything in its process group.

    SIGTERM to the group, wait for the worker itself, then give the processes it started
    a moment to go and SIGKILL whatever is left. The worker is waited on here because an
    unreaped worker would keep the group looking alive.
    """
    group = process.pid
    _signal_group(group, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        _signal_group(group, signal.SIGKILL)
        process.wait(timeout=grace)
    deadline = time.monotonic() + children_grace
    while time.monotonic() < deadline and _group_alive(group):
        time.sleep(0.1)
    _signal_group(group, signal.SIGKILL)


def _stop_live_groups():
    with _LIVE_GROUPS_LOCK:
        groups = list(_LIVE_GROUPS)
    for group in groups:
        _signal_group(group, signal.SIGTERM)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and any(_group_alive(g) for g in groups):
        time.sleep(0.1)
    for group in groups:
        _signal_group(group, signal.SIGKILL)


atexit.register(_stop_live_groups)


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
        self._ready_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._ready = False
        self._ready_error = None

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
                    if self.logger and self.is_error_line(line):
                        self.logger.warning(f"[{self.name} worker stderr] {line}")
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    _ERROR_MARKERS = ("error", "exception", "traceback", "critical", "fatal",
                      "out of memory", "killed", "segmentation")
    # Warnings printed by libraries at import that do not affect a worker: pyannote's
    # torchcodec notice mentions "Error message was", and vLLM logs each optional
    # import it could not do ("Traceback", "AssertionError") from import_utils.py.
    _NOISE_MARKERS = ("torchcodec", "[import_utils.py")

    @classmethod
    def is_error_line(cls, line: str) -> bool:
        """Whether a worker's stderr line reports a problem.

        Everything else (progress bars, vLLM's start-up chatter, stack dumps of
        healthy threads) still lands in the tail kept for a failed start, but is
        not written to the pipeline log."""
        lowered = line.lower()
        if any(marker in lowered for marker in cls._NOISE_MARKERS):
            return False
        return any(marker in lowered for marker in cls._ERROR_MARKERS)

    def stderr_tail(self, lines: int = 10) -> str:
        return " | ".join(list(self._stderr_tail)[-lines:])

    # ------------------------------------------------------------------
    # readiness
    # ------------------------------------------------------------------
    # True for workers whose libraries log to the same stdout (vLLM): only the JSON
    # {"status": "ready"} counts. A bare "line containing ready" matched vLLM's own
    # debug dump of VLLM_ENGINE_READY_TIMEOUT_S and declared the worker ready while
    # its engine was still compiling.
    ready_requires_json = False

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
            return (not self.ready_requires_json) and "ready" in stripped.lower()
        if isinstance(msg, dict):
            status = str(msg.get("status", "")).lower()
            if status == "ready":
                return True
            if status == "error":
                raise RuntimeError(f"{self.name} worker reported error: {msg.get('message')}")
        return False

    def log_ready_details(self, line: str) -> None:
        """Log what a worker chose to say in its ready message.

        stderr only reaches the log for lines that look like errors, so a worker that
        needs a fact recorded on every run (which device it really got) puts it here:
        {"status": "ready", "info": "...", "warning": true}.
        """
        try:
            msg = json.loads(line)
        except Exception:
            return
        if not isinstance(msg, dict) or not msg.get("info"):
            return
        emit = self.logger.warning if msg.get("warning") else self.logger.info
        emit(f"{self.name} worker: {msg['info']}")

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
        self._ready, self._ready_error = False, None

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
            # Its own process group: stop() then reaches the processes the worker
            # starts too. vLLM's engine is one, and it neither exits when its parent
            # is killed nor releases its GPU memory until it does.
            start_new_session=(os.name == "posix"),
        )
        if os.name == "posix":
            with _LIVE_GROUPS_LOCK:
                _LIVE_GROUPS.add(self.process.pid)

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self.process.stderr,), daemon=True
        )
        self._stderr_thread.start()

    def wait_ready(self):
        """Block until the worker signals ready.

        The handshake is read from stdout, so it must happen once per spawn: the
        first caller performs it and every other caller, from any thread, waits for
        that result. A failed start is raised to all of them.
        """
        with self._ready_lock:
            if self._ready_error is not None:
                raise self._ready_error
            if self._ready:
                return
            if self.process is None:
                raise RuntimeError(f"{self.name} worker was never spawned")
            try:
                self._wait_for_ready()
            except Exception as exc:
                self._ready_error = exc
                raise
            self._ready = True

    def _wait_for_ready(self):
        deadline = time.monotonic() + self.ready_timeout
        last_stdout = collections.deque(maxlen=5)
        process = self.process

        while True:
            if time.monotonic() > deadline:
                self._fail_start(
                    f"timed out after {self.ready_timeout:.0f}s waiting for ready",
                    last_stdout,
                )

            if process.poll() is not None:
                self._fail_start(
                    f"exited with code {process.returncode} before signalling ready",
                    last_stdout,
                )

            line = process.stdout.readline()
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
                        self.log_ready_details(stripped)
                    return
            except RuntimeError as e:
                self._fail_start(str(e), last_stdout)

            # Not the handshake: kept in last_stdout for a failed start, not logged.

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
        with self._stop_lock:
            self._stop_unlocked()

    def _stop_unlocked(self):
        if not self.process:
            return

        process = self.process
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        except Exception:
            pass

        try:
            if os.name == "posix":
                # The whole group, from the worker's own pid: also stops a vLLM engine,
                # whether the worker was serving or still loading its model.
                try:
                    stop_group(process)
                except subprocess.TimeoutExpired:
                    if self.logger:
                        self.logger.warning(f"{self.name} worker would not die.")
                with _LIVE_GROUPS_LOCK:
                    _LIVE_GROUPS.discard(process.pid)
            else:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if self.logger:
                        self.logger.warning(f"{self.name} worker ignored terminate; killing.")
                    process.kill()
                    process.wait(timeout=15)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error stopping {self.name} worker: {e}")

        if self._stderr_thread:
            self._stderr_thread.join(timeout=5)
            self._stderr_thread = None

        try:
            if process.stdout and not process.stdout.closed:
                process.stdout.close()
        except Exception:
            pass

        self.process = None
        self._ready, self._ready_error = False, None
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
