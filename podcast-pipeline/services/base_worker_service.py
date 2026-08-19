import collections
import json
import os
import subprocess
import threading
import time
from typing import Callable, List, Optional


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
        device_id: Optional[int] = None,
        ready_timeout: float = 900.0,
        logger=None,
    ):
        self.name = name
        self.python_bin = python_bin
        self.worker_script = worker_script
        self.extra_args = list(extra_args or [])
        self.device_id = device_id
        self.ready_timeout = ready_timeout
        self.logger = logger

        self.process = None
        self._stderr_tail = collections.deque(maxlen=self.STDERR_TAIL_LINES)
        self._stderr_thread = None

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
        if self.process is not None:
            return

        if not os.path.exists(self.python_bin):
            raise FileNotFoundError(
                f"{self.name} worker interpreter not found: {self.python_bin}"
            )

        env = os.environ.copy()
        if self.device_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        # tqdm progress from hf_hub_download would otherwise flood stderr before
        # the ready handshake, on top of being useless in a captured pipe.
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        cmd = self.build_command()
        if self.logger:
            target = f" on CUDA_VISIBLE_DEVICES={self.device_id}" if self.device_id is not None else ""
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
        stderr_log = self.stderr_tail(10)
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
