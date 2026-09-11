"""Timeout-aware JSON requests for the optional experiment workers."""

import json
import queue
import threading
import time

from services.base_worker_service import WorkerProcessService


class JsonWorkerService(WorkerProcessService):
    def spawn(self):
        if self.process is not None:
            return
        self._messages = queue.Queue()
        super().spawn()
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stdout_thread.start()

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        self._messages.put(value)
                except ValueError:
                    if self.logger:
                        self.logger.debug(f"[{self.name}] {line.rstrip()}")
        finally:
            self._messages.put(None)

    def _receive(self, timeout):
        try:
            message = self._messages.get(timeout=max(0, timeout))
        except queue.Empty:
            self.stop()
            raise TimeoutError(f"{self.name} response timed out") from None
        if message is None:
            raise RuntimeError(f"{self.name} closed stdout: {self.stderr_tail()}")
        if message.get("status") == "error" or "error" in message:
            raise RuntimeError(f"{self.name}: {message.get('error', message.get('message'))}")
        return message

    def _wait_for_ready(self):
        deadline = time.monotonic() + self.ready_timeout
        while True:
            message = self._receive(deadline - time.monotonic())
            if message.get("status") == "ready":
                return

    def request(self, payload, timeout=900):
        if self.process is None:
            self.start()
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return self._receive(timeout)

    def stop(self):
        super().stop()
        thread = getattr(self, "_stdout_thread", None)
        if thread:
            thread.join(timeout=5)
            self._stdout_thread = None
