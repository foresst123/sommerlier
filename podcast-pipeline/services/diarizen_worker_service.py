import json

from services.base_worker_service import WorkerProcessService


class DiarizenWorkerService(WorkerProcessService):
    """Manages the lifecycle of the isolated DiariZen worker process."""

    def __init__(self, python_env_path: str, worker_script_path: str, device_id: int = 1,
                 logger=None, env_name: str = "kaggle", config_path: str = "config.json"):
        super().__init__(
            name="DiariZen",
            python_bin=python_env_path,
            worker_script=worker_script_path,
            extra_args=["--config", config_path, "--env", env_name],
            device_id=device_id,
            logger=logger,
        )

    def request(self, payload: dict, *, response_id=None) -> dict:
        """Read through progress messages until DiariZen returns a result."""
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
                    if self.logger:
                        self.logger.debug(f"[{self.name} worker] {line.strip()}")
                    continue

                if "progress" in response:
                    if self.logger:
                        self.logger.info(response["progress"])
                    continue
                if "segments" in response or "error" in response:
                    return response
                if self.logger:
                    note = response.get("warning") or response.get("status")
                    if note:
                        self.logger.info(f"[{self.name} worker] {note}")
