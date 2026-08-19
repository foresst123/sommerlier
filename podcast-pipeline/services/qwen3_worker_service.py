from services.base_worker_service import WorkerProcessService


class Qwen3WorkerService(WorkerProcessService):
    """Manages the lifecycle of the isolated Qwen3-ASR worker process."""

    def __init__(self, python_env_path: str, worker_script_path: str, device_id: int = 1,
                 logger=None, env_name: str = "kaggle", config_path: str = "config.json"):
        super().__init__(
            name="Qwen3-ASR",
            python_bin=python_env_path,
            worker_script=worker_script_path,
            extra_args=["--config", config_path, "--env", env_name],
            device_id=device_id,
            logger=logger,
        )
