from services.base_worker_service import WorkerProcessService


class WhisperVLLMWorkerService(WorkerProcessService):
    """Lifecycle manager for the isolated vLLM Whisper process."""

    def __init__(self, python_env_path: str, worker_script_path: str,
                 device_id: int = 1, logger=None,
                 env_name: str = "a100", config_path: str = "config.json"):
        super().__init__(
            name="Whisper-vLLM",
            python_bin=python_env_path,
            worker_script=worker_script_path,
            extra_args=["--config", config_path, "--env", env_name],
            device_id=device_id,
            logger=logger,
            isolate_library_path=True,      # its own torch/CUDA build (vllm_env)
        )
