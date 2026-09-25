from services.base_worker_service import WorkerProcessService


class PhoWhisperWorkerService(WorkerProcessService):
    """Lifecycle of one PhoWhisper worker, pinned to one GPU.

    Same interpreter as the main process: PhoWhisper is CTranslate2, which lives in
    the main environment (unlike the vLLM models).
    """

    # CTranslate2 and torch may print to stdout; only the JSON line is the handshake.
    ready_requires_json = True

    def __init__(self, python_env_path, worker_script_path: str, device_id: int = 0,
                 logger=None, env_name: str = "a100", config_path: str = "config.json"):
        super().__init__(
            name="PhoWhisper",
            python_bin=python_env_path,
            worker_script=worker_script_path,
            extra_args=["--config", config_path, "--env", env_name],
            device_id=device_id,
            logger=logger,
            ready_timeout=900.0,
        )
