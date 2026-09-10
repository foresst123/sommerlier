from services.base_worker_service import WorkerProcessService


class SidonWorkerService(WorkerProcessService):
    """Manages the lifecycle of the isolated DialogueSidon worker process.

    Separate interpreter because Sidon needs `diffusers` and a torch build the
    rest of the pipeline does not agree with. `sidon_infer.load_models` pulls
    the .pt2 weights from the DialogueSidon HF repo -- honouring HF_HOME, so an
    offline profile still resolves from its own cache -- and there is no local
    checkpoint to point at, which is why startup is allowed longer than the
    other workers: a cold cache downloads before it can answer.
    """

    def __init__(self, python_env_path: str, worker_script_path: str, device_id: int = 0,
                 logger=None, env_name: str = "kaggle", config_path: str = "config.json"):
        super().__init__(
            name="Sidon",
            python_bin=python_env_path,
            worker_script=worker_script_path,
            extra_args=["--device", f"cuda:{device_id}",
                        "--config", config_path, "--env", env_name],
            device_id=device_id,
            logger=logger,
            ready_timeout=1800.0,
        )
