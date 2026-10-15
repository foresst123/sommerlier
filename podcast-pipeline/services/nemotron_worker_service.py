from services.diarizen_worker_service import DiarizenWorkerService


class NemotronWorkerService(DiarizenWorkerService):
    """Isolated Nemotron-3-Diarization worker; same JSON-lines protocol as DiariZen."""

    def __init__(self, python_env_path, worker_script_path, device_id=0,
                 logger=None, env_name: str = "kaggle", config_path: str = "config.json"):
        super().__init__(python_env_path, worker_script_path, device_id=device_id,
                         logger=logger, env_name=env_name, config_path=config_path)
        self.name = "Nemotron"
