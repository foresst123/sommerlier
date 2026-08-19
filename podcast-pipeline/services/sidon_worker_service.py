import os
from typing import Any, Dict

from services.base_worker_service import WorkerProcessService
from utils.worker_env import resolve_worker_python


class SidonWorkerService(WorkerProcessService):
    """Manages the lifecycle of the Sidon (TSE) worker subprocess."""

    def __init__(self, config: Dict[str, Any], args: Any, logger=None):
        self.config = config
        self.args = args
        self.env_profile = config.get("environments", {}).get(
            getattr(args, "env", "kaggle"), {}
        )

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # sidon_infer.load_models pulls the .pt2 weights from the DialogueSidon
        # HF repo (honouring HF_HOME, so an offline profile still resolves from
        # its own cache). There is no local checkpoint to point at.
        super().__init__(
            name="Sidon",
            python_bin=resolve_worker_python(
                "sidon", config=config, env_profile=self.env_profile, logger=logger
            ),
            worker_script=os.path.join(base_dir, "sidon_worker.py"),
            extra_args=["--device", f"cuda:{args.gpu_1}"],
            logger=logger,
            # Weight download on a cold cache is the slow part of startup.
            ready_timeout=1800.0,
        )

    def start(self):
        if not getattr(self.args, "tse", False):
            return
        super().start()
