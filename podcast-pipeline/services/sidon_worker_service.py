import os
from typing import Any, Dict

from services.base_worker_service import WorkerProcessService
from utils.worker_env import resolve_checkpoint, resolve_worker_python

# Deployment-specific locations the DialogueSidon checkpoint has been seen at.
# A config profile's checkpoints.dialogue_sidon overrides all of these.
_KAGGLE_CHECKPOINT_CANDIDATES = [
    "/kaggle/temp/tse_model/weights/dialogue_sidon.ckpt",
    "/kaggle/temp/dialogue_sidon.ckpt",
    "/kaggle/working/tse_model/weights/dialogue_sidon.ckpt",
    "/kaggle/working/sommerlier/tse_model/weights/dialogue_sidon.ckpt",
    "/kaggle/input/dialogue-sidon/dialogue_sidon.ckpt",
    "/kaggle/input/sidon-weights/dialogue_sidon.ckpt",
    "/kaggle/input/tse-model/weights/dialogue_sidon.ckpt",
]


class SidonWorkerService(WorkerProcessService):
    """Manages the lifecycle of the Sidon (TSE) worker subprocess."""

    def __init__(self, config: Dict[str, Any], args: Any, logger=None):
        self.config = config
        self.args = args
        self.env_profile = config.get("environments", {}).get(
            getattr(args, "env", "kaggle"), {}
        )

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tse_path = os.environ.get("TSE_PATH", os.path.join(base_dir, "..", "tse_model"))

        checkpoint_path = resolve_checkpoint(
            "DialogueSidon",
            [os.path.join(tse_path, "weights", "dialogue_sidon.ckpt")]
            + _KAGGLE_CHECKPOINT_CANDIDATES,
            env_profile=self.env_profile,
            config_key="dialogue_sidon",
            logger=logger,
        )

        super().__init__(
            name="Sidon",
            python_bin=resolve_worker_python(
                "sidon", config=config, env_profile=self.env_profile, logger=logger
            ),
            worker_script=os.path.join(base_dir, "sidon_worker.py"),
            extra_args=[
                "--device", f"cuda:{args.gpu_1}",
                "--model_path", checkpoint_path,
            ],
            logger=logger,
        )

    def start(self):
        if not getattr(self.args, "tse", False):
            return
        super().start()
