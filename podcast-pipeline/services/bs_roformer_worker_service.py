import json
from typing import Optional

from services.base_worker_service import WorkerProcessService


class BSRoformerWorkerService(WorkerProcessService):
    """Lifecycle of one BS-RoFormer worker process, one per GPU.

    Same interpreter as the main process (audio-separator is installed there),
    so unlike Sidon this needs no separate virtualenv. CUDA_VISIBLE_DEVICES maps
    the physical card to the worker's local cuda:0.
    """

    def __init__(self, python_env_path: str, worker_script_path: str,
                 device_id: Optional[int], remover_kwargs: dict, logger=None):
        super().__init__(
            name="BS-RoFormer",
            python_bin=python_env_path,
            worker_script=worker_script_path,
            extra_args=["--device", "cuda:0" if device_id is not None else "",
                        "--kwargs", json.dumps(remover_kwargs)],
            device_id=device_id,
            logger=logger,
            ready_timeout=900.0,
        )
