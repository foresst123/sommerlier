from typing import Optional

from services.base_worker_service import WorkerProcessService


class AssignmentWorkerService(WorkerProcessService):
    """Lifecycle of one speaker-assignment worker (Silero VAD + WeSpeaker).

    Same interpreter as the main process, which already has both models.
    CUDA_VISIBLE_DEVICES maps the physical card to the worker's local cuda:0;
    a None device id runs the worker on the CPU.
    """

    def __init__(self, python_env_path: str, worker_script_path: str,
                 device_id: Optional[int], *, threads: int = 2, logger=None):
        super().__init__(
            name="Assignment",
            python_bin=python_env_path,
            worker_script=worker_script_path,
            extra_args=["--device", "cuda:0" if device_id is not None else "cpu",
                        "--threads", str(int(threads))],
            device_id=device_id,
            logger=logger,
            ready_timeout=900.0,
        )
