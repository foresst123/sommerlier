from typing import Optional

from services.base_worker_service import WorkerProcessService


class WordAlignmentWorkerService(WorkerProcessService):
    """Lifecycle of one forced-alignment worker process.

    Same interpreter as the main process (whisperx is installed there).
    CUDA_VISIBLE_DEVICES maps the physical card to the worker's local cuda:0;
    a None device id runs the worker on the CPU.
    """

    def __init__(self, python_env_path: str, worker_script_path: str,
                 device_id: Optional[int], *, language: str, interpolate_method: str,
                 threads: int, model_name: Optional[str] = None,
                 model_dir: Optional[str] = None, model_cache_only: bool = False,
                 logger=None):
        extra = ["--language", language, "--interpolate", interpolate_method,
                 "--threads", str(int(threads)),
                 "--device", "cuda:0" if device_id is not None else "cpu"]
        if model_name:
            extra += ["--model-name", model_name]
        if model_dir:
            extra += ["--model-dir", model_dir]
        if model_cache_only:
            extra.append("--cache-only")
        super().__init__(
            name="WordAlign",
            python_bin=python_env_path,
            worker_script=worker_script_path,
            extra_args=extra,
            device_id=device_id,
            logger=logger,
            ready_timeout=900.0,
        )
