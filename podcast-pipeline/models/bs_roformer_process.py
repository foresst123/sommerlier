"""BS-RoFormer whose GPU half runs in a worker process.

Subclasses BSRoformerRemover and overrides only separate_raw(); the CPU-side
methods (postprocess_separated, separate_span_raw, ...) are inherited, and
music_service.py sees the same interface. See bs_roformer_worker.py for why the
separator lives in its own process.
"""

import os
import shutil
import sys
import tempfile
import threading
import uuid
from typing import Optional

import numpy as np

from models.bs_roformer import BSRoformerRemover
from services.bs_roformer_worker_service import BSRoformerWorkerService

_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bs_roformer_worker.py")


def physical_gpu_index(device) -> Optional[int]:
    """Host GPU index behind a torch device string, or None for CPU."""
    text = str(device or "")
    if not text.startswith("cuda"):
        return None
    _, _, index = text.partition(":")
    return int(index) if index.isdigit() else 0


class BSRoformerProcessRemover(BSRoformerRemover):
    def __init__(self, device=None, logger=None, python_bin=None,
                 worker_script=None, **kwargs):
        super().__init__(device=device, logger=logger, **kwargs)
        self._worker_kwargs = dict(kwargs)
        self._python_bin = python_bin or sys.executable
        self._worker_script = worker_script or _WORKER_SCRIPT
        self._service = None
        self._service_lock = threading.Lock()
        self._io_dir = None

    def _get_model(self):
        raise RuntimeError(
            "BSRoformerProcessRemover keeps the separator in a worker process")

    def _ensure_worker(self):
        with self._service_lock:
            if self._service is None:
                self._service = BSRoformerWorkerService(
                    self._python_bin, self._worker_script,
                    physical_gpu_index(self.device), self._worker_kwargs,
                    logger=self.logger)
            process = self._service.process
            if process is not None and process.poll() is not None:
                self._service.stop()          # reap a worker that died
            if self._service.process is None:
                self._service.start()
            if self._io_dir is None:
                self._io_dir = tempfile.mkdtemp(prefix="bsroformer_io_")
            return self._service

    def separate_raw(self, audio_array: np.ndarray, sample_rate: int):
        """GPU half of _run(), executed by the worker. None on any failure."""
        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.ndim > 2:
            audio = audio.reshape(len(audio), -1)
        if audio.size == 0:
            return None
        request_id = uuid.uuid4().hex
        in_path = out_path = None
        try:
            service = self._ensure_worker()
            in_path = os.path.join(self._io_dir, f"{request_id}.npy")
            np.save(in_path, audio)
            response = service.request(
                {"id": request_id, "audio_path": in_path,
                 "sample_rate": int(sample_rate)},
                response_id=request_id)
            if response.get("error"):
                raise RuntimeError(response["error"])
            out_path = response.get("out_path")
            if not out_path:
                return None
            out = np.load(out_path)
            return out, int(response["out_sr"]), bool(response["stereo_in"])
        except Exception as exc:
            if self.logger:
                self.logger.error(
                    f"BS-RoFormer worker failed ({type(exc).__name__}: {exc}); "
                    "keeping the mixture for this span")
            return None
        finally:
            for path in (in_path, out_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def unload(self):
        with self._service_lock:
            service, self._service = self._service, None
            io_dir, self._io_dir = self._io_dir, None
        if service is not None:
            service.stop()
        if io_dir:
            shutil.rmtree(io_dir, ignore_errors=True)


def build_bs_roformers(devices, cfg: dict, logger=None) -> list:
    """One separator per device; each in its own process when isolate_process is set."""
    cfg = dict(cfg)
    if cfg.pop("isolate_process", False):
        cls = BSRoformerProcessRemover
    else:
        from models.bs_roformer import BSRoformerRemover as cls
    return [cls(device=str(device), logger=logger, **cfg) for device in devices]
