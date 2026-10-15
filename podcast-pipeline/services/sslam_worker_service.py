"""SSLAM taggers as worker processes, seen from the pipeline as one tagger.

`build_sslam_pool` starts N `sslam_worker.py` processes spread over the GPUs and
returns a client with the one method the rest of the pipeline uses,
`tag_framewise(audio, sample_rate) -> (scores, fps)`. Each call leases whichever
worker is idle, so several files are swept at once, and the callers that arrive
while every worker is busy wait in the pool's queue.

The audio goes to the worker as a .npy file and the curves come back as an .npz;
a recording is many megabytes, and line JSON is for the request, not the payload.
"""
import os
import shutil
import sys
import tempfile
import uuid
from typing import Optional, Sequence

import numpy as np

from services.base_worker_service import WorkerProcessService
from services.worker_pool_service import WorkerPoolService

WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sslam_worker.py")


class SSLAMWorkerService(WorkerProcessService):
    """One SSLAM worker process on one physical GPU (or on the CPU)."""

    # transformers and its dependencies may print to stdout while loading; only the
    # JSON handshake counts as ready.
    ready_requires_json = True

    def __init__(self, device_id: Optional[int] = None, python_bin: Optional[str] = None,
                 worker_script: Optional[str] = None, logger=None,
                 ready_timeout: float = 900.0):
        super().__init__(
            name="SSLAM",
            python_bin=python_bin or sys.executable,
            worker_script=worker_script or WORKER_SCRIPT,
            # CUDA_VISIBLE_DEVICES masks the process to its one physical card, which
            # is then cuda:0 inside it.
            extra_args=["--device", "cuda:0" if device_id is not None else "cpu"],
            device_id=device_id,
            logger=logger,
            ready_timeout=ready_timeout,
        )


class SSLAMWorkerClient:
    """A pool of SSLAM worker processes, presented as a detector.

    `shared_safe` tells the pipeline not to put a lock of its own around a sweep:
    the pool already gives each one a worker.
    """

    shared_safe = True

    def __init__(self, pool, scratch_dir: Optional[str] = None):
        self.pool = pool
        self._dir = scratch_dir or tempfile.mkdtemp(prefix="sslam_exchange_")

    def tag_framewise(self, audio_array, sample_rate: int = 16000):
        request_id = uuid.uuid4().hex
        audio_path = os.path.join(self._dir, f"{request_id}.npy")
        scores_path = None
        np.save(audio_path, np.asarray(audio_array, dtype=np.float32).reshape(-1))
        try:
            reply = self.pool.request(
                {"id": request_id, "cmd": "tag", "audio_path": audio_path,
                 "sample_rate": int(sample_rate)},
                response_id=request_id)
            if reply.get("error"):
                raise RuntimeError(f"SSLAM worker: {reply['error']}")
            scores_path = reply["scores_path"]
            with np.load(scores_path) as data:
                scores = {name: np.array(data[name]) for name in data.files}
            return scores, float(reply["fps"])
        finally:
            for path in (audio_path, scores_path):
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def unload(self):
        """Stop every worker process, which is what frees their GPU memory."""
        try:
            self.pool.stop()
        finally:
            shutil.rmtree(self._dir, ignore_errors=True)


def build_sslam_pool(devices: Sequence[Optional[int]], logger=None) -> SSLAMWorkerClient:
    """Start one worker per entry of `devices` (physical GPU index, or None for CPU).

    All are launched first and joined afterwards, so the checkpoints load
    concurrently instead of one after another.
    """
    services = [SSLAMWorkerService(device_id=device, logger=logger) for device in devices]
    pool = WorkerPoolService(services, name="SSLAM")
    pool.spawn()
    try:
        pool.wait_ready()
    except Exception:
        pool.stop()
        raise
    return SSLAMWorkerClient(pool)
