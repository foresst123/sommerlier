import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qwen3_asr import Qwen3ASRClient
from models.separation_backends import SidonBackend
from services.base_worker_service import WorkerProcessService
from services.worker_pool_service import WorkerPoolService
from utils.checkpoint import CheckpointManager


def test_worker_cuda_mask_supports_one_or_multiple_physical_devices(tmp_path):
    worker = WorkerProcessService("x", sys.executable, str(tmp_path / "worker.py"),
                                  device_id=[2, 5])
    assert worker.cuda_visible_devices == "2,5"
    worker.device_id = 3
    assert worker.cuda_visible_devices == "3"


def test_worker_pool_dispatches_round_robin():
    class Worker:
        def __init__(self, name):
            self.name = name
            self.process = object()
            self.calls = 0

        def request(self, payload, response_id=None):
            self.calls += 1
            return {"worker": self.name}

        def spawn(self): pass
        def wait_ready(self): pass
        def stop(self): self.process = None

    left, right = Worker("left"), Worker("right")
    pool = WorkerPoolService([left, right])
    assert pool.request({})["worker"] == "left"
    assert pool.request({})["worker"] == "right"
    assert (left.calls, right.calls) == (1, 1)


def test_checkpoint_manifest_rejects_a_corrupt_result(tmp_path):
    checkpoints = CheckpointManager(str(tmp_path), "job")
    checkpoints.save("asr", {"ok": True}, fmt="json")
    path = checkpoints._get_stage_path("asr", "json")
    assert checkpoints.exists("asr", "json")
    assert os.path.exists(path + ".manifest.json")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("corrupt")
    assert not checkpoints.exists("asr", "json")
    assert checkpoints.load("asr", "json") is None


def test_qwen_batch_client_maps_results_by_id_not_response_order():
    class Pipe:
        def __init__(self):
            self.stdin = self
            self.stdout = self
            self.request = None

        def write(self, line):
            self.request = json.loads(line)

        def flush(self): pass

        def readline(self):
            return json.dumps({"results": [
                {"id": "b", "text": "second"},
                {"id": "a", "text": "first"},
            ]}) + "\n"

    client = Qwen3ASRClient(Pipe())
    assert client.transcribe_batch([("a", "/a.npy"), ("b", "/b.npy")]) == [
        "first", "second"]


def test_sidon_backend_can_use_a_pool_endpoint(tmp_path):
    class Endpoint:
        def request(self, payload, response_id=None):
            first = tmp_path / "first.npy"
            second = tmp_path / "second.npy"
            np.save(first, np.ones(8, np.float32))
            np.save(second, np.zeros(8, np.float32))
            return {"id": response_id, "track_1_path": str(first),
                    "track_2_path": str(second), "target_sr": 24000}

    backend = SidonBackend(process=Endpoint(), temp_dir=str(tmp_path))
    first, second, rate = backend.separate(np.zeros(16, np.float32), 16000)
    assert rate == 24000
    assert first.tolist() == [1.0] * 8
    assert second.tolist() == [0.0] * 8
