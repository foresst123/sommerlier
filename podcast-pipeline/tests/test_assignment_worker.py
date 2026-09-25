"""Silero VAD and WeSpeaker run in worker processes; the client in BssSeparator and
the worker's handle_request must agree on the protocol. No models: a fake separator
stands in for the worker's, and a fake pool carries the request across."""

import collections
import itertools
import json
import os
import sys
import tempfile
import threading

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assignment_worker
from models.bss_model import BssSeparator
from services.assignment_worker_service import AssignmentWorkerService
from utils import performance_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeWorkerSep:
    """What the worker calls: a probe filter and an embedder."""

    def _probe_from_segment(self, seg, sr, floor_db=-40.0, min_voiced_sec=None,
                            abs_floor_rms=1e-3):
        self.last_options = (floor_db, min_voiced_sec, abs_floor_rms)
        return None if not np.any(seg) else seg[: max(1, len(seg) // 2)]

    def _get_embedding(self, audio, sample_rate):
        return torch.tensor([float(len(audio)), float(sample_rate), 1.0])


class FakePool:
    """Runs the worker's handle_request in-process, like a pool of one worker."""

    def __init__(self, sep):
        self.sep, self.calls, self.lock = sep, [], threading.Lock()

    def request(self, payload, *, response_id=None):
        with self.lock:
            self.calls.append(payload["cmd"])
        reply = assignment_worker.handle_request(self.sep, json.loads(json.dumps(payload)))
        assert response_id is None or reply["id"] == response_id
        return reply


def _client(pool):
    sep = object.__new__(BssSeparator)
    sep.device = torch.device("cpu")
    sep._assignment = pool
    sep._temp_dir = tempfile.mkdtemp(prefix="assign_test_")
    sep._remote_counter = itertools.count()
    sep.timing = collections.Counter()
    sep._timing_lock = threading.Lock()
    sep.speaker_embedder = None
    sep._vad = None
    return sep


def test_an_embedding_comes_back_as_a_tensor_on_the_clients_device():
    client = _client(FakePool(FakeWorkerSep()))
    emb = client._get_embedding(np.ones(4800, np.float32), 24000)
    assert emb.tolist() == [4800.0, 24000.0, 1.0] and emb.device.type == "cpu"


def test_probe_embed_is_one_round_trip_and_normalised():
    pool = FakePool(FakeWorkerSep())
    client = _client(pool)
    track = np.ones(48000, np.float32)
    emb = client._probe_embedding(track, [(0, 4000), (8000, 12000)], 24000)
    assert pool.calls == ["probe_embed"]
    assert torch.linalg.vector_norm(emb).item() == pytest.approx(1.0, abs=1e-5)


def test_a_silent_track_gives_no_probe_and_no_embedding():
    client = _client(FakePool(FakeWorkerSep()))
    silent = np.zeros(48000, np.float32)
    assert client._probe_embedding(silent, [(0, 4000)], 24000) is None
    assert client._gather_probe(silent, [(0, 4000)], 24000) is None
    assert client._probe_embedding(np.ones(100, np.float32), [], 24000) is None


def test_the_probe_command_returns_the_probe_audio():
    client = _client(FakePool(FakeWorkerSep()))
    probe = client._gather_probe(np.ones(48000, np.float32), [(0, 4000)], 24000)
    assert probe.shape == (2000,) and probe.dtype == np.float32


def test_the_options_reach_the_worker_and_scratch_files_are_removed():
    sep = FakeWorkerSep()
    client = _client(FakePool(sep))
    client._probe_embedding(np.ones(9000, np.float32), [(0, 8000)], 24000,
                            floor_db=-30.0, min_voiced_sec=0.25, abs_floor_rms=0.01)
    assert sep.last_options == (-30.0, 0.25, 0.01)
    assert os.listdir(client._temp_dir) == []


def test_a_worker_error_becomes_an_exception_in_the_client():
    class Broken(FakeWorkerSep):
        def _get_embedding(self, audio, sample_rate):
            raise ValueError("bad audio")

    client = _client(FakePool(Broken()))
    with pytest.raises(RuntimeError, match="ValueError: bad audio"):
        client._get_embedding(np.ones(100, np.float32), 16000)


def test_an_unknown_command_is_reported_not_raised():
    reply = assignment_worker.handle_request(
        FakeWorkerSep(), {"cmd": "nope", "id": "x", "audio_path": "/nonexistent", "sr": 1})
    assert "error" in reply and reply["id"] == "x"


def test_parallel_scores_each_get_their_own_scratch_files():
    pool = FakePool(FakeWorkerSep())
    client = _client(pool)
    from concurrent.futures import ThreadPoolExecutor
    client._score_pool = ThreadPoolExecutor(max_workers=4)
    client.score_workers = 4
    tracks = [np.full(9000, i + 1, np.float32) for i in range(4)]
    results = client._run_scores([
        (lambda t=t: client._probe_embedding(t, [(0, 8000)], 24000)) for t in tracks])
    assert all(r is not None for r in results) and len(pool.calls) == 4
    assert os.listdir(client._temp_dir) == []


def test_the_worker_process_runs_on_its_gpu_and_the_main_interpreter():
    service = AssignmentWorkerService(sys.executable, "assignment_worker.py", device_id=1,
                                      threads=3)
    assert service.name == "Assignment"
    args = service.extra_args
    assert args[args.index("--device") + 1] == "cuda:0"       # remapped by CUDA_VISIBLE_DEVICES
    assert args[args.index("--threads") + 1] == "3"


def test_workers_are_off_by_default_and_on_in_the_a100_profiles():
    schema = performance_config._STAGES["separation"]["assignment_workers_per_gpu"]
    assert schema[1] == 0
    cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    for name in ("a100", "a100_hf"):
        stage = cfg["environments"][name]["performance"]["stages"]["separation"]
        assert stage["assignment_workers_per_gpu"] >= 1, name
    kaggle = cfg["environments"]["kaggle"]["performance"]["stages"]["separation"]
    assert kaggle.get("assignment_workers_per_gpu", 0) == 0


def test_the_separation_stage_starts_and_releases_the_assignment_workers():
    from services.pipeline_service import PipelineService
    assert "assignment" in PipelineService.WORKERS_FOR_STAGE["separation"]
    source = open(os.path.join(ROOT, "services", "pipeline_service.py"),
                  encoding="utf-8").read()
    assert '_release_worker(args, "assignment")' in source
