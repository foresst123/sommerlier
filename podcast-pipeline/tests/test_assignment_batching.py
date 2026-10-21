"""Batching preserves probe features, fallback and speaker assignment inputs."""
import collections
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bss_model import BssSeparator, _AssignmentBatchPolicy
from models.wespeaker_embedding import WeSpeakerONNXEmbedder
from test_assignment_worker import FakePool, FakeWorkerSep, _client
from utils import performance_config


def test_batching_is_opt_in_and_enabled_only_for_the_tuned_a100_profile():
    assert performance_config._STAGES["separation"]["assignment_batching"][1] is False
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"),
              encoding="utf-8") as stream:
        environments = json.load(stream)["environments"]
    separation = environments["a100"]["performance"]["stages"]["separation"]
    assert separation["assignment_batching"] is True
    assert separation["assignment_workers_per_gpu"] == 4
    assert separation["assignment_batch_warmup_requests"] == 20
    assert separation["assignment_batch_min_hit_rate"] == 0.5


class Session:
    def __init__(self, batch_axis="batch", fail_batch=False):
        self.batch_axis = batch_axis
        self.fail_batch = fail_batch
        self.inputs = []

    def get_inputs(self):
        return [SimpleNamespace(name="feats", shape=[self.batch_axis, "frames", 80])]

    def run(self, names, feeds):
        data = feeds["feats"]
        self.inputs.append(data.copy())
        if self.fail_batch and len(data) > 1:
            raise RuntimeError("batch unsupported/OOM")
        return [np.stack([data.mean(axis=(1, 2)), data.max(axis=(1, 2))], axis=1)]


def embedder(session):
    value = WeSpeakerONNXEmbedder("cpu")
    value._session = session
    # The production feature frontend is unchanged. Here audio encodes already
    # normalized features, making padding/cropping visible in the assertions.
    value._features = lambda audio, sr: torch.as_tensor(audio)
    return value


def test_equal_frames_batch_and_variable_frames_remain_unpadded_in_original_order():
    session = Session()
    model = embedder(session)
    features = [np.full((n, 80), v, np.float32)
                for n, v in [(11, 1), (18, 2), (11, 3), (25, 4)]]
    actual = model.embed_batch(features)
    expected = [model.embed(f) for f in features]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert [a.shape for a in session.inputs[:3]] == [(2, 11, 80), (1, 18, 80), (1, 25, 80)]
    np.testing.assert_array_equal(session.inputs[0][0], features[0])
    np.testing.assert_array_equal(session.inputs[0][1], features[2])
    assert model.batch_stats["batched_items"] == 2


@pytest.mark.parametrize("fixed_batch", [False, True])
def test_unsupported_batch_falls_back_to_individual_inference(fixed_batch):
    session = Session(batch_axis=1 if fixed_batch else "B", fail_batch=not fixed_batch)
    model = embedder(session)
    features = [np.full((8, 80), i, np.float32) for i in range(3)]
    values = model.embed_batch(features)
    assert [v.tolist() for v in values] == [[0, 0], [1, 1], [2, 2]]
    assert model.batch_stats["single_calls"] == 3
    assert model.batch_stats["batch_fallbacks"] == (0 if fixed_batch else 1)


def test_bad_feature_or_nonfinite_result_does_not_drop_other_items():
    model = embedder(Session())
    def features(audio, sr):
        if audio is None:
            raise ValueError("bad clip")
        return torch.as_tensor(audio)
    model._features = features
    values = model.embed_batch([None, np.ones((8, 80), np.float32),
                                np.full((8, 80), np.nan, np.float32)])
    assert isinstance(values[0], ValueError)
    assert values[1].tolist() == [1, 1]
    assert isinstance(values[2], ValueError)


class BatchWorker(FakeWorkerSep):
    def __init__(self):
        self.speaker_embedder = SimpleNamespace(batch_stats=collections.Counter())

    def _get_embeddings(self, audios, sr):
        if len(audios) > 1:
            self.speaker_embedder.batch_stats["batch_attempts"] += 1
            self.speaker_embedder.batch_stats["batches"] += 1
            self.speaker_embedder.batch_stats["batched_items"] += len(audios)
        else:
            self.speaker_embedder.batch_stats["single_calls"] += len(audios)
        return [self._get_embedding(a, sr) for a in audios]


def test_packed_probe_batch_matches_single_requests_including_silent_probe():
    pool = FakePool(BatchWorker())
    client = _client(pool)
    tracks = [np.ones(9000, np.float32), np.zeros(9000, np.float32),
              np.full(12000, 2, np.float32)]
    probes = [(track, [(0, len(track))]) for track in tracks]
    batched = client._probe_embeddings(probes, 24000)
    # VAD makes the two voiced probes different lengths, so neither is padded:
    # both embeddings fan out as independent worker requests.
    assert pool.calls.count("probe") == 3
    assert pool.calls.count("embed") == 2
    assert "embed_batch" not in pool.calls
    individual = [client._probe_embedding(track, spans, 24000) for track, spans in probes]
    for a, b in zip(batched, individual):
        if b is None:
            assert a is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert os.listdir(client._temp_dir) == []


def test_batch_scoring_reuses_full_context_for_both_speakers_and_preserves_fallback():
    client = _client(FakePool(BatchWorker()))
    tracks = [np.ones(9000, np.float32), np.full(9000, 2, np.float32)]
    # Clean probe is silent, full context is voiced.
    for t in tracks:
        t[:1000] = 0
    a = torch.tensor([1., 0., 0.])
    b = torch.tensor([0., 1., 0.])
    requests = [(tracks[0], [(0, 1000)], a, "track1_A"),
                (tracks[1], [(0, 1000)], a, "track2_A"),
                (tracks[0], None, b, "track1_B"),
                (tracks[1], None, b, "track2_B")]
    sources, errors = {}, {}
    scores = client._score_probe_batch(requests, 16000, [(0, 9000)], sources, errors)
    # Four unique first attempts: two silent clean probes, two full tracks.
    # The two fallback comparisons reuse those full-track embeddings.
    assert client._assignment.calls.count("probe") == 4
    assert client._assignment.calls.count("embed_batch") == 1
    assert not errors and set(sources.values()) == {"full_context"}
    for score, (track, _, target, _) in zip(scores, requests):
        embedding = client._probe_embedding(track, [(0, 9000)], 16000)
        assert score == float(torch.dot(target, embedding))


def test_enrollment_centroid_is_computed_once_when_windows_arrive_together():
    client = _client(FakePool(BatchWorker()))
    client.target_embed_cache = {}
    client._target_locks = {}
    client._target_locks_guard = threading.Lock()
    client.assignment_batching = True
    start = threading.Barrier(4)
    def ask():
        start.wait(timeout=5)
        return client._get_target_embedding([np.ones(9000, np.float32)], "speakerA", 16000)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: ask(), range(4)))
    assert client._assignment.calls == ["embed_batch"]
    for result in results[1:]:
        torch.testing.assert_close(result, results[0], rtol=0, atol=0)


@pytest.mark.parametrize("silent_clean", [False, True])
def test_complete_assignment_returns_same_tracks_scores_and_qc_diagnostics(silent_clean):
    t = np.arange(48000, dtype=np.float32) / 16000
    tracks = [np.sin(t * 220 * 2 * np.pi).astype(np.float32),
              np.sin(t * 350 * 2 * np.pi).astype(np.float32)]
    if silent_clean:
        for track in tracks:
            track[:16000] = 0
    outputs = []
    for batching in (False, True):
        client = _client(FakePool(BatchWorker()))
        client.target_embed_cache = {}
        client.assignment_batching = batching
        client.backend = SimpleNamespace(ordered=True)
        client._score_pool = None
        outputs.append(client.postprocess_separated(
            (tracks[0] + tracks[1]) * 0.5, (*tracks, 16000),
            enroll_A=[np.ones(16000, np.float32)],
            enroll_B=[np.ones(22000, np.float32)],
            sample_rate=16000, id_A="a", id_B="b",
            probe_A=[(0, 16000)], probe_B=[(32000, 48000)],
            core_range=(16000, 32000)))
    for a, b in zip(outputs[0][:2], outputs[1][:2]):
        np.testing.assert_array_equal(a, b)
    assert outputs[0][2:] == outputs[1][2:]


def test_packed_batch_keeps_successes_when_one_probe_errors():
    class BadProbe(BatchWorker):
        def _probe_from_segment(self, seg, sr, **kwargs):
            if seg[0] < 0:
                raise ValueError("bad probe")
            return super()._probe_from_segment(seg, sr, **kwargs)
    client = _client(FakePool(BadProbe()))
    tracks = [np.ones(9000, np.float32), -np.ones(9000, np.float32)]
    values = client._probe_embeddings([(t, [(0, 9000)]) for t in tracks], 16000)
    assert isinstance(values[0], torch.Tensor)
    assert isinstance(values[1], RuntimeError) and "bad probe" in str(values[1])


def test_low_batch_hit_rate_switches_the_rest_of_the_stage_to_direct_fanout():
    pool = FakePool(BatchWorker())
    client = _client(pool)
    client._assignment_batch_policy = None
    probes = [
        (np.ones(9000, np.float32), [(0, 9000)]),
        (np.ones(12000, np.float32), [(0, 12000)]),
    ]
    # Ten scoring calls observe 20 differently-sized, embeddable probes. None
    # can share an ONNX batch, so the one-way policy switches at the threshold.
    for _ in range(10):
        client._probe_embeddings(probes, 16000)
    before = len(pool.calls)
    client._probe_embeddings(probes, 16000)
    assert pool.calls[before:] == ["probe_embed", "probe_embed"]
    assert client._assignment_batch_policy.fanout is True
    assert client._assignment_batch_policy.total_items == 20


def test_high_batch_hit_rate_keeps_exact_length_batching_after_warmup():
    pool = FakePool(BatchWorker())
    client = _client(pool)
    client._assignment_batch_policy = None
    probes = [(np.full(9000, value, np.float32), [(0, 9000)])
              for value in (1, 2, 3, 4)]
    for _ in range(5):
        client._probe_embeddings(probes, 16000)
    before = len(pool.calls)
    client._probe_embeddings(probes, 16000)
    assert pool.calls[before:].count("probe") == 4
    assert pool.calls[before:].count("embed_batch") == 1
    assert client._assignment_batch_policy.fanout is False
    assert client._assignment_batch_policy.batched_items == 24


def test_exactly_fifty_percent_is_kept_in_batch_mode():
    policy = _AssignmentBatchPolicy(warmup_requests=20, min_hit_rate=0.5)
    snapshot = policy.observe(batched_items=10, total_items=20)
    assert snapshot["hit_rate"] == 0.5
    assert snapshot["fanout"] is False
    assert policy.should_batch() is True
