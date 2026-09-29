"""Word alignment batches fan out over a pool of worker processes."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.word_alignment_service import WordAlignmentService

FAKE_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fake_word_alignment_worker.py")


def _segments(count=6):
    return [SimpleNamespace(index=f"{i:05d}", start=i * 3.0, end=i * 3.0 + 2.0,
                            speaker="SPEAKER_00", text="xin chào bạn",
                            words=None, unseparated=None)
            for i in range(count)]


def _audio():
    return SimpleNamespace(waveform=np.zeros(30 * 16000, dtype=np.float32),
                           sample_rate=16000)


def _service(**kwargs):
    kwargs.setdefault("batch_seconds", 2.0)          # one clip per batch
    return WordAlignmentService(
        language="vi", device="cpu", workers_per_gpu=2,
        worker_python=sys.executable, worker_script=FAKE_WORKER, **kwargs)


def test_batches_are_spread_over_several_worker_processes():
    svc = _service()
    segments = _segments()
    try:
        result = svc.align(segments, _audio())
        processes = [s.process for s in svc._pool.services]

        assert len(processes) == 2 and all(p.poll() is None for p in processes)
        assert result.report["aligned_words"] == 18 and result.report["segments_complete"] == 6
        assert all(len(seg.words) == 3 for seg in segments)
        assert len({w["pid"] for seg in segments for w in seg.words}) == 2
        assert segments[1].words[0]["start"] == 3.0
    finally:
        svc.unload()
    assert svc._pool is None and all(p.poll() is not None for p in processes)


def test_a_failing_multi_clip_batch_is_retried_clip_by_clip(monkeypatch):
    monkeypatch.setenv("FAKE_WA_MODE", "fail_multi")
    svc = _service(batch_seconds=100.0)               # one batch holding all six
    segments = _segments()
    try:
        result = svc.align(segments, _audio())
        assert result.report["aligned_words"] == 18 and result.report["segments_failed"] == []
    finally:
        svc.unload()


def test_when_every_worker_call_fails_the_first_error_is_reported(monkeypatch):
    monkeypatch.setenv("FAKE_WA_MODE", "error")
    svc = _service()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            svc.align(_segments(3), _audio())
    finally:
        svc.unload()


def test_without_workers_nothing_is_spawned():
    svc = WordAlignmentService(language="vi", device="cpu")
    assert svc.workers_per_gpu == 0 and svc._pool is None


def test_negative_worker_counts_are_rejected():
    with pytest.raises(ValueError):
        WordAlignmentService(workers_per_gpu=-1)


def test_the_worker_gpu_list_defaults_to_the_service_device():
    assert WordAlignmentService(device="cuda:1", workers_per_gpu=1)._worker_gpu_ids() == [1]
    assert WordAlignmentService(device="cpu", workers_per_gpu=1)._worker_gpu_ids() == [None]
    assert WordAlignmentService(device="cuda:0", workers_per_gpu=1,
                                worker_gpus=[0, 1])._worker_gpu_ids() == [0, 1]
