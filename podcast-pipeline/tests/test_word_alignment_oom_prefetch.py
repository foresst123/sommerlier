"""OOM handling and CPU batch prefetch in the word alignment service."""

import logging
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.word_alignment_service import (
    WordAlignmentOutOfMemory, WordAlignmentService, is_out_of_memory)

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


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _logger():
    log = logging.getLogger(f"wa-test-{id(object())}")
    log.setLevel(logging.DEBUG)
    handler = _Capture()
    log.addHandler(handler)
    return log, handler


def _inproc(monkeypatch, fn, **kwargs):
    svc = WordAlignmentService(language="vi", device="cpu", **kwargs)
    monkeypatch.setattr(svc, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(svc, "_align_batch", fn)
    return svc


def _fake_words(batch):
    return {item["index"]: [{"word": w, "start": item["start"], "end": item["end"]}
                            for w in item["text"].split()] for item in batch}


def test_is_out_of_memory_recognises_relayed_text():
    assert is_out_of_memory(RuntimeError("OutOfMemoryError: CUDA out of memory"))
    assert is_out_of_memory(MemoryError())
    assert not is_out_of_memory(RuntimeError("boom"))


def test_oom_is_not_retried_per_segment_and_logs_one_line(monkeypatch):
    calls = []

    def fn(batch):
        calls.append(len(batch))
        raise RuntimeError("OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB")

    log, handler = _logger()
    svc = _inproc(monkeypatch, fn, batch_seconds=100.0, logger=log)
    with pytest.raises(WordAlignmentOutOfMemory, match="out of GPU memory"):
        svc.align(_segments(), _audio())
    assert calls == [6]                       # no per-segment retry
    problems = [r for r in handler.records if r.levelno >= logging.WARNING]
    assert len(problems) == 1 and problems[0].exc_info is None


def test_oom_during_segment_retry_stops_the_retry(monkeypatch):
    calls = []

    def fn(batch):
        calls.append(len(batch))
        if len(batch) > 1:
            raise RuntimeError("boom")
        raise RuntimeError("CUDA out of memory")

    svc = _inproc(monkeypatch, fn, batch_seconds=100.0)
    with pytest.raises(WordAlignmentOutOfMemory):
        svc.align(_segments(), _audio())
    assert calls == [6, 1]


def test_non_oom_failure_still_retries_segment_by_segment(monkeypatch):
    def fn(batch):
        if len(batch) > 1 or batch[0]["index"] == "00002":
            raise RuntimeError("boom")
        return _fake_words(batch)

    svc = _inproc(monkeypatch, fn, batch_seconds=100.0)
    result = svc.align(_segments(), _audio())
    assert [f["index"] for f in result.report["segments_failed"]] == ["00002"]
    assert result.report["segments_aligned"] == 5


def test_oom_through_worker_pool_fails_without_silent_gaps(monkeypatch):
    monkeypatch.setenv("FAKE_WA_MODE", "oom")
    log, handler = _logger()
    svc = WordAlignmentService(
        language="vi", device="cpu", workers_per_gpu=2, batch_seconds=2.0,
        prefetch_batches=2, worker_python=sys.executable, worker_script=FAKE_WORKER,
        logger=log)
    segments = _segments()
    try:
        with pytest.raises(WordAlignmentOutOfMemory):
            svc.align(segments, _audio())
    finally:
        svc.unload()
    assert all(seg.words is None for seg in segments)
    errors = [r for r in handler.records if "out of memory" in r.getMessage()]
    assert 1 <= len(errors) <= 1


@pytest.mark.parametrize("prefetch", [0, 1, 3])
def test_prefetch_keeps_batches_words_and_order(monkeypatch, prefetch):
    seen = []

    def fn(batch):
        seen.append([item["index"] for item in batch])
        return _fake_words(batch)

    svc = _inproc(monkeypatch, fn, batch_seconds=4.5, prefetch_batches=prefetch)
    segments = _segments()
    result = svc.align(segments, _audio())
    assert seen == [["00000", "00001"], ["00002", "00003"], ["00004", "00005"]]
    assert list(result.words_by_index) == [f"{i:05d}" for i in range(6)]
    assert all(len(seg.words) == 3 for seg in segments)


def test_prefetch_prepares_next_batch_while_gpu_is_busy(monkeypatch):
    prepared_at, aligned_at = [], []
    svc = WordAlignmentService(language="vi", device="cpu", batch_seconds=2.0,
                               prefetch_batches=2)
    original = svc._resample

    def slow_resample(waveform, rate):
        prepared_at.append(time.monotonic())
        return original(waveform, rate)

    def fn(batch):
        aligned_at.append(time.monotonic())
        time.sleep(0.15)
        return _fake_words(batch)

    monkeypatch.setattr(svc, "_resample", slow_resample)
    monkeypatch.setattr(svc, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(svc, "_align_batch", fn)
    svc.align(_segments(), _audio())
    # By the time the second batch starts, the third was already prepared.
    assert sum(t < aligned_at[1] + 0.05 for t in prepared_at) >= 3


def test_preparation_error_is_raised_and_thread_ends(monkeypatch):
    svc = _inproc(monkeypatch, _fake_words, prefetch_batches=1, batch_seconds=2.0)

    def bad(waveform, rate):
        raise ValueError("resample failed")

    monkeypatch.setattr(svc, "_resample", bad)
    with pytest.raises(ValueError, match="resample failed"):
        svc.align(_segments(), _audio())
    assert not [t for t in threading.enumerate() if t.name == "word-align-prep"]


def test_empty_input_still_raises_before_loading_model(monkeypatch):
    loaded = []
    svc = WordAlignmentService(language="vi", device="cpu", prefetch_batches=2)
    monkeypatch.setattr(svc, "_ensure_loaded", lambda: loaded.append(1))
    empty = [SimpleNamespace(index="0", start=0.0, end=1.0, text=" ", words=None)]
    with pytest.raises(RuntimeError, match="no non-empty"):
        svc.align(empty, _audio())
    assert not loaded
