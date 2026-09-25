"""Kiểm thử cache dựng cửa sổ trước (prefetch), tách khỏi test_separation_logic.py
vì đây là mối quan tâm khác: bộ nhớ đệm/luồng nền, không phải chính sách overlap.
Chạy trong podcast-pipeline: python -m pytest tests/test_separation_prefetch.py -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.audio import AudioData
from schemas.segment import Segment
from services.separation_service import SeparationService

SR = 24000


class FakeTSE:
    def __init__(self, sim_a=0.6, sim_b=0.6):
        self.sim_a, self.sim_b = sim_a, sim_b
        self.calls = []

    def separate_two_speakers(self, mixture_audio, enroll_A, enroll_B, sample_rate,
                              id_A, id_B, probe_A=None, probe_B=None, core_range=None):
        self.calls.append(len(mixture_audio))
        return (np.full(len(mixture_audio), 0.5, dtype=np.float32),
                np.full(len(mixture_audio), -0.5, dtype=np.float32),
                self.sim_a, self.sim_b,
                {"anchor_self": 0.6, "anchor_other": 0.1, "other_rms": 0.5})


def _audio(duration=60.0):
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 0.05, int(duration * SR)).astype(np.float32)
    wave[np.arange(len(wave)) % (SR // 2) < int(0.08 * SR)] = 0
    return AudioData(waveform=wave, sample_rate=SR, name="test", audio_segment=None,
                     duration=duration)


def _dialogue():
    return [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=14.0, end=14.4, speaker="SPEAKER_01"),
        Segment(index="00003", start=32.0, end=40.0, speaker="SPEAKER_01"),
        Segment(index="00004", start=42.0, end=48.0, speaker="SPEAKER_00"),
    ]


def test_prefetching_produces_the_same_result_as_building_inline():
    """The point of prefetching is that it changes nothing observable --
    only when the CPU work happens, not what it produces."""
    plain = SeparationService(FakeTSE(), logger=None)
    plain_out = plain.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)

    prefetched = SeparationService(FakeTSE(), logger=None)
    prefetched.prefetch_overlap_plan(_dialogue(), _audio(), "b.mp3")
    prefetched_out = prefetched.process_overlaps(
        _dialogue(), _audio(), overlap_threshold=0.1, audio_path="b.mp3")

    assert len(plain_out) == len(prefetched_out)
    for a, b in zip(plain_out, prefetched_out):
        assert a.index == b.index
        assert a.bss == b.bss
        assert a.bss_spans == b.bss_spans
    assert prefetched.stats["pairs"] == plain.stats["pairs"] == 1


def test_a_failed_prefetch_falls_back_to_building_now(monkeypatch):
    """A background build can fail for reasons that have nothing to do with
    the file (a transient pool error). process_overlaps must not propagate
    that -- it must build the plan itself, the same way it would with no
    prefetch at all."""
    calls = {"n": 0}
    original = SeparationService._build_overlap_plan

    def flaky(self, segments, audio, overlap_threshold=0.1):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return original(self, segments, audio, overlap_threshold)

    monkeypatch.setattr(SeparationService, "_build_overlap_plan", flaky)

    svc = SeparationService(FakeTSE(), logger=None)
    svc.prefetch_overlap_plan(_dialogue(), _audio(), "a.mp3")
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1, audio_path="a.mp3")

    assert calls["n"] == 2, "the prefetch attempt failed once, then process_overlaps built it itself"
    assert len(out) == len(_dialogue())
    assert any(s.bss for s in out), "the fallback build must still separate the overlap"


def test_drop_prefetched_plan_discards_an_unused_prefetch():
    svc = SeparationService(FakeTSE(), logger=None)
    svc.prefetch_overlap_plan(_dialogue(), _audio(), "d.mp3")
    svc.drop_prefetched_plan("d.mp3")
    assert "d.mp3" not in svc._prefetch_cache
    svc.drop_prefetched_plan("never-existed.mp3")  # must not raise


def test_close_prefetch_pool_is_idempotent():
    svc = SeparationService(FakeTSE(), logger=None)
    svc.prefetch_overlap_plan(_dialogue(), _audio(), "c.mp3")
    svc._take_prefetched_plan("c.mp3")  # drain it so the executor did real work
    svc.close_prefetch_pool()
    assert svc._prefetch_executor is None
    svc.close_prefetch_pool()  # must not raise when called twice


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, message, *a, **k):
        self.lines.append(message)

    debug = warning = error = info


class _AsyncModel(FakeTSE):
    """Split model: separate_raw runs on the GPU pool, postprocess_separated in the consumer."""

    def separate_raw(self, audio, sample_rate):
        return [audio, audio]

    def postprocess_separated(self, mixture_audio, raw_tracks, enroll_A, enroll_B,
                              sample_rate, id_A, id_B, probe_A=None, probe_B=None,
                              core_range=None):
        return self.separate_two_speakers(mixture_audio, enroll_A, enroll_B,
                                          sample_rate, id_A, id_B, probe_A, probe_B,
                                          core_range)


def test_the_stage_reports_whether_it_waits_for_sidon_or_for_speaker_assignment():
    logger = _Logger()
    svc = SeparationService(
        _AsyncModel(), logger=logger,
        performance_config={"enabled": True, "gpu_workers": 2, "postprocess_workers": 1,
                            "ordered_postprocess": True})
    try:
        svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    finally:
        svc.close_async_pools()
    timing = [line for line in logger.lines if "[TSE:timing]" in line]
    assert len(timing) == 1
    assert "waiting for Sidon" in timing[0] and "speaker assignment" in timing[0]
    assert timing[0].split("]")[1].split()[0].isdigit() and " 0 window" not in timing[0]
