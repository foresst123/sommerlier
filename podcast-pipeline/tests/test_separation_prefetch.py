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


def test_the_profile_splits_a_files_separation_time_and_nothing_is_logged():
    logger = _Logger()
    svc = SeparationService(
        _AsyncModel(), logger=logger,
        performance_config={"enabled": True, "gpu_workers": 2, "postprocess_workers": 1,
                            "ordered_postprocess": True})
    try:
        svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    finally:
        svc.close_async_pools()
    profile = svc.profile_snapshot()
    assert profile["windows"] >= 1 and profile["gpu_calls"] == profile["windows"]
    assert profile["consumer"] >= profile["raw_wait"] + profile["post"] - 1e-6
    assert profile["other"] == pytest.approx(
        profile["consumer"] - profile["raw_wait"] - profile["post"])
    assert profile["gpu_run"] >= 0 and profile["gpu_queue"] >= 0
    assert not [line for line in logger.lines if "[TSE:timing]" in line], (
        "the timings belong in the performance file, not the console log")


def test_the_profile_includes_the_assignment_and_sidon_counters_of_the_model():
    import collections

    class Timed(_AsyncModel):
        timing = collections.Counter(enrollment=0.5, probe_vad=3.0, probe_vad_calls=8,
                                     wespeaker=9.0, wespeaker_calls=8, sidon_calls=4,
                                     sidon_total=8.0, sidon_infer=6.0)

    svc = SeparationService(
        Timed(), logger=None,
        performance_config={"enabled": True, "gpu_workers": 2, "postprocess_workers": 1,
                            "ordered_postprocess": True})
    try:
        svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    finally:
        svc.close_async_pools()
    profile = svc.profile_snapshot()
    assert profile["probe_vad"] == 3.0 and profile["wespeaker_calls"] == 8.0
    assert profile["sidon_infer"] == 6.0


# --- speaker assignment of later windows runs while the consumer handles earlier ones ---

def _many_overlaps(count=6):
    """One long speaker with `count` short interjections, each its own overlap."""
    segments = [Segment(index="00000", start=0.0, end=count * 10.0 + 10.0,
                        speaker="SPEAKER_00")]
    for i in range(count):
        start = 10.0 * (i + 1)
        segments.append(Segment(index=f"{i + 1:05d}", start=start, end=start + 0.6,
                                speaker="SPEAKER_01"))
    return segments


class _ConcurrencyModel(_AsyncModel):
    """Counts how many windows are in speaker assignment at the same moment."""

    def __init__(self):
        super().__init__()
        import threading
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.order = []

    def postprocess_separated(self, mixture_audio, raw_tracks, enroll_A, enroll_B,
                              sample_rate, id_A, id_B, probe_A=None, probe_B=None,
                              core_range=None):
        import time
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.order.append(len(mixture_audio))
        try:
            time.sleep(0.15)          # the assignment step: mostly waiting on workers
            return super().postprocess_separated(
                mixture_audio, raw_tracks, enroll_A, enroll_B, sample_rate, id_A, id_B,
                probe_A, probe_B, core_range)
        finally:
            with self.lock:
                self.active -= 1


def _run(model, **config):
    svc = SeparationService(
        model, logger=None,
        performance_config={"enabled": True, "gpu_workers": 3, "postprocess_workers": 4,
                            "gpu_prefetch_per_worker": 2, "ordered_postprocess": True,
                            **config})
    try:
        out = svc.process_overlaps(_many_overlaps(), _audio(80.0), overlap_threshold=0.1)
    finally:
        svc.close_async_pools()
    return svc, out


def test_assignment_of_later_windows_starts_before_the_consumer_reaches_them():
    """The consumer used to submit each window's assignment and wait for it before
    touching the next, so one file never had two windows in assignment at once and
    the GPU workers sat idle behind it."""
    model = _ConcurrencyModel()
    _svc, out = _run(model)

    assert len(model.order) >= 4
    assert model.peak >= 3, f"windows were assigned one at a time (peak {model.peak})"


def test_assigning_ahead_changes_nothing_in_the_result():
    ahead_model, plain_model = _ConcurrencyModel(), _ConcurrencyModel()
    _ahead_svc, ahead = _run(ahead_model)
    _plain_svc, plain = _run(plain_model, postprocess_ahead=False)

    assert plain_model.peak == 1                    # the old behaviour, still selectable
    assert [(s.index, s.bss_spans) for s in ahead] == [
        (s.index, s.bss_spans) for s in plain]
    for a, b in zip(ahead, plain):
        assert np.array_equal(a.audio, b.audio)


def test_windows_whose_enrollment_depends_on_earlier_ones_are_still_assigned_in_order():
    """Enrollment memory feeds each window's accepted output into the next window's
    enrollment, so nothing may be assigned ahead when it is on."""
    from utils.enrollment_memory import EnrollmentMemory
    model = _ConcurrencyModel()
    svc = SeparationService(
        model, logger=None,
        performance_config={"enabled": True, "gpu_workers": 3, "postprocess_workers": 4,
                            "gpu_prefetch_per_worker": 2, "ordered_postprocess": True})
    svc.memory = EnrollmentMemory(enabled=True)
    try:
        svc.process_overlaps(_many_overlaps(), _audio(80.0), overlap_threshold=0.1)
    finally:
        svc.close_async_pools()
    assert model.peak == 1


def test_a_failing_assignment_still_fails_only_its_own_window():
    class Flaky(_ConcurrencyModel):
        calls = 0

        def postprocess_separated(self, *args, **kwargs):
            Flaky.calls += 1
            if Flaky.calls == 2:
                raise RuntimeError("worker died")
            return super().postprocess_separated(*args, **kwargs)

    Flaky.calls = 0
    svc, out = _run(Flaky())
    assert len(out) == len(_many_overlaps())        # nothing lost, nothing raised


# --- Sidon keeps going while what follows it is behind -----------------------------------

class _StalledDownstream(_AsyncModel):
    """Sidon returns at once; speaker assignment of the first window does not."""

    def __init__(self):
        super().__init__()
        import threading
        self.raw_done = 0
        self.lock = threading.Lock()
        self.release = threading.Event()

    def separate_raw(self, audio, sample_rate):
        with self.lock:
            self.raw_done += 1
        return super().separate_raw(audio, sample_rate)

    def postprocess_separated(self, *args, **kwargs):
        self.release.wait(10)
        return super().postprocess_separated(*args, **kwargs)


def _sidon_runs_ahead(prefetch_per_worker):
    import threading
    import time
    model = _StalledDownstream()
    svc = SeparationService(
        model, logger=None,
        performance_config={"enabled": True, "gpu_workers": 1, "postprocess_workers": 1,
                            "gpu_prefetch_per_worker": prefetch_per_worker,
                            "ordered_postprocess": True})
    worker = threading.Thread(target=lambda: svc.process_overlaps(
        _many_overlaps(8), _audio(120.0), overlap_threshold=0.1))
    worker.start()
    # Everything Sidon can do while downstream is stuck: wait until the count stops
    # moving (no fixed sleep, which flaked when the machine was busy).
    settled_since, last = time.monotonic(), -1
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        current = model.raw_done
        if current != last:
            last, settled_since = current, time.monotonic()
        elif time.monotonic() - settled_since >= 0.5 and current > 0:
            break
        time.sleep(0.02)
    ahead = model.raw_done
    model.release.set()
    worker.join(30)
    svc.close_async_pools()
    return ahead


def test_sidon_is_only_held_back_by_the_configured_lookahead():
    assert _sidon_runs_ahead(2) == 2


def test_with_a_deep_lookahead_sidon_finishes_every_window_while_downstream_is_stuck():
    assert _sidon_runs_ahead(64) == 8


def test_a_lookahead_of_zero_means_no_limit_at_all():
    assert _sidon_runs_ahead(0) == 8
