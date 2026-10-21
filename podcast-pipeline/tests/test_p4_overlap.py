"""Music decode-ahead and diarization tail completion (plan P4).

Fakes only: no model, GPU or pyannote is needed.
Run:  python -m pytest tests/test_p4_overlap.py -q   (from podcast-pipeline/)
"""
import os
import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# services/diarization_service.py imports pyannote.core at module scope; the
# code under test here never touches it, so a stub keeps these tests runnable
# where the real dependency is not installed.
def _stub_missing(name):
    try:
        __import__(name)
        return
    except ImportError:
        pass
    module = types.ModuleType(name)
    module.__path__ = []

    def _getattr(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return type(attr, (), {})

    module.__getattr__ = _getattr
    sys.modules[name] = module
    _stubbed.append(name)


_stubbed = []
for _name in ("pyannote", "pyannote.core", "pyannote.audio"):
    _stub_missing(_name)

from schemas.audio import AudioData
from services.music_service import MusicService
from utils.music_map import MUSIC, MusicMap

SR = 1000


def _audio(seconds=40):
    wave = np.linspace(-0.5, 0.5, seconds * SR, dtype=np.float32)
    return AudioData(name="t", waveform=wave, sample_rate=SR,
                     duration=float(seconds), audio_segment=None)


class _Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.decodes = []
        self.alive = 0
        self.max_alive = 0
        self.checked_out_during_decode = 0
        self.busy = 0

    def decoded(self):
        with self.lock:
            self.decodes.append(1)
            self.alive += 1
            self.max_alive = max(self.max_alive, self.alive)

    def posted(self):
        with self.lock:
            self.alive -= 1


class _HiResModel:
    """Split-interface separator whose 'decode' is a deterministic array."""

    def __init__(self, stats, fail_post=False, raw_delay=0.0):
        self.stats, self.fail_post, self.raw_delay = stats, fail_post, raw_delay

    def decode_span(self, source_path, start, end):
        with self.stats.lock:
            # a decode running while every instance is busy is the point
            self.stats.checked_out_during_decode += int(self.stats.busy > 0)
        self.stats.decoded()
        n = int((end - start) * 2 * SR)
        return np.full(n, start, dtype=np.float32), 2 * SR

    def separate_span_raw(self, source_path, start, end, decoded=None):
        if decoded is None:
            decoded = self.decode_span(source_path, start, end)
        mix, sr = decoded
        with self.stats.lock:
            self.stats.busy += 1
        try:
            time.sleep(self.raw_delay)
        finally:
            with self.stats.lock:
                self.stats.busy -= 1
        return {"mix": mix, "sr": sr, "start": start, "end": end}

    def separate_span_postprocess(self, context, reference, out_sr):
        try:
            if self.fail_post:
                raise RuntimeError("post boom")
            return (np.full(len(reference), context["start"], dtype=np.float32) * 0.5)
        finally:
            self.stats.posted()

    def separate_raw(self, audio, sr):
        return (np.asarray(audio, dtype=np.float32), sr, False)

    def postprocess_separated(self, raw, audio, sr):
        return raw[0]


class _Pool:
    def __init__(self, models):
        self.models = models


def _spans(n):
    return MusicMap([(2.0 + 4 * k, 4.0 + 4 * k, MUSIC) for k in range(n)])


def _run(ahead, n=6, models=2, **model_kw):
    stats = _Stats()
    pool = _Pool([_HiResModel(stats, **model_kw) for _ in range(models)])
    svc = MusicService(pool, logger=None, performance_config={
        "enabled": True, "ordered_postprocess": True, "postprocess_workers": 2,
        "hires_decode_ahead": ahead})
    try:
        patches = svc.strip_music_spans(_audio(), _spans(n), source_path="src.wav")
    finally:
        svc.close_async_pools()
    return patches, stats


def test_decode_ahead_gives_the_same_patches_and_decodes_each_span_once():
    base, base_stats = _run(0)
    ahead, ahead_stats = _run(4)
    assert len(base) == len(ahead) == 6
    for (s1, p1), (s2, p2) in zip(base, ahead):
        assert s1 == s2 and np.array_equal(p1, p2)
    assert len(base_stats.decodes) == len(ahead_stats.decodes) == 6


def test_decoded_spans_alive_at_once_never_exceed_the_bound():
    _, stats = _run(2, n=10, raw_delay=0.01)
    assert stats.max_alive <= 2
    assert stats.alive == 0, "every decoded span was consumed by postprocess"


def test_decode_does_not_wait_for_a_free_separator():
    _, stats = _run(4, n=8, models=1, raw_delay=0.05)
    assert stats.checked_out_during_decode > 0


def test_a_failing_postprocess_still_drains_every_job_before_raising():
    stats = _Stats()
    pool = _Pool([_HiResModel(stats, fail_post=True, raw_delay=0.01)])
    svc = MusicService(pool, logger=None, performance_config={
        "enabled": True, "postprocess_workers": 2, "hires_decode_ahead": 3})
    audio = _audio()
    with pytest.raises(RuntimeError, match="post boom"):
        svc.strip_music_spans(audio, _spans(6), source_path="src.wav")
    # nothing may still be running against the waveform, and no slot leaked
    assert stats.alive == 0
    state = svc._async_state
    slots = state["decode_slots"]
    for _ in range(3):
        assert slots.acquire(blocking=False)
    svc.close_async_pools()


def test_decode_ahead_is_off_by_default_and_without_a_decoder():
    stats = _Stats()
    model = _HiResModel(stats)
    svc = MusicService(_Pool([model]), logger=None,
                       performance_config={"enabled": True})
    assert svc._decode_runtime([model]) is None
    plain = SimpleNamespace(separate_raw=lambda a, s: None)
    svc2 = MusicService(_Pool([plain]), logger=None,
                        performance_config={"enabled": True, "hires_decode_ahead": 4})
    assert svc2._decode_runtime([plain]) is None


def test_remover_decode_span_is_the_same_decode_separate_span_raw_uses(monkeypatch, tmp_path):
    from models import bs_roformer as bsr

    src = tmp_path / "a.wav"
    src.write_bytes(b"x")
    calls = []

    def fake_load(path, sr, mono, offset, duration):
        calls.append((offset, duration))
        return np.ones((2, 100), dtype=np.float32), sr

    import librosa
    monkeypatch.setattr(librosa, "load", fake_load)
    remover = bsr.BSRoformerRemover(device=None, logger=None)
    remover.separate_raw = lambda mix, sr: (mix, sr, True)
    decoded = remover.decode_span(str(src), 1.0, 3.0)
    assert decoded[0].shape == (100, 2) and decoded[1] == bsr.NATIVE_SAMPLE_RATE
    ctx = remover.separate_span_raw(str(src), 1.0, 3.0, decoded=decoded)
    assert ctx["mix"] is decoded[0] and len(calls) == 1, "no second decode"
    assert remover.separate_span_raw(str(src), 1.0, 3.0)["mix"].shape == (100, 2)
    assert len(calls) == 2


# --- diarization tail --------------------------------------------------------

from services.diarization_service import DiarizationService  # noqa: E402
import services.diarization_service as _diar_module  # noqa: E402

# Do not leave stubs behind for other test files that probe for pyannote.
for _name in _stubbed:
    sys.modules.pop(_name, None)
if _stubbed:
    for _name in ("services.diarization_service", "algorithms.diarization.fusion"):
        sys.modules.pop(_name, None)


def _svc(**perf):
    svc = DiarizationService(performance_config=perf)
    result = SimpleNamespace(segments=[1], method="fake")
    svc.diarize_postprocess = lambda raw, audio, args: result
    return svc, result


def test_the_completion_future_covers_the_tail():
    svc, result = _svc()
    gate, started = threading.Event(), threading.Event()

    def tail(res):
        started.set()
        gate.wait(5)

    future = svc.submit_postprocess(None, None, None, then=tail)
    assert started.wait(2)
    assert not future.done(), "postprocess finished but the tail is still running"
    gate.set()
    assert future.result(timeout=2) is result
    svc.close_postprocess_pool()


def test_a_failing_tail_reaches_the_drain_as_a_failure():
    from utils.batch import _drain_pending_diarization

    svc, _ = _svc()

    def tail(res):
        raise RuntimeError("checkpoint boom")

    pipeline = SimpleNamespace(_pending_diar_jobs={
        "a.mp3": svc.submit_postprocess(None, None, None, then=tail)})
    failures = {}
    _drain_pending_diarization(pipeline, failures)
    assert "checkpoint boom" in failures["a.mp3"]
    assert not pipeline._pending_diar_jobs
    svc.close_postprocess_pool()


def test_without_a_tail_submit_postprocess_is_unchanged():
    svc, result = _svc()
    assert svc.submit_postprocess(None, None, None).result(timeout=2) is result
    svc.close_postprocess_pool()


class _FakeVad:
    def __init__(self):
        self.forks = 0

    def fork(self):
        self.forks += 1
        return _FakeVad()


def test_each_postprocess_thread_gets_its_own_vad_when_asked():
    vad = _FakeVad()
    svc = DiarizationService(vad_model=vad, performance_config={"vad_per_worker": True})
    seen = {}

    def grab(name):
        first, guard1 = svc._vad_for_thread()
        again, _ = svc._vad_for_thread()
        seen[name] = (first, again, guard1)

    threads = [threading.Thread(target=grab, args=(n,)) for n in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    a, b = seen["a"], seen["b"]
    assert a[0] is a[1] and b[0] is b[1], "one instance per thread, reused"
    assert a[0] is not b[0] and a[0] is not vad
    assert a[2] is not svc._vad_lock


def test_the_shared_vad_and_its_lock_stay_the_default():
    vad = _FakeVad()
    svc = DiarizationService(vad_model=vad)
    assert svc._vad_for_thread() == (vad, svc._vad_lock)
    # a model that cannot fork stays shared even when asked
    plain = SimpleNamespace()
    svc2 = DiarizationService(vad_model=plain, performance_config={"vad_per_worker": True})
    assert svc2._vad_for_thread() == (plain, svc2._vad_lock)


def test_the_boundary_finder_calls_the_vad_under_its_guard():
    _GuardedVad = _diar_module._GuardedVad

    held = []

    class Guard:
        def __enter__(self):
            held.append(True)

        def __exit__(self, *a):
            held.append(False)

    class Vad:
        def get_speech_timestamps(self, x, **kw):
            assert held and held[-1] is True
            return [x, kw]

    assert _GuardedVad(Vad(), Guard()).get_speech_timestamps(1, a=2) == [1, {"a": 2}]
    assert held == [True, False]


def test_concurrent_prefetch_requests_for_one_file_build_one_plan(monkeypatch):
    from services.separation_service import SeparationService

    builds = []
    gate = threading.Event()

    def slow_build(self, segments, audio, overlap_threshold=0.1):
        builds.append(1)
        gate.wait(2)
        return "plan"

    monkeypatch.setattr(SeparationService, "_build_overlap_plan", slow_build)
    svc = SeparationService(SimpleNamespace(), logger=None)
    threads = [threading.Thread(
        target=svc.prefetch_overlap_plan, args=([object()], _audio(1), "a.mp3"))
        for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    gate.set()
    assert svc._take_prefetched_plan("a.mp3") == "plan"
    assert len(builds) == 1
    svc.close_prefetch_pool()
