"""Regression cases for bounded windows and per-target recovery; no real models."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.audio import AudioData
from schemas.segment import Segment
from services.separation_service import SeparationService
from utils.acoustic_boundary import AcousticBoundaryFinder, ContextExpander
from utils.separation_quality import continuity_evidence, track_quality
from utils.separation_window import Piece, Window, WindowPlanner
from utils.window_pool import FileWindows


def test_qc_counts_the_last_partial_frame():
    host = np.full(59, 0.1)
    track = np.zeros(59)
    track[40:] = 0.1
    quality = track_quality(host, track, 1000)
    assert quality["frame_lengths"] == [20, 20, 19]
    assert quality["coverage"] == pytest.approx(19 / 59)
    assert quality["status"] == "insufficient_evidence"
    assert quality["accepted"] is False


def test_short_quiet_mixture_and_nonfinite_track_are_distinct():
    quiet = np.zeros(30)
    assert track_quality(quiet, quiet, 1000)["status"] == "quiet_mixture"
    assert track_quality(quiet, np.full(30, np.nan), 1000)["status"] == "nonfinite_audio"


@pytest.mark.parametrize("context", [0, 5, 20])
def test_stitch_fades_never_touch_core_and_length_prediction_is_exact(context):
    wave = np.random.default_rng(7).normal(0, 0.1, 6000).astype(np.float32)
    planner = WindowPlanner([], [], wave, 1000)
    prefix = (Piece(0, 1500, "A", "left", "support"),)
    suffix = (Piece(4000, 5500, "B", "right", "support"),)
    base = Piece(2000 - context, 2020 + context, "A+B", "base", "base")
    result = planner._assemble(prefix, base, suffix, ("A", "B"), 2000, 2020,
                               {"A": [], "B": []})
    assert np.array_equal(result.audio[slice(*result.core)], wave[2000:2020])
    assert result.layout["join_crossfade_samples"] == [min(20, context)] * 2
    assert len(result.audio) == planner._window_samples(prefix, base, suffix, 2000, 2020)


def test_context_maximum_cannot_be_overridden_by_minimum():
    finder = AcousticBoundaryFinder(np.ones(5000, np.float32), 1000)
    decision = ContextExpander(finder).expand(
        2500, "left", 2.0, minimum_seconds=2.0, maximum_seconds=0.2)
    assert 2300 <= decision.boundary_sample <= 2500
    with pytest.raises(ValueError):
        finder.find_cut(2000, search_min=3000, search_max=3500, hard_bounds=(1000, 2500))


def test_partially_consumed_support_retains_its_clean_remainder(monkeypatch):
    planner = WindowPlanner([], [], np.zeros(12000, np.float32), 1000)
    support = Piece(0, 10000, "A", "support", "support")
    monkeypatch.setattr(planner, "_support", lambda *args: [support])
    monkeypatch.setattr(planner.cuts, "analyse", lambda lo, hi: (
        {lo: "segment", lo + 100: "energy_pause", hi: "segment"}, [], "energy"))
    base = Piece(0, 2000, "A+B", "base", "base")
    options = planner._sequences("A", 1000, 0, 8000, base)
    assert any(seq and seq[0].start == 2100 and seq[0].end == 10000 for seq in options)


def test_context_fill_keeps_the_selected_padding():
    from algorithms.diarization.overlap import detect_overlapping_segments

    segments = [Segment("A1", 10, 20, "A"), Segment("B1", 15, 16, "B"),
                Segment("A2", 0, 5, "A"), Segment("B2", 30, 35, "B")]
    pairs = detect_overlapping_segments([s.__dict__ for s in segments], overlap_threshold=0)
    t = np.arange(40000) / 1000
    wave = (0.1 * np.sin(2 * np.pi * 83 * t)).astype(np.float32)
    wave[(t % 0.5) < 0.08] = 0
    planner = WindowPlanner(segments, pairs, wave, 1000)
    window = planner.build(pairs)
    assert window is not None
    initial = next(a for a in window.layout["actions"] if a["action"] == "select_clean_padding")
    selected = initial["prefix_source_samples"] + initial["suffix_source_samples"]
    final = [p["source_samples"] for p in window.layout["pieces"] if p["kind"] == "support"]
    assert selected and selected == final
    assert len(window.audio) <= 15000
    for action in window.layout["actions"]:
        if action["action"] == "expand_background_context":
            before, after = action["base_before"], action["base_after"]
            requested = action["requested_left_seconds"] + action["requested_right_seconds"]
            assert after[1] - after[0] - (before[1] - before[0]) <= round(requested * 1000)


def test_one_failed_core_does_not_discard_other_plans(monkeypatch):
    planner = WindowPlanner([], [], np.zeros(20000, np.float32), 1000)
    monkeypatch.setattr(planner, "_split_core_bounds", lambda *args: [(0, 5000), (5000, 10000)])
    expected = object()

    def build(group, core_bounds, initial_actions):
        if core_bounds[0] == 0:
            raise RuntimeError("first core failed")
        planner.reason, planner.detail = "ok", "ok"
        planner.actions = initial_actions
        return expected

    monkeypatch.setattr(planner, "build", build)
    plans = planner.build_many([{"overlap_start": 0, "overlap_end": 10}])
    assert plans[0].window is None and plans[0].reason == "window_error"
    assert plans[1].window is expected


def test_continuity_detects_swapped_tracks_on_shared_source_samples():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, 1000)).astype(np.float32)
    window = Window(a + b, (200, 400), {}, {
        "pieces": [{"kind": "base", "source_samples": [1000, 2000],
                    "window_samples": [0, 1000]}], "join_crossfade_samples": []})
    previous = {"bounds": (1000, 2000), "tracks": (a, b)}
    evidence = continuity_evidence(previous, window, (b, a), 1000)
    assert evidence["status"] == "measured"
    assert evidence["swap"] is True
    assert continuity_evidence(None, window, (b, a), 1000)["swap"] is False


def test_worker_exception_falls_back_and_keeps_later_jobs(monkeypatch):
    from utils import window_pool

    class FailedFuture:
        def result(self):
            raise RuntimeError("worker stopped")

    fw = object.__new__(FileWindows)
    fw._pool = SimpleNamespace(submit=lambda *args: FailedFuture())
    fw._ctx, fw._futures = {}, []
    monkeypatch.setattr(window_pool, "_build_job", lambda ctx, group: ([], "ok", str(group), []))
    results = list(fw.build_all(["first", "second"]))
    assert [r[2] for r in results] == ["first", "second"]
    assert all(r[3][0]["action"] == "worker_fallback_sequential" for r in results)


def test_retry_preserves_the_already_spliced_speaker(monkeypatch, tmp_path):
    from services import separation_service
    import json

    monkeypatch.setattr(separation_service, "_worth_pooling", lambda count: False)

    class RecoverSecondSpeaker:
        calls = 0

        def separate_two_speakers(self, mixture_audio, **kwargs):
            self.calls += 1
            a = np.full_like(mixture_audio, 0.2 if self.calls == 1 else 0.8)
            b = np.full_like(mixture_audio, 0.0 if self.calls == 1 else -0.2)
            return a, b, None, None, {}

    sr = 1000
    segments = [Segment("A1", 0, 20, "A"), Segment("B1", 10, 10.4, "B")]
    wave = np.random.default_rng(0).normal(0, 0.05, 20000).astype(np.float32)
    audio = AudioData(waveform=wave, sample_rate=sr, name="recovery",
                      audio_segment=None, duration=20)
    model = RecoverSecondSpeaker()
    svc = SeparationService(model, logger=None)
    svc.dump_dir = str(tmp_path)
    output = svc.process_overlaps(segments, audio)
    assert model.calls == 2
    assert svc.stats["retried"] == 1
    assert all(len(seg.bss_spans) == 1 and not seg.bss_failed_spans for seg in output)
    gain = svc.window_layouts[0]["level_calibration"]["A"]["gain"]
    assert output[0].audio[10200] == pytest.approx(0.2 * gain)
    assert svc.failure_artifacts[0]["final_outcome"] == "recovered"
    path = tmp_path / svc.failure_artifacts[0]["path"] / "metadata.json"
    assert json.loads(path.read_text())["recovery"]["outcome"] == "recovered"
