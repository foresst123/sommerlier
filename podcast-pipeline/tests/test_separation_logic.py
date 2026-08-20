"""Logic tests for TargetExtractionService that need no GPU and no models.

The separator is stubbed, so these check windowing, job grouping, per-track
gating and the dual-channel leakage mask -- not audio quality.

Run:  python -m pytest tests/test_separation_logic.py -q     (from podcast-pipeline/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.audio import AudioData
from schemas.segment import Segment
from services.separation_service import TargetExtractionService

SR = 24000


class FakeTSE:
    """Returns two tracks of a known constant so splices are identifiable."""

    def __init__(self, sim_a=0.6, sim_b=0.6):
        self.sim_a, self.sim_b = sim_a, sim_b
        self.calls = []

    def separate_two_speakers(self, mixture_audio, enroll_A, enroll_B, sample_rate,
                              id_A, id_B, probe_A=None, probe_B=None, core_range=None):
        self.calls.append({
            "len_sec": len(mixture_audio) / sample_rate,
            "probe_A_sec": sum(b - a for a, b in (probe_A or [])) / sample_rate,
            "probe_B_sec": sum(b - a for a, b in (probe_B or [])) / sample_rate,
        })
        return (np.full(len(mixture_audio), 0.5, dtype=np.float32),
                np.full(len(mixture_audio), -0.5, dtype=np.float32),
                self.sim_a, self.sim_b,
                {"anchor_self": 0.6, "anchor_other": 0.1, "other_rms": 0.5})


def _audio(duration=60.0):
    rng = np.random.default_rng(0)
    return AudioData(waveform=rng.normal(0, 0.05, int(duration * SR)).astype(np.float32),
                     sample_rate=SR, name="test", audio_segment=None, duration=duration)


def _dialogue():
    """A talks 0-30s; B has a 0.4s backchannel at 14.0s and a real turn 32-40s."""
    return [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=14.0, end=14.4, speaker="SPEAKER_01"),
        Segment(index="00003", start=32.0, end=40.0, speaker="SPEAKER_01"),
    ]


def test_window_reaches_target_and_contains_both_speakers():
    svc = TargetExtractionService(FakeTSE(), logger=None)
    by_spk = svc._intervals_by_speaker(_dialogue())
    built, reason = svc._build_window(by_spk, "SPEAKER_00", "SPEAKER_01", 14.0, 14.4, 60.0)
    assert built is not None, f"window should be buildable, got {reason}"
    lo, hi, solo_a, solo_b, anchor = built
    assert hi - lo >= 20.0, f"window {hi - lo:.1f}s is shorter than Sidon's chunk"
    # Only one speaker needs solo audio; the anchor is whichever has more.
    assert max(sum(b - a for a, b in solo_a), sum(b - a for a, b in solo_b)) >= 2.0
    assert anchor in ("SPEAKER_00", "SPEAKER_01")


def test_backchannel_gets_a_real_probe_not_the_whole_window():
    fake = FakeTSE()
    svc = TargetExtractionService(fake, logger=None)
    svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    assert len(fake.calls) == 1, "one job expected"
    call = fake.calls[0]
    assert call["len_sec"] >= 20.0
    assert call["probe_B_sec"] >= 2.0, "B must be scored on its solo turn"
    assert call["probe_B_sec"] < call["len_sec"], "probe must be a subset, not the window"


def test_low_scoring_track_does_not_discard_the_good_one():
    # A scores well, B fails QC -- the old code dropped both.
    fake = FakeTSE(sim_a=0.60, sim_b=0.05)
    svc = TargetExtractionService(fake, logger=None)
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    a_seg = next(s for s in out if s.index == "00001")
    b_seg = next(s for s in out if s.index == "00002")
    assert a_seg.tse is True and a_seg.tse_spans, "A's clean extraction must survive"
    assert b_seg.tse is False, "B failed QC and must keep the original mixture"
    assert b_seg.tse_failed_spans, "B's failure must be recorded, not silently dropped"
    assert b_seg.tse_status == "failed"


def test_nearby_overlaps_share_one_separation_call():
    segs = [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=10.0, end=10.4, speaker="SPEAKER_01"),
        Segment(index="00003", start=13.0, end=13.4, speaker="SPEAKER_01"),
        Segment(index="00004", start=32.0, end=40.0, speaker="SPEAKER_01"),
    ]
    fake = FakeTSE()
    svc = TargetExtractionService(fake, logger=None)
    svc.process_overlaps(segs, _audio(), overlap_threshold=0.1)
    assert len(fake.calls) == 1, f"two nearby overlaps should share one call, got {len(fake.calls)}"
    assert svc.stats["pairs"] == 2


def test_sdlm_export_zeroes_unseparated_overlap():
    fake = FakeTSE(sim_a=0.05, sim_b=0.05)   # everything fails QC
    svc = TargetExtractionService(fake, logger=None)
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    t0, _t1 = svc.export_sdlm_dual_channel(out, 60.0, SR, strict=True)
    lo, hi = int(14.0 * SR), int(14.4 * SR)
    assert np.allclose(t0[lo:hi], 0.0), "A's track must not carry B's overlapping speech"
    assert np.abs(t0[int(2.0 * SR):int(3.0 * SR)]).sum() > 0, "solo speech must survive"


def test_sdlm_export_keeps_separated_overlap():
    fake = FakeTSE(sim_a=0.6, sim_b=0.6)     # separation succeeds
    svc = TargetExtractionService(fake, logger=None)
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    t0, _t1 = svc.export_sdlm_dual_channel(out, 60.0, SR, strict=True)
    lo, hi = int(14.05 * SR), int(14.35 * SR)
    assert np.abs(t0[lo:hi]).sum() > 0, "a successfully separated overlap must be kept"


class DuplicatingTSE:
    """Sidon's realistic failure mode on short, lopsided overlaps: the dominant
    speaker is emitted on BOTH tracks. sim_A is high, sim_B is at chance."""

    def separate_two_speakers(self, mixture_audio, enroll_A, enroll_B, sample_rate,
                              id_A, id_B, probe_A=None, probe_B=None, core_range=None):
        a_voice = np.full(len(mixture_audio), 0.5, dtype=np.float32)
        # Identical tracks -> the anchor matches both equally -> no margin.
        return a_voice, a_voice.copy(), 0.46, -0.05, {
            "anchor_self": 0.46, "anchor_other": 0.45, "other_rms": 0.5}


def test_max_rule_would_paste_speaker_a_into_speaker_b():
    """Guard against the max(sim_A, sim_B) QC rule.

    Under max(), 0.46 passes and B's segment gets spliced with A's voice.
    Per-track gating must leave B untouched.
    """
    svc = TargetExtractionService(DuplicatingTSE(), logger=None)
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)

    a_seg = next(s for s in out if s.index == "00001")
    b_seg = next(s for s in out if s.index == "00002")

    assert a_seg.tse is True, "A's track scored 0.46 and should be spliced"
    assert b_seg.tse is False, (
        "B scored -0.05: splicing here would write A's voice into B's segment. "
        "This is exactly what max(sim_A, sim_B) < threshold would allow."
    )
    assert b_seg.tse_failed_spans[0][2] == "qc_sim"


def test_every_overlap_is_accounted_for():
    """The core invariant: no overlap may vanish without a recorded reason."""
    for fake in (FakeTSE(0.6, 0.6), FakeTSE(0.6, 0.05), FakeTSE(0.05, 0.05),
                 DuplicatingTSE()):
        svc = TargetExtractionService(fake, logger=None)
        out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
        n_ok = sum(len(s.tse_spans) for s in out)
        n_bad = sum(len(s.tse_failed_spans) for s in out)
        # one overlap x two segments (A's and B's side)
        assert n_ok + n_bad == 2, (
            f"{type(fake).__name__}: {n_ok} spliced + {n_bad} failed != 2 -- "
            "some code path discarded an overlap without recording it"
        )


def test_window_rejects_three_speakers():
    segs = [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=14.0, end=14.4, speaker="SPEAKER_01"),
        Segment(index="00003", start=16.0, end=18.0, speaker="SPEAKER_02"),
        Segment(index="00004", start=32.0, end=40.0, speaker="SPEAKER_01"),
    ]
    svc = TargetExtractionService(FakeTSE(), logger=None)
    by_spk = svc._intervals_by_speaker(segs)
    built, reason = svc._build_window(by_spk, "SPEAKER_00", "SPEAKER_01", 14.0, 14.4, 60.0)
    assert built is None and reason == "multi_speaker"


def test_no_window_does_not_block_later_overlaps():
    """A skipped overlap must not take the next one down with it."""
    segs = [
        Segment(index="00001", start=0.0, end=55.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=5.0, end=5.3, speaker="SPEAKER_01"),   # B has no solo nearby
        Segment(index="00003", start=50.0, end=50.4, speaker="SPEAKER_01"),
        Segment(index="00004", start=56.0, end=59.0, speaker="SPEAKER_01"), # B's only solo turn
    ]
    svc = TargetExtractionService(FakeTSE(), logger=None)
    out = svc.process_overlaps(segs, _audio(60.0), overlap_threshold=0.1)
    n_ok = sum(len(s.tse_spans) for s in out)
    n_bad = sum(len(s.tse_failed_spans) for s in out)
    assert n_ok + n_bad == 4, "both overlaps (x2 sides) must be accounted for"
    assert n_ok > 0, "the overlap near B's solo turn should still be processed"


class AllRejectTSE:
    """Rejects on the grouped window but succeeds on a single-overlap retry."""

    def __init__(self):
        self.calls = 0

    def separate_two_speakers(self, mixture_audio, enroll_A, enroll_B, sample_rate,
                              id_A, id_B, probe_A=None, probe_B=None, core_range=None):
        self.calls += 1
        sim = 0.05 if self.calls == 1 else 0.6
        return (np.full(len(mixture_audio), 0.5, dtype=np.float32),
                np.full(len(mixture_audio), -0.5, dtype=np.float32), sim, sim,
                {"anchor_self": 0.6, "anchor_other": 0.1, "other_rms": 0.5})


def test_failed_group_job_retries_each_overlap():
    segs = [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=10.0, end=10.4, speaker="SPEAKER_01"),
        Segment(index="00003", start=13.0, end=13.4, speaker="SPEAKER_01"),
        Segment(index="00004", start=32.0, end=40.0, speaker="SPEAKER_01"),
    ]
    fake = AllRejectTSE()
    svc = TargetExtractionService(fake, logger=None)
    out = svc.process_overlaps(segs, _audio(), overlap_threshold=0.1)
    assert svc.stats["retried"] == 1, "the rejected group job should have been split"
    assert fake.calls > 1, "retry must actually re-run separation"
    assert sum(len(s.tse_spans) for s in out) > 0, "retries should recover the overlaps"


def test_sdlm_mask_uses_failed_spans():
    svc = TargetExtractionService(FakeTSE(0.05, 0.05), logger=None)
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    assert all(s.tse_failed_spans or not s.tse_spans for s in out if s.index == "00002")
    t0, _ = svc.export_sdlm_dual_channel(out, 60.0, SR, strict=True)
    lo, hi = int(14.0 * SR), int(14.4 * SR)
    assert np.allclose(t0[lo:hi], 0.0)
