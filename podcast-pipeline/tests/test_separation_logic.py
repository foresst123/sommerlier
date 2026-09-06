"""Logic tests for TargetExtractionService that need no GPU and no models.

The separator is stubbed, so these check windowing, job grouping, per-track
gating and the dual-channel leakage mask -- not audio quality.

Run:  python -m pytest tests/test_separation_logic.py -q     (from podcast-pipeline/)
"""
import os
import sys

import numpy as np
import pytest

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




def test_a_backchannel_window_is_the_overlap_widened_to_the_model_window():
    """USEF is told who to extract by its enrollment, so the mixture carries
    the overlap and nothing else -- widened to the 2s the ONNX graph takes.

    This replaces a window of 5s solo A + 5s solo B + the overlap. That
    existed for DialogueSidon, which separates blind and collapsed to "one
    source carries everything" when the two speakers were unbalanced. A
    target-conditioned masker never needed the solo audio."""
    fake = FakeTSE()
    svc = TargetExtractionService(fake, logger=None)
    svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    assert len(fake.calls) == 1, "one job expected"
    call = fake.calls[0]
    assert call["len_sec"] == pytest.approx(2.0, abs=0.05), (
        f"window is {call['len_sec']:.2f}s; it should be the model's 2s")


def test_an_ordered_backend_scores_the_whole_track_not_a_probe():
    """With no solo speech in the window there is nothing to probe with, and
    nothing to probe for: track 1 IS speaker A by construction. An empty probe
    reads downstream as "score the whole track", which is the right question
    to ask of a track that should be one speaker end to end."""
    fake = FakeTSE()
    svc = TargetExtractionService(fake, logger=None)
    svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    call = fake.calls[0]
    assert not call.get("probe_A"), "no solo audio is fed any more"
    assert not call.get("probe_B")


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


def test_old_checkpoint_unpickles_without_new_fields():
    """Resuming a run checkpointed before tse_spans/tse_failed_spans existed.

    Pickle restores __dict__ directly and does not apply dataclass defaults, so
    without __setstate__ the first append raises AttributeError mid-pipeline.
    """
    import pickle
    from schemas.segment import EnhancedSegment

    seg = EnhancedSegment(index="00001", start=0.0, end=1.0, speaker="SPEAKER_00")
    del seg.__dict__["tse_spans"]
    del seg.__dict__["tse_failed_spans"]

    loaded = pickle.loads(pickle.dumps(seg))
    loaded.tse_failed_spans.append((0.0, 1.0, "qc_sim", "sim=0.10"))
    assert loaded.tse_status == "failed"

    svc = TargetExtractionService(FakeTSE(), logger=None)
    loaded.enhanced_audio = np.ones(SR, dtype=np.float32)
    t0, _ = svc.export_sdlm_dual_channel([loaded], 2.0, SR, strict=True)
    assert np.allclose(t0[:SR], 0.0), "failed span must still be masked after unpickling"


# --- stitched window + splice-site QC ---------------------------------------

def _svc():
    return TargetExtractionService.__new__(TargetExtractionService)


def _diar(spans):
    return [Segment(index=str(i).zfill(5), start=a, end=b, speaker=s)
            for i, (a, b, s) in enumerate(spans)]


def _buried_case():
    """A 0.34s backchannel inside a long turn -- the shape that broke Sidon."""
    segs = _diar([(730.0, 758.4, "2"), (758.4, 787.7, "1"),
                  (777.33, 777.67, "2"), (788.2, 816.5, "1")])
    svc = _svc()
    svc.logger = None
    return svc, svc._intervals_by_speaker(segs)








def test_qc_rejects_a_track_that_is_silent_where_the_mixture_speaks():
    """The observed failure: sim scored 0.67 on solo audio 19s away while the
    track held silence across the backchannel and the other speaker after it."""
    rng = np.random.default_rng(0)
    n = int(0.34 * SR)
    host = (rng.standard_normal(n) * 0.12).astype(np.float32)

    # Silent across the backchannel, the other speaker's tail at the end --
    # the measured shape of the real failure (00046: 11 of 17 frames flat).
    bad = np.zeros(n, dtype=np.float32)
    tail = int(n * 0.72)
    bad[tail:] = rng.standard_normal(n - tail) * 0.08
    assert TargetExtractionService._track_has_speech(host, bad) is False
    # Whole-clip RMS is far above the silence threshold, which is why the
    # existing gate let this through.
    assert float(np.sqrt(np.mean(bad ** 2))) > 0.002

    good = (host * 0.5).astype(np.float32)
    assert TargetExtractionService._track_has_speech(host, good) is True


def test_qc_accepts_a_quieter_track_and_rejects_an_empty_one():
    rng = np.random.default_rng(1)
    n = int(0.5 * SR)
    host = (rng.standard_normal(n) * 0.15).astype(np.float32)
    assert TargetExtractionService._track_has_speech(host, host * 0.08) is True
    assert TargetExtractionService._track_has_speech(host, np.zeros(n, np.float32)) is False
    # Nothing to preserve where the mixture is silent.
    z = np.zeros(n, dtype=np.float32)
    assert TargetExtractionService._track_has_speech(z, z) is True


def test_the_run_report_can_actually_be_built():
    """It could not, and no test noticed for a whole change set.

    _report_payload named TSE_MIN_SOLO, TSE_WINDOW_TARGET and TSE_WINDOW_MAX,
    three constants deleted with the Sidon window strategy. Every test called
    process_overlaps directly; only the pipeline calls this, and only after
    separation finishes -- so the run reached the end of the separation stage
    and died there with NameError, on both files, after 34 minutes."""
    fake = FakeTSE()
    svc = TargetExtractionService(fake, logger=None)
    svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)

    payload = svc.report_payload()
    assert set(payload) >= {"thresholds", "music_map", "stats", "failures"}
    assert payload["thresholds"]["model_window"] == 2.0


def test_no_module_reads_a_constant_nobody_defines():
    """The static version of the same failure: a deleted constant that some
    other line still names shows up only when that line runs."""
    import ast
    import builtins

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dangling = []
    for folder, _dirs, files in os.walk(root):
        if any(skip in folder for skip in ("__pycache__", ".git", "tests")):
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except SyntaxError:
                continue
            bound = set(dir(builtins))
            for node in ast.walk(tree):
                if isinstance(node, ast.alias):
                    bound.add((node.asname or node.name).split(".")[0])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bound.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    bound.add(node.id)
                elif isinstance(node, ast.arg):
                    bound.add(node.arg)
            for node in ast.walk(tree):
                # Constants only: lowercase names are far more likely to be a
                # false positive from a scope this walk does not model.
                if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        and node.id.isupper() and node.id not in bound):
                    dangling.append(f"{os.path.relpath(path, root)}:{node.lineno} {node.id}")
    assert not dangling, "constants used but never defined:\n  " + "\n  ".join(sorted(set(dangling)))
