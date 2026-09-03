"""Growing enrollments from the separations that went well, and fusing the
overlaps that were too close together to be balanced.

Both come from the same measurement. Similarity against the mined enrollment
sat flat at 0.58 -- flat with span length and with segment length, so a ceiling
rather than noise -- and the spans whose window held under 40% target audio
scored 0.518 against 0.613 for the rest. Roughly half the spans were paying
that.

Run:  python -m pytest tests/test_enrollment_memory.py -q   (from podcast-pipeline/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.enrollment_memory import EnrollmentMemory

SR = 24000


def _clip(seconds=1.0, seed=0):
    return np.random.default_rng(seed).standard_normal(int(SR * seconds)) * 0.1


def _memory(**kw):
    kw.setdefault("enabled", True)
    return EnrollmentMemory(**kw)


# --- the switch -------------------------------------------------------------

def test_it_is_off_unless_asked_for():
    """It changes what the separator is conditioned on, so it is opt-in."""
    off = EnrollmentMemory(enabled=False)
    assert not off.offer("1", _clip(), 0.9, SR)
    assert off.extend("1", ["mined"], SR) == ["mined"]


def test_both_profiles_declare_the_setting():
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for name, profile in config["environments"].items():
        assert "enrollment_memory" in profile["models"]["tse"], name


# --- what gets in -----------------------------------------------------------

def test_a_well_separated_track_is_kept():
    memory = _memory()
    assert memory.offer("1", _clip(), 0.80, SR)


def test_a_poorly_separated_track_is_refused():
    """Usable is a lower bar than exemplary; only the second belongs here."""
    memory = _memory()
    assert not memory.offer("1", _clip(), 0.30, SR)


def test_a_track_just_above_the_qc_gate_is_still_refused():
    """0.25 passes QC (0.20) but must not become reference audio."""
    memory = _memory()
    assert not memory.offer("1", _clip(), 0.25, SR)


def test_a_silent_track_is_refused_however_it_scored():
    memory = _memory()
    assert not memory.offer("1", np.zeros(SR), 0.95, SR)


def test_a_clip_too_short_to_embed_is_refused():
    memory = _memory()
    assert not memory.offer("1", _clip(0.2), 0.95, SR)


def test_a_missing_similarity_is_not_treated_as_zero_or_as_pass():
    """Spans with no solo region score None; they are unjudged, not bad."""
    memory = _memory()
    assert not memory.offer("1", _clip(), None, SR)


# --- what it does with them -------------------------------------------------

def test_the_mined_enrollment_is_extended_never_replaced():
    """The worst case has to be the original behaviour."""
    memory = _memory()
    memory.offer("1", _clip(seed=1), 0.9, SR)
    mined = ["mined-a", "mined-b"]
    grown = memory.extend("1", mined, SR)
    assert grown[:2] == mined
    assert len(grown) == 3


def test_a_speaker_with_nothing_earned_gets_its_enrollment_back():
    memory = _memory()
    mined = ["mined"]
    assert memory.extend("2", mined, SR) is mined


def test_memory_is_per_speaker():
    memory = _memory()
    memory.offer("1", _clip(seed=2), 0.9, SR)
    assert len(memory.extend("1", [], SR)) == 1
    assert memory.extend("2", [], SR) == []


def test_the_budget_keeps_the_strongest_clips():
    """Over budget, similarity decides what stays."""
    memory = _memory(budget=2.0)
    memory.offer("1", _clip(1.0, seed=3), 0.70, SR)
    memory.offer("1", _clip(1.0, seed=4), 0.95, SR)
    memory.offer("1", _clip(1.0, seed=5), 0.80, SR)
    held = memory.extend("1", [], SR)
    assert len(held) == 2, "budget should cap the stored audio"


def test_one_clip_is_kept_even_when_it_exceeds_the_budget():
    memory = _memory(budget=0.5)
    memory.offer("1", _clip(3.0, seed=6), 0.9, SR)
    assert len(memory.extend("1", [], SR)) == 1


# --- across files -----------------------------------------------------------

def test_reset_clears_everything():
    """Speaker "1" in the next file is a different person."""
    memory = _memory()
    memory.offer("1", _clip(seed=7), 0.9, SR)
    memory.reset()
    assert memory.extend("1", [], SR) == []
    assert memory.summary()["clips"] == 0


def test_the_separation_service_clears_it_between_files():
    """Wiring, not just the class: reset_stats() has to reach the memory."""
    import services.separation_service as sep

    service = sep.TargetExtractionService.__new__(sep.TargetExtractionService)
    service.logger = None
    service._tse_model = None
    service.model_loader = None
    service.memory = _memory()
    service.memory.offer("1", _clip(seed=8), 0.9, SR)
    service.reset_stats()
    assert service.memory.extend("1", [], SR) == []


# --- fusing overlaps that sit too close to be balanced ----------------------

def _pair(start, end, a="1", b="2"):
    return {"overlap_start": start, "overlap_end": end,
            "overlap_duration": end - start,
            "seg1": {"speaker": a}, "seg2": {"speaker": b}}


def _fuse(pairs, gap=0.6):
    import services.separation_service as sep
    return sep.TargetExtractionService._fuse_adjacent(pairs, gap=gap)


def test_two_overlaps_half_a_second_apart_become_one():
    """The measured case: gaps of 0.36-0.52s, which cost the stitched window."""
    fused = _fuse([_pair(1427.07, 1427.57), _pair(1428.07, 1428.11)])
    assert len(fused) == 1
    assert fused[0]["overlap_start"] == 1427.07
    assert fused[0]["overlap_end"] == 1428.11


def test_overlaps_far_apart_stay_separate():
    fused = _fuse([_pair(10.0, 10.5), _pair(30.0, 30.5)])
    assert len(fused) == 2


def test_the_fused_duration_is_recomputed():
    fused = _fuse([_pair(5.0, 5.4), _pair(5.8, 6.2)])
    assert abs(fused[0]["overlap_duration"] - 1.2) < 1e-9


def test_fusing_does_not_mutate_the_input():
    pairs = [_pair(5.0, 5.4), _pair(5.8, 6.2)]
    _fuse(pairs)
    assert pairs[0]["overlap_end"] == 5.4


def test_a_zero_gap_disables_fusing():
    pairs = [_pair(5.0, 5.4), _pair(5.5, 6.2)]
    assert len(_fuse(pairs, gap=0.0)) == 2


def test_a_single_overlap_is_returned_unchanged():
    pairs = [_pair(5.0, 5.4)]
    assert _fuse(pairs) is pairs
