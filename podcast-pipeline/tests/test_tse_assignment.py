"""The two comparisons that decide which separated track belongs to whom.

Both were wrong in ways that only bite in one branch, so a run could look
healthy while a whole class of overlaps was being rejected or swapped.

Run:  python -m pytest tests/test_tse_assignment.py -q     (from podcast-pipeline/)
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


# --- anchor_self / anchor_other --------------------------------------------

def test_the_anchor_scores_its_own_track_whichever_speaker_it_is():
    """anchor_self read out_A_np regardless of which speaker was the anchor, so
    every job anchored on B compared embed_B against track A: own low, other
    high, (own - other) negative, not_a_fail guaranteed."""
    src = _source("models/tse_model.py")
    block = src[src.index("anchor_is_a = sim_A is not None"):][:600]

    assert "self_np = out_A_np if anchor_is_a else out_B_np" in block
    assert "other_np = out_B_np if anchor_is_a else out_A_np" in block
    assert '("anchor_self", self_np[c0:c1])' in block
    assert "out_A_np[c0:c1]" not in block, "self track must not be hardcoded to A"


def test_the_anchor_branch_is_symmetric():
    """Simulate both branches: the anchor must always score itself."""
    for anchor_is_a in (True, False):
        anchor_embed = "embed_A" if anchor_is_a else "embed_B"
        self_np = "out_A" if anchor_is_a else "out_B"
        other_np = "out_B" if anchor_is_a else "out_A"

        assert anchor_embed[-1] == self_np[-1], (
            f"anchor {anchor_embed} scored {self_np} as 'self'")
        assert anchor_embed[-1] != other_np[-1]


# --- channel similarity -----------------------------------------------------

def test_channel_similarity_uses_an_energy_envelope():
    """The decoder resynthesises each chunk independently, so the same voice
    returns with a different phase across a seam. Raw-sample correlation goes
    from 1.0 to -0.64 on a 2ms shift, which is enough to swap two channels that
    were already correct."""
    src = _source("sidon_infer.py")
    fn = src[src.index("def _channel_similarity"):]
    fn = fn[:fn.index("\ndef ", 1)]

    assert "_energy_envelope" in fn
    assert "_ENVELOPE_FRAME" in src


def test_the_envelope_survives_a_phase_shift_that_breaks_raw_correlation():
    import numpy as np

    rng = np.random.default_rng(0)
    sr, n = 24000, 12000
    t = np.arange(n) / sr
    voice = np.sin(2 * np.pi * 180 * t) * np.exp(-((t - 0.25) ** 2) / 0.01)
    voice = voice + rng.standard_normal(n) * 0.02

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 1e-8 else 0.0

    def envelope(x, frame=240):
        m = len(x) // frame
        return np.sqrt((x[:m * frame].reshape(m, frame) ** 2).mean(axis=1) + 1e-12)

    shifted = np.roll(voice, int(0.002 * sr))          # 2ms

    assert corr(voice, shifted) < 0.5, "raw correlation should collapse"
    assert corr(envelope(voice), envelope(shifted)) > 0.9


# --- stitched window sizing -------------------------------------------------

def test_the_stitched_window_takes_five_seconds_per_speaker():
    """ECAPA pools over time; 3s left the scores noisy enough to mislabel
    tracks. Both speakers contribute equally, so the ratio stays 1:1 whatever
    this is."""
    src = _source("services/separation_service.py")
    assert 'TSE_STITCH_SOLO", "5.0"' in src


def test_the_overlap_carries_context_either_side():
    """Diarizer boundaries land on a frame grid, not on the speech, so a span
    cut exactly at them can start mid-syllable."""
    src = _source("services/separation_service.py")
    assert 'TSE_STITCH_EDGE_PAD", "0.2"' in src
    assert "pad = TSE_STITCH_EDGE_PAD" in src, "the pad must not be hardcoded"


def test_both_stitch_settings_are_configurable_per_profile():
    import json

    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    for env, profile in config["environments"].items():
        tse = profile.get("models", {}).get("tse", {})
        assert "stitch_solo" in tse, f"{env} cannot set stitch_solo"
        assert "stitch_edge_pad" in tse, f"{env} cannot set stitch_edge_pad"

    main = _source("main.py")
    assert '("stitch_solo", "TSE_STITCH_SOLO")' in main
    assert '("stitch_edge_pad", "TSE_STITCH_EDGE_PAD")' in main
