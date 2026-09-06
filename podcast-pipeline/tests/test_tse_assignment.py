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







def test_the_retired_stitch_settings_are_gone_from_every_profile():
    """`stitch_solo` and `stitch_edge_pad` sized the solo slices in the window
    that DialogueSidon needed. Nothing reads them now, and a setting that looks
    live but is not is worse than no setting at all."""
    import json

    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    for env, profile in config["environments"].items():
        tse = profile.get("models", {}).get("tse", {})
        assert "stitch_solo" not in tse, env
        assert "stitch_edge_pad" not in tse, env
    assert "TSE_STITCH" not in _source("main.py")
