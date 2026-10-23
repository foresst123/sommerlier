import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import separation_layout as sl  # noqa: E402

SR = 16000
CFG = sl.LayoutConfig()


def sec(x):
    return x / SR


# doc/cach_tao_du_lieu_tach_overlap.md section 5, W = 4 s: L -> (ctx per side, solo per speaker)
DOC_TABLE = [(0.3, 0.85, 1.0), (0.46, 0.77, 1.0), (1.0, 0.5, 1.0), (1.5, 0.25, 1.0), (2.0, 0.25, 0.75),
             (3.0, 0.5, 0.0), (3.2, 0.4, 0.0), (3.5, 0.25, 0.0), (4.0, 0.0, 0.0)]


@pytest.mark.parametrize("L,ctx,solo", DOC_TABLE)
def test_budget_matches_doc_table(L, ctx, solo):
    b = sl.plan_budget(sl.window_samples(4.0, SR), CFG.n(L), CFG)
    assert sec(b.ctx_left) == pytest.approx(ctx, abs=1e-4)
    assert sec(b.ctx_right) == pytest.approx(ctx, abs=1e-4)
    assert sec(b.solo[0]) == sec(b.solo[1]) == pytest.approx(solo, abs=1e-4)
    assert b.total == 4 * SR and b.pad_left == b.pad_right == 0


@pytest.mark.parametrize("W", [2.0, 3.0, 4.0, 5.5, 6.0, 8.0])
def test_budget_sums_to_window_and_respects_rules(W):
    rng = random.Random(0)
    n = sl.window_samples(W, SR)
    for _ in range(400):
        core = rng.randint(CFG.n(CFG.l_min), n)
        b = sl.plan_budget(n, core, CFG)
        assert b.total == n
        assert min(b.ctx_left, b.ctx_right, *b.solo, b.pad_left, b.pad_right) >= 0
        assert max(b.ctx_left, b.ctx_right) <= CFG.n(CFG.ctx_max)
        if n - core >= 2 * CFG.n(CFG.margin):
            assert min(b.ctx_left, b.ctx_right) >= CFG.n(CFG.margin)
        # solo exists iff L <= W - 1.1 s
        assert (b.solo[0] > 0) == (core <= n - CFG.n(1.1))
        assert b.solo[0] == 0 or b.solo[0] >= CFG.n(CFG.solo_min)


def test_large_window_caps_ctx_and_moves_rest_to_solo():
    b = sl.plan_budget(sl.window_samples(8.0, SR), CFG.n(0.3), CFG)
    assert b.ctx_left == b.ctx_right == CFG.n(2.0)
    assert b.solo[0] + b.solo[1] == 8 * SR - CFG.n(0.3) - 2 * CFG.n(2.0)
    assert b.pad_left == b.pad_right == 0


def test_tse_budget_has_no_solo_and_shrinks():
    t = sl.LayoutConfig(solo=False)
    b = sl.plan_budget(sl.window_samples(8.0, SR), t.n(0.5), t)
    assert b.solo == (0, 0) and b.ctx_left == b.ctx_right == t.n(2.0) and b.n == t.n(4.5)
    b = sl.plan_budget(sl.window_samples(4.0, SR), t.n(1.0), t)
    assert b.n == 4 * SR and b.ctx_left == b.ctx_right == t.n(1.5)
    fb, why = sl.fallback(b, sl.Avail(ctx_left=t.n(0.6), ctx_right=t.n(0.6)), t)
    assert fb.n == t.n(2.2) and why == ["tse_shrunk"]
    assert sl.fallback(b, sl.Avail(ctx_left=t.n(0.2), ctx_right=t.n(2)), t) == (None, "ctx_short")


def test_window_samples_refuses_fractional_samples():
    assert sl.window_samples(4.0, SR) == 64000
    assert sl.window_samples(3.5, SR, (24000,)) == 56000
    with pytest.raises(ValueError):
        sl.window_samples(4.00001, SR)


def test_len_bins_follow_window():
    assert sl.len_bins(4.0, 0.25) == [(0.25, 0.5), (0.5, 1.0), (1.0, 4.0)]
    assert sl.len_bins(0.8, 0.25) == [(0.25, 0.5), (0.5, 0.8)]
    assert sl.in_bin(4.0, 1.0, 4.0, last=True) and not sl.in_bin(1.0, 0.5, 1.0, last=False)


def test_fallback_chain():
    n = 4 * SR
    b = sl.plan_budget(n, CFG.n(1.0), CFG)               # ctx 0.5 / solo 1.0
    fb, why = sl.fallback(b, sl.Avail(), CFG)
    assert fb == b and why == []
    # ctx shorter than nominal -> extra solo
    fb, why = sl.fallback(b, sl.Avail(ctx_left=CFG.n(0.3), ctx_right=CFG.n(0.3), ctx_total=CFG.n(0.6)), CFG)
    assert (fb.ctx_left, fb.ctx_right) == (CFG.n(0.3), CFG.n(0.3)) and sum(fb.solo) == CFG.n(2.4)
    assert "ctx_short" in why and fb.total == n
    # below the 0.25 s minimum -> drop
    assert sl.fallback(b, sl.Avail(ctx_left=CFG.n(0.2)), CFG) == (None, "ctx_short")
    # no solo for B -> its time to ctx (cap 2 s), then silence
    fb, why = sl.fallback(b, sl.Avail(solo=(sl.INF, 0)), CFG)
    assert fb.solo[1] == 0 and "solo_missing_B" in why and "solo_to_ctx" in why and fb.total == n
    # nothing anywhere -> silence pad
    fb, why = sl.fallback(b, sl.Avail(ctx_left=CFG.n(0.5), ctx_right=CFG.n(0.5), solo=(0, 0)), CFG)
    assert fb.solo == (0, 0) and fb.pad_left + fb.pad_right == CFG.n(2.0) and "silence_pad" in why
    # a solo source shorter than solo_min is not used
    fb, why = sl.fallback(b, sl.Avail(solo=(CFG.n(0.2), sl.INF)), CFG)
    assert fb.solo[0] == 0 and fb.total == n


def _row(pattern="contained", L=1.0, order=("A", "B"), W=4.0, b=None, fallback=None):
    n = sl.window_samples(W, SR)
    b = b or sl.plan_budget(n, CFG.n(L), CFG)
    slots = sl.make_slots(b, pattern, order)
    runs = sl.make_runs(slots, pattern, CFG)
    pieces, src = [], 100.0
    for k, r in enumerate(runs):
        for p in r["pieces"]:
            a = src + (p["start"] - r["start"]) / SR
            pieces.append(dict(p, run=k, src_start=a, src_end=a + (p["end"] - p["start"]) / SR))
        src += 50
    c0, c1 = sl.core_range(slots)
    return {"id": "x", "samples": n, "sr": SR, "window": W, "pattern": pattern, "core": [c0, c1],
            "L": (c1 - c0) / SR, "slots": slots, "layout": pieces, "fallback": fallback,
            "layout_cfg": {"sr": SR}}


@pytest.mark.parametrize("pattern", ["contained", "partial", "full"])
@pytest.mark.parametrize("L", [0.25, 0.46, 1.0, 2.0, 2.9, 3.0, 3.6, 4.0])
@pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
def test_runs_are_valid_layouts(pattern, L, order):
    row = _row(pattern, L, order)
    assert sl.check_layout(row) == []
    f = CFG.n(CFG.fade)
    c0, c1 = row["core"]
    for q in row["layout"]:
        if q["kind"] == "core":       # fades only outside the core
            assert q["start"] + q["fade_in"] <= c0 and q["end"] - q["fade_out"] >= c1
    # every junction between runs at a slot boundary overlaps by exactly one fade
    for s, t in zip(row["slots"], row["slots"][1:]):
        if "solo" in (s["kind"], t["kind"]):
            left = [q for q in row["layout"] if q["start"] < s["end"] < q["end"] + 1 and q["end"] > s["end"]]
            assert left, (s, t)
            assert all(q["fade_out"] == f or q["fade_in"] == f for q in left)


def test_partial_ctx_owners_and_full_has_silent_ctx():
    row = _row("partial", 1.0)
    ctx = [s for s in row["slots"] if s["kind"] == "ctx"]
    assert [s["who"] for s in ctx] == ["A", "B"]
    row = _row("full", 1.0)
    assert all(s["who"] is None for s in row["slots"] if s["kind"] == "ctx")
    assert {q["kind"] for q in row["layout"]} == {"solo", "core"}


def _mutate(row, fn):
    import copy
    r = copy.deepcopy(row)
    fn(r)
    return sl.check_layout(r)


def test_check_layout_catches_violations():
    row = _row("contained", 1.0)
    assert sl.check_layout(row) == []
    core_b = next(i for i, q in enumerate(row["layout"]) if q["kind"] == "core" and q["who"] == "B")
    ctx_l = next(i for i, q in enumerate(row["layout"]) if q["kind"] == "ctx")
    solo = next(i for i, q in enumerate(row["layout"]) if q["kind"] == "solo")
    cases = {
        "length": lambda r: r.update(samples=r["samples"] + 1),
        "fade in core": lambda r: r["layout"][core_b].update(fade_in=r["layout"][core_b]["fade_in"] + 160,
                                                             start=r["layout"][core_b]["start"] + 160),
        "ctx not continuous": lambda r: r["layout"][ctx_l].update(src_start=r["layout"][ctx_l]["src_start"] - 1,
                                                                  src_end=r["layout"][ctx_l]["src_end"] - 1),
        "same solo twice": lambda r: [s.update(who="A") for s in r["slots"] if s["kind"] == "solo"],
        "budget": lambda r: r.update(L=r["L"], core=r["core"]) or r["slots"][0].update(end=r["slots"][0]["end"] - 1),
        "L field": lambda r: r.update(L=0.7),
        "too short": lambda r: r.update(layout_cfg={"sr": SR, "l_min": 1.5}),
        "big overlap": lambda r: r["layout"][solo].update(end=r["layout"][solo]["end"] + 2000),
        "solo source inside core run": lambda r: r["layout"][solo].update(
            src_start=r["layout"][ctx_l]["src_start"] if r["layout"][solo]["who"] == "A" else r["layout"][core_b]["src_start"],
            src_end=(r["layout"][ctx_l]["src_start"] if r["layout"][solo]["who"] == "A" else r["layout"][core_b]["src_start"])
            + (r["layout"][solo]["end"] - r["layout"][solo]["start"]) / SR),
        "pad without fallback": lambda r: r["slots"].insert(0, {"kind": "pad", "who": None, "start": -1, "end": 0}),
    }
    for name, fn in cases.items():
        assert _mutate(row, fn), name


def test_fallback_rows_skip_the_budget_check_but_not_the_rest():
    n = 4 * SR
    b = sl.plan_budget(n, CFG.n(1.0), CFG)
    fb, why = sl.fallback(b, sl.Avail(ctx_left=CFG.n(0.5), ctx_right=CFG.n(0.5), solo=(0, 0)), CFG)
    row = _row("contained", b=fb, fallback="+".join(why))
    assert sl.check_layout(row) == []
    assert [s["kind"] for s in row["slots"]] == ["pad", "ctx", "core", "ctx", "pad"]
    row["fallback"] = None
    assert sl.check_layout(row)


def test_pick_enrollment_nearest_outside_mix():
    clips = [(0, 2), (10, 13), (20, 21.5), (30, 34), (50, 52)]
    got = sl.pick_enrollment(clips, [(19.9, 25)], anchor=22.0, margin=0.3, k=3)
    assert got == [3, 1, 0]          # 8 s, 9 s and 20 s away; (20, 21.5) is inside the mix
    assert sl.default_enrollment(clips, got, 4.0, 1.0) == [3]
    assert sl.default_enrollment(clips, [1, 3], 4.0, 1.0) == [1, 3]
    assert sl.default_enrollment(clips, [], 4.0, 1.0) is None


def test_core_si_sdri_scores_the_core_only():
    rng = np.random.default_rng(0)
    ref = np.zeros(4 * SR)
    ref[SR:3 * SR] = rng.standard_normal(2 * SR)
    other = rng.standard_normal(4 * SR)
    mix = ref + other
    core = (int(1.5 * SR), int(2 * SR))
    assert sl.core_si_sdri(ref + 1e-3 * other, ref, mix, core, SR) > 20
    assert abs(sl.core_si_sdri(mix, ref, mix, core, SR)) < 1e-6


def test_budget_check_catches_unflagged_deviation():
    n = 4 * SR
    b = sl.plan_budget(n, CFG.n(1.0), CFG)
    fb, why = sl.fallback(b, sl.Avail(ctx_left=CFG.n(0.3), ctx_right=CFG.n(0.3), ctx_total=CFG.n(0.6)), CFG)
    assert fb.pad_left == fb.pad_right == 0 and why
    assert sl.check_layout(_row("contained", b=fb, fallback="+".join(why))) == []
    assert any("budget" in x for x in sl.check_layout(_row("contained", b=fb, fallback=None)))
