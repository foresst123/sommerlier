"""plan_chunk and the template helpers of render_mix_dataset: no audio, no soundfile."""
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import render_mix_dataset as rmd  # noqa: E402
from utils import separation_layout as sl  # noqa: E402

SR = 16000
CFG = sl.LayoutConfig()


def _t(L, pattern="contained", t=0.0):
    return {"overlap": L, "pattern": pattern, "first_len": 6.0, "second_len": L, "delta_db": 0.0, "t": t}


RICH = {"X": [(0.0, 30.0), (40.0, 70.0), (100.0, 103.0)], "Y": [(80.0, 95.0), (120.0, 121.0), (130.0, 150.0)]}


def _row(lay):
    return dict(lay, id="t", samples=lay["n"])


@pytest.mark.parametrize("W", [3.0, 4.0, 6.0])
def test_plan_chunk_layouts_pass_the_checks(W):
    rng = random.Random(1)
    n = sl.window_samples(W, SR)
    seen = set()
    for k in range(300):
        L = rng.uniform(CFG.l_min, W)
        pat = rng.choice(["contained", "partial", "full"])
        lay, why = rmd.plan_chunk(rng, _t(L, pat), RICH, n, CFG, 24000)
        assert lay is not None, why
        assert sl.check_layout(_row(lay)) == [], (pat, L)
        assert lay["n"] == n
        seen.add(tuple(lay["solo_order"]))
        for p in lay["layout"]:     # every source range lies in the recording, on its 24 kHz grid
            assert p["src_start"] >= 0
    assert ("A", "B") in seen and ("B", "A") in seen


def test_whole_short_clip_becomes_the_core():
    pools = {"X": [(0.0, 30.0)], "Y": [(50.0, 50.55), (60.0, 75.0)]}
    rng = random.Random(0)
    hits = 0
    for _ in range(200):
        lay, why = rmd.plan_chunk(rng, _t(0.5), pools, 4 * SR, CFG, 24000)
        assert lay is not None, why
        if lay["whole_clip_core"]:
            hits += 1
            core_b = next(p for p in lay["layout"] if p["kind"] == "core" and p["who"] == "B")
            c0, c1 = lay["core"]
            # the clip itself spans exactly the core; its fades lie outside
            assert core_b["src_start"] + (c0 - core_b["start"]) / SR == pytest.approx(50.0, abs=1e-4)
            assert lay["L"] == pytest.approx(0.55, abs=1e-4)
    assert 70 < hits < 130     # whenever the holder is X (p = 0.5), the short clip of Y is used


def test_short_sources_go_through_the_fallback_or_drop():
    rng = random.Random(0)
    # B has no second clip: no solo for B -> fallback
    pools = {"X": [(0.0, 30.0), (40.0, 70.0)], "Y": [(80.0, 82.0)]}
    lay, why = rmd.plan_chunk(rng, _t(1.0), pools, 4 * SR, CFG, 24000)
    assert lay is not None and "solo_missing" in lay["fallback"]
    assert sl.check_layout(_row(lay)) == []
    # A's stretch cannot hold core + 2 x 0.25 s: dropped (case 5)
    pools = {"X": [(0.0, 1.2)], "Y": [(80.0, 81.2)]}
    got = {rmd.plan_chunk(random.Random(s), _t(1.0), pools, 4 * SR, CFG, 24000)[1] for s in range(20)}
    assert got <= {"no_core_clip", "ctx_short"} and got
    # out of range (case 6) and a single speaker
    assert rmd.plan_chunk(rng, _t(0.2), RICH, 4 * SR, CFG, 24000) == (None, "out_of_range")
    assert rmd.plan_chunk(rng, _t(4.5), RICH, 4 * SR, CFG, 24000) == (None, "out_of_range")
    assert rmd.plan_chunk(rng, _t(1.0), {"X": RICH["X"]}, 4 * SR, CFG, 24000) == (None, "one_speaker")


def test_tse_plan_has_no_solo_and_window_shrinks_to_the_cap():
    t = sl.LayoutConfig(solo=False)
    lay, why = rmd.plan_chunk(random.Random(0), _t(0.5, "partial"), RICH, 8 * SR, t, 24000)
    assert lay["n"] == t.n(4.5) and not lay["solo_order"]
    assert sl.check_layout(_row(lay)) == []


def test_snap_moves_cuts_and_is_recorded():
    calls = []

    def snap(s, lo, hi):
        calls.append((s, lo, hi))
        assert lo <= s <= hi
        return lo, "energy_pause"
    lay, _ = rmd.plan_chunk(random.Random(0), _t(1.0), RICH, 4 * SR, CFG, 24000, snap=snap)
    assert calls and lay["cut_method"] == {"energy_pause": len(calls)}
    assert all(hi - lo <= 2 * rmd.SNAP + 1e-9 for _, lo, hi in calls)
    assert sl.check_layout(_row(lay)) == []


def test_templates_outside_l_min_and_window_are_never_drawn():
    ts = [_t(x) for x in (0.1, 0.24, 0.25, 0.8, 3.9, 4.0, 4.2, 7.0)]
    rng = random.Random(0)
    got = {rmd.pick_template(rng, "strat", [], ts, 0.25, 4.0)[0]["overlap"] for _ in range(400)}
    assert got == {0.25, 0.8, 3.9, 4.0}
    got = {rmd.pick_template(rng, "real", ts, ts, 0.25, 4.0)[0]["overlap"] for _ in range(400)}
    assert got == {0.25, 0.8, 3.9, 4.0}
    hard = [rmd.pick_template(rng, "hard", [], ts, 0.25, 4.0) for _ in range(50)]
    assert all(k == "hard" and 1.0 <= t["overlap"] <= 4.0 for t, k in hard)
    assert rmd.pick_template(rng, "real", [_t(5.0)], [_t(6.0)], 0.25, 4.0) == (None, "hard")


def test_template_draws_are_the_same_for_every_window():
    """Unfiltered lists, W-independent bins, a fixed number of draws: the k-th draw is the
    same template for W = 3 and W = 4 whenever it lies in range for both."""
    pool = [_t(x, t=float(i)) for i, x in enumerate(np.linspace(0.25, 3.9, 60))]
    own = pool[:15] + [_t(3.5)]

    def draws(W):
        rng = random.Random(7)
        out = []
        for _ in range(300):
            t, kind = rmd.pick_template(rng, rng.choices(["real", "strat", "hard"], [0.7, 0.2, 0.1])[0],
                                        own, pool, 0.25, W)
            rng.gauss(0, 1)
            rng.getrandbits(32)
            out.append((kind, t["overlap"]))
        return out
    a, b = draws(3.0), draws(4.0)
    both = [(x, y) for x, y in zip(a, b) if y[1] <= 3.0]
    assert len(both) > 200 and all(x == y for x, y in both)
    assert all(x[1] <= 3.0 for x in a)


def test_snapping_makes_core_edges_natural():
    """Periodic 200 ms syllables with 60 ms pauses: a random point is in a pause 23% of the
    time. Snapped core edges (partial A end, B start, contained B both edges) land in pauses."""
    sr = 24000
    period, on = int(0.26 * sr), int(0.20 * sr)
    x = (0.1 * np.random.default_rng(0).standard_normal(200 * sr)).astype("float32")
    x[(np.arange(len(x)) % period) >= on] *= 0.001
    pools = {"X": [(0.0, 90.0)], "Y": [(100.0, 190.0)]}
    snap = rmd.make_snapper(x, sr)

    def in_pause(t):
        return (int(round(t * sr)) % period) >= on - int(0.005 * sr)

    hits = {"partial_A_end": [], "partial_B_start": [], "contained_B_start": [], "contained_B_end": []}
    for i in range(120):
        for pat in ("partial", "contained"):
            lay, why = rmd.plan_chunk(random.Random(i), _t(random.Random(i).choice([0.5, 1.0, 2.0]), pat), pools,
                                      4 * SR, CFG, sr, snap=snap)
            assert lay is not None and sl.check_layout(_row(lay)) == [], why
            c0, c1 = lay["core"]
            for p in lay["layout"]:
                if p["kind"] != "core":
                    continue
                # source time of the core edges of this piece
                s0 = p["src_start"] + (c0 - p["start"]) / SR
                s1 = p["src_start"] + (c1 - p["start"]) / SR
                if pat == "partial" and p["who"] == "A":
                    hits["partial_A_end"].append(in_pause(s1))
                if pat == "partial" and p["who"] == "B":
                    hits["partial_B_start"].append(in_pause(s0))
                if pat == "contained" and p["who"] == "B":
                    hits["contained_B_start"].append(in_pause(s0))
                    hits["contained_B_end"].append(in_pause(s1))
            if pat == "contained":
                assert lay["core_edges"] == "snapped"
    rates = {k: float(np.mean(v)) for k, v in hits.items()}
    assert all(r > 0.6 for r in rates.values()), rates


def test_multi_pair_share_counts_events():
    ts = {"r": [_t(0.5, t=10.0), _t(0.5, t=10.8), _t(0.5, t=30.0)], "s": [_t(1.0, t=5.0)]}
    got = rmd.multi_pair_share(ts, merge_gap=1.0)
    assert got["pairs"] == 4 and got["events"] == 3 and got["multi_pair_event_share"] == pytest.approx(1 / 3, abs=1e-3)


def test_split_is_stratified_and_grouped():
    names = [f"show{i % 5}_ep{i}" for i in range(50)]
    rate = {nm: float(i) for i, nm in enumerate(names)}
    sp = rmd.split_recordings(names, rate, 0.1, 0)
    val = [nm for nm, s in sp.items() if s == "val"]
    assert len(val) == 5
    ranks = sorted(names.index(v) for v in val)
    assert all(10 * k <= r < 10 * (k + 1) for k, r in enumerate(ranks))     # one per rate stratum
    sp = rmd.split_recordings(names, rate, 0.2, 0, group_regex=r"^(show\d+)_")
    shows = {}
    for nm, s in sp.items():
        shows.setdefault(nm.split("_")[0], set()).add(s)
    assert all(len(v) == 1 for v in shows.values()) and any(v == {"val"} for v in shows.values())
