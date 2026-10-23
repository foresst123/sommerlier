import csv
import json
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import render_mix_dataset as rmd  # noqa: E402
import validate_mix_dataset as val  # noqa: E402
from utils import separation_layout as sl  # noqa: E402

sf = pytest.importorskip("soundfile")
SR = 24000


def _rec(name, seed, n_seg=8):
    rng = np.random.default_rng(seed)
    audio = (0.002 * rng.standard_normal(400 * SR)).astype("float32")
    segs, t = [], 0.0
    for i in range(n_seg):
        spk = "A" if i % 2 == 0 else "B"
        a, b = t, t + 30.0
        audio[int(a * SR):int(b * SR)] += (0.1 * (1 + 0.5 * (i % 2)) * rng.standard_normal(30 * SR)).astype("float32")
        segs.append({"index": str(i), "start": a, "end": b, "speaker": spk})
        t += 40.0       # 10 s gaps: noise bank material
    return {"path": f"cache/{name}", "segments": lambda: segs, "audio": lambda: (audio, SR)}


def _templates(names):
    rows = [
        {"t": 10.0, "overlap": 0.5, "pattern": "contained", "first_len": 6.0, "second_len": 0.5, "delta_db": -2.0},
        {"t": 20.0, "overlap": 1.5, "pattern": "contained", "first_len": 8.0, "second_len": 1.5, "delta_db": 1.0},
        {"t": 30.0, "overlap": 2.0, "pattern": "partial", "first_len": 7.0, "second_len": 5.0, "delta_db": -4.0},
        {"t": 40.0, "overlap": 3.0, "pattern": "full", "first_len": 3.0, "second_len": 3.2, "delta_db": 0.0},
        {"t": 50.0, "overlap": 0.15, "pattern": "contained", "first_len": 3.0, "second_len": 0.15, "delta_db": 0.0},
        {"t": 60.0, "overlap": 5.0, "pattern": "partial", "first_len": 9.0, "second_len": 6.0, "delta_db": 0.0}]
    return {n: [dict(r, rec=n) for r in rows] for n in names}


def test_resample_and_read_run_are_sample_exact():
    x = np.random.default_rng(0).standard_normal(4 * SR).astype("float32")
    assert len(rmd.resample(x, SR, 16000)) == 4 * 16000
    whole = rmd.read_run(x, SR, 16000, 1.0, 16000)
    a = rmd.read_run(x, SR, 16000, 1.0, 8000)
    b = rmd.read_run(x, SR, 16000, 1.5, 8000)
    assert len(whole) == 16000
    # two halves read and resampled on their own match the whole (padding hides the edges)
    assert np.max(np.abs(np.concatenate([a, b]) - whole)) < 0.02 * np.max(np.abs(whole))
    assert len(rmd.read_run(x, SR, 16000, 3.9, 4000)) == 4000    # past the end: zero padded


def test_render_layout_crossfades_sum_to_one():
    """A constant source rendered through a layout: wherever two runs crossfade, the stems sum
    to the constant (complementary ramps); a crossfade never dips."""
    n = 4 * 16000
    cfg = sl.LayoutConfig()
    b = sl.plan_budget(n, cfg.n(1.0), cfg)
    slots = sl.make_slots(b, "partial", ("A", "B"))
    runs = sl.make_runs(slots, "partial", cfg)
    pieces = []
    for k, r in enumerate(runs):
        for p in r["pieces"]:
            s = 10.0 * (k + 1) + (p["start"] - r["start"]) / 16000
            pieces.append(dict(p, run=k, src_start=s, src_end=s + (p["end"] - p["start"]) / 16000))
    lay = {"n": n, "sr": 16000, "layout": pieces}
    st = rmd.render_layout(lay, np.ones(60 * SR, "float32"), SR)
    tot = st["A"] + st["B"]
    c0, c1 = sl.core_range(slots)
    outside = np.r_[0:c0 - 400, c1 + 400:n]
    assert np.allclose(tot[outside], 1.0, atol=1e-3)
    assert np.allclose(tot[c0:c1], 2.0, atol=1e-3)        # both speakers in the core, no fade there


def test_build_renders_valid_dataset(tmp_path):
    names = [f"default_r{i}" for i in range(6)]
    recs = [_rec(n, i) for i, n in enumerate(names)]
    out = tmp_path / "out"
    rows = rmd.build(recs, _templates(names), str(out), n=60, window=4.0, seed=3, val_share=0.34,
                     noise_prob=0.5, workers=1)
    assert len(rows) >= 40
    assert {r["split"] for r in rows} == {"train", "val"}
    assert val.check(rows, check_n=200) == []
    assert all(0.25 <= r["L"] <= 4.0 for r in rows)                      # 0.15 and 5.0 never drawn
    r = next(r for r in rows if r["noise_snr_db"] is None)
    x = {t: sf.read(r["paths"][t])[0] for t in ("mix", "s1", "s2")}
    assert len(x["mix"]) == 4 * 16000
    assert np.allclose(x["mix"], x["s1"] + x["s2"], atol=3e-4)
    r = next(r for r in rows if r["noise_snr_db"] is not None)
    x = {t: sf.read(r["paths"][t])[0] for t in ("mix", "s1", "s2", "noise")}
    assert np.allclose(x["mix"], x["s1"] + x["s2"] + x["noise"], atol=3e-4)
    assert {"real", "strat", "hard"} <= {r["kind"] for r in rows}
    mf = (out / "manifests" / "mossformer2" / "tr_list.txt").read_text().splitlines()
    assert mf and len(mf[0].split(" ")) == 3
    tj = json.loads((out / "manifests" / "tiger" / "train" / "mix.json").read_text())
    assert tj and tj[0][1] == 4 * 16000
    with open(out / "manifests" / "speechbrain" / "train.csv") as f:
        sb = list(csv.DictReader(f))
    assert sb and all(os.path.exists(s["noise_wav"]) for s in sb)
    rep = json.loads((out / "render_report.json").read_text())
    assert rep["templates_out_of_range"] == 2 * len(names)
    assert rep["splits"]["train"]["rows"] + rep["splits"]["val"]["rows"] == len(rows)
    assert rep["multi_overlap"]["pairs"] == 6 * len(names)
    assert (out / "chunks.jsonl").read_text().count("\n") == len(rows)


def test_one_gain_per_speaker_keeps_the_level_between_pieces(tmp_path):
    """The holder's ctx and solo come from segments whose levels differ by 3.5 dB in the
    source; with one gain per speaker that difference survives in the stem."""
    names = ["default_g"]
    rec = _rec("default_g", 0)
    rows = rmd.build([rec], {"default_g": [dict(_templates(names)["default_g"][0])]}, str(tmp_path / "o"),
                     n=12, window=4.0, seed=0, val_share=0.0, noise_prob=0.0, subtype="FLOAT", snap=False)
    audio, _ = rec["audio"]()
    checked = 0
    for r in rows:
        if r["fallback"] or len(r["solo_order"]) != 2:
            continue
        s = {t: sf.read(r["paths"][t])[0] for t in ("s1", "s2")}
        stems = {"A": s["s1"], "B": s["s2"]} if r["target_order"] == "holder,interrupter" else {"A": s["s2"], "B": s["s1"]}
        solo = next(p for p in r["layout"] if p["kind"] == "solo" and p["who"] == "A")
        ctx = next(p for p in r["layout"] if p["kind"] == "ctx")

        def db(x):
            return 10 * np.log10(np.mean(x ** 2))

        def mid(p):
            return slice(p["start"] + 400, p["end"] - 400)

        def src(p):
            return np.asarray(audio[int(p["src_start"] * SR) + 600:int(p["src_end"] * SR) - 600])
        out_diff = db(stems["A"][mid(solo)]) - db(stems["A"][mid(ctx)])
        src_diff = db(src(solo)) - db(src(ctx))
        assert out_diff == pytest.approx(src_diff, abs=0.5)
        checked += 1
    assert checked


def test_noise_off_gives_exact_sums(tmp_path):
    names = ["default_a", "default_b"]
    rows = rmd.build([_rec(n, i) for i, n in enumerate(names)], _templates(names), str(tmp_path / "o"),
                     n=20, window=3.0, seed=1, val_share=0.0, noise_prob=0.0, subtype="FLOAT")
    assert rows and all(r["noise_snr_db"] is None and r["paths"]["noise"] is None for r in rows)
    r = rows[0]
    x = {t: sf.read(r["paths"][t])[0] for t in ("mix", "s1", "s2")}
    assert len(x["mix"]) == 3 * 16000
    assert np.max(np.abs(x["mix"] - x["s1"] - x["s2"])) < 1e-6


def test_template_stream_is_the_same_for_every_window(tmp_path):
    names = ["default_a", "default_b", "default_c"]
    recs = [_rec(n, i) for i, n in enumerate(names)]
    tp = {n: [t for t in _templates([n])[n] if 0.25 <= t["overlap"] <= 3.0] for n in names}
    a = rmd.build(recs, tp, str(tmp_path / "w3"), n=15, window=3.0, seed=2, val_share=0.0, noise_prob=0.0)
    b = rmd.build(recs, tp, str(tmp_path / "w4"), n=15, window=4.0, seed=2, val_share=0.0, noise_prob=0.0)
    seq = lambda rows: [(r["recording"], r["kind"], r["template_overlap"], r["pattern"]) for r in rows]  # noqa: E731
    assert seq(a) == seq(b)
    assert {r["samples"] for r in a} == {48000} and {r["samples"] for r in b} == {64000}


@pytest.mark.parametrize("mode", ["plain", "bss"])
def test_tse_rows_two_per_mix_with_enrollment_outside_the_mix(tmp_path, mode):
    names = [f"default_t{i}" for i in range(3)]
    out = tmp_path / mode
    rows = rmd.build([_rec(n, i, n_seg=10) for i, n in enumerate(names)], _templates(names), str(out), n=18,
                     window=4.0, seed=0, val_share=0.34, noise_prob=0.5, tse_mix=mode)
    trows = val.load_tse(str(out))
    mixes = val.load_rows(str(out), "tse_mixes.jsonl")
    assert trows and len(trows) == 2 * (len(mixes) if mode == "plain" else len(rows))
    assert val.check(rows, 50, tse_rows=trows, tse_mixes=mixes) == []
    for t in trows:
        assert os.path.exists(t["enroll_bank"]) and len(t["enroll_candidates"]) <= rmd.ENROLL_K
        bank = json.loads(open(t["enroll_bank"][:-4] + ".json").read())
        assert bank["speaker"] == t["target_speaker"]
    if mode == "plain":
        assert all(not m["solo_order"] for m in mixes)
        assert all(m["samples"] <= 4 * 16000 for m in mixes)
    # a candidate inside the mix is caught
    bad = json.loads(json.dumps(trows[0]))
    m = {r["id"]: r for r in rows + mixes}[bad["mix_id"]]
    p = m["layout"][0]
    bad["enroll_candidates"][0].update(src_start=p["src_start"], src_end=p["src_end"])
    assert val.check_tse([bad], {m["id"]: m})


def test_validator_flags_broken_rows(tmp_path):
    names = ["default_a", "default_b"]
    rows = rmd.build([_rec(n, i) for i, n in enumerate(names)], _templates(names), str(tmp_path / "o"),
                     n=8, window=4.0, seed=1, val_share=0.0, noise_prob=0.0)
    r = json.loads(json.dumps(rows[0]))
    r["samples"] = 48000
    assert val.check([r], 1)
    r = json.loads(json.dumps(rows[0]))
    r["layout"][0]["fade_in"] = 10_000
    assert val.check([r], 1)
    text = val.compare(rows, {}, np.array([0.1, 0.2, 0.5, 1.0, 6.0]))
    assert "core share" in text and "| 0.600 |" in text      # 3 of 5 real pairs outside [0.25, 4]
