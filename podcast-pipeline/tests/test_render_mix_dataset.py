import json
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import render_mix_dataset as rmd  # noqa: E402
import validate_mix_dataset as val  # noqa: E402

sf = pytest.importorskip("soundfile")
SR = 24000


def _rec(name, seed):
    rng = np.random.default_rng(seed)
    audio = (0.002 * rng.standard_normal(400 * SR)).astype("float32")
    segs, t = [], 0.0
    for i in range(8):
        spk = "A" if i % 2 == 0 else "B"
        a, b = t, t + 30.0
        audio[int(a * SR):int(b * SR)] += (0.1 * (1 + 0.5 * (i % 2)) * rng.standard_normal(30 * SR)).astype("float32")
        segs.append({"index": str(i), "start": a, "end": b, "speaker": spk})
        t += 40.0       # 10 s gaps: noise bank material
    return {"path": f"cache/{name}", "segments": lambda: segs, "audio": lambda: (audio, SR)}


def _templates(names):
    rows = [
        {"overlap": 0.5, "pattern": "contained", "first_len": 6.0, "second_len": 0.5, "delta_db": -2.0},
        {"overlap": 1.5, "pattern": "contained", "first_len": 8.0, "second_len": 1.5, "delta_db": 1.0},
        {"overlap": 2.0, "pattern": "partial", "first_len": 7.0, "second_len": 5.0, "delta_db": -4.0},
        {"overlap": 3.0, "pattern": "full", "first_len": 3.0, "second_len": 3.2, "delta_db": 0.0}]
    return {n: [dict(r, rec=n) for r in rows] for n in names}


def test_place_window_keeps_overlap_inside():
    rng = random.Random(0)
    for _ in range(200):
        ws = rmd.place_window(rng, 10.0, 10.5, 4.0)
        assert ws <= 10.0 - rmd.MARGIN + 1e-9 and ws + 4.0 >= 10.5 + rmd.MARGIN - 1e-9


def test_resample_ratio_and_snap():
    x = np.random.default_rng(0).standard_normal(4 * SR).astype("float32")
    assert len(rmd.resample(x, SR, 16000)) == 4 * 16000
    assert 0 <= rmd.snap(x, SR, SR) < len(x)


def test_build_renders_valid_dataset(tmp_path):
    names = [f"default_r{i}" for i in range(6)]
    recs = [_rec(n, i) for i, n in enumerate(names)]
    out = tmp_path / "out"
    rows = rmd.build(recs, _templates(names), str(out), n=60, window=4.0, seed=3, val_share=0.34,
                     noise_prob=0.5, workers=1)
    assert len(rows) >= 40
    assert {r["split"] for r in rows} == {"train", "val"}
    problems = val.check(rows, check_n=200)
    assert problems == []
    r = next(r for r in rows if r["noise_snr_db"] is None)
    x = {t: sf.read(r["paths"][t])[0] for t in ("mix", "s1", "s2")}
    assert len(x["mix"]) == 4 * 16000
    assert np.allclose(x["mix"], x["s1"] + x["s2"], atol=3e-4)
    assert 0 < r["overlap_in_window"] <= 4.0
    assert {"real", "strat", "hard"} <= {r["kind"] for r in rows}
    mf = (out / "manifests" / "mossformer2" / "tr_list.txt").read_text().splitlines()
    assert mf and len(mf[0].split(" ")) == 3
    tj = json.loads((out / "manifests" / "tiger" / "train" / "mix.json").read_text())
    assert tj and tj[0][1] == 4 * 16000
    assert (out / "manifests" / "speechbrain" / "valid.csv").exists()


def test_noise_off_gives_exact_sums(tmp_path):
    names = ["default_a", "default_b"]
    rows = rmd.build([_rec(n, i) for i, n in enumerate(names)], _templates(names), str(tmp_path / "o"),
                     n=20, window=3.0, seed=1, val_share=0.0, noise_prob=0.0, subtype="FLOAT")
    assert rows and all(r["noise_snr_db"] is None for r in rows)
    r = rows[0]
    x = {t: sf.read(r["paths"][t])[0] for t in ("mix", "s1", "s2")}
    assert np.max(np.abs(x["mix"] - x["s1"] - x["s2"])) < 1e-6
