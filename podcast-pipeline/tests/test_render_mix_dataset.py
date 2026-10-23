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
    """Regression: with a template between the two W values (3.5 s), W = 3 and W = 4 still
    draw the same template at every row whose template is in range for both."""
    names = ["default_a", "default_b", "default_c"]
    recs = [_rec(n, i) for i, n in enumerate(names)]
    tp = {n: [dict(t) for t in _templates([n])[n] if 0.25 <= t["overlap"] <= 3.0] +
          [{"t": 90.0, "overlap": 3.5, "pattern": "partial", "first_len": 6.0, "second_len": 4.0,
            "delta_db": 0.0, "rec": n}] for n in names}
    a = rmd.build(recs, tp, str(tmp_path / "w3"), n=30, window=3.0, seed=2, val_share=0.0, noise_prob=0.0)
    b = rmd.build(recs, tp, str(tmp_path / "w4"), n=30, window=4.0, seed=2, val_share=0.0, noise_prob=0.0)
    assert len(a) == len(b) == 30
    seq = lambda rows: [(r["recording"], r["kind"], r["template_overlap"], r["pattern"]) for r in rows]  # noqa: E731
    pairs = [(x, y) for x, y in zip(seq(a), seq(b)) if y[2] <= 3.0]
    assert len(pairs) >= 20 and all(x == y for x, y in pairs)
    assert any(y[2] == 3.5 for y in seq(b)) and all(x[2] <= 3.0 for x in seq(a))
    assert {r["samples"] for r in a} == {48000} and {r["samples"] for r in b} == {64000}


def _two_level_rec(name):
    """Each speaker has one loud and one quiet segment, 12 dB apart."""
    rng = np.random.default_rng(0)
    audio = (0.0005 * rng.standard_normal(200 * SR)).astype("float32")
    segs = []
    for spk, t, amp in (("A", 0, 0.2), ("B", 40, 0.2), ("A", 80, 0.05), ("B", 120, 0.05)):
        audio[t * SR:(t + 30) * SR] += (amp * rng.standard_normal(30 * SR)).astype("float32")
        segs.append({"start": float(t), "end": t + 30.0, "speaker": spk})
    return {"path": f"cache/{name}", "segments": lambda: segs, "audio": lambda: (audio, SR)}


def test_level_difference_in_the_core_is_the_templates(tmp_path):
    """Regression (round 3, B1): the gain comes from each speaker's core run, so the core
    level difference equals delta_db even when a solo comes from a 12 dB quieter clip."""
    rec = _two_level_rec("default_g")
    tp = {"default_g": [dict(t=0, overlap=1.0, pattern="contained", first_len=5, second_len=1, delta_db=0.0,
                             rec="default_g")]}
    rows = rmd.build([rec], tp, str(tmp_path / "o"), n=40, window=4.0, seed=0, val_share=0.0, noise_prob=0.0,
                     subtype="FLOAT", snap=False)
    err = []
    for r in rows:
        s1, s2 = sf.read(r["paths"]["s1"])[0], sf.read(r["paths"]["s2"])[0]
        A, B = (s1, s2) if r["target_order"] == "holder,interrupter" else (s2, s1)
        c0, c1 = r["core"]

        def db(x):
            return 10 * np.log10(np.mean(x ** 2))
        err.append(db(B[c0 + 400:c1 - 400]) - db(A[c0 + 400:c1 - 400]) - r["delta_db"])
    assert len(err) >= 30 and np.max(np.abs(err)) < 1.0, np.max(np.abs(err))


def test_multi_overlap_templates_give_no_rows(tmp_path):
    names = ["default_m"]
    tp = _templates(names)
    tp["default_m"] += [dict(t=200.0, overlap=0.6, pattern="contained", first_len=5, second_len=0.6, delta_db=0.0,
                             rec="default_m"),
                        dict(t=201.0, overlap=0.7, pattern="contained", first_len=5, second_len=0.7, delta_db=0.0,
                             rec="default_m")]
    out = tmp_path / "o"
    rows = rmd.build([_rec("default_m", 0)], tp, str(out), n=40, window=4.0, seed=0, val_share=0.0, noise_prob=0.0)
    assert rows and not {0.6, 0.7} & {r["template_overlap"] for r in rows}
    rep = json.loads((out / "render_report.json").read_text())
    assert rep["multi_overlap"]["multi_pair_templates_skipped"] == 2
    assert rep["multi_overlap"]["multi_pair_event_share"] == pytest.approx(1 / 7, abs=1e-3)


def test_stale_tse_outputs_are_removed(tmp_path):
    names = ["default_a", "default_b"]
    recs = [_rec(n, i, n_seg=10) for i, n in enumerate(names)]
    out = tmp_path / "o"
    rmd.build(recs, _templates(names), str(out), n=8, window=4.0, seed=0, val_share=0.5, tse_mix="plain")
    assert (out / "manifests" / "tse" / "train.jsonl").exists() and (out / "tse_mixes.jsonl").exists()
    rmd.build(recs, _templates(names), str(out), n=8, window=4.0, seed=0, val_share=0.5, tse_mix="none")
    assert not (out / "manifests" / "tse").exists() and not (out / "tse_mixes.jsonl").exists()


def test_stereo_input_is_downmixed():
    x = np.random.default_rng(0).standard_normal((2 * SR, 2)).astype("float32")
    got = rmd.read_run(x, SR, 16000, 0.5, 8000)
    assert np.allclose(got, rmd.read_run(x.mean(axis=1), SR, 16000, 0.5, 8000), atol=1e-6)
    rec = _rec("default_s", 0)
    audio, _ = rec["audio"]()
    st = np.stack([audio, audio], axis=1)
    stereo = {"path": "cache/default_s", "segments": rec["segments"], "audio": lambda: (st, SR)}
    assert rmd.noise_bank(rec["segments"](), st, SR) is not None
    assert rmd.mono(st).ndim == 1


def test_noise_options(tmp_path):
    sr = 16000
    rng = np.random.default_rng(0)
    steady = 0.01 * rng.standard_normal(3 * sr)
    burst = steady.copy()
    burst[sr:sr + sr // 3] += 0.3 * rng.standard_normal(sr // 3)
    assert rmd.stationary(steady, sr) and not rmd.stationary(burst, sr)
    names = ["default_n"]
    rows = rmd.build([_rec("default_n", 0)], _templates(names), str(tmp_path / "o"), n=30, window=4.0, seed=0,
                     val_share=0.0, noise_prob=1.0, rec_snr={"default_n": 20.0}, noise_snr_from_recording=True,
                     noise_clean_gaps=True)
    snr = [r["noise_snr_db"] for r in rows]
    assert snr and all(17.0 - 1e-6 <= x <= 23.0 + 1e-6 for x in snr)
    rows = rmd.build([_rec("default_n", 0)], _templates(names), str(tmp_path / "p"), n=30, window=4.0, seed=0,
                     val_share=0.0, noise_prob=1.0, rec_snr={"default_n": 20.0})
    assert any(x["noise_snr_db"] > 25 for x in rows)      # flag off: the fixed bands


def test_enrollment_from_short_clips_only():
    """Regression (round 3, B5): a speaker whose clean clips are all shorter than 1 s still
    gets an enrollment, concatenated from the nearest clips up to >= 1 s."""
    segs = [{"start": 0.0, "end": 30.0, "speaker": "A"}]
    t = 40.0
    for i in range(12):
        segs.append({"start": t, "end": t + 0.6, "speaker": "B"})
        t += 2.0
    clips = rmd.enroll_clips(segs, "B", 0.3, 0.25)
    assert len(clips) == 12 and all(b - a == pytest.approx(0.6) for a, b in clips)
    bank = {"B": {"path": "/x/default_e/B.wav", "clips": [{"offset": 0, "len": 9600, "src_start": a, "src_end": b}
                                                         for a, b in clips]},
            "A": {"path": "/x/default_e/A.wav", "clips": [{"offset": 0, "len": 64000, "src_start": 0.0,
                                                           "src_end": 4.0}]}}
    lay = {"n": 64000, "sr": 16000, "holder": "A", "interrupter": "B",
           "layout": [{"kind": "core", "who": "A", "src_start": 20.0, "src_end": 21.0},
                      {"kind": "core", "who": "B", "src_start": 50.0, "src_end": 50.6}]}
    from collections import Counter
    st = Counter()
    rows = rmd.tse_rows({"id": "m", "split": "train", "recording": "default_e", "mix": "m.wav", "A": "a.wav",
                         "B": "b.wav"}, lay, bank, 0.3, st)
    tb = next(r for r in rows if r["target_role"] == "interrupter")
    assert len(tb["enroll_default"]) >= 2
    assert sum(tb["enroll_candidates"][i]["len"] for i in tb["enroll_default"]) >= 16000
    assert all(not (c["src_start"] < 50.9 and c["src_end"] > 49.7) for c in tb["enroll_candidates"])


def test_exclude_held_removes_the_whole_group():
    recs = [{"path": f"cache/{n}"} for n in ("showA_1", "showA_2", "showB_1", "solo")]
    keep = rmd.exclude_held(recs, {"showA_1"}, r"^(show\w)_")
    assert [r["path"] for r in keep] == ["cache/showB_1", "cache/solo"]
    keep = rmd.exclude_held(recs, {"showA_1"}, None, {"showA_1": "ch1", "solo": "ch1"})
    assert [r["path"] for r in keep] == ["cache/showA_2", "cache/showB_1"]
    assert rmd.exclude_held(recs, {"showA_1"})[0]["path"] == "cache/showA_2"


def test_eval_separation_scores_the_core(tmp_path):
    import eval_separation as ev
    names = ["default_v"]
    rows = rmd.build([_rec("default_v", 0)], _templates(names), str(tmp_path / "o"), n=6, window=4.0, seed=0,
                     val_share=0.0, noise_prob=0.0, subtype="FLOAT")
    est = tmp_path / "est"
    est.mkdir()
    rng = np.random.default_rng(0)
    for r in rows:      # near-perfect estimates, in swapped order (PIT must undo it)
        for k, tag in ((1, "s2"), (2, "s1")):
            x = sf.read(r["paths"][tag])[0]
            sf.write(str(est / f"{r['id']}_est{k}.wav"), x + 1e-4 * rng.standard_normal(len(x)), 16000,
                     subtype="FLOAT")
    res = ev.evaluate_bss(rows, lambda i, k: str(est / f"{i}_est{k}.wav"))
    assert len(res) == 2 * len(rows)
    assert all(x["core_si_sdri"] > 20 and x["full_si_sdri"] > 20 for x in res)
    assert {x["role"] for x in res} == {"holder", "interrupter"}
    for r in rows:      # the mixture itself: no improvement
        for k in (1, 2):
            sf.write(str(est / f"{r['id']}_est{k}.wav"), sf.read(r["paths"]["mix"])[0], 16000, subtype="FLOAT")
    res = ev.evaluate_bss(rows, lambda i, k: str(est / f"{i}_est{k}.wav"))
    assert all(abs(x["core_si_sdri"]) < 1e-6 for x in res)
    text = ev.summarize(res)
    assert "pattern=contained" in text and "role=interrupter" in text


def test_eval_separation_tse(tmp_path):
    import eval_separation as ev
    names = [f"default_t{i}" for i in range(2)]
    out = tmp_path / "o"
    rmd.build([_rec(n, i, n_seg=10) for i, n in enumerate(names)], _templates(names), str(out), n=6,
              window=4.0, seed=0, val_share=0.5, noise_prob=0.0, tse_mix="plain", subtype="FLOAT")
    trows = val.load_tse(str(out))
    mixes = {m["id"]: m for m in val.load_rows(str(out), "tse_mixes.jsonl")}
    res = ev.evaluate_tse(trows, mixes, lambda i: next(t["target"] for t in trows if t["id"] == i))
    assert len(res) == len(trows) and all(x["core_si_sdri"] > 30 for x in res)


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
