"""Render training mixtures that follow the overlaps measured in real recordings.

Every sample is built from a REAL overlap template (measure_real_overlap_stats.py writes
pairs.csv): who held the floor for how long, how long the other speaker spoke, how much
they overlapped (full / contained / partial) and the level difference. The template is
realised with clean single-speaker stretches of the same recording (nobody else talks
within --margin seconds, no music), placed on one timeline and cut to a --window second
chunk that contains the overlap. Real overlaps here are mostly short interjections inside
a longer turn, so most chunks hold one speaker for most of the window and a brief second
voice -- not two long overlapping utterances.

  <out>/<split>/<id>_mix.wav, _s1.wav, _s2.wav   --sr (default 16 kHz) mono, PCM-16
  <out>/chunks.jsonl                              one row per chunk: the single source of
                                                  truth every per-model manifest derives from
  <out>/manifests/{mossformer2,tiger,speechbrain}/  lists in the format each trainer reads

What is drawn per chunk (--mix real:strat:hard, default 0.7:0.2:0.1):
  real   a template from the SAME recording (its own measured overlaps)
  strat  a template from the split's pool, equal share per overlap-length bin, so long
         overlaps are not starved by the short-heavy real distribution
  hard   a long overlap (>= 1 s) with a near-zero or very wide level difference
Level: B relative to A comes from the template (+/-1 dB jitter); hard cases override it.
Noise: with --noise-prob, background taken from the recording's own non-speech gaps is
added to the MIXTURE only (stems stay clean), at a speech-to-noise ratio mostly in the
range measured on the data (25-45 dB) with a minority lower. Where noise is added
mix != s1 + s2, by design. No synthetic reverb: the stems already carry the recording's room.

    python tools/render_mix_dataset.py --cache cache --stats overlap_stats \\
        --out mix_16k_w4 --n 20000 --window 4 --exclude-index overlap_samples/index.json
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_overlap_samples as cos  # noqa: E402
import measure_real_overlap_stats as mrs  # noqa: E402

LEN_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 1e9)]
BASE_DB = -23.0           # level of the holder's active speech
MARGIN = 0.2              # seconds of the window kept on each side of the overlap
MIN_ACTIVE = 0.15         # each stem needs this much active speech in the chunk
FRAME = 0.01


# ----------------------------------------------------------------------------- templates
def load_templates(stats_dir: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    with open(os.path.join(stats_dir, "pairs.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["recording"]].append({
                "rec": r["recording"], "overlap": float(r["overlap"]), "pattern": r["pattern"],
                "first_len": float(r["first_len"]), "second_len": float(r["second_len"]),
                "delta_db": float(r["delta_db"]) if r["delta_db"] != "" else None})
    return out


def pick_template(rng: random.Random, kind: str, own: List[dict], pool: List[dict]):
    if kind == "real" and own:
        return rng.choice(own), "real"
    if kind == "strat" or (kind == "real" and not own):
        lo, hi = rng.choice(LEN_BINS)
        cand = [t for t in pool if lo <= t["overlap"] < hi]
        if cand:
            return rng.choice(cand), "strat"
    cand = [t for t in pool if t["overlap"] >= 1.0] or pool
    t = dict(rng.choice(cand))
    t["delta_db"] = rng.uniform(-2, 2) if rng.random() < 0.5 else rng.choice([-1, 1]) * rng.uniform(6, 12)
    return t, "hard"


# ----------------------------------------------------------------------------- audio helpers
def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def snap(x: np.ndarray, sr: int, i: int, shift: float = 0.08) -> int:
    """Move cut point `i` to the quietest nearby spot so crops do not start mid-word."""
    n = max(1, int(FRAME * sr))
    lo, hi = max(0, i - int(shift * sr)), min(len(x) - n, i + int(shift * sr))
    if hi <= lo:
        return i
    idx = np.arange(lo, hi + 1, max(1, n // 2))
    return int(idx[int(np.argmin([float(np.mean(x[j:j + n] ** 2)) for j in idx]))])


def pull(rng: random.Random, clips: List[Tuple[float, float]], audio, sr: int,
         need: float) -> Optional[np.ndarray]:
    """A clean stretch of about `need` seconds; whole clip when it fits, else snapped crop."""
    cand = [c for c in clips if c[1] - c[0] >= need - 1e-6]
    if not cand:
        return None
    a, b = rng.choice(cand)
    x = np.asarray(audio[int(a * sr):int(b * sr)], dtype=np.float32)
    n = int(need * sr)
    if len(x) - n > int(0.05 * sr):
        s = rng.randint(0, len(x) - n)
        s, e = snap(x, sr, s), snap(x, sr, s + n)
        x = x[s:max(e, s + 1)]
    f = min(len(x) // 2, int(0.01 * sr))
    if f > 0:
        ramp = np.linspace(0, 1, f, dtype=np.float32)
        x = x.copy()
        x[:f] *= ramp
        x[-f:] *= ramp[::-1]
    return x


def level_db(x: np.ndarray, sr: int) -> float:
    lv = mrs.speech_level_db(x, sr)
    return lv if lv is not None else float(10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-12))


def active_seconds(x: np.ndarray, sr: int, floor_db: float = -45.0) -> float:
    n = int(FRAME * sr)
    if len(x) < n:
        return 0.0
    f = np.asarray(x[: len(x) // n * n], np.float64).reshape(-1, n)
    return float(np.sum(10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12) > floor_db)) * FRAME


def noise_bank(segs: List[dict], audio, sr: int, max_pieces: int = 30) -> Optional[np.ndarray]:
    cover = cos.merge_close([(s["start"], s["end"]) for s in segs], 0.0)
    gaps = [(cover[i][1], cover[i + 1][0]) for i in range(len(cover) - 1) if cover[i + 1][0] - cover[i][1] >= 0.8]
    parts = []
    for a, b in mrs._evenly(gaps, max_pieces):
        mid, half = (a + b) / 2, min((b - a) / 2, 1.5)
        parts.append(np.asarray(audio[int((mid - half) * sr):int((mid + half) * sr)], dtype=np.float32))
    bank = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return bank if len(bank) >= 2 * sr else None


# ----------------------------------------------------------------------------- one chunk
def place_window(rng: random.Random, ov_start: float, ov_end: float, window: float) -> float:
    ov = ov_end - ov_start
    if ov + 2 * MARGIN <= window:
        lo, hi = max(ov_end + MARGIN - window, -1.0), ov_start - MARGIN
        return rng.uniform(lo, max(lo, hi))
    return ov_start - MARGIN


def realize(rng: random.Random, t: dict, pools: Dict[str, list], audio, sr_src: int, window: float,
            sr_out: int, noise: Optional[np.ndarray], noise_prob: float, delta_jitter: float = 1.0):
    spk = [s for s, c in pools.items() if c]
    if len(spk) < 2:
        return None
    holder, other = rng.sample(spk, 2)
    F = min(t["first_len"], 20.0)
    S = min(t["second_len"], F)
    pattern = t["pattern"]
    if pattern == "contained":
        S = min(S, max(F - 0.05, 0.1))
        bs = rng.uniform(0.0, max(0.0, F - S))
    elif pattern == "full":
        bs = 0.0
    else:
        bs = max(0.0, F - min(t["overlap"], S, F))
    ov0, ov1 = bs, min(F, bs + S)
    ws = place_window(rng, ov0, ov1, window)
    a0, a1 = max(0.0, ws), min(F, ws + window)
    b0, b1 = max(bs, ws), min(bs + S, ws + window)
    if a1 - a0 < MIN_ACTIVE or b1 - b0 < MIN_ACTIVE:
        return None
    xa = pull(rng, pools[holder], audio, sr_src, a1 - a0)
    xb = pull(rng, pools[other], audio, sr_src, b1 - b0)
    if xa is None or xb is None:
        return None
    n = int(round(window * sr_src))
    A, B = np.zeros(n, np.float32), np.zeros(n, np.float32)
    ia, ib = int(round((a0 - ws) * sr_src)), int(round((b0 - ws) * sr_src))
    A[ia:ia + len(xa)] = xa[: max(0, n - ia)]
    B[ib:ib + len(xb)] = xb[: max(0, n - ib)]
    delta = t.get("delta_db")
    delta = (0.0 if delta is None else delta) + rng.gauss(0, delta_jitter) if t.get("kind") != "hard" \
        else t["delta_db"]
    A *= 10 ** ((BASE_DB - level_db(xa, sr_src)) / 20)
    B *= 10 ** ((BASE_DB + delta - level_db(xb, sr_src)) / 20)
    A, B = resample(A, sr_src, sr_out), resample(B, sr_src, sr_out)
    if active_seconds(A, sr_out) < MIN_ACTIVE or active_seconds(B, sr_out) < MIN_ACTIVE:
        return None
    M = A + B
    snr = None
    if noise is not None and rng.random() < noise_prob:
        snr = rng.uniform(10, 25) if rng.random() < 0.2 else rng.uniform(25, 45)
        m = int(round(window * sr_src))
        start = rng.randint(0, max(0, len(noise) - m))
        seg = np.resize(noise[start:start + m] if len(noise) >= m else noise, m)
        seg = resample(seg, sr_src, sr_out)
        p = float(np.mean(seg.astype(np.float64) ** 2)) + 1e-12
        M = M + seg * math.sqrt(10 ** ((BASE_DB - snr) / 10) / p)
    peak = max(float(np.max(np.abs(x))) for x in (A, B, M))
    if peak > 0.95:
        A, B, M = (x * (0.95 / peak) for x in (A, B, M))
    return {"A": A, "B": B, "M": M, "meta": {
        "pattern": pattern, "holder": holder, "interrupter": other, "delta_db": round(delta, 2),
        "noise_snr_db": None if snr is None else round(snr, 1),
        "overlap_in_window": round(max(0.0, min(ov1, ws + window) - max(ov0, ws)), 3),
        "overlap_start_in_window": round(max(ov0, ws) - ws, 3),
        "window_start_in_timeline": round(ws, 3), "template_overlap": round(t["overlap"], 3)}}


# ----------------------------------------------------------------------------- one recording
_JOBS: List[tuple] = []


def render_recording(job: tuple) -> List[dict]:
    import soundfile as sf
    rec, split, n, own, pool, cfg, seed, out = job
    rng = random.Random(seed)
    segs = rec["segments"]()
    audio, sr_src = rec["audio"]()
    pools = {}
    for s in sorted({x["speaker"] for x in segs}):
        c = cos.clean_clips(segs, s, cfg["margin"], cfg["min_clip"])
        if c:
            pools[s] = c
    noise = noise_bank(segs, audio, sr_src) if cfg["noise_prob"] > 0 else None
    name = os.path.basename(rec["path"].rstrip("/"))
    kinds, weights = ["real", "strat", "hard"], cfg["mix"]
    rows, tries = [], 0
    sdir = os.path.join(out, split)
    os.makedirs(sdir, exist_ok=True)
    while len(rows) < n and tries < n * 30:
        tries += 1
        t, kind = pick_template(rng, rng.choices(kinds, weights)[0], own, pool)
        t = dict(t, kind=kind)
        r = realize(rng, t, pools, audio, sr_src, cfg["window"], cfg["sr"], noise, cfg["noise_prob"])
        if r is None:
            continue
        cid = f"{name}_{len(rows):04d}"
        flip = rng.random() < 0.5
        s1, s2 = (r["B"], r["A"]) if flip else (r["A"], r["B"])
        paths = {}
        for tag, x in (("mix", r["M"]), ("s1", s1), ("s2", s2)):
            p = os.path.join(sdir, f"{cid}_{tag}.wav")
            sf.write(p, x, cfg["sr"], subtype=cfg["subtype"])
            paths[tag] = os.path.abspath(p)
        rows.append({"id": cid, "split": split, "recording": name, "kind": kind,
                     "window": cfg["window"], "sr": cfg["sr"], "samples": len(r["M"]), "paths": paths,
                     "target_order": "interrupter,holder" if flip else "holder,interrupter", **r["meta"]})
    return rows


def _run(i: int) -> List[dict]:
    return render_recording(_JOBS[i])


# ----------------------------------------------------------------------------- manifests
def write_manifests(out: str, rows: List[dict]) -> None:
    base = os.path.join(out, "manifests")
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val")}

    def lines(path, items):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(items) + ("\n" if items else ""))

    for s, name in (("train", "tr_list.txt"), ("val", "cv_list.txt")):
        lines(os.path.join(base, "mossformer2", name),
              [f"{r['paths']['mix']} {r['paths']['s1']} {r['paths']['s2']}" for r in by[s]])
    for s, d in (("train", "train"), ("val", "val"), ("val", "test")):
        for tag, key in (("mix", "mix"), ("s1", "s1"), ("s2", "s2")):
            p = os.path.join(base, "tiger", d, f"{tag}.json")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump([[r["paths"][key], r["samples"]] for r in by[s]], f)
    for s, name in (("train", "train.csv"), ("val", "valid.csv")):
        p = os.path.join(base, "speechbrain", name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ID", "duration", "mix_wav", "mix_wav_format", "mix_wav_opts",
                        "s1_wav", "s1_wav_format", "s1_wav_opts", "s2_wav", "s2_wav_format", "s2_wav_opts"])
            for r in by[s]:
                w.writerow([r["id"], r["window"], r["paths"]["mix"], "wav", "",
                            r["paths"]["s1"], "wav", "", r["paths"]["s2"], "wav", ""])


# ----------------------------------------------------------------------------- driver
def build(recs: List[dict], templates: Dict[str, List[dict]], out: str, n: int = 20000,
          window: float = 4.0, sr: int = 16000, seed: int = 0, val_share: float = 0.05,
          mix=(0.7, 0.2, 0.1), noise_prob: float = 0.5, margin: float = 0.3, min_clip: float = 0.25,
          subtype: str = "PCM_16", workers: int = 1, per_rec_cap: int = 100) -> List[dict]:
    global _JOBS
    rng = random.Random(seed)
    names = {os.path.basename(r["path"].rstrip("/")): r for r in recs}
    usable = [nm for nm in names if templates.get(nm)]
    if not usable:
        raise SystemExit("no recording has measured overlap templates")
    order = sorted(usable)
    rng.shuffle(order)
    n_val = max(1, round(val_share * len(order))) if val_share > 0 and len(order) > 1 else 0
    split_of = {nm: ("val" if i < n_val else "train") for i, nm in enumerate(order)}
    pool_of = {s: [t for nm in usable if split_of[nm] == s for t in templates[nm]] for s in ("train", "val")}
    w = {nm: min(len(templates[nm]), per_rec_cap) for nm in usable}
    # val gets its share of chunks by recording count, so it is not starved by few recordings
    total_w = sum(w.values())
    cfg = {"window": window, "sr": sr, "mix": list(mix), "noise_prob": noise_prob, "margin": margin,
           "min_clip": min_clip, "subtype": subtype}
    jobs = []
    for k, nm in enumerate(usable):
        cnt = max(1, round(n * w[nm] / total_w))
        jobs.append((names[nm], split_of[nm], cnt, templates[nm], pool_of[split_of[nm]], cfg,
                     seed * 100003 + k, out))
    _JOBS = jobs
    os.makedirs(out, exist_ok=True)
    rows: List[dict] = []
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as p:
            for k, r in enumerate(p.imap(_run, range(len(jobs))), 1):
                rows += r
                if k % 20 == 0 or k == len(jobs):
                    print(f"  rendered {k}/{len(jobs)} recordings, {len(rows)} chunks", flush=True)
    else:
        for k in range(len(jobs)):
            rows += _run(k)
    with open(os.path.join(out, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_manifests(out, rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache")
    p.add_argument("--runs", nargs="+", default=[])
    p.add_argument("--stats", required=True, help="folder with pairs.csv from measure_real_overlap_stats.py")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20000, help="chunks in total")
    p.add_argument("--window", type=float, default=4.0, help="seconds per chunk (4 for MossFormer2/SpeechBrain, 3 for TIGER)")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--val-share", type=float, default=0.05, help="share of recordings kept for validation")
    p.add_argument("--mix", default="0.7,0.2,0.1", help="real,strat,hard shares")
    p.add_argument("--noise-prob", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.3, help="nobody else may talk this close to a clean stretch")
    p.add_argument("--subtype", default="PCM_16", choices=["PCM_16", "FLOAT"])
    p.add_argument("--exclude-index", help="index.json of collect_overlap_samples: its recordings are never used")
    p.add_argument("--files", type=int, help="use only N recordings, picked at random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = p.parse_args()
    recs = cos.stage_recordings(a.runs) + (cos.cache_recordings(a.cache) if a.cache else [])
    if a.exclude_index:
        with open(a.exclude_index, encoding="utf-8") as f:
            held = {os.path.basename(s["recording"].rstrip("/")) for s in json.load(f)["samples"]}
        recs = [r for r in recs if os.path.basename(r["path"].rstrip("/")) not in held]
        print(f"excluded {len(held)} test recording(s)", flush=True)
    if a.files and a.files < len(recs):
        recs = random.Random(a.seed).sample(recs, a.files)
    print(f"{len(recs)} recording(s)", flush=True)
    rows = build(recs, load_templates(a.stats), a.out, a.n, a.window, a.sr, a.seed, a.val_share,
                 tuple(float(x) for x in a.mix.split(",")), a.noise_prob, a.margin, 0.25, a.subtype, a.workers)
    split = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
    print(f"{len(rows)} chunk(s) -> {a.out}  {split}  ({len(rows) * a.window / 3600:.1f} h)")


if __name__ == "__main__":
    main()
