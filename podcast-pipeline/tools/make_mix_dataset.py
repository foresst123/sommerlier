"""Build synthetic overlap training data from very clean single-speaker speech.

Reads a few recordings from the checkpoint cache (same source as
collect_overlap_samples.py), and for each sample takes one clean stretch from each
of two different speakers of the SAME recording, places them on a shared timeline
so they overlap, and sets their relative loudness at random. The clean sources are
kept, so the mixture has exact ground truth:

  <id>/clean_A.wav   speaker A alone, on the mixture's timeline (zeros elsewhere)
  <id>/clean_B.wav   speaker B alone, same timeline, at the level used in the mix
  <id>/mixture.wav   clean_A + clean_B, sample for sample (32-bit float)
  <id>/meta.json     recording, speakers, source times, offsets, overlap, levels
  manifest.jsonl     one line per sample; "split" is train or val, decided per
                     recording so a speaker/room never sits on both sides

"Clean" means nobody else talks within --margin seconds, no music, and the
stretch is clearly speech (level and active-frame checks) -- cut from the cached
post-music waveform, not from the separator's output.

    python tools/make_mix_dataset.py --cache cache --out mix_data --n 1000 --files 30
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_overlap_samples as cos  # noqa: E402

PATTERNS = (("partial", 0.5), ("contained", 0.25), ("full", 0.25))


def dbfs(x: np.ndarray) -> float:
    return 20 * math.log10(max(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))), 1e-9))


def looks_like_speech(x: np.ndarray, sr: int, floor_db: float = -50.0,
                      quiet_db: float = -55.0, max_quiet: float = 0.4) -> bool:
    """Loud enough overall and not mostly silence in 25 ms frames."""
    if len(x) < sr or dbfs(x) < floor_db or np.max(np.abs(x)) >= 0.999:
        return False
    n = int(0.025 * sr)
    frames = x[: len(x) // n * n].reshape(-1, n)
    quiet = np.mean([dbfs(f) < quiet_db for f in frames])
    return quiet <= max_quiet


def clean_pool(segs: List[dict], margin: float, min_len: float) -> Dict[str, List[Tuple[float, float]]]:
    pool = {}
    for spk in sorted({s["speaker"] for s in segs}):
        clips = cos.clean_clips(segs, spk, margin, min_len)
        if clips:
            pool[spk] = clips
    return pool


def eligible(pool: Dict[str, List[Tuple[float, float]]], min_total: float) -> List[str]:
    return [s for s, c in pool.items() if sum(b - a for a, b in c) >= min_total]


def crop(rng: random.Random, clips: List[Tuple[float, float]], audio, sr: int,
         lmin: float, lmax: float, tries: int = 12) -> Optional[Tuple[float, float, np.ndarray]]:
    """A random stretch of clean speech, `lmin`..`lmax` seconds, that passes the speech check."""
    usable = [c for c in clips if c[1] - c[0] >= lmin]
    for _ in range(tries):
        if not usable:
            return None
        a, b = rng.choices(usable, weights=[y - x for x, y in usable])[0]
        length = min(b - a, rng.uniform(lmin, lmax))
        start = a + rng.uniform(0, (b - a) - length)
        x = np.asarray(audio[int(start * sr):int((start + length) * sr)], dtype=np.float32)
        if looks_like_speech(x, sr):
            return start, start + length, x
    return None


def place(rng: random.Random, la: int, lb: int, sr: int) -> Tuple[str, int, int]:
    """(pattern, offset of A, offset of B) in samples, A first; the overlap is never empty."""
    pattern = rng.choices([p for p, _ in PATTERNS], weights=[w for _, w in PATTERNS])[0]
    if pattern == "full":
        return pattern, 0, 0
    if pattern == "contained":
        if lb >= la:
            pattern = "partial"
        else:
            return pattern, 0, rng.randint(0, la - lb)
    shortest = min(la, lb)
    ov = max(int(0.1 * sr), int(rng.uniform(0.15, 0.9) * shortest))
    return pattern, 0, la - ov


def mix(a: np.ndarray, b: np.ndarray, pattern_off: Tuple[str, int, int], db_b: float,
        base_db: float = -23.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Stems on one timeline at the chosen levels, and their sum. Returns A, B, mixture, overlap samples."""
    pattern, oa, ob = pattern_off
    if pattern == "full":                       # equal length so both really start and end together
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    a = a * 10 ** ((base_db - dbfs(a)) / 20)
    b = b * 10 ** ((base_db + db_b - dbfs(b)) / 20)
    total = max(oa + len(a), ob + len(b))
    A, B = np.zeros(total, np.float32), np.zeros(total, np.float32)
    A[oa:oa + len(a)] = a
    B[ob:ob + len(b)] = b
    M = A + B
    peak = float(np.max(np.abs(M)))
    if peak > 0.95:                              # one factor for all three keeps A + B == M
        A, B, M = (v * (0.95 / peak) for v in (A, B, M))
    overlap = max(0, min(oa + len(a), ob + len(b)) - max(oa, ob))
    return A, B, M, overlap


def build(recs: List[dict], out: str, n: int = 1000, seed: int = 0, lmin: float = 2.0,
          lmax: float = 8.0, db_range: float = 10.0, margin: float = 0.5,
          min_total: float = 10.0, val_share: float = 0.1, workers: int = 1) -> List[dict]:
    import soundfile as sf
    rng = random.Random(seed)
    segs_of = {}
    for rec, (_, segs) in zip(recs, cos.scan_all(recs, 1.0, 0.1, workers)):
        segs_of[rec["path"]] = segs
    usable = []
    for rec in recs:
        pool = clean_pool(segs_of[rec["path"]], margin, lmin)
        spk = eligible(pool, min_total)
        if len(spk) >= 2:
            usable.append((rec, pool, spk))
    if not usable:
        raise SystemExit("no recording has two speakers with enough clean speech")
    order = list(range(len(usable)))
    rng.shuffle(order)
    n_val = max(1, round(val_share * len(usable))) if val_share > 0 and len(usable) > 1 else 0
    val = {usable[i][0]["path"] for i in order[:n_val]}
    print(f"{len(usable)} usable recording(s), {len(val)} held out for val", flush=True)

    os.makedirs(out, exist_ok=True)
    rows, k = [], 0
    audio_of = {}
    attempts = 0
    while k < n and attempts < n * 20:
        rec, pool, spk = usable[attempts % len(usable)]
        attempts += 1
        if rec["path"] not in audio_of:
            audio_of.clear()                     # one waveform open at a time
            audio_of[rec["path"]] = rec["audio"]()
        audio, sr = audio_of[rec["path"]]
        sa, sb = rng.sample(spk, 2)
        ca = crop(rng, pool[sa], audio, sr, lmin, lmax)
        cb = crop(rng, pool[sb], audio, sr, lmin, lmax)
        if ca is None or cb is None:
            continue
        placed = place(rng, len(ca[2]), len(cb[2]), sr)
        db_b = rng.uniform(-db_range, db_range)
        A, B, M, ov = mix(ca[2], cb[2], placed, db_b)
        name = f"{k:04d}"
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        for fname, x in (("clean_A", A), ("clean_B", B), ("mixture", M)):
            sf.write(os.path.join(d, f"{fname}.wav"), x, sr, subtype="FLOAT")
        meta = {"id": name, "recording": os.path.relpath(rec["path"]), "sample_rate": sr,
                "split": "val" if rec["path"] in val else "train",
                "pattern": placed[0], "duration": round(len(M) / sr, 3),
                "overlap_seconds": round(ov / sr, 3), "level_B_vs_A_db": round(db_b, 2),
                "A": {"speaker": sa, "source": [round(ca[0], 3), round(ca[1], 3)],
                      "offset": round(placed[1] / sr, 3)},
                "B": {"speaker": sb, "source": [round(cb[0], 3), round(cb[1], 3)],
                      "offset": round(placed[2] / sr, 3)},
                "clock": "post-music waveform of the recording"}
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        rows.append(meta)
        k += 1
        if k % 100 == 0:
            print(f"  wrote {k}/{n}", flush=True)
    with open(os.path.join(out, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in rows:
            f.write(json.dumps({k2: m[k2] for k2 in ("id", "split", "pattern", "duration",
                                                     "overlap_seconds", "level_B_vs_A_db",
                                                     "recording")}, ensure_ascii=False) + "\n")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--files", type=int, default=30, help="recordings to read, picked at random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lmin", type=float, default=2.0, help="shortest stretch per speaker, seconds")
    p.add_argument("--lmax", type=float, default=8.0)
    p.add_argument("--db-range", type=float, default=10.0, help="B's level vs A is uniform in +-this dB")
    p.add_argument("--margin", type=float, default=0.5, help="nobody else may talk this close")
    p.add_argument("--min-total", type=float, default=10.0, help="clean seconds a speaker needs")
    p.add_argument("--val-share", type=float, default=0.1, help="share of recordings kept for validation")
    p.add_argument("--exclude-index", help="index.json of collect_overlap_samples; its recordings are skipped")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = p.parse_args()
    recs = cos.cache_recordings(a.cache)
    if a.exclude_index:
        with open(a.exclude_index, encoding="utf-8") as f:
            held = {s["recording"] for s in json.load(f)["samples"]}
        recs = [r for r in recs if os.path.relpath(r["path"]) not in held]
    print(f"{len(recs)} recording(s) available", flush=True)
    if a.files and a.files < len(recs):
        recs = random.Random(a.seed).sample(recs, a.files)
    rows = build(recs, a.out, a.n, a.seed, a.lmin, a.lmax, a.db_range, a.margin,
                 a.min_total, a.val_share, a.workers)
    split = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
    print(f"{len(rows)} sample(s) -> {a.out}  {split}")


if __name__ == "__main__":
    main()
