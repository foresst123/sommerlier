"""Collect overlap samples from finished runs, to keep as a test set.

Reads the stage outputs of every recording under --runs (needs 03_separation/
segments.json and 01_music/after_music.wav) and picks ~N overlap events spread
across short, medium and long, and across recordings. Each sample folder holds:

  overlap_original.wav   the mixture exactly over the overlap, no padding
  context.wav            --context seconds of the same audio around it (the
                         overlap is centred, the window grows equally both sides
                         and slides inward at the recording's edges), so it holds
                         every overlap that falls in it
  speakers/<SPK>/reference.wav   clean single-speaker speech of that speaker from
                         elsewhere in the recording (nobody else talking), plus
                         clips/ with the pieces it was built from
  separated/             the separator's output for segments touching the overlap,
                         when it was written
  meta.json              times in the pipeline clock (= after_music.wav), the
                         overlaps inside the context, separation outcome

Times are in after_music.wav's clock, the clock every stage output shares; use
01_music/music_map.json "timeline" to get back to the original recording.

    python tools/collect_overlap_samples.py --runs /data/data_run_2/_final --out overlap_samples
"""

import argparse
import glob
import json
import os
import random
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np

BINS = [("short", 0.0, 0.5), ("medium", 0.5, 2.0), ("long", 2.0, 5.0), ("xlong", 5.0, 1e9)]


def merge_close(spans: List[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def cross_speaker_overlaps(segs: List[dict]) -> List[Tuple[float, float]]:
    """Time where two different speakers' segments are both active."""
    segs = sorted(segs, key=lambda s: s["start"])
    found = []
    for i, s in enumerate(segs):
        for o in segs[i + 1:]:
            if o["start"] >= s["end"]:
                break
            if o["speaker"] != s["speaker"]:
                a, b = o["start"], min(s["end"], o["end"])
                if b > a:
                    found.append((a, b))
    return merge_close(found, 0.0)


def context_window(a: float, b: float, length: float, total: float) -> Tuple[float, float]:
    """`length` seconds around [a, b]; slides inward at the edges rather than shrinking."""
    if b - a >= length:
        return a, b
    if total <= length:
        return 0.0, total
    lo = (a + b) / 2 - length / 2
    lo = min(max(0.0, lo), total - length)
    return lo, lo + length


def clean_clips(segs: List[dict], speaker: str, margin: float, min_len: float) -> List[Tuple[float, float]]:
    """Stretches of `speaker` with nobody else active within `margin`, music-free."""
    others = merge_close([(s["start"], s["end"]) for s in segs if s["speaker"] != speaker], 0.0)
    out = []
    for s in segs:
        if s["speaker"] != speaker or s.get("has_music"):
            continue
        pieces = [(s["start"], s["end"])]
        for a, b in others:
            a, b = a - margin, b + margin
            nxt = []
            for x, y in pieces:
                if b <= x or a >= y:
                    nxt.append((x, y))
                    continue
                if a > x:
                    nxt.append((x, a))
                if b < y:
                    nxt.append((b, y))
            pieces = nxt
        out += [(x, y) for x, y in pieces if y - x >= min_len]
    return sorted(set(out))


def find_recordings(runs: List[str]) -> List[str]:
    found = []
    for run in runs:
        for seg in glob.glob(os.path.join(run, "**", "03_separation", "segments.json"), recursive=True):
            rec = os.path.dirname(os.path.dirname(seg))
            if os.path.exists(os.path.join(rec, "01_music", "after_music.wav")):
                found.append(rec)
    return sorted(set(found))


def events_of(rec: str, merge_gap: float, min_overlap: float) -> Tuple[List[dict], List[dict]]:
    with open(os.path.join(rec, "03_separation", "segments.json"), encoding="utf-8") as f:
        segs = [s for s in json.load(f) if "start" in s and "end" in s]
    spans = [(a, b) for a, b in cross_speaker_overlaps(segs) if b - a >= min_overlap]
    events = []
    for a, b in merge_close(spans, merge_gap):
        inside = [s for s in segs if s["start"] < b and s["end"] > a]
        events.append({"rec": rec, "start": a, "end": b, "dur": b - a,
                       "speakers": sorted({s["speaker"] for s in inside})})
    return events, segs


def select(events: List[dict], n: int, seed: int, context: float) -> List[dict]:
    """Equal share per length bin (what a bin cannot fill goes to the others),
    spread across recordings, never two samples whose context windows overlap."""
    rng = random.Random(seed)
    by_bin: Dict[str, List[dict]] = {name: [] for name, _, _ in BINS}
    for e in events:
        for name, lo, hi in BINS:
            if lo <= e["dur"] < hi:
                e["bin"] = name
                by_bin[name].append(e)
    for pool in by_bin.values():
        rng.shuffle(pool)
    taken: List[dict] = []
    used: Dict[str, List[Tuple[float, float]]] = {}
    quota = {name: n // len(BINS) for name in by_bin}

    def clash(e):
        return any(abs((e["start"] + e["end"]) / 2 - c) < context for c in
                   [(a + b) / 2 for a, b in used.get(e["rec"], [])])

    def take(name, k):
        got = 0
        pool = by_bin[name]
        # least-used recording first, so one long file does not fill the set
        while pool and got < k:
            pool.sort(key=lambda e: sum(t["rec"] == e["rec"] for t in taken))
            e = next((x for x in pool if not clash(x)), None)
            if e is None:
                break
            pool.remove(e)
            taken.append(e)
            used.setdefault(e["rec"], []).append((e["start"], e["end"]))
            got += 1
        return got

    short = 0
    for name in by_bin:
        short += quota[name] - take(name, quota[name])
    names = list(by_bin)
    while short > 0 and any(by_bin[nm] for nm in names):
        progressed = False
        for nm in names:
            if short > 0 and take(nm, 1):
                short -= 1
                progressed = True
        if not progressed:
            break
    return sorted(taken, key=lambda e: (e["bin"], e["rec"], e["start"]))


def write_sample(e: dict, segs: List[dict], out_dir: str, context: float,
                 ref_seconds: float, margin: float) -> dict:
    import soundfile as sf
    rec = e["rec"]
    audio, sr = sf.read(os.path.join(rec, "01_music", "after_music.wav"), always_2d=False)
    total = len(audio) / sr

    def cut(a, b):
        return audio[int(max(0.0, a) * sr):int(min(total, b) * sr)]

    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, "overlap_original.wav"), cut(e["start"], e["end"]), sr)
    c0, c1 = context_window(e["start"], e["end"], context, total)
    sf.write(os.path.join(out_dir, "context.wav"), cut(c0, c1), sr)

    in_ctx = [(a, b) for a, b in cross_speaker_overlaps(segs) if a < c1 and b > c0]
    speakers = sorted({s["speaker"] for s in segs if s["start"] < c1 and s["end"] > c0})
    refs = {}
    for spk in speakers:
        clips = clean_clips(segs, spk, margin, 1.0)
        # longest first, up to ref_seconds
        clips.sort(key=lambda c: c[0] - c[1])
        chosen, acc = [], 0.0
        for a, b in clips:
            if acc >= ref_seconds:
                break
            chosen.append((a, min(b, a + (ref_seconds - acc))))
            acc += chosen[-1][1] - chosen[-1][0]
        chosen.sort()
        if not chosen:
            refs[spk] = {"seconds": 0.0, "clips": []}
            continue
        d = os.path.join(out_dir, "speakers", spk)
        os.makedirs(os.path.join(d, "clips"), exist_ok=True)
        gap = np.zeros(int(0.2 * sr), dtype=audio.dtype)
        parts = []
        for a, b in chosen:
            piece = cut(a, b)
            sf.write(os.path.join(d, "clips", f"{a:.3f}-{b:.3f}.wav"), piece, sr)
            parts += [piece, gap]
        sf.write(os.path.join(d, "reference.wav"), np.concatenate(parts[:-1]), sr)
        refs[spk] = {"seconds": round(acc, 2), "clips": [[round(a, 3), round(b, 3)] for a, b in chosen]}

    copied = []
    sep_dir = os.path.join(rec, "03_separation", "audio", "separated")
    for s in segs:
        if s["start"] < e["end"] and s["end"] > e["start"]:
            for p in glob.glob(os.path.join(sep_dir, f"{s['index']}_{s['speaker']}_*.wav")):
                os.makedirs(os.path.join(out_dir, "separated"), exist_ok=True)
                shutil.copy(p, os.path.join(out_dir, "separated", os.path.basename(p)))
                copied.append(os.path.basename(p))

    touching = [s for s in segs if s["start"] < e["end"] and s["end"] > e["start"]]
    meta = {
        "recording": os.path.relpath(rec),
        "sample_rate": sr, "clock": "after_music.wav",
        "bin": e["bin"], "speakers": e["speakers"],
        "overlap": {"start": round(e["start"], 3), "end": round(e["end"], 3),
                    "duration": round(e["dur"], 3)},
        "context": {"start": round(c0, 3), "end": round(c1, 3), "duration": round(c1 - c0, 3),
                    "overlap_offset": round(e["start"] - c0, 3)},
        "overlaps_in_context": [{"start": round(a, 3), "end": round(b, 3),
                                 "offset": round(a - c0, 3)} for a, b in in_ctx],
        "separation": {
            "bss_spans": [[round(float(x), 3) for x in sp] for s in touching
                          for sp in s.get("bss_spans") or []],
            "bss_failed_spans": [list(sp) for s in touching for sp in s.get("bss_failed_spans") or []],
            "separated_clips": copied},
        "references": refs,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def build(runs: List[str], out: str, n: int = 100, context: float = 60.0, seed: int = 0,
          merge_gap: float = 1.0, min_overlap: float = 0.1, ref_seconds: float = 30.0,
          margin: float = 0.3) -> List[dict]:
    segs_of, events = {}, []
    for rec in find_recordings(runs):
        ev, segs = events_of(rec, merge_gap, min_overlap)
        segs_of[rec] = segs
        events += ev
    chosen = select(events, n, seed, context)
    os.makedirs(out, exist_ok=True)
    index = []
    for i, e in enumerate(chosen):
        name = f"{i:03d}_{e['bin']}_{os.path.basename(e['rec'])}_{e['start']:.0f}s"
        meta = write_sample(e, segs_of[e["rec"]], os.path.join(out, name), context, ref_seconds, margin)
        index.append({"folder": name, "bin": e["bin"], "overlap_duration": meta["overlap"]["duration"],
                      "recording": meta["recording"]})
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"count": len(index), "available_events": len(events), "samples": index},
                  f, ensure_ascii=False, indent=2)
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run folders (searched recursively)")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--context", type=float, default=60.0, help="seconds of context per sample")
    p.add_argument("--merge-gap", type=float, default=1.0, help="overlaps closer than this form one event")
    p.add_argument("--min-overlap", type=float, default=0.1)
    p.add_argument("--ref-seconds", type=float, default=30.0, help="clean reference length per speaker")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    idx = build(a.runs, a.out, a.n, a.context, a.seed, a.merge_gap, a.min_overlap, a.ref_seconds)
    counts = {}
    for s in idx:
        counts[s["bin"]] = counts.get(s["bin"], 0) + 1
    print(f"{len(idx)} sample(s) -> {a.out}  {counts}")


if __name__ == "__main__":
    main()
