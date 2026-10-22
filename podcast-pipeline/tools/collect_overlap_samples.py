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

With only the checkpoint cache (no stage output), read it instead; run from
podcast-pipeline/ so the pickles can import the segment classes. The audio is
the cached post-music waveform (24 kHz), which is the same clock:

    python tools/collect_overlap_samples.py --cache cache --out overlap_samples
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

CACHE_SR = 24000
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


def _plain(v):
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def stage_recordings(runs: List[str]) -> List[dict]:
    """Recordings that have 03_separation/segments.json and 01_music/after_music.wav."""
    found = []
    for run in runs:
        for seg in glob.glob(os.path.join(run, "**", "03_separation", "segments.json"), recursive=True):
            rec = os.path.dirname(os.path.dirname(seg))
            wav = os.path.join(rec, "01_music", "after_music.wav")
            if os.path.exists(wav):
                found.append(rec)
    return [_stage_recording(r) for r in sorted(set(found))]


def _stage_recording(rec: str) -> dict:
    def segments():
        with open(os.path.join(rec, "03_separation", "segments.json"), encoding="utf-8") as f:
            return [s for s in json.load(f) if "start" in s and "end" in s]

    def audio():
        import soundfile as sf
        return sf.read(os.path.join(rec, "01_music", "after_music.wav"), always_2d=False)

    def clips(touching, out_dir):
        names, src = [], os.path.join(rec, "03_separation", "audio", "separated")
        for s in touching:
            for p in glob.glob(os.path.join(src, f"{s['index']}_{s['speaker']}_*.wav")):
                os.makedirs(out_dir, exist_ok=True)
                shutil.copy(p, os.path.join(out_dir, os.path.basename(p)))
                names.append(os.path.basename(p))
        return names

    return {"path": rec, "segments": segments, "audio": audio, "clips": clips}


def _newest(pattern: str) -> Optional[str]:
    hits = glob.glob(pattern, recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None


def cache_recordings(cache: str) -> List[dict]:
    """Recordings with a separation checkpoint and a cached post-music waveform."""
    out = []
    for job in sorted(glob.glob(os.path.join(cache, "*"))):
        sep = _newest(os.path.join(job, "separation", "**", "result.pkl"))
        wav = _newest(os.path.join(job, "processed_audio", "**", "result.npy"))
        if sep and wav:
            out.append(_cache_recording(job, sep, wav))
    return out


def _cache_recording(job: str, sep: str, wav: str) -> dict:
    import pickle

    def load_rows():
        with open(sep, "rb") as f:
            return pickle.load(f)

    def segments():
        # Scanning every recording must not keep its waveforms: drop them at once.
        out = []
        for i, r in enumerate(load_rows()):
            d = dict(r.__dict__) if hasattr(r, "__dict__") else dict(r)
            d.pop("audio", None)
            out.append({k: _plain(v) for k, v in d.items()} | {"_i": i})
        return out

    def audio():
        return np.load(wav, mmap_mode="r", allow_pickle=False), CACHE_SR

    def clips(touching, out_dir):
        import soundfile as sf
        rows, names = load_rows(), []     # only for the few recordings actually sampled
        for s in touching:
            a = getattr(rows[s["_i"]], "audio", None)
            if a is None or not len(a) or not (s.get("bss_spans") or s.get("bss_failed_spans")):
                continue
            os.makedirs(out_dir, exist_ok=True)
            name = f"{s['index']}_{s['speaker']}.wav"
            sf.write(os.path.join(out_dir, name), np.asarray(a), CACHE_SR)
            names.append(name)
        return names

    return {"path": job, "segments": segments, "audio": audio, "clips": clips}


def events_of(rec: dict, merge_gap: float, min_overlap: float) -> Tuple[List[dict], List[dict]]:
    segs = rec["segments"]()
    spans = [(a, b) for a, b in cross_speaker_overlaps(segs) if b - a >= min_overlap]
    events = []
    for a, b in merge_close(spans, merge_gap):
        inside = [s for s in segs if s["start"] < b and s["end"] > a]
        events.append({"rec": rec["path"], "start": a, "end": b, "dur": b - a,
                       "speakers": sorted({s["speaker"] for s in inside})})
    return events, segs


_RECS: List[dict] = []


def _scan_one(args):
    i, merge_gap, min_overlap = args
    return events_of(_RECS[i], merge_gap, min_overlap)


def scan_all(recs: List[dict], merge_gap: float, min_overlap: float, workers: int):
    """events_of for every recording. Reading a cache pickle (it carries the separated
    audio) dominates the run, so recordings are read in parallel worker processes;
    only the small segment lists travel back."""
    global _RECS
    _RECS = recs
    jobs = [(i, merge_gap, min_overlap) for i in range(len(recs))]
    results = []
    if workers > 1 and len(recs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            for k, r in enumerate(pool.imap(_scan_one, jobs), 1):
                results.append(r)
                if k % 20 == 0 or k == len(jobs):
                    print(f"  scanned {k}/{len(jobs)}", flush=True)
    else:
        for k, j in enumerate(jobs, 1):
            results.append(_scan_one(j))
            if k % 20 == 0 or k == len(jobs):
                print(f"  scanned {k}/{len(jobs)}", flush=True)
    return results


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
    quota = {name: n // len(BINS) + (i < n % len(BINS)) for i, name in enumerate(by_bin)}

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


def write_sample(e: dict, rec: dict, segs: List[dict], out_dir: str, context: float,
                 ref_seconds: float, margin: float) -> dict:
    import soundfile as sf
    audio, sr = rec["audio"]()
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
        gap = np.zeros(int(0.2 * sr), dtype=np.asarray(audio[:1]).dtype)
        parts = []
        for a, b in chosen:
            piece = cut(a, b)
            sf.write(os.path.join(d, "clips", f"{a:.3f}-{b:.3f}.wav"), piece, sr)
            parts += [piece, gap]
        sf.write(os.path.join(d, "reference.wav"), np.concatenate(parts[:-1]), sr)
        refs[spk] = {"seconds": round(acc, 2), "clips": [[round(a, 3), round(b, 3)] for a, b in chosen]}

    touching = [s for s in segs if s["start"] < e["end"] and s["end"] > e["start"]]
    copied = rec["clips"](touching, os.path.join(out_dir, "separated"))
    meta = {
        "recording": os.path.relpath(rec["path"]),
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
            "bss_failed_spans": [_plain(sp) for s in touching for sp in s.get("bss_failed_spans") or []],
            "separated_clips": copied},
        "references": refs,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def build(recs: List[dict], out: str, n: int = 100, context: float = 60.0, seed: int = 0,
          merge_gap: float = 1.0, min_overlap: float = 0.1, ref_seconds: float = 30.0,
          margin: float = 0.3, workers: int = 1) -> List[dict]:
    segs_of, events = {}, []
    by_path = {r["path"]: r for r in recs}
    for rec, (ev, segs) in zip(recs, scan_all(recs, merge_gap, min_overlap, workers)):
        segs_of[rec["path"]] = segs
        events += ev
    chosen = select(events, n, seed, context)
    os.makedirs(out, exist_ok=True)
    index = []
    for i, e in enumerate(chosen):
        name = f"{i:03d}_{e['bin']}_{os.path.basename(e['rec'].rstrip('/'))}_{e['start']:.0f}s"
        meta = write_sample(e, by_path[e["rec"]], segs_of[e["rec"]], os.path.join(out, name), context, ref_seconds, margin)
        index.append({"folder": name, "bin": e["bin"], "overlap_duration": meta["overlap"]["duration"],
                      "recording": meta["recording"]})
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"count": len(index), "available_events": len(events), "samples": index},
                  f, ensure_ascii=False, indent=2)
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=[], help="run folders with stage outputs (searched recursively)")
    p.add_argument("--cache", help="checkpoint cache folder (cache/default_<file>/...), used when there is no stage output")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--context", type=float, default=60.0, help="seconds of context per sample")
    p.add_argument("--merge-gap", type=float, default=1.0, help="overlaps closer than this form one event")
    p.add_argument("--min-overlap", type=float, default=0.1)
    p.add_argument("--ref-seconds", type=float, default=30.0, help="clean reference length per speaker")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                   help="processes reading recordings in parallel")
    p.add_argument("--files", type=int, help="read only N recordings, picked at random (--seed)")
    a = p.parse_args()
    recs = stage_recordings(a.runs) + (cache_recordings(a.cache) if a.cache else [])
    if a.files and a.files < len(recs):
        recs = random.Random(a.seed).sample(recs, a.files)
    print(f"{len(recs)} recording(s) found", flush=True)
    idx = build(recs, a.out, a.n, a.context, a.seed, a.merge_gap, a.min_overlap, a.ref_seconds, workers=a.workers)
    counts = {}
    for s in idx:
        counts[s["bin"]] = counts.get(s["bin"], 0) + 1
    print(f"{len(idx)} sample(s) -> {a.out}  {counts}")


if __name__ == "__main__":
    main()
