"""Measure what real overlaps look like, so synthetic mixtures can copy them.

Reads the segments of every recording in the checkpoint cache (or stage outputs) and
measures, over many real cross-speaker overlaps:

  duration        how long each overlap event is (events closer than --merge-gap merge)
  pattern         per overlapping pair: full (same start and end), contained (the later
                  speaker starts and ends inside the earlier one: an interjection or
                  backchannel) or partial (the later speaker starts inside the earlier
                  one and runs on after it ends: a hand-over)
  level           later speaker's level minus earlier speaker's, in dB. Proxy: each
                  speaker's level is taken from their own clean stretches (nobody else
                  talking) nearest in time, because the overlap itself is a sum
  density         events per minute, share of audio that is overlap, other events within
                  +-30 s (how many overlaps a training window should hold)
  noise           background level in stretches no segment covers, and the speech-to-noise
                  ratio against the speakers' level

Distributions are pooled with at most --max-per-file events per recording, so a few long
files do not dominate, and every headline number carries a 95% interval from resampling
RECORDINGS (the unit that varies), not events.

Writes events.csv, pairs.csv, recordings.csv, summary.json and summary.md under --out; the
CSVs let a renderer sample from the measured distribution instead of a guessed one.

    python tools/measure_real_overlap_stats.py --cache cache --out overlap_stats
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_overlap_samples as cos  # noqa: E402

PATTERNS = ("full", "contained", "partial")
DUR_EDGES = [0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 1e9]
DELTA_EDGES = [-1e9, -15, -10, -5, -2, 2, 5, 10, 15, 1e9]
QUANTILES = (5, 10, 25, 50, 75, 90, 95)
FRAME = 0.025


def speech_level_db(x: np.ndarray, sr: int) -> Optional[float]:
    """Mean-power level (dBFS) over the active 25 ms frames of `x`; None if there is none."""
    n = int(FRAME * sr)
    if len(x) < 5 * n:
        return None
    frames = np.asarray(x[: len(x) // n * n], dtype=np.float64).reshape(-1, n)
    power = np.mean(frames ** 2, axis=1)
    fdb = 10 * np.log10(power + 1e-12)
    active = fdb > max(-55.0, float(np.percentile(fdb, 95)) - 30.0)
    if active.sum() < 5:
        return None
    return float(10 * np.log10(power[active].mean() + 1e-12))


def classify(first, second, tol: float = 0.25) -> str:
    """`first` starts no later than `second`; both are (start, end)."""
    (a1, b1), (a2, b2) = first, second
    if abs(a1 - a2) <= tol and abs(b1 - b2) <= tol:
        return "full"
    if b2 <= b1 + tol:
        return "contained"
    return "partial"


def local_level(points, t: float, k: int = 5) -> Optional[float]:
    if not points:
        return None
    near = sorted(points, key=lambda p: abs(p[0] - t))[:k]
    return float(np.median([p[1] for p in near]))


def _evenly(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, n).astype(int)]


def measure_recording(rec: dict, merge_gap: float, min_overlap: float, max_clips: int = 40) -> dict:
    segs = rec["segments"]()
    audio, sr = rec["audio"]()
    total = len(audio) / sr

    def read(a, b):
        return np.asarray(audio[int(max(0.0, a) * sr):int(min(total, b) * sr)], dtype=np.float32)

    levels: Dict[str, list] = {}
    for spk in sorted({s["speaker"] for s in segs}):
        pts = []
        for a, b in _evenly(cos.clean_clips(segs, spk, 0.2, 1.0), max_clips):
            mid, half = (a + b) / 2, min((b - a) / 2, 3.0)
            lv = speech_level_db(read(mid - half, mid + half), sr)
            if lv is not None:
                pts.append((mid, lv))
        levels[spk] = pts

    ordered = sorted(segs, key=lambda s: (s["start"], -s["end"]))
    pairs, spans = [], []
    for i, s in enumerate(ordered):
        for o in ordered[i + 1:]:
            if o["start"] >= s["end"]:
                break
            if o["speaker"] == s["speaker"]:
                continue
            ov = min(s["end"], o["end"]) - o["start"]
            if ov < min_overlap:
                continue
            la, lb = local_level(levels.get(s["speaker"]), o["start"]), local_level(levels.get(o["speaker"]), o["start"])
            first_len, second_len = s["end"] - s["start"], o["end"] - o["start"]
            pairs.append({
                "t": o["start"], "overlap": ov,
                "pattern": classify((s["start"], s["end"]), (o["start"], o["end"])),
                "first_len": first_len, "second_len": second_len,
                "overlap_ratio": ov / min(first_len, second_len),
                "delta_db": (lb - la) if (la is not None and lb is not None) else None})
            spans.append((o["start"], o["start"] + ov))
    events = [{"start": a, "end": b, "duration": b - a} for a, b in cos.merge_close(spans, merge_gap)]
    starts = np.array([e["start"] for e in events])
    for e in events:
        e["neighbors_30s"] = int(np.sum(np.abs(starts - e["start"]) <= 30.0)) - 1

    cover = cos.merge_close([(s["start"], s["end"]) for s in segs], 0.0)
    gaps = [(cover[i][1], cover[i + 1][0]) for i in range(len(cover) - 1) if cover[i + 1][0] - cover[i][1] >= 0.5]
    gdb = []
    for a, b in _evenly(gaps, max_clips):
        mid, half = (a + b) / 2, min((b - a) / 2, 1.0)
        x = read(mid - half, mid + half)
        if len(x):
            gdb.append(float(10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-12)))
    noise = float(np.percentile(gdb, 25)) if len(gdb) >= 5 else None
    allpts = [p[1] for v in levels.values() for p in v]
    speech = float(np.median(allpts)) if allpts else None
    return {"name": os.path.basename(rec["path"].rstrip("/")), "path": rec["path"], "duration": total,
            "pairs": pairs, "events": events, "noise_db": noise, "speech_db": speech,
            "snr_db": (speech - noise) if (speech is not None and noise is not None) else None}


_RECS: List[dict] = []


def _one(i):
    return measure_recording(_RECS[i], *_ARGS)


_ARGS = (1.0, 0.1)


def measure_all(recs: List[dict], merge_gap: float, min_overlap: float, workers: int) -> List[dict]:
    global _RECS, _ARGS
    _RECS, _ARGS = recs, (merge_gap, min_overlap)
    out = []
    if workers > 1 and len(recs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            for k, r in enumerate(pool.imap(_one, range(len(recs))), 1):
                out.append(r)
                if k % 20 == 0 or k == len(recs):
                    print(f"  measured {k}/{len(recs)}", flush=True)
    else:
        for k in range(len(recs)):
            out.append(_one(k))
            if (k + 1) % 20 == 0 or k + 1 == len(recs):
                print(f"  measured {k + 1}/{len(recs)}", flush=True)
    return out


def quantiles(x) -> dict:
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if not len(x):
        return {}
    return {f"p{q}": round(float(np.percentile(x, q)), 3) for q in QUANTILES} | {"n": int(len(x))}


def histogram(x, edges) -> list:
    x = np.asarray([v for v in x if v is not None], dtype=float)
    counts, _ = np.histogram(x, bins=edges)
    n = max(1, counts.sum())
    return [{"from": edges[i], "to": edges[i + 1], "count": int(c), "share": round(float(c) / n, 4)}
            for i, c in enumerate(counts)]


def boot_ci(per_rec: List[np.ndarray], stat: Callable[[np.ndarray], float], n: int, rng: random.Random):
    """95% interval of `stat` from resampling recordings (the clustering unit)."""
    per_rec = [a for a in per_rec if len(a)]
    if len(per_rec) < 3:
        return None
    vals = []
    for _ in range(n):
        pick = [per_rec[rng.randrange(len(per_rec))] for _ in per_rec]
        vals.append(stat(np.concatenate(pick)))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def summarize(results: List[dict], max_per_file: int, seed: int, boot: int) -> dict:
    rng = random.Random(seed)

    def cap(rows):
        return rows if len(rows) <= max_per_file else rng.sample(rows, max_per_file)

    ev_dur = [np.array([e["duration"] for e in cap(r["events"])], float) for r in results]
    pairs = [cap(r["pairs"]) for r in results]
    pat_code = [np.array([PATTERNS.index(p["pattern"]) for p in rows], int) for rows in pairs]
    delta = [np.array([p["delta_db"] for p in rows if p["delta_db"] is not None], float) for rows in pairs]
    ratio = [np.array([p["overlap_ratio"] for p in rows], float) for rows in pairs]
    neigh = [np.array([e["neighbors_30s"] for e in cap(r["events"])], float) for r in results]
    minutes = sum(r["duration"] for r in results) / 60.0
    n_events = sum(len(r["events"]) for r in results)
    ov_time = sum(e["duration"] for r in results for e in r["events"])
    rec_rate = [len(r["events"]) / (r["duration"] / 60.0) for r in results if r["duration"] > 0]

    def cat(lst):
        lst = [a for a in lst if len(a)]
        return np.concatenate(lst) if lst else np.array([])

    S: dict = {
        "recordings": len(results), "recordings_with_overlap": sum(1 for r in results if r["events"]),
        "events_total": n_events, "pairs_total": sum(len(r["pairs"]) for r in results),
        "events_used": int(sum(len(a) for a in ev_dur)), "audio_minutes": round(minutes, 1),
        "max_per_file": max_per_file,
        "events_per_minute": {"pooled": round(n_events / minutes, 3) if minutes else None,
                              "per_recording": quantiles(rec_rate)},
        "overlap_share_of_audio": round(ov_time / (minutes * 60.0), 4) if minutes else None,
    }
    d = cat(ev_dur)
    S["event_duration_s"] = quantiles(d) | {
        "median_ci95": boot_ci(ev_dur, lambda a: float(np.median(a)), boot, rng),
        "p90_ci95": boot_ci(ev_dur, lambda a: float(np.percentile(a, 90)), boot, rng),
        "histogram": histogram(d, DUR_EDGES)}
    codes = cat(pat_code)
    S["pattern"] = {p: {"share": round(float(np.mean(codes == i)), 4) if len(codes) else None,
                        "ci95": boot_ci(pat_code, lambda a, i=i: float(np.mean(a == i)), boot, rng)}
                    for i, p in enumerate(PATTERNS)}
    dl = cat(delta)
    S["later_minus_earlier_speaker_db"] = quantiles(dl) | {
        "median_ci95": boot_ci(delta, lambda a: float(np.median(a)), boot, rng),
        "abs_median": round(float(np.median(np.abs(dl))), 3) if len(dl) else None,
        "abs_p90_ci95": boot_ci([np.abs(a) for a in delta], lambda a: float(np.percentile(a, 90)), boot, rng),
        "histogram": histogram(dl, DELTA_EDGES),
        "note": "proxy: speaker levels come from their own clean stretches nearest in time"}
    S["overlap_ratio_of_shorter_segment"] = quantiles(cat(ratio))
    S["other_events_within_30s"] = quantiles(cat(neigh))
    S["noise_floor_db"] = quantiles([r["noise_db"] for r in results])
    S["speech_db"] = quantiles([r["speech_db"] for r in results])
    S["speech_to_noise_db"] = quantiles([r["snr_db"] for r in results])
    return S


def write_outputs(results: List[dict], summary: dict, out: str) -> None:
    os.makedirs(out, exist_ok=True)

    def dump(name, header, rows):
        with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    dump("events.csv", ["recording", "start", "end", "duration", "neighbors_30s"],
         [[r["name"], f"{e['start']:.3f}", f"{e['end']:.3f}", f"{e['duration']:.3f}", e["neighbors_30s"]]
          for r in results for e in r["events"]])
    dump("pairs.csv", ["recording", "t", "overlap", "pattern", "first_len", "second_len", "overlap_ratio", "delta_db"],
         [[r["name"], f"{p['t']:.3f}", f"{p['overlap']:.3f}", p["pattern"], f"{p['first_len']:.3f}",
           f"{p['second_len']:.3f}", f"{p['overlap_ratio']:.3f}",
           "" if p["delta_db"] is None else f"{p['delta_db']:.2f}"] for r in results for p in r["pairs"]])
    dump("recordings.csv", ["recording", "minutes", "events", "events_per_min", "noise_db", "speech_db", "snr_db"],
         [[r["name"], f"{r['duration'] / 60:.2f}", len(r["events"]),
           f"{len(r['events']) / (r['duration'] / 60):.3f}" if r["duration"] else "",
           "" if r["noise_db"] is None else f"{r['noise_db']:.1f}",
           "" if r["speech_db"] is None else f"{r['speech_db']:.1f}",
           "" if r["snr_db"] is None else f"{r['snr_db']:.1f}"] for r in results])
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(summary))


def render_markdown(S: dict) -> str:
    def ci(v):
        return "n/a" if not v else f"[{v[0]}, {v[1]}]"

    d, dl = S["event_duration_s"], S["later_minus_earlier_speaker_db"]
    lines = [
        f"Recordings {S['recordings']} ({S['recordings_with_overlap']} with overlap), "
        f"{S['events_total']} events / {S['pairs_total']} pairs over {S['audio_minutes']} min; "
        f"pooled with at most {S['max_per_file']} per recording ({S['events_used']} events used).", "",
        f"- events per minute: {S['events_per_minute']['pooled']} pooled; overlap is "
        f"{S['overlap_share_of_audio']} of the audio",
        f"- event duration (s): median {d.get('p50')} (95% CI {ci(d.get('median_ci95'))}), "
        f"p10 {d.get('p10')}, p90 {d.get('p90')} (CI {ci(d.get('p90_ci95'))})",
    ]
    for p, v in S["pattern"].items():
        lines.append(f"- pattern {p}: {v['share']} (95% CI {ci(v['ci95'])})")
    lines += [
        f"- later minus earlier speaker (dB): median {dl.get('p50')} (CI {ci(dl.get('median_ci95'))}), "
        f"|diff| median {dl.get('abs_median')}, |diff| p90 CI {ci(dl.get('abs_p90_ci95'))}",
        f"- other events within 30 s of an event: median {S['other_events_within_30s'].get('p50')}, "
        f"p90 {S['other_events_within_30s'].get('p90')}",
        f"- noise floor (dBFS) per recording: median {S['noise_floor_db'].get('p50')}, "
        f"p10 {S['noise_floor_db'].get('p10')}, p90 {S['noise_floor_db'].get('p90')}",
        f"- speech-to-noise (dB) per recording: median {S['speech_to_noise_db'].get('p50')}, "
        f"p10 {S['speech_to_noise_db'].get('p10')}, p90 {S['speech_to_noise_db'].get('p90')}",
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", help="checkpoint cache folder (cache/default_<file>/...)")
    p.add_argument("--runs", nargs="+", default=[], help="run folders with stage outputs")
    p.add_argument("--out", required=True)
    p.add_argument("--files", type=int, help="measure only N recordings, picked at random (--seed)")
    p.add_argument("--max-per-file", type=int, default=50, help="cap per recording when pooling distributions")
    p.add_argument("--merge-gap", type=float, default=1.0)
    p.add_argument("--min-overlap", type=float, default=0.1)
    p.add_argument("--boot", type=int, default=300, help="bootstrap resamples for the intervals")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = p.parse_args()
    recs = cos.stage_recordings(a.runs) + (cos.cache_recordings(a.cache) if a.cache else [])
    print(f"{len(recs)} recording(s) found", flush=True)
    if a.files and a.files < len(recs):
        recs = random.Random(a.seed).sample(recs, a.files)
    results = measure_all(recs, a.merge_gap, a.min_overlap, a.workers)
    summary = summarize(results, a.max_per_file, a.seed, a.boot)
    write_outputs(results, summary, a.out)
    print(render_markdown(summary))
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
