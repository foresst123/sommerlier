"""How much clean single-speaker material there is, and how often the layout rule can use it.

Reads only the segments of every recording (no audio), so it is cheap on the full cache:

  per recording and speaker   clean seconds and clip count at each --min-clip (nobody else
                              within --margin, no music); the speaker with less clean time
                              in a recording is the "minor" one (a proxy for the interrupter)
  unique clean hours          sum of the clean clips over all speakers (clips are disjoint)
  layout feasibility          with --stats (pairs.csv): every usable template of a recording
                              is planned with render_mix_dataset.plan_chunk for each
                              --window, without audio, and the outcome is counted: nominal,
                              kept through a fallback (which one), or dropped (why).
                              This estimates drop and fallback rates before any rendering.

Writes speakers.csv, feasibility.csv and summary.json under --out.

    python tools/audit_clean_pool.py --cache cache --stats overlap_stats --out clean_audit \\
        --window 3 4 6 --min-clip 0.25 0.3
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_overlap_samples as cos  # noqa: E402
import render_mix_dataset as rmd  # noqa: E402
from utils import separation_layout as sl  # noqa: E402


def speaker_rows(name: str, segs: List[dict], margin: float, min_clips: List[float]) -> List[dict]:
    rows = []
    for spk in sorted({s["speaker"] for s in segs}):
        r = {"recording": name, "speaker": spk}
        for mc in min_clips:
            clips = cos.clean_clips(segs, spk, margin, mc)
            r[f"clean_s@{mc}"] = round(sum(b - a for a, b in clips), 3)
            r[f"clips@{mc}"] = len(clips)
            r[f"clips_ge_2s@{mc}"] = sum(b - a >= 2.0 for a, b in clips)
        rows.append(r)
    if len(rows) >= 2:
        key = f"clean_s@{min_clips[0]}"
        minor = min(rows, key=lambda r: r[key])
        for r in rows:
            r["minor"] = r is minor
    return rows


def feasibility(name: str, segs: List[dict], templates: List[dict], windows: List[float], margin: float,
                min_clip: float, l_min: float, seed: int, sr: int = 16000) -> Dict[float, Counter]:
    pools = {}
    for s in sorted({x["speaker"] for x in segs}):
        c = cos.clean_clips(segs, s, margin, min_clip)
        if c:
            pools[s] = c
    out = {}
    for w in windows:
        cfg = sl.LayoutConfig(sr=sr, l_min=l_min)
        n = sl.window_samples(w, sr)
        c = Counter()
        rng = random.Random(seed)
        for t in templates:
            c["templates"] += 1
            if not l_min - 1e-9 <= t["overlap"] <= w + 1e-9:
                c["drop:out_of_range"] += 1
                continue
            lay, why = rmd.plan_chunk(rng, t, pools, n, cfg, sr)
            if lay is None:
                c[f"drop:{why}"] += 1
            elif lay["fallback"]:
                c["fallback"] += 1
                for x in lay["fallback"].split("+"):
                    c[f"fallback:{x}"] += 1
            else:
                c["nominal"] += 1
        out[w] = c
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache")
    p.add_argument("--runs", nargs="+", default=[])
    p.add_argument("--stats", help="folder with pairs.csv; enables the layout feasibility count")
    p.add_argument("--out", required=True)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--min-clip", type=float, nargs="+", default=[0.25, 0.3])
    p.add_argument("--window", type=float, nargs="+", default=[4.0])
    p.add_argument("--l-min", type=float, default=0.25)
    p.add_argument("--files", type=int)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    recs = cos.stage_recordings(a.runs) + (cos.cache_recordings(a.cache) if a.cache else [])
    if a.files and a.files < len(recs):
        recs = random.Random(a.seed).sample(recs, a.files)
    templates = rmd.load_templates(a.stats) if a.stats else {}
    summary = audit(recs, templates, a.out, a.margin, a.min_clip, a.window, a.l_min, a.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def audit(recs, templates, out, margin=0.3, min_clips=(0.25, 0.3), windows=(4.0,), l_min=0.25, seed=0) -> dict:
    min_clips, windows = list(min_clips), list(windows)
    os.makedirs(out, exist_ok=True)
    spk_rows, feas = [], {w: Counter() for w in windows}
    for k, rec in enumerate(recs):
        name = os.path.basename(rec["path"].rstrip("/"))
        segs = rec["segments"]()
        spk_rows += speaker_rows(name, segs, margin, min_clips)
        if templates.get(name):
            for w, c in feasibility(name, segs, templates[name], windows, margin, min_clips[0], l_min,
                                    seed + k).items():
                feas[w].update(c)
    keys = sorted({k for r in spk_rows for k in r})
    with open(os.path.join(out, "speakers.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(spk_rows)
    reasons = sorted({k for c in feas.values() for k in c})
    with open(os.path.join(out, "feasibility.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window"] + reasons)
        for win, c in feas.items():
            w.writerow([win] + [c.get(r, 0) for r in reasons])
    summary = {"recordings": len(recs), "speakers": len(spk_rows), "margin": margin,
               "unique_clean_hours": {str(mc): round(sum(r[f"clean_s@{mc}"] for r in spk_rows) / 3600, 6)
                                      for mc in min_clips},
               "minor_speaker_clean_hours": {str(mc): round(sum(r[f"clean_s@{mc}"] for r in spk_rows
                                                                if r.get("minor")) / 3600, 6)
                                             for mc in min_clips},
               "feasibility": {str(w): dict(c) for w, c in feas.items()}}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    main()
