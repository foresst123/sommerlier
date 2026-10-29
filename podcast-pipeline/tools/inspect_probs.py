#!/usr/bin/env python3
"""What Nemotron's windowed slot probabilities look like to a DiariZen-style clustering. No torch, no GPU.

    python tools/inspect_probs.py PROBS_DIR [--median-s 0.22] [--edge-s 2.0]

PROBS_DIR is the folder an arm G/I/J stage 1 wrote (plan.json and piece_NNN.npz). For several slot thresholds it
prints what decides how many embeddings the clustering gets and how good they are:

  active slots/win  slots with any active frame in a window (the embeddings of one window);
  speech            share of frames where at least one slot is on;
  overlap           share of speech frames with two or more slots on; >=3 shows slots firing together beyond what
                    DiariZen's powerset allows (at most two);
  clean>=10%        share of active (window, slot) with at least 10% of the window's frames on alone, the rule that
                    lets an embedding into the clustering;
  uncertain         share of frames whose strongest slot lies between 0.3 and 0.7, so the threshold decides them;
  edge/mid speech   speech share in the first and last --edge-s seconds of a window against the rest: a model that is
                    less sure at the edges of a window shows a gap there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import utils.dz_pipeline as dz  # noqa: E402
import utils.windowed_clustering as wc  # noqa: E402

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def load_windows(probs_dir: str):
    """[(window span, probabilities (frames, slots))] over every piece of the plan, and frame_s."""
    with open(os.path.join(probs_dir, "plan.json"), encoding="utf-8") as fh:
        plan = json.load(fh)
    wins = []
    for entry in plan["pieces"]:
        _, spans, probs, _ = wc.load_piece_probs(os.path.join(probs_dir, entry["file"]))
        wins.extend(zip(spans, probs))
    return wins, float(plan["frame_s"])


def summarise(wins, frame_s: float, thr: float, median_s: float = 0.22, edge_s: float = 2.0) -> dict:
    active_slots, clean_ok, active_total = [], 0, 0
    speech = overlap = three = frames = uncertain = 0
    edge_speech = edge_frames = mid_speech = mid_frames = 0
    edge_n = max(1, int(round(edge_s / frame_s)))
    median_frames = dz.frames_for_seconds(median_s, frame_s)
    for _, probs in wins:
        probs = np.asarray(probs, dtype=float)
        binary = dz.binarise_window(probs, thr, median_frames)
        n_on = binary.sum(axis=1)
        frames += len(binary)
        speech += int((n_on > 0).sum())
        overlap += int((n_on >= 2).sum())
        three += int((n_on >= 3).sum())
        top = probs.max(axis=1) if probs.size else np.zeros(len(binary))
        uncertain += int(((top > 0.3) & (top < 0.7)).sum())
        on = n_on > 0
        edge = np.zeros(len(on), dtype=bool)
        edge[:edge_n] = True
        edge[-edge_n:] = True
        edge_speech += int(on[edge].sum())
        edge_frames += int(edge.sum())
        mid_speech += int(on[~edge].sum())
        mid_frames += int((~edge).sum())
        min_frames = round(0.1 * len(binary))
        alone = n_on == 1
        n_act = 0
        for j in range(binary.shape[1]):
            if binary[:, j].sum() > 0:
                n_act += 1
                active_total += 1
                clean_ok += int((binary[:, j] * alone).sum() >= min_frames)
        active_slots.append(n_act)

    def share(a, b):
        return a / b if b else float("nan")

    return {"thr": thr, "windows": len(wins), "active_slots_per_window": float(np.mean(active_slots)) if wins else 0.0,
            "speech": share(speech, frames), "overlap": share(overlap, speech), "overlap_3plus": share(three, speech),
            "clean_10pct": share(clean_ok, active_total), "uncertain": share(uncertain, frames),
            "edge_speech": share(edge_speech, edge_frames), "mid_speech": share(mid_speech, mid_frames)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probs_dir")
    ap.add_argument("--median-s", type=float, default=0.22)
    ap.add_argument("--edge-s", type=float, default=2.0)
    a = ap.parse_args(argv)
    wins, frame_s = load_windows(a.probs_dir)
    print(f"{len(wins)} windows, frame_s {frame_s}, slots {wins[0][1].shape[1] if wins else 0}")
    head = ("thr", "slots/win", "speech", "overlap", ">=3 on", "clean>=10%", "uncertain", "edge speech", "mid speech")
    print("  ".join(f"{h:>11}" for h in head))
    for thr in THRESHOLDS:
        r = summarise(wins, frame_s, thr, a.median_s, a.edge_s)
        row = (r["thr"], r["active_slots_per_window"], r["speech"], r["overlap"], r["overlap_3plus"], r["clean_10pct"],
               r["uncertain"], r["edge_speech"], r["mid_speech"])
        print("  ".join(f"{v:>11.3f}" for v in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
