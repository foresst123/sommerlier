#!/usr/bin/env python3
"""Compare the arms of tools/diar_arms_experiment.py with one of them taken as the reference (DiariZen by default).

    python tools/compare_arms.py OUT/diar_arms [--ref diarizen_auto] [--arms nemotron,nemotron_dz] [--collar 0.25]
                                              [--csv table.csv]

Reads OUT/diar_arms/*/segments_<arm>.json (already cropped to the clip). The reference is not the truth unless it is --ref labels (hand labels, with their own timing tolerance and their
"not sure" regions left out): the table
says how far each arm is from it, so a gap means "they disagree, listen", not "wrong". Per sample and arm:

  spk >=1s      speakers with at least one second of speech in the clip (ref in the first row of a sample)
  ref s / arm s speech seconds (any speaker) of the reference and of the arm
  miss / FA     speech the reference has and the arm lacks / the arm has and the reference lacks (% of reference speech)
  conf          reference speech given to another speaker after the best one-to-one speaker mapping (%)
  DER           miss + FA + conf (NIST frame count, 10 ms frames, no scoring within --collar seconds of the
                reference's segment edges)
  short <1s     reference segments under one second (backchannels): share the arm covers with speech (any speaker),
                and share it covers with the speaker it maps to
  speaker recall  per reference speaker with >=1 s: share of that speaker's time the mapped arm speaker covers

Arms A, B and H were post-processed by the pipeline (merge of nearby turns and so on); the windowed arms are raw.
--postprocess applies the pipeline's own merge_gap / bridge_gap steps to the raw ones, so the table compares like with like.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

STEP = 0.01
# Arms whose segments.json is the raw output of tools/windowed_*: arms A, B and H went through the pipeline's own
# post-processing (main.py), these did not. --postprocess gives them the same one.
RAW_ARMS = ("nemotron_windowed", "nemotron_windowed_vbx", "nemotron_dense_vbx", "nemotron_dz")
MERGE_GAP, BRIDGE_GAP, MAX_SEGMENT = 0.5, 3.0, 20.0     # config.json, a100: merge_gap, bridge_gap, max_segment_length
SHORT_S = 1.0
MIN_SPEAKER_S = 1.0


def load_segments(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data.get("segments", data) if isinstance(data, dict) else data
    out = []
    for r in rows or []:
        try:
            s, e = float(r["start"]), float(r["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s:
            out.append({"start": s, "end": e, "speaker": str(r.get("speaker", "?"))})
    return out


def to_frames(segments: list[dict], n_frames: int, step: float = STEP):
    """(frames, speakers) bool matrix and the speaker labels, in order of first appearance."""
    labels: list[str] = []
    for s in sorted(segments, key=lambda r: r["start"]):
        if s["speaker"] not in labels:
            labels.append(s["speaker"])
    mat = np.zeros((n_frames, len(labels)), dtype=bool)
    for s in segments:
        a = max(0, int(math.floor(s["start"] / step + 1e-9)))
        b = min(n_frames, int(math.ceil(s["end"] / step - 1e-9)))
        if b > a:
            mat[a:b, labels.index(s["speaker"])] = True
    return mat, labels


def collar_mask(segments: list[dict], n_frames: int, collar: float, step: float = STEP) -> np.ndarray:
    """True where a frame is scored: not within `collar` seconds of any segment edge of the reference."""
    keep = np.ones(n_frames, dtype=bool)
    if collar <= 0:
        return keep
    for s in segments:
        for t in (s["start"], s["end"]):
            a = max(0, int(math.floor((t - collar) / step)))
            b = min(n_frames, int(math.ceil((t + collar) / step)))
            keep[a:b] = False
    return keep


def best_mapping(ref: np.ndarray, hyp: np.ndarray, scored: np.ndarray) -> dict[int, int]:
    """hyp speaker index -> ref speaker index, one to one, maximising the overlapping scored frames."""
    if ref.shape[1] == 0 or hyp.shape[1] == 0:
        return {}
    overlap = (ref[scored].T.astype(np.int64)) @ hyp[scored].astype(np.int64)      # (ref, hyp)
    rows, cols = linear_sum_assignment(overlap, maximize=True)
    return {int(c): int(r) for r, c in zip(rows, cols) if overlap[r, c] > 0}


def compare(ref_segs: list[dict], hyp_segs: list[dict], duration: float, collar: float = 0.25,
            step: float = STEP, ignore: Optional[list[tuple[float, float]]] = None) -> dict:
    n = int(math.ceil(duration / step))
    ref, ref_labels = to_frames(ref_segs, n, step)
    hyp, hyp_labels = to_frames(hyp_segs, n, step)
    scored = collar_mask(ref_segs, n, collar, step)
    for a, b in ignore or ():                                  # regions the labeller was not sure about: not scored
        scored[max(0, int(math.floor(a / step))):min(n, int(math.ceil(b / step)))] = False
    mapping = best_mapping(ref, hyp, scored)

    n_ref = ref.sum(axis=1)
    n_hyp = hyp.sum(axis=1)
    correct = np.zeros(n, dtype=np.int64)
    for h, r in mapping.items():
        correct += (ref[:, r] & hyp[:, h]).astype(np.int64)
    miss = np.maximum(0, n_ref - n_hyp)
    fa = np.maximum(0, n_hyp - n_ref)
    conf = np.minimum(n_ref, n_hyp) - correct
    total = float(n_ref[scored].sum())

    def pct(v):
        return 100.0 * float(v[scored].sum()) / total if total > 0 else float("nan")

    ref_any, hyp_any = n_ref > 0, n_hyp > 0
    inv = {r: h for h, r in mapping.items()}

    # reference segments shorter than a second: does the arm speak there, and with the mapped speaker?
    short = [s for s in ref_segs if s["end"] - s["start"] < SHORT_S]
    seen_any = seen_spk = 0
    for s in short:
        a = max(0, int(math.floor(s["start"] / step)))
        b = min(n, max(a + 1, int(math.ceil(s["end"] / step))))
        seen_any += int(hyp_any[a:b].mean() >= 0.5)
        h = inv.get(ref_labels.index(s["speaker"]))
        seen_spk += int(h is not None and hyp[a:b, h].mean() >= 0.5)

    recall = {}
    for i, name in enumerate(ref_labels):
        t = float(ref[scored, i].sum())
        if t * step >= MIN_SPEAKER_S:
            h = inv.get(i)
            recall[name] = round(float((ref[scored, i] & hyp[scored, h]).sum()) / t, 3) if h is not None else 0.0

    def speakers(mat):
        return int((mat.sum(axis=0) * step >= MIN_SPEAKER_S).sum())

    return {"ref_speakers": speakers(ref), "arm_speakers": speakers(hyp),
            "ref_s": round(float(ref_any.sum()) * step, 1), "arm_s": round(float(hyp_any.sum()) * step, 1),
            "miss": round(pct(miss), 1), "fa": round(pct(fa), 1), "conf": round(pct(conf), 1),
            "der": round(pct(miss + fa + conf), 1), "short_n": len(short),
            "short_any": round(100.0 * seen_any / len(short), 1) if short else None,
            "short_spk": round(100.0 * seen_spk / len(short), 1) if short else None,
            "speaker_recall": recall}


def pipeline_postprocess(segments: list[dict]) -> list[dict]:
    """The steps of DiarizationService.diarize_postprocess that need no audio: drop sub-0.2 s jitter, merge nearby turns
    of one speaker (merge_gap), bridge an interrupted turn (bridge_gap). VAD, seam split and the length split are left out.
    """
    from utils import segment_utils as su
    rows = [dict(index=f"{i:05d}", **s) for i, s in enumerate(sorted(segments, key=lambda r: (r["start"], r["end"])))]
    rows = su.filter_diarizer_noise(rows)
    rows = su.cut_by_speaker_label(rows, merge_gap=MERGE_GAP, max_segment_length=MAX_SEGMENT)
    rows = su.bridge_interrupted_speaker_turns(rows, bridge_gap=BRIDGE_GAP)
    return [{"start": float(r["start"]), "end": float(r["end"]), "speaker": str(r["speaker"])} for r in rows]


def clip_duration(sample_dir: str, all_segs: list[list[dict]]) -> float:
    clip = os.path.join(sample_dir, "clip.wav")
    try:
        import soundfile as sf
        info = sf.info(clip)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return max((s["end"] for segs in all_segs for s in segs), default=0.0)


def arm_of(path: str) -> str:
    return os.path.basename(path)[len("segments_"):-len(".json")]


def load_labels(sample_dir: str) -> Optional[dict]:
    path = os.path.join(sample_dir, "labels.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc if doc.get("segments") else None


def run(root: str, ref_arm: str, arms: Optional[list[str]], collar: Optional[float],
        postprocess: bool = False) -> list[dict]:
    """ref_arm "labels" takes the hand labels of tools/label_page.html as the reference: their own timing tolerance is
    the collar unless one is given, and their "not sure" regions are left out of the scoring."""
    rows = []
    for sdir in sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)):
        files = {arm_of(f): f for f in glob.glob(os.path.join(sdir, "segments_*.json"))}
        ignore, tol = None, collar
        if ref_arm == "labels":
            doc = load_labels(sdir)
            if doc is None or not files:
                continue
            ref_segs = [{"start": s["start"], "end": s["end"], "speaker": str(s["speaker"])} for s in doc["segments"]]
            ignore = [(i["start"], i["end"]) for i in doc.get("ignore", [])]
            tol = float(doc.get("tolerance_s", 0.25)) if collar is None else collar
            segs = {a: load_segments(f) for a, f in files.items()}
            candidates = sorted(segs)
        else:
            if ref_arm not in files:
                continue
            segs = {a: load_segments(f) for a, f in files.items()}
            ref_segs = segs[ref_arm]
            tol = 0.25 if collar is None else collar
            candidates = sorted(a for a in segs if a != ref_arm)
        if postprocess:
            segs = {a: pipeline_postprocess(v) if a in RAW_ARMS else v for a, v in segs.items()}
        duration = clip_duration(sdir, [ref_segs] + list(segs.values()))
        for arm in candidates:
            if arms and arm not in arms:
                continue
            r = compare(ref_segs, segs[arm], duration, tol, ignore=ignore)
            rows.append(dict(sample=os.path.basename(sdir), arm=arm, **r))
    return rows


def print_table(rows: list[dict], ref_arm: str) -> None:
    if ref_arm == "labels":
        print("reference: your hand labels (edge tolerance and 'not sure' regions as saved with them)")
    else:
        print(f"reference: {ref_arm}  (a gap means the two disagree, not that the arm is wrong)")
    head = ("arm", "spk>=1s", "ref s", "arm s", "miss%", "FA%", "conf%", "DER%", "short any%", "short spk%")
    last = None
    for r in rows:
        if r["sample"] != last:
            last = r["sample"]
            print(f"\n== {last}")
            print("  ".join(f"{h:>20}" if i == 0 else f"{h:>10}" for i, h in enumerate(head)))
        spk = f"{r['arm_speakers']} (ref {r['ref_speakers']})"
        short = (f"{r['short_any']}/{r['short_n']}" if r["short_n"] else "-", f"{r['short_spk']}" if r["short_n"] else "-")
        cells = (r["arm"], spk, r["ref_s"], r["arm_s"], r["miss"], r["fa"], r["conf"], r["der"], short[0], short[1])
        print("  ".join(f"{str(c):>20}" if i == 0 else f"{str(c):>10}" for i, c in enumerate(cells)))
        if r["speaker_recall"]:
            print(f"{'':>20}  speaker recall: " + ", ".join(f"{k}={v}" for k, v in r["speaker_recall"].items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the diar_arms output folder (one sub-folder per sample)")
    ap.add_argument("--ref", default="diarizen_auto",
                    help="the arm taken as the reference, or 'labels' for the hand labels saved by tools/label_server.py")
    ap.add_argument("--arms", help="comma list of arms to compare (default: every other arm found)")
    ap.add_argument("--collar", type=float, default=None,
                    help="seconds left out around every reference edge (default 0.25; with --ref labels, the tolerance of each sample's labels)")
    ap.add_argument("--postprocess", action="store_true",
                    help="give the raw windowed arms (nemotron_windowed*, nemotron_dense_vbx, nemotron_dz) the pipeline's "
                         "own merge and bridge steps, as arms A and B already went through them")
    ap.add_argument("--csv", help="also write the rows to this file")
    a = ap.parse_args(argv)
    arms = [x.strip() for x in a.arms.split(",")] if a.arms else None
    rows = run(a.root, a.ref, arms, a.collar, a.postprocess)
    if not rows:
        what = "labels.json" if a.ref == "labels" else f"segments_{a.ref}.json"
        print(f"no sample with {what} under {a.root}", file=sys.stderr)
        return 1
    print_table(rows, a.ref)
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            keys = [k for k in rows[0] if k != "speaker_recall"]
            w = csv.DictWriter(fh, fieldnames=keys + ["speaker_recall"])
            w.writeheader()
            for r in rows:
                w.writerow(dict({k: r[k] for k in keys}, speaker_recall=json.dumps(r["speaker_recall"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
