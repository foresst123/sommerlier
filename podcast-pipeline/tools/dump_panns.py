#!/usr/bin/env python3
"""Write out everything PANNs saw: all 527 scores, every frame, one recording.

The pipeline reads 6 of the 527 AudioSet groups and routes on 4 thresholds.
Every one of those thresholds was chosen without seeing the distribution it
sits in. This writes the raw numbers so the choosing can stop being a guess.

What comes out is the matrix the pipeline itself routed on -- the same
resample, the same peak normalisation, the same chunking -- because it calls
`PANNSDetector.framewise_raw`, the one preprocessing path. An analysis of this
file is an analysis of the real decision input, not of a second signal that
resembles it.

    python tools/dump_panns.py --audio path/to/podcast.mp3

Writes `<name>_panns.npz` plus `<name>_panns.json` (metadata, readable without
numpy). Add `--csv` for a spreadsheet of the strongest labels per block.

## Size, and why the default loses nothing

Cnn14_DecisionLevelMax pools time by 2 five times over: it makes one decision
per 32 output frames and repeats it across all 32. The advertised 100 fps is a
presentation of a 3.125 fps signal.

So the default stores one row per decision block, which is 32x smaller and
**discards nothing**. A 50-minute recording is ~10MB instead of ~630MB. The
repetition is verified rather than assumed -- `--full-frames` keeps every frame
if you want to check, and the metadata records whether the blocks really were
constant.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def block_reduce(framewise, block):
    """One row per decision block, and how far from constant each block was.

    Returns (reduced, max_within_block_spread). A spread of 0 means the model
    really did repeat one decision across the block and nothing was lost.
    """
    if not len(framewise):
        return framewise, 0.0
    n = len(framewise) // block * block
    head, tail = framewise[:n], framewise[n:]

    spread = 0.0
    if n:
        cube = head.reshape(-1, block, framewise.shape[1])
        spread = float(np.max(cube.max(axis=1) - cube.min(axis=1)))
        reduced = cube[:, 0, :]
    else:
        reduced = framewise[:0]

    # The final partial block is its own row rather than dropped: it is real
    # audio at the end of the recording.
    if len(tail):
        reduced = np.concatenate([reduced, tail[:1]], axis=0)
    return reduced, spread


def derive_flags(scores, config):
    """The routing decision, frame by frame, exactly as music_map makes it.

    Carried in the dump so the analysis can ask "which frames became SONG, and
    what did the other 526 labels say there?" without reimplementing the rule
    and getting it subtly different.
    """
    speech, singing, music = scores["speech"], scores["singing"], scores["music"]
    is_singing = ((singing >= config["singing_threshold"])
                  & (singing >= speech + config["singing_margin"]))
    loud_music = (music >= config["music_threshold"]) & ~is_singing
    is_song = loud_music & (speech < config["speech_present"])
    is_music = loud_music & ~is_song
    return {"is_singing": is_singing, "is_song": is_song, "is_music": is_music}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dump every PANNs score for every frame of one recording.")
    parser.add_argument("--audio", required=True, help="Audio file to analyse.")
    parser.add_argument("--out", default=None,
                        help="Output .npz (default: <audio stem>_panns.npz).")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Rate the pipeline loads at; keep 16000 to match it.")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    parser.add_argument("--full-frames", action="store_true",
                        help="Keep all 100fps rows instead of one per 320ms "
                             "decision block. 32x larger, same information.")
    parser.add_argument("--float32", action="store_true",
                        help="Store float32 instead of float16. Probabilities "
                             "live in [0,1], where float16 already resolves "
                             "~0.0005 -- far finer than any threshold here.")
    parser.add_argument("--csv", action="store_true",
                        help="Also write a per-block CSV of the strongest labels.")
    parser.add_argument("--csv-top", type=int, default=8,
                        help="How many labels per block in the CSV (default 8).")
    parser.add_argument("--limit-seconds", type=float, default=None,
                        help="Analyse only the first N seconds. For a quick look.")
    args = parser.parse_args(argv)

    from models.panns import PANNSDetector
    from services.audio_service import AudioService
    from utils import music_map as mm

    stem = os.path.splitext(os.path.basename(args.audio))[0]
    out_path = args.out or f"{stem}_panns.npz"
    json_path = os.path.splitext(out_path)[0] + ".json"

    # Loaded exactly as the pipeline loads it, gain and all: analysing a
    # differently-decoded copy would describe a signal PANNs never saw.
    print(f"[1/4] Loading {args.audio} at {args.sr} Hz")
    audio = AudioService().load_audio(args.audio, target_sr=args.sr)
    waveform = audio.waveform
    if args.limit_seconds:
        waveform = waveform[:int(args.limit_seconds * args.sr)]
    duration = len(waveform) / float(args.sr)
    print(f"      {duration:.1f}s ({duration / 60:.1f} min)")

    print(f"[2/4] Running PANNs SoundEventDetection")
    detector = PANNSDetector(device=args.device)
    framewise, fps, scale = detector.framewise_raw(waveform, args.sr)
    if not len(framewise):
        print("      Nothing to analyse (under one second of audio).")
        return 1
    print(f"      {len(framewise)} frames x {framewise.shape[1]} labels at {fps:.0f} fps")
    print(f"      waveform was scaled by {scale:.4f} to reach peak 0.9")

    scores = detector.group_scores(framewise)
    config = {"music_threshold": mm.MUSIC_THRESHOLD,
              "singing_threshold": mm.SINGING_THRESHOLD,
              "singing_margin": mm.SINGING_MARGIN,
              "speech_present": mm.SPEECH_PRESENT,
              "min_span_seconds": mm.MIN_SPAN_SECONDS,
              "merge_gap_seconds": mm.MERGE_GAP_SECONDS,
              "pad_seconds": mm.PAD_SECONDS}
    flags = derive_flags(scores, config)

    block = detector.SED_DECISION_FRAMES
    if args.full_frames:
        matrix, spread, row_seconds = framewise, None, 1.0 / fps
        print(f"[3/4] Keeping all {len(matrix)} frames")
    else:
        matrix, spread = block_reduce(framewise, block)
        row_seconds = block / fps
        print(f"[3/4] Reduced to {len(matrix)} decision blocks of {row_seconds * 1000:.0f}ms")
        print(f"      largest spread within a block: {spread:.6f} "
              + ("(constant -- nothing lost)" if spread < 1e-6
                 else "(NOT constant: re-run with --full-frames)"))

    dtype = np.float32 if args.float32 else np.float16
    payload = {
        "scores": matrix.astype(dtype),
        "labels": np.array(detector.labels, dtype=object),
        "fps": np.float32(fps),
        "row_seconds": np.float32(row_seconds),
        "block_frames": np.int32(block),
        "scale_applied": np.float32(scale),
        "duration_seconds": np.float32(duration),
        "sample_rate": np.int32(args.sr),
    }
    # The six curves and the three decisions, at full frame rate, so the
    # analysis can line raw labels up against what the pipeline concluded.
    for key, curve in scores.items():
        payload[f"group_{key}"] = curve.astype(np.float32)
    for key, flag in flags.items():
        payload[f"flag_{key}"] = flag

    print(f"[4/4] Writing {out_path}")
    np.savez_compressed(out_path, **payload)
    size_mb = os.path.getsize(out_path) / 1024 ** 2

    meta = {
        "audio": os.path.abspath(args.audio),
        "duration_seconds": round(duration, 2),
        "sample_rate": args.sr,
        "frames": int(len(framewise)),
        "rows_stored": int(len(matrix)),
        "row_seconds": round(float(row_seconds), 5),
        "fps_advertised": fps,
        "block_frames": int(block),
        "block_spread_max": None if spread is None else round(float(spread), 8),
        "lossless_block_reduction": None if spread is None else bool(spread < 1e-6),
        "labels": int(framewise.shape[1]),
        "dtype": str(np.dtype(dtype)),
        "scale_applied": round(float(scale), 6),
        "file_mb": round(size_mb, 2),
        "thresholds": config,
        "group_seconds": {k: round(float(v.sum()) / fps, 2)
                          for k, v in {kk: (vv >= 0.5) for kk, vv in scores.items()}.items()},
        "flag_seconds": {k: round(float(v.sum()) / fps, 2) for k, v in flags.items()},
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    if args.csv:
        csv_path = os.path.splitext(out_path)[0] + "_top.csv"
        _write_top_csv(csv_path, matrix, detector.labels, row_seconds, args.csv_top)
        print(f"      {csv_path}")

    print(f"\n{out_path}  {size_mb:.1f} MB")
    print(f"{json_path}")
    print(f"\nseconds over 0.5 per group: {meta['group_seconds']}")
    print(f"seconds per routing decision: {meta['flag_seconds']}")
    return 0


def _write_top_csv(path, matrix, labels, row_seconds, top):
    """A readable slice: the strongest labels in each block, with their scores.

    The npz is for numpy; this is for looking at a suspicious minute by eye
    without writing code.
    """
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["start_s", "end_s"]
                        + [c for i in range(top) for c in (f"label_{i+1}", f"score_{i+1}")])
        for row_index, row in enumerate(matrix):
            order = np.argsort(row)[::-1][:top]
            cells = []
            for column in order:
                cells += [labels[column], round(float(row[column]), 4)]
            writer.writerow([round(row_index * row_seconds, 3),
                             round((row_index + 1) * row_seconds, 3)] + cells)


if __name__ == "__main__":
    raise SystemExit(main())
