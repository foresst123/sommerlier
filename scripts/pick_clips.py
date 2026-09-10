#!/usr/bin/env python3
"""
Download YouTube audio clips for the sommelier pipeline.

Input: a TSV file with columns
  label, url, start, end, channel, genre
(comments starting with `#` allowed; blank lines ignored).

For each row:
  - Calls yt-dlp to extract the [start, end] window as 16kHz mono WAV.
  - Writes <out_dir>/<label>.wav + <out_dir>/<label>.meta.json with the
    source metadata so the normalizer can later pick it up.

Usage:
  python scripts/pick_clips.py --tsv data/clips_to_pull.tsv --out data/raw

Requires `yt-dlp` and `ffmpeg` on PATH (already on the laptop via brew).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 4.0  # 4, 8, 16, 32 seconds between retries (with jitter)


def hms_to_seconds(s: str) -> float:
    """Accept 'M:SS', 'H:MM:SS', or raw seconds."""
    s = s.strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(s)


def safe_label(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", s.strip())
    return re.sub(r"_+", "_", s).strip("_") or "clip"


def pull_one(
    label: str, url: str, start: float, end: float,
    channel: str, genre: str, out_dir: Path,
) -> Path | None:
    safe = safe_label(label)
    out_wav = out_dir / f"{safe}.wav"
    out_meta = out_dir / f"{safe}.meta.json"

    if out_wav.exists():
        print(f"  [skip] {safe}.wav already exists")
        return out_wav
    if out_meta.exists():
        print(f"  [skip] {safe} already pulled (meta.json present, WAV pruned)")
        return None

    # Whole-episode mode: end=0 (or end<=start) means "pull the entire video,
    # no --download-sections". Useful when we want long contiguous conversations
    # (e.g. for full-duplex training) rather than 4-min slices.
    whole_episode = end <= start
    duration = 0.0 if whole_episode else end - start

    cmd = [
        "yt-dlp",
        # -k keeps the pre-postprocessing file (original Opus/m4a at 48/44.1 kHz)
        # alongside the 16 kHz WAV that the pipeline reads.  Costs ~1 MB/min
        # extra disk and is our escape hatch if the video later goes 403 and
        # we want higher-fidelity audio for TTS/vocoder work.
        "-k",
        "-x", "--audio-format", "wav", "--audio-quality", "0",
        *([] if whole_episode else [
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
        ]),
        # Rate-limit awareness: signed URLs expire fast under heavy concurrent
        # load from a single IP.  These slow us down slightly but massively
        # reduce 403 attrition.
        "--sleep-requests", "1",
        "--retries", "5",
        "--fragment-retries", "5",
        # YouTube's obfuscated player needs a JS runtime; without one, formats
        # get dropped and pages fail with "No title found in player responses".
        # deno is auto-detected only if on PATH; explicit path is safer.
        *(["--js-runtimes", "deno:/home/tuanda67/.local/bin/deno"]
          if os.path.exists("/home/tuanda67/.local/bin/deno") else []),
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(out_dir / f"{safe}.%(ext)s"),
        url,
    ]
    print(f"  [pull] {safe}  ({duration:.1f}s)  {url}", flush=True)

    last_err = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            break
        except subprocess.CalledProcessError as e:
            last_err = (e.stderr or "")[-500:]
            transient = ("403" in last_err) or ("HTTP Error 5" in last_err) or ("timed out" in last_err)
            if not transient or attempt == MAX_ATTEMPTS:
                print(f"  [error] {safe}: yt-dlp failed after {attempt} attempts: {last_err}",
                      file=sys.stderr, flush=True)
                return None
            # Clean any partial file from the failed attempt before retrying.
            for stale in out_dir.glob(f"{safe}.*"):
                if stale.suffix in (".wav", ".part", ".temp", ".ytdl") and stale.name != out_meta.name:
                    stale.unlink(missing_ok=True)
            delay = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"  [retry] {safe}: attempt {attempt} failed (transient 403/5xx), "
                  f"sleeping {delay:.1f}s", flush=True)
            time.sleep(delay)

    if not out_wav.exists():
        print(f"  [error] expected {out_wav} not produced", file=sys.stderr, flush=True)
        return None

    # Discover the retained source-format file (opus / m4a / webm) that -k left behind.
    source_file = None
    for ext in ("opus", "webm", "m4a", "aac", "ogg", "mp3"):
        candidate = out_dir / f"{safe}.{ext}"
        if candidate.exists():
            source_file = candidate
            break

    meta = {
        "label": label,
        "safe_label": safe,
        "source_url": url,
        "source_channel": channel,
        "source_genre": genre,
        "youtube_start_s": start,
        "youtube_end_s": end,
        "clip_duration_s": duration,
        "sample_rate": 16000,
        "channels": 1,
        # Preserved for future higher-fidelity re-transcoding (TTS/vocoder work).
        "source_audio_file": source_file.name if source_file else None,
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_wav


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", type=Path, required=True, help="TSV file: label, url, start, end, channel, genre")
    p.add_argument("--out", type=Path, default=Path("data/raw"), help="output directory (default: data/raw)")
    p.add_argument("--workers", type=int, default=1, help="parallel yt-dlp workers (default: 1 serial)")
    args = p.parse_args()

    if not shutil.which("yt-dlp"):
        sys.exit("yt-dlp not found on PATH (try `brew install yt-dlp`)")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (try `brew install ffmpeg`)")

    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    with args.tsv.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for raw in reader:
            if not raw or raw[0].lstrip().startswith("#"):
                continue
            if len(raw) < 4:
                print(f"  [skip] short row: {raw}", file=sys.stderr)
                continue
            label, url, start, end = raw[0], raw[1], raw[2], raw[3]
            channel = raw[4] if len(raw) > 4 else ""
            genre = raw[5] if len(raw) > 5 else ""
            rows.append((label, url, hms_to_seconds(start), hms_to_seconds(end), channel, genre))

    if not rows:
        sys.exit(f"no clips found in {args.tsv}")

    print(f"Pulling {len(rows)} clips to {args.out}/  (workers={args.workers})")
    ok = 0
    if args.workers <= 1:
        for r in rows:
            if pull_one(*r, out_dir=args.out):
                ok += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(pull_one, *r, out_dir=args.out): r[0] for r in rows}
            for fut in as_completed(futs):
                if fut.result():
                    ok += 1
    print(f"\nDone: {ok}/{len(rows)} clips downloaded to {args.out}/")


if __name__ == "__main__":
    main()
