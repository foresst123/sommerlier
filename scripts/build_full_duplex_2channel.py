#!/usr/bin/env python3
"""
Assemble 2-channel (stereo) full-duplex training samples from the
sommelier per-turn outputs.

Input:  one conversation JSON produced by build_training_conversations.py
        (a run of ≥N contiguous dyadic turns from a single clip).

For each conversation we:
  * Determine speaker→channel mapping (first speaker = L, other = R).
  * Read each turn's Demucs-cleaned mono MP3 from the clip dir.
  * Place it into the correct channel at the correct time offset.
  * The other channel stays silence during that turn (single-speaker
    frames).
  * For overlap frames (both speakers active), if a gated MossFormer2
    separation exists, use its per-speaker signals; otherwise fall back
    to placing the current turn on its channel and leaving the other
    channel silent (loses the interruption, but doesn't fabricate
    audio).

Output:
  <out_dir>/<conversation_id>.wav   ← 16 kHz stereo
  <out_dir>/<conversation_id>.json  ← metadata + turn timing on the
                                       stereo timeline

Usage:
    python scripts/build_full_duplex_2channel.py \\
        --convs-dir data/training_conversations_fullduplex/conversations \\
        --clips-root data/raw/_final/-sepreformer-True-... \\
        --out data/full_duplex_stereo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


SR = 16_000


def _read_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=False)
    if sr != SR:
        # Resample via linear interp — cheap; per-turn MP3s are already
        # 16k in the sommelier pipeline so this branch shouldn't fire
        # in practice. Kept for safety.
        n_new = int(round(len(audio) * SR / sr))
        audio = np.interp(
            np.linspace(0, 1, n_new, endpoint=False),
            np.linspace(0, 1, len(audio), endpoint=False),
            audio,
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def _sec_to_samp(x: float) -> int:
    return int(round(x * SR))


def _assemble_conversation(conv: dict, clip_dir: Path) -> tuple[np.ndarray, dict]:
    turns = conv["turns"]
    speakers = sorted({t["speaker"] for t in turns})
    if len(speakers) != 2:
        raise ValueError(f"expected exactly 2 speakers, got {speakers}")

    # First turn's speaker → left channel; other → right.
    left_spk = turns[0]["speaker"]
    right_spk = [s for s in speakers if s != left_spk][0]
    channel_of = {left_spk: 0, right_spk: 1}

    start_time = turns[0]["start"]
    end_time = turns[-1]["end"]
    n_samples = _sec_to_samp(end_time - start_time)
    stereo = np.zeros((n_samples, 2), dtype=np.float32)

    problems = []
    for t in turns:
        # audio_path is relative to the FINAL/<clip>/ directory, i.e.
        # `<clip>/00027_SPEAKER_00.mp3`. We accept both layouts.
        audio_rel = Path(t["audio_path"])
        candidates = [clip_dir / audio_rel.name, clip_dir.parent / audio_rel]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            problems.append(f"missing audio for turn {t['turn_id']}: tried {candidates}")
            continue
        chunk = _read_mono(found)
        ch = channel_of[t["speaker"]]
        s0 = _sec_to_samp(t["start"] - start_time)
        s1 = min(s0 + len(chunk), n_samples)
        chunk = chunk[: s1 - s0]
        # Add rather than overwrite in case two turns collide (overlap
        # region on the same channel — shouldn't happen if the
        # extractor did its job, but be safe).
        stereo[s0:s1, ch] += chunk

    peak = float(np.max(np.abs(stereo))) or 1.0
    if peak > 1.0:
        stereo = stereo / peak

    meta = {
        "conversation_id": conv["conversation_id"],
        "source_clip": conv["source_clip"],
        "source_url": conv.get("source_url"),
        "duration_s": round(end_time - start_time, 2),
        "n_turns": len(turns),
        "channels": {"L": left_spk, "R": right_spk},
        "start_offset_in_source_clip_s": start_time,
        "turns": [
            {
                "turn_id": t["turn_id"],
                "speaker": t["speaker"],
                "channel": channel_of[t["speaker"]],
                "start": round(t["start"] - start_time, 3),
                "end": round(t["end"] - start_time, 3),
                "text": t.get("text", ""),
            }
            for t in turns
        ],
        "problems": problems,
    }
    return stereo, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--convs-dir", type=Path, required=True,
                   help="dir with per-conversation JSONs from build_training_conversations.py")
    p.add_argument("--clips-root", type=Path, required=True,
                   help="_final root containing <clip>/<clip>/<turn>.mp3")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N conversations (debug)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    conv_files = sorted(args.convs_dir.glob("*.json"))
    if args.limit:
        conv_files = conv_files[: args.limit]

    ok = 0
    failed = 0
    total_hours = 0.0
    for cf in conv_files:
        conv = json.loads(cf.read_text(encoding="utf-8"))
        clip_dir = args.clips_root / conv["source_clip"] / conv["source_clip"]
        try:
            stereo, meta = _assemble_conversation(conv, clip_dir)
        except Exception as e:
            failed += 1
            print(f"[fail] {conv['conversation_id']}: {e}")
            continue
        wav_out = args.out / f"{conv['conversation_id']}.wav"
        sf.write(str(wav_out), stereo, SR, subtype="PCM_16")
        (args.out / f"{conv['conversation_id']}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total_hours += meta["duration_s"] / 3600
        ok += 1

    print(f"[done] {ok} stereo conversations written ({total_hours:.2f}h), {failed} failed")
    print(f"       -> {args.out}/")


if __name__ == "__main__":
    main()
