#!/usr/bin/env python3
"""
Extract contiguous multi-turn conversations from per-clip JSONL for
downstream PT / SFT / RL training.

Input:  the _final root (or an HF-mirror dir with the same layout).
        For each clip dir with a <clip>.jsonl, we split the turn list
        into runs that are BOTH:
          - index-contiguous  (turn_index[i+1] == turn_index[i] + 1)
          - time-contiguous   (turns[i+1].start - turns[i].end <= max_gap_s)
        The two conditions catch different failure modes:
          * index jump = the normalizer dropped a turn for cause
                         (BGM, misattribution, phantom, multi-speaker)
          * time gap   = long silence Whisper/VAD missed a segment

Output: one JSON per conversation, with the fields a trainer actually reads
        (text, speaker, audio_path, timing, and optional word timing / flags).

Also emits a summary TSV + a small preview HTML you can eyeball.

Usage:
    python scripts/build_training_conversations.py \\
        --root data/raw/_final/-sepreformer-True-.../   \\
        --out  data/training_conversations \\
        --min-turns 6 --max-gap-s 2.5
"""
from __future__ import annotations

import argparse
import json
import html
from collections import Counter
from pathlib import Path


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def _split_runs(turns: list[dict], max_gap_s: float) -> list[list[dict]]:
    """Split into contiguous runs. Turns already sorted by start."""
    if not turns:
        return []
    runs: list[list[dict]] = [[turns[0]]]
    for prev, cur in zip(turns, turns[1:]):
        idx_ok = cur["turn_index"] - prev["turn_index"] == 1
        time_ok = cur["start"] - prev["end"] <= max_gap_s
        if idx_ok and time_ok:
            runs[-1].append(cur)
        else:
            runs.append([cur])
    return runs


def _turn_to_row(t: dict) -> dict:
    """Slim the turn dict to what a trainer actually consumes."""
    row = {
        "turn_id": t["turn_id"],
        "turn_index": t["turn_index"],
        "speaker": t["speaker"],
        "start": t["start"],
        "end": t["end"],
        "duration": t["duration"],
        "audio_path": t["audio_path"],
        # Prefer gold, then rover, then any single-model text
        "text": (t.get("gold") or {}).get("text")
                or (t.get("asr") or {}).get("rover")
                or t.get("text_written")
                or (t.get("asr") or {}).get("whisper")
                or "",
        "text_source": "gold" if (t.get("gold") or {}).get("text")
                       else "rover" if (t.get("asr") or {}).get("rover")
                       else "whisper",
        "flags": t.get("flags", {}),
    }
    if t.get("words"):
        row["words"] = t["words"]
    if t.get("contested_words"):
        row["contested_words"] = t["contested_words"]
    return row


def _build_conversation(clip: str, run: list[dict], run_idx: int,
                        source_url: str, source_channel: str,
                        source_genre: str) -> dict:
    return {
        "conversation_id": f"{clip}__run{run_idx:02d}",
        "source_clip": clip,
        "source_url": source_url,
        "source_channel": source_channel,
        "source_genre": source_genre,
        "n_turns": len(run),
        "duration_s": round(run[-1]["end"] - run[0]["start"], 2),
        "unique_speakers": sorted({t["speaker"] for t in run}),
        "start_turn_index": run[0]["turn_index"],
        "end_turn_index": run[-1]["turn_index"],
        "turns": [_turn_to_row(t) for t in run],
    }


HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>Training conversations preview</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#222}
h2{border-bottom:1px solid #ddd;padding-bottom:.3rem}
.conv{border:1px solid #ddd;border-radius:6px;padding:.8rem;margin:1rem 0;background:#fafafa}
.meta{color:#666;font-size:.85rem;margin-bottom:.5rem}
.turn{display:grid;grid-template-columns:80px 1fr;gap:.5rem;padding:.3rem 0;border-top:1px dashed #eee}
.turn:first-child{border:0}
.sp{font-weight:600;font-family:ui-monospace,monospace;font-size:.8rem}
.sp0{color:#c62828} .sp1{color:#1565c0} .sp2{color:#2e7d32} .sp3{color:#6a1b9a}
.text{white-space:pre-wrap}
audio{width:100%;height:24px;margin-top:.3rem}
</style></head><body>
<h1>Training conversations preview</h1>
"""


def _write_html(convs: list[dict], out: Path, clips_root: Path,
                per_conv_max: int, embed_audio: bool) -> None:
    with out.open("w") as f:
        f.write(HTML_HEAD)
        for c in convs[:per_conv_max]:
            f.write(f'<div class="conv"><h2>{html.escape(c["conversation_id"])}</h2>')
            url = c.get("source_url") or ""
            url_html = (f'<a href="{html.escape(url)}" target="_blank">source</a>'
                        if url else '<span>(no source URL)</span>')
            f.write(f'<div class="meta">clip <code>{html.escape(c["source_clip"])}</code> · '
                    f'{c["n_turns"]} turns · {c["duration_s"]:.1f}s · '
                    f'{"/".join(c["unique_speakers"])} · '
                    f'{url_html}</div>')
            for t in c["turns"]:
                spk_short = t["speaker"].split("_")[-1]
                try:
                    sp_class = f"sp{int(spk_short) % 4}"
                except ValueError:
                    sp_class = "sp0"
                f.write(f'<div class="turn"><div class="sp {sp_class}">{html.escape(t["speaker"])}<br>'
                        f'{t["start"]:.2f}-{t["end"]:.2f}s</div>'
                        f'<div><div class="text">{html.escape(t["text"])}</div>')
                if embed_audio:
                    audio_rel = str(clips_root / t["audio_path"])
                    f.write(f'<audio controls src="{html.escape(audio_rel)}"></audio>')
                f.write('</div></div>')
            f.write('</div>')
        f.write('</body></html>')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="_final root or HF-mirror layout (each clip in its own dir)")
    p.add_argument("--out", type=Path, required=True,
                   help="output dir; will contain conversations/*.json + summary.tsv + preview.html")
    p.add_argument("--min-turns", type=int, default=6,
                   help="drop conversations shorter than this (default 6)")
    p.add_argument("--max-gap-s", type=float, default=2.5,
                   help="max allowed silence between turns (default 2.5s)")
    p.add_argument("--min-duration-s", type=float, default=0.0,
                   help="drop conversations shorter than this (default 0)")
    p.add_argument("--only-two-speaker", action="store_true",
                   help="drop conversations that don't have exactly 2 unique speakers")
    p.add_argument("--exclude-bgm", action="store_true",
                   help="drop a whole conversation if any turn has flags.has_bgm")
    p.add_argument("--exclude-multi-speaker", action="store_true",
                   help="drop conversations with >2 unique speakers")
    p.add_argument("--preview-html", type=Path, default=None,
                   help="if set, also write a preview HTML with the first N conversations")
    p.add_argument("--preview-n", type=int, default=20,
                   help="how many conversations to show in the preview HTML (default 20)")
    p.add_argument("--preview-embed-audio", action="store_true",
                   help="reference audio files via relative paths in the preview HTML")
    p.add_argument("--meta-dir", type=Path, default=None,
                   help="dir with <clip>.meta.json (source URL/channel) if not in the JSONL")
    args = p.parse_args()

    convs_dir = args.out / "conversations"
    convs_dir.mkdir(parents=True, exist_ok=True)

    all_convs: list[dict] = []
    stats: dict[str, int | float] = Counter()
    per_channel = Counter()
    for clip_dir in sorted(args.root.iterdir()):
        if not clip_dir.is_dir() or clip_dir.name.startswith("_"):
            continue
        jsonl = clip_dir / f"{clip_dir.name}.jsonl"
        if not jsonl.exists():
            continue
        turns = _load_jsonl(jsonl)
        if not turns:
            continue
        stats["clips_seen"] += 1
        stats["turns_total"] += len(turns)

        # Pull source metadata from the first turn (all turns share it)
        first = turns[0]
        source_url = first.get("source_url", "")
        source_channel = first.get("source_channel", "")
        source_genre = first.get("source_genre", "")

        runs = _split_runs(turns, args.max_gap_s)
        for run_idx, run in enumerate(runs):
            if len(run) < args.min_turns:
                stats["runs_dropped_short"] += 1
                continue
            if args.min_duration_s and (run[-1]["end"] - run[0]["start"]) < args.min_duration_s:
                stats["runs_dropped_duration"] += 1
                continue
            unique_speakers = {t["speaker"] for t in run}
            if args.exclude_multi_speaker and len(unique_speakers) > 2:
                stats["runs_dropped_multispk"] += 1
                continue
            if args.only_two_speaker and len(unique_speakers) != 2:
                stats["runs_dropped_not_dyadic"] += 1
                continue
            if args.exclude_bgm and any((t.get("flags") or {}).get("has_bgm") for t in run):
                stats["runs_dropped_bgm"] += 1
                continue
            conv = _build_conversation(
                clip_dir.name, run, run_idx, source_url, source_channel, source_genre,
            )
            (convs_dir / f"{conv['conversation_id']}.json").write_text(
                json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            all_convs.append(conv)
            stats["conversations_kept"] += 1
            stats["turns_kept"] += len(run)
            stats["duration_kept_s"] += conv["duration_s"]
            per_channel[(source_channel or "unknown")[:40]] += 1

    # --- summary TSV ---
    with (args.out / "summary.tsv").open("w") as f:
        f.write("conversation_id\tsource_clip\tn_turns\tduration_s\tspeakers\tsource_url\n")
        for c in all_convs:
            f.write(f"{c['conversation_id']}\t{c['source_clip']}\t{c['n_turns']}\t"
                    f"{c['duration_s']}\t{'/'.join(c['unique_speakers'])}\t{c['source_url']}\n")

    # --- stats file ---
    stats_out = dict(stats)
    stats_out["duration_kept_h"] = round(stats.get("duration_kept_s", 0) / 3600, 2)
    stats_out["per_channel_top10"] = per_channel.most_common(10)
    (args.out / "stats.json").write_text(json.dumps(stats_out, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    # --- preview HTML ---
    if args.preview_html:
        # Sort by duration desc so the preview shows meaty conversations first
        all_convs.sort(key=lambda c: c["duration_s"], reverse=True)
        _write_html(all_convs, args.preview_html, args.root,
                    args.preview_n, args.preview_embed_audio)

    print(f"[done] clips {stats.get('clips_seen', 0)} · "
          f"conversations {stats.get('conversations_kept', 0)} · "
          f"turns_kept {stats.get('turns_kept', 0)} · "
          f"{stats_out['duration_kept_h']}h kept")
    print(f"       -> {convs_dir}/")


if __name__ == "__main__":
    main()
