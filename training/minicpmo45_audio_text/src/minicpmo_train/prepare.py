"""Build deterministic train/dev manifests from conversation.json exports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .manifest import ManifestRow, rows_from_conversation, rows_from_text_replay, write_manifests


def build_rows(args: argparse.Namespace) -> list[ManifestRow]:
    tasks = set(args.tasks)
    rows: list[ManifestRow] = []
    files = sorted(args.exports_root.rglob("conversation.json"))
    for path in files:
        rows.extend(
            rows_from_conversation(
                path,
                tasks=tasks,
                strict_tracks_only=not args.allow_time_gated,
                min_audio_seconds=args.min_audio_seconds,
                max_audio_seconds=args.max_audio_seconds,
                history_turns=args.history_turns,
            )
        )
    if args.text_replay:
        rows.extend(rows_from_text_replay(args.text_replay))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=("asr", "dialogue"), default=("asr", "dialogue"))
    parser.add_argument("--text-replay", type=Path)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-audio-seconds", type=float, default=0.35)
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--allow-time-gated", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = build_rows(args)
    train_path = args.output_dir / "train.jsonl"
    dev_path = args.output_dir / "dev.jsonl"
    counts = write_manifests(
        rows,
        train_path=train_path,
        dev_path=dev_path,
        seed=args.seed,
        dev_ratio=args.dev_ratio,
    )
    task_counts = Counter(row.task for row in rows)
    summary = {
        "conversation_files_scanned": len(list(args.exports_root.rglob("conversation.json"))),
        "candidate_rows": len(rows),
        "tasks": dict(sorted(task_counts.items())),
        "splits": counts,
        "strict_tracks_only": not args.allow_time_gated,
        "train_manifest": str(train_path.resolve()),
        "dev_manifest": str(dev_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
