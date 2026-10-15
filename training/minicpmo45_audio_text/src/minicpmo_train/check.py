"""Fast CPU checks for config, manifests and referenced audio files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from .config import load_config
from .manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-audio", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = load_manifest(config.data.train_manifest)
    tasks = Counter(row.task for row in rows)
    checked = 0
    for row in rows:
        if row.audio_path is None or checked >= args.check_audio:
            continue
        path = Path(row.audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing audio for {row.id}: {path}")
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Unreadable audio for {row.id}: {path}")
        if row.audio_end is not None and row.audio_end > info.duration + 0.02:
            raise ValueError(f"Audio bound exceeds file for {row.id}: {row.audio_end} > {info.duration}")
        checked += 1
    print(
        json.dumps(
            {
                "ok": True,
                "train_rows": len(rows),
                "tasks": dict(sorted(tasks.items())),
                "audio_files_checked": checked,
                "model": config.model.name_or_path,
                "output_dir": config.train.output_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
