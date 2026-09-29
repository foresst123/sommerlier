from __future__ import annotations

import json
import wave
from pathlib import Path

from minicpmo_train.manifest import rows_from_conversation, write_manifests


def _wav(path: Path, seconds: int = 4, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * seconds * sample_rate)


def _export(tmp_path: Path, *, source: str = "episode-1") -> Path:
    folder = tmp_path / source / "clip"
    folder.mkdir(parents=True)
    _wav(folder / "speaker_A.wav")
    _wav(folder / "speaker_B.wav")
    doc = {
        "id": f"{source}-clip",
        "folder": source,
        "files": {"speaker_A": "speaker_A.wav", "speaker_B": "speaker_B.wav"},
        "channels_2ch": {"method": "strict_separation_tracks"},
        "verification": {"ok": True},
        "conversation": [
            {"speaker": "A", "start": 0.0, "end": 1.0, "text": "Bạn khỏe không?"},
            {"speaker": "B", "start": 1.1, "end": 2.2, "text": "Mình khỏe, cảm ơn bạn."},
            {"speaker": "A", "start": 2.3, "end": 3.4, "text": "Hôm nay bạn làm gì?"},
        ],
    }
    path = folder / "conversation.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def test_builds_asr_and_cross_speaker_dialogue_rows(tmp_path: Path) -> None:
    rows = list(
        rows_from_conversation(
            _export(tmp_path),
            tasks={"asr", "dialogue"},
            strict_tracks_only=True,
            min_audio_seconds=0.3,
            max_audio_seconds=30.0,
            history_turns=2,
        )
    )
    assert [row.task for row in rows].count("asr") == 3
    assert [row.task for row in rows].count("dialogue") == 2
    dialogue = [row for row in rows if row.task == "dialogue"][0]
    assert dialogue.response == "Mình khỏe, cảm ơn bạn."
    assert dialogue.audio_path and dialogue.audio_path.endswith("speaker_A.wav")
    assert dialogue.messages[-1]["role"] == "user"


def test_split_keeps_each_source_group_in_one_partition(tmp_path: Path) -> None:
    rows = []
    for source in ("episode-1", "episode-2", "episode-3"):
        rows.extend(
            rows_from_conversation(
                _export(tmp_path, source=source),
                tasks={"asr"},
                strict_tracks_only=True,
                min_audio_seconds=0.3,
                max_audio_seconds=30.0,
                history_turns=0,
            )
        )
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    write_manifests(rows, train_path=train, dev_path=dev, seed=42, dev_ratio=0.5)
    partitions: dict[str, set[str]] = {}
    for name, path in (("train", train), ("dev", dev)):
        for line in path.read_text(encoding="utf-8").splitlines():
            group = json.loads(line)["source_group"]
            partitions.setdefault(group, set()).add(name)
    assert all(len(names) == 1 for names in partitions.values())


def test_time_gated_export_is_rejected_by_default(tmp_path: Path) -> None:
    path = _export(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["channels_2ch"]["method"] = "time_gated"
    path.write_text(json.dumps(doc), encoding="utf-8")
    rows = list(
        rows_from_conversation(
            path,
            tasks={"asr"},
            strict_tracks_only=True,
            min_audio_seconds=0.3,
            max_audio_seconds=30.0,
            history_turns=0,
        )
    )
    assert rows == []
