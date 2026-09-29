"""Manifest contracts and conversion from Sommelier conversation exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

TaskKind = Literal["asr", "dialogue", "text"]

ASR_INSTRUCTION = (
    "Hãy nghe kỹ đoạn âm thanh tiếng Việt và chép lại chính xác nội dung đã nói."
)
DIALOGUE_INSTRUCTION = (
    "Hãy nghe lời người dùng và trả lời tự nhiên, đúng ngữ cảnh bằng tiếng Việt."
)
DEFAULT_SYSTEM = (
    "Bạn là trợ lý hội thoại tiếng Việt. Trả lời đúng trọng tâm, tự nhiên và không bịa "
    "nội dung không có trong ngữ cảnh."
)


@dataclass(frozen=True, kw_only=True)
class ManifestRow:
    id: str
    task: TaskKind
    messages: list[dict[str, str]]
    response: str
    audio_path: str | None
    audio_start: float | None
    audio_end: float | None
    source_group: str
    metadata: dict[str, object]

    def validate(self) -> None:
        if not self.id or not self.source_group:
            raise ValueError("manifest row requires id and source_group")
        if self.task not in {"asr", "dialogue", "text"}:
            raise ValueError(f"unsupported task: {self.task}")
        if not self.response.strip():
            raise ValueError(f"empty response in {self.id}")
        if not self.messages or self.messages[-1].get("role") != "user":
            raise ValueError(f"{self.id} must end with a user message before response")
        if self.task == "text":
            if self.audio_path is not None:
                raise ValueError(f"text row {self.id} cannot carry audio")
            return
        if self.audio_path is None or self.audio_start is None or self.audio_end is None:
            raise ValueError(f"audio row {self.id} is missing audio bounds")
        if self.audio_start < 0 or self.audio_end <= self.audio_start:
            raise ValueError(f"invalid audio bounds in {self.id}")


def _stable_id(*parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _split_name(source_group: str, *, seed: int, dev_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}:{source_group}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return "dev" if fraction < dev_ratio else "train"


def _audio_file(meta: dict, directory: Path, speaker: str) -> Path | None:
    files = meta.get("files")
    if not isinstance(files, dict):
        return None
    candidate = files.get(f"speaker_{speaker}") or files.get("mixture")
    if not isinstance(candidate, str):
        return None
    path = (directory / candidate).resolve()
    return path if path.is_file() else None


def _source_group(meta: dict, json_path: Path) -> str:
    for key in ("source_id", "folder"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json_path.parent.parent.name or json_path.parent.name


def rows_from_conversation(
    json_path: Path,
    *,
    tasks: set[TaskKind],
    strict_tracks_only: bool,
    min_audio_seconds: float,
    max_audio_seconds: float,
    history_turns: int,
) -> Iterator[ManifestRow]:
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        return
    verification = meta.get("verification")
    if isinstance(verification, dict) and verification.get("ok") is False:
        return
    channels = meta.get("channels_2ch")
    method = channels.get("method") if isinstance(channels, dict) else None
    if strict_tracks_only and method != "strict_separation_tracks":
        return
    turns = meta.get("conversation")
    if not isinstance(turns, list):
        return
    source_group = _source_group(meta, json_path)
    conversation_id = str(meta.get("id") or json_path.parent.name)

    clean_turns: list[dict] = []
    for position, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        try:
            start, end = float(turn["start"]), float(turn["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(turn.get("text") or "").strip()
        speaker = str(turn.get("speaker") or "")
        duration = end - start
        audio_path = _audio_file(meta, json_path.parent, speaker)
        if (
            speaker not in {"A", "B"}
            or not text
            or audio_path is None
            or duration < min_audio_seconds
            or duration > max_audio_seconds
        ):
            continue
        clean_turns.append(
            {
                "position": position,
                "speaker": speaker,
                "text": text,
                "start": start,
                "end": end,
                "audio": audio_path,
            }
        )

    if "asr" in tasks:
        for index, turn in enumerate(clean_turns):
            row = ManifestRow(
                id=_stable_id(conversation_id, "asr", index, turn["start"], turn["end"]),
                task="asr",
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": ASR_INSTRUCTION},
                ],
                response=str(turn["text"]),
                audio_path=str(turn["audio"]),
                audio_start=float(turn["start"]),
                audio_end=float(turn["end"]),
                source_group=source_group,
                metadata={"conversation_id": conversation_id, "speaker": turn["speaker"], "method": method},
            )
            row.validate()
            yield row

    if "dialogue" in tasks:
        for index in range(1, len(clean_turns)):
            user_turn, answer_turn = clean_turns[index - 1], clean_turns[index]
            if (
                user_turn["speaker"] == answer_turn["speaker"]
                or int(answer_turn["position"]) != int(user_turn["position"]) + 1
            ):
                continue
            context: list[dict[str, str]] = [{"role": "system", "content": DEFAULT_SYSTEM}]
            history = clean_turns[max(0, index - 1 - history_turns) : index - 1]
            while history and history[0]["speaker"] != user_turn["speaker"]:
                history = history[1:]
            for historic in history:
                role = "user" if historic["speaker"] == user_turn["speaker"] else "assistant"
                if context[-1]["role"] == role:
                    context[-1]["content"] += "\n" + str(historic["text"])
                else:
                    context.append({"role": role, "content": str(historic["text"])})
            if context[-1]["role"] == "user":
                context[-1]["content"] += "\n" + DIALOGUE_INSTRUCTION
            else:
                context.append({"role": "user", "content": DIALOGUE_INSTRUCTION})
            row = ManifestRow(
                id=_stable_id(conversation_id, "dialogue", index, user_turn["start"], answer_turn["start"]),
                task="dialogue",
                messages=context,
                response=str(answer_turn["text"]),
                audio_path=str(user_turn["audio"]),
                audio_start=float(user_turn["start"]),
                audio_end=float(user_turn["end"]),
                source_group=source_group,
                metadata={
                    "conversation_id": conversation_id,
                    "user_speaker": user_turn["speaker"],
                    "assistant_speaker": answer_turn["speaker"],
                    "method": method,
                },
            )
            row.validate()
            yield row


def rows_from_text_replay(path: Path) -> Iterator[ManifestRow]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            messages = raw.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_number}: messages must contain user and assistant")
            final = messages[-1]
            if not isinstance(final, dict) or final.get("role") != "assistant":
                raise ValueError(f"{path}:{line_number}: final message must be assistant")
            prompt_messages = messages[:-1]
            normalized = [
                {"role": str(message["role"]), "content": str(message["content"])}
                for message in prompt_messages
            ]
            source_group = str(raw.get("source_group") or f"text:{line_number}")
            row = ManifestRow(
                id=str(raw.get("id") or _stable_id(path, line_number)),
                task="text",
                messages=normalized,
                response=str(final.get("content") or ""),
                audio_path=None,
                audio_start=None,
                audio_end=None,
                source_group=source_group,
                metadata={"text_replay": str(path), "line": line_number},
            )
            row.validate()
            yield row


def write_manifests(
    rows: Iterable[ManifestRow],
    *,
    train_path: Path,
    dev_path: Path,
    seed: int,
    dev_ratio: float,
) -> dict[str, int]:
    if not 0 <= dev_ratio < 1:
        raise ValueError("dev_ratio must be in [0, 1)")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "dev": 0}
    seen: set[str] = set()
    with train_path.open("w", encoding="utf-8") as train_handle, dev_path.open(
        "w", encoding="utf-8"
    ) as dev_handle:
        for row in rows:
            row.validate()
            if row.id in seen:
                continue
            seen.add(row.id)
            split = _split_name(row.source_group, seed=seed, dev_ratio=dev_ratio)
            handle = dev_handle if split == "dev" else train_handle
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            counts[split] += 1
    return counts


def load_manifest(path: str | Path) -> list[ManifestRow]:
    manifest_path = Path(path)
    rows: list[ManifestRow] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = ManifestRow(**json.loads(line))
                row.validate()
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{manifest_path}:{line_number}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


__all__ = [
    "ManifestRow",
    "TaskKind",
    "load_manifest",
    "rows_from_conversation",
    "rows_from_text_replay",
    "write_manifests",
]
