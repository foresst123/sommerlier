"""Dataset and model-specific collation without depending on LLaMA-Factory."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from torch.utils.data import Dataset, WeightedRandomSampler

from .manifest import ManifestRow, load_manifest


def find_last_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return start
    return -1


class ManifestDataset(Dataset[ManifestRow]):
    def __init__(self, manifest_path: str | Path) -> None:
        self.rows = load_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> ManifestRow:
        return self.rows[index]

    def weighted_sampler(self, task_weights: dict[str, float], *, seed: int) -> WeightedRandomSampler:
        counts = Counter(row.task for row in self.rows)
        missing = sorted(set(counts) - set(task_weights))
        if missing:
            raise ValueError(f"Missing task weights for: {', '.join(missing)}")
        weights = [task_weights[row.task] / counts[row.task] for row in self.rows]
        generator = torch.Generator()
        generator.manual_seed(seed)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.rows),
            replacement=True,
            generator=generator,
        )


def _load_audio(row: ManifestRow, target_sample_rate: int, max_audio_seconds: float) -> np.ndarray:
    if row.audio_path is None or row.audio_start is None or row.audio_end is None:
        raise ValueError(f"row {row.id} does not contain audio")
    info = sf.info(row.audio_path)
    start_frame = max(0, round(row.audio_start * info.samplerate))
    end_frame = min(info.frames, round(row.audio_end * info.samplerate))
    if end_frame <= start_frame:
        raise ValueError(f"row {row.id} resolves to an empty audio slice")
    audio, sample_rate = sf.read(
        row.audio_path,
        start=start_frame,
        stop=end_frame,
        dtype="float32",
        always_2d=True,
    )
    mono = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        waveform = torch.from_numpy(mono).unsqueeze(0)
        mono = audio_functional.resample(waveform, sample_rate, target_sample_rate).squeeze(0).numpy()
    limit = round(max_audio_seconds * target_sample_rate)
    if len(mono) > limit:
        raise ValueError(f"row {row.id} exceeds max_audio_seconds after loading")
    return np.ascontiguousarray(mono, dtype=np.float32)


@dataclass(kw_only=True)
class MiniCPMODataCollator:
    processor: object
    max_length: int
    sample_rate: int = 16000
    max_audio_seconds: float = 30.0

    def __call__(self, rows: list[ManifestRow]) -> dict[str, object]:
        tokenizer = self.processor.tokenizer
        prompts: list[str] = []
        audios: list[list[np.ndarray]] = []
        responses: list[str] = []
        tasks: list[str] = []
        ids: list[str] = []

        for row in rows:
            messages = [dict(message) for message in row.messages]
            sample_audios: list[np.ndarray] = []
            if row.audio_path is not None:
                sample_audios.append(_load_audio(row, self.sample_rate, self.max_audio_seconds))
                messages[-1]["content"] = messages[-1]["content"].rstrip() + "\n<audio>./</audio>"
            full_messages = messages + [{"role": "assistant", "content": row.response}]
            prompt = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
                use_tts_template=False,
                enable_thinking=False,
            )
            prompts.append(prompt)
            audios.append(sample_audios)
            responses.append(row.response)
            tasks.append(row.task)
            ids.append(row.id)

        encoded = self.processor(
            text=prompts,
            images=None,
            audios=audios,
            max_length=self.max_length,
            stream_input=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].long()
        attention_mask = encoded["attention_mask"].bool()
        labels = torch.full_like(input_ids, -100, dtype=torch.long)
        for batch_index, response in enumerate(responses):
            answer_ids = tokenizer.encode(response, add_special_tokens=False)
            full_ids = input_ids[batch_index].tolist()
            start = find_last_subsequence(full_ids, answer_ids)
            if start < 0:
                raise ValueError(
                    f"Response tokens were truncated or changed for sample {ids[batch_index]}; "
                    "raise model.max_length or shorten its context"
                )
            non_padding = torch.where(attention_mask[batch_index])[0]
            end = int(non_padding[-1]) + 1
            labels[batch_index, start:end] = input_ids[batch_index, start:end]

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(~attention_mask, 0)
        data = dict(encoded)
        data.pop("attention_mask", None)
        data.pop("image_sizes", None)
        data["input_ids"] = input_ids
        data["position_ids"] = position_ids
        # init_vision=False is safe only when the remote forward receives this key.
        data["vision_hidden_states"] = [[] for _ in rows]
        return {
            "data": data,
            "attention_mask": attention_mask,
            "labels": labels,
            "tasks": tasks,
            "ids": ids,
        }


__all__ = ["ManifestDataset", "MiniCPMODataCollator", "find_last_subsequence"]
