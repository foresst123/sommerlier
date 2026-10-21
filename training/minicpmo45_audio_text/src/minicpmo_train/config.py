"""Strict configuration loading for the MiniCPM-o trainer."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TypeVar

import yaml


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    name_or_path: str = "openbmb/MiniCPM-o-4_5"
    revision: str = "1f761131fa83f5ed3cd6f2f22b225c4501d154fa"
    trust_remote_code: bool = True
    max_length: int = 4096
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    train_audio_projector: bool = True
    train_audio_encoder_top_layers: int = 0


@dataclass(frozen=True, kw_only=True)
class DataConfig:
    train_manifest: str
    eval_manifest: str | None = None
    num_workers: int = 4
    sample_rate: int = 16000
    max_audio_seconds: float = 30.0
    task_weights: dict[str, float] = field(
        default_factory=lambda: {"asr": 0.25, "dialogue": 0.50, "text": 0.25}
    )


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    output_dir: str
    seed: int = 42
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_steps: int = 1000
    learning_rate_lora: float = 2.0e-5
    learning_rate_projector: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    eval_steps: int = 100
    save_steps: int = 100
    gradient_checkpointing: bool = True


@dataclass(frozen=True, kw_only=True)
class Config:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig


T = TypeVar("T")


def _strict_dataclass(cls: type[T], raw: object, section: str) -> T:
    if not isinstance(raw, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in '{section}': {', '.join(unknown)}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ValueError(f"Invalid config section '{section}': {exc}") from exc


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Training config must be a YAML mapping")
    unknown = sorted(set(raw) - {"model", "data", "train"})
    if unknown:
        raise ValueError(f"Unknown top-level config sections: {', '.join(unknown)}")
    missing = [name for name in ("model", "data", "train") if name not in raw]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    config = Config(
        model=_strict_dataclass(ModelConfig, raw["model"], "model"),
        data=_strict_dataclass(DataConfig, raw["data"], "data"),
        train=_strict_dataclass(TrainConfig, raw["train"], "train"),
    )
    _validate(config)
    return config


def _validate(config: Config) -> None:
    if config.model.max_length < 512:
        raise ValueError("model.max_length must be at least 512")
    if config.model.lora_rank <= 0 or config.model.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0 <= config.model.train_audio_encoder_top_layers <= 24:
        raise ValueError("train_audio_encoder_top_layers must be between 0 and 24")
    if config.data.sample_rate != 16000:
        raise ValueError("MiniCPM-o 4.5 audio input must use 16 kHz")
    if config.data.max_audio_seconds <= 0 or config.data.max_audio_seconds > 30:
        raise ValueError("data.max_audio_seconds must be in (0, 30]")
    unknown_tasks = sorted(set(config.data.task_weights) - {"asr", "dialogue", "text"})
    if unknown_tasks:
        raise ValueError(f"Unknown data.task_weights: {', '.join(unknown_tasks)}")
    if not config.data.task_weights or any(value <= 0 for value in config.data.task_weights.values()):
        raise ValueError("data.task_weights values must be positive")
    if config.train.per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be positive")
    if config.train.gradient_accumulation_steps <= 0 or config.train.max_steps <= 0:
        raise ValueError("gradient_accumulation_steps and max_steps must be positive")
    if min(config.train.logging_steps, config.train.eval_steps, config.train.save_steps) <= 0:
        raise ValueError("logging_steps, eval_steps and save_steps must be positive")


__all__ = ["Config", "DataConfig", "ModelConfig", "TrainConfig", "load_config"]
