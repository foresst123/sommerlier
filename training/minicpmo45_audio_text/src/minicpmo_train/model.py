"""MiniCPM-o 4.5 loading, freezing and LoRA targeting."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoProcessor

from .config import ModelConfig

LLM_LORA_PATTERN = (
    r"^llm\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
)


def load_processor(config: ModelConfig):
    return AutoProcessor.from_pretrained(
        config.name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )


def load_trainable_model(config: ModelConfig, *, adapter_path: str | None = None):
    model = AutoModel.from_pretrained(
        config.name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        init_vision=False,
        init_audio=True,
        init_tts=False,
        low_cpu_mem_usage=True,
    )
    model.config.stream_input = False
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.llm.config, "use_cache"):
        model.llm.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    modules_to_save: list[str] = []
    if config.train_audio_projector:
        modules_to_save.append("audio_projection_layer")
    top_layers = config.train_audio_encoder_top_layers
    if top_layers:
        modules_to_save.extend(f"apm.layers.{index}" for index in range(24 - top_layers, 24))
    if adapter_path is not None:
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=LLM_LORA_PATTERN,
        modules_to_save=modules_to_save or None,
    )
    model = get_peft_model(model, lora_config)
    return model


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    base = getattr(model, "base_model", model)
    remote = getattr(base, "model", base)
    llm = getattr(remote, "llm", None)
    if llm is None or not hasattr(llm, "gradient_checkpointing_enable"):
        raise RuntimeError("Qwen3 gradient checkpointing is unavailable on this checkpoint")
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(llm, "enable_input_require_grads"):
        llm.enable_input_require_grads()


def named_trainable_parameters(model: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    return ((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)


def parameter_groups(model: torch.nn.Module, *, lora_lr: float, projector_lr: float, weight_decay: float):
    lora: list[torch.nn.Parameter] = []
    projector: list[torch.nn.Parameter] = []
    audio_encoder: list[torch.nn.Parameter] = []
    unexpected: list[str] = []
    for name, parameter in named_trainable_parameters(model):
        if "lora_" in name:
            lora.append(parameter)
        elif "audio_projection_layer" in name:
            projector.append(parameter)
        elif re.search(r"(?:^|\.)apm\.layers\.\d+\.", name):
            audio_encoder.append(parameter)
        else:
            unexpected.append(name)
    if unexpected:
        raise RuntimeError("Unexpected trainable parameters: " + ", ".join(unexpected[:20]))
    groups = []
    if lora:
        groups.append({"params": lora, "lr": lora_lr, "weight_decay": weight_decay, "name": "lora"})
    if projector:
        groups.append(
            {"params": projector, "lr": projector_lr, "weight_decay": weight_decay, "name": "projector"}
        )
    if audio_encoder:
        groups.append(
            {"params": audio_encoder, "lr": lora_lr, "weight_decay": weight_decay, "name": "audio_encoder"}
        )
    if not groups:
        raise RuntimeError("No trainable parameters were selected")
    return groups


def trainable_summary(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


__all__ = [
    "LLM_LORA_PATTERN",
    "enable_gradient_checkpointing",
    "load_processor",
    "load_trainable_model",
    "parameter_groups",
    "trainable_summary",
]
