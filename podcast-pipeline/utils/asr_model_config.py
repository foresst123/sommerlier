"""Configuration and placement of the in-process ASR models.

Kept free of heavy imports so it can be tested without the model libraries;
ModelLoader builds the models from what these return.
"""

import torch

# Settings that only the vLLM Whisper backend understands.
_VLLM_WHISPER_KEYS = ("backend", "model_name", "gpu_memory_utilization",
                      "max_model_len", "max_new_tokens", "torch_dtype")


def phowhisper_kwargs(models_cfg: dict, batch_size=None) -> dict:
    cfg = dict((models_cfg or {}).get("phowhisper", {}))
    if batch_size:
        cfg["batch_size"] = int(batch_size)
    return cfg


def whisper_ct2_kwargs(models_cfg: dict, batch_size=None) -> dict:
    cfg = dict((models_cfg or {}).get("whisper", {}))
    for key in _VLLM_WHISPER_KEYS:
        cfg.pop(key, None)
    if batch_size:
        cfg["batch_size"] = int(batch_size)
    return cfg


def whisper_backend(models_cfg: dict) -> str:
    return str((models_cfg or {}).get("whisper", {}).get("backend", "ctranslate2")).lower()


def asr_device(placement, kind: str, default, cuda_available: bool):
    """The device for `kind` under a placement, or `default` when none applies."""
    if placement and cuda_available and kind in placement:
        return torch.device(f"cuda:{placement[kind]}")
    return default
