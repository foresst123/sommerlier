"""How the ASR models are configured and where they are placed (no heavy imports)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.asr_model_config import (
    asr_device, phowhisper_kwargs, whisper_backend, whisper_ct2_kwargs)

MODELS = {
    "phowhisper": {"batch_size": 16, "compute_type": "float32"},
    "whisper": {"backend": "ctranslate2", "batch_size": 16, "model_name": "x",
                "torch_dtype": "bfloat16", "gpu_memory_utilization": 0.4,
                "max_model_len": 1, "max_new_tokens": 2, "compute_type": "float32"},
}


def test_the_boost_batch_replaces_the_configured_one_without_touching_the_profile():
    assert phowhisper_kwargs(MODELS, 48) == {"batch_size": 48, "compute_type": "float32"}
    assert phowhisper_kwargs(MODELS) == {"batch_size": 16, "compute_type": "float32"}
    assert MODELS["phowhisper"]["batch_size"] == 16


def test_vllm_only_settings_never_reach_the_ctranslate2_wrapper():
    kwargs = whisper_ct2_kwargs(MODELS, 48)
    assert kwargs == {"batch_size": 48, "compute_type": "float32"}
    assert "backend" in MODELS["whisper"]        # the profile itself is unchanged


def test_the_whisper_backend_defaults_to_ctranslate2():
    assert whisper_backend(MODELS) == "ctranslate2"
    assert whisper_backend({"whisper": {"backend": "VLLM"}}) == "vllm"
    assert whisper_backend({}) == "ctranslate2"


def test_a_placement_picks_the_gpu_and_falls_back_to_the_default_device():
    default = torch.device("cpu")
    placement = {"qwen3": 0, "whisper": 1, "phowhisper": 1}
    assert asr_device(placement, "phowhisper", default, True) == torch.device("cuda:1")
    assert asr_device(placement, "qwen3", default, True) == torch.device("cuda:0")
    assert asr_device(None, "phowhisper", default, True) is default
    assert asr_device(placement, "phowhisper", default, False) is default
    assert asr_device(placement, "unknown", default, True) is default
