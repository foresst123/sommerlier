"""Configuration and partition invariants for two-GPU LLM refinement."""

import json
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pipeline_class():
    # The repository's lightweight test environment does not always install
    # torch. Device-map construction is pure Python, so a tiny import stub lets
    # that invariant remain covered there as well as in the Kaggle environment.
    import importlib.util

    module_name = "_refinement_pipeline_pool_config_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "services" / "refinement_pipeline_pool.py")
    module = importlib.util.module_from_spec(spec)
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = types.SimpleNamespace(Tensor=object)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
        sys.modules.pop(module_name, None)
    return module.RefinementPipelinePool


def test_qwen_weights_are_partitioned_once_across_the_two_gpus():
    cls = _pipeline_class()
    mapping = cls.build_device_map(36, devices=(0, 1), split_layer=20)

    assert mapping["model.embed_tokens"] == 0
    assert mapping["model.rotary_emb"] == 0
    assert mapping["model.norm"] == 1
    assert mapping["lm_head"] == 1
    assert [mapping[f"model.layers.{i}"] for i in range(36)] == [0] * 20 + [1] * 16


def test_shipped_profiles_enable_one_pipelined_model_not_two_replicas():
    config = json.loads((ROOT / "config.json").read_text())
    for profile_name in ("kaggle", "a100"):
        refinement = config["environments"][profile_name]["performance"]["stages"]["refinement"]
        assert refinement["placement"] == "pipelined"
        assert refinement["workers"] == 2
        assert refinement["micro_batch_size"] >= 1
        assert 0.2 <= refinement["pipeline_split_ratio"] <= 0.8
