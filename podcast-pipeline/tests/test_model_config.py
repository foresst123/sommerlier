"""Each model reads its own config, not another model's.

Run:  python -m pytest tests/test_model_config.py -q     (from podcast-pipeline/)
"""
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _init_params(module_path, cls_name):
    with open(os.path.join(ROOT, module_path), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                    return [a.arg for a in sub.args.args if a.arg != "self"]
    raise AssertionError(f"{cls_name}.__init__ not found in {module_path}")


def test_every_profile_configures_phowhisper_separately():
    """It used to take ASRService.batch_size, which is read from models.qwen3 --
    a different model, on a different device, with its own memory budget."""
    for env, profile in _config()["environments"].items():
        models = profile.get("models", {})
        assert "phowhisper" in models, f"{env} has no models.phowhisper"
        assert "batch_size" in models["phowhisper"], f"{env} phowhisper lacks batch_size"


def test_phowhisper_config_keys_match_its_constructor():
    params = _init_params("models/phowhisper.py", "PhoWhisperASR")
    for env, profile in _config()["environments"].items():
        for key in profile.get("models", {}).get("phowhisper", {}):
            assert key in params, f"{env}: models.phowhisper.{key} is not a constructor arg"


def test_asr_service_no_longer_passes_its_own_batch_size_to_phowhisper():
    with open(os.path.join(ROOT, "services/asr_service.py"), encoding="utf-8") as f:
        src = f.read()
    call = src[src.index("self.phowhisper.transcribe_batch"):][:200]
    assert "batch_size=self.batch_size" not in call


def test_tf32_is_opt_out_and_present_in_every_profile():
    for env, profile in _config()["environments"].items():
        assert "allow_tf32" in profile, f"{env} has no allow_tf32"
        assert isinstance(profile["allow_tf32"], bool)


def test_main_gates_tf32_on_the_config_flag():
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as f:
        src = f.read()
    assert 'env_profile.get("allow_tf32"' in src
    assert "torch.backends.cuda.matmul.allow_tf32" in src
    assert "torch.backends.cudnn.allow_tf32" in src
