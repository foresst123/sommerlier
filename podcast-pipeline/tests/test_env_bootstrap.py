import logging

import pytest

from utils import env_bootstrap

LOG = logging.getLogger("bootstrap-test")


def test_healthy_env_is_not_reinstalled(tmp_path):
    calls = []
    ready = env_bootstrap.ensure_worker_envs(
        ["qwen3"], LOG, root=str(tmp_path), check=lambda d, w: True,
        installers={"qwen3": lambda d, l: calls.append(d)})
    assert calls == []
    assert ready == {"qwen3": str(tmp_path / "qwen3_env")}


def test_missing_env_is_built_then_rechecked(tmp_path, monkeypatch):
    monkeypatch.setattr(env_bootstrap, "_ensure_uv", lambda logger: None)
    state = {"built": False}

    def build(env_dir, logger):
        state["built"] = True

    ready = env_bootstrap.ensure_worker_envs(
        ["sidon"], LOG, root=str(tmp_path), check=lambda d, w: state["built"],
        installers={"sidon": build})
    assert state["built"] and "sidon" in ready


def test_env_that_fails_probe_after_install_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(env_bootstrap, "_ensure_uv", lambda logger: None)
    with pytest.raises(RuntimeError, match="still fails"):
        env_bootstrap.ensure_worker_envs(
            ["vllm"], LOG, root=str(tmp_path), check=lambda d, w: False,
            installers={"vllm": lambda d, l: None})


def test_explicit_python_variable_skips_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN3_PYTHON", "/somewhere/python")
    ready = env_bootstrap.ensure_worker_envs(
        ["qwen3"], LOG, root=str(tmp_path), check=lambda d, w: False,
        installers={"qwen3": lambda d, l: pytest.fail("must not install")})
    assert ready == {}


def _clear_token_env(monkeypatch):
    for name in ("HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_bootstrap, "_cached_hf_token", lambda: None)


def test_token_from_config_is_not_prompted(monkeypatch):
    _clear_token_env(monkeypatch)
    cfg = {"huggingface_token": "hf_cfg"}
    token = env_bootstrap.ensure_hf_token(
        cfg, LOG, prompt=lambda _: pytest.fail("must not prompt"), interactive=True)
    assert token == "hf_cfg"


def test_missing_token_prompts_and_exports(monkeypatch):
    _clear_token_env(monkeypatch)
    cfg = {"huggingface_token": ""}
    token = env_bootstrap.ensure_hf_token(cfg, LOG, prompt=lambda _: " hf_typed ",
                                          interactive=True)
    assert token == "hf_typed"
    assert cfg["huggingface_token"] == "hf_typed"
    import os
    assert os.environ["HF_TOKEN"] == "hf_typed"


def test_missing_token_without_terminal_raises(monkeypatch):
    _clear_token_env(monkeypatch)
    with pytest.raises(RuntimeError, match="No HuggingFace token"):
        env_bootstrap.ensure_hf_token({}, LOG, interactive=False)


from types import SimpleNamespace


def _args(**kw):
    return SimpleNamespace(**{"audio": None, "audio_dir": None, "env": "a100", **kw})


def _cfg(out=""):
    return {"entrypoint": {"input_folder_path": "/in"},
            "environments": {"a100": {"outputs": {"clean_two_channel_dir": out}}}}


def test_enter_accepts_config_defaults():
    cfg = _cfg("/out")
    args = env_bootstrap.ask_paths(_args(), cfg, prompt=lambda _: "", interactive=True)
    assert args.audio_dir == "/in"
    assert cfg["environments"]["a100"]["outputs"]["clean_two_channel_dir"] == "/out"


def test_typed_answers_override_config():
    cfg = _cfg("/out")
    answers = iter(["/mine", "/dest"])
    args = env_bootstrap.ask_paths(_args(), cfg, prompt=lambda _: next(answers), interactive=True)
    assert args.audio_dir == "/mine"
    assert cfg["environments"]["a100"]["outputs"]["clean_two_channel_dir"] == "/dest"


def test_input_flag_skips_input_question():
    asked = []
    args = _args(audio_dir="/cli")
    env_bootstrap.ask_paths(args, _cfg(), prompt=lambda q: asked.append(q) or "", interactive=True)
    assert args.audio_dir == "/cli" and len(asked) == 1


def test_non_interactive_uses_config_silently():
    cfg = _cfg("/out")
    args = env_bootstrap.ask_paths(_args(), cfg, interactive=False)
    assert args.audio_dir == "/in"
