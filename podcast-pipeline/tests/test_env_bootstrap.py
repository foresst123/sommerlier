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


def test_main_env_complete_installs_nothing():
    ran = []
    changed = env_bootstrap.ensure_main_env(
        LOG, "req.txt", missing=[], run=lambda cmd, logger: ran.append(cmd))
    assert changed is False and ran == []


def test_main_env_missing_torch_installs_pins_then_requirements_then_gpu_ort(monkeypatch):
    ran = []
    monkeypatch.setattr(env_bootstrap, "missing_main_modules", lambda: [])   # fine after install
    changed = env_bootstrap.ensure_main_env(
        LOG, "req.txt", missing=["torch"], python="/py", ensure_uv=lambda l: None,
        run=lambda cmd, logger: ran.append(cmd))
    assert changed is True and len(ran) == 3
    assert "torch==2.8.0" in ran[0] and "/py" in ran[0]
    assert "-r" in ran[1] and "unsafe-best-match" in ran[1]
    assert "onnxruntime-gpu==1.23.2" in ran[2] and "--no-deps" in ran[2]


def test_main_env_still_broken_after_install_raises(monkeypatch):
    monkeypatch.setattr(env_bootstrap, "missing_main_modules", lambda: ["torch"])
    with pytest.raises(RuntimeError, match="still lacks"):
        env_bootstrap.ensure_main_env(LOG, "req.txt", missing=["torch"], python="/py",
                                      ensure_uv=lambda l: None, run=lambda c, l: None)


def test_default_env_root_is_outside_the_repository(monkeypatch):
    import os
    monkeypatch.delenv("PIPELINE_ENV_ROOT", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)   # no /kaggle
    repo = os.path.dirname(os.path.dirname(os.path.abspath(env_bootstrap.__file__)))
    root = env_bootstrap.env_root()
    assert os.path.dirname(os.path.dirname(repo)) == root


def test_ffmpeg_present_installs_nothing():
    assert env_bootstrap.ensure_ffmpeg(
        LOG, which=lambda name: "/usr/bin/ffmpeg",
        run=lambda c, l: pytest.fail("must not install")) is False


def test_ffmpeg_missing_in_conda_env_is_installed(monkeypatch):
    found = {"ffmpeg": False}
    ran = []
    which = lambda name: ("/x/conda" if name == "conda" else
                          ("/env/bin/ffmpeg" if found["ffmpeg"] else None))

    def run(cmd, logger):
        ran.append(cmd)
        found["ffmpeg"] = True

    monkeypatch.setenv("PATH", "/usr/bin")
    assert env_bootstrap.ensure_ffmpeg(LOG, which=which, run=run, conda_prefix="/env") is True
    assert ran[0][:2] == ["conda", "install"] and "ffmpeg" in ran[0] and "/env" in ran[0]


def test_ffmpeg_missing_outside_conda_explains_what_to_do():
    with pytest.raises(RuntimeError, match="apt install ffmpeg"):
        env_bootstrap.ensure_ffmpeg(LOG, which=lambda name: None, conda_prefix="")
