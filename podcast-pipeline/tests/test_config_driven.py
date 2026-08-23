"""Everything the run needs comes from the profile; the command line only
overrides.

Two environments live side by side in config.json -- `kaggle` and `a100` --
and `--env` picks one. Which stages run, their thresholds and every model's
batch size are profile values, so switching machines is one flag rather than a
different command line.

Run:  python -m pytest tests/test_config_driven.py -q     (from podcast-pipeline/)
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Flags that decide whether a stage runs at all.
STAGE_FLAGS = ("vad", "tse", "panns", "ASRMoE", "llm_refinement",
               "qwen3omni", "dia3", "keep_models")


def _config():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _resolve(argv):
    """Run main.py's setup section and report the resolved args."""
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as f:
        src = f.read()
    head = src[:src.index('if env_profile.get("offline_mode"')]
    probe = os.path.join(ROOT, "_pytest_probe_main.py")
    with open(probe, "w", encoding="utf-8") as f:
        f.write(head + '\nimport json as _j\nprint("RESULT " + _j.dumps('
                '{k: v for k, v in vars(args).items()}))\n')
    try:
        run = subprocess.run([sys.executable, probe] + argv,
                             capture_output=True, text=True, cwd=ROOT)
        for line in run.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[7:])
        raise AssertionError(f"probe failed: {run.stderr[-400:]}")
    finally:
        os.remove(probe)


def test_both_environments_exist():
    envs = _config()["environments"]
    assert "kaggle" in envs and "a100" in envs


def test_stage_flags_live_in_every_profile():
    """Otherwise a stage silently stays off: argparse store_true defaults to
    False, so a flag the profile cannot set is a flag that never runs."""
    for env, profile in _config()["environments"].items():
        pipeline = profile.get("pipeline", {})
        for flag in STAGE_FLAGS:
            assert flag in pipeline, f"{env}.pipeline is missing {flag}"
            assert isinstance(pipeline[flag], bool)


def test_profile_alone_turns_the_stages_on():
    args = _resolve(["--audio", "x.mp3", "--env", "kaggle"])
    for flag in ("vad", "tse", "panns", "ASRMoE", "llm_refinement"):
        assert args[flag] is True, f"{flag} should come from the profile"


def test_the_two_profiles_differ_where_the_hardware_does():
    kaggle = _resolve(["--audio", "x.mp3", "--env", "kaggle"])
    a100 = _resolve(["--audio", "x.mp3", "--env", "a100"])
    # One 40GB card holds every model; two 14.5GB cards do not.
    assert kaggle["keep_models"] is False
    assert a100["keep_models"] is True


def test_the_command_line_wins_over_the_profile():
    """The profile sets gpu_1=0; passing it explicitly must not be overwritten."""
    args = _resolve(["--audio", "x.mp3", "--env", "a100", "--gpu_1", "3"])
    assert args["gpu_1"] == 3


def test_a_threshold_can_be_overridden_without_editing_config():
    base = _resolve(["--audio", "x.mp3", "--env", "a100"])
    over = _resolve(["--audio", "x.mp3", "--env", "a100", "--merge_gap", "1.0"])
    assert base["merge_gap"] == 0.3
    assert over["merge_gap"] == 1.0


def test_every_model_reads_its_own_batch_size():
    for env, profile in _config()["environments"].items():
        models = profile.get("models", {})
        for name in ("diarizen", "whisper", "phowhisper", "qwen3", "refinement"):
            assert name in models, f"{env}.models is missing {name}"
            assert "batch_size" in models[name], f"{env}.models.{name} has no batch_size"
