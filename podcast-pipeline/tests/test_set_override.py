"""--set KEY=VALUE profile overrides. Run: python -m pytest tests/test_set_override.py -q"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.config_override import apply_overrides, parse_set


def _profile():
    return {"steps": {"music_removal": True}, "pipeline": {"music_scope": "spans"},
            "models": {"bs_roformer": {"model": "a.ckpt", "x": 1}}, "gpu_1": 0}


def test_parse_types():
    assert parse_set("steps.a=false") == (["steps", "a"], False)
    assert parse_set("pipeline.n=3")[1] == 3 and parse_set("pipeline.f=0.5")[1] == 0.5
    assert parse_set("pipeline.z=null")[1] is None
    assert parse_set('pipeline.s="true"')[1] == "true"
    assert parse_set("models.bs_roformer.model=x.ckpt") == (["models", "bs_roformer", "model"], "x.ckpt")
    assert parse_set("pipeline.e=a=b")[1] == "a=b"
    assert parse_set("pipeline.l=[1, 2]")[1] == [1, 2]


@pytest.mark.parametrize("bad", ["novalue", "=x", "a..b=1", ".=1"])
def test_bad_format(bad):
    with pytest.raises(ValueError, match="KEY=VALUE|empty segment"):
        parse_set(bad)


def test_apply_nested_and_log():
    p = _profile()
    log = apply_overrides(p, ["steps.music_removal=false", "pipeline.music_scope=full",
                              "models.bs_roformer.model=x.ckpt", "models.new.deep=1", "gpu_1=1"])
    assert p["steps"]["music_removal"] is False and p["pipeline"]["music_scope"] == "full"
    assert p["models"]["bs_roformer"] == {"model": "x.ckpt", "x": 1}
    assert p["models"]["new"] == {"deep": 1} and p["gpu_1"] == 1
    assert log[0] == "steps.music_removal=false" and log[2] == 'models.bs_roformer.model="x.ckpt"'
    q = _profile()
    assert apply_overrides(q, []) == [] and q == _profile()


def test_unknown_section_and_non_section():
    with pytest.raises(ValueError, match="unknown section 'step'"):
        apply_overrides(_profile(), ["step.music_removal=false"])
    with pytest.raises(ValueError, match="is not a section"):
        apply_overrides(_profile(), ["gpu_1.x=1"])


def _resolve(argv):
    """Run main.py's setup section (as test_config_driven does) and return args plus stdout."""
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    head = src[:src.index('if env_profile.get("offline_mode"')]
    probe = os.path.join(ROOT, "_pytest_probe_set.py")
    with open(probe, "w", encoding="utf-8") as f:
        f.write(head + '\nimport json as _j\nprint("RESULT " + _j.dumps('
                '{k: v for k, v in vars(args).items()}))\nprint("MODEL " + str(env_profile["models"]["bs_roformer"].get("model")))\n')
    try:
        env = dict(os.environ, SOMMELIER_SKIP_BOOTSTRAP="1")
        run = subprocess.run([sys.executable, probe, "--audio_dir", "x"] + argv,
                             capture_output=True, text=True, cwd=ROOT, env=env)
        res = [json.loads(l[7:]) for l in run.stdout.splitlines() if l.startswith("RESULT ")]
        if not res:
            return None, run.stdout + run.stderr
        return res[0], run.stdout
    finally:
        os.remove(probe)


def test_main_applies_set_like_config():
    base, _ = _resolve(["--env", "a100"])
    got, out = _resolve(["--env", "a100", "--set", "steps.music_removal=false",
                         "--set", "pipeline.music_scope=full", "--set", "models.bs_roformer.model=z.ckpt"])
    assert got["step_music_removal"] is False and got["music_scope"] == "full"
    assert "MODEL z.ckpt" in out and "Profile overrides" in out
    # no --set: same resolved args as before, apart from the two new flag attributes
    for k in ("set_overrides", "music_scope"):
        base.pop(k, None)
    again, _ = _resolve(["--env", "a100"])
    for k in ("set_overrides", "music_scope"):
        again.pop(k, None)
    assert base == again
    flag, _ = _resolve(["--env", "a100", "--music_scope", "full"])
    assert flag["music_scope"] == "full"


def test_main_rejects_bad_section():
    got, out = _resolve(["--env", "a100", "--set", "nope.x=1"])
    assert got is None and "unknown section 'nope'" in out


def test_denoise_section_can_be_created_for_shared_and_per_backend_keys():
    """main.py's --set denoise.chunk_s=30 used to die with "unknown section 'denoise'"; the denoise service reads config["denoise"]."""
    prof = _profile()
    applied = apply_overrides(prof, ["denoise.chunk_s=30", "denoise.re_use.chunk_s=10", "denoise.re_use.hop_portion=0.5"])
    assert prof["denoise"] == {"chunk_s": 30, "re_use": {"chunk_s": 10, "hop_portion": 0.5}}
    assert applied[0] == "denoise.chunk_s=30"
