"""RE-USE backend: registry, availability, the subprocess contract and the worker's chunk scheme.
Run: PYTHONPATH=. python -m pytest tests/test_denoise_re_use.py -q

The real model needs CUDA and mamba_ssm, so these tests use a fake worker script or the worker's pure helpers.
"""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from models.denoise import BACKENDS, available_backends, get_backend, parse_chain
from models.denoise.base import DenoiseError
from models.denoise.re_use import ReUseBackend, ReUseLaughterBackend

PY = sys.executable
WORKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "denoise", "workers", "re_use_worker.py")

_HEAD = """
import argparse, json, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp"); ap.add_argument("--out"); ap.add_argument("--sr", type=int); ap.add_argument("--cfg-json")
a = ap.parse_args()
x = np.load(a.inp)
cfg = json.load(open(a.cfg_json))
"""


def _worker(tmp_path, body, name="w.py"):
    p = tmp_path / name
    p.write_text(_HEAD + textwrap.dedent(body))
    return str(p)


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("re_use_worker_under_test", WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ registry and availability

def test_registered_and_chainable():
    assert BACKENDS["re_use"] is ReUseBackend and BACKENDS["re_use_laughter"] is ReUseLaughterBackend
    assert parse_chain("re_use>mf2se48") == ["re_use", "mf2se48"]
    assert isinstance(get_backend("re_use_laughter"), ReUseLaughterBackend)


def test_unavailable_without_python_or_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("REUSE_PYTHON", raising=False)
    monkeypatch.delenv("REUSE_DIR", raising=False)
    monkeypatch.delenv("REUSE_LAUGHTER_DIR", raising=False)
    ok, why = ReUseBackend.check_available({})
    assert not ok and "REUSE_PYTHON" in why
    ok, why = ReUseBackend.check_available({"python": PY})
    assert not ok and "REUSE_DIR" in why
    ok, why = ReUseBackend.check_available({"python": PY, "repo_dir": str(tmp_path)})
    assert not ok and "does not look like a RE-USE download" in why
    ok, why = ReUseLaughterBackend.check_available({"python": PY})
    assert not ok and "REUSE_LAUGHTER_DIR" in why
    res = available_backends()
    assert res["re_use"][0] is False and res["re_use_laughter"][0] is False


def test_available_with_a_fake_download(tmp_path):
    repo = tmp_path / "dl"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "generator_SEMamba_time_d4.py").write_text("")
    (repo / "model.safetensors").write_bytes(b"x")
    ok, why = ReUseBackend.check_available({"python": PY, "repo_dir": str(repo), "modules": ["numpy"]})
    assert not ok and "recipe" in why
    recipe = tmp_path / "r.yaml"
    recipe.write_text("a: 1\n")
    ok, why = ReUseBackend.check_available({"python": PY, "repo_dir": str(repo), "recipe": str(recipe), "modules": ["numpy"]})
    assert ok, why
    ok, why = ReUseBackend.check_available({"python": PY, "repo_dir": str(repo), "recipe": str(recipe), "modules": ["no_such_module_xyz"]})
    assert not ok and "no_such_module_xyz" in why


def test_weights_missing_needs_hf_repo(tmp_path):
    repo = tmp_path / "dl"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "generator_SEMamba_time_d4.py").write_text("")
    recipe = tmp_path / "r.yaml"
    recipe.write_text("a: 1\n")
    base = {"python": PY, "repo_dir": str(repo), "recipe": str(recipe), "modules": ["numpy"]}
    ok, why = ReUseBackend.check_available(base)
    assert not ok and "model.safetensors" in why
    assert ReUseBackend.check_available({**base, "hf_repo": "x/y"})[0] is True


# ------------------------------------------------------------------ the subprocess contract

def test_enhance_runs_the_worker_at_24k_with_config(tmp_path):
    log = tmp_path / "seen.json"
    w = _worker(tmp_path, f"""
    json.dump({{"sr": a.sr, "n": int(len(x)), "chunk_s": cfg["chunk_s"], "hop_portion": cfg["hop_portion"],
               "hf_repo": cfg["hf_repo"], "seed": cfg["seed"]}}, open({str(log)!r}, "w"))
    np.save(a.out, (x * 0.5).astype(np.float32))
    """)
    be = get_backend("re_use", {"python": PY, "worker": w, "delay_compensation": False})
    x = (0.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)   # 1 s at 16 kHz
    y = be.enhance(x, 16000)
    assert y.dtype == np.float32 and y.shape == x.shape                 # same length and rate as the input
    seen = json.loads(log.read_text())
    assert seen["sr"] == 24000 and seen["n"] == 24000                   # the model ran at its native 24 kHz (16 kHz in, resampled up)
    assert seen["chunk_s"] == 5.0 and seen["hop_portion"] == 0.5
    assert seen["hf_repo"] == "nvidia/NVIDIA-NemotronLabs-RE-USE"
    lau = get_backend("re_use_laughter", {"python": PY, "worker": w, "delay_compensation": False})
    lau.enhance(x, 16000)
    assert json.loads(log.read_text())["hf_repo"].endswith("RE-USE-laughter")


def test_worker_failure_is_a_denoise_error_with_the_stderr_tail(tmp_path):
    w = _worker(tmp_path, "sys.stderr.write('re_use_worker: CUDA is not available\\n'); sys.exit(2)")
    be = get_backend("re_use", {"python": PY, "worker": w})
    with pytest.raises(DenoiseError, match="CUDA is not available"):
        be.enhance(np.zeros(4800, dtype=np.float32) + 0.01, 24000)


def test_the_real_worker_refuses_a_folder_that_is_not_the_download(tmp_path):
    x = tmp_path / "in.npy"
    np.save(x, np.zeros(2400, dtype=np.float32))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"repo_dir": str(tmp_path)}))
    r = subprocess.run([PY, WORKER, "--in", str(x), "--out", str(tmp_path / "o.npy"), "--sr", "24000", "--cfg-json", str(cfg)],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "not a RE-USE download" in r.stderr


def test_the_real_worker_refuses_a_rate_outside_the_model_range(tmp_path):
    x = tmp_path / "in.npy"
    np.save(x, np.zeros(2400, dtype=np.float32))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"repo_dir": str(tmp_path)}))
    r = subprocess.run([PY, WORKER, "--in", str(x), "--out", str(tmp_path / "o.npy"), "--sr", "96000", "--cfg-json", str(cfg)],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "supports 8000-48000" in r.stderr


# ------------------------------------------------------------------ the worker's chunk scheme

def test_stft_sizes_scale_with_the_sample_rate():
    w = _load_worker_module()
    assert w.scaled_stft(320, 40, 320, 8000, 8000) == (320, 40, 320)
    assert w.scaled_stft(320, 40, 320, 8000, 24000) == (960, 120, 960)
    assert w.scaled_stft(320, 40, 320, 8000, 22050) == (882, 110, 882)      # make_even(881) and make_even(110.25)
    assert w.make_even(881) == 882 and w.make_even(110) == 110


@pytest.mark.parametrize("n", [100, 1000, 1001, 1500, 2999, 5000])
def test_overlap_add_with_an_identity_model_returns_the_input(n):
    w = _load_worker_module()
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n).astype(np.float32)
    calls = []

    def run(seg):
        calls.append(len(seg))
        return seg.copy()

    y = w.overlap_add(x, run, chunk=1000, hop_portion=0.5)
    assert y.dtype == np.float32 and y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=1e-5)
    assert all(c <= 1000 for c in calls)
    if n <= 1000:
        assert calls == [n]                                                  # one pass, as upstream does for a short file


def test_overlap_add_takes_the_last_chunk_from_the_end_and_keeps_a_constant_gain():
    w = _load_worker_module()
    x = np.ones(2300, dtype=np.float32)
    starts = []

    def run(seg):
        starts.append(int(len(seg)))
        return seg * 0.25

    y = w.overlap_add(x, run, chunk=1000, hop_portion=0.5)
    np.testing.assert_allclose(y, 0.25, atol=1e-6)                          # a constant gain survives the cross-fades
    assert starts[-1] == 1000                                               # the last chunk is a full chunk from the end


def test_overlap_add_rejects_a_model_that_changes_the_length():
    w = _load_worker_module()
    with pytest.raises(ValueError, match="samples for a"):
        w.overlap_add(np.zeros(3000, dtype=np.float32), lambda s: s[:-1], chunk=1000, hop_portion=0.5)
