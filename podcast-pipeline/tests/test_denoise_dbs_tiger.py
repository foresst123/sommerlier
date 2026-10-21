import subprocess
import sys

import numpy as np
import pytest

from models.denoise.base import DenoiseError
from models.denoise.dbs_tiger import DBSTigerBackend, mix_stems, window_average

SR = 44100


def _sig(n, seed=0):
    return np.random.default_rng(seed).uniform(-0.5, 0.5, n).astype(np.float32)


def _fake(monkeypatch, run, **cfg):
    """Backend whose model is a numpy function (n,) -> (3, n); no torch, no weights."""
    monkeypatch.setattr(DBSTigerBackend, "_load", lambda self: None)
    monkeypatch.setattr(DBSTigerBackend, "_run_model", lambda self, piece: run(piece))
    return DBSTigerBackend(cfg)


def _known(piece):  # music = 1x, sfx = 2x, speech = 4x the input
    return np.stack([piece, 2 * piece, 4 * piece])


def _identity3(piece):
    return np.stack([piece, piece, piece])


# ------------------------------------------------------------ window average

@pytest.mark.parametrize("n", [1, 5, 1000, 3 * SR - 1, 3 * SR, 3 * SR + 1, 6 * SR - 1, 6 * SR, 6 * SR + 1,
                               9 * SR - 1, 9 * SR, 9 * SR + 1, 15 * SR + 123])
def test_window_average_identity(n):
    x = _sig(n)
    y = window_average(_identity3, x, 6 * SR, 3 * SR)
    assert y.shape == (3, n)
    assert np.max(np.abs(y - x)) < 1e-5


def test_window_average_small_params_identity_and_calls():
    x = _sig(25)
    seen = []

    def run(p):
        seen.append(len(p))
        return _identity3(p)

    y = window_average(run, x, 10, 5)
    assert np.max(np.abs(y - x)) < 1e-6
    assert set(seen) == {10}  # every window is full length (last one zero-padded)
    assert len(seen) == 4  # starts 0, 5, 10, 15; the window at 15 reaches the end


def test_window_average_uniform_weights():
    # a model that marks each window with its call index: overlap region is the plain mean
    calls = iter(range(1, 10))

    def run(p):
        k = next(calls)
        return np.full((3, len(p)), float(k), dtype=np.float32)

    y = window_average(run, np.zeros(15, dtype=np.float32), 10, 5)
    # windows [0,10) = 1 and [5,15) = 2 (it reaches the end, so no third)
    assert np.allclose(y[0, :5], 1) and np.allclose(y[0, 5:10], 1.5) and np.allclose(y[0, 10:], 2)


def test_window_average_short_input_padding():
    seen = []

    def run(p):
        seen.append(len(p))
        return _identity3(p)

    window_average(run, _sig(4), 10, 5)    # <= chunk // 2: padded to a full window
    window_average(run, _sig(7), 10, 5)    # > chunk // 2: unpadded
    assert seen == [10, 7]


# --------------------------------------------------------------------- stems

def test_stem_selection_order(monkeypatch):
    x = _sig(2000)
    for stem, k in (("music", 1), ("sfx", 2), ("speech", 4)):
        b = _fake(monkeypatch, _known, stem=stem)
        assert np.allclose(b.enhance(x, SR), k * x, atol=1e-6)


def test_default_stem_is_speech(monkeypatch):
    x = _sig(2000)
    assert np.allclose(_fake(monkeypatch, _known).enhance(x, SR), 4 * x, atol=1e-6)


def test_residual_gain_math(monkeypatch):
    x = _sig(2000)
    out = _fake(monkeypatch, _known, residual_gain=0.25).enhance(x, SR)
    assert np.allclose(out, (4 + 0.25 * 3) * x, atol=1e-5)
    out = _fake(monkeypatch, _known, stem="music", residual_gain=0.5).enhance(x, SR)
    assert np.allclose(out, (1 + 0.5 * 6) * x, atol=1e-5)


def test_mix_stems_zero_gain_is_exact_stem():
    s = np.stack([_sig(10, 1), _sig(10, 2), _sig(10, 3)])
    assert np.array_equal(mix_stems(s, "speech", 0.0), s[2])


@pytest.mark.parametrize("cfg", [{"stem": "vocals"}, {"residual_gain": -0.1}, {"residual_gain": float("nan")},
                                 {"window_s": 0}, {"hop_s": 7.0}, {"hop_s": 0}])
def test_bad_config(cfg):
    with pytest.raises(ValueError):
        DBSTigerBackend(cfg)


# ----------------------------------------------------------------- enhance()

@pytest.mark.parametrize("n", [1, 2, 7, 479, 24000, 24001, 144000, 144001, 240123])
def test_24k_roundtrip_exact_length(monkeypatch, n):
    x = _sig(n)
    y = _fake(monkeypatch, _identity3).enhance(x, 24000)
    assert y.shape == (n,) and y.dtype == np.float32 and np.all(np.isfinite(y))


def test_model_sees_44k_audio(monkeypatch):
    lens = []  # 1 s is under half a window, so the model gets one zero-padded 6 s window

    def run(p):
        lens.append(len(p))
        return _identity3(p)

    _fake(monkeypatch, run).enhance(_sig(24000), 24000)
    assert lens == [6 * SR]


def test_windowing_at_44k_with_long_input(monkeypatch):
    lens = []

    def run(p):
        lens.append(len(p))
        return _identity3(p)

    n = 13 * 24000  # 13 s -> 573300 samples at 44.1 kHz: windows starting 0, 3, 6, 9 s (the 9 s one reaches the end)
    y = _fake(monkeypatch, run).enhance(_sig(n), 24000)
    assert len(y) == n
    assert lens == [6 * SR] * 4


def test_input_not_mutated(monkeypatch):
    x = _sig(5000)
    ref = x.copy()
    _fake(monkeypatch, lambda p: np.stack([p * 0, p * 0, p * 0])).enhance(x, SR)
    _fake(monkeypatch, _identity3).enhance(x, 24000)
    assert np.array_equal(x, ref)


def test_empty_input(monkeypatch):
    assert len(_fake(monkeypatch, _identity3).enhance(np.zeros(0, dtype=np.float32), 24000)) == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_output_raises(monkeypatch, bad):
    def run(p):
        out = _identity3(p).copy()
        out[2, 3] = bad
        return out

    with pytest.raises(DenoiseError):
        _fake(monkeypatch, run).enhance(_sig(3000), SR)


def test_deterministic(monkeypatch):
    x = _sig(20000)
    b = _fake(monkeypatch, _known, residual_gain=0.1)
    assert np.array_equal(b.enhance(x, 24000), b.enhance(x, 24000))


# ------------------------------------------------------------- availability

def test_check_available_messages(monkeypatch, tmp_path):
    from models.denoise import dbs_tiger as mod
    monkeypatch.delenv("DBS_TIGER_DIR", raising=False)
    monkeypatch.delenv("DBS_TIGER_CKPT", raising=False)
    monkeypatch.setenv("SOMMELIER_DENOISE_DIR", str(tmp_path / "w"))
    monkeypatch.setattr(mod, "missing_modules", lambda *n: ["torch", "safetensors"])
    ok, why = DBSTigerBackend.check_available({})
    assert not ok and "torch" in why and "safetensors" in why
    monkeypatch.setattr(mod, "missing_modules", lambda *n: [])
    ok, why = DBSTigerBackend.check_available({})
    assert not ok and "DBS_TIGER_DIR" in why
    ok, why = DBSTigerBackend.check_available({"repo_dir": str(tmp_path)})
    assert not ok and "tiger.py" in why
    (tmp_path / "dbs" / "models").mkdir(parents=True)
    (tmp_path / "dbs" / "models" / "tiger.py").write_text("")
    ok, why = DBSTigerBackend.check_available({"repo_dir": str(tmp_path)})
    assert not ok and "DBS_TIGER_CKPT" in why
    ok, why = DBSTigerBackend.check_available({"repo_dir": str(tmp_path), "checkpoint": str(tmp_path / "nope.safetensors")})
    assert not ok and "checkpoint not found" in why
    ck = tmp_path / "model.safetensors"
    ck.write_bytes(b"x")
    assert DBSTigerBackend.check_available({"repo_dir": str(tmp_path), "checkpoint": str(ck)}) == (True, "")
    monkeypatch.setenv("DBS_TIGER_DIR", str(tmp_path))
    monkeypatch.setenv("DBS_TIGER_CKPT", str(ck))
    assert DBSTigerBackend.check_available({}) == (True, "")
    monkeypatch.delenv("DBS_TIGER_CKPT")
    assert DBSTigerBackend.check_available({"auto_download": True}) == (True, "")


def test_load_raises_denoise_error_when_unavailable(monkeypatch):
    monkeypatch.delenv("DBS_TIGER_DIR", raising=False)
    with pytest.raises(DenoiseError):
        DBSTigerBackend({}).load()


def test_importing_module_does_not_import_torch():
    code = ("import sys; import models.denoise.dbs_tiger as m; "
            "assert m.DBSTigerBackend.name == 'dbs_tiger' and m.DBSTigerBackend.native_sr == 44100; "
            "assert 'torch' not in sys.modules, 'torch imported'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**__import__('os').environ, "PYTHONPATH": "."})
    assert r.returncode == 0, r.stderr
