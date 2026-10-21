import io
import subprocess
import sys

import numpy as np
import pytest

from models.denoise.base import DenoiseError
from models.denoise.bsrnn_urgent import BSRNN26Backend, _CkptUnpickler, model_dims, model_sr


def _sig(n, sr, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    x = 0.2 * np.sin(2 * np.pi * 300 * t) * (1 + np.sin(2 * np.pi * 2 * t)) + 0.02 * rng.standard_normal(n)
    return x.astype(np.float32)


def _backend(model, **cfg):
    """Backend with the network replaced by `model(chunk, fs)`; records what it was fed."""
    b = BSRNN26Backend(cfg)
    b.loaded = True
    b.calls = []

    def run(chunk, fs):
        b.calls.append((len(chunk), fs))
        return model(chunk, fs)

    b._run_model = run
    return b


def test_no_torch_import_at_module_import():
    code = "import sys, models.denoise.bsrnn_urgent; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("sr,want", [(8000, 8000), (16000, 16000), (22050, 24000), (24000, 24000), (32000, 32000),
                                     (44100, 48000), (48000, 48000), (96000, 48000), (6000, 8000)])
def test_model_sr(sr, want):
    assert model_sr(sr) == want


def test_arbitrary_model_level_is_mapped_back_to_input_level():
    b = _backend(lambda c, fs: 7.0 * c + 0.0)  # scale-free model: right shape, wrong level
    x = _sig(24000 * 3, 24000)
    y = b.enhance(x, 24000)
    assert y.dtype == np.float32 and len(y) == len(x)
    assert np.allclose(y, x, atol=1e-5)
    assert b.calls == [(len(x), 24000)]  # 24 kHz runs natively, one pass


def test_silence_and_unrelated_output_give_zeros():
    rng = np.random.default_rng(1)
    b = _backend(lambda c, fs: rng.standard_normal(len(c)).astype(np.float32))  # garbage even for silence
    assert not np.any(b.enhance(np.zeros(24000, np.float32), 24000))
    b = _backend(lambda c, fs: -c)  # anti-correlated: gain clamps at 0 rather than flipping the sign
    assert not np.any(b.enhance(_sig(8000, 16000), 16000))


@pytest.mark.parametrize("sr,n", [(8000, 8003), (16000, 16001), (24000, 72007), (32000, 5), (44100, 44101), (22050, 22049), (48000, 96013)])
def test_length_and_rate_invariants(sr, n):
    b = _backend(lambda c, fs: c * 3)
    x = _sig(n, sr)
    y = b.enhance(x, sr)
    assert y.shape == x.shape and y.dtype == np.float32 and np.all(np.isfinite(y))
    assert all(fs == model_sr(sr) for _, fs in b.calls)


def test_resampled_rate_round_trip_keeps_in_band_signal():
    b = _backend(lambda c, fs: c)
    x = (0.3 * np.sin(2 * np.pi * 300 * np.arange(44100 * 2) / 44100)).astype(np.float32)  # far below every filter edge
    y = b.enhance(x, 44100)
    assert [fs for _, fs in b.calls] == [48000]
    mid = slice(4410, -4410)
    assert np.max(np.abs(y[mid] - x[mid])) < 5e-3


def test_chunked_run_matches_single_pass_and_respects_chunk_size():
    x = _sig(24000 * 10, 24000)
    whole = _backend(lambda c, fs: 2 * c, chunk_s=0).enhance(x, 24000)
    b = _backend(lambda c, fs: 2 * c, chunk_s=3, overlap_s=0.5)
    y = b.enhance(x, 24000)
    assert len(b.calls) > 1 and max(n for n, _ in b.calls) <= 3 * 24000
    assert np.allclose(y, whole, atol=1e-5)


def test_short_inputs_are_padded_for_the_model_and_trimmed_back():
    b = _backend(lambda c, fs: c)
    for n in (1, 100, 479):
        y = b.enhance(_sig(n, 24000), 24000)
        assert len(y) == n
    assert min(n for n, _ in b.calls) >= 4 * (960 * 24000 // 48000)
    assert len(b.enhance(np.zeros(0, np.float32), 24000)) == 0


def test_input_is_not_mutated():
    b = _backend(lambda c, fs: c * 0.5)
    x = _sig(24000, 24000)
    x0 = x.copy()
    b.enhance(x, 24000)
    assert np.array_equal(x, x0)


def test_is_deterministic():
    x = _sig(24000 * 2, 24000)
    a = _backend(lambda c, fs: c * 1.7, chunk_s=1, overlap_s=0.25).enhance(x, 24000)
    c = _backend(lambda c, fs: c * 1.7, chunk_s=1, overlap_s=0.25).enhance(x, 24000)
    assert np.array_equal(a, c)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_model_output_raises(bad):
    def model(c, fs):
        out = c.copy()
        out[10] = bad
        return out

    with pytest.raises(DenoiseError):
        _backend(model).enhance(_sig(24000, 24000), 24000)


def test_wrong_length_model_output_raises():
    with pytest.raises(DenoiseError):
        _backend(lambda c, fs: c[:-5]).enhance(_sig(24000, 24000), 24000)


def test_bad_input_and_config():
    b = _backend(lambda c, fs: c)
    with pytest.raises(ValueError):
        b.enhance(np.zeros((2, 100), np.float32), 24000)
    with pytest.raises(ValueError):
        b.enhance(np.full(100, np.nan, np.float32), 24000)
    with pytest.raises(ValueError):
        BSRNN26Backend({"chunk_s": 4, "overlap_s": 2})


def test_check_available_messages(tmp_path, monkeypatch):
    for v in ("URGENT26_DIR", "URGENT26_BSRNN_CKPT"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("models.denoise.bsrnn_urgent.missing_modules", lambda *n: [])
    ok, why = BSRNN26Backend.check_available({})
    assert not ok and "URGENT26_DIR" in why
    ok, why = BSRNN26Backend.check_available({"repo_dir": str(tmp_path)})
    assert not ok and "baseline_code/models/bsrnn.py" in why
    (tmp_path / "baseline_code" / "models").mkdir(parents=True)
    (tmp_path / "baseline_code" / "models" / "bsrnn.py").write_text("")
    ok, why = BSRNN26Backend.check_available({"repo_dir": str(tmp_path)})
    assert not ok and "URGENT26_BSRNN_CKPT" in why
    ok, why = BSRNN26Backend.check_available({"repo_dir": str(tmp_path), "checkpoint": str(tmp_path / "nope.ckpt")})
    assert not ok and "checkpoint not found" in why
    ck = tmp_path / "bsrnn.ckpt"
    ck.write_bytes(b"x")
    assert BSRNN26Backend.check_available({"repo_dir": str(tmp_path), "checkpoint": str(ck)}) == (True, "")


def test_check_available_reports_missing_packages(monkeypatch):
    monkeypatch.setattr("models.denoise.bsrnn_urgent.missing_modules", lambda *n: ["espnet2"])
    ok, why = BSRNN26Backend.check_available({})
    assert not ok and "espnet2" in why and "requirements-denoise-bsrnn.txt" in why


def test_checkpoint_unpickler_stubs_training_classes_and_refuses_others():
    u = _CkptUnpickler(io.BytesIO(b""))
    assert u.find_class("baseline_code.config", "Config").__name__ == "Config"
    assert u.find_class("collections", "OrderedDict") is not None
    with pytest.raises(DenoiseError):
        u.find_class("os", "system")
    with pytest.raises(DenoiseError):
        u.find_class("builtins", "eval")


def test_model_dims_from_state_dict():
    sd = {"bsrnn.bsrnn.band_split.fc.0.weight": np.zeros((196, 10, 1))}
    for i in range(6):
        sd[f"bsrnn.bsrnn.rnn_time.{i}.weight_ih_l0"] = np.zeros(1)
        sd[f"bsrnn.bsrnn.rnn_time.{i}.bias_ih_l0"] = np.zeros(1)
    assert model_dims(sd) == (196, 6)
