import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import demucs_vocals as D  # noqa: E402


def test_to_stereo_repeats_the_channel():
    mono = np.arange(5, dtype=np.float32)
    st = D.to_stereo(mono)
    assert st.shape == (2, 5) and st.dtype == np.float32
    assert np.array_equal(st[0], mono) and np.array_equal(st[1], mono)


def test_mono_mix_averages_channels_and_keeps_mono():
    stereo = np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32)
    assert np.allclose(D.mono_mix(stereo), [2.0, 4.0])
    assert np.allclose(D.mono_mix(np.array([1.0, 2.0])), [1.0, 2.0])


def test_normalise_then_denormalise_gives_the_input_back():
    rng = np.random.default_rng(0)
    wav = (rng.standard_normal((2, 1000)) * 0.3 + 0.1).astype(np.float32)
    norm, mean, std = D.normalise(wav)
    ref = wav.mean(axis=0)
    assert mean == pytest.approx(float(ref.mean()), rel=1e-5) and std == pytest.approx(float(ref.std()), rel=1e-5)
    assert float(norm.mean(axis=0).mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(norm.mean(axis=0).std()) == pytest.approx(1.0, rel=1e-4)
    assert np.allclose(D.denormalise(norm, mean, std), wav, atol=1e-5)


def test_normalise_of_silence_stays_silent_not_nan():
    norm, mean, std = D.normalise(np.zeros((2, 10), dtype=np.float32))
    assert std == 1.0 and np.all(norm == 0.0) and np.all(np.isfinite(norm))


def test_fit_length_trims_and_pads_to_exactly_n():
    x = np.arange(10, dtype=np.float64)
    longer = D.fit_length(x, 4)
    assert longer.dtype == np.float32 and np.array_equal(longer, [0, 1, 2, 3])
    shorter = D.fit_length(x, 13)
    assert len(shorter) == 13 and np.array_equal(shorter[:10], x) and np.all(shorter[10:] == 0)
    assert len(D.fit_length(x, 10)) == 10


def test_load_mono_16k_mixes_stereo_and_refuses_other_rates(tmp_path):
    stereo = tmp_path / "s.wav"
    sf.write(str(stereo), np.stack([np.full(160, 0.2), np.full(160, 0.6)], axis=1).astype(np.float32), 16000,
             subtype="FLOAT")
    assert np.allclose(D.load_mono_16k(str(stereo)), 0.4, atol=1e-6)
    other = tmp_path / "o.wav"
    sf.write(str(other), np.zeros(800, dtype=np.float32), 8000)
    with pytest.raises(SystemExit, match="16000 Hz"):
        D.load_mono_16k(str(other))


def test_main_exits_2_with_a_message_when_demucs_is_not_installed(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "demucs", None)           # any import of demucs now fails
    monkeypatch.setitem(sys.modules, "demucs.apply", None)
    monkeypatch.setitem(sys.modules, "demucs.pretrained", None)
    src = tmp_path / "in.wav"
    sf.write(str(src), np.zeros(1600, dtype=np.float32), 16000)
    assert D.main(["--in", str(src), "--out", str(tmp_path / "v.wav")]) == 2
    assert "demucs" in capsys.readouterr().err
    assert not (tmp_path / "v.wav").exists()
