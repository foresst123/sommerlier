"""wpe backend: dereverberation keeps length and the direct sound, and lowers the reverberant tail."""

import numpy as np
import pytest

pytest.importorskip("nara_wpe")

from models.denoise import BACKENDS, get_backend

SR = 16000


def _reverberant(rt60=0.6, seconds=12):
    rng = np.random.default_rng(0)
    n = SR * seconds
    t = np.arange(n) / SR
    x = np.zeros(n)
    for s, e, f0 in [(0.5, 1.5, 120), (3.0, 4.2, 115), (6.0, 7.0, 125), (9.0, 10.5, 118)]:
        i = slice(int(s * SR), int(e * SR))
        tt = t[i] - s
        ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.9 * tt))) / SR
        env = np.clip(np.sin(np.pi * tt * 4.0), 0, None) * np.minimum(1, np.minimum(tt, (e - s) - tt) * 10)
        x[i] += env * sum(np.sin(k * ph) / k ** 0.8 for k in range(1, 25))
    tr = np.arange(int(rt60 * SR)) / SR
    rir = rng.standard_normal(len(tr)) * np.exp(-6.9 * tr / rt60)
    rir[0] = 1.0
    return np.convolve(x, rir)[:n].astype(np.float32)


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a, dtype=np.float64)) + 1e-12))


def test_registered_and_available():
    assert "wpe" in BACKENDS
    assert BACKENDS["wpe"].check_available()[0]


def test_tail_drops_and_direct_sound_stays_close():
    y = _reverberant()
    z = get_backend("wpe").enhance(y, SR)
    assert z.dtype == np.float32 and z.shape == y.shape and np.all(np.isfinite(z))
    tails = [slice(int(1.6 * SR), int(2.0 * SR)), slice(int(4.3 * SR), int(4.7 * SR))]
    for s in tails:
        assert 20 * np.log10(_rms(y[s]) / _rms(z[s])) > 2.0          # about 3-4 dB measured
    direct = slice(int(3.2 * SR), int(4.0 * SR))
    assert abs(20 * np.log10(_rms(z[direct]) / _rms(y[direct]))) < 3.0   # about -1.8 dB measured


def test_other_sample_rates_keep_length():
    y = _reverberant()[: SR * 6]
    ds = y[::2].copy()
    z = get_backend("wpe").enhance(ds, SR // 2)
    assert z.shape == ds.shape


def test_audio_shorter_than_the_filter_is_returned_unchanged():
    x = (np.random.default_rng(1).standard_normal(1500) * 0.1).astype(np.float32)
    assert np.array_equal(get_backend("wpe").enhance(x, SR), x)


def test_bad_config_is_refused():
    with pytest.raises(ValueError):
        get_backend("wpe", {"taps": 0})
    with pytest.raises(ValueError):
        get_backend("wpe", {"stft_size": 128, "stft_shift": 256})
