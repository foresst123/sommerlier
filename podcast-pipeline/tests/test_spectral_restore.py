"""The EQ that undoes DialogueSidon's decoder tilt.

The curve is the inverse of a measurement taken over 43 separated tracks against
their own mixtures, so the tests that matter are the ones showing it actually
flattens a tilt rather than merely running.

Run:  python -m pytest tests/test_spectral_restore.py -q     (from podcast-pipeline/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.spectral_restore import correction_curve, enabled, restore

SR = 24000


def _band_db(x, sr, lo, hi):
    """Energy in [lo, hi) as dB of the whole-signal energy."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    band = spec[(freqs >= lo) & (freqs < hi)].sum()
    return 10 * np.log10(band / spec.sum() + 1e-20)


def _noise(n=SR * 2, seed=0):
    return np.random.default_rng(seed).standard_normal(n) * 0.1


# --- the curve itself -------------------------------------------------------

def test_bass_is_lifted_and_low_mids_are_cut():
    """The two ends of the measured tilt, in the direction that undoes it."""
    freqs = np.array([150.0, 1600.0])
    gain = correction_curve(freqs)
    assert gain[0] > 1.0, "150 Hz should be boosted"
    assert gain[1] < 1.0, "1.6 kHz should be cut"


def test_the_curve_is_smooth_rather_than_stepped():
    """A hard step between bands rings; neighbouring bins must stay close."""
    freqs = np.linspace(20, 8000, 4000)
    gain_db = 20 * np.log10(correction_curve(freqs))
    assert np.abs(np.diff(gain_db)).max() < 0.5


def test_no_correction_is_applied_at_dc():
    assert correction_curve(np.array([0.0]))[0] == 1.0


def test_gains_stay_within_the_measured_range():
    freqs = np.linspace(20, 12000, 6000)
    gain_db = 20 * np.log10(correction_curve(freqs))
    assert gain_db.max() < 5.0
    assert gain_db.min() > -5.0


def test_sub_bass_is_deliberately_under_corrected():
    """Rumble lives below 100 Hz; restoring the measured -7.1 dB would lift it."""
    assert 20 * np.log10(correction_curve(np.array([60.0]))[0]) < 4.0


# --- behaviour on signals ---------------------------------------------------

def test_a_tilt_matching_the_decoder_is_flattened():
    """The point of the module: apply the measured tilt, then undo it."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(SR * 2) * 0.1

    # Impose the tilt the decoder was measured to produce: bass down, low-mids up.
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
    tilt = np.ones_like(freqs)
    tilt[(freqs >= 100) & (freqs < 200)] = 10 ** (-4.2 / 20)
    tilt[(freqs >= 1000) & (freqs < 2000)] = 10 ** (3.5 / 20)
    tilted = np.fft.irfft(spec * tilt, n=x.size)

    before = abs(_band_db(tilted, SR, 100, 200) - _band_db(x, SR, 100, 200))
    after = abs(_band_db(restore(tilted, SR), SR, 100, 200) - _band_db(x, SR, 100, 200))
    assert after < before, f"bass deviation grew: {before:.2f} -> {after:.2f} dB"


def test_the_peak_level_is_preserved():
    """This is a tone control; audio_normalize owns level."""
    x = _noise()
    out = restore(x, SR)
    assert np.isclose(np.abs(out).max(), np.abs(x).max(), rtol=1e-6)


def test_length_and_dtype_survive():
    x = _noise().astype(np.float32)
    out = restore(x, SR)
    assert out.shape == x.shape
    assert out.dtype == np.float32


def test_silence_is_returned_untouched():
    """A track that failed separation must not have its noise floor raised."""
    x = np.zeros(SR, dtype=np.float32)
    assert np.array_equal(restore(x, SR), x)


def test_an_empty_track_does_not_raise():
    x = np.array([], dtype=np.float32)
    assert restore(x, SR).size == 0


def test_a_nonsense_sample_rate_is_refused_rather_than_guessed():
    x = _noise()
    assert np.array_equal(restore(x, 0), x)


def test_the_result_is_finite():
    """An FFT round trip that produces NaN would poison every later stage."""
    assert np.all(np.isfinite(restore(_noise(), SR)))


# --- the switch -------------------------------------------------------------

def test_restoration_is_off_unless_asked_for():
    """It changes how the audio sounds, so it may not appear by default."""
    saved = os.environ.pop("SIDON_SPECTRAL_RESTORE", None)
    try:
        assert not enabled()
    finally:
        if saved is not None:
            os.environ["SIDON_SPECTRAL_RESTORE"] = saved


def test_the_switch_accepts_the_usual_spellings():
    saved = os.environ.get("SIDON_SPECTRAL_RESTORE")
    try:
        for value in ("1", "true", "TRUE", "yes", "on"):
            os.environ["SIDON_SPECTRAL_RESTORE"] = value
            assert enabled(), value
        for value in ("0", "false", "no", "off", ""):
            os.environ["SIDON_SPECTRAL_RESTORE"] = value
            assert not enabled(), value
    finally:
        if saved is None:
            os.environ.pop("SIDON_SPECTRAL_RESTORE", None)
        else:
            os.environ["SIDON_SPECTRAL_RESTORE"] = saved


def test_both_profiles_declare_the_setting():
    """A run's audio must be reproducible from its profile alone."""
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for name, profile in config["environments"].items():
        assert "spectral_restore" in profile["models"]["sidon"], name
