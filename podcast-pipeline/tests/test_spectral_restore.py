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


def test_loudness_is_matched_on_energy_not_peak():
    """This is a tone control; audio_normalize owns level.

    Matching the old *peak* would undo the correction: a bass boost raises the
    peak more than the energy, so rescaling to the previous peak divides the
    boost straight back out and the filter measures as applied but is inaudible.
    """
    # Quiet enough that the anti-clip ceiling never binds; when it does, it
    # takes precedence over energy matching and this equality would not hold.
    x = _noise() * 0.05
    out = restore(x, SR)
    rms_in = np.sqrt(np.mean(x ** 2))
    rms_out = np.sqrt(np.mean(out ** 2))
    assert np.isclose(rms_out, rms_in, rtol=1e-6)


def test_the_output_never_clips():
    """Energy matching can raise the ceiling; it may not push past full scale."""
    loud = _noise() * 9.0
    assert np.abs(restore(loud, SR)).max() <= max(np.abs(loud).max(), 0.99) + 1e-9


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


# --- consonants are spared --------------------------------------------------

def _voiced(n=SR, f0=120.0, seed=2):
    """A buzzy harmonic tone: low zero-crossing rate, like a vowel."""
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return x / np.abs(x).max() * 0.3


def _unvoiced(n=SR, seed=3):
    """High-passed noise: high zero-crossing rate, like /s/ or a /k/ burst."""
    x = np.random.default_rng(seed).standard_normal(n)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec[freqs < 2000] = 0
    return np.fft.irfft(spec, n=n) * 0.3


def _shape_db(x, lo, hi):
    """Band share of total energy, in dB -- independent of overall level."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    return 10 * np.log10(spec[(freqs >= lo) & (freqs < hi)].sum() / spec.sum() + 1e-20)


def test_a_vowel_gets_its_low_end_back():
    v = _voiced()
    gained = _shape_db(restore(v, SR), 100, 200) - _shape_db(v, 100, 200)
    assert gained > 0.5, f"vowel gained only {gained:.2f} dB"


def test_a_consonant_keeps_its_high_frequency_energy():
    """The whole point of cách 2: the burst that makes /c/ audible survives.

    The flat curve cuts 2.7-3.5 dB across this range, which is what turned
    "khác" into "khá". Unvoiced frames take only a fraction of that.
    """
    c = _unvoiced()
    lost = _shape_db(c, 2000, 4000) - _shape_db(restore(c, SR), 2000, 4000)
    assert lost < 1.5, f"consonant band lost {lost:.2f} dB"


def test_voicing_separates_the_two():
    from utils.spectral_restore import _FRAME, _voicing
    v = _voiced()[:_FRAME * 3].reshape(3, _FRAME)
    c = _unvoiced()[:_FRAME * 3].reshape(3, _FRAME)
    assert _voicing(v).mean() > 0.8
    assert _voicing(c).mean() < 0.2


def test_unmodified_audio_reconstructs_exactly():
    """Overlap-add must be transparent when the filter is identity."""
    import utils.spectral_restore as mod
    original = mod.correction_curve
    mod.correction_curve = lambda f: np.ones_like(np.asarray(f, dtype=float))
    try:
        x = _noise()
        assert np.abs(mod.restore(x, SR) - x).max() < 1e-9
    finally:
        mod.correction_curve = original


def test_a_short_clip_keeps_its_loudness():
    """The edge of the overlap-add must not cost the whole clip its level.

    Dividing by a window sum that tapers to zero produced samples tens of times
    full scale; the anti-clip ceiling then scaled everything down, and a word
    came back 20 dB quieter with its vowel gone -- audible as "phù" losing its
    "ù" and leaving only the "ph".
    """
    x = _voiced(n=int(SR * 0.42))
    out = restore(x, SR)
    change = 20 * np.log10(np.sqrt(np.mean(out ** 2)) / np.sqrt(np.mean(x ** 2)))
    assert abs(change) < 1.0, f"level moved {change:+.2f} dB"


def test_no_sample_explodes_at_the_edges():
    for seconds in (0.1, 0.42, 1.0, 3.0):
        x = _voiced(n=int(SR * seconds))
        assert np.abs(restore(x, SR)).max() < 1.0, seconds


def test_a_consonant_vowel_syllable_keeps_its_balance():
    """A vowel may not be pushed down relative to the consonant before it."""
    consonant = _unvoiced(n=int(SR * 0.12)) * 0.5
    vowel = _voiced(n=int(SR * 0.30))
    x = np.concatenate([consonant, vowel])
    out = restore(x, SR)
    n = len(consonant)

    def rms(a):
        return np.sqrt(np.mean(a ** 2))

    before = rms(x[n:]) / rms(x[:n])
    after = rms(out[n:]) / rms(out[:n])
    assert 20 * np.log10(after / before) > -1.0, "vowel lost ground to the consonant"
