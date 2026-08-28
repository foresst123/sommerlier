"""Undoing the spectral tilt that DialogueSidon's decoder leaves behind.

Sidon is a generative separator: the diffusion head predicts latents and a VAE
decoder resynthesises the waveform. Nothing is masked, so the output carries no
musical noise -- but it also does not carry the input's spectrum. Measured over
43 separated tracks against their own mixtures, with each spectrum normalised
by its total energy so loudness does not enter into it:

      0-100 Hz   -7.1 dB      1000-1400 Hz   +3.5 dB
    100-200 Hz   -4.2 dB      1400-2000 Hz   +3.5 dB
    200-300 Hz   -2.2 dB      2000-2800 Hz   +2.7 dB
    300-500 Hz   -0.2 dB      2800-4000 Hz   +1.2 dB

That is a systematic tilt, not per-file variation: the quartiles stay on the
same side of unity in every band that matters. It is also exactly what the tilt
sounds like -- thin and hard, missing the warmth of the original voice.

This module inverts that tilt and nothing else. It is deliberately separate
from `audio_normalize`, whose contract is that it only ever applies a scalar
gain; reshaping a spectrum is precisely what that module promises not to do,
and for a good reason. Enhancement that alters spectral content is well known
to raise *perceived* quality while lowering ASR accuracy, because the
recogniser meets an artifact it was not trained on. The mitigations here are:

  * only Sidon's own output is touched, never the 95% of segments that reach
    ASR straight from the mixture;
  * the correction is the inverse of a measured curve, so it moves the audio
    back towards natural speech rather than towards a preference;
  * gains are capped, and the sub-100 Hz band is deliberately under-corrected
    because a full +7 dB there would lift rumble and handling noise that is not
    voice at all.

It is off unless `SIDON_SPECTRAL_RESTORE=1`, and the ASR effect must be
measured before it is trusted -- pleasant is not the same as accurate.
"""

import os

import numpy as np

# Inverse of the measured tilt, as (upper edge in Hz, gain in dB). The last
# entry extends to Nyquist. Bands are the ones the measurement used.
#
# Two departures from a straight inversion, both deliberate:
#
#   * below 100 Hz the measurement says -7.1 dB, and the correction is +3.0.
#     Voice fundamentals start around 85 Hz and most of what lives below that
#     in podcast audio is rumble, mic handling and desk thump. Restoring it
#     faithfully would restore the noise with it.
#   * 4000-5600 Hz measured +2.1 dB but with quartiles spanning 0.90 to 2.74 --
#     the one band where files disagree about the sign. An unreliable number is
#     not corrected at all.
_CORRECTION_DB = (
    (100.0, 3.0),
    (200.0, 4.2),
    (300.0, 2.2),
    (500.0, 0.2),
    (700.0, -1.2),
    (1000.0, -1.8),
    (1400.0, -3.5),
    (2000.0, -3.5),
    (2800.0, -2.7),
    (4000.0, -1.2),
    (None, 0.0),
)

def enabled() -> bool:
    """Whether spectral restoration is switched on for this run."""
    return os.environ.get("SIDON_SPECTRAL_RESTORE", "0").strip().lower() in (
        "1", "true", "yes", "on")


def correction_curve(freqs: np.ndarray) -> np.ndarray:
    """Linear gain for each frequency in ``freqs``, smoothly interpolated.

    Interpolation runs on log-frequency because hearing and the measurement
    bands are both logarithmic; a linear ramp would spend most of its width
    inside the top band and leave the bass corners abrupt.
    """
    edges, gains = [], []
    low = 20.0
    for upper, db in _CORRECTION_DB:
        hi = float(upper) if upper is not None else max(float(freqs[-1]), low * 2)
        # Anchor each band's gain at its geometric centre, then let the
        # interpolation form the ramps between neighbours.
        edges.append(np.sqrt(max(low, 1.0) * max(hi, low * 1.0001)))
        gains.append(db)
        low = hi

    edges = np.asarray(edges, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)

    f = np.maximum(np.asarray(freqs, dtype=np.float64), 1.0)
    db = np.interp(np.log2(f), np.log2(edges), gains,
                   left=gains[0], right=gains[-1])
    # DC carries no voice and a gain there only adds offset.
    db = np.where(np.asarray(freqs) < 20.0, 0.0, db)
    return 10.0 ** (db / 20.0)


# Analysis frame for the voiced/unvoiced split. 32 ms at 24 kHz resolves the
# ~100 Hz bins the low bands need while still being short enough that a stop
# burst is not averaged into the vowel beside it.
_FRAME = 768
_HOP = _FRAME // 4

# Above this zero-crossing rate a frame is friction rather than phonation: /s/,
# /x/, /ch/ and the release of /k/, /t/, /c/ all sit well above it, while
# vowels and nasals sit far below.
_UNVOICED_ZCR = 0.18

# Share of the correction applied to unvoiced frames. Not zero: leaving them
# completely untouched puts a seam at every consonant, and the tilt is real
# there too -- just far less damaging than losing the burst.
_UNVOICED_MIX = 0.25


def _voicing(frames: np.ndarray) -> np.ndarray:
    """Per-frame weight: 1.0 for fully voiced, 0.0 for fully unvoiced.

    Zero-crossing rate alone is the right instrument here. Pitch tracking would
    be more precise but costs far more, and the decision only has to separate
    friction from phonation, which ZCR does cleanly at this frame length.
    The ramp rather than a hard threshold keeps a frame that sits near the
    boundary from flipping between two different filters mid-word.
    """
    zcr = (np.diff(np.sign(frames), axis=1) != 0).mean(axis=1)
    return np.clip((_UNVOICED_ZCR - zcr) / (_UNVOICED_ZCR * 0.5) + 0.5, 0.0, 1.0)


def restore(track: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the inverse tilt, sparing the consonants.

    The measured tilt is an average over whole segments, and applying it that
    way is what a single whole-signal FFT does. That turned out to be wrong in
    a way listening caught before the numbers did: the correction cuts 2.7-3.5
    dB across 1.4-4 kHz, which is exactly where the burst of a final /c/, /t/
    or /k/ lives, while lifting 100-200 Hz by 4.2 dB so the vowel then masks
    what is left. The result is warmer and less intelligible -- "khác" heard as
    "khá".

    So the correction is applied per frame, in full to voiced frames and at
    `_UNVOICED_MIX` to unvoiced ones. Vowels get their warmth back; consonants
    keep the high-frequency energy that makes them audible as consonants.

    Overlap-add with a Hann window at 75% overlap: the window sums to a
    constant, so unmodified audio reconstructs exactly and a frame-varying
    filter cross-fades instead of stepping.
    """
    x = np.asarray(track, dtype=np.float64).reshape(-1)
    if x.size == 0 or sample_rate <= 0:
        return track

    peak_in = float(np.abs(x).max())
    if peak_in < 1e-9:
        # Silence has no spectrum to correct, and scaling it would only raise
        # the noise floor of a track that failed separation.
        return track

    # Too short to frame: fall back to the whole-signal filter rather than
    # returning the audio uncorrected.
    if x.size < _FRAME * 2:
        spectrum = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(x.size, 1.0 / float(sample_rate))
        out = np.fft.irfft(spectrum * correction_curve(freqs), n=x.size)
    else:
        window = np.hanning(_FRAME + 1)[:_FRAME]
        starts = range(0, x.size - _FRAME + 1, _HOP)
        frames = np.stack([x[s:s + _FRAME] for s in starts])
        weight = _voicing(frames)

        gain = correction_curve(np.fft.rfftfreq(_FRAME, 1.0 / float(sample_rate)))
        spectra = np.fft.rfft(frames * window, axis=1)

        # One filter per frame, interpolated in dB between the full correction
        # and the reduced one so the transition is smooth in the perceptual
        # domain rather than in raw amplitude.
        gain_db = 20.0 * np.log10(np.maximum(gain, 1e-9))
        per_frame = 10.0 ** ((gain_db[None, :] * (
            weight[:, None] + (1.0 - weight[:, None]) * _UNVOICED_MIX)) / 20.0)

        # Windowed a second time before overlap-add. Analysis and synthesis
        # windows together make the reconstruction tolerant of the per-frame
        # filter, which would otherwise let frame edges disagree and cancel.
        filtered = np.fft.irfft(spectra * per_frame, n=_FRAME, axis=1) * window

        out = np.zeros(x.size)
        norm = np.zeros(x.size)
        for i, s in enumerate(starts):
            out[s:s + _FRAME] += filtered[i]
            norm[s:s + _FRAME] += window ** 2
        # Edges see fewer overlapping windows; dividing by the achieved sum
        # keeps their level right instead of fading the first and last frames.
        # Only divide where the windows actually overlap. At the very edges the
        # sum tapers to zero, and dividing there turns a handful of samples into
        # a spike tens of times full scale -- which the anti-clip ceiling then
        # answers by scaling the *whole* clip down, quietly costing 20 dB on
        # every syllable. Those samples keep their original value instead.
        covered = norm > 0.05
        out = np.where(covered, out / np.maximum(norm, 1e-9), x)

    # Match loudness on RMS, not peak. A bass boost raises the peak more than
    # it raises the energy, so normalising to the old peak divides the boost
    # straight back out -- the correction measures as applied and is inaudible.
    rms_in = float(np.sqrt(np.mean(x ** 2)))
    rms_out = float(np.sqrt(np.mean(out ** 2)))
    if rms_out > 1e-9 and rms_in > 1e-9:
        out *= rms_in / rms_out

    # Keep the headroom promise that peak matching used to give for free: this
    # is a tone control, so it may raise the ceiling but must not clip.
    peak_out = float(np.abs(out).max())
    ceiling = max(peak_in, 0.99)
    if peak_out > ceiling:
        out *= ceiling / peak_out

    return out.astype(np.asarray(track).dtype, copy=False)
