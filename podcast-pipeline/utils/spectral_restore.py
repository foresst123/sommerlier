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


def restore(track: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the inverse tilt to one separated track.

    Overlap-add is unnecessary here: a separated span is a few seconds at most
    and is processed whole, so a single FFT is both exact and cheaper. The
    result is scaled back to the input's peak, because this is a tone control
    and must not double as a level change -- `audio_normalize` owns level.
    """
    x = np.asarray(track, dtype=np.float64).reshape(-1)
    if x.size == 0 or sample_rate <= 0:
        return track

    peak_in = float(np.abs(x).max())
    if peak_in < 1e-9:
        # Silence has no spectrum to correct, and scaling it would only raise
        # the noise floor of a track that failed separation.
        return track

    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / float(sample_rate))
    out = np.fft.irfft(spectrum * correction_curve(freqs), n=x.size)

    peak_out = float(np.abs(out).max())
    if peak_out > 1e-9:
        out *= peak_in / peak_out

    return out.astype(np.asarray(track).dtype, copy=False)
