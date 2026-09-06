"""Verify what separation produced, and recover when it made things worse.

The pipeline's QC gate decides whether to splice at all, using ECAPA similarity
on solo regions. That answers "is this the right speaker", not "is this audio
better than what we had". A resynthesising separator can hand back a track that
is unmistakably the right speaker and still be worse input for ASR: level jumps
at the splice, VAE artefacts, low-frequency junk, or interference that was never
actually removed.

This module runs after splicing and asks the second question, per span:

    did BAK improve?   (DNSMOS background-intrusiveness: is the other speaker
                        quieter than in the mixture -- a reference-free stand-in
                        for SIR, which needs ground truth we do not have)
    did SIG survive?   (speech quality: did resynthesis cost more than the
                        interference removal gained)

and picks one of three outcomes: keep, blend, or revert. Blending exists because
all-or-nothing throws away partial wins -- a track with the interferer 6dB down
but some vocoder roughness is often better ASR input at alpha=0.6 than either
the raw mixture or the full separation.

DNSMOS is optional: pass `scorer=None` and the level/DC repairs still run, which
are worth having on their own.
"""
import numpy as np

# Keep the separated track when the background got at least this much quieter
# and speech quality did not fall further than the tolerance. Both are MOS
# points on DNSMOS's 1-5 scale. NOT CALIBRATED -- see docs/ note below.
VERIFY_BAK_GAIN = 0.15
VERIFY_SIG_DROP = 0.30
VERIFY_BLEND_ALPHA = 0.6

HIGHPASS_HZ = 60.0


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def remove_dc(x: np.ndarray) -> np.ndarray:
    """Strip the DC offset a VAE decoder can leave behind.

    A constant offset is inaudible but shifts every frame's energy, which moves
    VAD boundaries and log-mel features that ASR front-ends compute.
    """
    if x.size == 0:
        return x
    return (x - np.mean(x)).astype(np.float32)


def highpass(x: np.ndarray, sr: int, cutoff: float = HIGHPASS_HZ) -> np.ndarray:
    """One-pole high-pass. Generative decoders leave rumble below the voice band
    that carries no information but does move the energy statistics ASR uses."""
    if x.size < 2:
        return x
    dt = 1.0 / sr
    rc = 1.0 / (2.0 * np.pi * cutoff)
    a = rc / (rc + dt)
    out = np.empty_like(x, dtype=np.float32)
    out[0] = x[0]
    prev_x, prev_y = x[0], x[0]
    for i in range(1, x.size):
        y = a * (prev_y + x[i] - prev_x)
        out[i] = y
        prev_x, prev_y = x[i], y
    return out


def restore_gain(track: np.ndarray, track_sum: np.ndarray, mixture: np.ndarray,
                 max_gain_db: float = 12.0) -> np.ndarray:
    """Put the separated tracks back at the level of the audio around them.

    A spliced span landing several dB away from its neighbours is a seam ASR
    front-ends read as an event. Written for DialogueSidon, which normalised
    every 20s chunk by its own peak and so returned tracks at an arbitrary
    level; USEF masks rather than resynthesises and mostly does not drift, so
    the correction is now a check that usually finds nothing. Kept because a
    near-unity gain costs nothing and the failure it guards is silent.

    The gain is derived from the SUM of both separated tracks against the
    mixture, not from this track against its neighbours. Matching a track to the
    audio beside it would drag a backchannel -- naturally 10-20dB below the main
    speaker -- up to the main speaker's level, destroying exactly the relative
    dynamics the recording had. One common gain fixes the chunk normalisation
    while leaving the two speakers' relative levels alone.

    Clamped so a near-silent track cannot be amplified into noise.
    """
    r_sum, r_mix = rms(track_sum), rms(mixture)
    if r_sum < 1e-8 or r_mix < 1e-8:
        return np.asarray(track, dtype=np.float32)
    limit = 10.0 ** (max_gain_db / 20.0)
    gain = float(np.clip(r_mix / r_sum, 1.0 / limit, limit))
    return (np.asarray(track, dtype=np.float32) * gain).astype(np.float32)


def blend(separated: np.ndarray, original: np.ndarray, alpha: float) -> np.ndarray:
    """alpha=1 keeps the separation, alpha=0 keeps the mixture."""
    n = min(len(separated), len(original))
    a = float(np.clip(alpha, 0.0, 1.0))
    return (a * separated[:n] + (1.0 - a) * original[:n]).astype(np.float32)


def decide(before: dict, after: dict,
           bak_gain: float = VERIFY_BAK_GAIN,
           sig_drop: float = VERIFY_SIG_DROP) -> tuple:
    """Choose keep / blend / revert from DNSMOS scores.

    Returns (action, detail).
    """
    d_bak = after["BAK"] - before["BAK"]
    d_sig = after["SIG"] - before["SIG"]
    detail = f"dBAK={d_bak:+.2f} dSIG={d_sig:+.2f}"

    if d_bak < bak_gain:
        # The interferer is no quieter than in the mixture. Whatever the track
        # is, it is not a separation, so the artefacts buy nothing.
        return "revert", detail + " (no interference removed)"
    if d_sig < -sig_drop:
        # Interference did drop, but resynthesis cost more speech quality than
        # the gain is worth at full strength.
        return "blend", detail + " (quality cost too high for full strength)"
    return "keep", detail


def verify_span(separated: np.ndarray, original: np.ndarray, sr: int,
                scorer=None, other_track: np.ndarray = None,
                alpha: float = VERIFY_BLEND_ALPHA):
    """Repair, score and choose for one spliced span.

    `scorer(audio, sr) -> {"SIG":..., "BAK":..., "OVRL":...}`; None skips scoring
    and keeps the repaired separation.
    `other_track` is the separator's OTHER output over the same span; passing it
    lets the common gain be derived from both tracks against the mixture. Without
    it the gain falls back to this track alone, which is right for a dominant
    speaker and wrong for a quiet backchannel.
    """
    audio = highpass(remove_dc(np.asarray(separated, dtype=np.float32)), sr)
    if other_track is not None and len(other_track):
        n = min(len(audio), len(other_track))
        audio = restore_gain(audio, audio[:n] + np.asarray(other_track[:n], dtype=np.float32), original)
    else:
        audio = restore_gain(audio, audio, original)

    if scorer is None:
        return audio, "keep", "no scorer"

    try:
        before, after = scorer(original, sr), scorer(audio, sr)
    except Exception as e:
        # A scorer failure must not lose the separation; fall through to keep.
        return audio, "keep", f"scorer failed: {type(e).__name__}"

    action, detail = decide(before, after)
    if action == "revert":
        return np.asarray(original, dtype=np.float32), action, detail
    if action == "blend":
        return blend(audio, original, alpha), action, detail
    return audio, action, detail
