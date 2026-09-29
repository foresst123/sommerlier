"""Tests for the level-conditioning helpers in utils/audio_normalize.py.

These pin the two properties that make the module safe to put in front of an
ASR model: every correction is a single scalar (so no spectral artifact can be
introduced), and no correction is allowed to amplify silence or a failed
extraction into noise.

Run:  python -m pytest tests/test_audio_normalize.py -q     (from podcast-pipeline/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio_normalize import (
    MAX_GAIN,
    TARGET_RMS,
    match_splice_level,
    measure,
    normalize_for_asr,
    remove_dc,
    safe_limit,
)

SR = 24000


def _noise(dur, scale, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(dur * SR)) * scale).astype(np.float32)


# --- normalize_for_asr -------------------------------------------------------

def test_gain_is_a_pure_scalar_so_the_spectrum_is_untouched():
    """The whole safety argument rests on this: a scalar cannot invent an
    artifact the recogniser has not heard before."""
    x = _noise(0.5, 0.02)
    y = normalize_for_asr(x)
    ratios = y[np.abs(x) > 1e-9] / x[np.abs(x) > 1e-9]
    assert np.std(ratios) < 1e-4


def test_quiet_and_loud_segments_converge_on_one_level():
    levels = [np.sqrt(np.mean(normalize_for_asr(_noise(1.0, s, seed=i)) ** 2))
              for i, s in enumerate([0.004, 0.01, 0.05, 0.12, 0.35])]
    assert max(levels) / min(levels) < 4.0
    assert all(abs(v - TARGET_RMS) < TARGET_RMS for v in levels)


def test_output_never_clips():
    for scale in (0.004, 0.35, 0.9, 2.0):
        assert measure(normalize_for_asr(_noise(0.2, scale)))["peak"] <= 0.951


def test_silence_is_not_amplified_into_noise():
    """A near-silent segment is silent because nobody spoke. Lifting it to the
    target level would hand the decoder a noise floor to hallucinate over."""
    assert measure(normalize_for_asr(_noise(1.0, 2e-6)))["rms"] < 1e-4
    assert measure(normalize_for_asr(np.zeros(SR, dtype=np.float32)))["peak"] == 0.0


def test_gain_is_capped():
    x = _noise(1.0, 1e-4)
    g = measure(normalize_for_asr(x))["rms"] / measure(x)["rms"]
    assert g <= MAX_GAIN + 1e-2


def test_empty_and_none_are_safe():
    assert normalize_for_asr(None) is None
    assert len(normalize_for_asr(np.array([], dtype=np.float32))) == 0


# --- remove_dc ---------------------------------------------------------------

def test_dc_offset_is_removed_but_a_negligible_one_is_left_alone():
    biased = _noise(1.0, 0.05) + 0.02
    assert abs(measure(remove_dc(biased))["dc"]) < 1e-6
    clean = _noise(1.0, 0.05, seed=3)
    assert remove_dc(clean) is clean          # untouched, not merely equal


# --- match_splice_level ------------------------------------------------------

def test_a_quiet_separated_patch_is_lifted_to_the_level_it_replaces():
    """Without this, a 0.25s backchannel spliced in at half the surrounding
    level has an audible step across most of its length."""
    host = _noise(0.25, 0.12, seed=1)
    patch = _noise(0.25, 0.045, seed=2)
    before = measure(host)["rms"] / measure(patch)["rms"]
    after = measure(host)["rms"] / measure(match_splice_level(host, patch))["rms"]
    assert before > 2.0
    assert 0.85 < after < 1.18


def test_a_failed_extraction_is_not_amplified_without_bound():
    host = _noise(0.2, 0.15, seed=4)
    near_silent = _noise(0.2, 1e-5, seed=5)
    out = match_splice_level(host, near_silent)
    assert measure(out)["rms"] / measure(near_silent)["rms"] <= 3.01


def test_splice_matching_is_also_a_pure_scalar():
    host = _noise(0.2, 0.12, seed=6)
    patch = _noise(0.2, 0.04, seed=7)
    out = match_splice_level(host, patch)
    ratios = out[np.abs(patch) > 1e-9] / patch[np.abs(patch) > 1e-9]
    assert np.std(ratios) < 1e-4


# --- safe_limit --------------------------------------------------------------

def test_limiter_preserves_the_waveform_where_a_hard_clip_would_fold_it():
    over = _noise(1.0, 0.3, seed=8) + _noise(1.0, 0.3, seed=9)
    assert measure(over)["peak"] > 1.0

    limited, gain = safe_limit(over)
    assert np.allclose(limited / gain, over, atol=1e-6)   # shape intact
    assert measure(limited)["clipped"] == 0

    hard = np.clip(over, -1.0, 1.0)
    assert not np.allclose(hard, over, atol=1e-6)         # clipping destroys it


def test_limiter_is_a_no_op_when_there_is_headroom():
    quiet = _noise(1.0, 0.1, seed=10)
    out, gain = safe_limit(quiet)
    assert gain == 1.0
    assert out is quiet


# --- edge-weighted splice matching ------------------------------------------

def test_the_edges_decide_the_gain_not_the_whole_span():
    """A dynamic span makes the mean the wrong average.

    Whole-span RMS closes the gap between the two *means*, which can leave both
    ends visibly off -- and the ends are where a splice is heard. Three of the
    worst joins on the measured corpus sat inside spans holding 22-35 dB of
    internal range.
    """
    sr = 24000
    rng = np.random.default_rng(0)

    # Host: quiet at both ends, loud in the middle.
    host = rng.standard_normal(sr) * 0.02
    host[sr // 3: 2 * sr // 3] *= 12.0
    # Patch: the same shape at half the level, so the edges need a gain of 2.
    patch = host * 0.5

    matched = match_splice_level(host, patch)
    edge = slice(0, 480)
    ratio = (np.sqrt(np.mean(host[edge] ** 2))
             / np.sqrt(np.mean(matched[edge] ** 2)))
    assert abs(20 * np.log10(ratio)) < 1.0, "edges should line up"


def test_a_uniform_span_is_unaffected_by_the_change():
    """With no internal dynamics the edge gain and the mean gain agree."""
    sr = 24000
    host = np.random.default_rng(1).standard_normal(sr) * 0.1
    matched = match_splice_level(host, host * 0.4)
    assert np.allclose(matched, host, rtol=0.05)


def test_a_silent_edge_falls_back_to_the_whole_span():
    """At a genuine speech onset the step is the recording, not an artifact."""
    sr = 24000
    host = np.zeros(sr)
    host[sr // 2:] = np.random.default_rng(2).standard_normal(sr // 2) * 0.1
    patch = host * 0.5
    matched = match_splice_level(host, patch)
    # Still scaled towards the host, via the whole-span figure.
    assert np.sqrt(np.mean(matched ** 2)) > np.sqrt(np.mean(patch ** 2))


def test_the_gain_stays_within_max_adjust():
    """A loud consonant at one edge may not set the level for everything."""
    sr = 24000
    rng = np.random.default_rng(3)
    host = rng.standard_normal(sr) * 0.5
    patch = rng.standard_normal(sr) * 0.001          # needs 500x
    matched = match_splice_level(host, patch, max_adjust=3.0)
    gain = np.sqrt(np.mean(matched ** 2)) / np.sqrt(np.mean(patch ** 2))
    assert gain <= 3.0 + 1e-6


def test_a_span_too_short_to_have_edges_still_returns_audio():
    sr = 24000
    host = np.random.default_rng(4).standard_normal(200) * 0.1
    out = match_splice_level(host, host * 0.5)
    assert len(out) == len(host)
    assert np.all(np.isfinite(out))
