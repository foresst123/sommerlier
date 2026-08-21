"""Sanity checks for the SI-SDR/SIR/SAR implementation in tools/eval_separation.py.

No audio, no models: these pin the maths against signals whose answer is known.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from eval_separation import _align, bss_metrics


def _sources(n=24000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n), rng.normal(0, 1, n)


def test_perfect_separation_reports_a_ceiling_not_nan():
    s1, s2 = _sources()
    sdr, sir, sar = bss_metrics(s1.copy(), s1, s2)
    assert sdr > 50 and sir > 50 and sar > 50
    assert not any(np.isnan([sdr, sir, sar]))


def test_metrics_are_scale_invariant():
    s1, s2 = _sources()
    a = bss_metrics(s1.copy(), s1, s2)
    b = bss_metrics(3.0 * s1, s1, s2)
    assert np.allclose(a, b, atol=0.1), "a gain change must not move the score"


def test_leakage_lowers_sir_monotonically():
    s1, s2 = _sources()
    sirs = [bss_metrics(s1 + k * s2, s1, s2)[1] for k in (0.05, 0.1, 0.5)]
    assert sirs[0] > sirs[1] > sirs[2]


def test_emitting_the_wrong_speaker_is_caught_by_sir_not_sar():
    """Sidon's known failure mode: a clean copy of the wrong speaker.

    SAR stays high because nothing was invented; only SIR exposes it. A quality
    check that reads SAR alone would call this a success.
    """
    s1, s2 = _sources()
    sdr, sir, sar = bss_metrics(s2.copy(), s1, s2)
    assert sir < -20, f"SIR {sir:.1f} should be strongly negative"
    assert sar > 50, f"SAR {sar:.1f} stays high: no artefacts, just the wrong source"


def test_invented_audio_lowers_sar():
    s1, s2 = _sources()
    noise = np.random.default_rng(7).normal(0, 1, len(s1))
    assert bss_metrics(noise, s1, s2)[2] < 0


def test_alignment_recovers_a_decoder_time_shift():
    s1, s2 = _sources()
    shifted = np.concatenate([np.zeros(120), s1[:-120]])
    before = bss_metrics(shifted, s1, s2)[0]
    after = bss_metrics(_align(shifted, s1, 240), s1, s2)[0]
    assert before < 0 < after, (
        f"5ms of decoder latency scored {before:.1f}dB unaligned and {after:.1f}dB "
        "aligned; without alignment SI-SDR punishes latency as separation error"
    )
