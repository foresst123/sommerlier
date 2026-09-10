"""Tests for post-separation verification and repair.

No models: DNSMOS is injected as a callable, so the decision logic and the
signal repairs are pinned without a GPU.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.verification_service import (
    blend, decide, highpass, remove_dc, restore_gain, rms, verify_span,
)

SR = 24000


def _tone(freq, n=SR, amp=0.2, sr=SR):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _scorer(sig, bak):
    return lambda audio, sr: {"SIG": sig, "BAK": bak, "OVRL": (sig + bak) / 2}


# --- repairs -------------------------------------------------------------

def test_dc_offset_is_removed():
    x = _tone(200) + 0.05
    assert abs(np.mean(remove_dc(x))) < 1e-6


def test_highpass_attenuates_rumble_and_keeps_voice():
    rumble, voice = _tone(20, amp=0.5), _tone(300, amp=0.5)
    out = highpass(rumble + voice, SR)
    kept = rms(highpass(voice, SR)) / rms(voice)
    killed = rms(highpass(rumble, SR)) / rms(rumble)
    assert kept > 0.8, f"voice band lost {1 - kept:.0%}"
    assert killed < kept / 2, f"rumble {killed:.2f} vs voice {kept:.2f}"


def test_common_gain_restores_the_mixture_level():
    """Sidon peak-normalises each 20s chunk, so both tracks come back at an
    arbitrary level and the spliced seam becomes a step change."""
    loud, quiet = _tone(300, amp=0.20), _tone(700, amp=0.02)
    mixture = loud + quiet
    # The separator returns both at 1/4 of the original level.
    sep_loud, sep_quiet = loud * 0.25, quiet * 0.25
    out = restore_gain(sep_loud, sep_loud + sep_quiet, mixture)
    assert abs(rms(out) - rms(loud)) / rms(loud) < 0.05


def test_a_quiet_backchannel_keeps_its_relative_level():
    """The point of one common gain: matching the quiet track to its loud
    neighbour would drag a backchannel up to the main speaker's level."""
    loud, quiet = _tone(300, amp=0.20), _tone(700, amp=0.02)
    mixture = loud + quiet
    sep_loud, sep_quiet = loud * 0.25, quiet * 0.25
    out = restore_gain(sep_quiet, sep_loud + sep_quiet, mixture)
    ratio = rms(out) / rms(loud)
    assert 0.05 < ratio < 0.15, (
        f"backchannel came back at {ratio:.2f} of the main speaker; it started "
        "at 0.10 and must not be normalised up to match it"
    )


def test_gain_will_not_amplify_near_silence():
    silence = np.zeros(SR, dtype=np.float32) + 1e-9
    out = restore_gain(silence, silence, _tone(300, amp=0.2))
    assert np.abs(out).max() < 0.01, "an empty track must not be gained up into noise"


def test_gain_is_clamped():
    tiny = _tone(300, amp=0.001)
    out = restore_gain(tiny, tiny, _tone(300, amp=0.5), max_gain_db=12.0)
    assert rms(out) / rms(tiny) <= 10 ** (12.0 / 20.0) + 1e-3


def test_blend_endpoints():
    a, b = _tone(300), _tone(700)
    assert np.allclose(blend(a, b, 1.0), a)
    assert np.allclose(blend(a, b, 0.0), b)
    assert np.allclose(blend(a, b, 0.5), 0.5 * a + 0.5 * b, atol=1e-6)


# --- decision ------------------------------------------------------------

def test_clear_improvement_is_kept():
    action, _ = decide({"SIG": 3.0, "BAK": 2.0}, {"SIG": 3.0, "BAK": 3.5})
    assert action == "keep"


def test_no_interference_removed_reverts():
    """The separator returned something, but the other speaker is just as loud.
    Whatever artefacts came with it bought nothing."""
    action, detail = decide({"SIG": 3.0, "BAK": 3.0}, {"SIG": 2.6, "BAK": 3.0})
    assert action == "revert", detail


def test_big_quality_cost_blends_instead_of_all_or_nothing():
    action, detail = decide({"SIG": 3.5, "BAK": 2.0}, {"SIG": 2.6, "BAK": 3.4})
    assert action == "blend", detail


# --- orchestration -------------------------------------------------------

def test_verify_span_reverts_to_the_original_audio():
    sep, orig = _tone(700, amp=0.2), _tone(300, amp=0.2)
    scores = iter([{"SIG": 3.0, "BAK": 3.0}, {"SIG": 2.9, "BAK": 3.0}])
    out, action, _ = verify_span(sep, orig, SR, scorer=lambda a, s: next(scores))
    assert action == "revert"
    assert np.allclose(out, orig), "revert must return the mixture untouched"


def test_verify_span_without_a_scorer_still_repairs():
    sep = _tone(300, amp=0.05) + 0.05          # quiet and DC-shifted
    orig = _tone(300, amp=0.20)
    out, action, _ = verify_span(sep, orig, SR, scorer=None)
    assert action == "keep"
    assert abs(np.mean(out)) < 1e-3, "DC should be gone even with no scorer"
    assert rms(out) > rms(remove_dc(sep)) * 1.5, "level should have been restored upward"


def test_scorer_failure_does_not_lose_the_separation():
    def boom(audio, sr):
        raise RuntimeError("onnx session died")

    sep, orig = _tone(700, amp=0.2), _tone(300, amp=0.2)
    out, action, detail = verify_span(sep, orig, SR, scorer=boom)
    assert action == "keep" and "scorer failed" in detail
    assert len(out) == len(sep)
