"""Boundary placement in the diarization post-processing chain.

The diarizer decides who speaks; these functions decide where the edges land.
A forced split that ignores the audio cuts mid-word, and a clipped word is a
transcription error the recogniser cannot recover from.

Run:  python -m pytest tests/test_segment_boundaries.py -q   (from podcast-pipeline/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.segment_utils import _quietest_cut, split_long_segments

SR = 24000


def _speech_with_pauses(dur, pauses, seed=0):
    """Continuous noise with short near-silent gaps at `pauses` (seconds)."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal(int(dur * SR)) * 0.1).astype(np.float32)
    for t in pauses:
        w[int((t - 0.15) * SR):int((t + 0.15) * SR)] *= 0.001
    return w


def _rms_at(w, t, win=0.02):
    i = int((t - win / 2) * SR)
    return float(np.sqrt(np.mean(w[max(0, i):i + int(win * SR)] ** 2) + 1e-12))


def test_split_lands_on_a_pause_instead_of_the_stopwatch():
    w = _speech_with_pauses(50.0, (18.6, 38.4))
    seg = [{"start": 0.0, "end": 50.0, "speaker": "1", "index": "00000"}]

    blind = split_long_segments([dict(seg[0])], max_duration=20.0)
    aware = split_long_segments([dict(seg[0])], max_duration=20.0,
                                waveform=w, sample_rate=SR)

    assert abs(blind[0]["end"] - 20.0) < 1e-6          # exactly the deadline
    assert abs(aware[0]["end"] - 18.6) < 0.2           # the actual pause
    # And the chosen point is genuinely quiet, not merely different.
    assert _rms_at(w, aware[0]["end"]) < _rms_at(w, blind[0]["end"]) / 5


def test_split_never_exceeds_the_deadline_and_stays_gapless():
    w = _speech_with_pauses(50.0, (18.6, 38.4))
    out = split_long_segments(
        [{"start": 0.0, "end": 50.0, "speaker": "1", "index": "00000"}],
        max_duration=20.0, waveform=w, sample_rate=SR)

    assert all(s["end"] - s["start"] <= 20.0 + 1e-6 for s in out)
    assert out[0]["start"] == 0.0
    assert abs(out[-1]["end"] - 50.0) < 1e-6
    for a, b in zip(out, out[1:]):
        assert abs(a["end"] - b["start"]) < 1e-6      # contiguous, no overlap


def test_splitting_without_audio_still_works():
    """The waveform is optional; callers without it must not break."""
    out = split_long_segments(
        [{"start": 0.0, "end": 45.0, "speaker": "1", "index": "00000"}],
        max_duration=20.0)
    assert [round(s["end"] - s["start"], 3) for s in out] == [20.0, 20.0, 5.0]


def test_a_segment_under_the_limit_is_untouched():
    seg = {"start": 3.0, "end": 9.0, "speaker": "1", "index": "00007"}
    out = split_long_segments([dict(seg)], max_duration=20.0,
                              waveform=_speech_with_pauses(12.0, (5.0,)),
                              sample_rate=SR)
    assert len(out) == 1
    assert (out[0]["start"], out[0]["end"]) == (3.0, 9.0)


def test_quietest_cut_finds_the_pause_and_degrades_safely():
    w = _speech_with_pauses(10.0, (6.0,))
    t = _quietest_cut(w, SR, 4.0, 8.0)
    assert t is not None and abs(t - 6.0) < 0.2

    assert _quietest_cut(None, SR, 0.0, 5.0) is None
    assert _quietest_cut(w, SR, 5.0, 5.0) is None      # empty band
    assert _quietest_cut(w, SR, 0.0, 0.001) is None    # shorter than a frame
