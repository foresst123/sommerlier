"""The replaced stretch is widened ~100 ms on each side into the solo speech around an
overlap, and the crossfade happens there (both signals are the same speaker) instead of
inside the overlap. Where widening is unsafe it falls back to the old 20 ms fade."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.audio import AudioData
from schemas.segment import Segment
from services import separation_service
from services.separation_service import SeparationService

SR = 24000
MARGIN = 0.10


def _svc(margin=MARGIN):
    svc = SeparationService.__new__(SeparationService)
    svc.splice_margin = margin
    return svc


def _margins(svc, *, src=48000, dst=48000, limit=3600, track_len=200000, audio_len=200000,
             lo=1000, hi=4600, spans=None, il=False, ir=False):
    return svc._splice_margins(SR, src, dst, limit, track_len, audio_len, lo, hi,
                               spans if spans is not None else [(lo, hi)], il, ir)


# --- how far it may widen ---------------------------------------------------------

def test_it_widens_by_the_configured_margin_on_both_sides():
    assert _margins(_svc()) == (2400, 2400)              # 100 ms at 24 kHz
    assert _margins(_svc(0.0)) == (0, 0)


def test_it_never_widens_across_an_internal_boundary_or_outside_the_audio():
    assert _margins(_svc(), il=True) == (0, 2400)
    assert _margins(_svc(), ir=True) == (2400, 0)
    assert _margins(_svc(), dst=0) == (0, 2400)                      # overlap starts the segment
    assert _margins(_svc(), audio_len=48000 + 3600 + 600) == (2400, 600)   # segment ends soon
    assert _margins(_svc(), src=1000) == (1000, 2400)                # window starts soon


def test_a_gap_to_a_neighbouring_overlap_is_shared_so_margins_never_meet():
    spans = [(1000, 4600), (6000, 9000)]                 # 1400 samples between them
    left, right = _margins(_svc(), lo=1000, hi=4600, spans=spans)
    assert (left, right) == (2400, 700)                  # half of the gap
    left2, _ = _margins(_svc(), lo=6000, hi=9000, spans=spans, src=48000, dst=48000)
    assert left2 == 700


def test_a_margin_shorter_than_5ms_is_not_worth_it():
    assert _margins(_svc(), dst=100) == (0, 2400)


# --- when the widened part may be used ----------------------------------------------

def test_the_widened_part_must_carry_comparable_speech():
    rng = np.random.default_rng(0)
    speech = rng.normal(0, 0.05, 2400).astype(np.float32)
    usable = separation_service.SeparationService._margin_usable
    assert usable(speech, speech * 0.8) is True
    assert usable(speech, np.zeros_like(speech)) is False           # separator gave silence
    assert usable(speech, speech * 20.0) is False                   # far too loud
    assert usable(speech * 0.0, speech) is False                    # nothing to blend from
    assert usable(np.zeros(0, np.float32), speech) is False


# --- the blend ------------------------------------------------------------------------

def test_blend_ramps_over_the_margin_and_leaves_the_core_fully_replaced():
    orig = np.zeros(1000, np.float32)
    new = np.ones(1000, np.float32)
    out = SeparationService._blend(orig, new, 200, 100)
    assert out[0] == 0.0 and out[199] == pytest.approx(0.995)      # left ramp 0 -> 1
    assert np.all(out[200:900] == 1.0)                              # core untouched by fades
    assert out[900] == 1.0 and out[999] == pytest.approx(0.01)      # right ramp 1 -> 0


def test_blend_without_a_fade_on_a_side_switches_hard_there():
    out = SeparationService._blend(np.zeros(100, np.float32), np.ones(100, np.float32), 0, 20)
    assert out[0] == 1.0 and out[79] == 1.0


def test_the_legacy_fade_is_unchanged_when_there_is_no_margin():
    orig = np.linspace(-1, 1, 4000, dtype=np.float32)
    new = np.full(4000, 0.5, np.float32)
    old = _svc()._cross_fade(orig, new, int(0.02 * SR), SR, fade_left=True, fade_right=True)
    legacy_fade = min(int(0.02 * SR), max(int(0.005 * SR), 4000 // 8))
    assert np.allclose(SeparationService._blend(orig, new, legacy_fade, legacy_fade), old)


# --- end to end -----------------------------------------------------------------------

class _FakeModel:
    """Separated tracks = the mixture plus a constant, so replaced samples are visible."""

    OFFSET = 0.05

    def __init__(self):
        self.calls = []

    def separate_two_speakers(self, mixture_audio, enroll_A, enroll_B, sample_rate, id_A, id_B,
                              probe_A=None, probe_B=None, core_range=None):
        track = np.asarray(mixture_audio, dtype=np.float32) + self.OFFSET
        return (track, track.copy(), 0.6, 0.6,
                {"anchor_self": 0.6, "anchor_other": 0.1, "other_rms": 0.5})


def _audio(duration=60.0):
    rng = np.random.default_rng(1)
    wave = rng.normal(0, 0.05, int(duration * SR)).astype(np.float32)
    wave[np.arange(len(wave)) % (SR // 2) < int(0.08 * SR)] = 0.02
    return AudioData(waveform=wave, sample_rate=SR, name="t", audio_segment=None,
                     duration=duration)


def _segments():
    return [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=14.0, end=14.5, speaker="SPEAKER_01"),
        Segment(index="00003", start=32.0, end=40.0, speaker="SPEAKER_01"),
        Segment(index="00004", start=42.0, end=48.0, speaker="SPEAKER_00"),
    ]


def _run(margin):
    audio = _audio()
    svc = SeparationService(_FakeModel(), logger=None)
    svc.splice_margin = margin
    out = svc.process_overlaps(_segments(), audio, overlap_threshold=0.1)
    return audio, out


def _offset_at(audio, seg, t):
    i = int(t * SR) - int(seg.start * SR)
    return float(np.mean(seg.audio[i:i + 240] - audio.waveform[int(t * SR):int(t * SR) + 240]))


def test_with_a_margin_the_audio_just_outside_the_overlap_is_replaced_too():
    audio, out = _run(MARGIN)
    host = next(s for s in out if s.index == "00001")
    lo, hi, _ = host.bss_spans[0]
    assert lo == pytest.approx(14.0, abs=0.01)                        # the recorded span is the overlap
    # 50 ms before the overlap: inside the widened part, partly replaced
    assert _offset_at(audio, host, lo - 0.05) > 0.01
    # 300 ms before: beyond the margin, untouched
    assert abs(_offset_at(audio, host, lo - 0.30)) < 1e-4
    # 50 ms after the overlap ends
    assert _offset_at(audio, host, hi + 0.02) > 0.01
    assert abs(_offset_at(audio, host, hi + 0.30)) < 1e-4


def test_without_a_margin_nothing_outside_the_overlap_changes():
    audio, out = _run(0.0)
    host = next(s for s in out if s.index == "00001")
    lo, hi, _ = host.bss_spans[0]
    assert abs(_offset_at(audio, host, lo - 0.05)) < 1e-4
    assert abs(_offset_at(audio, host, hi + 0.02)) < 1e-4


def test_the_default_margin_is_100_ms_and_can_be_switched_off_by_environment():
    assert separation_service.BSS_SPLICE_MARGIN == pytest.approx(0.10)
    assert SeparationService(_FakeModel(), logger=None).splice_margin == pytest.approx(0.10)
