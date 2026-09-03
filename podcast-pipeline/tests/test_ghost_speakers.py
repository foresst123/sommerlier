"""Two fixes for overlaps that separation refused for the wrong reason.

On one 41-minute two-person podcast, eleven of thirteen unseparable overlaps
traced back to a single cause: the diarizer emitted a third speaker holding 3.2
seconds across the whole recording. That speaker was never in any overlap, but

  * it could not be enrolled (under 1.5s of clean audio), and
  * it fell inside the *window* of a merged job, which was tested for a third
    voice and rejected.

Run:  python -m pytest tests/test_ghost_speakers.py -q     (from podcast-pipeline/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.segment_utils import merge_ghost_speakers


def _seg(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def _shares(segments):
    held = {}
    for s in segments:
        held[s["speaker"]] = held.get(s["speaker"], 0.0) + (s["end"] - s["start"])
    return held


# --- dissolving a ghost -----------------------------------------------------

def test_a_three_second_speaker_in_an_hour_is_dissolved():
    """The case this was written for, at the measured proportions."""
    # The real proportions: 2.3s of ghost against ~41 minutes of conversation.
    segments = ([_seg(i * 10.0, i * 10.0 + 8.0, "1") for i in range(250)]
                + [_seg(i * 10.0 + 8.0, i * 10.0 + 9.5, "2") for i in range(250)]
                + [_seg(100.1, 101.3, "0"), _seg(102.2, 103.3, "0")])
    merged = merge_ghost_speakers(segments)
    assert "0" not in _shares(merged)
    assert len(merged) == len(segments), "segments are relabelled, never dropped"


def test_a_ghost_joins_whoever_was_speaking_around_it():
    segments = ([_seg(0.0, 60.0, "1"), _seg(60.0, 120.0, "2")]
                + [_seg(300.0, 360.0, "1"), _seg(360.0, 420.0, "2")]
                + [_seg(119.0, 119.6, "0")])
    merged = merge_ghost_speakers(segments)
    ghost = [s for s in merged if 119.0 <= s["start"] < 119.7][0]
    assert ghost["speaker"] in ("1", "2")


def test_total_speech_time_is_unchanged():
    segments = ([_seg(i * 10.0, i * 10.0 + 9.0, "1") for i in range(40)]
                + [_seg(i * 10.0 + 9.0, i * 10.0 + 9.8, "2") for i in range(40)]
                + [_seg(55.0, 56.0, "0")])
    before = sum(s["end"] - s["start"] for s in segments)
    after = sum(s["end"] - s["start"] for s in merge_ghost_speakers(segments))
    assert abs(before - after) < 1e-9


# --- and not dissolving anyone else -----------------------------------------

def test_a_quiet_but_real_participant_is_left_alone():
    """Someone who speaks little still holds whole turns; a ghost holds scraps."""
    segments = ([_seg(i * 10.0, i * 10.0 + 8.0, "1") for i in range(30)]
                + [_seg(i * 10.0 + 8.0, i * 10.0 + 9.5, "2") for i in range(30)]
                + [_seg(500.0, 512.0, "3")])          # one 12s turn
    assert "3" in _shares(merge_ghost_speakers(segments))


def test_two_speakers_are_never_reduced_to_one():
    """With two speakers the quieter one is a participant, however quiet."""
    segments = [_seg(0.0, 600.0, "1"), _seg(600.0, 600.9, "2")]
    assert set(_shares(merge_ghost_speakers(segments))) == {"1", "2"}


def test_a_balanced_conversation_is_untouched():
    segments = [_seg(i * 10.0, i * 10.0 + 9.0, str(i % 3)) for i in range(60)]
    assert merge_ghost_speakers(segments) == sorted(segments, key=lambda s: s["start"])


def test_an_empty_list_is_returned_as_is():
    assert merge_ghost_speakers([]) == []


# --- the window test that rejected two-speaker overlaps ---------------------

def _service():
    import services.separation_service as sep
    svc = sep.TargetExtractionService.__new__(sep.TargetExtractionService)
    svc.logger = None
    svc.music_map = None
    return svc


def test_a_third_voice_between_overlaps_no_longer_blocks_the_job():
    """Measured case: eight two-speaker overlaps across 43s, rejected because a
    third speaker said something once, in a gap between them."""
    svc = _service()
    by_spk = {
        "1": [(820.0, 880.0)],
        "2": [(831.4, 831.9), (874.2, 874.3)],
        "0": [(840.1, 841.3)],          # between the overlaps, not in one
    }
    overlaps = [(831.4, 831.9), (874.2, 874.3)]
    built, reason = svc._build_window(by_spk, "1", "2", 831.4, 874.3, 2978.0,
                                      overlaps=overlaps)
    assert reason != "multi_speaker"


def test_a_third_voice_inside_an_overlap_still_blocks_it():
    """Sidon emits two sources; three in the span really is unseparable."""
    svc = _service()
    by_spk = {
        "1": [(820.0, 880.0)],
        "2": [(850.0, 851.0)],
        "0": [(850.2, 850.8)],          # inside the overlap
    }
    built, reason = svc._build_window(by_spk, "1", "2", 850.0, 851.0, 2978.0,
                                      overlaps=[(850.0, 851.0)])
    assert reason == "multi_speaker"


def test_the_hull_is_used_when_no_overlaps_are_given():
    """Callers that pass no span list keep the old, conservative behaviour."""
    svc = _service()
    by_spk = {"1": [(0.0, 100.0)], "2": [(10.0, 11.0)], "0": [(50.0, 51.0)]}
    built, reason = svc._build_window(by_spk, "1", "2", 10.0, 60.0, 200.0)
    assert reason == "multi_speaker"
