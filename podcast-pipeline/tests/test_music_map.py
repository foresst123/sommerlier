"""Keeping music out of the audio TSE enrols speakers on.

Separation runs before music removal, so its search for clean solo speech sees
the music bed that music removal has not taken out yet. An ECAPA embedding
built from speech-over-music describes both, which is not what the assignment
is matching against.

Run:  python -m pytest tests/test_music_map.py -q     (from podcast-pipeline/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.music_map import MusicMap, build


class FakeDetector:
    """Reports music wherever the sample values are above a marker level."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.calls = 0

    def detect_music(self, audio, sample_rate):
        self.calls += 1
        loud = float(np.abs(audio).mean()) > self.threshold
        return loud, 0.9 if loud else 0.1


# --- querying the map -------------------------------------------------------

def test_an_empty_map_never_blocks_anything():
    """No detector, or no music: both mean nothing to avoid."""
    empty = MusicMap()
    assert not empty
    assert not empty.overlaps(0.0, 100.0)
    assert empty.clean_parts(2.0, 5.0) == [(2.0, 5.0)]


def test_a_span_clear_of_the_music_is_left_whole():
    music = MusicMap([(10.0, 20.0)])
    assert music.clean_parts(0.0, 5.0) == [(0.0, 5.0)]
    assert not music.overlaps(0.0, 5.0)


def test_a_span_inside_the_music_disappears():
    music = MusicMap([(10.0, 20.0)])
    assert music.clean_parts(12.0, 15.0) == []
    assert music.overlaps(12.0, 15.0)


def test_a_span_straddling_the_music_keeps_its_clean_half():
    music = MusicMap([(10.0, 20.0)])
    assert music.clean_parts(5.0, 15.0) == [(5.0, 10.0)]
    assert music.clean_parts(15.0, 25.0) == [(20.0, 25.0)]


def test_music_in_the_middle_splits_a_span_in_two():
    music = MusicMap([(10.0, 12.0)])
    assert music.clean_parts(8.0, 15.0) == [(8.0, 10.0), (12.0, 15.0)]


def test_touching_music_at_the_boundary_is_not_an_overlap():
    """A span ending exactly where music starts is still clean."""
    music = MusicMap([(10.0, 20.0)])
    assert not music.overlaps(5.0, 10.0)
    assert not music.overlaps(20.0, 25.0)


def test_several_musical_stretches_are_all_removed():
    music = MusicMap([(2.0, 3.0), (6.0, 7.0)])
    assert music.clean_parts(0.0, 10.0) == [(0.0, 2.0), (3.0, 6.0), (7.0, 10.0)]


# --- building it ------------------------------------------------------------

def test_no_detector_gives_an_empty_map():
    """PANNs off is not the same as an error; it means the check did not run."""
    assert not build(np.zeros(48000), 24000, None)


def test_a_silent_recording_has_no_music():
    detector = FakeDetector()
    assert not build(np.zeros(24000 * 5), 24000, detector)
    assert detector.calls > 0, "the sweep should have run"


def test_music_is_found_where_it_plays():
    sr = 24000
    audio = np.full(sr * 10, 0.1)
    audio[sr * 4:sr * 6] = 1.0          # two loud seconds
    found = build(audio, sr, FakeDetector())
    assert found
    assert found.overlaps(4.5, 5.5)


def test_adjacent_detections_merge_into_one_stretch():
    sr = 24000
    audio = np.full(sr * 10, 0.1)
    audio[sr * 2:sr * 8] = 1.0
    found = build(audio, sr, FakeDetector())
    # Six seconds of music should read as one stretch, not six.
    assert len(found) == 1


def test_a_detector_that_raises_does_not_take_the_run_down():
    class Broken:
        def detect_music(self, audio, sample_rate):
            raise RuntimeError("model died")

    assert not build(np.ones(24000 * 4), 24000, Broken())


def test_a_recording_shorter_than_one_window_is_not_guessed_at():
    """PANNs needs about a second; below that its verdict means nothing."""
    detector = FakeDetector()
    assert not build(np.ones(1000), 24000, detector)
    assert detector.calls == 0


# --- surviving a checkpoint -------------------------------------------------

def test_the_map_round_trips_through_json():
    original = MusicMap([(1.5, 2.25), (9.0, 11.5)])
    revived = MusicMap.from_json(original.to_json())
    assert revived.spans == original.spans
    assert revived.overlaps(10.0, 10.5)


def test_a_missing_checkpoint_reads_as_no_music():
    assert not MusicMap.from_json(None)
    assert not MusicMap.from_json({})


# --- the reason all of the above exists -------------------------------------

def test_solo_spans_over_music_are_not_offered_for_enrollment():
    """The behaviour this module was written for.

    _pick_solo chooses on length and proximity alone, so without this filter a
    five-second stretch of speech over a music bed is a perfectly good
    candidate -- and Sidon is then enrolled on the speaker plus the backing
    track.
    """
    import services.separation_service as sep

    service = sep.TargetExtractionService.__new__(sep.TargetExtractionService)
    service.music_map = MusicMap([(100.0, 110.0)])
    service.logger = None

    # One clean stretch, one buried in music, one straddling the edge.
    by_spk = {"1": [(90.0, 96.0), (101.0, 108.0), (108.0, 115.0)],
              "2": [(0.0, 1.0)]}
    picked, total = service._pick_solo(by_spk, "1", centre=105.0, want=5.0)

    for start, end in picked:
        assert not service.music_map.overlaps(start, end), \
            f"picked {start:.1f}-{end:.1f}s which has music under it"
    assert total > 0, "the clean stretches should still be usable"
