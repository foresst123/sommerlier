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

from utils.music_map import MUSIC, SINGING, MusicMap, build


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


# --- building it -----------------------------------------------------------

class FrameDetector:
    """Stands in for PANNs SED: returns the scores it was handed."""

    def __init__(self, speech=None, singing=None, music=None, frames=500, fps=100.0):
        def _arr(v):
            return np.full(frames, v, dtype=np.float32) if np.isscalar(v) else np.asarray(v, np.float32)
        self.scores = {"speech": _arr(speech if speech is not None else 0.0),
                       "singing": _arr(singing if singing is not None else 0.0),
                       "music": _arr(music if music is not None else 0.0)}
        self.fps = fps
        self.calls = 0

    def tag_framewise(self, waveform, sample_rate):
        self.calls += 1
        return self.scores, self.fps


def _audio(seconds=5.0, sr=24000):
    return np.zeros(int(sr * seconds), dtype=np.float32)


def test_no_detector_gives_an_empty_map():
    """PANNs off is not the same as an error; it means the check did not run."""
    assert not build(_audio(), 24000, None)


def test_a_detector_without_frame_tagging_is_reported_not_guessed_at():
    class Old:
        def detect_music(self, audio, sr):
            return True, 0.9
    assert not build(_audio(), 24000, Old())


def test_a_clean_recording_has_no_spans():
    detector = FrameDetector(speech=0.9, singing=0.01, music=0.02)
    assert not build(_audio(), 24000, detector)
    assert detector.calls == 1


def test_music_under_speech_is_marked_music_not_singing():
    """The common case: a bed to remove, with someone talking over it."""
    detector = FrameDetector(speech=0.9, singing=0.05, music=0.8)
    found = build(_audio(), 24000, detector)
    assert found.total_of(MUSIC) > 0
    assert found.total_of(SINGING) == 0


def test_a_sung_stretch_is_marked_singing():
    """Singing must be skipped, not cleaned: the thing to remove is the voice."""
    detector = FrameDetector(speech=0.2, singing=0.9, music=0.8)
    found = build(_audio(), 24000, detector)
    assert found.total_of(SINGING) > 0
    assert found.total_of(MUSIC) == 0, "a sung frame is not also a bed to strip"


def test_speech_over_music_is_not_mistaken_for_singing():
    """Both light up the vocal range; singing has to lead by a margin.

    Without it, ordinary speech with a bed under it reads as singing and gets
    dropped from the transcript."""
    detector = FrameDetector(speech=0.85, singing=0.40, music=0.7)
    found = build(_audio(), 24000, detector)
    assert found.total_of(SINGING) == 0
    assert found.total_of(MUSIC) > 0


def test_a_brief_flicker_is_not_a_span():
    frames = np.zeros(500, dtype=np.float32)
    frames[100:110] = 0.9              # 0.1s, under MIN_SPAN_SECONDS
    found = build(_audio(), 24000, FrameDetector(music=frames))
    assert not found


def test_a_dip_below_the_threshold_does_not_split_one_stretch():
    """Music dips on a beat rest without stopping."""
    frames = np.zeros(500, dtype=np.float32)
    frames[100:200] = 0.9
    frames[200:220] = 0.0              # 0.2s gap, under MERGE_GAP_SECONDS
    frames[220:320] = 0.9
    found = build(_audio(), 24000, FrameDetector(music=frames))
    assert len(found) == 1


def test_the_frame_rate_survives_into_the_map():
    found = build(_audio(), 24000, FrameDetector(music=0.9, fps=50.0))
    assert found.fps == 50.0


def test_a_detector_that_raises_does_not_take_the_run_down():
    class Broken:
        def tag_framewise(self, waveform, sample_rate):
            raise RuntimeError("model died")
    assert not build(_audio(), 24000, Broken())


def test_empty_scores_are_not_a_division_by_zero():
    detector = FrameDetector(frames=0)
    assert not build(_audio(), 24000, detector)


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
