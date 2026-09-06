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

from utils.music_map import MUSIC, SINGING, SONG, MusicMap, build


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

def test_enrollment_is_never_mined_from_speech_over_music():
    """The behaviour this module was written for.

    mine_enrollments picks on clip length alone, so without this filter a five
    second stretch of speech over a music bed is a perfectly good candidate --
    and USEF is then conditioned on the speaker plus the backing track. It is
    conditioned on that audio and nothing else, so a contaminated enrollment is
    a contaminated extraction.

    The guard used to live on the stitched window's solo picker, which went
    with DialogueSidon. It has to sit on the mining now, because that is the
    only place an enrollment comes from.
    """
    import numpy as np
    import services.separation_service as sep
    from schemas.audio import AudioData
    from schemas.segment import Segment

    sr = 16000
    service = sep.TargetExtractionService.__new__(sep.TargetExtractionService)
    service.music_map = MusicMap([(100.0, 110.0)])
    service.logger = None

    audio = AudioData(name="t", waveform=np.ones(120 * sr, dtype=np.float32),
                      sample_rate=sr, duration=120.0, audio_segment=None)
    segments = [
        Segment(index="1", start=90.0, end=96.0, speaker="A"),    # clean
        Segment(index="2", start=101.0, end=108.0, speaker="A"),  # buried in music
        Segment(index="3", start=108.0, end=115.0, speaker="A"),  # straddles the edge
    ]
    picked = service.mine_enrollments(segments, audio)["A"]
    assert picked, "the clean stretches should still be usable"

    total = sum(len(c) for c in picked) / sr
    clean = sum(b - a for s in segments
                for a, b in service.music_map.clean_parts(s.start, s.end))
    assert total <= clean + 0.05, (
        f"mined {total:.2f}s but only {clean:.2f}s is clear of the bed")


# --- the two thresholds, against what the tagger actually produces -----------
#
# Both numbers below were set from a full 527-label dump of three recordings
# (tools/dump_panns.py, data/panns/). Before that they were guesses, and both
# guesses were wrong in the same way: set past what the model can produce, so
# the code they guarded never ran.

def test_the_singing_level_is_reachable_by_this_tagger():
    """0.35 sat above the ceiling of the singing group -- measured at 0.205 on
    the one recording that contains a real song -- so the branch never fired
    once and every sung stretch fell through to SONG."""
    from utils.music_map import SINGING_THRESHOLD
    measured_ceiling = 0.205
    false_positive_floor = 0.114     # "Male singing" on ordinary speech
    assert false_positive_floor < SINGING_THRESHOLD < measured_ceiling


def test_singing_at_the_measured_level_is_caught():
    """At 0.35 none of the real song cleared the bar; at 0.12, 10.2s does.

    Speech is near zero here because that is what the recording looks like:
    SINGING_MARGIN still requires singing to lead speech by 0.15, and with the
    singing group topping out at 0.205 that leaves room only where nobody is
    talking. Lowering the level revived the branch; the margin is what now
    bounds it."""
    detector = FrameDetector(speech=0.02, singing=0.18, music=0.6)
    found = build(_audio(), 24000, detector)
    assert found.total_of(SINGING) > 0


def test_the_margin_is_now_the_binding_constraint_not_the_level():
    """Worth pinning because it changes which knob to reach for next: on the
    one file with real singing, the level admits 23.0s and the margin cuts
    that to 10.2s."""
    from utils.music_map import SINGING_MARGIN, SINGING_THRESHOLD
    measured_ceiling = 0.205
    assert SINGING_MARGIN > measured_ceiling - SINGING_THRESHOLD - 0.01, (
        "the margin no longer binds; re-measure before assuming it does")


def test_ordinary_speech_that_brushes_the_singing_label_is_not_cut():
    """vimeanhphanchiatay peaks at 0.114 on "Male singing" with nobody
    singing. Cutting there would delete conversation."""
    detector = FrameDetector(speech=0.9, singing=0.11, music=0.6)
    found = build(_audio(), 24000, detector)
    assert found.total_of(SINGING) == 0
    assert found.total_of(MUSIC) > 0


# --- deleting audio asks for more evidence than cleaning it -----------------

def test_cutting_demands_a_longer_run_than_stripping():
    """The two decisions are not equally reversible: MUSIC writes vocals back
    over a bed, SINGING and SONG delete the audio for good."""
    from utils.music_map import MIN_SPAN_EXCISED, MIN_SPAN_SECONDS
    assert MIN_SPAN_EXCISED > MIN_SPAN_SECONDS


def test_the_stripping_level_still_admits_a_single_decision_block():
    """Cnn14_DecisionLevelMax decides once per 320ms. Asking MUSIC for more
    than one block throws away real beds -- 10 spans and 6.1s of them on
    vimeanhphanchiatay."""
    from utils.music_map import MIN_SPAN_SECONDS
    assert MIN_SPAN_SECONDS <= 0.32


def test_the_cutting_level_is_a_whole_number_of_decision_blocks():
    """A level between blocks cannot be reached exactly, so it silently
    behaves as the next block up and the written number misleads."""
    from utils.music_map import MIN_SPAN_EXCISED
    blocks = MIN_SPAN_EXCISED / 0.32
    assert abs(blocks - round(blocks)) < 1e-6, f"{MIN_SPAN_EXCISED} is not n x 320ms"
    assert round(blocks) >= 3


def test_a_single_block_of_song_is_not_enough_to_delete_audio():
    """Nine one-block spans appeared across three recordings, each grown to
    0.92s by padding. Deleting a second of a recording on one 320ms decision
    is not a trade worth making."""
    frames = np.zeros(500, dtype=np.float32)
    frames[100:132] = 0.9                       # exactly one 320ms block
    detector = FrameDetector(speech=0.02, singing=0.0, music=frames)
    found = build(_audio(), 24000, detector)
    assert found.total_of(SONG) == 0


def test_a_single_block_of_music_under_speech_is_still_stripped():
    """The same run on the non-destructive path survives: cleaning a bed that
    was not there costs a separator pass, not a piece of the recording."""
    frames = np.zeros(500, dtype=np.float32)
    frames[100:132] = 0.9
    detector = FrameDetector(speech=0.9, singing=0.0, music=frames)
    found = build(_audio(), 24000, detector)
    assert found.total_of(MUSIC) > 0


def test_three_blocks_of_song_do_clear_the_bar():
    frames = np.zeros(500, dtype=np.float32)
    frames[100:196] = 0.9                       # three 320ms blocks
    detector = FrameDetector(speech=0.02, singing=0.0, music=frames)
    found = build(_audio(), 24000, detector)
    assert found.total_of(SONG) > 0
