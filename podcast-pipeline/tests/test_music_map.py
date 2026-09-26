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

from utils.music_map import MUSIC, SONG, MusicMap, build


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
    """Stands in for a frame-level tagger: returns the scores it was handed."""

    def __init__(self, speech=None, music=None, frames=500, fps=100.0):
        def _arr(v):
            return np.full(frames, v, dtype=np.float32) if np.isscalar(v) else np.asarray(v, np.float32)
        self.scores = {"speech": _arr(speech if speech is not None else 0.0),
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
    detector = FrameDetector(speech=0.9, music=0.02)
    assert not build(_audio(), 24000, detector)
    assert detector.calls == 1


def test_music_under_speech_is_stripped_not_cut():
    """The common case: a bed to remove, with someone talking over it."""
    detector = FrameDetector(speech=0.9, music=0.8)
    found = build(_audio(), 24000, detector)
    assert found.total_of(MUSIC) > 0
    assert found.total_of(SONG) == 0


def test_music_with_nobody_talking_is_cut():
    """An intro or a sting. Handing it to a vocal separator would ask the model
    to invent a voice out of an instrumental, so it leaves instead."""
    detector = FrameDetector(speech=0.02, music=0.8)
    found = build(_audio(), 24000, detector)
    assert found.total_of(SONG) > 0
    assert found.total_of(MUSIC) == 0, "a frame is one kind or the other"


def test_the_two_kinds_are_exclusive_and_split_on_speech():
    """SPEECH_PRESENT is the whole boundary between cleaning and deleting, so
    it is worth pinning that nothing lands in both."""
    from utils.music_map import SPEECH_PRESENT
    for speech in (0.0, SPEECH_PRESENT - 0.01, SPEECH_PRESENT, 0.9):
        found = build(_audio(), 24000, FrameDetector(speech=speech, music=0.8))
        assert not (found.total_of(MUSIC) and found.total_of(SONG))


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
    service = sep.SeparationService.__new__(sep.SeparationService)
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
# --- the singing branch, and why it is gone ---------------------------------

def test_nothing_reintroduces_a_singing_kind():
    """There used to be a third kind for a voice that sings rather than speaks.

    It is gone, and this pins that rather than trusting a comment. Under PANNs
    it fired for 10s, 0s and 0s across three recordings; under SSLAM every
    detection that cleared the speech margin was a single isolated frame, so no
    run ever reached the two consecutive frames MIN_SPAN_EXCISED asks for and
    the branch could not produce a span at all. A branch that cannot fire is
    worse than no branch: it reads as a working safeguard.
    """
    import utils.music_map as mm
    assert not hasattr(mm, "SINGING")
    assert not hasattr(mm, "SINGING_THRESHOLD")
    assert not hasattr(mm, "SINGING_MARGIN")
    assert mm.EXCISED == (mm.SONG,)


def test_the_singing_labels_stay_out_of_the_music_group():
    """Folding them in was the obvious repair and it was measured, not assumed:
    on vimeanhphanchiatay it flags 47 more seconds, 39.5 of which carry speech
    -- "Male singing" and "Humming" firing on ordinary Vietnamese speech. It
    would send that through a separator with nothing to remove and cut the
    rest."""
    from models.audioset import MUSIC_LABELS, SINGING_LABELS
    assert not set(MUSIC_LABELS) & set(SINGING_LABELS)


def test_ordinary_speech_that_brushes_a_singing_label_is_not_cut():
    """vimeanhphanchiatay peaks at 0.114 on "Male singing" with nobody singing.
    Since that label no longer routes anywhere, speech over a bed is a bed."""
    found = build(_audio(), 24000, FrameDetector(speech=0.9, music=0.6))
    assert found.total_of(SONG) == 0
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
    detector = FrameDetector(speech=0.02, music=frames)
    found = build(_audio(), 24000, detector)
    assert found.total_of(SONG) == 0


def test_a_single_block_of_music_under_speech_is_still_stripped():
    """The same run on the non-destructive path survives: cleaning a bed that
    was not there costs a separator pass, not a piece of the recording."""
    frames = np.zeros(500, dtype=np.float32)
    frames[100:132] = 0.9
    detector = FrameDetector(speech=0.9, music=frames)
    found = build(_audio(), 24000, detector)
    assert found.total_of(MUSIC) > 0


def test_three_blocks_of_song_do_clear_the_bar():
    frames = np.zeros(500, dtype=np.float32)
    frames[100:196] = 0.9                       # three 320ms blocks
    detector = FrameDetector(speech=0.02, music=frames)
    found = build(_audio(), 24000, detector)
    assert found.total_of(SONG) > 0
