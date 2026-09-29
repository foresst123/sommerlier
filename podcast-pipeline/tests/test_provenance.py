"""Marking each segment with where it came from, and whether its timing is real.

The pipeline cuts sung and standalone-music stretches out and crossfades the
remainder together, then runs everything else in that shortened timeline. These
pin the translation back -- and, more importantly, the one thing the
translation alone does not give you: which silences between turns are real
silences and which are the seam where audio was removed.

Run:  python -m pytest tests/test_provenance.py -q   (from podcast-pipeline/)
"""
import os
import sys
from dataclasses import dataclass, field

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.excise import TimelineMap
from utils.provenance import annotate, summary


@dataclass
class Seg:
    start: float
    end: float
    orig_spans: list = field(default_factory=list)
    crosses_cut: bool = False
    gap_before: float = None
    noise_score: float = None
    noise_breakdown: dict = None
    noise_kind: str = None


def _timeline():
    """20s of audio with 5.0-8.0 and 12.0-13.0 removed.

    kept (original)  ->  cut
      0.0 -  5.0     ->  0.0 -  5.0
      8.0 - 12.0     ->  5.0 -  9.0
     13.0 - 20.0     ->  9.0 - 16.0
    """
    return TimelineMap([(0.0, 5.0, 0.0), (8.0, 12.0, 5.0), (13.0, 20.0, 9.0)])


# --- provenance --------------------------------------------------------------

def test_a_segment_inside_one_piece_names_one_original_range():
    seg = Seg(6.0, 7.0)
    annotate([seg], _timeline())
    assert seg.orig_spans == [{"start": 9.0, "end": 10.0}]
    assert seg.crosses_cut is False


def test_a_segment_straddling_a_join_names_both_pieces():
    """A single (start, end) pair would claim 4.0-9.0 of the original, which
    includes the 5.0-8.0 that was cut -- audio this segment never held."""
    seg = Seg(4.0, 6.0)
    annotate([seg], _timeline())
    assert seg.orig_spans == [{"start": 4.0, "end": 5.0}, {"start": 8.0, "end": 9.0}]
    assert seg.crosses_cut is True
    held = sum(s["end"] - s["start"] for s in seg.orig_spans)
    assert held == pytest.approx(seg.end - seg.start)


def test_an_uncut_recording_still_gets_marked():
    """No excision is the ordinary case; provenance must not be a special case
    the consumer has to branch on."""
    segs = [Seg(0.0, 2.0), Seg(3.0, 5.0)]
    annotate(segs, None)
    assert segs[0].orig_spans == [{"start": 0.0, "end": 2.0}]
    assert segs[1].crosses_cut is False
    assert segs[1].gap_before == 1.0


# --- the gap, which is the point ---------------------------------------------

def test_a_gap_within_one_piece_is_a_real_pause():
    segs = [Seg(5.5, 6.0), Seg(6.5, 7.0)]
    annotate(segs, _timeline())
    assert segs[1].gap_before == 0.5


def test_a_gap_spanning_a_join_is_unknowable_not_zero():
    """In the cut timeline these two turns are 0.2s apart. In the recording
    they are separated by the 3s that was removed at the join. Neither number
    is the pause between the speakers, so no number is offered."""
    segs = [Seg(4.0, 4.9), Seg(5.1, 6.0)]
    annotate(segs, _timeline())
    assert segs[1].gap_before is None


def test_a_turn_ending_exactly_on_a_join_breaks_the_next_gap():
    segs = [Seg(4.0, 5.0), Seg(5.2, 6.0)]
    annotate(segs, _timeline())
    assert segs[1].gap_before is None


def test_the_first_segment_has_no_gap_to_measure():
    """The lead-in is not a pause between speakers."""
    segs = [Seg(6.0, 7.0), Seg(7.5, 8.0)]
    annotate(segs, _timeline())
    assert segs[0].gap_before is None
    assert segs[1].gap_before == 0.5


def test_overlapping_turns_keep_a_negative_gap():
    """Interruption is the phenomenon a full-duplex corpus exists to capture;
    clamping it to zero would erase it."""
    segs = [Seg(6.0, 7.0), Seg(6.8, 8.0)]
    annotate(segs, _timeline())
    assert segs[1].gap_before == pytest.approx(-0.2)


# --- the noise score is read over the original ranges ------------------------

def test_noise_is_scored_over_what_the_segment_actually_holds():
    """A glued segment must be judged on its two real pieces, not on the range
    between them -- which includes audio that was removed."""
    seen = []

    class Track:
        def score_spans(self, spans):
            seen.append(list(spans))
            return 0.42

        def breakdown(self, spans):
            return {"noise_speech": 0.42, "noise_env": 0.0, "noise_room": 0.0}

    seg = Seg(4.0, 6.0)
    annotate([seg], _timeline(), noise=Track())
    assert seg.noise_score == 0.42
    assert seen == [[(4.0, 5.0), (8.0, 9.0)]]


def test_no_detector_leaves_the_score_unset_rather_than_zero():
    """Zero would read as "measured, and clean". Unset reads as "not checked"."""
    seg = Seg(6.0, 7.0)
    annotate([seg], _timeline())
    assert seg.noise_score is None


# --- what kind of noise it is ------------------------------------------------

def _stub_track(breakdown, score=0.3):
    class Track:
        def score_spans(self, spans):
            return score

        def breakdown(self, spans):
            return dict(breakdown)
    return Track()


def test_a_segment_is_labelled_with_the_noise_it_carries():
    """noise_score says how much; this says what -- a filter treats voices
    behind the speaker differently from traffic."""
    numbers = {"noise_speech": 0.05, "noise_env": 0.31, "noise_room": 0.02}
    seg = Seg(6.0, 7.0)

    annotate([seg], _timeline(), noise=_stub_track(numbers))

    assert seg.noise_kind == "noise_env"
    assert seg.noise_breakdown == numbers, "the evidence travels with the word"


def test_below_the_noticeable_level_is_clean_not_left_unset():
    """Unset means "not checked". A segment that WAS checked and heard nothing
    worth naming must not look like one that was never looked at."""
    numbers = {"noise_speech": 0.02, "noise_env": 0.01, "noise_room": 0.03}
    seg = Seg(6.0, 7.0)

    annotate([seg], _timeline(), noise=_stub_track(numbers, score=0.03))

    assert seg.noise_kind == "clean"
    assert seg.noise_breakdown == numbers


def test_an_unmeasured_segment_has_no_kind_and_no_numbers():
    """No frames under the span: three Nones would look like a result."""
    nothing = {"noise_speech": None, "noise_env": None, "noise_room": None}
    seg = Seg(6.0, 7.0)

    annotate([seg], _timeline(), noise=_stub_track(nothing, score=None))

    assert seg.noise_score is None
    assert seg.noise_breakdown is None
    assert seg.noise_kind is None


def test_no_detector_leaves_the_kind_unset_too():
    seg = Seg(6.0, 7.0)
    annotate([seg], _timeline())
    assert seg.noise_kind is None and seg.noise_breakdown is None


def test_the_real_track_labels_a_glued_segment_from_its_two_pieces_only():
    """End to end on the real NoiseTrack: voices in the first piece, silence in
    the second. The removed stretch between them must not be scored."""
    import numpy as np
    from utils.noise_map import NoiseTrack

    fps = 100
    speech = np.zeros(20 * fps, dtype=np.float32)
    speech[4 * fps:5 * fps] = 0.6          # inside the first kept piece
    speech[5 * fps:8 * fps] = 0.9          # inside the REMOVED stretch
    track = NoiseTrack({"noise_speech": speech,
                        "noise_env": np.zeros_like(speech),
                        "noise_room": np.zeros_like(speech)}, fps=fps)
    seg = Seg(4.0, 6.0)  # cut timeline: original 4-5 and 8-9

    annotate([seg], _timeline(), noise=track)

    assert seg.noise_kind == "noise_speech"
    assert seg.noise_breakdown["noise_speech"] == pytest.approx(0.6, abs=0.05)


# --- the manifest line -------------------------------------------------------

def test_the_summary_makes_a_badly_cut_run_visible():
    segs = [Seg(4.0, 4.9), Seg(5.1, 6.0), Seg(6.5, 7.0)]
    annotate(segs, _timeline())
    out = summary(segs)
    assert out["segments"] == 3
    assert out["segments_crossing_a_cut"] == 0
    # Only the middle one: the first segment's None is expected, not a break.
    assert out["gaps_broken_by_a_cut"] == 1


def test_the_summary_reports_noise_when_it_was_measured():
    class Track:
        def __init__(self): self.n = 0
        def score_spans(self, spans):
            self.n += 1
            return 0.1 * self.n
        def breakdown(self, spans):
            return {"noise_speech": 0.0, "noise_env": 0.0, "noise_room": 0.0}

    segs = [Seg(6.0, 6.5), Seg(7.0, 7.5), Seg(8.0, 8.5)]
    annotate(segs, _timeline(), noise=Track())
    out = summary(segs)
    assert out["noise_p50"] == 0.2
    assert out["noise_max"] == 0.3


def test_the_summary_omits_noise_when_nothing_was_measured():
    segs = [Seg(6.0, 6.5)]
    annotate(segs, _timeline())
    assert "noise_p50" not in summary(segs)


# --- the three modules actually connect --------------------------------------

def test_one_sweep_feeds_the_map_the_track_and_the_segments():
    """music_map.build_maps -> NoiseTrack -> annotate is the whole chain, and
    each link was written separately. This pins that they fit."""
    import numpy as np
    from utils.music_map import MUSIC, build_maps

    fps = 100.0
    frames = 2000                                   # 20 seconds

    class Detector:
        """One PANNs sweep: music in 5-8s, traffic in 12-14s."""
        calls = 0

        def tag_framewise(self, waveform, sample_rate):
            Detector.calls += 1
            zero = lambda: np.zeros(frames, dtype=np.float32)
            speech = np.full(frames, 0.9, dtype=np.float32)
            music = zero(); music[500:800] = 0.9
            env = zero(); env[1200:1400] = 0.8
            return ({"speech": speech, "singing": zero(), "music": music,
                     "noise_speech": zero(), "noise_env": env,
                     "noise_room": zero()}, fps)

    music_map, noise = build_maps(np.zeros(20 * 16000, dtype=np.float32),
                                  16000, Detector())
    assert Detector.calls == 1, "the tagger must be swept once, not twice"
    assert music_map.total_of(MUSIC) > 0
    assert noise, "the noise track came out of the same sweep"

    # Nothing was excised in this run, so cut time is original time.
    segs = [Seg(6.0, 7.0), Seg(12.5, 13.5)]
    annotate(segs, TimelineMap(), noise=noise)

    assert segs[0].noise_score is not None
    assert segs[0].noise_score < 0.2, "the music stretch is not noise"
    assert segs[1].noise_score > 0.5, "the traffic stretch is"


def test_a_segment_glued_over_a_cut_is_scored_on_its_real_audio():
    """The end-to-end version of the trap: the removed stretch must not be
    swept back into the score by the naive range."""
    import numpy as np
    from utils.noise_map import NoiseTrack

    curve = np.zeros(2000, dtype=np.float32)
    curve[500:800] = 0.95                     # 5-8s: removed, filthy
    noise = NoiseTrack({"noise_env": curve}, fps=100.0)

    # 5.0-8.0 excised, so cut 4.5-5.5 is original 4.5-5.0 + 8.0-8.5.
    timeline = TimelineMap([(0.0, 5.0, 0.0), (8.0, 20.0, 5.0)])
    seg = Seg(4.5, 5.5)
    annotate([seg], timeline, noise=noise)

    assert seg.crosses_cut is True
    assert seg.noise_score < 0.1, "scored the audio it holds, not the gap"


def test_the_summary_counts_how_many_segments_carry_each_kind():
    segs = [Seg(6.0, 6.5), Seg(7.0, 7.5), Seg(8.0, 8.5)]
    kinds = iter(["noise_env", "clean", "noise_env"])
    for seg in segs:
        seg.noise_kind = next(kinds)

    assert summary(segs)["noise_kinds"] == {"noise_env": 2, "clean": 1}


def test_the_summary_omits_the_kinds_when_none_was_assigned():
    assert "noise_kinds" not in summary([Seg(6.0, 6.5)])
