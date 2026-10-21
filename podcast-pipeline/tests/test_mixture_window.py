"""Choosing the audio handed to USEF-TSE around one overlap.

The model's mixture window is fixed at 2 seconds. Most overlaps on this corpus
are shorter than that -- a "dạ" or a "vâng" lasting a third of a second -- so
the question every job asks is how to reach two seconds around one.

Padding with silence is the wrong answer where real audio exists: the model has
to hear the voices it is separating. These pin the widening, and the one thing
it must never do -- reach across a seam left by excising a sung or musical
stretch, which would drag in audio from a different part of the recording
entirely.

Run:  python -m pytest tests/test_mixture_window.py -q   (from podcast-pipeline/)
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mixture_window import WINDOW_TARGET, bounds, widen, window_for


# --- the ordinary case ------------------------------------------------------

def test_an_overlap_already_long_enough_is_left_alone():
    """Past the target the separator chunks internally; nothing to arrange."""
    assert window_for(10.0, 35.0, [], 90.0) == (10.0, 35.0)
    assert window_for(10.0, 30.0, [], 90.0) == (10.0, 30.0)


def test_a_short_overlap_is_grown_evenly_on_both_sides():
    """A 0.34s backchannel needs 19.66s, so 9.83s goes each way."""
    lo, hi = window_for(10.0, 10.34, [], 60.0)
    assert lo == pytest.approx(0.17)
    assert hi == pytest.approx(20.17)
    assert hi - lo == pytest.approx(WINDOW_TARGET)


def test_the_overlap_stays_centred_when_there_is_room():
    lo, hi = window_for(30.0, 30.5, [], 60.0)
    assert 30.0 - lo == pytest.approx(hi - 30.5)


# --- walls: the ends of the recording ---------------------------------------

def test_an_overlap_at_the_start_grows_only_forwards():
    lo, hi = window_for(0.1, 0.5, [], 60.0)
    assert lo == 0.0
    assert hi - lo == pytest.approx(WINDOW_TARGET), "the shortfall moves right"


def test_an_overlap_at_the_end_grows_only_backwards():
    lo, hi = window_for(59.6, 59.9, [], 60.0)
    assert hi == pytest.approx(60.0)
    assert hi - lo == pytest.approx(WINDOW_TARGET)


# --- walls: seams left by excising ------------------------------------------

def test_widening_stops_at_a_seam_and_takes_the_rest_from_the_far_side():
    """Reaching across a join picks up audio from elsewhere in the recording,
    whose speakers have nothing to do with this overlap."""
    lo, hi = window_for(10.0, 10.4, seams=[9.8], duration=60.0)
    assert lo == pytest.approx(9.8), "stopped at the seam"
    assert hi - lo == pytest.approx(WINDOW_TARGET)
    assert hi == pytest.approx(29.8), "the shortfall came from the right"


def test_a_seam_on_the_right_pushes_the_window_left():
    """The window still gets its full length; it just takes it all from the
    side that has room."""
    lo, hi = window_for(60.0, 60.4, seams=[65.0], duration=120.0)
    assert hi == pytest.approx(65.0), "stopped at the seam"
    assert lo == pytest.approx(45.0)
    assert hi - lo == pytest.approx(WINDOW_TARGET)


def test_seams_on_both_sides_cap_the_window_short():
    """A stretch between two joins can simply be shorter than the separator
    wants. Reaching past one would pull in a different part of the recording,
    so a short window is the correct answer rather than a failure."""
    lo, hi = window_for(10.0, 10.4, seams=[9.7, 10.9], duration=60.0)
    assert (lo, hi) == (pytest.approx(9.7), pytest.approx(10.9))
    assert hi - lo < WINDOW_TARGET


def test_only_the_nearest_seam_on_each_side_matters():
    lo, hi = window_for(10.0, 10.4, seams=[2.0, 5.0, 9.5, 12.0, 40.0], duration=60.0)
    assert lo >= 9.5 and hi <= 12.0


def test_a_seam_far_away_does_not_constrain_anything():
    lo, hi = window_for(30.0, 30.4, seams=[1.0, 59.0], duration=60.0)
    assert hi - lo == pytest.approx(WINDOW_TARGET)


# --- the bounds helper on its own -------------------------------------------

def test_bounds_default_to_the_ends_of_the_recording():
    assert bounds(10.0, 11.0, [], 60.0) == (0.0, 60.0)


def test_a_seam_exactly_on_the_edge_of_the_overlap_still_walls_it():
    """Touching counts: what lies beyond is unrelated audio either way."""
    assert bounds(10.0, 11.0, [10.0], 60.0)[0] == 10.0
    assert bounds(10.0, 11.0, [11.0], 60.0)[1] == 11.0


# --- widen on its own -------------------------------------------------------

def test_widen_never_returns_less_than_it_was_given():
    lo, hi = widen(5.0, 5.2, 5.0, 5.2)
    assert (lo, hi) == (5.0, 5.2)


def test_widen_is_symmetric_about_the_overlap():
    lo, hi = widen(10.0, 10.4, 0.0, 60.0)
    assert 10.0 - lo == pytest.approx(hi - 10.4)


def test_the_target_matches_the_separator_s_own_chunk_length():
    """Sidon processes 20s chunks (CHUNK_SECONDS in sidon_infer.py). A window
    shorter than one chunk runs it outside its design point, and a blind
    separator given 2s -- which is what USEF's ONNX graph fixed this at -- had
    no solo stretch to identify either speaker from: ECAPA similarity fell to
    p50 0.15, the score two unrelated speakers get."""
    assert WINDOW_TARGET == 20.0


# --- seams come from the excise timeline ------------------------------------

def test_the_timeline_reports_its_joins_in_cut_time():
    from utils.excise import TimelineMap
    # 0-5 and 8-12 and 13-20 kept -> cut timeline 0-5, 5-9, 9-16
    tl = TimelineMap([(0.0, 5.0, 0.0), (8.0, 12.0, 5.0), (13.0, 20.0, 9.0)])
    assert tl.seams() == [5.0, 9.0]


def test_an_uncut_recording_has_no_joins():
    from utils.excise import TimelineMap
    assert TimelineMap().seams() == []
    assert TimelineMap([(0.0, 20.0, 0.0)]).seams() == []


def test_a_join_reported_by_seams_is_one_cut_between_agrees_with():
    """Two views of the same fact; they must not drift apart."""
    from utils.excise import TimelineMap
    tl = TimelineMap([(0.0, 5.0, 0.0), (8.0, 12.0, 5.0), (13.0, 20.0, 9.0)])
    for seam in tl.seams():
        assert tl.cut_between(seam - 0.01, seam + 0.01)
