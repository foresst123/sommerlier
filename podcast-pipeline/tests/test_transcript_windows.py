"""Windows over a transcript: sized by tokens, overlapped, each position owned once.

Run:  python -m pytest tests/test_transcript_windows.py -q     (from podcast-pipeline/)
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.transcript_windows import Window, build_windows, clock, format_line, line_number


def _owners(windows, n):
    return [[i for i, w in enumerate(windows) if w.owns(p)] for p in range(n)]


def test_nothing_gives_no_windows():
    assert build_windows([], budget=100, overlap=4) == []


def test_a_transcript_that_fits_is_one_window_that_owns_everything():
    windows = build_windows([10, 10, 10], budget=100, overlap=4)
    assert windows == [Window(0, 3, 0, 3)]


def test_every_position_is_owned_by_exactly_one_window():
    counts = [7, 3, 9, 4, 8, 2, 6, 5, 9, 1, 4, 7, 3, 8, 2, 6, 5, 4, 9, 3]
    for overlap in (0, 1, 3, 4, 6):
        windows = build_windows(counts, budget=25, overlap=overlap)
        owners = _owners(windows, len(counts))
        assert all(len(o) == 1 for o in owners), (overlap, owners)


def test_a_window_only_owns_positions_it_shows():
    counts = [5] * 40
    for w in build_windows(counts, budget=30, overlap=6):
        assert w.start <= w.core_start <= w.core_stop <= w.stop


def test_windows_stay_within_the_token_budget():
    counts = [7, 3, 9, 4, 8, 2, 6, 5, 9, 1, 4, 7, 3, 8, 2, 6]
    for w in build_windows(counts, budget=25, overlap=3):
        assert sum(counts[w.start:w.stop]) <= 25


def test_consecutive_windows_share_the_requested_overlap():
    counts = [5] * 40
    windows = build_windows(counts, budget=50, overlap=4)
    assert len(windows) > 1
    for a, b in zip(windows, windows[1:]):
        assert a.stop - b.start == 4


def test_a_segment_larger_than_the_budget_still_gets_a_window():
    """Not showing a segment is a silent way of never checking it."""
    windows = build_windows([5, 500, 5], budget=20, overlap=1)
    assert any(w.shows(1) for w in windows)
    assert all(len(o) == 1 for o in _owners(windows, 3))


def test_an_overlap_as_long_as_the_window_still_makes_progress():
    windows = build_windows([10] * 12, budget=30, overlap=50)
    assert len(windows) < 100
    assert windows[-1].stop == 12
    assert all(len(o) == 1 for o in _owners(windows, 12))


def test_the_first_window_owns_the_start_and_the_last_owns_the_end():
    windows = build_windows([5] * 30, budget=40, overlap=4)
    assert windows[0].core_start == 0
    assert windows[-1].core_stop == 30


# --- one line per segment ----------------------------------------------------

def _seg(**kw):
    base = dict(index="00012", start=65.25, end=70.0, speaker="SPEAKER_01",
                text="xin chào", gap_before=0.4)
    base.update(kw)
    return SimpleNamespace(**base)


def test_clock_runs_minutes_past_fifty_nine():
    assert clock(65.5) == "01:05.5"
    assert clock(4503.0) == "75:03.0"
    assert clock(-1) == "00:00.0"


def test_a_line_carries_number_time_speaker_gap_and_text():
    line = format_line(_seg(), 7, "B")
    assert line.startswith("[7] 01:05.2-01:10.0 B ")
    assert "(gap +0.4s)" in line and line.endswith("xin chào")


def test_a_line_shows_the_name_it_is_given_never_the_diarizers_label_or_index():
    line = format_line(_seg(), 3, "A")
    assert "SPEAKER_01" not in line and "00012" not in line


def test_a_negative_gap_is_shown_as_an_interruption():
    assert "(gap -0.3s)" in format_line(_seg(gap_before=-0.3), 1, "A")


def test_an_unknown_gap_is_left_out_not_shown_as_zero():
    assert "gap" not in format_line(_seg(gap_before=None), 1, "A")


def test_a_locked_segment_is_marked():
    assert "[cố định]" in format_line(_seg(), 1, "A", locked=True)
    assert "[cố định]" not in format_line(_seg(), 1, "A", locked=False)


def test_newlines_in_the_text_do_not_break_the_line():
    assert "\n" not in format_line(_seg(text="một\nhai\n\nba"), 1, "A")


# --- the number a model wrote back ------------------------------------------

def test_the_ways_a_model_writes_a_line_number_all_read_as_that_number():
    for written in (12, 12.0, "12", " 12 ", "#12", "[12]", "dòng 12"):
        assert line_number(written) == 12, written


def test_what_is_no_line_number_is_none():
    for written in (0, -1, 1.5, True, False, None, "", "đầu", float("nan"), float("inf")):
        assert line_number(written) is None, written
