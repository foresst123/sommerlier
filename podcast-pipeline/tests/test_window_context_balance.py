"""The window must carry context on both sides of the core, not just the left.

Measured on a 1002s two-speaker file: mean left context 5.58s against 1.39s on
the right, no window short of context on the left, 26 of 30 short on the right.
A backchannel is evidenced by the host resuming after it, and that evidence sits
entirely on the right.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.segment import Segment
from utils.separation_window import WindowPlanner
from algorithms.diarization.overlap import detect_overlapping_segments
from services.separation_service import SeparationService

SR = 1000


def waveform(duration=120):
    t = np.arange(duration * SR) / SR
    wave = (0.1 * np.sin(2 * np.pi * 83 * t)).astype(np.float32)
    wave[(t % 0.5) < 0.08] = 0
    return wave


def plan(rows):
    segs = [Segment(str(i), a, b, s) for i, (a, b, s) in enumerate(rows)]
    pairs = detect_overlapping_segments([s.__dict__ for s in segs], overlap_threshold=0)
    planner = WindowPlanner(segs, pairs, waveform(), SR)
    jobs = SeparationService()._group_jobs(pairs)
    return planner, planner.build(jobs[0][2])


# A 0.45s backchannel inside a long host turn, with material for both speakers
# elsewhere -- the shape of nearly every job in the real file.
BACKCHANNEL = [(10, 30, "A"), (20, 20.45, "B"), (0, 8, "B"),
               (40, 48, "B"), (50, 62, "A"), (70, 80, "A")]


def test_the_right_of_the_core_gets_real_context():
    planner, result = plan(BACKCHANNEL)
    assert result is not None, planner.detail
    left, right = result.layout["context_seconds"]
    assert right >= 2.0, f"only {right:.2f}s after the core; the host's resumption is the evidence"
    assert left / right <= 3.0, f"context leans {left/right:.1f}:1 left ({left:.2f}s vs {right:.2f}s)"


def test_reported_shortfall_matches_what_was_scored():
    """context_shortfall_seconds existed but was never fed into the score.

    It is the metric that shows the defect, so it has to stay honest: a window
    the planner considers good should not be reporting a shortfall.
    """
    planner, result = plan(BACKCHANNEL)
    assert result is not None, planner.detail
    assert result.layout["context_shortfall_seconds"][1] == pytest.approx(0.0, abs=1e-6)


def test_both_speakers_keep_comparable_voice():
    """Freeing budget for the right must not be paid for out of the balance.

    The prefix support piece is drawn from the host's own speech, so buying
    left context with a prefix hands the host more voice, not less. A window
    holding one speaker against a sliver of the other is what makes the
    separator emit a source and silence.
    """
    planner, result = plan(BACKCHANNEL)
    assert result is not None, planner.detail
    assert result.layout["ratio"] <= 1.35, \
        f"voice ratio {result.layout['ratio']:.2f}: {result.layout['estimated_voice_seconds']}"


def test_a_core_near_the_start_of_the_host_still_builds():
    """base_core_min was 3.0s, so a core less than 3s clear of everything to its
    left had no legal base start and the job was dropped as
    no_safe_left_cut_for_3_10s_base -- 25 of 27 window failures on the real file
    carried that reason. Here B's own earlier turn floors the search at 9.8s
    while the core starts at 11.8s, leaving 2.0s: under the old floor, nothing.
    At 1.5s the base can sit close to the core and a prefix carries it to the
    anchor.
    """
    planner, result = plan([(10, 30, "A"), (11.8, 12.25, "B"), (8.5, 9.8, "B"),
                            (40, 48, "B"), (50, 62, "A"), (70, 80, "A")])
    assert result is not None, planner.detail


def test_the_core_sits_inside_the_modules_own_anchor_band():
    planner, result = plan(BACKCHANNEL)
    assert result is not None, planner.detail
    assert 5.0 <= result.layout["core_position_seconds"] <= 8.0


def test_the_base_never_outgrows_its_budget():
    """Host retention was scored against the full 15s target, so on a long host
    the score kept rewarding a wider base up to swallowing the window. The cap
    is the 7s the layout budgets for the base.
    """
    planner, result = plan(BACKCHANNEL)
    assert result is not None, planner.detail
    base = next(p for p in result.layout["pieces"] if p["kind"] == "base")
    width = (base["source_samples"][1] - base["source_samples"][0]) / SR
    assert width <= 10.0, f"base grew to {width:.2f}s and left nothing for the other speaker"
