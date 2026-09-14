"""Overlap must survive merging, ghost dissolution and the VAD cursor.

Every regression here was a real way to lose a backchannel. The separator can
only work on an overlap that reaches it, so a stage that quietly drops one
costs a training example and reports nothing.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.segment_utils import (
    bridge_interrupted_speaker_turns,
    cut_by_speaker_label,
    merge_ghost_speakers,
    split_long_segments,
)


def overlaps(segments):
    """Every cross-speaker overlap, as (a, b, seconds)."""
    found = []
    ordered = sorted(segments, key=lambda s: s["start"])
    for i, first in enumerate(ordered):
        for second in ordered[i + 1:]:
            if second["start"] >= first["end"]:
                break
            if first["speaker"] == second["speaker"]:
                continue
            span = min(first["end"], second["end"]) - max(first["start"], second["start"])
            if span > 0:
                found.append((first["speaker"], second["speaker"], round(span, 4)))
    return found


def test_merging_keeps_a_backchannel_that_overlaps_a_turn():
    """A 0.4s backchannel inside a turn is the shortest thing we must keep.

    Measured floor on this corpus is 0.24s, so min_segment_length has to sit
    below it -- the default 0.2 is too close to be safe.
    """
    segments = [
        {"index": "00000", "start": 10.0, "end": 14.0, "speaker": "A"},
        {"index": "00001", "start": 13.6, "end": 14.2, "speaker": "B"},
        {"index": "00002", "start": 14.5, "end": 18.0, "speaker": "A"},
    ]
    assert overlaps(segments) == [("A", "B", 0.4)]

    merged = cut_by_speaker_label(segments, merge_gap=2.0, min_segment_length=0.1,
                                  max_segment_length=30.0)
    found = overlaps(merged)
    assert found, "the backchannel was lost in merging"
    assert any(s["speaker"] == "B" for s in merged), "speaker B disappeared entirely"
    # Joining A's halves puts B wholly inside one turn, so the overlap grows
    # from 0.4s to B's full 0.6s. Growing is right -- B really does speak
    # entirely while A holds the floor. Shrinking would be the bug.
    assert found[0][2] == pytest.approx(0.6, abs=1e-6)


def test_merging_keeps_a_micro_fragment_when_it_carries_an_overlap():
    """Sub-100ms diarizer output is only jitter when it overlaps nobody."""
    segments = [
        {"index": "00000", "start": 10.0, "end": 12.0, "speaker": "A"},
        {"index": "00001", "start": 10.5, "end": 10.54, "speaker": "B"},
    ]
    merged = cut_by_speaker_label(
        segments, merge_gap=0.3, min_segment_length=0.1, max_segment_length=20.0
    )
    assert any(segment["speaker"] == "B" for segment in merged)
    assert overlaps(merged) == [("A", "B", 0.04)]


def test_bridge_requires_the_interrupter_to_overlap_both_a_edges():
    """An ordinary B turn in A's silent gap must not invent an A overlap."""
    segments = [
        {"index": "00000", "start": 10.0, "end": 14.0, "speaker": "A"},
        {"index": "00001", "start": 14.1, "end": 14.4, "speaker": "B"},
        {"index": "00002", "start": 14.5, "end": 18.0, "speaker": "A"},
    ]
    assert overlaps(segments) == [], "precondition: the split hides the overlap"
    assert bridge_interrupted_speaker_turns(segments, bridge_gap=3.0) == segments


def test_bridge_rebuilds_one_meaningful_overlap_from_two_edge_slivers():
    segments = [
        {"index": "00000", "start": 10.0, "end": 10.25, "speaker": "A"},
        {"index": "00001", "start": 10.2, "end": 12.28, "speaker": "B"},
        {"index": "00002", "start": 12.26, "end": 14.0, "speaker": "A"},
    ]
    bridged = bridge_interrupted_speaker_turns(segments, bridge_gap=3.0)
    a = next(segment for segment in bridged if segment["speaker"] == "A")
    assert (a["start"], a["end"]) == (10.0, 14.0)
    assert overlaps(bridged) == [("A", "B", 2.08)]


def test_bridge_stops_at_a_seam_or_a_third_speaker():
    segments = [
        {"index": "00000", "start": 10.0, "end": 10.25, "speaker": "A"},
        {"index": "00001", "start": 10.2, "end": 12.28, "speaker": "B"},
        {"index": "00002", "start": 12.26, "end": 14.0, "speaker": "A"},
        {"index": "00003", "start": 11.0, "end": 11.2, "speaker": "C"},
    ]
    assert len(bridge_interrupted_speaker_turns(segments, bridge_gap=3.0)) == 4
    assert len(bridge_interrupted_speaker_turns(segments[:3], bridge_gap=3.0,
                                                seams=[11.0])) == 3


def test_split_keeps_an_overlap_in_one_segment_when_space_exists():
    segments = [
        {"index": "00000", "start": 0.0, "end": 70.0, "speaker": "A"},
        {"index": "00001", "start": 25.0, "end": 35.0, "speaker": "B"},
    ]
    out = split_long_segments(segments, max_duration=30.0, min_piece=0.2)
    a = [segment for segment in out if segment["speaker"] == "A"]
    assert [(segment["start"], segment["end"]) for segment in a] == [
        (0.0, 25.0), (25.0, 55.0), (55.0, 70.0)]
    assert overlaps(out) == [("A", "B", 10.0)]


def test_overlong_full_overlap_is_not_cut_through_post_diarization():
    segments = [
        {"index": "00000", "start": 0.0, "end": 70.0, "speaker": "A"},
        {"index": "00001", "start": 0.0, "end": 70.0, "speaker": "B"},
    ]
    out = split_long_segments(segments, max_duration=30.0, min_piece=0.2)
    by_speaker = {
        speaker: [(segment["start"], segment["end"]) for segment in out
                  if segment["speaker"] == speaker]
        for speaker in ("A", "B")
    }
    assert by_speaker["A"] == by_speaker["B"] == [(0.0, 70.0)]


def test_split_output_is_time_ordered_before_indices_are_assigned():
    segments = [
        {"index": "00000", "start": 0.0, "end": 70.0, "speaker": "A"},
        {"index": "00001", "start": 25.0, "end": 35.0, "speaker": "B"},
    ]
    out = split_long_segments(segments, max_duration=30.0, min_piece=0.2)
    assert [(segment["start"], segment["end"], segment["speaker"]) for segment in out] == [
        (0.0, 25.0, "A"),
        (25.0, 35.0, "B"),
        (25.0, 55.0, "A"),
        (55.0, 70.0, "A"),
    ]
    assert [segment["index"] for segment in out] == ["00000", "00001", "00002", "00003"]


def test_merging_never_bridges_a_seam():
    """Two turns either side of an excision are minutes apart in the source."""
    segments = [
        {"index": "00000", "start": 10.0, "end": 14.0, "speaker": "A"},
        {"index": "00001", "start": 14.2, "end": 18.0, "speaker": "A"},
    ]
    merged = cut_by_speaker_label(segments, merge_gap=2.0, min_segment_length=0.1,
                                  max_segment_length=30.0, seams=[14.1])
    assert len(merged) == 2, "a seam must stop the merge"


def test_ghost_dissolution_keeps_a_two_speaker_conversation_intact():
    """With two speakers there is no third to dissolve, however quiet one is."""
    segments = [
        {"index": "00000", "start": 0.0, "end": 100.0, "speaker": "A"},
        {"index": "00001", "start": 50.0, "end": 50.5, "speaker": "B"},
    ]
    assert merge_ghost_speakers(segments) == segments


def test_ghost_dissolution_removes_the_third_speaker_but_keeps_the_overlap():
    """Speaker 0 held under 1.5s across a 1002s file and blocked 8 overlaps.

    Relabelling it must not change any segment's timing -- only its label --
    so the overlap it sat inside survives with the same span.
    """
    segments = [
        {"index": "00000", "start": 0.0, "end": 400.0, "speaker": "A"},
        {"index": "00001", "start": 200.0, "end": 200.6, "speaker": "GHOST"},
        {"index": "00002", "start": 500.0, "end": 900.0, "speaker": "B"},
    ]
    before = overlaps(segments)
    assert ("A", "GHOST", 0.6) in before

    after = merge_ghost_speakers(segments)
    assert not any(s["speaker"] == "GHOST" for s in after), "ghost should be gone"
    spans = sorted(s["end"] - s["start"] for s in after)
    assert spans == sorted(s["end"] - s["start"] for s in segments), \
        "ghost dissolution changed segment timing; it must only relabel"


def test_ghost_with_one_uncertain_fragment_still_repairs_supported_fragments():
    segments = [
        {"index": "00000", "start": 0.0, "end": 300.0, "speaker": "A"},
        {"index": "00001", "start": 400.0, "end": 500.0, "speaker": "B"},
        {"index": "00002", "start": 100.0, "end": 100.4, "speaker": "G"},
        {"index": "00003", "start": 350.0, "end": 350.4, "speaker": "G"},
    ]
    out = merge_ghost_speakers(segments)
    assert not any(
        segment["speaker"] == "G" and segment["start"] < 300.0
        for segment in out
    )
    assert any(
        segment["speaker"] == "G" and segment["start"] == 350.0
        for segment in out
    )


def test_vad_cursor_is_per_speaker_so_cross_speaker_overlap_survives():
    """A global cursor clips one speaker's start against another's end.

    That is the whole overlap. The cursor exists to drop a speaker overlapping
    themselves -- a labelling artefact -- and must not reach across speakers.
    """

    frame = pd.DataFrame([
        {"start": 10.0, "end": 14.0, "speaker": "A"},
        {"start": 13.6, "end": 14.2, "speaker": "B"},
        {"start": 14.5, "end": 18.0, "speaker": "A"},
    ])

    # Replay the cursor from vad() without loading the ONNX model.
    out, cursors = [], {}
    for _, row in frame.iterrows():
        start, end, speaker = float(row["start"]), float(row["end"]), row["speaker"]
        seen = cursors.get(speaker, 0.0)
        if end <= seen:
            continue
        start = max(start, seen)
        cursors[speaker] = end
        out.append({"start": start, "end": end, "speaker": speaker})

    assert ("A", "B", 0.4) in overlaps(out), "per-speaker cursor lost the overlap"
    # Read the constant from source: importing the module pulls in librosa and
    # onnxruntime, which this control-flow test has no use for.
    import re
    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models", "silero_vad.py"), encoding="utf-8").read()
    threshold = float(re.search(r'VAD_THRESHOLD = float\(os\.environ\.get\(\s*"VAD_THRESHOLD",\s*"([\d.]+)"', source).group(1))
    assert threshold <= 0.5, \
        "above this, short turns keep diarizer edges and long ones get Silero's"


def test_vad_cursor_still_drops_a_speaker_overlapping_themselves():
    frame = pd.DataFrame([
        {"start": 10.0, "end": 14.0, "speaker": "A"},
        {"start": 12.0, "end": 13.0, "speaker": "A"},
        {"start": 13.0, "end": 16.0, "speaker": "A"},
    ])
    out, cursors = [], {}
    for _, row in frame.iterrows():
        start, end, speaker = float(row["start"]), float(row["end"]), row["speaker"]
        seen = cursors.get(speaker, 0.0)
        if end <= seen:
            continue
        start = max(start, seen)
        cursors[speaker] = end
        out.append({"start": start, "end": end, "speaker": speaker})

    assert len(out) == 2, "the fully-contained duplicate should be dropped"
    assert out[1]["start"] == 14.0, "the partial duplicate should be clipped"
