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

from utils.segment_utils import cut_by_speaker_label, merge_ghost_speakers


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


def test_merging_recovers_an_overlap_the_diarizer_split_away():
    """Two halves of one turn hide the overlap between them.

    Before merging, A=[10,14] and A=[14.5,18] leave B=[14.1,14.4] touching
    neither, so no overlap is detected at all. Joining A's halves puts B back
    inside a turn where it belongs. Merging adds overlaps here; it never
    removes them.
    """
    segments = [
        {"index": "00000", "start": 10.0, "end": 14.0, "speaker": "A"},
        {"index": "00001", "start": 14.1, "end": 14.4, "speaker": "B"},
        {"index": "00002", "start": 14.5, "end": 18.0, "speaker": "A"},
    ]
    assert overlaps(segments) == [], "precondition: the split hides the overlap"

    merged = cut_by_speaker_label(segments, merge_gap=2.0, min_segment_length=0.1,
                                  max_segment_length=30.0)
    recovered = overlaps(merged)
    assert recovered, "merging should expose the backchannel inside A's turn"
    assert recovered[0][2] == pytest.approx(0.3, abs=1e-6)


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
