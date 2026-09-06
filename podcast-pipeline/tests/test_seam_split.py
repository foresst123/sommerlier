"""No segment may span a join left by excising.

Cutting the sung and standalone-music stretches out leaves points where two
parts of the recording that were never adjacent now touch. A segment straddling
one is two pieces of the conversation glued together, minutes apart in the
original -- and every stage after it inherits the lie: the separator's mixture
window widens across it, the dual-channel reconstruction writes it as
continuous speech, and anything measuring turn timing reads a pause that never
happened.

Marking it afterwards (`crosses_cut`) says it happened. This stops it.

Run:  python -m pytest tests/test_seam_split.py -q   (from podcast-pipeline/)
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.excise import TimelineMap
from utils.segment_utils import split_at_seams


def _seg(start, end, speaker="A"):
    return {"start": start, "end": end, "speaker": speaker, "index": "00000"}


def test_a_segment_spanning_a_join_becomes_two():
    out = split_at_seams([_seg(4.0, 6.0)], seams=[5.0])
    assert [(s["start"], s["end"]) for s in out] == [(4.0, 5.0), (5.0, 6.0)]
    assert all(s["speaker"] == "A" for s in out), "the speaker travels with both"


def test_a_segment_clear_of_every_join_is_untouched():
    original = _seg(6.0, 7.0)
    out = split_at_seams([original], seams=[5.0, 9.0])
    assert len(out) == 1
    assert (out[0]["start"], out[0]["end"]) == (6.0, 7.0)


def test_a_segment_that_ends_exactly_on_a_join_is_not_split():
    """It is already one contiguous piece; there is nothing on the far side."""
    out = split_at_seams([_seg(4.0, 5.0)], seams=[5.0])
    assert len(out) == 1


def test_a_segment_crossing_two_joins_becomes_three():
    out = split_at_seams([_seg(4.0, 10.0)], seams=[5.0, 9.0])
    assert [(s["start"], s["end"]) for s in out] == [(4.0, 5.0), (5.0, 9.0), (9.0, 10.0)]


def test_a_sliver_beside_a_join_is_dropped_not_kept():
    """Twenty milliseconds on one side of a join is the tail of a crossfade,
    not speech, and it would reach ASR as a segment of its own."""
    out = split_at_seams([_seg(4.98, 7.0)], seams=[5.0])
    assert len(out) == 1
    assert out[0]["start"] == 5.0


def test_no_seams_is_not_a_special_case_for_the_caller():
    segs = [_seg(0.0, 3.0), _seg(4.0, 6.0)]
    assert split_at_seams(segs, seams=[]) == segs
    assert split_at_seams(segs, seams=None) == segs


def test_indices_are_dense_and_ordered_after_a_split():
    """Downstream reads them as a position, and a split has just made the old
    ones neither dense nor unique."""
    out = split_at_seams([_seg(0.0, 2.0), _seg(4.0, 6.0), _seg(8.0, 9.0)], seams=[5.0])
    assert [s["index"] for s in out] == ["00000", "00001", "00002", "00003"]


def test_total_speech_time_is_preserved():
    segs = [_seg(4.0, 10.0), _seg(12.0, 13.0)]
    before = sum(s["end"] - s["start"] for s in segs)
    after = sum(s["end"] - s["start"] for s in split_at_seams(segs, seams=[5.0, 9.0]))
    assert after == pytest.approx(before)


# --- the case that actually produces these ----------------------------------

def test_merging_cannot_put_a_join_back_inside_a_segment():
    """The split runs before VAD, so the one step that could undo it is the
    merge: two turns of one speaker either side of a join sit 0.05s apart in
    the cut timeline, well inside merge_gap of 0.3s, while in the recording
    they are minutes apart with a song between them."""
    from utils.segment_utils import cut_by_speaker_label
    pieces = [_seg(4.0, 5.0), _seg(5.05, 6.0)]

    joined = cut_by_speaker_label(pieces, merge_gap=0.3, max_segment_length=20.0)
    assert len(joined) == 1, "without seams the merge bridges the join"

    kept = cut_by_speaker_label(pieces, merge_gap=0.3, max_segment_length=20.0,
                                seams=[5.0])
    assert len(kept) == 2, "with seams it must not"


def test_a_gap_that_is_not_a_join_still_merges():
    """The guard has to be about joins, not about gaps in general."""
    from utils.segment_utils import cut_by_speaker_label
    pieces = [_seg(4.0, 5.0), _seg(5.05, 6.0)]
    out = cut_by_speaker_label(pieces, merge_gap=0.3, max_segment_length=20.0,
                               seams=[12.0])
    assert len(out) == 1


def test_the_seams_come_from_the_timeline_the_cut_produced():
    timeline = TimelineMap([(0.0, 5.0, 0.0), (8.0, 12.0, 5.0), (13.0, 20.0, 9.0)])
    out = split_at_seams([_seg(0.0, 16.0)], timeline.seams())
    assert [(s["start"], s["end"]) for s in out] == [(0.0, 5.0), (5.0, 9.0), (9.0, 16.0)]


def test_every_piece_maps_to_one_contiguous_span_of_the_source():
    """The property the whole thing is for: after this, provenance never has to
    report a segment as glued."""
    timeline = TimelineMap([(0.0, 5.0, 0.0), (8.0, 12.0, 5.0), (13.0, 20.0, 9.0)])
    for start, end in ((4.0, 6.0), (0.0, 16.0), (8.5, 9.5), (2.0, 3.0)):
        for piece in split_at_seams([_seg(start, end)], timeline.seams()):
            spans = timeline.spans_to_original(piece["start"], piece["end"])
            assert len(spans) == 1, (
                f"{piece['start']}-{piece['end']} still spans a join: {spans}")


# --- wiring ------------------------------------------------------------------

def test_diarization_splits_at_seams_before_vad():
    """The joins bound what is genuinely continuous, so everything downstream
    should work inside one piece rather than repair a segment spanning two."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "services", "diarization_service.py"),
                  encoding="utf-8").read()
    split = source.index("split_at_seams(")
    vad = source.index("self.vad_model.vad(")
    merge = source.index("cut_by_speaker_label(\n")
    assert split < vad < merge, "seam split, then VAD, then merge"


def test_the_merge_is_told_about_the_seams():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "services", "diarization_service.py"),
                  encoding="utf-8").read()
    assert "seams=seams)" in source, (
        "merging is the only step that could undo the split")


def test_the_pipeline_hands_diarization_the_timeline():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "services", "pipeline_service.py"),
                  encoding="utf-8").read()
    assert "self.diarization_svc.timeline = timeline" in source
