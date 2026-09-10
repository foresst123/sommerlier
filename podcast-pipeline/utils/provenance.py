"""Where each finished segment came from in the recording as delivered.

Everything from diarization onward works in the cut timeline -- the recording
with its sung and standalone-music stretches removed and the remainder
crossfaded together. That timeline is an internal convenience. The file on
disk, the review page and anyone assembling conversations out of this corpus
all work in the original one, and until now nothing translated between them:
`TimelineMap.to_original` existed and was called only by its own tests.

Three marks are needed, not one, and the third is the one that is easy to miss:

    orig_spans   the original ranges the segment's audio is made of. A list,
                 because a segment straddling a join is two pieces of the
                 recording glued together and a single (start, end) pair would
                 name a range that includes the audio removed between them.

    crosses_cut  whether that happened. Such a segment is still perfectly good
                 speech, but it is not a contiguous observation of the source,
                 which matters to anything measuring timing.

    gap_before   the silence before this segment, in seconds -- or None when a
                 join lies in that silence. None means "unknowable", not zero.
                 A pause measured across a join is however much audio was
                 removed there, and a full-duplex corpus that reads it as turn
                 timing learns a gap that never happened. That is the same
                 failure excising was introduced to prevent, one stage later
                 and harder to see.

No audio is touched here. This module only writes down what earlier stages
already did.
"""

from utils.excise import TimelineMap


def annotate(segments, timeline=None, noise=None):
    """Mark each segment with where it came from. Returns the same list.

    `segments` are in the cut timeline, ordered by start. `timeline` may be
    None or empty, which is the ordinary case for a recording nothing was cut
    from -- every segment then maps to itself and no gap is suspect.

    `noise` is an optional NoiseTrack, scored over the original ranges rather
    than the cut ones so a segment glued from two pieces is judged on the audio
    it actually holds.
    """
    timeline = timeline or TimelineMap()
    previous_end = None

    for seg in segments:
        start, end = float(seg.start), float(seg.end)
        spans = timeline.spans_to_original(start, end)

        seg.orig_spans = [{"start": round(a, 3), "end": round(b, 3)}
                          for a, b in spans]
        seg.crosses_cut = len(spans) > 1

        if previous_end is None:
            # Nothing before the first segment to measure a turn against. The
            # lead-in is not a pause between speakers.
            seg.gap_before = None
        elif timeline.cut_between(previous_end, start):
            seg.gap_before = None
        else:
            # Negative for overlapping turns, which is real and worth keeping:
            # interruption is the phenomenon a full-duplex corpus is for.
            seg.gap_before = round(start - previous_end, 3)

        if noise is not None:
            seg.noise_score = noise.score_spans(spans)

        previous_end = end

    return segments


def summary(segments) -> dict:
    """Counts worth putting in a manifest, so a bad run is visible at a glance."""
    total = len(segments)
    glued = sum(1 for s in segments if getattr(s, "crosses_cut", False))
    # The first segment's None is expected; anything beyond that is a join.
    unknown_gap = sum(1 for s in segments[1:]
                      if getattr(s, "gap_before", None) is None)
    scored = [s.noise_score for s in segments
              if getattr(s, "noise_score", None) is not None]
    out = {"segments": total,
           "segments_crossing_a_cut": glued,
           "gaps_broken_by_a_cut": unknown_gap}
    if scored:
        ordered = sorted(scored)
        out["noise_p50"] = round(ordered[len(ordered) // 2], 3)
        out["noise_max"] = round(ordered[-1], 3)
    return out
