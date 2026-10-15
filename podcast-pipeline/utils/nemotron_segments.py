"""Turn Nemotron-3-Diarization output into the pipeline's diarization segments.

No NeMo or torch import here, so it is importable (and testable) anywhere. The
worker returns the same {"start", "end", "speaker"} dicts as the DiariZen one.
"""

import re

# Model card: up to eight speakers, channels ordered by first arrival.
MAX_SPEAKERS = 8


def _parse_one(raw):
    """Return (start, end, speaker_id) for one raw segment.

    Accepted shapes: "0.00 3.20 speaker_0" (assumed NeMo format; the card only
    says 'begin_seconds, end_seconds, speaker_index'), (start, end, speaker),
    and the Transformers processor's {"Start", "End", "Speaker"} dict.
    """
    if isinstance(raw, dict):
        return float(raw["Start"]), float(raw["End"]), raw["Speaker"]
    if isinstance(raw, str):
        parts = raw.replace(",", " ").split()
    else:
        parts = list(raw)
    if len(parts) != 3:
        raise ValueError(f"Unrecognised Nemotron segment: {raw!r}")
    return float(parts[0]), float(parts[1]), parts[2]


def _speaker_index(label):
    """'speaker_3' / 3 / '3' -> 3."""
    match = re.search(r"\d+", str(label))
    if match is None:
        raise ValueError(f"Speaker label without an index: {label!r}")
    return int(match.group())


def to_pipeline_segments(raw_segments, merge_gap=0.0, min_duration=0.0):
    """Convert raw model segments to sorted pipeline segments.

    Speakers are renamed SPEAKER_00, SPEAKER_01, ... in order of first
    appearance (DiariZen's naming). Same-speaker turns separated by at most
    ``merge_gap`` seconds are joined, then turns shorter than ``min_duration``
    are dropped. Overlapping turns of different speakers are kept as-is.
    """
    turns = []
    for raw in raw_segments or []:
        start, end, speaker = _parse_one(raw)
        if end > start:
            turns.append((start, end, _speaker_index(speaker)))
    turns.sort()

    names = {}
    for _, _, speaker in turns:
        names.setdefault(speaker, f"SPEAKER_{len(names):02d}")

    merged = []
    open_turn = {}  # speaker -> index into merged of its latest turn
    for start, end, speaker in turns:
        at = open_turn.get(speaker)
        if at is not None and start - merged[at]["end"] <= merge_gap:
            merged[at]["end"] = max(merged[at]["end"], end)
            continue
        open_turn[speaker] = len(merged)
        merged.append({"start": start, "end": end, "speaker": names[speaker]})

    return [seg for seg in merged if seg["end"] - seg["start"] >= min_duration]
