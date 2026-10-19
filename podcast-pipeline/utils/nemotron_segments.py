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


# NeMo's post-processing YAML for Sortformer (a `parameters:` block). Values are
# those NeMo ships for CallHome part 1 (sortformer_diar_4spk-v1_callhome-part1.yaml);
# they were tuned for other Sortformer models, not for Nemotron-3, so treat them as
# a starting point. Keys not given in the profile take these values.
POSTPROCESSING_KEYS = ("onset", "offset", "pad_onset", "pad_offset",
                       "min_duration_on", "min_duration_off")
POSTPROCESSING_BASE = {"onset": 0.53, "offset": 0.49, "pad_onset": 0.23, "pad_offset": 0.01,
                       "min_duration_on": 0.42, "min_duration_off": 0.34}


def postprocessing_params(cfg):
    """The six thresholds from ``models.nemotron.postprocessing``, or None when it is not set.

    Raises ValueError on an unknown key so a typo is not a silently ignored setting.
    """
    section = (cfg or {}).get("postprocessing")
    if not section:
        return None
    unknown = sorted(set(section) - set(POSTPROCESSING_KEYS))
    if unknown:
        raise ValueError(f"unknown models.nemotron.postprocessing key(s): {', '.join(unknown)}")
    return {key: float(section.get(key, POSTPROCESSING_BASE[key])) for key in POSTPROCESSING_KEYS}


def write_postprocessing_yaml(params, directory):
    """Write ``params`` as the YAML NeMo's ``postprocessing_yaml`` expects; returns its path."""
    import os
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "nemotron_postprocessing.yaml")
    lines = ["parameters:"] + [f"  {key}: {params[key]}" for key in POSTPROCESSING_KEYS]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path
