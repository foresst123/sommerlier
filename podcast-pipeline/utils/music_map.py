"""Where music plays in a recording, as a timeline the whole pipeline can read.

Separation runs before music removal, so when the stitched window goes looking
for clean solo speech to enrol a speaker on, it is searching audio that still
has whatever music was under it. `_pick_solo` picks on length and proximity
alone, so nothing stops it handing Sidon five seconds of speech over a music
bed -- and the ECAPA embedding taken from that describes the speaker plus the
music, which is not the thing being matched against.

Reordering the stages would fix it too, but music removal depends on Demucs
having run over the whole file and separation already checkpoints between the
two, so the cheap version is this: sweep PANNs across the timeline once while
the diarizer's audio is already in memory, keep the verdicts, and let any later
stage ask "was there music here?" without loading a model.

The map is deliberately coarse. It answers one question -- is this stretch safe
to enrol on -- and a false positive costs a slightly worse choice of solo span,
never a dropped segment.
"""

import os

import numpy as np

# PANNs (Cnn14) needs about a second of audio to pass its pooling stack; below
# that it returns "no music" regardless of content. The window is set well
# above that minimum so a verdict is never taken from a clip too short to have
# one.
WINDOW_SECONDS = float(os.environ.get("MUSIC_MAP_WINDOW", "2.0"))

# Half-window steps: a music cue that starts mid-window still lands fully
# inside the next one, so onsets are caught within a second.
HOP_SECONDS = float(os.environ.get("MUSIC_MAP_HOP", "1.0"))

# Probability above which a window counts as music. PANNs' own detect_music
# defaults to 0.5; this sits lower because the cost here is asymmetric -- a
# window wrongly called musical only removes one candidate from a pool of
# hundreds, while a missed one contaminates an enrollment.
THRESHOLD = float(os.environ.get("MUSIC_MAP_THRESHOLD", "0.35"))


class MusicMap:
    """Musical stretches of one recording, queryable by time span."""

    def __init__(self, spans=None, window=WINDOW_SECONDS):
        # Sorted, merged (start, end) in seconds.
        self.spans = sorted(spans or [])
        self.window = window

    def __bool__(self):
        return bool(self.spans)

    def __len__(self):
        return len(self.spans)

    @property
    def total(self) -> float:
        return sum(b - a for a, b in self.spans)

    def overlaps(self, start: float, end: float) -> bool:
        """Whether [start, end) touches any musical stretch."""
        if not self.spans or end <= start:
            return False
        # Linear is fine: a map has tens of spans, not thousands, and this is
        # called once per candidate rather than per sample.
        for a, b in self.spans:
            if a >= end:
                break
            if b > start:
                return True
        return False

    def clean_parts(self, start: float, end: float):
        """`[start, end)` with every musical stretch cut out of it."""
        if end <= start:
            return []
        parts = [(start, end)]
        for a, b in self.spans:
            if a >= end:
                break
            if b <= start:
                continue
            trimmed = []
            for lo, hi in parts:
                if b <= lo or a >= hi:
                    trimmed.append((lo, hi))
                    continue
                if lo < a:
                    trimmed.append((lo, a))
                if b < hi:
                    trimmed.append((b, hi))
            parts = trimmed
        return parts

    def to_json(self) -> dict:
        return {"window": self.window,
                "spans": [[round(a, 3), round(b, 3)] for a, b in self.spans]}

    @classmethod
    def from_json(cls, payload) -> "MusicMap":
        if not payload:
            return cls()
        return cls([tuple(s) for s in payload.get("spans", [])],
                   window=payload.get("window", WINDOW_SECONDS))


def _merge(spans, gap=0.0):
    """Merge overlapping or touching spans."""
    out = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return [tuple(s) for s in out]


def build(waveform, sample_rate, detector, logger=None,
          window=WINDOW_SECONDS, hop=HOP_SECONDS, threshold=THRESHOLD) -> MusicMap:
    """Sweep `detector` across the recording and collect the musical stretches.

    `detector` is a PANNSDetector; anything exposing
    ``detect_music(array, sample_rate) -> (bool, float)`` works.

    Returns an empty map when there is no detector, which is the same answer as
    "no music found" for every caller -- they all treat an empty map as "no
    reason to avoid anything", and that is the right default when the check did
    not run.
    """
    if detector is None or waveform is None or not len(waveform):
        return MusicMap()

    step = max(1, int(hop * sample_rate))
    size = max(step, int(window * sample_rate))
    total = len(waveform)

    hits = []
    checked = 0
    for start in range(0, max(1, total - size + 1), step):
        chunk = waveform[start:start + size]
        if len(chunk) < size:
            break
        try:
            _, probability = detector.detect_music(chunk, sample_rate)
        except Exception as exc:  # pragma: no cover - depends on the model
            if logger:
                logger.warning(f"Music sweep failed at {start / sample_rate:.1f}s: {exc}")
            break
        checked += 1
        if probability >= threshold:
            hits.append((start / sample_rate, (start + size) / sample_rate))

    spans = _merge(hits)
    if logger:
        span_total = sum(b - a for a, b in spans)
        logger.info(
            f"Music map: {len(spans)} musical stretch(es), {span_total:.1f}s of "
            f"{total / sample_rate:.1f}s ({span_total / max(total / sample_rate, 1e-9) * 100:.1f}%), "
            f"from {checked} window(s)")
    return MusicMap(spans, window=window)
