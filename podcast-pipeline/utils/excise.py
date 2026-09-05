"""Cutting stretches out of a recording, and finding your way back afterwards.

Sung passages and standalone music have to leave before anything else runs.
Marking them and letting later stages skip does not work: the diarizer still
clusters on them, ASR still receives them, and song lyrics still reach the
transcript scored as dialogue -- dirty data for a full-duplex corpus, which
would learn a song's timing as conversational turn taking.

Cutting has one consequence that has to be handled rather than lived with:
**every timestamp after a cut moves**. Diarization, separation and ASR all work
in the shortened timeline, while the review page, the exported audio and the
user's own ears work in the original one. `TimelineMap` is the translation, and
carrying it is the price of cutting at all.

The joins are crossfaded rather than butted together. A hard cut between two
unrelated waveforms is a step discontinuity: it clicks, and a diarizer or a VAD
reads it as an acoustic event -- a spurious onset exactly where the evidence is
weakest.
"""

import os

import numpy as np

# Crossfade length at each join. 30ms is long enough that the step is inaudible
# and short enough that it never spans a whole syllable: speech runs here have
# a median of 0.40s, so a fade of this size touches under a tenth of one.
FADE_SECONDS = float(os.environ.get("EXCISE_FADE", "0.030"))


class TimelineMap:
    """Translates between the shortened recording and the original one.

    Held as the kept stretches in original time, in order. Position in the cut
    timeline is their cumulative length, so both directions are a search plus
    an offset.
    """

    def __init__(self, kept=None, fade=0.0):
        # [(orig_start, orig_end, cut_start)] in seconds.
        self.kept = list(kept or [])
        self.fade = fade

    def __bool__(self):
        return bool(self.kept)

    @property
    def removed(self) -> float:
        """Seconds cut out."""
        if not self.kept:
            return 0.0
        span = self.kept[-1][1] - self.kept[0][0]
        return max(0.0, span - sum(b - a for a, b, _ in self.kept))

    def to_original(self, t: float) -> float:
        """Where a cut-timeline instant sits in the recording as delivered."""
        if not self.kept:
            return t
        for start, end, cut_start in self.kept:
            if t < cut_start + (end - start):
                return start + (t - cut_start)
        # Past the end: pin to the last kept sample rather than extrapolating
        # into a stretch that was removed.
        start, end, cut_start = self.kept[-1]
        return end + (t - cut_start - (end - start))

    def to_cut(self, t: float):
        """Where an original instant sits after cutting, or None if removed."""
        if not self.kept:
            return t
        for start, end, cut_start in self.kept:
            if start <= t < end:
                return cut_start + (t - start)
        return None

    def to_json(self) -> dict:
        return {"fade": self.fade,
                "kept": [[round(a, 4), round(b, 4), round(c, 4)] for a, b, c in self.kept]}

    @classmethod
    def from_json(cls, payload) -> "TimelineMap":
        if not payload:
            return cls()
        return cls([tuple(row) for row in payload.get("kept", [])],
                   fade=payload.get("fade", 0.0))


def _merge(spans):
    """Overlapping or touching spans as one."""
    out = []
    for a, b in sorted((float(s[0]), float(s[1])) for s in spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def excise(waveform, sample_rate, spans, fade=FADE_SECONDS, min_keep=0.05):
    """Remove `spans` from `waveform`, crossfading each join.

    Returns (audio, TimelineMap). Kept stretches shorter than `min_keep` are
    dropped with the span beside them: a 20ms island between two cuts is not
    speech anyone can use, and keeping it would put two crossfades back to back
    over almost nothing.
    """
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    total = len(audio)
    if total == 0:
        return audio, TimelineMap()

    duration = total / float(sample_rate)
    cuts = [(max(0.0, a), min(duration, b)) for a, b in _merge(spans) if b > a]
    if not cuts:
        return audio, TimelineMap([(0.0, duration, 0.0)], fade=0.0)

    # Complement of the cuts: what survives.
    kept_spans, cursor = [], 0.0
    for a, b in cuts:
        if a - cursor >= min_keep:
            kept_spans.append((cursor, a))
        cursor = max(cursor, b)
    if duration - cursor >= min_keep:
        kept_spans.append((cursor, duration))

    if not kept_spans:
        return np.zeros(0, dtype=np.float32), TimelineMap()

    fade_n = max(0, int(fade * sample_rate))

    # Overlap-add, not fade-out-then-concatenate. Fading each side to zero and
    # butting the results together dips to silence at the join -- the very
    # artifact the fade exists to avoid. The tail of one piece has to sit on top
    # of the head of the next.
    #
    # Equal-power ramps (sin/cos) because the two sides are uncorrelated speech
    # and sum in power: complementary linear ramps would leave a hole at 0.71
    # amplitude halfway through.
    out = None
    kept, position = [], 0.0

    for start, end in kept_spans:
        i, j = int(start * sample_rate), min(int(end * sample_rate), total)
        piece = audio[i:j].astype(np.float32, copy=True)
        if len(piece) == 0:
            continue

        if out is None:
            kept.append((start, end, 0.0))
            position = len(piece) / float(sample_rate)
            out = piece
            continue

        n = min(fade_n, len(out), len(piece))
        if n > 0:
            ramp = np.sin(np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32))
            head = piece[:n] * ramp                 # fading in
            out[-n:] = out[-n:] * ramp[::-1] + head  # fading out, summed
            tail = piece[n:]
        else:
            tail = piece

        # The piece begins where its fade-in begins, which is `n` samples back
        # from the end of what was already written.
        kept.append((start, end, position - n / float(sample_rate)))
        out = np.concatenate([out, tail])
        position = len(out) / float(sample_rate)

    if out is None:
        return np.zeros(0, dtype=np.float32), TimelineMap()
    return out, TimelineMap(kept, fade=fade)
