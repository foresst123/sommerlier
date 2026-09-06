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

    def seams(self):
        """Where the joins are, in cut-timeline seconds.

        Each one is a point where two stretches of the recording that were
        never adjacent now are. Anything that widens a span has to stop at
        them: reaching across a seam picks up audio from somewhere else in the
        recording entirely.
        """
        out, position = [], 0.0
        for start, end, _cut_start in self.kept[:-1]:
            position += end - start
            out.append(position)
        return out

    def removed_spans(self, duration: float = None):
        """What this timeline took out, in original time.

        The complement of `kept`, and the reason it exists: re-entering the
        pipeline for a later stage reloads the whole recording and has to cut
        it the same way again. Recomputing the cuts from the music map is not
        the same thing -- the map is rebuilt from settings that may have
        changed since, and a mismatch there silently puts diarization on a
        different timeline from every timestamp that follows it. This is the
        record of what actually happened.

        `duration` is the original length, needed only to notice a recording
        that ended inside a cut; without it the tail is assumed kept.
        """
        if not self.kept:
            return []
        out = []
        first_start = self.kept[0][0]
        if first_start > 0:
            out.append((0.0, first_start))
        for (_, prev_end, _), (next_start, _, _) in zip(self.kept, self.kept[1:]):
            if next_start > prev_end:
                out.append((prev_end, next_start))
        last_end = self.kept[-1][1]
        if duration is not None and duration > last_end:
            out.append((last_end, duration))
        return out

    def spans_to_original(self, start: float, end: float):
        """The original ranges a cut-timeline interval is actually made of.

        `to_original` maps an instant, which is not enough for an interval that
        crosses a join: [to_original(start), to_original(end)] would name a
        range in the recording that includes the stretch removed between them --
        audio this interval does not contain and never did. A segment straddling
        a join is two pieces of the original glued together, and the honest
        answer is both of them.

        Returns [] for an empty or reversed interval, and a single span covering
        the whole request when nothing was cut.
        """
        if end <= start:
            return []
        if not self.kept:
            return [(start, end)]

        out = []
        for orig_start, orig_end, cut_start in self.kept:
            cut_end = cut_start + (orig_end - orig_start)
            lo, hi = max(start, cut_start), min(end, cut_end)
            if hi <= lo:
                continue
            out.append((orig_start + (lo - cut_start),
                        orig_start + (hi - cut_start)))
        return out

    def crosses_cut(self, start: float, end: float) -> bool:
        """Whether this interval is made of more than one piece of the original."""
        return len(self.spans_to_original(start, end)) > 1

    def cut_between(self, start: float, end: float) -> bool:
        """Whether a join lies in the cut-timeline interval [start, end].

        This is what makes a silence between two turns trustworthy or not. A
        pause measured across a join is not a pause: the speakers were separated
        by however much was removed there, and a full-duplex corpus that reads
        it as turn timing learns a gap that never happened.

        Touching endpoints count. A turn ending exactly on a join is followed by
        removed audio whatever the arithmetic says.
        """
        if end < start or not self.kept:
            return False
        for orig_start, orig_end, cut_start in self.kept[:-1]:
            join = cut_start + (orig_end - orig_start)
            if start <= join <= end:
                return True
        return False

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
