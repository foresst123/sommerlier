"""How much non-speech, non-music contamination sits under each stretch.

The pipeline routed on three AudioSet groups out of 527 -- speech, singing,
music -- so a segment recorded beside a motorbike and one recorded in a treated
room were the same thing to it. Cnn14 predicts all 527 on every forward pass,
so the other 524 were being computed and thrown away.

This reads three of them back. It exists to let a dirty segment be **found and
excluded**, not repaired: enhancement was ruled out for this corpus because a
denoiser alters the recording, and what it invents becomes training data for a
conversation that never happened. Fewer honest segments beat more repaired
ones. When the recordings are two-mic the calculation changes and this becomes
a routing decision instead of a filter; until then it is a filter.

Held as the framewise curve rather than as spans, unlike MusicMap. Music is a
routing decision -- strip this stretch, cut that one -- so it has to become
intervals. Noise is a judgement each consumer makes at its own threshold, and
collapsing it to spans this early would bake one threshold into everything
downstream before anyone has seen the distribution on real audio.
"""

import os

import numpy as np

# The three groups, kept separate because they mean different things. Voices in
# the background break diarization and put words in the transcript nobody in
# the conversation said; the other two mostly cost ASR accuracy.
KINDS = ("noise_speech", "noise_env", "noise_room")

# What counts as worth noticing, for logging and as the starting point for any
# filter. Measured, not guessed -- a full 527-label dump of three recordings
# (tools/dump_sslam.py) says the tagger is far sparser on this material than a
# 0..1 score suggests:
#
#     only 2 of 527 labels ever exceed 0.5 in any file: Speech and Music
#     the highest any noise group reached, across all three:
#         noise_speech  0.284   (lm8, crowd/cheering under a song)
#         noise_env     0.257   (vimeanh, birds through a window)
#         noise_room    0.073   (vimeanh, dishes)
#
# An earlier 0.35 here sat above every value ever observed -- it could not have
# fired on anything. 0.10 sits inside the range the three groups actually
# occupy while staying well clear of the 0.001-0.005 floor of a quiet room.
#
# Still provisional in one direction: all three recordings are indoor. Nothing
# here has seen the `Outdoor / noisy environment` stratum of clip_selection.csv,
# which is where the ceiling would rise if it rises anywhere.
NOTICEABLE = float(os.environ.get("NOISE_NOTICEABLE", "0.10"))

# What a segment is labelled when it was measured and no group reached
# NOTICEABLE. A word for "nothing worth naming", not proof of a clean
# recording: NOTICEABLE is provisional and was set from indoor material only.
CLEAN = "clean"


def dominant_kind(breakdown, floor=None):
    """Name the noise a segment carries, from its per-group scores.

    The loudest of the three groups, provided it reaches `floor` (NOTICEABLE
    unless told otherwise); CLEAN when it was measured and nothing does; None
    when nothing was measured at all, which is "not checked", not "clean".

    This puts one threshold into a label, which noise_score deliberately does
    not -- so the per-group numbers travel beside it (`noise_breakdown`), and
    anyone who wants a different cut-off, or a stricter one for voices, applies
    it to those instead of to this word.

    A tie goes to the earlier group in KINDS. That is speech, the one that
    breaks diarization, so the label errs toward the worse diagnosis.
    """
    floor = NOTICEABLE if floor is None else floor
    measured = {k: v for k, v in (breakdown or {}).items()
                if k in KINDS and v is not None}
    if not measured:
        return None
    kind = max(KINDS, key=lambda k: measured.get(k, -1.0))
    return kind if measured[kind] >= floor else CLEAN


class NoiseTrack:
    """Framewise noise strength for one recording, in the ORIGINAL timeline.

    Built before excising, so every lookup takes spans in original time --
    which is exactly what `utils.provenance` puts on each segment.
    """

    def __init__(self, curves=None, fps=100.0):
        self.curves = {k: np.asarray(curves.get(k, []), dtype=np.float32)
                       for k in KINDS} if curves else {k: np.zeros(0, np.float32)
                                                       for k in KINDS}
        self.fps = float(fps) or 100.0

    def __bool__(self):
        return any(len(c) for c in self.curves.values())

    @property
    def combined(self):
        """The loudest of the three at each frame.

        Max rather than sum: a room with an air conditioner and a keyboard is
        not twice as contaminated as one with either, and summing would push
        ordinary rooms past any threshold set for genuinely bad audio.
        """
        present = [c for c in self.curves.values() if len(c)]
        if not present:
            return np.zeros(0, dtype=np.float32)
        width = min(len(c) for c in present)
        return np.max(np.stack([c[:width] for c in present]), axis=0)

    def _frames(self, spans, curve):
        if not len(curve):
            return np.zeros(0, dtype=np.float32)
        pieces = []
        for start, end in spans:
            lo = max(0, int(float(start) * self.fps))
            hi = min(len(curve), int(np.ceil(float(end) * self.fps)))
            if hi > lo:
                pieces.append(curve[lo:hi])
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def score_spans(self, spans, percentile: float = 90.0):
        """One number for a set of original-time spans, or None if unmeasured.

        The 90th percentile rather than the mean or the max. The mean lets a
        long clean segment hide a second of traffic; the max makes one frame of
        a door closing condemn a whole turn. What matters for excluding a
        segment is whether the contamination is sustained.
        """
        frames = self._frames(spans, self.combined)
        if not len(frames):
            return None
        return round(float(np.percentile(frames, percentile)), 4)

    def breakdown(self, spans, percentile: float = 90.0):
        """The same score per kind, so a filter can treat voices differently."""
        out = {}
        for kind in KINDS:
            frames = self._frames(spans, self.curves[kind])
            out[kind] = (round(float(np.percentile(frames, percentile)), 4)
                         if len(frames) else None)
        return out

    # -- where the contamination is, given a level per group ---------------
    #
    # `limits` maps a group name to the level at which that group is worth
    # noticing ({"noise_speech": 0.10, "noise_env": 0.15}). A frame is "over" when
    # ANY listed group reaches its own level, exactly at the level counting --
    # the same rule `dominant_kind` uses. Groups left out of `limits`, or that
    # were not measured, are ignored, so asking about voices alone is one entry.
    # Everything returns None or [] for a span nobody measured: "not checked" is
    # never turned into "clean".

    def _over(self, lo, hi, limits):
        """Boolean per frame in [lo, hi), or None when no listed group has any."""
        pieces = []
        for kind, limit in limits.items():
            curve = self.curves.get(kind)
            if curve is None or not len(curve):
                continue
            top = min(len(curve), hi)
            if top > lo:
                pieces.append(curve[lo:top] >= limit)
        if not pieces:
            return None
        width = min(len(p) for p in pieces)
        return np.logical_or.reduce([p[:width] for p in pieces])

    def _span_frames(self, start, end):
        return (max(0, int(float(start) * self.fps)),
                int(np.ceil(float(end) * self.fps)))

    @staticmethod
    def _longest_true_run(flags) -> int:
        if flags is None or not len(flags) or not flags.any():
            return 0
        edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
        return int((np.where(edges == -1)[0] - np.where(edges == 1)[0]).max())

    def share_over(self, spans, limits):
        """Fraction of the frames in `spans` that are over, or None if unmeasured."""
        over = total = 0
        for start, end in spans:
            flags = self._over(*self._span_frames(start, end), limits)
            if flags is not None:
                over += int(flags.sum())
                total += len(flags)
        return round(over / total, 4) if total else None

    def longest_run_over(self, spans, limits):
        """Longest unbroken stretch over, in seconds; None if unmeasured.

        Per span, never across two: the pieces of a glued segment are not
        contiguous in the recording, and a run that appears to bridge them would
        be two short ones.
        """
        longest, measured = 0, False
        for start, end in spans:
            flags = self._over(*self._span_frames(start, end), limits)
            if flags is not None:
                measured = True
                longest = max(longest, self._longest_true_run(flags))
        return round(longest / self.fps, 3) if measured else None

    def sustained_spans(self, limits, min_seconds: float):
        """Stretches of the whole recording over for at least `min_seconds`.

        (start, end) in the original timeline, in order. The filter that breaks a
        conversation at a lasting noise reads this once per recording, so a run
        that straddles two turns is one stretch rather than two short ones that
        each look harmless.
        """
        widest = max((len(c) for c in self.curves.values()), default=0)
        flags = self._over(0, widest, limits)
        if flags is None:
            return []
        need = max(1, int(np.ceil(float(min_seconds) * self.fps)))
        edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        return [(round(float(a) / self.fps, 3), round(float(b) / self.fps, 3))
                for a, b in zip(starts, ends) if b - a >= need]

    def to_json(self, decimals: int = 3) -> dict:
        """Rounded so a 50-minute track is kilobytes, not megabytes.

        300k frames of float32 is 1.2MB per kind before rounding. Three
        decimals is well below any threshold anyone will set on a 0..1 score.
        """
        return {"fps": self.fps,
                "curves": {k: [round(float(v), decimals) for v in c]
                           for k, c in self.curves.items() if len(c)}}

    @classmethod
    def from_json(cls, payload) -> "NoiseTrack":
        if not payload:
            return cls()
        return cls(payload.get("curves", {}), fps=payload.get("fps", 100.0))


def build(scores, fps) -> NoiseTrack:
    """A track from the framewise scores a tagger's `tag_framewise` returns.

    Takes the scores rather than the waveform on purpose: tagging a recording
    costs a full sweep, and the music map already pays for one. Reading
    the noise groups out of that same result is free; a second sweep would
    double the cost of the stage for nothing.
    """
    if not scores:
        return NoiseTrack()
    curves = {k: scores[k] for k in KINDS if k in scores and len(scores[k])}
    return NoiseTrack(curves, fps=fps) if curves else NoiseTrack()
