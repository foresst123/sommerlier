"""What kind of audio is playing, at 10ms resolution, across a recording.

Three questions get different answers and need to be told apart:

    someone is singing        -> mark it, and keep it out of the transcript.
                                 Song lyrics scored as dialogue are dirty data
                                 for a full-duplex corpus: the model would
                                 learn a song's timing as conversational turn
                                 taking.
    speech over a music bed   -> separate, then transcribe.
    speech, nothing under it  -> leave it alone. On this corpus that is
                                 almost everything: 30s of music in 50 minutes.

PANNs answers all three at once -- Cnn14 predicts 527 AudioSet labels on every
forward pass, `Speech`, `Singing` and `Music` among them. The earlier version
of this module read one of those labels and slid a 2s window; this one uses
`SoundEventDetection`, which labels every 10ms frame directly, and reads three
groups instead of one.

The map is also what keeps enrolment off music beds: separation runs before
music removal, so its search for clean solo speech would otherwise happen over
audio that still has the bed in it, and an ECAPA embedding taken from
speech-over-music describes both.
"""

import os

import numpy as np

# Above this a frame counts as carrying the thing. Deliberately below the 0.5
# a classifier would use: the cost is asymmetric. A frame wrongly called
# musical removes one enrolment candidate from a pool of hundreds, while a
# missed one contaminates an enrolment or sends song lyrics to ASR.
MUSIC_THRESHOLD = float(os.environ.get("MUSIC_MAP_THRESHOLD", "0.35"))
SINGING_THRESHOLD = float(os.environ.get("MUSIC_MAP_SINGING", "0.35"))

# Singing and speech both light up the vocal range, so a frame can score on
# both. It is called singing only when singing leads by this margin -- without
# it, ordinary speech with a music bed under it reads as singing and gets
# dropped from the transcript.
SINGING_MARGIN = float(os.environ.get("MUSIC_MAP_SINGING_MARGIN", "0.15"))

# Shortest run worth recording. Below this the label is a flicker between two
# genuinely different stretches rather than a stretch of its own.
MIN_SPAN_SECONDS = float(os.environ.get("MUSIC_MAP_MIN_SPAN", "0.30"))

# Gap below which two runs of the same kind are one. Music dips under the
# threshold on a beat rest without stopping.
MERGE_GAP_SECONDS = float(os.environ.get("MUSIC_MAP_MERGE_GAP", "0.50"))

# What a span can be, and what each one means for the audio.
#
#   MUSIC    a bed with someone talking over it. Strip the bed, keep the speech.
#   SINGING  the voice itself is singing. Cut it: the thing to remove is the
#            voice, so stripping a bed would leave the lyrics behind.
#   SONG     music with nobody speaking -- an intro, an outro, a sting. Cut it
#            too. Running vocal separation here would extract whatever the
#            model imagines a voice to be out of an instrumental.
MUSIC = "music"
SINGING = "singing"
SONG = "song"

# Spans that leave the recording entirely rather than being cleaned.
EXCISED = (SINGING, SONG)

# Below this the frame carries no speech worth keeping, so music there is a
# song rather than a bed.
SPEECH_PRESENT = float(os.environ.get("MUSIC_MAP_SPEECH_PRESENT", "0.20"))


class MusicMap:
    """Musical and sung stretches of one recording, queryable by time span."""

    def __init__(self, spans=None, fps=100.0):
        # Sorted (start, end, kind) in seconds. A bare (start, end) is taken as
        # music, which is what every span meant before singing was told apart.
        normalised = [tuple(s) if len(s) > 2 else (s[0], s[1], MUSIC)
                      for s in (spans or [])]
        self.spans = sorted(normalised, key=lambda s: s[0])
        self.fps = fps

    def __bool__(self):
        return bool(self.spans)

    def __len__(self):
        return len(self.spans)

    def _of_kind(self, kind=None):
        return [s for s in self.spans if kind is None or s[2] == kind]

    @property
    def total(self) -> float:
        return sum(b - a for a, b, _ in self.spans)

    def total_of(self, kind) -> float:
        return sum(b - a for a, b, k in self.spans if k == kind)

    def overlaps(self, start: float, end: float, kind=None) -> bool:
        """Whether [start, end) touches a stretch of `kind` (or any kind)."""
        if end <= start:
            return False
        for a, b, k in self.spans:
            if a >= end:
                break
            if b > start and (kind is None or k == kind):
                return True
        return False

    def is_singing(self, start: float, end: float) -> bool:
        return self.overlaps(start, end, SINGING)

    def clean_parts(self, start: float, end: float, kind=None):
        """`[start, end)` with every matching stretch cut out of it."""
        if end <= start:
            return []
        parts = [(start, end)]
        for a, b, k in self.spans:
            if kind is not None and k != kind:
                continue
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
        return {"fps": self.fps,
                "spans": [[round(a, 3), round(b, 3), k] for a, b, k in self.spans]}

    @classmethod
    def from_json(cls, payload) -> "MusicMap":
        if not payload:
            return cls()
        spans = []
        for row in payload.get("spans", []):
            # Maps written before spans carried a kind held [start, end] only.
            spans.append((row[0], row[1], row[2] if len(row) > 2 else MUSIC))
        return cls(spans, fps=payload.get("fps", 100.0))

    def remap(self, timeline):
        """This map expressed in a cut recording's timeline.

        After the sung stretches are removed, every later stage works in the
        shortened timeline while these spans are still in the original one.
        Comparing the two would silently mis-locate every bed.

        Spans that were themselves cut out disappear; a span straddling a cut
        keeps whichever parts survived.
        """
        if not timeline or not self.spans:
            return MusicMap(fps=self.fps)

        moved = []
        for start, end, kind in self.spans:
            for keep_start, keep_end, cut_start in timeline.kept:
                lo, hi = max(start, keep_start), min(end, keep_end)
                if hi <= lo:
                    continue
                offset = cut_start - keep_start
                moved.append((lo + offset, hi + offset, kind))
        return MusicMap(moved, fps=self.fps)

    def excised_spans(self):
        """The stretches that leave the recording, in order."""
        return [(a, b, k) for a, b, k in self.spans if k in EXCISED]

    def summary(self) -> dict:
        return {"spans": len(self.spans),
                "music_seconds": round(self.total_of(MUSIC), 2),
                "singing_seconds": round(self.total_of(SINGING), 2),
                "song_seconds": round(self.total_of(SONG), 2)}


def _runs(flags, fps, min_span, merge_gap):
    """Contiguous True runs of `flags`, merged and filtered, in seconds."""
    if not len(flags):
        return []
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    spans = []
    for i, j in zip(starts, ends):
        a, b = i / fps, j / fps
        if spans and a - spans[-1][1] <= merge_gap:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return [s for s in spans if s[1] - s[0] >= min_span]


def build(waveform, sample_rate, detector, logger=None,
          music_threshold=MUSIC_THRESHOLD, singing_threshold=SINGING_THRESHOLD,
          singing_margin=SINGING_MARGIN):
    """Label the recording frame by frame and collapse the result into spans.

    An empty map is returned when there is no detector, which every caller
    already reads as "no reason to avoid anything" -- the right default when
    the check did not run rather than a claim that the audio is clean.
    """
    if detector is None or waveform is None or not len(waveform):
        return MusicMap()

    tag = getattr(detector, "tag_framewise", None)
    if tag is None:
        if logger:
            logger.warning("Detector has no frame-level tagging; music map is empty")
        return MusicMap()

    try:
        scores, fps = tag(waveform, sample_rate)
    except Exception as exc:                       # pragma: no cover - model path
        if logger:
            logger.warning(f"Music sweep failed: {exc}")
        return MusicMap()

    speech, singing, music = scores["speech"], scores["singing"], scores["music"]
    if not len(music):
        return MusicMap(fps=fps)

    # Singing first, and exclusively: a sung frame is not also a bed to clean up,
    # because the thing to remove would be the voice itself.
    is_singing = (singing >= singing_threshold) & (singing >= speech + singing_margin)

    # Then split the rest of the music by whether anyone is talking over it. A
    # bed under speech is worth stripping; music with no speech is worth cutting,
    # and asking a vocal separator to work on it would only invent a voice.
    loud_music = (music >= music_threshold) & ~is_singing
    is_song = loud_music & (speech < SPEECH_PRESENT)
    is_music = loud_music & ~is_song

    spans = ([(a, b, SINGING) for a, b in
              _runs(is_singing, fps, MIN_SPAN_SECONDS, MERGE_GAP_SECONDS)]
             + [(a, b, SONG) for a, b in
                _runs(is_song, fps, MIN_SPAN_SECONDS, MERGE_GAP_SECONDS)]
             + [(a, b, MUSIC) for a, b in
                _runs(is_music, fps, MIN_SPAN_SECONDS, MERGE_GAP_SECONDS)])

    found = MusicMap(spans, fps=fps)
    if logger:
        duration = len(waveform) / max(sample_rate, 1)
        logger.info(
            f"Music map: {found.total_of(SINGING):.1f}s singing, "
            f"{found.total_of(MUSIC):.1f}s music under speech, of "
            f"{duration:.1f}s ({found.total / max(duration, 1e-9) * 100:.1f}%), "
            f"{len(found)} span(s) at {fps:.0f} fps")
    return found
