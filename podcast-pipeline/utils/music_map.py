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
MUSIC_THRESHOLD = float(os.environ.get("MUSIC_MAP_THRESHOLD", "0.10"))

# 0.35 was above anything this tagger produces, so the SINGING branch never
# fired once: a full 527-label dump (tools/dump_panns.py) of three recordings
# put the ceiling of the singing group at 0.205, on the one file that actually
# contains a song. Every sung stretch fell through to SONG instead.
#
# 0.12 is the lowest level that still separates the two cases cleanly:
#
#     level   two files with no singing    the file with a real song
#     0.05            1.3s                        103.4s
#     0.10            0.6s                         39.4s
#     0.12            0.0s                         23.0s   <- clean
#     0.20            0.0s                          0.3s
#     0.35            0.0s                          0.0s   <- what it was
#
# Below 0.12 the "Male singing" label starts firing on ordinary speech in
# vimeanhphanchiatay; above it, real singing is thrown away for nothing.
#
# Expect no change in what gets cut on the current corpus: every frame over
# 0.10 already sat inside a SONG span, which is excised anyway. What this
# restores is the branch's ability to fire at all -- which matters for the one
# case SONG cannot catch, someone singing *over* speech.
#
# Thin evidence, and worth saying so: one recording with real singing.
SINGING_THRESHOLD = float(os.environ.get("MUSIC_MAP_SINGING", "0.12"))

# Singing and speech both light up the vocal range, so a frame can score on
# both. It is called singing only when singing leads by this margin -- without
# it, ordinary speech with a music bed under it reads as singing and gets
# dropped from the transcript.
SINGING_MARGIN = float(os.environ.get("MUSIC_MAP_SINGING_MARGIN", "0.15"))

# Shortest run worth recording -- and it is two numbers, because the two
# decisions this map drives are not equally reversible.
#
# The old single 0.30 filtered nothing, and the code said so without acting on
# it: Cnn14_DecisionLevelMax decides once per 320ms and repeats that decision
# across the 32 frames it covers, so the shortest run it can produce is already
# longer than 0.30. Every single 320ms block became a span, then grew by
# PAD_SECONDS on each side into a 0.92s one. Nine such spans across three
# recordings.
#
# MUSIC only strips a bed and writes vocals back. A false one costs a separator
# pass on speech that did not need it -- wasteful, not destructive -- so it
# stays at one block, and raising it would throw away real music: on
# vimeanhphanchiatay, 0.96 would drop 10 spans and 6.1s of genuine bed.
#
# SINGING and SONG delete audio from the recording permanently. That asks for
# more than one block of evidence, and the cost of asking is small: across the
# two files with sung or standalone music, 0.96 drops 6 fragment spans and
# 2.9 seconds total.
MIN_SPAN_SECONDS = float(os.environ.get("MUSIC_MAP_MIN_SPAN", "0.32"))
MIN_SPAN_EXCISED = float(os.environ.get("MUSIC_MAP_MIN_SPAN_CUT", "0.96"))

# Gap below which two runs of the same kind are one. Music dips under the
# threshold on a beat rest without stopping.
MERGE_GAP_SECONDS = float(os.environ.get("MUSIC_MAP_MERGE_GAP", "0.50"))

# Margin to pad around detected music/singing spans to catch the fade-in/out
PAD_SECONDS = float(os.environ.get("MUSIC_MAP_PAD", "0.30"))

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


def _runs(flags, fps, min_span, merge_gap, pad=0.0):
    """Contiguous True runs of `flags`, merged and filtered, in seconds."""
    if not len(flags):
        return []
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    raw_spans = []
    for i, j in zip(starts, ends):
        a, b = i / fps, j / fps
        if raw_spans and a - raw_spans[-1][1] <= merge_gap:
            raw_spans[-1] = (raw_spans[-1][0], b)
        else:
            raw_spans.append((a, b))
            
    max_dur = len(flags) / fps
    spans = []
    for a, b in raw_spans:
        if b - a < min_span:
            continue
            
        a = max(0.0, a - pad)
        b = min(max_dur, b + pad)
        
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], b))
        else:
            spans.append((a, b))
            
    return spans


def build(waveform, sample_rate, detector, logger=None,
          music_threshold=MUSIC_THRESHOLD, singing_threshold=SINGING_THRESHOLD,
          singing_margin=SINGING_MARGIN):
    """The music map alone. See `build_maps` for the noise track beside it."""
    return build_maps(waveform, sample_rate, detector, logger=logger,
                      music_threshold=music_threshold,
                      singing_threshold=singing_threshold,
                      singing_margin=singing_margin)[0]


def build_maps(waveform, sample_rate, detector, logger=None,
               music_threshold=MUSIC_THRESHOLD, singing_threshold=SINGING_THRESHOLD,
               singing_margin=SINGING_MARGIN):
    """Label the recording frame by frame; return (MusicMap, NoiseTrack).

    One PANNs sweep produces both. Cnn14 predicts all 527 AudioSet labels on
    every forward pass, so the noise groups cost nothing beyond reading columns
    that were already computed -- a second sweep would double the price of this
    stage for no new information.

    Empty maps are returned when there is no detector, which every caller
    already reads as "no reason to avoid anything" -- the right default when
    the check did not run rather than a claim that the audio is clean.
    """
    from utils.noise_map import NoiseTrack, build as build_noise

    if detector is None or waveform is None or not len(waveform):
        return MusicMap(), NoiseTrack()

    tag = getattr(detector, "tag_framewise", None)
    if tag is None:
        if logger:
            logger.warning("Detector has no frame-level tagging; music map is empty")
        return MusicMap(), NoiseTrack()

    try:
        scores, fps = tag(waveform, sample_rate)
    except Exception as exc:                       # pragma: no cover - model path
        if logger:
            logger.warning(f"Music sweep failed: {exc}")
        return MusicMap(), NoiseTrack()

    noise = build_noise(scores, fps)

    speech, singing, music = scores["speech"], scores["singing"], scores["music"]
    if not len(music):
        return MusicMap(fps=fps), noise

    # Singing first, and exclusively: a sung frame is not also a bed to clean up,
    # because the thing to remove would be the voice itself.
    is_singing = (singing >= singing_threshold) & (singing >= speech + singing_margin)

    # Then split the rest of the music by whether anyone is talking over it. A
    # bed under speech is worth stripping; music with no speech is worth cutting,
    # and asking a vocal separator to work on it would only invent a voice.
    loud_music = (music >= music_threshold) & ~is_singing
    is_song = loud_music & (speech < SPEECH_PRESENT)
    is_music = loud_music & ~is_song

    # SINGING and SONG leave the recording; MUSIC is only cleaned. The first two
    # therefore have to clear a longer run than the third -- see MIN_SPAN_*.
    spans = ([(a, b, SINGING) for a, b in
              _runs(is_singing, fps, MIN_SPAN_EXCISED, MERGE_GAP_SECONDS, PAD_SECONDS)]
             + [(a, b, SONG) for a, b in
                _runs(is_song, fps, MIN_SPAN_EXCISED, MERGE_GAP_SECONDS, PAD_SECONDS)]
             + [(a, b, MUSIC) for a, b in
                _runs(is_music, fps, MIN_SPAN_SECONDS, MERGE_GAP_SECONDS, PAD_SECONDS)])

    found = MusicMap(spans, fps=fps)
    if logger:
        duration = len(waveform) / max(sample_rate, 1)
        logger.info(
            f"Music map: {found.total_of(SINGING):.1f}s singing, "
            f"{found.total_of(MUSIC):.1f}s music under speech, of "
            f"{duration:.1f}s ({found.total / max(duration, 1e-9) * 100:.1f}%), "
            f"{len(found)} span(s) at {fps:.0f} fps")
        if noise:
            combined = noise.combined
            from utils.noise_map import NOTICEABLE
            share = float((combined >= NOTICEABLE).mean()) * 100 if len(combined) else 0.0
            logger.info(
                f"Noise: p50={float(np.percentile(combined, 50)):.3f} "
                f"p90={float(np.percentile(combined, 90)):.3f} "
                f"max={float(combined.max()):.3f}, "
                f"{share:.1f}% of frames over {NOTICEABLE}. Nothing is removed "
                "for this -- it marks segments so they can be left out.")
    return found, noise
