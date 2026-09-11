"""What kind of audio is playing across a recording, frame by frame.

Two questions get different answers and need to be told apart:

    music with nobody talking -> cut it out. An intro, an outro, a sting: there
                                 is no dialogue in it to keep, and handing it to
                                 a vocal separator would only ask the model to
                                 invent a voice out of an instrumental.
    speech over a music bed   -> separate, then transcribe.
    speech, nothing under it  -> leave it alone. On this corpus that is
                                 almost everything.

There used to be a third kind, SINGING, for a voice that is singing rather than
speaking. It is gone. On three recordings it fired for 10s, 0s and 0s under
PANNs, and under SSLAM every detection that survived the speech-margin gate was
a single isolated frame -- no run ever reached the two consecutive frames that
MIN_SPAN_EXCISED asks for, so the branch could not produce a span at all. A
branch that cannot fire is worse than no branch: it reads as a safeguard that
is present and working. What it was meant to catch, someone singing over
speech, now falls to MUSIC and is cleaned rather than cut.

The detector supplies all of this at once: both taggers score all 527 AudioSet
labels on every frame, `Speech`, `Singing` and `Music` among them, so reading
several groups costs no more than reading one.

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
# SONG deletes audio from the recording permanently. That asks for more than one
# block of evidence, and the cost of asking is small: across the two files with
# standalone music, 0.96 drops 6 fragment spans and 2.9 seconds total.
#
# Both numbers were calibrated on PANNs' 320ms grid. On SSLAM they quantise
# rather than transfer: at its 2 fps a frame IS 0.5s, so 0.32 becomes "one
# frame" and 0.96 becomes "two consecutive frames". Measured on the same three
# recordings, that costs 6.5s out of 417.5s of music -- 1.6%, spread over 13
# single-frame runs. The names no longer describe the behaviour, but the
# behaviour is still the intended one.
MIN_SPAN_SECONDS = float(os.environ.get("MUSIC_MAP_MIN_SPAN", "0.32"))
MIN_SPAN_EXCISED = float(os.environ.get("MUSIC_MAP_MIN_SPAN_CUT", "0.96"))

# Gap below which two runs of the same kind are one. Music dips under the
# threshold on a beat rest without stopping.
MERGE_GAP_SECONDS = float(os.environ.get("MUSIC_MAP_MERGE_GAP", "0.50"))

# Margin to pad around detected music spans to catch the fade-in/out
PAD_SECONDS = float(os.environ.get("MUSIC_MAP_PAD", "0.30"))

# What a span can be, and what each one means for the audio.
#
#   MUSIC    a bed with someone talking over it. Strip the bed, keep the speech.
#   SONG     music with nobody speaking -- an intro, an outro, a sting. Cut it
#            too. Running vocal separation here would extract whatever the
#            model imagines a voice to be out of an instrumental.
MUSIC = "music"
SONG = "song"
SINGING = "singing"
EXCISE_SINGING = os.environ.get("MUSIC_MAP_EXCISE_SINGING", "0") == "1"
SINGING_THRESHOLD = float(os.environ.get("MUSIC_MAP_SINGING_THRESHOLD", "0.10"))

# Spans that leave the recording entirely rather than being cleaned.
EXCISED = (SONG, SINGING) if EXCISE_SINGING else (SONG,)

# Below this the frame carries no speech worth keeping, so music there is a
# song rather than a bed.
SPEECH_PRESENT = float(os.environ.get("MUSIC_MAP_SPEECH_PRESENT", "0.20"))


class MusicMap:
    """Musical and sung stretches of one recording, queryable by time span."""

    def __init__(self, spans=None, fps=100.0):
        # Sorted (start, end, kind) in seconds. A bare (start, end) is taken as
        # music, which is what every span meant before the kinds were split.
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
                "song_seconds": round(self.total_of(SONG), 2),
                "singing_seconds": round(self.total_of(SINGING), 2)}


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
          music_threshold=MUSIC_THRESHOLD):
    """The music map alone. See `build_maps` for the noise track beside it."""
    return build_maps(waveform, sample_rate, detector, logger=logger,
                      music_threshold=music_threshold)[0]


def build_maps(waveform, sample_rate, detector, logger=None,
               music_threshold=MUSIC_THRESHOLD):
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

    speech, music = scores["speech"], scores["music"]
    if not len(music):
        return MusicMap(fps=fps), noise

    # Split the music by whether anyone is talking over it. A bed under speech
    # is worth stripping; music with no speech is worth cutting, and asking a
    # vocal separator to work on it would only invent a voice.
    loud_music = (music >= music_threshold)
    is_song = loud_music & (speech < SPEECH_PRESENT)
    is_music = loud_music & ~is_song
    is_singing = (np.asarray(scores.get("singing", np.zeros_like(music)))
                  >= SINGING_THRESHOLD) & (speech < SPEECH_PRESENT)
    if EXCISE_SINGING:
        is_song &= ~is_singing

    # SONG leaves the recording; MUSIC is only cleaned. Deleting audio asks for
    # more evidence than cleaning it, so SONG clears a longer run -- see
    # MIN_SPAN_*.
    spans = ([(a, b, SONG) for a, b in
                _runs(is_song, fps, MIN_SPAN_EXCISED, MERGE_GAP_SECONDS, PAD_SECONDS)]
             + [(a, b, MUSIC) for a, b in
                _runs(is_music, fps, MIN_SPAN_SECONDS, MERGE_GAP_SECONDS, PAD_SECONDS)])

    found = MusicMap(spans, fps=fps)
    if EXCISE_SINGING:
        spans.extend((a, b, SINGING) for a, b in _runs(
            is_singing, fps, MIN_SPAN_EXCISED, MERGE_GAP_SECONDS, PAD_SECONDS))
        found = MusicMap(spans, fps=fps)
    if logger:
        duration = len(waveform) / max(sample_rate, 1)
        logger.info(
            f"Music map: {found.total_of(SONG):.1f}s standalone music, "
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
