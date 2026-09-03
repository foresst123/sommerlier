"""Growing each speaker's enrollment from the extractions that went well.

Enrollment is mined once, before any separation, from whatever solo audio the
diarizer found. That audio is clean but it is not the audio being separated:
different sentence, different loudness, sometimes minutes away. Similarity
against it sat flat at 0.58 across this corpus -- flat with span length, flat
with segment length, so not noise but a ceiling -- where natural speech against
a natural enrollment scores 0.70-0.90.

A separated span that scores well is, by construction, a recording of the target
speaker from inside the same conversation. Adding those back into the enrollment
gives later separations a reference drawn from the audio they are actually
working on. The idea is EvoTSE's (arXiv 2604.06810), which reports SI-SDRi
rising from 2.09 to 10.73 dB out of domain by evolving the enrollment at
inference time -- no retraining, the same extractor throughout.

Two things make it safe here rather than a way to compound errors:

  * only spans above a similarity floor are admitted, so a mis-assigned track
    cannot teach the speaker its own mistake;
  * the mined enrollment is never displaced, only extended, so the worst case
    is the original behaviour.

The premise was checked before this was written: on the measured file 21% of
spans clear the floor, and the first one arrives at position 3 of 24 for one
speaker and 6 of 24 for the other -- early enough that most of the file is
separated with an enrollment that has already grown.
"""

import os

import numpy as np

# Similarity a separated track must reach before it is trusted as reference
# audio. Above the QC threshold that decides whether a span is usable at all
# (0.20): usable is a lower bar than exemplary, and only the second belongs in
# an enrollment. On this corpus 0.65 admits the top fifth of spans.
ADMIT_SIMILARITY = float(os.environ.get("TSE_MEMORY_SIM", "0.65"))

# Shortest admissible clip. Below this there is too little voiced audio for the
# embedder to gain anything, and short clips are where assignment is least sure.
MIN_CLIP_SECONDS = float(os.environ.get("TSE_MEMORY_MIN_CLIP", "0.4"))

# Total added audio per speaker. Enrollment gains saturate around 4s and the
# mined budget is already 8s, so this is deliberately modest: the value here is
# reference audio from *this* conversation, not more seconds.
BUDGET_SECONDS = float(os.environ.get("TSE_MEMORY_BUDGET", "4.0"))

ENABLED = os.environ.get("TSE_MEMORY", "0").strip().lower() in ("1", "true", "yes", "on")


class EnrollmentMemory:
    """Per-speaker store of well-separated audio, used to extend enrollments."""

    def __init__(self, admit=ADMIT_SIMILARITY, min_clip=MIN_CLIP_SECONDS,
                 budget=BUDGET_SECONDS, logger=None, enabled=None):
        # Off unless asked for: this changes what the separator is conditioned
        # on, so it has to be a deliberate choice rather than a default that
        # arrives with an upgrade.
        self.enabled = ENABLED if enabled is None else enabled
        self.admit = admit
        self.min_clip = min_clip
        self.budget = budget
        self.logger = logger
        # speaker -> [(similarity, clip)], best first
        self._clips = {}
        self.added = 0
        self.rejected = 0

    def reset(self):
        """Clear everything. Speaker labels are per-file and must not carry."""
        self._clips.clear()
        self.added = 0
        self.rejected = 0

    def offer(self, speaker, clip, similarity, sample_rate):
        """Consider one separated track for the speaker's memory.

        Returns True when it was kept. A clip is judged on its similarity to the
        enrollment that produced it, which is the same number the QC gate uses,
        so nothing new has to be computed.
        """
        if not self.enabled:
            return False
        if speaker is None or clip is None or similarity is None:
            return False
        if similarity < self.admit:
            self.rejected += 1
            return False

        clip = np.asarray(clip).reshape(-1)
        if sample_rate <= 0 or len(clip) / sample_rate < self.min_clip:
            self.rejected += 1
            return False
        # A track that came back silent scores against silence, not speech.
        if float(np.sqrt(np.mean(clip ** 2))) < 1e-4:
            self.rejected += 1
            return False

        held = self._clips.setdefault(speaker, [])
        held.append((float(similarity), clip.astype(np.float32, copy=True)))
        # Best first, so trimming to budget keeps the strongest references.
        held.sort(key=lambda item: item[0], reverse=True)

        total, kept = 0.0, []
        for score, audio in held:
            duration = len(audio) / sample_rate
            if total + duration > self.budget and kept:
                break
            kept.append((score, audio))
            total += duration
        self._clips[speaker] = kept

        self.added += 1
        return True

    def extend(self, speaker, enrollment, sample_rate):
        """The mined enrollment plus whatever this speaker has earned.

        The mined clips come first and are never dropped: the memory is
        additional evidence, not a replacement for it.
        """
        if not self.enabled:
            return enrollment
        held = self._clips.get(speaker)
        if not held:
            return enrollment
        return list(enrollment) + [audio for _, audio in held]

    def summary(self):
        return {
            "speakers": len(self._clips),
            "clips": sum(len(v) for v in self._clips.values()),
            "admitted": self.added,
            "rejected": self.rejected,
            "threshold": self.admit,
        }
