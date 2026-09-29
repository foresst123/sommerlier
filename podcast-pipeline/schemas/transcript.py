from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class TranscriptSegment:
    index: str
    start: float
    end: float
    speaker: str
    text: str                     # Ensemble text
    text_whisper: str
    text_phowhisper: str
    text_qwen3: str
    language: str
    bs_roformer: bool             # Đã qua BS-RoFormer?
    bss: bool                     # Đã qua TSE?
    # PANNs' music verdict, kept whether or not BS-RoFormer then ran. `bs_roformer` only
    # says the audio was replaced, which is a different question: a segment can
    # be flagged for music and still be left alone, and a TSE segment skips
    # music detection altogether. Reviewers need to see the detection.
    has_music: bool = False
    # Overlapping spans TSE could not pull apart, as (start, end, reason). The
    # audio for these still carries both voices, so a reviewer transcribing
    # them is hearing a mixture rather than one speaker.
    unseparated: Optional[List[Dict[str, Any]]] = None
    qwen3omni_caption: Optional[str] = None # Caption từ Qwen3-Omni (optional)
    words: Optional[List[Dict[str, Any]]] = None # Word-level timestamps (optional)

    # --- provenance: see utils/provenance.py ---------------------------------
    # `start` and `end` above are in the CUT timeline -- the recording with its
    # sung and standalone-music stretches removed. These three say what that
    # corresponds to in the file as delivered, and they are written at export
    # by utils.provenance.annotate.
    #
    # Where this segment's audio sits in the original recording. A list because
    # a segment straddling a join is two pieces glued together; one (start,
    # end) pair would name a range covering the audio removed between them.
    orig_spans: Optional[List[Dict[str, float]]] = None
    # Whether that happened. Still good speech, but not a contiguous
    # observation of the source.
    crosses_cut: bool = False
    # Silence before this segment in seconds, or None when a join lies in it.
    # None means unknowable, NOT zero: a pause measured across a join is
    # however much audio was removed there. Anything learning turn timing must
    # skip these rather than read them as fast responses. Negative values are
    # real -- that is an interruption, which is the point of the corpus.
    gap_before: Optional[float] = None
    # Non-speech, non-music noise over the span, 0..1, or None when the
    # detector did not run. None is "not checked", not "clean".
    noise_score: Optional[float] = None
    # The same measure taken on each AudioSet group separately, so a filter can
    # treat voices differently from a fan: {"noise_speech", "noise_env",
    # "noise_room"} -> 0..1. Background voices break diarization and put words
    # in the transcript that nobody said, which is why the groups are kept
    # apart. None when the detector did not run, like noise_score.
    noise_breakdown: Optional[Dict[str, Optional[float]]] = None
    # Which group is loudest, as a word: "noise_speech", "noise_env" or
    # "noise_room" when it reaches utils.noise_map.NOTICEABLE, "clean" when it
    # was measured and nothing does, None when it was not measured. "clean" only
    # means below that provisional level; the numbers above are the evidence.
    noise_kind: Optional[str] = None

    # The diarizer's label, kept only when the speaker relabel pass changed
    # `speaker`. None means the label is untouched. Nothing but `speaker` is ever
    # rewritten by that pass, so this is the one trace that it ran on a segment.
    speaker_original: Optional[str] = None
