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
    demucs: bool                  # Đã qua Demucs?
    tse: bool                     # Đã qua TSE?
    # PANNs' music verdict, kept whether or not Demucs then ran. `demucs` only
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
