from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from schemas.segment import EnhancedSegment

@dataclass
class ASRResult:
    text: str                     # Text cuối cùng (ensemble)
    text_whisper: str             # Output Whisper
    text_phowhisper: str          # Output PhoWhisper
    text_qwen3: str               # Output Qwen3
    language: str                 # Ngôn ngữ detected
    confidence: Optional[float]   # Confidence score (optional)

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
    qwen3omni_caption: Optional[str] = None # Caption từ Qwen3-Omni (optional)
    words: Optional[List[Dict[str, Any]]] = None # Word-level timestamps (optional)
