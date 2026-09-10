from dataclasses import dataclass
from typing import List
from schemas.segment import Segment

@dataclass
class DiarizationResult:
    segments: List[Segment]       # Danh sách segment đã diarize
    num_speakers: int             # Số speaker phát hiện được
    method: str                   # "pyannote" hoặc "sortformer"

@dataclass
class OverlapPair:
    seg1: Segment
    seg2: Segment
    overlap_start: float
    overlap_end: float
    overlap_duration: float
