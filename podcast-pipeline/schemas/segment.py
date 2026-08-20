import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Segment:
    index: str                    # "00001", "00002"...
    start: float                  # Thời gian bắt đầu (giây)
    end: float                    # Thời gian kết thúc (giây)
    speaker: str                  # "SPEAKER_00", "SPEAKER_01"...

@dataclass
class EnhancedSegment(Segment):
    enhanced_audio: Optional[np.ndarray] = None  # Audio đã qua TSE
    tse: bool = False                            # Có được xử lý bởi TSE không
    demucs: bool = False                         # Có được xử lý bởi Demucs không
    # Absolute (start, end, sim) of every span actually replaced by TSE output.
    # `tse` alone cannot say which part of a segment is separated and which is
    # still raw mixture; the dual-channel export needs that distinction to avoid
    # writing the interfering speaker into the target track.
    tse_spans: List[Tuple[float, float, float]] = field(default_factory=list)
    # (start, end, reason, detail) for overlapping spans that could NOT be
    # separated. Every overlap must land in exactly one of these two lists --
    # a span in neither means some code path discarded it silently.
    tse_failed_spans: List[Tuple[float, float, str, str]] = field(default_factory=list)

    @property
    def tse_status(self) -> str:
        if not self.tse_spans and not self.tse_failed_spans:
            return "clean"        # no overlap touched this segment
        if not self.tse_failed_spans:
            return "separated"
        if not self.tse_spans:
            return "failed"
        return "partial"
