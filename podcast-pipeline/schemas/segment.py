import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Segment:
    index: str                    # "00001", "00002"...
    start: float                  # Thời gian bắt đầu (giây)
    end: float                    # Thời gian kết thúc (giây)
    speaker: str                  # "SPEAKER_00", "SPEAKER_01"...

@dataclass
class EnhancedSegment(Segment):
    enhanced_audio: Optional[np.ndarray] = None  # Audio đã qua SR-CorrNet
    srcorrnet: bool = False                      # Có được tách bởi SR-CorrNet không
    demucs: bool = False                         # Có được xử lý bởi Demucs không
