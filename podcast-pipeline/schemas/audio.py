import numpy as np
from dataclasses import dataclass
from pydub import AudioSegment

@dataclass
class AudioData:
    waveform: np.ndarray          # float32, 1D, normalized [-1, 1]
    sample_rate: int              # 16000: tốc độ lấy mẫu của toàn pipeline (services/audio_service.PIPELINE_SAMPLE_RATE)
    name: str                     # Tên file gốc
    audio_segment: AudioSegment   # Pydub object (cho export)
    duration: float               # Tổng thời lượng (giây)

@dataclass
class DiarizationChunk:
    path: str                     # Đường dẫn file chunk tạm
    offset: float                 # Vị trí bắt đầu trong audio gốc (giây)
    duration: float               # Độ dài chunk (giây)
