import numpy as np
from dataclasses import dataclass

@dataclass
class SeparationResult:
    source_1: np.ndarray          # Waveform speaker 1
    source_2: np.ndarray          # Waveform speaker 2
    speaker_1_id: str             # Speaker label gán cho source 1
    speaker_2_id: str             # Speaker label gán cho source 2
    confidence: float             # Cosine similarity score
