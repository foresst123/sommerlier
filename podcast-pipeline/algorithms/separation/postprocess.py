import numpy as np
from utils.audio_math import rms_energy

def match_target_amplitude(source_wav: np.ndarray, target_wav: np.ndarray) -> np.ndarray:
    """
    Match the volume (RMS) of source_wav to that of target_wav.
    """
    epsilon = 1e-10
    src_rms = rms_energy(source_wav)
    tgt_rms = rms_energy(target_wav)
    
    if src_rms < epsilon:
        return source_wav
    
    gain = tgt_rms / (src_rms + epsilon)
    adjusted_wav = source_wav * gain
    return np.clip(adjusted_wav, -1.0, 1.0)
