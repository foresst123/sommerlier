import numpy as np
from pydub import AudioSegment
from schemas.audio import AudioData

class AudioService:
    """Service to handle audio loading, normalization and basic manipulations."""
    
    def __init__(self, logger=None):
        self.logger = logger
        
    def load_audio(self, file_path: str, target_sr: int = 16000) -> AudioData:
        """Load audio file into memory, normalize and return AudioData schema."""
        if self.logger:
            self.logger.info(f"Loading audio from {file_path} at {target_sr}Hz")
            
        # Use pydub for standardization (matches main branch exactly)
        pydub_audio = AudioSegment.from_file(file_path)
        pydub_audio = pydub_audio.set_frame_rate(target_sr).set_sample_width(2).set_channels(1)
        
        # Volume normalization to -20 dBFS
        target_dBFS = -20
        gain = target_dBFS - pydub_audio.dBFS
        if self.logger:
            self.logger.info(f"Applying gain: {gain:.2f} dB")
        pydub_audio = pydub_audio.apply_gain(min(max(gain, -3), 3))
        
        # Extract waveform
        waveform = np.array(pydub_audio.get_array_of_samples(), dtype=np.float32)
        max_amplitude = np.max(np.abs(waveform))
        if max_amplitude > 0:
            waveform /= max_amplitude
            
        duration = len(waveform) / target_sr
        
        return AudioData(
            waveform=waveform,
            sample_rate=target_sr,
            name=file_path.split('/')[-1],
            audio_segment=pydub_audio,
            duration=duration
        )
