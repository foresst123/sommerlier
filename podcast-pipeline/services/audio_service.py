import numpy as np
from pydub import AudioSegment
from schemas.audio import AudioData

# Widest correction allowed when normalizing loudness. Enough to lift a
# quiet recording to target without amplifying a near-silent one into noise.
MAX_GAIN_DB = 12.0

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
        
        # Loudness normalization to -20 dBFS. The ASR models were trained on
        # normalized speech, so leaving a quiet recording quiet costs accuracy
        # exactly on the soft passages (backchannels) this pipeline cares about.
        # The old +/-3 dB clamp silently under-corrected: a -27 dBFS podcast
        # asked for +7 dB and got +3.
        target_dBFS = -20
        gain = target_dBFS - pydub_audio.dBFS
        applied_gain = min(max(gain, -MAX_GAIN_DB), MAX_GAIN_DB)
        if self.logger:
            if abs(applied_gain - gain) > 0.01:
                self.logger.info(
                    f"Applying gain: {applied_gain:.2f} dB (capped from {gain:.2f} dB)"
                )
            else:
                self.logger.info(f"Applying gain: {applied_gain:.2f} dB")
        pydub_audio = pydub_audio.apply_gain(applied_gain)

        # Extract waveform as float32 in [-1, 1]. Dividing by the peak here
        # would undo the loudness normalization just applied, and would let one
        # loud transient scale down the whole file, so scale by the sample width
        # instead and only fall back to peak scaling if something still clips.
        waveform = np.array(pydub_audio.get_array_of_samples(), dtype=np.float32)
        waveform /= float(1 << (8 * pydub_audio.sample_width - 1))

        max_amplitude = np.max(np.abs(waveform)) if waveform.size else 0.0
        if max_amplitude > 1.0:
            waveform /= max_amplitude
            
        duration = len(waveform) / target_sr
        
        return AudioData(
            waveform=waveform,
            sample_rate=target_sr,
            name=file_path.split('/')[-1],
            audio_segment=pydub_audio,
            duration=duration
        )
