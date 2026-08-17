import librosa
from pydub import AudioSegment
from schemas.audio import AudioData

class AudioService:
    """Service to handle audio loading, normalization and basic manipulations."""
    
    def __init__(self, logger=None):
        self.logger = logger
        
    def load_audio(self, file_path: str, target_sr: int = 24000) -> AudioData:
        """Load audio file into memory, normalize and return AudioData schema."""
        if self.logger:
            self.logger.info(f"Loading audio from {file_path} at {target_sr}Hz")
            
        # librosa handles normalization by default
        waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
        duration = len(waveform) / sr
        
        # Load pydub for MP3 export later
        pydub_audio = AudioSegment.from_file(file_path)
        
        return AudioData(
            waveform=waveform,
            sample_rate=sr,
            name=file_path.split('/')[-1],
            audio_segment=pydub_audio,
            duration=duration
        )
