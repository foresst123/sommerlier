import torch
from panns_inference import AudioTagging

class PANNSDetector:
    """Wrapper for PANNS Audio Tagging (background music detection)."""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = AudioTagging(checkpoint_path=None, device=self.device)
        
    def detect_music(self, audio_array, sample_rate: int = 32000, threshold: float = 0.5) -> tuple:
        """Detect if music is present in audio."""
        import librosa
        if sample_rate != 32000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=32000)
            
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1) if audio_array.shape[1] == 2 else audio_array
            
        clipwise_output, _ = self.model.inference(audio_array[None, :])
        
        music_idx = None
        for i, label in enumerate(self.model.labels):
            if label.lower() == "music":
                music_idx = i
                break
                
        if music_idx is not None:
            music_prob = float(clipwise_output[0, music_idx])
            has_music = music_prob > threshold
            return has_music, music_prob
        return False, 0.0
