import torch
from pyannote.audio import Pipeline

class PyannoteDiarizer:
    """Wrapper for Pyannote speaker diarization."""
    
    def __init__(self, token: str, device: torch.device, use_community: bool = True, kwargs: dict = None):
        model_name = "pyannote/speaker-diarization-community-1" if use_community else "pyannote/speaker-diarization"
        self.pipeline = Pipeline.from_pretrained(model_name, token=token)
        self.pipeline.to(device)
        
        if not use_community and kwargs:
            self.pipeline.instantiate(kwargs)
            
    def diarize(self, audio_path: str):
        """Run diarization on audio file."""
        return self.pipeline(audio_path)
