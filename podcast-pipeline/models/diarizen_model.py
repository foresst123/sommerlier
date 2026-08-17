import torch
from diarizen.pipelines.inference import DiariZenPipeline

class DiariZenDiarizer:
    """Wrapper for DiariZen WavLM-Large s80-md-v2."""
    
    def __init__(self, device: torch.device):
        self.device = device
        # Load the v2 model which supports up to 4 overlapping speakers
        self.pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
        # Ensure it runs on the correct device
        self.pipeline.to(device)
        
    def diarize(self, audio_path: str):
        """
        Run DiariZen inference on audio file.
        Returns an Annotation object similar to Pyannote.
        """
        return self.pipeline(audio_path)
