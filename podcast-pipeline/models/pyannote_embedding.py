import torch
from pyannote.audio import Model, Inference

class PyannoteEmbedder:
    """Wrapper for Pyannote speaker embedding (for clustering/fusion)."""
    
    def __init__(self, token: str, device: torch.device):
        self.model = Model.from_pretrained("pyannote/embedding", token=token)
        self.inference = Inference(self.model, device=device, window="whole")
        
    def embed(self, input_data):
        """
        Extract embeddings. 
        input_data must be a dict: {"waveform": torch.Tensor, "sample_rate": int}
        """
        return self.inference(input_data)
