import torch
from nemo.collections.asr.models import SortformerEncLabelModel

class SortformerDiarizer:
    """Wrapper for NVIDIA NeMo Sortformer diarization."""
    
    def __init__(self, device: torch.device):
        self.model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
        self.model = self.model.to(device)
        self.model.eval()
        
    def diarize(self, audio_paths: list):
        """Run diarization on list of audio paths (usually chunked)."""
        return self.model.diarize(audio_paths)
