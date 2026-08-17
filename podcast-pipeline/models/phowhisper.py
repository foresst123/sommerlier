import torch
from transformers import pipeline

class PhoWhisperASR:
    """Wrapper for PhoWhisper-large via transformers pipeline."""
    
    def __init__(self, device: torch.device, dtype=None):
        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model="vinai/PhoWhisper-large",
            device=device,
            torch_dtype=dtype,
        )
    
    def transcribe(self, audio_16k_array) -> str:
        """Run inference and return Vietnamese text."""
        result = self.pipeline(
            {"array": audio_16k_array, "sampling_rate": 16000},
            generate_kwargs={"language": "vi"}
        )
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("text", "")
        elif isinstance(result, dict):
            return result.get("text", "")
        return str(result)
