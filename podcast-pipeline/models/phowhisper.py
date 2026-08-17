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

    def transcribe_batch(self, audio_16k_arrays: list, batch_size: int = 16) -> list:
        """Run batched inference and return list of Vietnamese texts."""
        if not audio_16k_arrays:
            return []
            
        inputs = [{"array": arr, "sampling_rate": 16000} for arr in audio_16k_arrays]
        results = self.pipeline(
            inputs,
            generate_kwargs={"language": "vi"},
            batch_size=batch_size
        )
        
        texts = []
        for res in results:
            if isinstance(res, list) and len(res) > 0:
                texts.append(res[0].get("text", ""))
            elif isinstance(res, dict):
                texts.append(res.get("text", ""))
            else:
                texts.append(str(res))
        return texts
