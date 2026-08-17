import torch
from transformers import pipeline

class PhoWhisperASR:
    """Wrapper for PhoWhisper-large via transformers pipeline."""
    
    def __init__(self, device: torch.device, dtype=None):
        import os
        if dtype is None:
            use_bf16 = os.environ.get("SOMMELIER_USE_BF16") == "1"
            if device.type == "cuda":
                dtype = torch.bfloat16 if use_bf16 else torch.float16
            else:
                dtype = torch.float32
            
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

    def transcribe_batch(self, audio_16k_arrays: list, batch_size: int = 16, logger=None, callback=None) -> list:
        """Run batched inference and return list of Vietnamese texts."""
        if not audio_16k_arrays:
            return []
            
        def data_generator():
            for arr in audio_16k_arrays:
                yield {"array": arr, "sampling_rate": 16000}
                
        results = self.pipeline(
            data_generator(),
            generate_kwargs={"language": "vi"},
            batch_size=batch_size
        )
        
        texts = []
        for i, res in enumerate(results):
            if callback: callback()
            if isinstance(res, list) and len(res) > 0:
                texts.append(res[0].get("text", ""))
            elif isinstance(res, dict):
                texts.append(res.get("text", ""))
            else:
                texts.append(str(res))
                
        return texts
