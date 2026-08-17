import os
import torch
from models.whisper import load_asr_model

class WhisperASR:
    """Wrapper for the Whisper ASR model."""
    def __init__(self, model_size="large-v3-turbo", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        device_index = 0
        if isinstance(self.device, torch.device) and self.device.index is not None:
            device_index = self.device.index
            
        self.model = load_asr_model(
            whisper_arch=model_size,
            device="cuda" if self.device.type == "cuda" else "cpu",
            device_index=device_index,
            compute_type="float16" if self.device.type == "cuda" else "int8",
            language=None, # Will auto-detect or be overridden during transcribe
            vad_model=None,
            vad_options=None,
            threads=4
        )

    def transcribe(self, audio_16k, dummy_vad, language="en", batch_size=1):
        """Transcribe an audio segment."""
        # The underlying model is a VadFreeFasterWhisperPipeline
        result = self.model.transcribe(
            audio_16k,
            dummy_vad,
            batch_size=batch_size,
            language=language,
            print_progress=False
        )
        
        # Result format should mimic what main_original_ASR_MoE.py expects:
        # result is typically a dict with "segments", "language"
        text = ""
        words = []
        det_lang = language
        
        if result and "segments" in result:
            text = " ".join([s["text"] for s in result["segments"]]).strip()
            det_lang = result.get("language", language)
            for s in result["segments"]:
                if "words" in s:
                    words.extend(s["words"])
                    
        return {
            "text": text,
            "language": det_lang,
            "words": words
        }
