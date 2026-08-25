import os
import torch
from models.whisper import load_asr_model

class PhoWhisperASR:
    """Wrapper for PhoWhisper-large using faster-whisper (CTranslate2)."""
    
    def __init__(self, device: torch.device, dtype=None,
                 compute_type: str = None, batch_size: int = 16, threads: int = 4):
        self.batch_size = int(batch_size)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        device_index = 0
        if isinstance(self.device, torch.device) and self.device.index is not None:
            device_index = self.device.index
            
        # An explicit compute_type from config wins; the env var stays as the
        # override that needs no config edit. bfloat16 only pays off from
        # Ampere onwards -- Turing has no bf16 tensor cores and emulates it.
        if compute_type is None:
            use_bf16 = os.environ.get("SOMMELIER_USE_BF16") == "1"
            compute_type = "bfloat16" if use_bf16 else "float16"
        if self.device.type != "cuda":
            compute_type = "int8"
            
        self.model = load_asr_model(
            whisper_arch="kiendt/PhoWhisper-large-ct2",
            device="cuda" if self.device.type == "cuda" else "cpu",
            device_index=device_index,
            compute_type=compute_type,
            language="vi",
            vad_model=None,
            vad_options=None,
            # Same reasoning as the Whisper wrapper: priming the decoder makes
            # it complete the prompt rather than transcribe on short clips, and
            # temperature fallback re-rolls near-silence into invented text.
            asr_options={
                "initial_prompt": None,
                "temperatures": [0.0],
                "hallucination_silence_threshold": 1.0,
            },
            threads=threads
        )
    
    def transcribe(self, audio_16k_array) -> str:
        """Run inference and return Vietnamese text."""
        dummy_vad = [{"start": 0.0, "end": len(audio_16k_array) / 16000.0}]
        result = self.model.transcribe(
            audio_16k_array,
            dummy_vad,
            batch_size=1,
            language="vi",
            print_progress=False
        )
        if result and "segments" in result:
            return " ".join([s["text"] for s in result["segments"]]).strip()
        return ""

    def transcribe_batch(self, audio_16k_arrays: list, batch_size: int = None, logger=None, callback=None) -> list:
        """Run batched inference and return list of Vietnamese texts.

        Falls back to the batch size this model was configured with. The caller
        used to pass the Qwen3 worker's batch size, which is sized for a
        different model on a different device.
        """
        if not audio_16k_arrays:
            return []

        batch_size = int(batch_size or self.batch_size)
            
        texts = []
        for arr in audio_16k_arrays:
            texts.append(self.transcribe(arr))
            if callback: callback()
                
        return texts
