import os
import torch
import numpy as np
from models.whisper import load_asr_model

# What CTranslate2 and PyTorch say when the GPU has no memory left.
_OOM_MARKERS = ("out of memory", "outofmemory", "cudaerrormemoryallocation",
                "failed to allocate")


def is_out_of_memory(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _OOM_MARKERS)


class PhoWhisperASR:
    """Wrapper for PhoWhisper-large using faster-whisper (CTranslate2)."""
    
    def __init__(self, device: torch.device, dtype=None,
                 compute_type: str = None, batch_size: int = 16, threads: int = 4):
        self.batch_size = int(batch_size)
        self.oom_retries = 0
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
        for start in range(0, len(audio_16k_arrays), batch_size):
            texts.extend(self._transcribe_halving(
                audio_16k_arrays[start:start + batch_size], logger, callback))
        return texts

    def _transcribe_halving(self, audio_arrays: list, logger, callback) -> list:
        """One batch; if the GPU runs out of memory, run it again as two halves.

        The GPU is shared with the Whisper and Qwen3 engines, whose memory grows
        while they load, so a batch that fitted a moment ago can fail. A single clip
        that still does not fit is raised: there is nothing smaller to try.
        """
        try:
            return self._transcribe_chunk(audio_arrays, callback)
        except Exception as exc:
            if len(audio_arrays) < 2 or not is_out_of_memory(exc):
                raise
            half = len(audio_arrays) // 2
            self.oom_retries += 1
            if logger:
                logger.warning(
                    f"[PhoWhisper] out of memory on a batch of {len(audio_arrays)}; "
                    f"retrying as {half} + {len(audio_arrays) - half}")
            return (self._transcribe_halving(audio_arrays[:half], logger, callback)
                    + self._transcribe_halving(audio_arrays[half:], logger, callback))

    def _transcribe_chunk(self, audio_arrays: list, callback) -> list:
        arrays = [np.ascontiguousarray(arr, dtype=np.float32) for arr in audio_arrays]
        spans = []
        offset = 0
        for array in arrays:
            spans.append({"start": offset / 16000.0,
                          "end": (offset + len(array)) / 16000.0})
            offset += len(array)
        carrier = np.concatenate(arrays) if arrays else np.empty(0, np.float32)
        result = self.model.transcribe(
            carrier, spans, batch_size=len(arrays), language="vi",
            print_progress=False)
        segments = list((result or {}).get("segments", []))
        texts = []
        for index in range(len(arrays)):
            segment = segments[index] if index < len(segments) else {}
            texts.append(str(segment.get("text", "")).strip())
            if callback:
                callback()
        return texts
