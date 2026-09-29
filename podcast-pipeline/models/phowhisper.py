import os
import torch
import numpy as np
from models.whisper import load_asr_model

# Each out-of-memory retries with a batch this much smaller.
_OOM_STEP = 4
# After an out-of-memory the batch size stays lowered until this many chunks fit,
# then goes up one step: the memory another model took may have been freed.
_RECOVER_AFTER_FITS = 10

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
        self._batch_ceiling = None      # None until an out-of-memory lowers it
        self._fits_at_ceiling = 0
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

        requested = int(batch_size or self.batch_size)
        # A size that ran out of memory is not asked for again straight away: the
        # scheduler keeps handing out its own size, and each try that fails first
        # costs more than the batch itself.
        size = requested if self._batch_ceiling is None else min(requested, self._batch_ceiling)

        texts = []
        for start in range(0, len(audio_16k_arrays), size):
            texts.extend(self._transcribe_stepping(
                audio_16k_arrays[start:start + size], logger, callback))
            self._note_fit(requested)
        return texts

    def _note_fit(self, requested: int) -> None:
        """A chunk went through: after enough of them, try one step up again."""
        if self._batch_ceiling is None:
            return
        self._fits_at_ceiling += 1
        if self._fits_at_ceiling >= _RECOVER_AFTER_FITS:
            self._fits_at_ceiling = 0
            raised = self._batch_ceiling + _OOM_STEP
            self._batch_ceiling = None if raised >= requested else raised

    def _transcribe_stepping(self, audio_arrays: list, logger, callback) -> list:
        """One batch; if the GPU runs out of memory, run it again 4 clips smaller.

        The batch goes out as pieces of the smaller size (48 -> 44 + 4), and a piece
        that still does not fit steps down again. The GPU is shared with the Whisper
        and Qwen3 engines, whose memory changes while they load and finish, so a
        batch that fitted a moment ago can fail. A single clip that still does not
        fit is raised: there is nothing smaller to try.
        """
        try:
            return self._transcribe_chunk(audio_arrays, callback)
        except Exception as exc:
            count = len(audio_arrays)
            if count < 2 or not is_out_of_memory(exc):
                raise
            smaller = max(1, count - _OOM_STEP)
            self.oom_retries += 1
            self._batch_ceiling = (smaller if self._batch_ceiling is None
                                   else min(self._batch_ceiling, smaller))
            self._fits_at_ceiling = 0
            if logger:
                logger.warning(
                    f"[PhoWhisper] out of memory on a batch of {count}; retrying with "
                    f"{smaller}; later batches are limited to {self._batch_ceiling} "
                    f"until {_RECOVER_AFTER_FITS} chunks fit")
            texts = []
            for start in range(0, count, smaller):
                texts.extend(self._transcribe_stepping(
                    audio_arrays[start:start + smaller], logger, callback))
            return texts

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
