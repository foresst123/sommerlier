import os
import torch
import numpy as np
from models.whisper import load_asr_model

class WhisperASR:
    """Wrapper for the Whisper ASR model."""
    def __init__(self, model_size="large-v3", device=None, compute_type=None, batch_size=16):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.batch_size = batch_size
        device_index = 0
        if isinstance(self.device, torch.device) and self.device.index is not None:
            device_index = self.device.index
            
        if compute_type is None:
            use_bf16 = os.environ.get("SOMMELIER_USE_BF16") == "1"
            compute_type = "bfloat16" if use_bf16 else "float16"
        if self.device.type != "cuda": compute_type = "int8"
            
        self.model = load_asr_model(
            whisper_arch=model_size,
            device="cuda" if self.device.type == "cuda" else "cpu",
            device_index=device_index,
            compute_type=compute_type,
            language=None, # Will auto-detect or be overridden during transcribe
            vad_model=None,
            vad_options=None,
            # Only the settings that differ from whisperx's defaults; the rest
            # (no_speech_threshold, condition_on_previous_text, ...) are already
            # what we want.
            asr_options={
                # No initial_prompt: on the sub-second backchannels this pipeline
                # cares about, priming the decoder makes it complete the prompt
                # instead of transcribing, which is where the "Hẹn gặp lại các
                # bạn..." outros came from.
                "initial_prompt": None,
                # Greedy only. Temperature fallback re-rolls a clip that failed
                # its quality thresholds, and on near-silence each retry is
                # another chance to invent an outro.
                "temperatures": [0.0],
                # Treat a shorter run of silence as suspicious than the 2.0s
                # default, since backchannel gaps are brief.
                "hallucination_silence_threshold": 1.0,
            },
            threads=4
        )

    def transcribe(self, audio_16k, dummy_vad, language="en", batch_size=None):
        """Transcribe an audio segment."""
        # The underlying model is a VadFreeFasterWhisperPipeline
        bs = batch_size if batch_size is not None else self.batch_size
        result = self.model.transcribe(
            audio_16k,
            dummy_vad,
            batch_size=bs,
            language=language,
            print_progress=False
        )
        
        # whisperx returns a dict with "segments" and "language".
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

    def transcribe_batch(self, audios_16k, vad_segments, language="en",
                         batch_size=None, callback=None):
        """Batch independent clips while preserving one result per clip.

        The underlying WhisperX pipeline batches VAD spans. Packing clips into
        one carrier array only supplies those independent spans to that API;
        no decoder context crosses clip boundaries.
        """
        if not audios_16k:
            return []
        limit = max(1, int(batch_size or self.batch_size))
        results = []
        for start in range(0, len(audios_16k), limit):
            arrays = [np.ascontiguousarray(a, dtype=np.float32)
                      for a in audios_16k[start:start + limit]]
            spans = []
            offset = 0
            for array, local in zip(arrays, vad_segments[start:start + limit]):
                duration = len(array) / 16000.0
                item = local[0] if local else {"start": 0.0, "end": duration}
                lo = min(max(0.0, float(item.get("start", 0.0))), duration)
                hi = min(max(lo, float(item.get("end", duration))), duration)
                spans.append({"start": offset / 16000.0 + lo,
                              "end": offset / 16000.0 + hi})
                offset += len(array)
            carrier = np.concatenate(arrays) if arrays else np.empty(0, np.float32)
            output = self.model.transcribe(
                carrier, spans, batch_size=limit, language=language,
                print_progress=False)
            segments = list((output or {}).get("segments", []))
            detected = (output or {}).get("language", language)
            for index in range(len(arrays)):
                segment = segments[index] if index < len(segments) else {}
                words = list(segment.get("words", []))
                results.append({"text": str(segment.get("text", "")).strip(),
                                "language": detected, "words": words})
                if callback:
                    callback()
        return results
