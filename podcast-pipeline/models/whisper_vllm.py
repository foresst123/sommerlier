"""Thin IPC client exposing the same interface as ``WhisperASR``."""

import os
import tempfile

import numpy as np


class WhisperVLLMClient:
    """Send independent audio clips to a vLLM Whisper worker.

    Audio stays as float32 NumPy data.  The temporary files are only the IPC
    boundary; this avoids WAV encoding and lets the worker batch all requests
    in one ``LLM.generate`` call.
    """

    def __init__(self, endpoint, batch_size=16):
        self.endpoint = endpoint
        self.batch_size = max(1, int(batch_size))

    @staticmethod
    def _clip(audio, vad):
        array = np.ascontiguousarray(audio, dtype=np.float32)
        if not vad:
            return array
        span = vad[0]
        lo = max(0, min(len(array), round(float(span.get("start", 0.0)) * 16000)))
        hi = max(lo, min(len(array), round(float(
            span.get("end", len(array) / 16000.0)) * 16000)))
        return array[lo:hi]

    def transcribe(self, audio_16k, dummy_vad, language="en", batch_size=None):
        results = self.transcribe_batch(
            [audio_16k], [dummy_vad], language=language,
            batch_size=batch_size)
        return results[0] if results else {
            "text": "", "language": language, "words": []}

    def transcribe_batch(self, audios_16k, vad_segments, language="en",
                         batch_size=None, callback=None):
        if not audios_16k:
            return []
        if self.endpoint is None:
            return [{"text": "", "language": language, "words": []}
                    for _ in audios_16k]

        limit = max(1, int(batch_size or self.batch_size))
        results = []
        for start in range(0, len(audios_16k), limit):
            paths = []
            try:
                jobs = []
                local_vads = vad_segments[start:start + limit]
                for offset, (audio, vad) in enumerate(zip(
                        audios_16k[start:start + limit], local_vads)):
                    handle = tempfile.NamedTemporaryFile(
                        prefix="sommelier_whisper_", suffix=".npy", delete=False)
                    handle.close()
                    path = handle.name
                    np.save(path, self._clip(audio, vad))
                    paths.append(path)
                    jobs.append({"id": str(start + offset), "audio_path": path})

                response = self.endpoint.request({
                    "cmd": "transcribe_batch",
                    "language": language,
                    "jobs": jobs,
                })
                if response.get("error"):
                    raise RuntimeError(response["error"])
                by_id = {str(item.get("id")): item
                         for item in response.get("results", [])}
                for offset in range(len(jobs)):
                    item = by_id.get(str(start + offset), {})
                    results.append({
                        "text": str(item.get("text", "")).strip(),
                        "language": item.get("language") or language,
                        # vLLM Whisper is used for transcript voting here;
                        # word timing is produced by the later aligner stage.
                        "words": [],
                    })
                    if callback:
                        callback()
            finally:
                for path in paths:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        return results

    def close(self):
        stop = getattr(self.endpoint, "stop", None)
        if callable(stop):
            stop()

    unload = close
