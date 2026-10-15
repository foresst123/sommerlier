"""Client with PhoWhisperASR's interface, backed by worker process(es).

`endpoint` is a WorkerProcessService or a WorkerPoolService. With a pool, several
threads may call `transcribe_batch` at once and each request goes to whichever worker
is idle, which is how one lane uses more than one worker.
"""

import os
import tempfile
import uuid

import numpy as np


class PhoWhisperClient:
    def __init__(self, endpoint, batch_size: int = 16):
        self.endpoint = endpoint
        self.batch_size = max(1, int(batch_size))

    def transcribe(self, audio_16k_array) -> str:
        return self.transcribe_batch([audio_16k_array])[0]

    def transcribe_batch(self, audio_16k_arrays: list, batch_size: int = None,
                         logger=None, callback=None) -> list:
        if not audio_16k_arrays:
            return []
        limit = max(1, int(batch_size or self.batch_size))
        texts = []
        for start in range(0, len(audio_16k_arrays), limit):
            chunk = [np.ascontiguousarray(a, dtype=np.float32)
                     for a in audio_16k_arrays[start:start + limit]]
            texts.extend(self._send(chunk, limit))
            if callback:
                for _ in chunk:
                    callback()
        return texts

    def _send(self, clips: list, batch_size: int) -> list:
        handle = tempfile.NamedTemporaryFile(
            prefix="sommelier_pho_", suffix=".npy", delete=False)
        handle.close()
        try:
            np.save(handle.name, np.concatenate(clips))
            request_id = uuid.uuid4().hex
            response = self.endpoint.request({
                "cmd": "transcribe_batch",
                "id": request_id,
                "audio_path": handle.name,
                "lengths": [len(clip) for clip in clips],
                "ids": [str(i) for i in range(len(clips))],
                "batch_size": batch_size,
            }, response_id=request_id)
            if response.get("error"):
                raise RuntimeError(response["error"])
            by_id = {str(item.get("id")): str(item.get("text", "")).strip()
                     for item in response.get("results", [])}
            return [by_id.get(str(i), "") for i in range(len(clips))]
        finally:
            try:
                os.remove(handle.name)
            except OSError:
                pass

    def close(self):
        stop = getattr(self.endpoint, "stop", None)
        if callable(stop):
            stop()

    unload = close
