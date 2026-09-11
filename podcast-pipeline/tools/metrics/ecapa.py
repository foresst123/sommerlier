"""Independent ECAPA inference; does not construct Sidon, BSS or VAD."""

import os

import numpy as np


class ECAPAScorer:
    def __init__(self, device="cpu", savedir=None):
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir or os.path.join(os.environ.get("BSS_PATH", "pretrained_models"), "ecapa"),
            run_opts={"device": device})

    def embed(self, audio, sr):
        import librosa
        import torch
        if len(audio) < sr * 0.25 or np.sqrt(np.mean(audio ** 2)) < 1e-6:
            return None
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        with torch.inference_mode():
            value = self.model.encode_batch(torch.from_numpy(audio.copy()).unsqueeze(0).to(self.device))
        value = value.detach().cpu().numpy().reshape(-1)
        return value / max(float(np.linalg.norm(value)), 1e-12)

    @staticmethod
    def similarity(left, right):
        return float(np.dot(left, right)) if left is not None and right is not None else None

    def enrollment(self, audio, sr, lengths):
        if not lengths:
            return self.embed(audio, sr)
        offset, embeddings = 0, []
        for length in lengths:
            value = self.embed(audio[offset:offset + length], sr)
            offset += length
            if value is not None:
                embeddings.append(value)
        if not embeddings:
            return None
        mean = np.mean(embeddings, axis=0)
        return mean / max(float(np.linalg.norm(mean)), 1e-12)
