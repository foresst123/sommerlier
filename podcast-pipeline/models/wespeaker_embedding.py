"""WeSpeaker ONNX speaker embedding without the incompatible Python package.

The official ``wespeaker`` package pins NumPy 1.22, while this pipeline uses
NumPy 2.x. The released ONNX checkpoint only needs the existing Torch audio
frontend and ONNX Runtime, so inference stays deliberately small here.
"""

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from huggingface_hub import hf_hub_download


SAMPLE_RATE = 16000
DEFAULT_REPOSITORY = "Wespeaker/wespeaker-voxceleb-resnet293-LM"
DEFAULT_FILENAME = "voxceleb_resnet293_LM.onnx"


class WeSpeakerONNXEmbedder:
    """Extract ResNet293-LM speaker embeddings from mono waveform arrays."""

    def __init__(self, device, repository=DEFAULT_REPOSITORY,
                 filename=DEFAULT_FILENAME, revision=None):
        self.device = torch.device(device)
        self.repository = repository
        self.filename = filename
        self.revision = revision
        self._session = None

    def _model_path(self):
        override = os.environ.get("WESPEAKER_MODEL_PATH")
        if override:
            path = Path(override).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"WESPEAKER_MODEL_PATH does not exist: {path}")
            return str(path)

        return hf_hub_download(
            repo_id=self.repository,
            filename=self.filename,
            revision=self.revision,
            cache_dir=os.environ.get("WESPEAKER_CACHE_DIR"),
        )

    def _get_session(self):
        if self._session is None:
            providers = ["CPUExecutionProvider"]
            if self.device.type == "cuda":
                providers.insert(0, "CUDAExecutionProvider")
            self._session = ort.InferenceSession(self._model_path(), providers=providers)
        return self._session

    @staticmethod
    def _features(audio, sample_rate):
        waveform = torch.as_tensor(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        if waveform.shape[-1] < int(0.025 * SAMPLE_RATE):
            raise ValueError("WeSpeaker needs at least 25ms of audio")

        # WeSpeaker's official ONNX frontend: 80-bin Kaldi fbank, utterance
        # mean normalization, and PCM-scale waveform input.
        fbank = kaldi.fbank(
            waveform * (1 << 15),
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=SAMPLE_RATE,
            window_type="hamming",
            use_energy=False,
        )
        return fbank - fbank.mean(dim=0, keepdim=True)

    def embed(self, audio, sample_rate=SAMPLE_RATE):
        features = self._features(audio, sample_rate).unsqueeze(0).cpu().numpy()
        output = self._get_session().run(["embs"], {"feats": features})[0]
        embedding = np.asarray(output, dtype=np.float32).reshape(-1)
        if not embedding.size or not np.isfinite(embedding).all():
            raise ValueError("WeSpeaker returned an invalid embedding")
        return torch.from_numpy(embedding).to(self.device)
