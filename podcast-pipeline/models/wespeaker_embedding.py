"""WeSpeaker ONNX speaker embedding without the incompatible Python package.

The official ``wespeaker`` package pins NumPy 1.22, while this pipeline uses
NumPy 2.x. The released ONNX checkpoint only needs the existing Torch audio
frontend and ONNX Runtime, so inference stays deliberately small here.
"""

import os
import threading
from collections import Counter, defaultdict
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
# ONNX Runtime's CUDA provider defaults to "EXHAUSTIVE": every input length it has
# not seen yet triggers a search over convolution algorithms. On this ResNet that
# cost ~2.5 s per new length (3.3 s for the first), against ~80 ms with "DEFAULT"
# (measured on an A100), and probes almost never repeat a length.
DEFAULT_CUDNN_CONV_ALGO_SEARCH = "DEFAULT"


class WeSpeakerONNXEmbedder:
    """Extract ResNet293-LM speaker embeddings from mono waveform arrays."""

    def __init__(self, device, repository=DEFAULT_REPOSITORY,
                 filename=DEFAULT_FILENAME, revision=None, threads=None,
                 cudnn_conv_algo_search=DEFAULT_CUDNN_CONV_ALGO_SEARCH):
        self.device = torch.device(device)
        self.cudnn_conv_algo_search = cudnn_conv_algo_search
        # ONNX Runtime intra-op threads; None keeps the process's CPU budget.
        self.threads = threads
        self.repository = repository
        self.filename = filename
        self.revision = revision
        self._session = None
        self._session_lock = threading.Lock()
        self.batch_stats = Counter()

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
            with self._session_lock:
                if self._session is None:
                    from utils.cpu_plan import onnx_session_options
                    providers = ["CPUExecutionProvider"]
                    if self.device.type == "cuda":
                        providers.insert(0, (
                            "CUDAExecutionProvider",
                            {"device_id": int(self.device.index or 0),
                             "cudnn_conv_algo_search": self.cudnn_conv_algo_search},
                        ))
                    # sess_options caps ORT's intra-op thread pool; see cpu_plan.
                    self._session = ort.InferenceSession(
                        self._model_path(),
                        sess_options=onnx_session_options(self.threads),
                        providers=providers,
                    )
        return self._session

    def active_providers(self):
        """ONNX Runtime providers the session really got; CUDA is dropped silently
        when its libraries cannot be loaded, leaving only the CPU provider."""
        return list(self._get_session().get_providers())

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

    def embed_batch(self, audios, sample_rate=SAMPLE_RATE, max_batch_size=4):
        """Embed equal-frame inputs together, without padding or cropping.

        Different lengths and fixed-batch-one exports run individually. Each
        result is a tensor or its exception, so a bad clip cannot discard its
        neighbours. Batch inference failures retry the same features singly.
        """
        results = [None] * len(audios)
        groups = defaultdict(list)
        for index, audio in enumerate(audios):
            try:
                features = self._features(audio, sample_rate).cpu().numpy()
                groups[features.shape].append((index, features))
            except Exception as exc:
                results[index] = exc
        if not groups:
            return results
        session = self._get_session()
        shape = next(item.shape for item in session.get_inputs() if item.name == "feats")
        dynamic_batch = bool(shape) and not isinstance(shape[0], int)
        limit = max(1, int(max_batch_size)) if dynamic_batch else 1

        def infer(items):
            data = np.stack([features for _, features in items])
            output = np.asarray(session.run(["embs"], {"feats": data})[0],
                                dtype=np.float32)
            if output.ndim != 2 or output.shape[0] != len(items) or not output.shape[1]:
                raise ValueError("WeSpeaker returned an invalid embedding batch shape")
            for (index, _), embedding in zip(items, output):
                if not np.isfinite(embedding).all():
                    raise ValueError("WeSpeaker returned an invalid embedding")
                results[index] = torch.from_numpy(embedding.copy()).to(self.device)

        for group in groups.values():
            for start in range(0, len(group), limit):
                items = group[start:start + limit]
                if len(items) > 1:
                    self.batch_stats["batch_attempts"] += 1
                    try:
                        infer(items)
                    except Exception:
                        self.batch_stats["batch_fallbacks"] += 1
                    else:
                        self.batch_stats["batches"] += 1
                        self.batch_stats["batched_items"] += len(items)
                        continue
                for item in items:
                    self.batch_stats["single_calls"] += 1
                    try:
                        infer([item])
                    except Exception as exc:
                        results[item[0]] = exc
        return results
