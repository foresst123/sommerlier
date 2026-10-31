"""Speaker embedders for the embedder trials of arm J (tools/windowed_vbx_stage.py --clustering dz-ahc).

EXPERIMENT. The real models were NOT run here (no weights, no GPU in the sandbox). Only the fakes in
tests/test_embedders.py ran. Every adapter below except r34lm is written from the public API or from memory and is
marked UNVERIFIED where that matters.

Kinds (`make_embedder(kind, ...)`):
  r34lm              pyannote's WeSpeaker ResNet34-LM, the model of arm J (DiariZen's PLDA was trained for it). Takes a
                     waveform AND a frame mask: windowed_vbx_stage calls it through make_masked_embedder, exactly as
                     before. Its embed(wave) is an unmasked call, for completeness only.
  wespeaker_r293     the repo's WeSpeaker ResNet-293 ONNX (models/wespeaker_embedding.py, voxceleb_resnet293_LM.onnx).
                     --r293-onnx PATH overrides the Hugging Face download. Runs with onnxruntime.
  wespeaker_simam100 WeSpeaker SimAM-ResNet100, pre-trained on VoxBlink2 and fine-tuned on VoxCeleb2. Preferred:
                     --simam-onnx PATH, run exactly like the R293 adapter (UNVERIFIED: that the export uses the
                     "feats"/"embs" names of the R293 export). Fallback: --simam-dir DIR with config.yaml and
                     avg_model.pt, loaded with the `wespeaker` package (UNVERIFIED API, from memory). The weights are
                     CC BY-NC-SA 4.0 (non-commercial). Fetch them from the WeSpeaker pretrained list or
                     voxblink2.github.io; nothing here downloads them.
  espnet_spk         ESPnet-SPK: espnet2.bin.spk_inference.Speech2Embedding.from_pretrained(model_tag=TAG), default tag
                     espnet/voxcelebs12_ska_wavlm_joint (UNVERIFIED, from memory of the ESPnet API).

Every adapter has `name`, `id_string` (a file-name-safe id for cache keys), `load()` and `embed(wave_16k) -> (D,)`.
Heavy imports happen inside load(); a missing package or missing weights raises RuntimeError with a message that
names the package or the path. embed() loads on its first call, so a run whose embeddings all come from a cache never
imports the model.

make_concat_embedder(adapter, min_s, sr) turns the frame mask of the dz pipeline into audio: the samples of the frames
where mask > 0.5 are concatenated, and a window with less than min_s seconds of them gives None (the caller's None
handling applies). Only r34lm takes the mask itself.
"""

from __future__ import annotations

import math
import os
import re
import sys
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SR = 16000
KINDS = ("r34lm", "wespeaker_r293", "wespeaker_simam100", "espnet_spk")
DEFAULT_ESPNET_TAG = "espnet/voxcelebs12_ska_wavlm_joint"
DEFAULT_MIN_S = 0.5
EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "x"


def _vector(out) -> np.ndarray:
    """A torch tensor, numpy array or list -> float64 (D,)."""
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    if isinstance(out, (tuple, list)):
        out = out[-1]
    return np.asarray(out, dtype=np.float64).reshape(-1)


def _missing(kind: str, what: str, exc: BaseException) -> RuntimeError:
    return RuntimeError(f"embedder {kind}: {what} is not available here ({exc}). Install it in this env, or pick "
                        f"another --embedder-kind.")


class _Adapter:
    """Shared lazy loading. Subclasses set name, id_string and implement _load() and _embed(wave)."""

    name = ""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._state = None

    @property
    def id_string(self) -> str:
        return _safe(self.name)

    def load(self):
        if self._state is None:
            self._state = self._load()
        return self._state

    def embed(self, wave: np.ndarray) -> np.ndarray:
        self.load()
        return self._embed(np.asarray(wave, dtype=np.float32).reshape(-1))

    def _load(self):
        raise NotImplementedError

    def _embed(self, wave: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _OnnxAdapter(_Adapter):
    """A WeSpeaker-style ONNX model run by models/wespeaker_embedding.WeSpeakerONNXEmbedder (fbank 80, input 'feats')."""

    def __init__(self, name: str, onnx_path: Optional[str], device: str = "cpu"):
        super().__init__(device)
        self.name = name
        self.onnx_path = onnx_path

    @property
    def id_string(self) -> str:
        base = os.path.basename(self.onnx_path) if self.onnx_path else "hub-default"
        return _safe(f"{self.name}-{base}")

    def _load(self):
        if self.onnx_path and not os.path.isfile(self.onnx_path):
            raise RuntimeError(f"embedder {self.name}: no ONNX file at {self.onnx_path}")
        try:
            from models.wespeaker_embedding import WeSpeakerONNXEmbedder
        except ImportError as exc:
            raise _missing(self.name, "models.wespeaker_embedding (onnxruntime, torch, torchaudio)", exc)
        path = self.onnx_path

        class _FixedPath(WeSpeakerONNXEmbedder):
            def _model_path(self):          # the constructor only knows the HF hub or WESPEAKER_MODEL_PATH
                return path or super()._model_path()

        try:
            session = _FixedPath(device=self.device)
            session.active_providers()      # opens the ONNX session now: a bad file or no download fails here
        except Exception as exc:            # e.g. the hub download fails without a network or a cached copy
            raise RuntimeError(f"embedder {self.name}: cannot open the model ({exc})") from exc
        return session

    def _embed(self, wave):
        out = self._state.embed(wave, SR)
        return _vector(out)


class R34LMAdapter(_Adapter):
    """pyannote's WeSpeaker ResNet34-LM (arm J's embedder). masked_embed_fn() is the exact call of arm J."""

    name = "r34lm"

    def __init__(self, model: str = EMBEDDING_REPO, device: str = "cpu"):
        super().__init__(device)
        self.model = model

    @property
    def id_string(self) -> str:
        return _safe(f"r34lm-{os.path.basename(self.model.rstrip('/'))}")

    def _load(self):
        from tools import windowed_vbx_stage as stage     # lazy: that module imports torch only inside its main()
        try:
            emb, torch, path = stage.build_embedder(self.model, self.device)
        except SystemExit as exc:           # build_embedder reports a missing package or download this way
            raise RuntimeError(f"embedder r34lm: {exc}") from exc
        return {"emb": emb, "torch": torch, "path": path, "stage": stage}

    def _embed(self, wave):
        torch = self._state["torch"]
        x = torch.from_numpy(np.asarray(wave, dtype=np.float32)).to(self.device).reshape(1, 1, -1)
        with torch.no_grad():
            return _vector(self._state["emb"](x))

    def masked_embed_fn(self):
        """(embed(clip, mask), failed counter) as make_masked_embedder makes them. Loads the model."""
        state = self.load()
        return state["stage"].make_masked_embedder(state["emb"], state["torch"], self.device)


class SimamDirAdapter(_Adapter):
    """WeSpeaker SimAM-ResNet100 from a model folder (config.yaml, avg_model.pt) with the `wespeaker` package.

    UNVERIFIED, written from memory of the wespeaker package: get_speaker_model(cfg["model"])(**cfg["model_args"]),
    load_checkpoint(model, avg_model.pt), kaldi fbank with the feature_args of config.yaml, mean-normalised over time,
    and model(feats) returning the embedding (or a tuple whose last item is it). Check each call before a real run.
    """

    name = "wespeaker_simam100"

    def __init__(self, model_dir: str, device: str = "cpu"):
        super().__init__(device)
        self.model_dir = model_dir

    @property
    def id_string(self) -> str:
        return _safe(f"wespeaker_simam100-{os.path.basename(os.path.normpath(self.model_dir))}")

    def _load(self):
        if os.path.isfile(os.path.join(self.model_dir, "model.safetensors")):
            return self._load_safetensors()
        cfg_path = os.path.join(self.model_dir, "config.yaml")
        ckpt = os.path.join(self.model_dir, "avg_model.pt")
        if not (os.path.isfile(cfg_path) and os.path.isfile(ckpt)):
            raise RuntimeError(f"embedder wespeaker_simam100: {self.model_dir} needs config.yaml and avg_model.pt, "
                               f"or model.safetensors and config.json")
        try:
            import torch
            import yaml
            import torchaudio.compliance.kaldi as kaldi
            from wespeaker.models.speaker_model import get_speaker_model
            from wespeaker.utils.checkpoint import load_checkpoint
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            model = get_speaker_model(cfg["model"])(**cfg.get("model_args", {}))
            load_checkpoint(model, ckpt)
            model.eval()
        except ImportError as exc:
            raise _missing(self.name, "the wespeaker, torchaudio and yaml packages", exc)
        except Exception as exc:            # UNVERIFIED API: the message says so
            raise RuntimeError(f"embedder wespeaker_simam100: loading {self.model_dir} with the wespeaker package "
                               f"failed ({exc}). The wespeaker API may differ from what this adapter assumes "
                               f"(UNVERIFIED); use --simam-onnx instead.") from exc
        model.to(self.device)
        feat = cfg.get("feature_args", {})
        return {"model": model, "torch": torch, "kaldi": kaldi,
                "bins": int(feat.get("num_mel_bins", 80)), "frame_length": float(feat.get("frame_length", 25)),
                "frame_shift": float(feat.get("frame_shift", 10))}

    def _load_safetensors(self):
        """The Hugging Face layout of the model card: model.safetensors (PyTorch key names) and config.json.
        SimAM_ResNet100_ASP of wespeaker/models/samresnet.py, loaded strict; the fbank settings are in config.json.
        Checked against the card and that source file; not yet against a real run."""
        import json
        cfg_path = os.path.join(self.model_dir, "config.json")
        if not os.path.isfile(cfg_path):
            raise RuntimeError(f"embedder wespeaker_simam100: {self.model_dir} has model.safetensors but no config.json")
        try:
            import torch
            import torchaudio.compliance.kaldi as kaldi
            from safetensors.torch import load_file
            try:
                from wespeaker_vendor.samresnet import SimAM_ResNet100_ASP     # tools/ on sys.path
            except ImportError:
                from tools.wespeaker_vendor.samresnet import SimAM_ResNet100_ASP
        except ImportError as exc:
            raise _missing(self.name, "the vendored WeSpeaker model (tools/wespeaker_vendor), safetensors and torchaudio", exc)
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        model = SimAM_ResNet100_ASP(in_planes=cfg["in_planes"], embed_dim=cfg["embed_dim"],
                                    acoustic_dim=cfg["acoustic_dim"], dropout=cfg.get("dropout", 0))
        try:
            model.load_state_dict(load_file(os.path.join(self.model_dir, "model.safetensors")), strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"embedder wespeaker_simam100: the weights do not fit SimAM_ResNet100_ASP ({exc})") from exc
        model.eval()
        model.to(self.device)
        fb = cfg["fbank"]
        return {"model": model, "torch": torch, "kaldi": kaldi, "bins": int(fb["num_mel_bins"]),
                "frame_length": float(fb["frame_length_ms"]), "frame_shift": float(fb["frame_shift_ms"])}

    def _embed(self, wave):
        st = self._state
        torch = st["torch"]
        x = torch.as_tensor(wave * (1 << 15), dtype=torch.float32).reshape(1, -1)
        feats = st["kaldi"].fbank(x, num_mel_bins=st["bins"], frame_length=st["frame_length"],
                                  frame_shift=st["frame_shift"], dither=0.0, sample_frequency=SR,
                                  window_type="hamming", use_energy=False)
        feats = feats - feats.mean(dim=0, keepdim=True)
        with torch.no_grad():
            return _vector(st["model"](feats.unsqueeze(0).to(self.device)))


class EspnetAdapter(_Adapter):
    """ESPnet-SPK through espnet2.bin.spk_inference.Speech2Embedding. UNVERIFIED, written from memory of the API."""

    name = "espnet_spk"

    def __init__(self, tag: str = DEFAULT_ESPNET_TAG, device: str = "cpu"):
        super().__init__(device)
        self.tag = tag

    @property
    def id_string(self) -> str:
        return _safe(f"espnet_spk-{self.tag.split('/')[-1]}")

    def _load(self):
        try:
            import torch
            from espnet2.bin.spk_inference import Speech2Embedding
        except ImportError as exc:
            raise _missing(self.name, "espnet and torch (espnet2.bin.spk_inference)", exc)
        try:
            s2e = Speech2Embedding.from_pretrained(model_tag=self.tag)     # UNVERIFIED call
        except Exception as exc:
            raise RuntimeError(f"embedder espnet_spk: cannot load model tag {self.tag!r} ({exc}). Check the tag "
                               f"on the espnet Hugging Face hub (UNVERIFIED API).") from exc
        # Speech2Embedding keeps its network on the CPU by default; move every torch module it holds to the device.
        for value in list(vars(s2e).values()):
            if isinstance(value, torch.nn.Module):
                value.to(self.device)
        return {"s2e": s2e, "torch": torch}

    def _embed(self, wave):
        torch = self._state["torch"]
        out = self._state["s2e"](torch.as_tensor(wave, dtype=torch.float32))   # UNVERIFIED: 16 kHz tensor in
        return _vector(out)


def validate_args(kind: str, r293_onnx=None, simam_onnx=None, simam_dir=None) -> None:
    """Refuses bad combinations before any model is touched. Raises ValueError with the message to show."""
    if kind not in KINDS:
        raise ValueError(f"unknown embedder kind {kind!r}, one of {', '.join(KINDS)}")
    if kind == "wespeaker_simam100":
        if bool(simam_onnx) == bool(simam_dir):
            raise ValueError("wespeaker_simam100 needs exactly one of --simam-onnx PATH or --simam-dir DIR")


def make_embedder(kind: str, *, device: str = "cpu", r293_onnx: Optional[str] = None,
                  simam_onnx: Optional[str] = None, simam_dir: Optional[str] = None,
                  espnet_tag: str = DEFAULT_ESPNET_TAG) -> _Adapter:
    """The adapter of one kind, not loaded yet (call load(), or embed(), which loads)."""
    validate_args(kind, r293_onnx=r293_onnx, simam_onnx=simam_onnx, simam_dir=simam_dir)
    if kind == "r34lm":
        return R34LMAdapter(device=device)
    if kind == "wespeaker_r293":
        return _OnnxAdapter("wespeaker_r293", r293_onnx, device=device)
    if kind == "wespeaker_simam100":
        if simam_onnx:
            return _OnnxAdapter("wespeaker_simam100", simam_onnx, device=device)
        return SimamDirAdapter(simam_dir, device=device)
    return EspnetAdapter(espnet_tag, device=device)


def make_concat_embedder(adapter, min_s: float = DEFAULT_MIN_S, sr: int = SR):
    """embed_fn(window_audio, frame_mask) -> (D,) float64 or None, for the dz pipeline (not r34lm).

    The frame mask has len(mask) frames over the window: sample j belongs to frame floor(j * F / N). The samples of the
    frames with mask > 0.5 are concatenated and given to adapter.embed. Fewer than min_s seconds of them gives None.
    A ValueError from the adapter (one clip refused) also gives None and is counted. Other errors, such as a missing
    package, propagate: they are not a per-window refusal. embed_fn.counts holds the calls, the short windows and the
    refusals.
    """
    if not (math.isfinite(min_s) and min_s >= 0):
        raise ValueError(f"min_s must be 0 or more, got {min_s}")
    counts = {"calls": 0, "too_short": 0, "refused": 0}

    def embed_fn(window_audio, frame_mask):
        counts["calls"] += 1
        wave = np.asarray(window_audio, dtype=np.float32).reshape(-1)
        mask = np.asarray(frame_mask, dtype=float).reshape(-1)
        n, f = len(wave), len(mask)
        if n == 0 or f == 0:
            counts["too_short"] += 1
            return None
        frame_of = (np.arange(n, dtype=np.int64) * f) // n
        clean = wave[mask[frame_of] > 0.5]
        if len(clean) < min_s * sr:
            counts["too_short"] += 1
            return None
        try:
            vec = adapter.embed(clean)
        except ValueError:
            counts["refused"] += 1
            return None
        return np.asarray(vec, dtype=np.float64).reshape(-1)

    embed_fn.counts = counts
    return embed_fn
