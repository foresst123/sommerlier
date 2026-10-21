"""MP-SENet (yxlu-0102/MP-SENet), 16 kHz. Not a pip package: point MPSENET_DIR at a clone of the repo."""
import contextlib
import json
import os
import sys
import threading
import types
from typing import Optional, Tuple

import numpy as np

from .base import DenoiseBackend, DenoiseError, has_module, missing_modules

_IMPORT_LOCK = threading.Lock()


class AttrDict(dict):
    """dict with attribute access, like MP-SENet's env.AttrDict (MPNet reads h.dense_channel etc.)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def _repo_dir(cfg: dict) -> str:
    return str(cfg.get("repo_dir") or os.environ.get("MPSENET_DIR") or "")


def _ckpt_path(cfg: dict) -> str:
    explicit = cfg.get("checkpoint") or os.environ.get("MPSENET_CKPT")
    if explicit:
        return str(explicit)
    # best_ckpt/g_best_vb (VoiceBank+DEMAND) or g_best_dns (DNS challenge), both shipped in the repo
    return os.path.join(_repo_dir(cfg), "best_ckpt", str(cfg.get("ckpt_name", "g_best_vb")))


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _unavailable(*_a, **_k):
    raise ImportError("stubbed out by models.denoise.mpsenet (only needed for training metrics)")


@contextlib.contextmanager
def _isolated_repo_imports(repo_dir: str):
    """Make `import models.model` / `import utils` resolve inside the MP-SENet clone.

    MP-SENet uses the top-level names `models` and `utils`, which this project also uses. We set
    ours aside, import theirs, then put ours back. The imported classes keep working afterwards.
    pesq / joblib / matplotlib are imported by upstream at module level but only used for
    training metrics and plots, so they are stubbed when not installed.
    """
    shadow = lambda k: k in ("models", "utils") or k.startswith(("models.", "utils."))  # noqa: E731
    with _IMPORT_LOCK:
        saved = {k: v for k, v in sys.modules.items() if shadow(k)}
        for k in saved:
            del sys.modules[k]
        stubs = {}
        if not has_module("pesq"):
            stubs["pesq"] = _stub("pesq", pesq=_unavailable)
        if not has_module("joblib"):
            stubs["joblib"] = _stub("joblib", Parallel=_unavailable, delayed=_unavailable)
        if not has_module("matplotlib"):
            plt = _stub("matplotlib.pylab")
            stubs["matplotlib"] = _stub("matplotlib", use=lambda *a, **k: None, pylab=plt)
            stubs["matplotlib.pylab"] = plt
        stubbed = [k for k in stubs if k not in sys.modules]
        for k in stubbed:
            sys.modules[k] = stubs[k]
        # upstream's models/ has no __init__.py, so a regular package of ours would win the import: give it an explicit one
        pkg = types.ModuleType("models")
        pkg.__path__ = [os.path.join(repo_dir, "models")]
        sys.modules["models"] = pkg
        sys.path.insert(0, repo_dir)
        try:
            yield
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(repo_dir)
            for k in [k for k in sys.modules if shadow(k)]:
                del sys.modules[k]
            sys.modules.update(saved)
            for k in stubbed:
                sys.modules.pop(k, None)


def load_mpnet_class(repo_dir: str):
    with _isolated_repo_imports(repo_dir):
        from models.model import MPNet
    return MPNet


class MPSENetBackend(DenoiseBackend):
    """MP-SENet magnitude+phase enhancement, 16 kHz.

    Follows MP-SENet/inference.py: norm_factor = sqrt(N / sum(x^2)) over the whole input, input
    scaled by it, compressed magnitude/phase STFT (n_fft 400, hop 100, win 400, centre=True,
    compress_factor 0.3), MPNet, inverse STFT, divided by norm_factor. The factor is computed once
    for the whole file and shared by all chunks so chunk levels match. The STFT is centred, so no
    delay is expected; delay_compensation="auto" only verifies that cheaply.

    Config: repo_dir (clone of the repo, env MPSENET_DIR), checkpoint (explicit file, env
    MPSENET_CKPT), ckpt_name (default g_best_vb; or g_best_dns, looked up in <repo_dir>/best_ckpt),
    config_json (default <checkpoint folder>/config.json, as upstream), chunk_s (default 10),
    overlap_s (default 1), device (default auto), seed.
    Weights: see scripts/denoise_weights.md.
    """
    name = "mpsenet"
    native_sr = 16000
    default_chunk_s = 10.0
    default_overlap_s = 1.0
    default_delay = "auto"

    @classmethod
    def check_available(cls, cfg: Optional[dict] = None) -> Tuple[bool, str]:
        cfg = cfg or {}
        miss = missing_modules("torch", "einops")
        if miss:
            return False, f"missing python package(s): {', '.join(miss)} (pip install -r requirements-denoise.txt)"
        repo = _repo_dir(cfg)
        if not repo:
            return False, "MPSENET_DIR is not set (clone yxlu-0102/MP-SENet and point MPSENET_DIR at it)"
        if not os.path.isfile(os.path.join(repo, "models", "model.py")):
            return False, f"MPSENET_DIR={repo} does not look like an MP-SENet clone (models/model.py missing)"
        ckpt = _ckpt_path(cfg)
        if not os.path.isfile(ckpt):
            return False, f"checkpoint not found: {ckpt}"
        if not os.path.isfile(str(cfg.get("config_json") or os.path.join(os.path.dirname(ckpt), "config.json"))):
            return False, f"config.json not found next to {ckpt}"
        return True, ""

    def _load(self):
        ok, why = self.check_available(self.cfg)
        if not ok:
            raise DenoiseError(f"{self.name}: {why}")
        import torch

        ckpt = _ckpt_path(self.cfg)
        with open(str(self.cfg.get("config_json") or os.path.join(os.path.dirname(ckpt), "config.json"))) as f:
            h = AttrDict(json.load(f))
        if int(h.sampling_rate) != self.native_sr:
            raise DenoiseError(f"{self.name}: config sampling_rate {h.sampling_rate} != {self.native_sr}")
        MPNet = load_mpnet_class(_repo_dir(self.cfg))
        self._seed_all()
        dev = torch.device("cuda" if self.device == "auto" and torch.cuda.is_available() else
                           "cpu" if self.device == "auto" else self.device)
        try:
            state = torch.load(ckpt, map_location=dev)
        except Exception:  # checkpoints saved with extra pickled objects need the unsafe loader (local trusted file)
            state = torch.load(ckpt, map_location=dev, weights_only=False)
        model = MPNet(h).to(dev)
        model.load_state_dict(state["generator"])
        self._h, self._dev, self._model = h, dev, model.eval()

    def _unload(self):
        self._model = None
        if has_module("torch"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # mag_pha_stft / mag_pha_istft from MP-SENet/dataset.py, reproduced here so importing
    # `dataset` (which pulls in librosa) is not needed.
    def _stft(self, y):
        import torch
        h = self._h
        window = torch.hann_window(h.win_size).to(y.device)
        spec = torch.view_as_real(torch.stft(y, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=window,
                                             center=True, pad_mode="reflect", normalized=False, return_complex=True))
        mag = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        pha = torch.atan2(spec[:, :, :, 1] + 1e-10, spec[:, :, :, 0] + 1e-5)
        return torch.pow(mag, h.compress_factor), pha

    def _istft(self, mag, pha):
        import torch
        h = self._h
        mag = torch.pow(mag, 1.0 / h.compress_factor)
        com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
        window = torch.hann_window(h.win_size).to(com.device)
        return torch.istft(com, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=window, center=True)

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        import torch

        energy = float(np.sum(x.astype(np.float64) ** 2))
        if energy <= 0.0:
            return x.copy()
        norm_factor = float(np.sqrt(len(x) / energy))
        min_len = 2 * int(self._h.n_fft)  # reflect padding needs more than n_fft // 2 samples

        def run(chunk: np.ndarray) -> np.ndarray:
            n = len(chunk)
            if n < min_len:
                chunk = np.concatenate([chunk, np.zeros(min_len - n, dtype=np.float32)])
            with torch.no_grad():
                wav = torch.from_numpy(np.ascontiguousarray(chunk * np.float32(norm_factor))).unsqueeze(0).to(self._dev)
                amp, pha = self._stft(wav)
                amp_g, pha_g, _ = self._model(amp, pha)
                out = self._istft(amp_g, pha_g) / norm_factor
            return out.squeeze(0).cpu().numpy()[:n]

        return self._chunked(run, x)
