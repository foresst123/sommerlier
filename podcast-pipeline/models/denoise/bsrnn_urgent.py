"""URGENT 2026 Track 1 discriminative baseline: BSRNN, 8-48 kHz (urgent-challenge/urgent2026_challenge_track1).

Not a pip package: point URGENT26_DIR at a clone of the repo and URGENT26_BSRNN_CKPT at bsrnn.ckpt
(huggingface.co/lichenda/icassp_2026_urgent_baseline, MIT). Needs espnet2 (see requirements-denoise-bsrnn.txt).
"""
import contextlib
import os
import pickle
import sys
import threading
import types
from typing import Optional, Tuple

import numpy as np

from .base import DenoiseBackend, DenoiseError, has_module, missing_modules

# espnet's STFTEncoder asserts fs % 48000 == 0 or 48000 % fs == 0, so 22.05 / 44.1 kHz cannot run natively
SUPPORTED_SR = (8000, 16000, 24000, 32000, 48000)
_IMPORT_LOCK = threading.Lock()


def model_sr(sr: int) -> int:
    """Rate the model runs at for input rate sr: sr itself when supported, else the next one up (48 kHz above 48 kHz)."""
    for s in SUPPORTED_SR:
        if sr <= s:
            return s
    return SUPPORTED_SR[-1]


def _repo_dir(cfg: dict) -> str:
    return str(cfg.get("repo_dir") or os.environ.get("URGENT26_DIR") or "")


def _ckpt_path(cfg: dict) -> str:
    return str(cfg.get("checkpoint") or os.environ.get("URGENT26_BSRNN_CKPT") or "")


class _Stub:
    """Stand-in for the pickled training objects (baseline_code Config, Lightning classes) in the checkpoint."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _CkptUnpickler(pickle.Unpickler):
    """Lets torch.load read a Lightning checkpoint without importing training code or running arbitrary globals."""
    _STUB = ("baseline_code", "pytorch_lightning", "lightning")
    _OK = ("torch", "collections", "numpy")
    _BUILTINS = {"set", "frozenset", "slice", "complex", "bytearray", "range"}

    def find_class(self, module, name):
        top = module.split(".")[0]
        if top in self._STUB:
            return type(name, (_Stub,), {})
        if top in self._OK or (module == "builtins" and name in self._BUILTINS):
            return super().find_class(module, name)
        raise DenoiseError(f"bsrnn26: checkpoint references an unexpected global {module}.{name}; refusing to load it")


def load_state_dict(path: str):
    """The `se_model.` weights of a SEModel Lightning checkpoint, as a plain state dict (prefix stripped)."""
    import torch

    pm = types.SimpleNamespace(Unpickler=_CkptUnpickler, load=pickle.load, loads=pickle.loads, dumps=pickle.dumps,
                               UnpicklingError=pickle.UnpicklingError, __name__="bsrnn26_pickle")
    ck = torch.load(path, map_location="cpu", weights_only=False, pickle_module=pm)  # the restricted unpickler above does the filtering
    sd = {k[len("se_model."):]: v for k, v in ck["state_dict"].items() if k.startswith("se_model.")}
    if not sd:
        raise DenoiseError(f"bsrnn26: no 'se_model.*' weights in {path} (flow_bsrnn.ckpt is the generative model and is not supported)")
    return sd


def model_dims(sd: dict) -> Tuple[int, int]:
    """(num_channel, num_layer) read off the weights, so no hyper-parameter has to be guessed."""
    ch = int(sd["bsrnn.bsrnn.band_split.fc.0.weight"].shape[0])
    layers = len({k.split(".")[3] for k in sd if k.startswith("bsrnn.bsrnn.rnn_time.")})
    return ch, layers


@contextlib.contextmanager
def _repo_on_path(repo_dir: str):
    with _IMPORT_LOCK:
        sys.path.insert(0, repo_dir)
        try:
            yield
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(repo_dir)


class BSRNN26Backend(DenoiseBackend):
    """BSRNN (38M parameters, 196 channels, 6 layers), discriminative complex-mask + residual model.

    Follows baseline_code/inference.py: model.se_model(wav (1, T), lengths, fs) with an adaptive STFT
    (Hann 20 ms / hop 10 ms at any fs, centred) and the band split truncated to the input bandwidth,
    so 8, 16, 24, 32 and 48 kHz run natively (24 kHz needs no resampling). Other rates are resampled to
    the next supported one (44.1 -> 48 kHz) and back with exact length. The decoder is told the input
    length, so output length equals input length; the centred STFT adds no delay (measured 0).

    The model is trained with a scale-invariant loss, so its output level is arbitrary (peaks of 3-6 and
    a non-zero output for silence were measured). Upstream renormalises to a 0.9 peak; here each chunk is
    instead multiplied by the least-squares gain a = <x, y> / <y, y> (clamped at 0), which returns the
    speech to the input level and maps silence to silence. Input is fed unscaled (training peak was <= 0.9).

    Config: repo_dir (clone of the Track 1 repo, env URGENT26_DIR), checkpoint (bsrnn.ckpt, env
    URGENT26_BSRNN_CKPT), chunk_s (default 20), overlap_s (default 2), device (default auto), seed.
    Weights: see scripts/denoise_bsrnn_urgent.md.
    """
    name = "bsrnn26"
    native_sr = None  # set per call in enhance(): the rate the model runs at
    default_chunk_s = 20.0
    default_overlap_s = 2.0
    default_delay = "auto"  # centred STFT: expect 0, but verify cheaply

    def enhance(self, wav: np.ndarray, sr: int) -> np.ndarray:
        self.native_sr = model_sr(int(round(sr)))
        return super().enhance(wav, sr)

    @classmethod
    def check_available(cls, cfg: Optional[dict] = None) -> Tuple[bool, str]:
        cfg = cfg or {}
        miss = missing_modules("torch", "torch_complex", "typeguard", "espnet2")
        if miss:
            return False, f"missing python package(s): {', '.join(miss)} (pip install -r requirements-denoise-bsrnn.txt)"
        repo = _repo_dir(cfg)
        if not repo:
            return False, "URGENT26_DIR is not set (clone urgent-challenge/urgent2026_challenge_track1 and point URGENT26_DIR at it)"
        if not os.path.isfile(os.path.join(repo, "baseline_code", "models", "bsrnn.py")):
            return False, f"URGENT26_DIR={repo} does not look like the Track 1 repo (baseline_code/models/bsrnn.py missing)"
        ckpt = _ckpt_path(cfg)
        if not ckpt:
            return False, "URGENT26_BSRNN_CKPT is not set (download bsrnn.ckpt, see scripts/denoise_bsrnn_urgent.md)"
        if not os.path.isfile(ckpt):
            return False, f"checkpoint not found: {ckpt}"
        return True, ""

    def _load(self):
        ok, why = self.check_available(self.cfg)
        if not ok:
            raise DenoiseError(f"{self.name}: {why}")
        import torch

        with _repo_on_path(_repo_dir(self.cfg)):
            from baseline_code.models.bsrnn import BSRNN_SE
        sd = load_state_dict(_ckpt_path(self.cfg))
        ch, layers = model_dims(sd)
        model = BSRNN_SE(num_channel=ch, num_layer=layers)
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            raise DenoiseError(f"{self.name}: checkpoint does not match the model ({str(e)[:200]})") from e
        self._seed_all()
        self._dev = torch.device("cuda" if self.device == "auto" and torch.cuda.is_available() else
                                 "cpu" if self.device == "auto" else self.device)
        self._model = model.to(self._dev).eval()

    def _unload(self):
        self._model = None
        if has_module("torch"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_model(self, chunk: np.ndarray, fs: int) -> np.ndarray:
        """Raw model output for one float32 chunk at rate fs, same length, arbitrary level."""
        import torch

        with torch.no_grad():
            wav = torch.from_numpy(np.ascontiguousarray(chunk)).unsqueeze(0).to(self._dev)
            lengths = torch.tensor([wav.shape[-1]], device=self._dev)
            out, _ = self._model(wav, lengths, fs)  # BSRNN_SE.forward(speech_mix, speech_lengths, fs)
        return out.squeeze(0).float().cpu().numpy()

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        fs = int(self.native_sr)
        min_len = 4 * (960 * fs // 48000)  # the centred STFT reflect-pads n_fft // 2 and needs more samples than that

        def run(chunk: np.ndarray) -> np.ndarray:
            n = len(chunk)
            if n < min_len:
                chunk = np.concatenate([chunk, np.zeros(min_len - n, dtype=np.float32)])
            y = np.asarray(self._run_model(chunk, fs), dtype=np.float32).reshape(-1)[:n]
            if len(y) != n or not np.all(np.isfinite(y)):
                raise DenoiseError(f"{self.name}: model returned {len(y)} samples for {n}, finite={bool(np.all(np.isfinite(y)))}")
            x64, y64 = chunk[:n].astype(np.float64), y.astype(np.float64)
            denom = float(y64 @ y64)
            gain = max(0.0, float(x64 @ y64) / denom) if denom > 1e-12 else 0.0
            return (y * np.float32(gain)).astype(np.float32)

        return self._chunked(run, x)
