"""DBS-TIGER (JusperLee/Dynamic-Band-Splitting, arXiv 2610.14993), 44.1 kHz dialogue/music/sfx separator.

STATUS: ANALYSIS-ONLY (project policy). The dialogue stem is a model output, not a cleaned
recording. The model was trained only on DnR v2 (clean English audiobook speech + FMA music +
FSD50k effects) and has not been tested on Vietnamese conversation. Do not export its output as
training audio without a listening test and the verification gates.

What it does: it removes music and sound effects. It does NOT remove crowd/background voices
(there was no such class in training; they are likely kept, smeared or partly cut).
Unknown (no evidence in the paper or repo): behaviour on laughs, breaths, coughs and backchannels
(they may be routed to sfx and deleted), sung vocals in music beds, reverberant or overlapped
speech, real-time factor and memory. residual_gain exists because of the first one.

Config (all optional):
  stem            "speech" (default, output index 2) | "music" (0) | "sfx" (1); the non-speech stems
                  let you LISTEN to what the model removes
  residual_gain   >= 0, default 0.0: out = stem + residual_gain * (sum of the other two stems);
                  0.25 keeps a quarter of what was removed
  window_s / hop_s  upstream protocol, default 6.0 / 3.0 (uniform-weight overlap average, as
                  dbs/engine/inference.py `chunked`, replicated here so the repo's training
                  dependencies are not needed)
  repo_dir        clone of the repo (env DBS_TIGER_DIR)
  checkpoint      local model.safetensors (env DBS_TIGER_CKPT); config.json next to it is used for
                  the model arguments when present, else the class defaults (= released config)
  auto_download   false by default; true fetches model.safetensors + config.json from the Hugging
                  Face repo `hf_repo` (default JusperLee/DBS-TIGER-DnR) into SOMMELIER_DENOISE_DIR
  device, seed    as the base class
Setup: scripts/denoise_dbs_tiger.md.
"""
import contextlib
import inspect
import json
import os
import sys
import threading
from typing import Callable, Optional, Tuple

import numpy as np

from .base import DenoiseBackend, DenoiseError, has_module, missing_modules, weights_root

STEMS = {"music": 0, "sfx": 1, "speech": 2}  # upstream SOURCES = ("music", "sfx", "speech")
DEFAULT_HF_REPO = "JusperLee/DBS-TIGER-DnR"
_IMPORT_LOCK = threading.Lock()


def _repo_dir(cfg: dict) -> str:
    return str(cfg.get("repo_dir") or os.environ.get("DBS_TIGER_DIR") or "")


def _hf_dir(cfg: dict) -> str:
    return os.path.join(weights_root(), "dbs_tiger")


def _ckpt_path(cfg: dict) -> str:
    explicit = cfg.get("checkpoint") or os.environ.get("DBS_TIGER_CKPT")
    if explicit:
        return str(explicit)
    return ""


def window_average(run: Callable[[np.ndarray], np.ndarray], x: np.ndarray, chunk: int, hop: int) -> np.ndarray:
    """Upstream `chunked` (dbs/engine/inference.py) on a 1-D array: run(piece) -> (S, len(piece)).

    Windows of `chunk` every `hop`, the last one zero-padded and cropped, averaged with uniform
    weights. A signal up to `chunk` long runs in one call: unpadded when longer than chunk // 2,
    else zero-padded to `chunk` (the STFT needs more than n_fft // 2 samples). Returns (S, len(x)).
    """
    n = len(x)
    chunk = max(1, int(chunk))
    hop = max(1, min(int(hop), chunk))
    if n <= chunk:
        if n > chunk // 2:
            return np.asarray(run(x), dtype=np.float32)[:, :n]
        pad = np.concatenate([x, np.zeros(chunk - n, dtype=x.dtype)])
        return np.asarray(run(pad), dtype=np.float32)[:, :n]
    out = weight = None
    start = 0
    while start < n:
        valid = min(chunk, n - start)
        piece = x[start:start + valid]
        if valid < chunk:
            piece = np.concatenate([piece, np.zeros(chunk - valid, dtype=x.dtype)])
        est = np.asarray(run(piece), dtype=np.float64)[:, :valid]
        if out is None:
            out = np.zeros((est.shape[0], n), dtype=np.float64)
            weight = np.zeros(n, dtype=np.float64)
        out[:, start:start + valid] += est
        weight[start:start + valid] += 1.0
        if start + chunk >= n:
            break
        start += hop
    return (out / weight).astype(np.float32)


def mix_stems(stems: np.ndarray, stem: str, residual_gain: float) -> np.ndarray:
    """stems (3, L) ordered music, sfx, speech -> chosen stem + residual_gain * sum of the other two."""
    k = STEMS[stem]
    out = stems[k].astype(np.float32, copy=True)
    if residual_gain:
        rest = sum(stems[i].astype(np.float32) for i in range(3) if i != k)
        out = out + np.float32(residual_gain) * rest
    return out


@contextlib.contextmanager
def _repo_on_path(repo_dir: str):
    """`dbs` is a unique top-level name, so the clone only needs to be importable while we import it."""
    with _IMPORT_LOCK:
        sys.path.insert(0, repo_dir)
        try:
            yield
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(repo_dir)


class DBSTigerBackend(DenoiseBackend):
    """Dialogue (default), music or sfx stem of DBS-TIGER at 44.1 kHz. ANALYSIS-ONLY: see the module docstring."""
    name = "dbs_tiger"
    native_sr = 44100
    default_chunk_s = 0.0  # windowing is done by window_s / hop_s, not by the base class
    default_overlap_s = 0.0

    def __init__(self, cfg: Optional[dict] = None, logger=None):
        super().__init__(cfg, logger)
        self.stem = str(self.cfg.get("stem", "speech")).strip().lower()
        if self.stem not in STEMS:
            raise ValueError(f"{self.name}: stem must be one of {sorted(STEMS)}, got {self.stem!r}")
        self.residual_gain = float(self.cfg.get("residual_gain", 0.0))
        if not np.isfinite(self.residual_gain) or self.residual_gain < 0:
            raise ValueError(f"{self.name}: residual_gain must be >= 0, got {self.residual_gain}")
        self.window_s = float(self.cfg.get("window_s", 6.0))
        self.hop_s = float(self.cfg.get("hop_s", 3.0))
        if self.window_s <= 0 or not 0 < self.hop_s <= self.window_s:
            raise ValueError(f"{self.name}: need 0 < hop_s <= window_s, got window_s={self.window_s}, hop_s={self.hop_s}")
        self._model = None

    @classmethod
    def check_available(cls, cfg: Optional[dict] = None) -> Tuple[bool, str]:
        cfg = cfg or {}
        miss = missing_modules("torch", "safetensors", "einops", "rotary_embedding_torch", "huggingface_hub")
        if miss:
            return False, f"missing python package(s): {', '.join(miss)} (pip install -r requirements-denoise-dbs-tiger.txt)"
        repo = _repo_dir(cfg)
        if not repo:
            return False, "DBS_TIGER_DIR is not set (clone JusperLee/Dynamic-Band-Splitting and point DBS_TIGER_DIR at it)"
        if not os.path.isfile(os.path.join(repo, "dbs", "models", "tiger.py")):
            return False, f"DBS_TIGER_DIR={repo} does not look like a Dynamic-Band-Splitting clone (dbs/models/tiger.py missing)"
        ckpt = _ckpt_path(cfg)
        if ckpt:
            if not os.path.isfile(ckpt):
                return False, f"checkpoint not found: {ckpt}"
        elif cfg.get("auto_download"):
            pass  # fetched from the hub in load()
        else:
            cached = os.path.join(_hf_dir(cfg), "model.safetensors")
            if not os.path.isfile(cached):
                return False, ("no checkpoint: set DBS_TIGER_CKPT to a local model.safetensors "
                               "(JusperLee/DBS-TIGER-DnR) or pass auto_download=true")
        return True, ""

    def _resolve_checkpoint(self) -> str:
        ckpt = _ckpt_path(self.cfg)
        if ckpt:
            return ckpt
        cached = os.path.join(_hf_dir(self.cfg), "model.safetensors")
        if os.path.isfile(cached) or not self.cfg.get("auto_download"):
            return cached
        from huggingface_hub import hf_hub_download
        repo_id = str(self.cfg.get("hf_repo", DEFAULT_HF_REPO))
        self._log(f"downloading {repo_id} to {_hf_dir(self.cfg)}")
        with contextlib.suppress(Exception):  # config.json is optional (class defaults match the release)
            hf_hub_download(repo_id, "config.json", local_dir=_hf_dir(self.cfg))
        return hf_hub_download(repo_id, "model.safetensors", local_dir=_hf_dir(self.cfg))

    def _build_model(self, ckpt: str):
        """Instantiate DBSTIGER from the clone and load the safetensors weights (strict)."""
        from safetensors.torch import load_file

        with _repo_on_path(_repo_dir(self.cfg)):
            from dbs.models import DBSTIGER
        kwargs = {}
        conf = os.path.join(os.path.dirname(ckpt), "config.json")
        if os.path.isfile(conf):
            with open(conf) as f:
                saved = json.load(f)
            accepted = set(inspect.signature(DBSTIGER.__init__).parameters)
            kwargs = {k: v for k, v in saved.items() if k in accepted}
        model = DBSTIGER(**kwargs)
        model.load_state_dict(load_file(ckpt), strict=True)
        return model

    def _load(self):
        ok, why = self.check_available(self.cfg)
        if not ok:
            raise DenoiseError(f"{self.name}: {why}")
        import torch

        self._seed_all()
        dev = torch.device("cuda" if self.device == "auto" and torch.cuda.is_available() else
                           "cpu" if self.device == "auto" else self.device)
        model = self._build_model(self._resolve_checkpoint())
        self._dev, self._model = dev, model.to(dev).eval()

    def _unload(self):
        self._model = None
        if has_module("torch"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_model(self, piece: np.ndarray) -> np.ndarray:
        """One window (n,) -> (3, n) stems."""
        import torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(piece, dtype=np.float32)).view(1, 1, -1).to(self._dev)
            est = self._model(x)
            est = est[0] if isinstance(est, (tuple, list)) else est
        return est[0].float().cpu().numpy()

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        chunk = int(round(self.window_s * self.native_sr))
        hop = int(round(self.hop_s * self.native_sr))
        stems = window_average(self._run_model, x, chunk, hop)
        if not np.all(np.isfinite(stems)):
            raise DenoiseError(f"{self.name}: model produced non-finite samples")
        return mix_stems(stems, self.stem, self.residual_gain)
