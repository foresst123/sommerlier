"""DenoiseBackend: the interface every noise filter implements.

Subclasses provide _load/_unload/_enhance_native (work at native_sr on a float32 mono array);
enhance() does validation, resampling to and from native_sr, delay compensation, the finite
check and the exact-length guarantee.
"""
import importlib.util
import os
from typing import Optional, Tuple

import numpy as np

from .chunking import align_delay, chunk_apply, compensate_delay, fit_length, resample_to


class DenoiseError(RuntimeError):
    """A denoise backend failed to load or produced unusable audio."""


def weights_root() -> str:
    """Default folder for downloaded denoise weights (env SOMMELIER_DENOISE_DIR)."""
    return os.environ.get("SOMMELIER_DENOISE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sommelier", "denoise")


def has_module(name: str) -> bool:
    """True when `name` is importable, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def missing_modules(*names: str) -> list:
    return [n for n in names if not has_module(n)]


class DenoiseBackend:
    """Base class. Common config keys (all optional):

    chunk_s (float)    longest span fed to the model at once, seconds of native_sr audio; 0 disables chunking
    overlap_s (float)  crossfaded overlap between spans (at most chunk_s / 4)
    device (str)       "auto" | "cpu" | "cuda" | "cuda:1"
    seed (int)         seed for anything stochastic (default 0)
    delay_compensation False | True/"auto" | int samples at native_sr; "auto" measures the lag on the
                       loudest part of the file and removes it (default depends on the backend)
    max_delay_ms (float) search range for "auto" (default 20)
    """
    name = ""
    native_sr: Optional[int] = None  # None = works at any rate, no resampling
    default_chunk_s = 30.0
    default_overlap_s = 1.0
    default_delay = False

    def __init__(self, cfg: Optional[dict] = None, logger=None):
        cfg = dict(cfg or {})
        self.cfg = cfg
        self.logger = logger
        self.chunk_s = float(cfg.get("chunk_s", self.default_chunk_s))
        self.overlap_s = float(cfg.get("overlap_s", self.default_overlap_s))
        if self.chunk_s < 0 or self.overlap_s < 0 or 4 * self.overlap_s > self.chunk_s > 0:
            raise ValueError(f"{self.name}: need 0 <= overlap_s <= chunk_s / 4, got chunk_s={self.chunk_s}, overlap_s={self.overlap_s}")
        self.device = str(cfg.get("device", "auto"))
        self.seed = int(cfg.get("seed", 0))
        delay = cfg.get("delay_compensation", self.default_delay)
        self.delay = "auto" if delay is True or delay == "auto" else (False if delay in (False, None) else int(delay))
        self.max_delay_ms = float(cfg.get("max_delay_ms", 20.0))
        self.loaded = False

    # ---------------------------------------------------------------- registry
    @classmethod
    def check_available(cls, cfg: Optional[dict] = None) -> Tuple[bool, str]:
        """(importable and weights present, reason if not). Must not import heavy dependencies."""
        return True, ""

    # ------------------------------------------------------------- lifecycle
    def load(self):
        if not self.loaded:
            self._load()
            self.loaded = True
        return self

    def unload(self):
        if self.loaded:
            self._unload()
            self.loaded = False

    def __enter__(self):
        return self.load()

    def __exit__(self, *exc):
        self.unload()

    def _load(self):
        pass

    def _unload(self):
        pass

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _log(self, msg: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(f"[denoise:{self.name}] {msg}")

    def _seed_all(self):
        np.random.seed(self.seed)
        if has_module("torch"):
            import torch
            torch.manual_seed(self.seed)

    def _chunked(self, fn, x: np.ndarray) -> np.ndarray:
        """Run fn over x in chunk_s spans with a crossfade (a single call when chunk_s is 0)."""
        if self.chunk_s <= 0 or not self.native_sr:
            return fn(x)
        return chunk_apply(x, fn, int(self.chunk_s * self.native_sr), int(self.overlap_s * self.native_sr))

    def _check_finite(self, y: np.ndarray, where: str):
        if not np.all(np.isfinite(y)):
            bad = int(np.count_nonzero(~np.isfinite(y)))
            raise DenoiseError(f"{self.name}: model produced {bad} non-finite sample(s) {where}")

    # ---------------------------------------------------------------- public
    def enhance(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """float32 mono 1-D at any sr -> float32 mono 1-D with the same length, sr and sample alignment."""
        wav = np.asarray(wav)
        if wav.ndim != 1:
            raise ValueError(f"{self.name}: enhance expects a mono 1-D array, got shape {wav.shape}")
        if not np.issubdtype(wav.dtype, np.floating):
            raise ValueError(f"{self.name}: enhance expects floating-point audio, got {wav.dtype}")
        sr = int(round(sr))
        n = len(wav)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        if not np.all(np.isfinite(wav)):
            raise ValueError(f"{self.name}: input audio contains NaN/inf")
        self.load()
        self._seed_all()
        x = resample_to(wav, sr, self.native_sr) if self.native_sr and sr != self.native_sr else np.array(wav, dtype=np.float32, copy=True)
        y = np.asarray(self._enhance_native(x), dtype=np.float32)
        self._check_finite(y, "(after the model)")
        y = fit_length(y, len(x))
        if self.delay:
            native = self.native_sr or sr
            if self.delay == "auto":
                y, lag = align_delay(x, y, max_lag=max(1, int(self.max_delay_ms * native / 1000)), probe=10 * native)
            else:
                y, lag = compensate_delay(y, self.delay), self.delay
            if lag:
                self._log(f"removed a {lag}-sample model delay", "debug")
        if self.native_sr and sr != self.native_sr:
            y = fit_length(resample_to(y, self.native_sr, sr), n)
        y = np.ascontiguousarray(y, dtype=np.float32)
        self._check_finite(y, "(after resampling)")
        return y
