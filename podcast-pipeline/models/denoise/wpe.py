"""Single-channel WPE dereverberation (nara_wpe, numpy), 16 kHz.

WPE predicts the late reverberation of each STFT bin from the bins `delay` frames or more earlier and subtracts it,
so the direct sound and its first reflections are left as they were. It is a linear filter fitted to the file itself:
no weights to download and no model that can invent speech.
"""
from typing import Optional, Tuple

import numpy as np

from .base import DenoiseBackend, DenoiseError, missing_modules


class WPEBackend(DenoiseBackend):
    """Config: taps (default 20), delay (3 frames), iterations (3), stft_size (512), stft_shift (128),
    chunk_s (default 60), overlap_s (default 2).

    A single channel is the weak case for WPE: on a synthetic harmonic signal with a 0.6 s decay the tail drops about
    3-4 dB at taps 10-30, more with longer taps but then the direct sound loses energy too (-3 dB at taps 40,
    delay 2). It is not validated on Vietnamese speech; judge it by listening before it goes in a profile.
    """
    name = "wpe"
    native_sr = 16000
    default_chunk_s = 60.0
    default_overlap_s = 2.0

    def __init__(self, cfg: Optional[dict] = None, logger=None):
        super().__init__(cfg, logger)
        self.taps = int(self.cfg.get("taps", 20))
        self.wpe_delay = int(self.cfg.get("delay", 3))
        self.iterations = int(self.cfg.get("iterations", 3))
        self.size = int(self.cfg.get("stft_size", 512))
        self.shift = int(self.cfg.get("stft_shift", 128))
        if min(self.taps, self.wpe_delay, self.iterations, self.size, self.shift) < 1 or self.shift > self.size:
            raise ValueError(f"wpe: taps, delay, iterations, stft_size and stft_shift must be >= 1 and shift <= size, "
                             f"got {self.taps}/{self.wpe_delay}/{self.iterations}/{self.size}/{self.shift}")

    @classmethod
    def check_available(cls, cfg: Optional[dict] = None) -> Tuple[bool, str]:
        miss = missing_modules("nara_wpe")
        if miss:
            return False, "missing python package nara_wpe (pip install -r requirements-denoise.txt)"
        return True, ""

    def _load(self):
        ok, why = self.check_available(self.cfg)
        if not ok:
            raise DenoiseError(f"wpe: {why}")
        from nara_wpe.wpe import wpe
        from nara_wpe.utils import istft, stft
        self._wpe, self._stft, self._istft = wpe, stft, istft

    def _unload(self):
        self._wpe = self._stft = self._istft = None

    def _one(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        # fewer frames than taps + delay: nothing to predict from, hand the audio back
        if n < (self.taps + self.wpe_delay + 2) * self.shift + self.size:
            return x.astype(np.float32)
        spec = self._stft(x.astype(np.float64), size=self.size, shift=self.shift)      # (T, F)
        out = self._wpe(spec.T[:, None, :], taps=self.taps, delay=self.wpe_delay,
                        iterations=self.iterations, statistics_mode="full")           # (F, 1, T)
        y = self._istft(out[:, 0, :].T, size=self.size, shift=self.shift)[:n]
        return np.pad(y, (0, n - len(y))).astype(np.float32)

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        return self._chunked(self._one, x)
