"""Test-only backends: no dependencies, work at any sample rate."""
import numpy as np

from .base import DenoiseBackend


class IdentityBackend(DenoiseBackend):
    """Returns the audio unchanged. Config: none needed."""
    name = "identity"

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        return x


class FakeHighPassBackend(DenoiseBackend):
    """Deterministic zero-phase Butterworth high-pass that stands in for a real filter in tests.

    Config: cutoff_hz (default 200), order (default 4; applied forward and backward).
    """
    name = "fake_hp"

    def __init__(self, cfg=None, logger=None):
        super().__init__(cfg, logger)
        self.cutoff_hz = float(self.cfg.get("cutoff_hz", 200.0))
        self.order = int(self.cfg.get("order", 4))

    def enhance(self, wav, sr):
        if not 0 < self.cutoff_hz < sr / 2:
            raise ValueError(f"fake_hp: cutoff_hz={self.cutoff_hz} must be inside (0, sr/2={sr / 2})")
        self._sr = int(sr)
        return super().enhance(wav, sr)

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import butter, sosfilt, sosfiltfilt
        sos = butter(self.order, self.cutoff_hz, btype="highpass", fs=self._sr, output="sos")
        padlen = 3 * (2 * len(sos) + 1)
        if len(x) > padlen:
            return sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)
        return sosfilt(sos, x.astype(np.float64)).astype(np.float32)  # too short for filtfilt padding
