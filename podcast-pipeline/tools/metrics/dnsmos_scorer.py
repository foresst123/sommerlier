"""DNSMOS used only as a measurement, never a gate on pipeline audio."""

import os
from pathlib import Path

import numpy as np

from utils.worker_env import resolve_checkpoint

WARNINGS = [
    "DNSMOS resamples to 16 kHz: AudioSR's extra 48 kHz bandwidth is not measured.",
    "Two-speaker mixtures are outside DNSMOS's single-speaker training distribution; compare rankings, not absolute BAK.",
    "Only axis F has matched windows. Upstream axes change the window population; aggregate by recording.",
]
MODEL_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"


class DNSMOSScorer:
    def __init__(self, path=None, profile=None):
        from models.dnsmos import ComputeScore
        root = Path(__file__).resolve().parents[3]
        configured = (profile or {}).get("models", {}).get("dnsmos", {}).get("model")
        resolved = resolve_checkpoint("DNSMOS", [path, os.environ.get("DNSMOS_MODEL"),
            configured, str(root / "offline_weights/dnsmos/sig_bak_ovr.onnx")],
            env_profile=profile, config_key="dnsmos")
        self.model = ComputeScore(resolved, device="cpu")

    def __call__(self, audio, sr):
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("DNSMOS requires nonempty finite audio")
        # The vendored scorer floors seconds before calculating its hops. An
        # exact 9.01s signal otherwise produces zero model evaluations.
        original_seconds = len(audio) / sr
        if original_seconds < 10:
            audio = np.tile(audio, int(np.ceil(10 * sr / len(audio))))[:10 * sr]
        result = self.model(audio, sr, False)
        result = {k: float(result[k]) for k in
                  ("SIG", "BAK", "OVRL", "SIG_raw", "BAK_raw", "OVRL_raw")}
        if not all(np.isfinite(v) for v in result.values()):
            raise ValueError("DNSMOS produced nonfinite scores")
        return dict(result, original_seconds=original_seconds)
