"""Lazy, model-free-testable input quality gate using DNSMOS.

DNSMOS returns OVRL MOS on a roughly 1--5 scale.  The pipeline uses the
explicit normalized contract ``clamp((OVRL - 1) / 4, 0, 1)``; therefore the
default threshold 0.3 means raw OVRL >= 2.2.  The ONNX scorer is imported only
when an enabled gate actually evaluates a file.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Callable, Mapping, Optional


GATE_VERSION = "dnsmos-input-v1"
DEFAULT_MIN_NORMALIZED_QUALITY = 0.30


class AudioQualityGateError(RuntimeError):
    """The enabled quality check could not produce a trustworthy score."""


class AudioQualityRejected(AudioQualityGateError):
    """The input's normalized DNSMOS quality is below the configured floor."""

    def __init__(self, source_path: Optional[str], report: Mapping[str, Any]):
        super().__init__(
            f"DNSMOS rejected {source_path or 'audio'}: "
            f"normalized_quality={report['normalized_quality']:.3f} < "
            f"threshold={report['threshold']:.3f} (OVRL={report['OVRL']:.3f})")
        self.report = dict(report)


def normalized_quality(ovrl: float) -> float:
    value = float(ovrl)
    if not math.isfinite(value):
        raise ValueError("DNSMOS OVRL must be finite")
    return max(0.0, min(1.0, (value - 1.0) / 4.0))


class DNSMOSInputGate:
    """Score and accept/reject audio, with an injectable scorer for tests."""

    def __init__(
        self,
        *,
        profile: Optional[Mapping[str, Any]] = None,
        model_path: Optional[str] = None,
        min_normalized_quality: float = DEFAULT_MIN_NORMALIZED_QUALITY,
        enabled: bool = True,
        on_error: str = "reject",
        device: str = "cpu",
        scorer: Optional[Callable[[Any, int], Mapping[str, Any]]] = None,
        scorer_factory: Optional[Callable[[], Callable[[Any, int], Mapping[str, Any]]]] = None,
    ):
        threshold = float(min_normalized_quality)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("min_normalized_quality must be a finite value in [0, 1]")
        if on_error not in {"reject", "skip"}:
            raise ValueError("on_error must be 'reject' or 'skip'")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        self.profile = profile or {}
        # Not part of config_fingerprint: both devices give the same score, so
        # moving the scorer must not re-score every checkpointed file.
        self.device = device
        self.model_path = model_path
        self.threshold = threshold
        self.enabled = bool(enabled)
        self.on_error = on_error
        self._scorer = scorer
        self._scorer_factory = scorer_factory
        # Files scored at once must share one ONNX session, not build one each.
        self._scorer_lock = threading.Lock()

    @property
    def config_fingerprint(self) -> dict:
        return {"gate_version": GATE_VERSION, "threshold": self.threshold,
                "model_path": self.model_path, "on_error": self.on_error}

    def _get_scorer(self):
        if self._scorer is None:
            with self._scorer_lock:
                if self._scorer is None:
                    if self._scorer_factory is not None:
                        self._scorer = self._scorer_factory()
                    else:
                        if not self.model_path:
                            raise ValueError("DNSMOS model path is required")
                        model_path = os.fspath(self.model_path)
                        if not os.path.isabs(model_path):
                            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            model_path = os.path.abspath(os.path.join(project_dir, model_path))
                        if not os.path.isfile(model_path):
                            raise FileNotFoundError(
                                f"DNSMOS model not found: {model_path}; run "
                                "download_offline_weights.py to provision it")

                        # This is the repository's checked-in DNSMOS adapter.  Keep the
                        # import and ONNX session lazy so disabled profiles and unit tests
                        # neither load the model nor require an execution provider.
                        from models.dnsmos import ComputeScore
                        model = ComputeScore(model_path, device=self.device)

                        def score(audio, sample_rate):
                            return model(audio, sample_rate, is_personalized_MOS=False)

                        self._scorer = score
        return self._scorer

    def assess(self, waveform, sample_rate: int, source_path: Optional[str] = None) -> dict:
        if not self.enabled:
            return dict(self.config_fingerprint, enabled=False, passed=True,
                        skipped=True, reason="disabled")
        try:
            if waveform is None or len(waveform) == 0:
                raise ValueError("audio waveform is empty")
            scores = dict(self._get_scorer()(waveform, sample_rate))
            if "OVRL" not in scores:
                raise ValueError("DNSMOS scorer did not return OVRL")
            ovrl = float(scores["OVRL"])
            quality = normalized_quality(ovrl)
            report = dict(self.config_fingerprint, enabled=True,
                          passed=quality >= self.threshold, skipped=False,
                          source_path=source_path, OVRL=ovrl,
                          normalized_quality=quality, threshold=self.threshold)
            for key in ("SIG", "BAK", "SIG_raw", "BAK_raw", "OVRL_raw"):
                if key in scores:
                    value = float(scores[key])
                    if not math.isfinite(value):
                        raise ValueError(f"DNSMOS {key} must be finite")
                    report[key] = value
            return report
        except Exception as exc:
            if self.on_error == "skip":
                return dict(self.config_fingerprint, enabled=True, passed=True,
                            skipped=True, source_path=source_path,
                            reason=f"scorer_error:{type(exc).__name__}")
            if isinstance(exc, AudioQualityGateError):
                raise
            raise AudioQualityGateError(
                f"DNSMOS input check failed for {source_path or 'audio'}: "
                f"{type(exc).__name__}: {exc}") from exc

    def check(self, waveform, sample_rate: int, source_path: Optional[str] = None) -> dict:
        report = self.assess(waveform, sample_rate, source_path)
        if not report.get("passed", False):
            raise AudioQualityRejected(source_path, report)
        return report
