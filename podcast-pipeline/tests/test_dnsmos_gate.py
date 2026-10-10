import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.dnsmos_gate import (
    AudioQualityGateError,
    AudioQualityRejected,
    DNSMOSInputGate,
    normalized_quality,
)
from services.pipeline_service import PipelineService


class _Checkpoint:
    def __init__(self, report=None):
        self.report = report

    def load_if_committed(self, _name, fmt=None):
        return self.report

    def save(self, _name, value, fmt=None):
        self.report = value


class _Gate:
    enabled = True
    config_fingerprint = {"gate_version": "test", "threshold": 0.3}

    def __init__(self):
        self.calls = 0

    def check(self, waveform, sample_rate, source_path=None):
        self.calls += 1
        return {**self.config_fingerprint, "passed": True, "skipped": False,
                "OVRL": 3.0, "normalized_quality": 0.5, "threshold": 0.3,
                "source_path": source_path}


class _AudioService:
    def __init__(self):
        self.calls = 0

    def load_audio(self, _path, target_sr):
        self.calls += 1
        return SimpleNamespace(waveform=[0.1], sample_rate=target_sr)


def test_normalized_dns_mos_contract():
    assert normalized_quality(1.0) == pytest.approx(0.0)
    assert normalized_quality(2.2) == pytest.approx(0.3)
    assert normalized_quality(5.0) == pytest.approx(1.0)


def test_fake_scorer_accepts_at_threshold():
    gate = DNSMOSInputGate(
        scorer=lambda _audio, _sr: {"OVRL": 2.2},
        min_normalized_quality=0.3,
    )
    report = gate.check([0.0], 24000, "clean.wav")
    assert report["passed"] is True
    assert report["normalized_quality"] == pytest.approx(0.3)


def test_fake_scorer_rejects_below_threshold():
    gate = DNSMOSInputGate(scorer=lambda _audio, _sr: {"OVRL": 2.0})
    with pytest.raises(AudioQualityRejected, match="OVRL=2.000"):
        gate.check([0.0], 24000, "noisy.wav")


def test_disabled_gate_never_loads_scorer():
    loaded = []
    gate = DNSMOSInputGate(enabled=False, scorer_factory=lambda: loaded.append(True))
    assert gate.assess([0.0], 24000)["skipped"] is True
    assert loaded == []


def test_enabled_gate_fails_closed_when_the_local_model_is_missing(tmp_path):
    gate = DNSMOSInputGate(model_path=str(tmp_path / "missing.onnx"))
    with pytest.raises(AudioQualityGateError, match="download_offline_weights.py"):
        gate.check([0.0], 24000, "input.wav")


def test_skip_policy_returns_a_loggable_report_on_scorer_error():
    def broken(_audio, _sr):
        raise RuntimeError("broken scorer")

    report = DNSMOSInputGate(scorer=broken, on_error="skip").check([0.0], 24000)
    assert report["passed"] is True
    assert report["skipped"] is True
    assert report["reason"] == "scorer_error:RuntimeError"


def test_empty_audio_fails_before_loading_the_scorer():
    loaded = []
    gate = DNSMOSInputGate(scorer_factory=lambda: loaded.append(True))
    with pytest.raises(AudioQualityGateError, match="audio waveform is empty"):
        gate.check([], 24000, "empty.wav")
    assert loaded == []


def test_pipeline_cache_is_bound_to_the_current_source_file(tmp_path):
    source = tmp_path / "input.wav"
    source.write_bytes(b"first")
    service = PipelineService.__new__(PipelineService)
    service.audio_svc = _AudioService()
    service.logger = None
    gate, checkpoint = _Gate(), _Checkpoint()

    first, decoded = service._check_input_audio_quality(gate, checkpoint, str(source))
    assert decoded is not None and gate.calls == 1 and service.audio_svc.calls == 1
    second, decoded = service._check_input_audio_quality(gate, checkpoint, str(source))
    assert second == first and decoded is None and gate.calls == 1

    source.write_bytes(b"replacement with a different identity")
    third, decoded = service._check_input_audio_quality(gate, checkpoint, str(source))
    assert decoded is not None and gate.calls == 2 and service.audio_svc.calls == 2
    assert third["source_size"] == source.stat().st_size
