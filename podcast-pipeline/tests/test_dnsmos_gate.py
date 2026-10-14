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


def test_gate_loads_the_scorer_on_the_configured_device(tmp_path, monkeypatch):
    import models.dnsmos
    model = tmp_path / "dnsmos.onnx"
    model.write_bytes(b"")
    seen = []

    class _Score:
        def __init__(self, path, device="cpu"):
            seen.append(device)

        def __call__(self, audio, sample_rate, is_personalized_MOS):
            return {"OVRL": 3.0}

    monkeypatch.setattr(models.dnsmos, "ComputeScore", _Score)
    DNSMOSInputGate(model_path=str(model), device="cuda").check([0.1], 24000)
    assert seen == ["cuda"]


def test_gate_rejects_an_unknown_device():
    with pytest.raises(ValueError, match="device"):
        DNSMOSInputGate(device="tpu")


def test_device_does_not_invalidate_checkpointed_scores():
    # CPU and CUDA give the same score, so moving devices must not re-score a corpus.
    assert DNSMOSInputGate(device="cuda").config_fingerprint == \
        DNSMOSInputGate(device="cpu").config_fingerprint


def test_pipeline_reads_the_device_from_the_profile():
    service = PipelineService.__new__(PipelineService)
    service.input_audio_gate = None
    config = {"environments": {"a100": {"audio_quality": {"enabled": True, "device": "cuda"}}}}
    gate = service._resolve_input_audio_gate(config, SimpleNamespace(env="a100"))
    assert gate.device == "cuda"


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


def test_files_scored_at_once_share_one_scorer():
    import threading
    import time
    built = []

    def factory():
        time.sleep(0.05)              # wide enough for every thread to race in
        built.append(1)
        return lambda _audio, _sr: {"OVRL": 3.0}

    gate = DNSMOSInputGate(scorer_factory=factory)
    threads = [threading.Thread(target=gate.check, args=([0.1], 24000))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1


def test_score_survives_a_cleared_cache_through_the_progress_ledger(tmp_path):
    from utils.progress import ProgressLedger

    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    ledger = ProgressLedger(str(tmp_path))
    service = PipelineService.__new__(PipelineService)
    service.audio_svc = _AudioService()
    service.logger = None
    service.progress_ledger = ledger
    gate = _Gate()

    first, _ = service._check_input_audio_quality(gate, _Checkpoint(), str(source))
    assert gate.calls == 1

    # The cache is wiped (a fresh checkpoint) and the ledger is read back from disk.
    service.progress_ledger = ProgressLedger(str(tmp_path))
    second, decoded = service._check_input_audio_quality(gate, _Checkpoint(), str(source))
    assert second == first and decoded is None
    assert gate.calls == 1 and service.audio_svc.calls == 1


def test_a_changed_source_or_gate_is_scored_again_despite_the_ledger(tmp_path):
    from utils.progress import ProgressLedger

    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    service = PipelineService.__new__(PipelineService)
    service.audio_svc = _AudioService()
    service.logger = None
    service.progress_ledger = ProgressLedger(str(tmp_path))
    gate = _Gate()
    service._check_input_audio_quality(gate, _Checkpoint(), str(source))

    source.write_bytes(b"a different recording")
    service._check_input_audio_quality(gate, _Checkpoint(), str(source))
    assert gate.calls == 2


def test_skipped_scores_are_not_kept_in_the_ledger(tmp_path):
    from utils.progress import ProgressLedger

    ledger = ProgressLedger(str(tmp_path))
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")

    class SkippingGate(_Gate):
        def check(self, *args, **kwargs):
            self.calls += 1
            return {"skipped": True, "reason": "scorer unavailable"}

    service = PipelineService.__new__(PipelineService)
    service.audio_svc = _AudioService()
    service.logger = None
    service.progress_ledger = ledger
    service._check_input_audio_quality(SkippingGate(), _Checkpoint(), str(source))
    assert ledger.get_quality(str(source)) is None
