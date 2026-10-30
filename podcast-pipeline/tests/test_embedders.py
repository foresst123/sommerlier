"""tools/embedders.py with fakes only. The real models were NOT run: no weights, no GPU, no torch in the test env."""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import embedders as EMB  # noqa: E402


class RecordingAdapter:
    """Stands in for an adapter: records the audio it is asked to embed and returns a fixed vector."""

    name = "fake"
    id_string = "fake"

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def embed(self, wave):
        self.calls.append(np.asarray(wave, dtype=np.float32).copy())
        if self.fail_on is not None and len(wave) == self.fail_on:
            raise ValueError("refused")
        return np.array([float(len(wave)), 1.0])


# ---- make_concat_embedder ----

def test_concat_embedder_passes_exactly_the_masked_samples_to_the_adapter():
    adapter = RecordingAdapter()
    embed = EMB.make_concat_embedder(adapter, min_s=0.0)
    wave = np.arange(16000, dtype=np.float32)                     # 1 s
    mask = np.zeros(10)
    mask[2:5] = 1.0                                               # frames 2, 3, 4 of 10 are clean
    mask[7] = 0.5                                                 # 0.5 is not > 0.5: dropped
    vec = embed(wave, mask)
    frame_of = (np.arange(16000) * 10) // 16000
    expected = wave[np.isin(frame_of, [2, 3, 4])]
    assert len(adapter.calls) == 1
    assert np.array_equal(adapter.calls[0], expected)
    assert vec.dtype == np.float64 and vec.tolist() == [float(len(expected)), 1.0]


def test_concat_embedder_returns_none_under_min_s_and_does_not_call_the_adapter():
    adapter = RecordingAdapter()
    embed = EMB.make_concat_embedder(adapter, min_s=0.5)
    wave = np.ones(16000, dtype=np.float32)
    mask = np.zeros(10)
    mask[0] = 1.0                                                 # 0.1 s of clean audio
    assert embed(wave, mask) is None
    assert adapter.calls == [] and embed.counts["too_short"] == 1
    mask[:5] = 1.0                                                # 0.5 s exactly: not under the minimum
    assert embed(wave, mask) is not None
    assert embed.counts == {"calls": 2, "too_short": 1, "refused": 0}


def test_concat_embedder_turns_a_refusal_into_none_and_counts_it():
    adapter = RecordingAdapter(fail_on=8000)
    embed = EMB.make_concat_embedder(adapter, min_s=0.0)
    wave = np.ones(16000, dtype=np.float32)
    mask = np.zeros(10)
    mask[:5] = 1.0                                                # 8000 clean samples: refused by the fake
    assert embed(wave, mask) is None and embed.counts["refused"] == 1


def test_concat_embedder_other_errors_propagate_and_bad_min_s_is_refused():
    class Broken:
        def embed(self, wave):
            raise RuntimeError("no package")

    with pytest.raises(RuntimeError):
        EMB.make_concat_embedder(Broken(), min_s=0.0)(np.ones(100), np.ones(4))
    with pytest.raises(ValueError):
        EMB.make_concat_embedder(RecordingAdapter(), min_s=-1.0)
    with pytest.raises(ValueError):
        EMB.make_concat_embedder(RecordingAdapter(), min_s=float("nan"))


# ---- registry and validation ----

def test_registry_lists_the_four_kinds_and_refuses_unknown_ones():
    assert EMB.KINDS == ("r34lm", "wespeaker_r293", "wespeaker_simam100", "espnet_spk")
    with pytest.raises(ValueError, match="unknown embedder kind"):
        EMB.make_embedder("nope")


def test_simam_needs_exactly_one_of_onnx_or_dir():
    with pytest.raises(ValueError):
        EMB.validate_args("wespeaker_simam100")
    with pytest.raises(ValueError):
        EMB.validate_args("wespeaker_simam100", simam_onnx="a.onnx", simam_dir="d")
    EMB.validate_args("wespeaker_simam100", simam_onnx="a.onnx")
    EMB.validate_args("r34lm")


def test_adapters_have_file_name_safe_ids():
    assert EMB.make_embedder("r34lm").id_string == "r34lm-wespeaker-voxceleb-resnet34-LM"
    assert EMB.make_embedder("wespeaker_r293", r293_onnx="/x/voxceleb_resnet293_LM.onnx").id_string == \
        "wespeaker_r293-voxceleb_resnet293_LM.onnx"
    assert EMB.make_embedder("wespeaker_r293").id_string == "wespeaker_r293-hub-default"
    assert EMB.make_embedder("espnet_spk").id_string == "espnet_spk-voxcelebs12_ska_wavlm_joint"
    assert EMB.make_embedder("wespeaker_simam100", simam_dir="/m/simam_v1").id_string == \
        "wespeaker_simam100-simam_v1"
    assert EMB._safe("a b/c") == "a_b_c"


def test_missing_onnx_file_raises_runtime_error_naming_the_path(tmp_path):
    ad = EMB.make_embedder("wespeaker_simam100", simam_onnx=str(tmp_path / "gone.onnx"))
    with pytest.raises(RuntimeError, match="gone.onnx"):
        ad.embed(np.zeros(16000, dtype=np.float32))


def test_missing_onnx_package_raises_runtime_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "models.wespeaker_embedding", None)    # import fails
    ad = EMB.make_embedder("wespeaker_r293")
    with pytest.raises(RuntimeError, match="not available here"):
        ad.load()


def test_missing_torch_for_r34lm_raises_runtime_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    ad = EMB.make_embedder("r34lm")
    with pytest.raises(RuntimeError, match="r34lm"):
        ad.embed(np.zeros(16000, dtype=np.float32))


def test_missing_espnet_package_raises_runtime_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "espnet2.bin.spk_inference", None)
    ad = EMB.make_embedder("espnet_spk")
    with pytest.raises(RuntimeError, match="espnet"):
        ad.embed(np.zeros(16000, dtype=np.float32))


def _fake_espnet(monkeypatch, from_pretrained):
    """espnet2.bin.spk_inference and torch, as far as EspnetAdapter touches them."""
    mod = types.ModuleType("espnet2.bin.spk_inference")
    mod.Speech2Embedding = types.SimpleNamespace(from_pretrained=staticmethod(from_pretrained))
    fake_torch = types.ModuleType("torch")
    fake_torch.as_tensor = lambda x, dtype=None: np.asarray(x)      # numpy stands in for the tensor
    fake_torch.float32 = np.float32
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "espnet2", types.ModuleType("espnet2"))
    monkeypatch.setitem(sys.modules, "espnet2.bin", types.ModuleType("espnet2.bin"))
    monkeypatch.setitem(sys.modules, "espnet2.bin.spk_inference", mod)


def test_espnet_bad_tag_raises_runtime_error_with_the_tag(monkeypatch):
    def refuse(model_tag):
        raise OSError("no such model")

    _fake_espnet(monkeypatch, refuse)
    ad = EMB.make_embedder("espnet_spk", espnet_tag="espnet/nope")
    with pytest.raises(RuntimeError, match="espnet/nope"):
        ad.load()


def test_espnet_adapter_returns_a_float64_vector_from_a_fake_model(monkeypatch):
    class FakeS2E:
        def __call__(self, wave):
            return np.arange(4, dtype=np.float32) + float(wave.shape[0]) * 0

    _fake_espnet(monkeypatch, lambda model_tag: FakeS2E())
    vec = EMB.make_embedder("espnet_spk").embed(np.zeros(16000))
    assert vec.dtype == np.float64 and vec.shape == (4,)
