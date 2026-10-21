"""One ASR model per run: nothing else loads, and there is no ROVER vote."""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.asr_service import ASRService
from utils.asr_models import ALL_ASR_MODELS, enabled_asr_models


def test_no_setting_keeps_all_three_models():
    assert enabled_asr_models({}) == ALL_ASR_MODELS
    assert enabled_asr_models({"pipeline": {}}) == ALL_ASR_MODELS


def test_profile_picks_models_in_stable_order():
    assert enabled_asr_models({"pipeline": {"asr_models": ["whisper", "qwen3"]}}) == (
        "qwen3", "whisper")
    assert enabled_asr_models({"pipeline": {"asr_models": "qwen3"}}) == ("qwen3",)


@pytest.mark.parametrize("bad", [[], ["qwen"], ["qwen3", "canary"]])
def test_unknown_or_empty_selection_is_refused(bad):
    with pytest.raises(ValueError, match="asr_models"):
        enabled_asr_models({"pipeline": {"asr_models": bad}})


def _seg(index=1):
    return SimpleNamespace(index=index, start=0.0, end=3.0, speaker="S1", bs_roformer=False,
                           bss=False, has_music=False, bss_failed_spans=[])


def test_single_model_text_is_the_answer_without_rover(monkeypatch):
    class Boom:
        def __init__(self):
            raise AssertionError("ROVER must not be built for a single model")

    monkeypatch.setattr("services.asr_service.RoverEnsembler", Boom)
    svc = ASRService(asr_models=("qwen3",), edge_pad=0.0)
    results, _dropped = svc._vote(
        [_seg()], [("whisper text", "vi", [])], ["pho text"], ["xin chào các bạn"], [0.0])
    assert results[0].text == "xin chào các bạn"


def test_several_models_still_vote(monkeypatch):
    calls = []

    class Rover:
        def align_and_vote(self, order):
            calls.append(order)
            return "voted"

    monkeypatch.setattr("services.asr_service.RoverEnsembler", Rover)
    svc = ASRService(edge_pad=0.0)
    results, _ = svc._vote(
        [_seg()], [("một hai ba bốn", "vi", [])], ["một hai ba năm"], ["một hai ba sáu"], [0.0])
    assert results[0].text == "voted" and len(calls) == 1


def test_missing_model_is_only_warned_about_when_it_was_asked_for():
    messages = []
    logger = SimpleNamespace(warning=messages.append, info=lambda *_: None)
    ASRService(asr_models=("qwen3",), logger=logger, edge_pad=0.0)._warn_missing()
    assert not any("Whisper" in m for m in messages)
    assert any("Qwen3-ASR is not loaded" in m for m in messages)


def test_loader_builds_only_the_selected_model(monkeypatch):
    ml = pytest.importorskip("services.model_loader", reason="needs the ASR stack installed")

    loader = object.__new__(ml.ModelLoader)
    loader.models, loader.logger = {}, None
    loader.args = SimpleNamespace(env="a100", ASRMoE=True, lang="vi", stop_after=None)
    loader.config = {"environments": {"a100": {"pipeline": {"asr_models": ["qwen3"]}}}}
    monkeypatch.setattr(ml, "Qwen3ASRClient", lambda process: ("qwen3-client", process))
    loader.make_phowhisper = lambda *a, **k: pytest.fail("PhoWhisper must not load")
    loader.make_whisper_ct2 = lambda *a, **k: pytest.fail("Whisper must not load")

    loader.load_asr_models(SimpleNamespace(process="p"), whisper_service=None,
                           phowhisper_service=None)
    assert list(loader.models) == ["qwen3"]
    loader.load_asr_models(SimpleNamespace(process="p"))     # second call: already loaded
    assert list(loader.models) == ["qwen3"]
