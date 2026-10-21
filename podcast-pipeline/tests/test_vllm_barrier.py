"""Whatever used vLLM must be gone, and its VRAM back, before the next step starts. The
refinement engine holds 95% of each card, so a step that follows it -- word alignment was
the one that failed, every segment out of memory -- has nothing to run in until the engine
process is dead and the card is free. Fakes only, no GPU."""

import logging
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.gpu_memory as gpu_memory
from services.diarization_refinement_service import DiarizationRefinementService
from services.pipeline_service import PipelineService


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _logger():
    logger = logging.getLogger(f"barrier-{id(object())}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = Capture()
    logger.addHandler(handler)
    return logger, handler


class FakeWorker:
    def __init__(self, events):
        self.events = events

    def stop(self):
        self.events.append("worker stopped")


def _service(events, devices=(0, 1)):
    logger, handler = _logger()
    svc = DiarizationRefinementService.__new__(DiarizationRefinementService)
    svc.logger = logger
    svc._worker = FakeWorker(events)
    svc._worker_devices = list(devices)
    svc.model = None
    svc.tokenizer = None
    return svc, handler


def test_unloading_the_engine_returns_only_once_the_cards_are_free(monkeypatch):
    events = []
    calls = []

    def fake_wait(devices, fraction, **kwargs):
        calls.append((list(devices), fraction))
        events.append("vram checked")
        return True

    monkeypatch.setattr(gpu_memory, "wait_for_free_vram", fake_wait)
    svc, handler = _service(events)

    svc.unload()

    assert events == ["worker stopped", "vram checked"]
    assert calls == [([0, 1], 0.90)]
    assert svc._worker is None
    assert any("released" in message for message in handler.messages)


def test_a_card_that_never_empties_is_reported_not_hidden(monkeypatch):
    monkeypatch.setattr(gpu_memory, "wait_for_free_vram", lambda *a, **k: False)
    svc, handler = _service([])

    svc.unload()

    assert any("still" in message and "not free" in message for message in handler.messages)


def test_unloading_with_nothing_resident_does_nothing(monkeypatch):
    monkeypatch.setattr(gpu_memory, "wait_for_free_vram",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no wait")))
    svc, _ = _service([])
    svc._worker = None
    svc.unload()                      # must not raise, must not wait for anything


def test_the_service_says_whether_an_engine_is_resident():
    svc, _ = _service([])
    assert svc.is_resident() is True
    svc._worker = None
    assert svc.is_resident() is False


class Recorder:
    def __init__(self, name, events, resident=True):
        self.name, self.events, self.resident = name, events, resident

    def is_resident(self):
        return self.resident

    def unload(self):
        self.events.append(f"{self.name} unloaded")
        self.resident = False


def _pipeline(events, resident=True):
    pipe = PipelineService.__new__(PipelineService)
    pipe.logger, _ = _logger()
    pipe.refinement_svc = Recorder("llm", events, resident)
    return pipe


def test_a_step_that_follows_the_llm_releases_it_first_whatever_the_flags_say():
    events = []
    pipe = _pipeline(events)
    pipe._release_llm_before(types.SimpleNamespace(keep_models=False), "word alignment")
    assert events == ["llm unloaded"]


def test_keep_models_leaves_the_engine_alone():
    events = []
    pipe = _pipeline(events)
    pipe._release_llm_before(types.SimpleNamespace(keep_models=True), "word alignment")
    assert events == []


def test_nothing_is_unloaded_when_no_engine_is_resident():
    events = []
    pipe = _pipeline(events, resident=False)
    pipe._release_llm_before(types.SimpleNamespace(keep_models=False), "word alignment")
    assert events == []


def test_word_alignment_is_only_reached_through_that_release():
    source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "services", "pipeline_service.py"), encoding="utf-8").read()
    at = source.index("transcripts = self._align_words(")
    before = source[:at]
    assert before.rindex("self._release_llm_before(") > before.rindex("alignment_cached =")
