import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def phowhisper(monkeypatch):
    """models.phowhisper imports faster_whisper through models.whisper; stub that."""
    stub = types.ModuleType("models.whisper")
    stub.load_asr_model = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "models.whisper", stub)
    sys.modules.pop("models.phowhisper", None)
    module = importlib.import_module("models.phowhisper")
    globals()["PhoWhisperASR"] = module.PhoWhisperASR
    globals()["is_out_of_memory"] = module.is_out_of_memory
    yield module
    sys.modules.pop("models.phowhisper", None)


class FakeModel:
    """Transcribes a batch as long as it has at most `fits` clips, else runs out of memory."""

    def __init__(self, fits, message="CUDA failed with error out of memory"):
        self.fits = fits
        self.message = message
        self.batches = []

    def transcribe(self, carrier, spans, batch_size=None, language=None, print_progress=False):
        self.batches.append(len(spans))
        if len(spans) > self.fits:
            raise RuntimeError(self.message)
        return {"segments": [{"text": f"len{round((s['end'] - s['start']) * 16000)}"}
                             for s in spans]}


def _asr(model):
    asr = PhoWhisperASR.__new__(PhoWhisperASR)
    asr.model, asr.batch_size, asr.oom_retries = model, 16, 0
    return asr


def _clips(n):
    return [np.zeros(100 + k, dtype=np.float32) for k in range(n)]


class Log:
    def __init__(self):
        self.lines = []

    def warning(self, message):
        self.lines.append(message)


def test_a_batch_that_does_not_fit_is_run_again_in_halves_and_keeps_its_order():
    model = FakeModel(fits=3)
    asr = _asr(model)
    log = Log()
    texts = asr.transcribe_batch(_clips(8), batch_size=8, logger=log)
    assert texts == [f"len{100 + k}" for k in range(8)]
    assert model.batches == [8, 4, 2, 2, 4, 2, 2]          # 8 -> 4+4, each 4 -> 2+2
    assert asr.oom_retries == 3
    assert len(log.lines) == 3 and all("out of memory" in line for line in log.lines)


def test_a_batch_that_fits_is_run_once():
    model = FakeModel(fits=16)
    asr = _asr(model)
    assert len(asr.transcribe_batch(_clips(5), batch_size=16)) == 5
    assert model.batches == [5] and asr.oom_retries == 0


def test_the_progress_callback_fires_once_per_clip_even_after_a_retry():
    ticks = []
    asr = _asr(FakeModel(fits=2))
    asr.transcribe_batch(_clips(6), batch_size=6, callback=lambda: ticks.append(1))
    assert len(ticks) == 6


def test_a_single_clip_that_does_not_fit_is_raised():
    with pytest.raises(RuntimeError, match="out of memory"):
        _asr(FakeModel(fits=0)).transcribe_batch(_clips(1), batch_size=1)


def test_an_error_that_is_not_out_of_memory_is_not_retried():
    model = FakeModel(fits=0, message="something else broke")
    with pytest.raises(RuntimeError, match="something else"):
        _asr(model).transcribe_batch(_clips(4), batch_size=4)
    assert model.batches == [4]


def test_the_ways_a_gpu_reports_running_out_of_memory_are_recognised():
    assert is_out_of_memory(RuntimeError("CUDA failed with error out of memory"))
    assert is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert is_out_of_memory(RuntimeError("cudaErrorMemoryAllocation"))
    assert not is_out_of_memory(RuntimeError("shape mismatch"))
