import sys
import types

import pytest

from utils.llm_sampling import THINKING_SAMPLING, hf_generate_kwargs, vllm_sampling_kwargs


def test_thinking_calls_sample_with_the_recommended_settings():
    kwargs = vllm_sampling_kwargs(True, 8192)
    assert kwargs["temperature"] == 0.6 and kwargs["top_p"] == 0.95 and kwargs["top_k"] == 20
    assert kwargs["presence_penalty"] == 1.0 and kwargs["max_tokens"] == 8192
    assert kwargs["seed"] == THINKING_SAMPLING["seed"]
    assert kwargs["repetition_detection"] == {
        "max_pattern_size": 256, "min_pattern_size": 16, "min_count": 3}


def test_calls_without_thinking_stay_greedy():
    assert vllm_sampling_kwargs(False, 512) == {
        "temperature": 0.0, "max_tokens": 512, "repetition_penalty": 1.0}
    assert hf_generate_kwargs(False) == {
        "do_sample": False, "temperature": None, "top_p": None, "top_k": None,
        "repetition_penalty": 1.0}


def test_the_transformers_path_samples_when_thinking_and_has_no_presence_penalty():
    kwargs = hf_generate_kwargs(True)
    assert kwargs["do_sample"] is True and kwargs["temperature"] == 0.6
    assert kwargs["top_k"] == 20 and "presence_penalty" not in kwargs


class _Tokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return messages[-1]["content"]


class _Output:
    def __init__(self):
        self.prompt_token_ids = [1, 2]
        self.outputs = [types.SimpleNamespace(text="[]", token_ids=[3])]


class _Model:
    def generate(self, texts, sampling_params=None, use_tqdm=False):
        self.params = sampling_params
        return [_Output() for _ in texts]


@pytest.fixture
def worker(monkeypatch):
    """refinement_worker with vllm faked, so the arguments handed to it can be read."""
    fake = types.ModuleType("vllm")
    fake.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
    params = types.ModuleType("vllm.sampling_params")
    params.RepetitionDetectionParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
    fake.sampling_params = params
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", params)
    monkeypatch.setitem(sys.modules, "worker_vllm_env", types.ModuleType("worker_vllm_env"))
    sys.modules.pop("refinement_worker", None)
    import refinement_worker
    yield refinement_worker
    sys.modules.pop("refinement_worker", None)


def test_the_vllm_worker_hands_the_thinking_settings_to_the_engine(worker):
    model = _Model()
    worker.generate_with_usage(model, _Tokenizer(), "sys", ["a"], 8192, True, "vllm")
    assert model.params.temperature == 0.6 and model.params.presence_penalty == 1.0
    assert model.params.top_k == 20 and model.params.max_tokens == 8192


def test_the_vllm_worker_stays_greedy_without_thinking(worker):
    model = _Model()
    worker.generate_with_usage(model, _Tokenizer(), "sys", ["a"], 512, False, "vllm")
    assert model.params.temperature == 0.0 and not hasattr(model.params, "top_k")


def test_thinking_calls_ask_vllm_to_stop_a_repeating_reply(worker):
    model = _Model()
    worker.generate_with_usage(model, _Tokenizer(), "sys", ["a"], 8192, True, "vllm")
    detection = model.params.repetition_detection
    assert (detection.max_pattern_size, detection.min_pattern_size,
            detection.min_count) == (256, 16, 3)


def test_calls_without_thinking_do_not_detect_repetition(worker):
    model = _Model()
    worker.generate_with_usage(model, _Tokenizer(), "sys", ["a"], 512, False, "vllm")
    assert not hasattr(model.params, "repetition_detection")


def test_replies_stopped_by_repetition_are_counted(worker):
    class Stopped(_Output):
        def __init__(self):
            super().__init__()
            self.outputs[0].finish_reason = "repetition"
    model = _Model()
    model.generate = lambda texts, sampling_params=None, use_tqdm=False: [Stopped(), _Output()]
    _, usage = worker.generate_with_usage(
        model, _Tokenizer(), "sys", ["a", "b"], 8192, True, "vllm")
    assert usage["repetition_stops"] == 1


def test_an_older_vllm_without_the_parameter_still_generates(worker, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", None)   # import raises
    model = _Model()
    worker.generate_with_usage(model, _Tokenizer(), "sys", ["a"], 8192, True, "vllm")
    assert not hasattr(model.params, "repetition_detection")
