"""A thinking budget for the vLLM worker: cap the thinking, then let the model answer."""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("torch")

import refinement_worker as worker  # noqa: E402
from utils.llm_sampling import THINKING_CLOSE, thinking_continuation  # noqa: E402


class _Params:
    def __init__(self, **kw):
        self.kw = kw


class _Out:
    def __init__(self, text, tokens, reason, prompt=7):
        self.prompt_token_ids = list(range(prompt))
        self.outputs = [types.SimpleNamespace(text=text, token_ids=list(range(tokens)),
                                              finish_reason=reason)]


class _Model:
    """generate() answers from a script: one list of outputs per call."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        self.calls.append((list(prompts), sampling_params.kw["max_tokens"]))
        return self.script.pop(0)


@pytest.fixture(autouse=True)
def _reset():
    worker.set_thinking_budget(0)
    yield
    worker.set_thinking_budget(0)


def test_continuation_rules():
    assert thinking_continuation("done </think>{}", "stop") is None
    assert thinking_continuation("still thinking", "length") == THINKING_CLOSE
    assert thinking_continuation("x</think>{\"a\"", "length") == ""
    assert thinking_continuation("loop", "repetition") is None
    assert thinking_continuation(None, "length") == THINKING_CLOSE


def test_without_a_budget_it_is_one_plain_call():
    model = _Model([[_Out("a</think>b", 30, "stop")]])
    replies, tokens, reasons, prompts = worker._vllm_generate(model, _Params, ["P"], True, 16384)
    assert replies == ["a</think>b"] and tokens == [30] and len(model.calls) == 1
    assert model.calls[0][1] == 16384


def test_a_reply_cut_in_its_thinking_is_closed_and_continued():
    worker.set_thinking_budget(1000)
    model = _Model([[_Out("long thought", 1000, "length")], [_Out("[{\"keep\": true}]", 40, "stop")]])
    replies, tokens, reasons, _ = worker._vllm_generate(model, _Params, ["PROMPT "], True, 16384)
    assert model.calls[0][1] == 1000
    assert model.calls[1][0] == ["PROMPT long thought" + THINKING_CLOSE]
    assert model.calls[1][1] == 16384 - 1000
    assert replies == ["long thought" + THINKING_CLOSE + "[{\"keep\": true}]"]
    assert tokens == [1040] and reasons == ["stop"]


def test_a_reply_cut_while_answering_just_goes_on():
    worker.set_thinking_budget(1000)
    model = _Model([[_Out("t</think>[{\"a\": 1,", 1000, "length")], [_Out(" \"b\": 2}]", 12, "stop")]])
    replies, *_ = worker._vllm_generate(model, _Params, ["P"], True, 4000)
    assert model.calls[1][0] == ["Pt</think>[{\"a\": 1,"]          # nothing inserted
    assert replies == ["t</think>[{\"a\": 1, \"b\": 2}]"]


def test_replies_that_finished_are_left_alone_and_only_cut_ones_are_sent_again():
    worker.set_thinking_budget(500)
    model = _Model([[_Out("x</think>ok", 80, "stop"), _Out("thinking", 500, "length")],
                    [_Out("answer", 20, "stop")]])
    replies, tokens, *_ = worker._vllm_generate(model, _Params, ["A", "B"], True, 3000)
    assert len(model.calls[1][0]) == 1                           # only the second reply continued
    assert replies[0] == "x</think>ok" and tokens[0] == 80
    assert replies[1].endswith("answer")


def test_the_answer_always_gets_some_room_even_with_a_budget_near_the_cap():
    worker.set_thinking_budget(3900)
    model = _Model([[_Out("t", 3900, "length")], [_Out("a", 5, "stop")]])
    worker._vllm_generate(model, _Params, ["P"], True, 4000)
    assert model.calls[1][1] == 512


def test_a_request_without_thinking_ignores_the_budget():
    worker.set_thinking_budget(500)
    model = _Model([[_Out("answer", 400, "length")]])
    worker._vllm_generate(model, _Params, ["P"], False, 2000)
    assert len(model.calls) == 1 and model.calls[0][1] == 2000


def test_a_budget_not_below_the_cap_changes_nothing():
    worker.set_thinking_budget(20000)
    model = _Model([[_Out("a", 100, "stop")]])
    worker._vllm_generate(model, _Params, ["P"], True, 16384)
    assert len(model.calls) == 1 and model.calls[0][1] == 16384


def test_the_budget_reaches_the_worker_command_line_only_when_set():
    from services.refinement_worker_service import RefinementWorkerService
    base = dict(python_env_path="/x/python", worker_script_path="/x/w.py", model_name="m", backend="vllm")
    assert "--thinking-budget" not in RefinementWorkerService(**base)._extra_args if hasattr(
        RefinementWorkerService(**base), "_extra_args") else True
    with_budget = RefinementWorkerService(thinking_budget=3000, **base)
    args = getattr(with_budget, "_extra_args", None) or getattr(with_budget, "extra_args", None)
    assert args and args[args.index("--thinking-budget") + 1] == "3000"
