"""Refinement on the API backend: the same fusion, nothing loaded locally."""
import json
import os
import sys
from concurrent.futures import Future

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("torch")

import services.diarization_refinement_service as refinement  # noqa: E402
from schemas.transcript import TranscriptSegment  # noqa: E402
from services.api_llm import ApiLLM  # noqa: E402

TEXT = "xin chào các bạn đến với chương trình hôm nay"


class FakeRemote:
    """The surface the refinement service reads from an API backend."""
    api_model = "fake"
    replica_count = 3

    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def ensure_loaded(self):
        return True

    def token_usage(self):
        return {"prompt_tokens": 10, "completion_tokens": 2}

    def generate_texts(self, system_prompt, messages, max_new_tokens, use_prefix, labels, thinking):
        self.calls.append((list(messages), max_new_tokens, use_prefix, labels))
        return (False, []) if self.fail else (True, [TEXT for _ in messages])


def _seg(index):
    return TranscriptSegment(
        index=index, start=float(index * 6), end=float(index * 6 + 5), speaker="SPEAKER_00",
        text=TEXT, text_whisper=TEXT, text_phowhisper=TEXT, text_qwen3=TEXT,
        language="vi", bs_roformer=False, bss=False)


def _svc(remote, **kw):
    svc = refinement.DiarizationRefinementService(logger=None, **kw)
    svc.use_remote(remote)
    return svc


@pytest.mark.parametrize("shared_queue", [False, True])
def test_refine_runs_through_the_remote_without_loading_anything_locally(shared_queue):
    remote = FakeRemote()
    svc = _svc(remote, batch_size=2, shared_queue=shared_queue, chunk_size=2)
    segments = [_seg(i) for i in range(5)]
    out = svc.refine(segments)
    assert [s.text for s in out] == [TEXT] * 5
    assert svc.model is None and svc.tokenizer is None and svc._worker is None
    assert sum(len(m) for m, *_ in remote.calls) == 5
    assert all(call[2] is False for call in remote.calls)       # the local prefix cache is never asked for


def test_the_remote_decides_whether_refinement_can_run():
    class Down(FakeRemote):
        def ensure_loaded(self):
            return False
    with pytest.raises(RuntimeError):
        _svc(Down()).refine([_seg(0)])


def test_most_failures_still_fail_the_file_instead_of_landing_unrefined():
    with pytest.raises(RuntimeError):
        _svc(FakeRemote(fail=True), batch_size=2).refine([_seg(i) for i in range(4)])


def test_submit_texts_returns_a_future_and_replica_count_and_usage_come_from_the_remote():
    remote = FakeRemote()
    svc = _svc(remote)
    future = svc.submit_texts("sys", ["a", "b"], labels=[1, 2])
    assert isinstance(future, Future) and future.result() == (True, [TEXT, TEXT])
    assert svc.replica_count == 3 and svc.token_usage() == {"prompt_tokens": 10, "completion_tokens": 2}


def test_without_a_remote_nothing_changes():
    svc = refinement.DiarizationRefinementService(logger=None)
    assert svc._remote() is None and svc.replica_count == 1 and svc.token_usage() == {}


def test_refinement_through_the_real_api_class_sends_each_segment_once():
    sent = []

    def transport(url, body, headers, timeout):
        sent.append(json.loads(body)["messages"][1]["content"])
        reply = {"choices": [{"message": {"content": TEXT}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        return 200, json.dumps(reply).encode(), {}

    os.environ["OPENAI_API_KEY"] = "sk-test"
    api = ApiLLM("gpt-x", checkpoint_model_name="Qwen/Qwen3.5-9B", transport=transport,
                 concurrency=4, sleep=lambda s: None)
    svc = _svc(api, batch_size=4, shared_queue=True, chunk_size=2)
    out = svc.refine([_seg(i) for i in range(6)])
    assert [s.text for s in out] == [TEXT] * 6
    assert len(sent) == 6
    assert api.token_usage() == {"prompt_tokens": 42, "completion_tokens": 18}


def test_the_completion_floor_keeps_room_for_the_answer_when_reasoning_is_on():
    api = ApiLLM("m", checkpoint_model_name="x", reasoning_effort="low")
    assert json.loads(api._body("s", "u", 200))["max_completion_tokens"] == 1024
    assert json.loads(ApiLLM("m", checkpoint_model_name="x")._body("s", "u", 200))["max_completion_tokens"] == 200
    assert json.loads(ApiLLM("m", checkpoint_model_name="x", reasoning_effort="low",
                             completion_floor=0)._body("s", "u", 200))["max_completion_tokens"] == 200
