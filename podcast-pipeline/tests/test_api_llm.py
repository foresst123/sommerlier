import json
import types

import pytest

from services.api_llm import (ApiError, ApiLLM, STAGES, build_pass_llm, resolve_backend,
                              resolve_stages, stage_llm)
from services.transcript_review_service import TranscriptReviewService
from utils.llm_batches import ask_in_batches

LOCAL_NAME = "Qwen/Qwen3.5-9B"


def _ok(text="ok", prompt=11, completion=3):
    body = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}
    return 200, json.dumps(body).encode(), {}


class FakeTransport:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, body, headers, timeout):
        self.calls.append((url, json.loads(body), headers))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")


def _llm(transport, **kw):
    return ApiLLM("gpt-x", checkpoint_model_name=LOCAL_NAME, transport=transport,
                  sleep=lambda s: None, **kw)


def test_checkpoint_name_is_the_local_model_so_finished_work_is_reused():
    llm = _llm(FakeTransport([_ok()]))
    assert llm.model_name == LOCAL_NAME and llm.api_model == "gpt-x"
    local = types.SimpleNamespace(model_name=LOCAL_NAME, batch_size=4, max_batch_tokens=0)
    kw = dict(window_tokens=3500, thinking=False)
    assert (TranscriptReviewService(llm, **kw).checkpoint_namespace
            == TranscriptReviewService(local, **kw).checkpoint_namespace)


def test_the_surface_the_passes_read_is_present_and_one_message_per_call():
    llm = _llm(FakeTransport([_ok()]), concurrency=5)
    assert llm.ensure_loaded() is True
    assert llm.count_tokens("x" * 70) == 20 and llm.count_tokens("") == 1
    assert (llm.batch_size, llm.max_batch_tokens, llm.parallel_windows, llm.replica_count) == (1, 0, True, 5)


def test_generate_texts_keeps_order_and_sends_the_expected_request():
    transport = FakeTransport([_ok("a"), _ok("b"), _ok("c")])
    llm = _llm(transport, concurrency=1, reasoning_effort="low")
    ok, texts = llm.generate_texts("SYS", ["m1", "m2", "m3"], max_new_tokens=300, thinking=True)
    assert ok and texts == ["a", "b", "c"]
    url, body, headers = transport.calls[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert body["model"] == "gpt-x" and body["reasoning_effort"] == "low"
    assert body["max_completion_tokens"] == 1024        # reasoning on: room for the thinking and the answer
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "m1"}]
    assert headers["Authorization"] == "Bearer sk-test-secret"


def test_reasoning_effort_is_only_sent_when_configured():
    transport = FakeTransport([_ok()])
    _llm(transport).generate_texts("S", ["m"], max_new_tokens=300)
    assert "reasoning_effort" not in transport.calls[0][1]
    assert transport.calls[0][1]["max_completion_tokens"] == 300      # no floor without reasoning


def test_concurrent_calls_return_in_message_order():
    transport = FakeTransport([_ok("x")])
    llm = _llm(transport, concurrency=4)
    ok, texts = llm.generate_texts("S", [f"m{i}" for i in range(8)])
    assert ok and len(texts) == 8 and len(transport.calls) == 8


def test_ask_in_batches_runs_through_the_api_with_one_message_per_call():
    transport = FakeTransport([_ok("r")])
    llm = _llm(transport, concurrency=3)
    replies, unanswered = ask_in_batches(llm, "S", ["a", "b", "c", "d"], per_call=1,
                                         max_new_tokens=50, label="t")
    assert replies == ["r"] * 4 and unanswered == 0


def test_retries_then_succeeds_and_never_leaks_the_key():
    transport = FakeTransport([(429, b'{"error": {"message": "slow down"}}', {"Retry-After": "0"}),
                               TimeoutError("t"), _ok("fine")])
    llm = _llm(transport, max_retries=3)
    assert llm.generate_texts("S", ["m"]) == (True, ["fine"])
    assert len(transport.calls) == 3


def test_a_request_that_cannot_succeed_is_reported_not_raised_and_hides_the_key():
    transport = FakeTransport([(401, b'{"error": {"message": "bad key sk-test-secret"}}', {})])
    seen = []
    logger = types.SimpleNamespace(warning=seen.append, info=lambda *a, **k: None)
    llm = _llm(transport, logger=logger)
    assert llm.generate_texts("S", ["m"]) == (False, [])
    assert len(transport.calls) == 1                          # 401 is not retried
    assert llm.stats["failures"] == 1
    assert seen and "sk-test-secret" not in " ".join(seen) and "***" in " ".join(seen)


def test_gives_up_after_the_configured_retries():
    transport = FakeTransport([(503, b"{}", {})])
    llm = _llm(transport, max_retries=2)
    assert llm.generate_texts("S", ["m"]) == (False, [])
    assert len(transport.calls) == 3


def test_an_empty_reply_counts_as_a_failure_so_the_pass_treats_it_as_no_change():
    llm = _llm(FakeTransport([_ok("   ")]))
    assert llm.generate_texts("S", ["m"]) == (False, [])


def test_usage_is_counted():
    llm = _llm(FakeTransport([_ok("x", prompt=100, completion=20)]))
    llm.generate_texts("S", ["a", "b"])
    assert llm.stats["prompt_tokens"] == 200 and llm.stats["completion_tokens"] == 40


def test_ensure_loaded_is_false_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    assert _llm(FakeTransport([_ok()])).ensure_loaded() is False


def _args(**kw):
    return types.SimpleNamespace(**{"llm_backend": None, "api_model": None, **kw})


def test_the_default_backend_is_the_local_model_itself():
    local = object()
    assert build_pass_llm(_args(), {}, local) is local
    assert resolve_backend(_args(), {"models": {"llm_backend": "local"}}) == "local"


def test_flag_beats_profile_and_unknown_values_raise():
    assert resolve_backend(_args(llm_backend="api"), {"models": {"llm_backend": "local"}}) == "api"
    with pytest.raises(ValueError):
        resolve_backend(_args(), {"models": {"llm_backend": "gemini"}})


def test_api_backend_builds_from_the_profile_and_keeps_the_local_checkpoint_name():
    local = types.SimpleNamespace(model_name=LOCAL_NAME)
    profile = {"models": {"api_llm": {"model": "gpt-x", "concurrency": 2}}}
    llm = build_pass_llm(_args(llm_backend="api"), profile, local)
    assert isinstance(llm, ApiLLM) and llm.model_name == LOCAL_NAME and llm.replica_count == 2
    assert build_pass_llm(_args(llm_backend="api", api_model="other"), profile, local).api_model == "other"


def test_api_backend_fails_at_start_up_not_per_file():
    local = types.SimpleNamespace(model_name=LOCAL_NAME)
    with pytest.raises(ValueError):                           # no model id anywhere
        build_pass_llm(_args(llm_backend="api"), {}, local)
    with pytest.raises(ValueError):                           # a typo is an error
        build_pass_llm(_args(llm_backend="api"), {"models": {"api_llm": {"model": "m", "modell": 1}}}, local)


def test_a_missing_key_stops_the_run_before_any_file(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    local = types.SimpleNamespace(model_name=LOCAL_NAME)
    with pytest.raises(RuntimeError):
        build_pass_llm(_args(llm_backend="api", api_model="m"), {}, local)


def test_conversation_selection_checkpoint_name_is_the_same_for_api_and_local():
    from services.conversation_export_service import ConversationExportService
    api = _llm(FakeTransport([_ok()]))
    local = types.SimpleNamespace(model_name=LOCAL_NAME, batch_size=4, max_batch_tokens=0)
    assert (ConversationExportService(api).checkpoint_namespace
            == ConversationExportService(local).checkpoint_namespace)


def test_in_flight_requests_never_exceed_the_configured_concurrency():
    import threading
    import time
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def transport(url, body, headers, timeout):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return _ok("x")

    llm = _llm(transport, concurrency=3)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=12) as pool:           # several callers at once, as refinement queues chunks
        results = list(pool.map(lambda i: llm.generate_texts("S", ["a", "b", "c", "d"]), range(6)))
    assert all(ok for ok, _ in results) and state["peak"] <= 3


def test_cached_prompt_tokens_are_counted_separately():
    body = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 30,
                      "prompt_tokens_details": {"cached_tokens": 1536}}}
    llm = _llm(FakeTransport([(200, json.dumps(body).encode(), {})]))
    llm.generate_texts("S", ["a", "b"])
    assert llm.stats["prompt_tokens"] == 4000 and llm.stats["cached_tokens"] == 3072


def test_every_stage_calls_the_api_unless_narrowed():
    assert resolve_stages(_args(), {}) == frozenset(STAGES)
    assert resolve_stages(_args(api_stages="review, selection"), {}) == {"review", "selection"}
    assert resolve_stages(_args(), {"stages": ["review"]}) == {"review"}
    assert resolve_stages(_args(api_stages="selection"), {"stages": ["review"]}) == {"selection"}   # flag wins
    with pytest.raises(ValueError):
        resolve_stages(_args(api_stages="review,summary"), {})


def test_only_the_chosen_stages_get_the_api_and_the_rest_keep_the_local_model():
    local = types.SimpleNamespace(model_name=LOCAL_NAME)
    profile = {"models": {"api_llm": {"model": "gpt-x"}}}
    llm = build_pass_llm(_args(llm_backend="api", api_stages="review,selection"), profile, local)
    assert stage_llm(llm, local, "review") is llm and stage_llm(llm, local, "selection") is llm
    assert stage_llm(llm, local, "refinement") is local and stage_llm(llm, local, "relabel") is local
    everything = build_pass_llm(_args(llm_backend="api"), profile, local)
    assert all(stage_llm(everything, local, name) is everything for name in STAGES)
    assert stage_llm(local, local, "review") is local             # local backend: nothing changes


def _cache_key_of(call):
    return json.loads(json.dumps(call[1])).get("prompt_cache_key")


def test_the_same_system_prompt_gets_the_same_cache_key_and_another_prompt_another():
    transport = FakeTransport([_ok()])
    llm = _llm(transport)
    llm.generate_texts("PROMPT A", ["m1"])
    llm.generate_texts("PROMPT A", ["m2"])
    llm.generate_texts("PROMPT B", ["m3"])
    keys = [_cache_key_of(c) for c in transport.calls]
    assert keys[0] == keys[1] and keys[0] != keys[2] and keys[0].startswith("sommelier-")


def test_a_busy_prefix_is_spread_over_more_keys_to_stay_under_the_request_rate():
    transport = FakeTransport([_ok()])
    now = {"t": 0.0}
    llm = _llm(transport, clock=lambda: now["t"], concurrency=1)
    for i in range(60):                                       # 60 requests inside one minute
        now["t"] = i * 0.5
        llm.generate_texts("SAME", ["m"])
    keys = {_cache_key_of(c) for c in transport.calls}
    assert 3 <= len(keys) <= 32                               # about rpm / 12, never one key for all
    quiet = _llm(FakeTransport([_ok()]), clock=lambda: 0.0, concurrency=1)
    quiet.generate_texts("SAME", ["m"])


def test_a_slow_prefix_keeps_one_key_so_its_cache_is_reused():
    transport = FakeTransport([_ok()])
    now = {"t": 0.0}
    llm = _llm(transport, clock=lambda: now["t"])
    for i in range(5):
        now["t"] = i * 30.0                                   # one request every half minute
        llm.generate_texts("SAME", ["m"])
    assert len({_cache_key_of(c) for c in transport.calls}) <= 2


def test_caching_fields_are_not_sent_to_another_endpoint_unless_asked():
    transport = FakeTransport([_ok()])
    other = _llm(transport, base_url="https://gateway.example/v1")
    other.generate_texts("S", ["m"])
    assert "prompt_cache_key" not in transport.calls[0][1]
    forced = _llm(FakeTransport([_ok()]), base_url="https://gateway.example/v1", prompt_cache=True)
    forced.generate_texts("S", ["m"])
    assert forced._transport.calls[0][1]["prompt_cache_key"].startswith("sommelier-")


def test_retention_is_sent_only_when_configured():
    plain = FakeTransport([_ok()])
    _llm(plain).generate_texts("S", ["m"])
    assert "prompt_cache_retention" not in plain.calls[0][1]
    kept = FakeTransport([_ok()])
    _llm(kept, prompt_cache_retention="24h").generate_texts("S", ["m"])
    assert kept.calls[0][1]["prompt_cache_retention"] == "24h"


def test_a_server_that_rejects_the_caching_fields_gets_the_request_again_without_them():
    transport = FakeTransport([(400, b'{"error": {"message": "Unknown parameter: prompt_cache_key"}}', {}), _ok("fine")])
    llm = _llm(transport, max_retries=0)
    assert llm.generate_texts("S", ["m"]) == (True, ["fine"])
    assert "prompt_cache_key" in transport.calls[0][1] and "prompt_cache_key" not in transport.calls[1][1]
    llm.generate_texts("S", ["again"])
    assert "prompt_cache_key" not in transport.calls[2][1]    # and it does not try them again
