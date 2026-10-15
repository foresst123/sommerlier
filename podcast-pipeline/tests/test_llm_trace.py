"""Every Qwen call is written to <output_dir>/llm_trace/<stage>.jsonl."""

import json
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import llm_trace, profiling
from services.refinement_worker_service import (
    RefinementWorkerPoolService, RefinementWorkerService)


def _lines(out, stage):
    path = os.path.join(out, "llm_trace", f"{stage}.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _record(trace, messages=("a", "b"), **kw):
    args = dict(system_prompt="SYS", user_messages=list(messages),
                outputs=[m.upper() for m in messages], details=None, ok=True,
                error=None, thinking=False, max_new_tokens=64, duration_s=0.5,
                backend="worker")
    args.update(kw)
    llm_trace.record_call(trace, **args)


# --- module ------------------------------------------------------------------------

def test_one_line_per_message_with_full_fields(tmp_path):
    out = str(tmp_path / "ep1")
    with llm_trace.trace_to("transcript_review", out):
        _record(llm_trace.current(), replica=1, call_id="abc",
                details=[{"finish_reason": "stop", "prompt_tokens": 10,
                          "completion_tokens": 3}, None])
    rows = _lines(out, "transcript_review")
    assert [r["input"] for r in rows] == ["a", "b"]
    assert [r["output"] for r in rows] == ["A", "B"]
    first = rows[0]
    assert (first["stage"], first["file"], first["call_id"], first["index"],
            first["n_messages"], first["backend"], first["replica"]) == (
        "transcript_review", "ep1", "abc", 0, 2, "worker", 1)
    assert (first["finish_reason"], first["prompt_tokens"], first["completion_tokens"]) == (
        "stop", 10, 3)
    assert rows[1]["finish_reason"] is None and rows[1]["completion_tokens"] is None
    assert first["run_id"] == llm_trace.RUN_ID and first["ok"] is True
    assert first["max_new_tokens"] == 64 and first["thinking"] is False


def test_system_prompt_stored_once_by_hash(tmp_path):
    out = str(tmp_path)
    with llm_trace.trace_to("refinement", out):
        _record(llm_trace.current())
        _record(llm_trace.current())
    rows = _lines(out, "refinement")
    sha = rows[0]["system_prompt_sha1"]
    assert {r["system_prompt_sha1"] for r in rows} == {sha}
    with open(os.path.join(out, "llm_trace", "refinement.system_prompts.json")) as handle:
        assert json.load(handle) == {sha: "SYS"}
    assert "SYS" not in open(os.path.join(out, "llm_trace", "refinement.jsonl")).read()


def test_retry_appends_instead_of_overwriting(tmp_path):
    out = str(tmp_path)
    with llm_trace.trace_to("refinement", out):
        _record(llm_trace.current(), messages=["a"], ok=False, outputs=None, error="OOM")
        _record(llm_trace.current(), messages=["a"])
    rows = _lines(out, "refinement")
    assert [(r["ok"], r["output"], r["error"]) for r in rows] == [
        (False, None, "OOM"), (True, "A", None)]


def test_non_ascii_kept_and_none_trace_is_a_noop(tmp_path):
    _record(None)
    with llm_trace.trace_to("refinement", str(tmp_path)):
        _record(llm_trace.current(), messages=["xin chào"], outputs=["tiếng Việt"])
    raw = open(tmp_path / "llm_trace" / "refinement.jsonl", encoding="utf-8").read()
    assert "tiếng Việt" in raw


def test_disabled_trace_has_no_current_and_writes_nothing(tmp_path):
    with llm_trace.trace_to("refinement", str(tmp_path), enabled=False):
        assert llm_trace.current() is None
        _record(llm_trace.current())
    assert not (tmp_path / "llm_trace").exists()


def test_io_error_is_swallowed(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    # output_dir is a file, so llm_trace/ cannot be created
    with llm_trace.trace_to("refinement", str(blocker)):
        _record(llm_trace.current())


def test_context_restored_after_block(tmp_path):
    assert llm_trace.current() is None
    with llm_trace.trace_to("a", str(tmp_path)):
        with llm_trace.trace_to("b", str(tmp_path)):
            assert llm_trace.current().stage == "b"
        assert llm_trace.current().stage == "a"
    assert llm_trace.current() is None


def test_bind_carries_trace_to_another_thread_and_restores(tmp_path):
    seen = {}

    def work():
        seen["inside"] = llm_trace.current()

    with llm_trace.trace_to("persona_profile", str(tmp_path)):
        bound = llm_trace.bind(work)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(bound).result()
        seen["after"] = pool.submit(llm_trace.current).result()
    assert seen["inside"].stage == "persona_profile"
    assert seen["after"] is None


def test_profiling_bind_also_carries_trace(tmp_path):
    seen = []
    with llm_trace.trace_to("speaker_relabel", str(tmp_path)):
        bound = profiling.bind(lambda: seen.append(llm_trace.current()))
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(bound).result()
    assert seen[0].stage == "speaker_relabel"


def test_many_threads_write_whole_lines(tmp_path):
    out = str(tmp_path)

    def work(i):
        with llm_trace.trace_to("refinement", out):
            for j in range(20):
                _record(llm_trace.current(), messages=[f"{i}-{j}"])

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(_lines(out, "refinement")) == 160


# --- worker service ---------------------------------------------------------------------

class _Pipe:
    def __init__(self, reply):
        self.reply = reply

    def write(self, text):
        pass

    def flush(self):
        pass

    def readline(self):
        return json.dumps(self.reply)


def _service(reply):
    service = RefinementWorkerService.__new__(RefinementWorkerService)
    service._usage_lock = threading.Lock()
    service._usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "repetition_stops": 0}
    service._lock = threading.Lock()
    service.logger = None
    service.model_name = "fake"
    pipe = _Pipe(reply)
    service.process = types.SimpleNamespace(stdin=pipe, stdout=pipe, poll=lambda: None)
    return service


def test_worker_service_records_texts_and_details(tmp_path):
    out = str(tmp_path / "ep")
    details = [{"finish_reason": "stop", "prompt_tokens": 5, "completion_tokens": 2},
               {"finish_reason": "length", "prompt_tokens": 6, "completion_tokens": 9}]
    service = _service({"ok": True, "texts": ["o1", "o2"], "details": details,
                        "usage": {"prompt_tokens": 11, "completion_tokens": 11}})
    with llm_trace.trace_to("transcript_review", out):
        assert service.generate_texts("S", ["i1", "i2"], 128, True) == (True, ["o1", "o2"])
    rows = _lines(out, "transcript_review")
    assert [r["output"] for r in rows] == ["o1", "o2"]
    assert [r["finish_reason"] for r in rows] == ["stop", "length"]
    assert rows[0]["backend"] == "worker" and rows[0]["thinking"] is True
    assert rows[0]["max_new_tokens"] == 128 and rows[0]["duration_s"] >= 0


def test_worker_service_without_details_still_works(tmp_path):
    out = str(tmp_path)
    service = _service({"ok": True, "texts": ["o1"]})
    with llm_trace.trace_to("refinement", out):
        assert service.generate_texts("S", ["i1"]) == (True, ["o1"])
    row = _lines(out, "refinement")[0]
    assert row["output"] == "o1" and row["finish_reason"] is None


def test_worker_service_records_failure(tmp_path):
    out = str(tmp_path)
    service = _service({"ok": False, "error": "OOM: boom"})
    with llm_trace.trace_to("refinement", out):
        assert service.generate_texts("S", ["i1"]) == (False, [])
    row = _lines(out, "refinement")[0]
    assert (row["ok"], row["output"], row["error"]) == (False, None, "OOM: boom")


def test_worker_service_without_trace_writes_nothing(tmp_path):
    service = _service({"ok": True, "texts": ["o1"]})
    assert service.generate_texts("S", ["i1"])[0] is True
    assert not list(tmp_path.iterdir())


def test_pool_traces_the_calling_files_not_the_replica_thread(tmp_path):
    pool = RefinementWorkerPoolService([_service({"ok": True, "texts": ["x"]}),
                                        _service({"ok": True, "texts": ["y"]})])
    pool.start = lambda: None
    first, second = str(tmp_path / "one"), str(tmp_path / "two")
    try:
        with llm_trace.trace_to("refinement", first):
            future_a = pool.submit("S", ["m1"])
        with llm_trace.trace_to("refinement", second):
            future_b = pool.submit("S", ["m2"])
        assert future_a.result()[0] and future_b.result()[0]
    finally:
        pool._stop_threads()
    assert [r["input"] for r in _lines(first, "refinement")] == ["m1"]
    assert [r["input"] for r in _lines(second, "refinement")] == ["m2"]
    assert _lines(first, "refinement")[0]["replica"] in (0, 1)


def test_pool_generate_texts_records_every_message_in_callers_file(tmp_path):
    pool = RefinementWorkerPoolService([_service({"ok": True, "texts": ["x"]}),
                                        _service({"ok": True, "texts": ["y"]})])
    out = str(tmp_path)
    try:
        with llm_trace.trace_to("refinement", out):
            assert pool.generate_texts("S", ["m1", "m2"])[0] is True
    finally:
        pool._stop_threads()
    assert sorted(r["input"] for r in _lines(out, "refinement")) == ["m1", "m2"]


# --- refinement_worker: per-output details -------------------------------------------------

import pytest


class _Tok:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return messages[-1]["content"]


def _vllm_output(text, prompt, completion, reason):
    return types.SimpleNamespace(
        prompt_token_ids=list(range(prompt)),
        outputs=[types.SimpleNamespace(text=text, token_ids=list(range(completion)),
                                       finish_reason=reason)])


@pytest.fixture
def worker(monkeypatch):
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


def test_vllm_worker_reports_details_for_each_output(worker):
    model = types.SimpleNamespace(generate=lambda texts, sampling_params=None, use_tqdm=False: [
        _vllm_output("x", 4, 2, "stop"), _vllm_output("y", 6, 9, "length")])
    texts, usage, details = worker.generate_with_details(
        model, _Tok(), "sys", ["a", "b"], 9, False, "vllm")
    assert texts == ["x", "y"]
    assert details == [
        {"finish_reason": "stop", "prompt_tokens": 4, "completion_tokens": 2},
        {"finish_reason": "length", "prompt_tokens": 6, "completion_tokens": 9}]
    assert usage["prompt_tokens"] == 10


def test_generate_with_usage_keeps_its_two_value_shape(worker):
    model = types.SimpleNamespace(generate=lambda texts, sampling_params=None, use_tqdm=False: [
        _vllm_output("x", 4, 2, "stop")])
    assert len(worker.generate_with_usage(model, _Tok(), "sys", ["a"], 9, False, "vllm")) == 2


def test_hf_details_split_tokens_per_row(worker):
    import torch
    generated = torch.tensor([[5, 6, 0, 0], [5, 6, 7, 8]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    details = worker._hf_details(generated, mask, pad_token_id=0, max_new_tokens=4)
    assert details == [
        {"finish_reason": "stop", "prompt_tokens": 3, "completion_tokens": 2},
        {"finish_reason": "length", "prompt_tokens": 2, "completion_tokens": 4}]


# --- in-process generation ----------------------------------------------------------------

from services.diarization_refinement_service import DiarizationRefinementService


def _local_service(result):
    service = DiarizationRefinementService.__new__(DiarizationRefinementService)
    service._ensure_worker_started = lambda: False
    service.model = object()
    service.tokenizer = types.SimpleNamespace(pad_token_id=0, padding_side="right",
                                              pad_token="p", eos_token="e")
    service.logger = None
    service.last_failure = None
    service._generate_batch = lambda *a, **k: result
    return service


def test_in_process_generation_is_traced(tmp_path):
    out = str(tmp_path)
    service = _local_service((True, ["o1", "o2"]))
    with llm_trace.trace_to("speaker_relabel", out):
        assert service.generate_texts("S", ["i1", "i2"], 100, thinking=True) == (
            True, ["o1", "o2"])
    rows = _lines(out, "speaker_relabel")
    assert [(r["input"], r["output"], r["backend"]) for r in rows] == [
        ("i1", "o1", "in_process"), ("i2", "o2", "in_process")]
    assert rows[0]["thinking"] is True and rows[0]["max_new_tokens"] == 100


def test_in_process_failure_is_traced_and_untraced_call_writes_nothing(tmp_path):
    out = str(tmp_path / "t")
    service = _local_service((False, []))
    service.last_failure = "out of memory: x"
    with llm_trace.trace_to("refinement", out):
        assert service.generate_texts("S", ["i1"]) == (False, [])
    row = _lines(out, "refinement")[0]
    assert (row["ok"], row["output"], row["error"]) == (False, None, "out of memory: x")
    service.generate_texts("S", ["i1"])
    assert len(_lines(out, "refinement")) == 1


# --- no data lost -------------------------------------------------------------------------

def test_prompt_file_failure_does_not_lose_jsonl_lines(tmp_path):
    out = str(tmp_path)
    folder = tmp_path / "llm_trace"
    (folder / "refinement.system_prompts.json").mkdir(parents=True)   # cannot be written
    with llm_trace.trace_to("refinement", out):
        _record(llm_trace.current())
    rows = _lines(out, "refinement")
    assert [r["output"] for r in rows] == ["A", "B"]
    assert all(r["system_prompt_sha1"] for r in rows)


def test_prompt_is_rewritten_when_the_output_dir_was_recreated(tmp_path):
    import shutil
    out = str(tmp_path / "ep")
    with llm_trace.trace_to("refinement", out):
        _record(llm_trace.current())
    shutil.rmtree(out)
    with llm_trace.trace_to("refinement", out):
        _record(llm_trace.current())
    with open(os.path.join(out, "llm_trace", "refinement.system_prompts.json")) as handle:
        assert list(json.load(handle).values()) == ["SYS"]


def test_prompt_is_not_cached_when_its_write_failed(tmp_path):
    out = str(tmp_path)
    blocker = tmp_path / "llm_trace" / "refinement.system_prompts.json"
    blocker.mkdir(parents=True)
    with llm_trace.trace_to("refinement", out):
        _record(llm_trace.current())
    blocker.rmdir()
    with llm_trace.trace_to("refinement", out):
        _record(llm_trace.current())
    with open(blocker) as handle:
        assert list(json.load(handle).values()) == ["SYS"]


def test_corrupt_prompt_file_is_kept_aside_not_overwritten(tmp_path):
    out = str(tmp_path)
    folder = tmp_path / "llm_trace"
    folder.mkdir()
    (folder / "refinement.system_prompts.json").write_text('{"abc": "old pro', encoding="utf-8")
    with llm_trace.trace_to("refinement", out):
        _record(llm_trace.current())
    kept = [p for p in folder.iterdir() if ".bad-" in p.name]
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == '{"abc": "old pro'
    with open(folder / "refinement.system_prompts.json") as handle:
        assert list(json.load(handle).values()) == ["SYS"]
    assert not list(folder.glob("*.tmp"))


def test_lone_surrogate_does_not_lose_the_call(tmp_path):
    out = str(tmp_path)
    with llm_trace.trace_to("refinement", out):
        _record(llm_trace.current(), messages=["a", "b"], outputs=["bad \ud800 text", "fine"])
    rows = _lines(out, "refinement")
    assert len(rows) == 2 and rows[1]["output"] == "fine"
    assert rows[0]["output"] == "bad \ud800 text"


def test_in_process_error_names_the_exception_not_a_stale_failure(tmp_path):
    out = str(tmp_path)
    service = _local_service((True, []))
    service.last_failure = "stale failure"

    def boom(*a, **k):
        raise RuntimeError("kaput")
    service._generate_batch = boom
    with llm_trace.trace_to("refinement", out):
        with pytest.raises(RuntimeError):
            service.generate_texts("S", ["i1"])
    assert _lines(out, "refinement")[0]["error"] == "RuntimeError: kaput"
