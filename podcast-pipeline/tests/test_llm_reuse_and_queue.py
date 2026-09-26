"""P5: the Qwen replicas are reused across Refinement and Speaker relabel, share one
work queue, and progress goes to the log.

Fakes only: no model, no GPU, no subprocess.

Run:  python -m pytest tests/test_llm_reuse_and_queue.py -q     (from podcast-pipeline/)
"""
import json
import logging
import os
import sys
import threading
import time
import types
from concurrent.futures import Future

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import batch as B
from utils.llm_batches import ask_in_batches, llm_concurrency
from utils.llm_progress import LlmProgress
from utils.performance_config import resolve
from services.refinement_worker_service import (
    RefinementWorkerPoolService, RefinementWorkerService)


# --- fakes ---------------------------------------------------------------------------

class _Process:
    def __init__(self):
        self.dead = False

    def poll(self):
        return 1 if self.dead else None


class _Replica:
    """A replica service: answers `<msg>!` after `delay` seconds and remembers what it saw."""

    def __init__(self, name, delay=0.0, fail=False, die=True):
        self.name, self.delay, self.fail, self.die = name, delay, fail, die
        self.process = _Process()
        self.model_name = "fake"
        self.seen = []
        self.spawned = self.stopped = 0
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def spawn(self):
        self.spawned += 1
        self.process = _Process()

    def wait_ready(self):
        pass

    def ping(self):
        return True

    def stop(self):
        self.stopped += 1
        self.process = None

    def generate_texts(self, system_prompt, messages, max_new_tokens=512, thinking=False):
        time.sleep(self.delay)
        self.seen.append(list(messages))
        if self.fail:
            self.process.dead = self.die
            return False, []
        return True, [f"{m}!" for m in messages]


# --- shared queue ----------------------------------------------------------------------

def test_a_free_replica_takes_the_next_job_instead_of_waiting_for_the_other_one():
    slow, fast = _Replica("slow", delay=0.3), _Replica("fast", delay=0.0)
    pool = RefinementWorkerPoolService([slow, fast])
    try:
        futures = [pool.submit("sys", [f"m{i}"]) for i in range(6)]
        results = [f.result(timeout=10) for f in futures]
    finally:
        pool.stop()
    assert all(ok for ok, _ in results)
    assert [t for _, t in results] == [[f"m{i}!"] for i in range(6)]
    # A round robin would give each three; the queue lets the quick one do the rest.
    assert len(fast.seen) > len(slow.seen)


def test_generate_texts_keeps_its_contract_over_the_queue():
    pool = RefinementWorkerPoolService([_Replica("a"), _Replica("b")])
    try:
        ok, texts = pool.generate_texts("sys", [f"m{i}" for i in range(5)])
    finally:
        pool.stop()
    assert ok and texts == [f"m{i}!" for i in range(5)]
    assert pool.generate_texts("sys", []) == (True, [])


def test_a_failed_part_fails_the_whole_call():
    pool = RefinementWorkerPoolService([_Replica("b", fail=True, die=False)])
    try:
        ok, texts = pool.generate_texts("sys", ["x", "y"])
    finally:
        pool.stop()
    assert not ok and texts == []


def test_a_job_of_a_dead_replica_goes_to_the_one_still_up():
    dead, alive = _Replica("dead", fail=True), _Replica("alive", delay=0.05)
    pool = RefinementWorkerPoolService([dead, alive])
    try:
        results = [pool.submit("sys", [f"m{i}"]).result(timeout=10) for i in range(4)]
    finally:
        pool.stop()
    assert all(ok for ok, _ in results)
    assert [t for _, t in results] == [[f"m{i}!"] for i in range(4)]


def test_replicas_are_started_once_and_stopped_together():
    a, b = _Replica("a"), _Replica("b")
    pool = RefinementWorkerPoolService([a, b])
    pool.start()
    pool.stop()
    assert (a.spawned, b.spawned, a.stopped, b.stopped) == (1, 1, 1, 1)


def test_the_worker_counts_the_tokens_the_engine_reports():
    service = RefinementWorkerService.__new__(RefinementWorkerService)
    service._usage_lock = threading.Lock()
    service._usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "repetition_stops": 0}
    service._lock = threading.Lock()
    service.logger = None

    class _Pipe:
        def __init__(self):
            self.written = []

        def write(self, text):
            self.written.append(text)

        def flush(self):
            pass

        def readline(self):
            return json.dumps({"ok": True, "texts": ["a", "b"],
                               "usage": {"prompt_tokens": 100, "completion_tokens": 7,
                                         "repetition_stops": 1}})

    pipe = _Pipe()
    service.process = types.SimpleNamespace(stdin=pipe, stdout=pipe)
    assert service.generate_texts("s", ["x", "y"]) == (True, ["a", "b"])
    service.generate_texts("s", ["x", "y"])
    assert service.usage == {"requests": 4, "prompt_tokens": 200, "completion_tokens": 14,
                             "repetition_stops": 2}


# --- progress ------------------------------------------------------------------------------

class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, text):
        self.lines.append(text)

    warning = info


def test_progress_is_a_plain_log_line_with_rate_and_eta():
    clock, log = _Clock(), _Log()
    progress = LlmProgress(450, logger=log, interval=15, clock=clock)
    clock.now = 38.7
    progress.advance(120)
    assert log.lines == ["[LLM] 120/450 (26%, 3.1 seg/s, ~106s)"]
    assert "\r" not in log.lines[0]


def test_progress_never_goes_back_and_ends_complete_once():
    clock, log = _Clock(), _Log()
    progress = LlmProgress(10, logger=log, interval=1000, clock=clock)
    clock.now = 1
    progress.advance(4)
    progress.advance(4)       # throttled
    progress.advance(4)       # clamped to the total, and complete: written
    progress.finish()         # nothing new to say
    done = [int(line.split()[1].split("/")[0]) for line in log.lines]
    assert done == sorted(done) and done[-1] == 10
    assert log.lines[-1].startswith("[LLM] 10/10 (100%")
    assert len(log.lines) == 2


def test_progress_reports_engine_tokens_when_known():
    clock, log = _Clock(), _Log()
    progress = LlmProgress(2, logger=log, clock=clock, usage=lambda: {
        "prompt_tokens": 500, "completion_tokens": 20})
    clock.now = 1
    progress.advance(1)
    assert log.lines[0].endswith("| tokens in=500 out=20")


# --- refinement loop ---------------------------------------------------------------------

refinement = pytest.importorskip("services.diarization_refinement_service")
from schemas.transcript import TranscriptSegment


def _segments(n):
    return [TranscriptSegment(
        index=f"{i:05d}", start=i * 5.0, end=i * 5.0 + 4.0, speaker="0",
        text=f"c{i}", text_whisper=f"cau so {i} hom nay", text_phowhisper=f"cau so {i} hom nay",
        text_qwen3=f"cau so {i} hom nay", language="vi",
        bs_roformer=False, bss=False) for i in range(n)]


class _Pool:
    """A pool double: records the order of events; answers from a script."""

    def __init__(self, events, replicas=2, fail_when=None):
        self.services = [object()] * replicas
        self.events = events
        self.fail_when = fail_when or (lambda messages: False)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.calls = 0

    def submit(self, system_prompt, messages, max_new_tokens=512, thinking=False):
        self.events.append(("submit", messages[0]))
        future = Future()
        if self.fail_when(messages):
            future.set_result((False, []))
        else:
            future.set_result((True, [m.split("Bản 1: ")[1].split("\n")[0] + " ok" for m in messages]))
        return future

    def generate_texts(self, system_prompt, messages, max_new_tokens=512, thinking=False):
        self.events.append(("sync", messages[0]))
        if self.fail_when(messages):
            return False, []
        return True, [m.split("Bản 1: ")[1].split("\n")[0] + " ok" for m in messages]

    def ping(self):
        return True


def _service(events, shared_queue, batch_size=8, **pool_kw):
    svc = refinement.DiarizationRefinementService(
        logger=None, batch_size=batch_size, backend="vllm", workers=2,
        shared_queue=shared_queue, config={})
    svc._worker_env_error = None
    svc._worker = _Pool(events, **pool_kw)
    svc._worker_devices = [0, 1]
    # Acceptance is the real one; the fake answer is the segment's own text plus a word.
    return svc


def test_queued_refinement_gives_the_same_texts_as_the_sequential_loop():
    a, b = _segments(23), _segments(23)
    _service([], shared_queue=False).refine(a)
    _service([], shared_queue=True).refine(b)
    assert [s.text for s in a] == [s.text for s in b]
    assert all(s.text.endswith(" ok") for s in b)


def test_the_next_chunks_are_queued_before_the_first_answers_are_checked():
    events = []
    svc = _service(events, shared_queue=True, batch_size=8)      # 4 per replica-chunk
    original = svc._apply_batch

    def apply(batch, decoded):
        events.append(("apply", batch[0][1]))
        return original(batch, decoded)

    svc._apply_batch = apply
    svc.refine(_segments(24))                                   # 6 chunks
    kinds = [kind for kind, _ in events]
    first_apply = kinds.index("apply")
    # replicas + 1 chunks are already queued when the first answer is checked, and
    # the next one is queued before that check runs.
    assert kinds[:first_apply].count("submit") >= 4
    assert kinds.count("apply") == 6 and kinds.count("submit") == 6
    # Answers are applied in chunk order.
    applied = [msg for kind, msg in events if kind == "apply"]
    submitted = [msg for kind, msg in events if kind == "submit"]
    assert applied == submitted


def test_a_chunk_that_fails_is_halved_and_only_the_bad_request_is_lost():
    poison = "cau so 5 hom nay"
    svc = _service([], shared_queue=True, batch_size=8,
                   fail_when=lambda messages: any(poison in m for m in messages))
    segs = _segments(16)
    svc.refine(segs)
    unrefined = [s.index for s in segs if not s.text.endswith(" ok")]
    assert unrefined == ["00005"]


def test_fast_replica_gets_more_chunks_while_first_chunk_is_still_running():
    """The admission queue must not wait for source-order result application."""
    from concurrent.futures import ThreadPoolExecutor

    svc = _service([], shared_queue=True, batch_size=2)
    first = Future()
    refilled = threading.Event()
    original = svc._worker.submit
    held = []
    calls = []

    def submit(*args, **kwargs):
        answer = original(*args, **kwargs)
        calls.append(answer)
        if len(calls) == 1:
            held.append(answer.result())
            return first
        if len(calls) >= 4:       # past the initial depth of replicas + 1
            refilled.set()
        return answer

    svc._worker.submit = submit
    segments = _segments(8)
    with ThreadPoolExecutor(max_workers=1) as executor:
        work = executor.submit(svc.refine, segments)
        try:
            assert refilled.wait(5), "fast lane stalled behind the first chunk"
            assert not work.done()
        finally:
            first.set_result(held[0] if held else (False, []))
        result = work.result(timeout=5)
    assert [s.index for s in result] == [s.index for s in segments]
    assert all(s.text.endswith(" ok") for s in result)


def test_most_chunks_failing_still_raises_the_failure_limit():
    svc = _service([], shared_queue=True, fail_when=lambda messages: True)
    with pytest.raises(RuntimeError, match="LLM refinement failed"):
        svc.refine(_segments(16))


def test_refinement_no_longer_draws_a_tqdm_bar_and_logs_progress(caplog):
    logger = logging.getLogger("p5-progress")
    svc = _service([], shared_queue=True, batch_size=4)
    svc.logger = logger
    with caplog.at_level(logging.INFO, logger="p5-progress"):
        svc.refine(_segments(12))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[LLM] ")
             and "/" in r.getMessage()]
    assert lines and lines[-1].startswith("[LLM] 12/12 (100%")
    import inspect
    assert "tqdm" not in inspect.getsource(refinement.DiarizationRefinementService.refine)


# --- relabel: independent windows on both cards --------------------------------------------

def test_two_independent_windows_are_in_flight_together_when_parallel_is_on():
    gate = threading.Barrier(2, timeout=5)

    class _Llm:
        parallel_windows, replica_count = True, 2

        def generate_texts(self, system_prompt, messages, max_new_tokens=512,
                           use_prefix=False, labels=None, thinking=False):
            gate.wait()            # only passes when the other window is asked at the same time
            return True, [f"r:{m}" for m in messages]

    replies, unanswered = ask_in_batches(
        _Llm(), "sys", ["w0", "w1"], per_call=1, max_new_tokens=4096, label="relabel window")
    assert replies == ["r:w0", "r:w1"] and unanswered == 0


def test_without_the_switch_windows_are_asked_one_at_a_time_as_before():
    order = []

    class _Llm:
        replica_count = 2      # parallel_windows missing: off

        def generate_texts(self, system_prompt, messages, max_new_tokens=512,
                           use_prefix=False, labels=None, thinking=False):
            order.append(("in", messages[0]))
            time.sleep(0.02)
            order.append(("out", messages[0]))
            return True, list(messages)

    assert llm_concurrency(_Llm()) == 1
    ask_in_batches(_Llm(), "sys", ["a", "b", "c"], per_call=1, max_new_tokens=8, label="x")
    assert [k for k, _ in order] == ["in", "out"] * 3


def test_parallel_replies_land_at_their_own_window_and_a_failed_window_is_halved_alone():
    class _Llm:
        parallel_windows, replica_count = True, 2

        def generate_texts(self, system_prompt, messages, max_new_tokens=512,
                           use_prefix=False, labels=None, thinking=False):
            if len(messages) > 1 or messages[0] == "bad":
                return False, []
            return True, [m.upper() for m in messages]

    replies, unanswered = ask_in_batches(
        _Llm(), "sys", ["a", "bad", "c", "d", "e"], per_call=2, max_new_tokens=8, label="x")
    assert replies == ["A", None, "C", "D", "E"] and unanswered == 1


def test_relabel_keeps_its_window_prompt_and_output_budget_and_uses_both_replicas():
    from services.speaker_relabel_service import SpeakerRelabelService

    seen = []
    lock = threading.Lock()
    gate = threading.Barrier(2, timeout=5)

    class _Llm:
        model_name = "fake"
        batch_size = 1024
        max_batch_tokens = 32768       # the whole pool's budget, as in the a100 profile
        parallel_windows, replica_count = True, 2

        def ensure_loaded(self):
            return True

        def count_tokens(self, text):
            return len(text.split())

        def generate_texts(self, system_prompt, messages, max_new_tokens=512,
                           use_prefix=False, labels=None, thinking=False):
            with lock:
                seen.append((len(messages), max_new_tokens, system_prompt))
            if len(seen) <= 2:
                gate.wait()
            return True, ["[]" for _ in messages]

    segs = [TranscriptSegment(
        index=f"{i:05d}", start=i * 5.0, end=i * 5.0 + 4.0, speaker="AB"[i % 2],
        text="một hai ba bốn năm sáu", text_whisper="w", text_phowhisper="p",
        text_qwen3="q", language="vi", bs_roformer=False, bss=False) for i in range(80)]
    svc = SpeakerRelabelService(_Llm(), window_tokens=32000, overlap_segments=30,
                                max_new_tokens=4096, max_window_segments=20)
    result = svc.relabel(segs)
    assert result.windows >= 2 and result.failed_windows == 0
    # Each window is its own request with the unchanged 4096 output budget.
    assert {n for n, _, _ in seen} == {1}
    assert {t for _, t, _ in seen} == {4096}
    assert len({p for _, _, p in seen}) == 1


# --- config -----------------------------------------------------------------------------------

def test_the_new_switches_default_off_and_the_a100_profile_sets_them():
    stage = resolve({"performance": {"enabled": True}})["stages"]["refinement"]
    assert (stage["keep_llm_across_relabel"], stage["shared_queue"],
            stage["parallel_windows"]) == (False, False, False)
    config = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")))
    tuned = resolve(config["environments"]["a100"])["stages"]["refinement"]
    assert tuned["keep_llm_across_relabel"] is True
    assert tuned["shared_queue"] is True
    assert tuned["chunk_size"] == 32
    assert resolve(config["environments"]["a100"])["stages"]["refinement"]["parallel_windows"]


# --- lifecycle: how many times the replicas are started ---------------------------------------

from services.pipeline_service import PipelineService


class _Spawner:
    """Counts replica spawns through the real DiarizationRefinementService."""

    def __init__(self):
        self.spawns = 0
        self.stops = 0


def _lifecycle(monkeypatch, keep, relabel=True):
    from services import refinement_worker_service as rws
    counter = _Spawner()

    class _Service(_Replica):
        def __init__(self, **kw):
            super().__init__("r")
            self.model_name = kw.get("model_name", "fake")

        def spawn(self):
            counter.spawns += 1

        def stop(self):
            counter.stops += 1
            self.process = None

    monkeypatch.setattr(rws, "RefinementWorkerService", _Service)
    monkeypatch.setattr("utils.gpu_memory.wait_for_free_vram", lambda *a, **k: True)
    svc = refinement.DiarizationRefinementService(
        logger=None, backend="vllm", workers=2, pipeline_devices=[0, 1], config={})
    svc._worker_env_error, svc._worker_python = None, "python"
    pipe = PipelineService.__new__(PipelineService)
    pipe.logger = None
    pipe.refinement_svc = svc
    pipe.relabel_svc = object() if relabel else None
    pipe.model_loader = None
    pipe.worker_services = {}
    pipe.defer_free = pipe.defer_workers = pipe.defer_callbacks = None
    perf = {"enabled": True, "stages": {"refinement": {"keep_llm_across_relabel": keep}}}

    def stage_pass(stage, continues=True, uses_llm=True):
        args = types.SimpleNamespace(
            stop_after=stage, performance_config=perf, batch_continues=continues,
            keep_models=False, step_speaker_relabel=True)
        pipe.begin_stage_scope()
        try:
            if uses_llm:
                assert svc.ensure_loaded()
            pipe._release_llm_after(args, stage)
        finally:
            pipe.end_stage_scope()

    return counter, svc, pipe, stage_pass


def test_by_stage_refinement_then_relabel_starts_the_replicas_once(monkeypatch):
    counter, svc, pipe, stage_pass = _lifecycle(monkeypatch, keep=True)
    stage_pass("refinement")
    assert svc.is_resident()                       # handed on
    stage_pass("speaker_relabel")
    assert not svc.is_resident()                   # released before word alignment
    stage_pass("conversation_exports")             # the accepted second load
    assert counter.spawns == 4 and counter.stops == 4      # 2 replicas x 2 loads


def test_the_old_lifecycle_starts_them_three_times(monkeypatch):
    counter, svc, pipe, stage_pass = _lifecycle(monkeypatch, keep=False)
    stage_pass("refinement")
    assert not svc.is_resident()
    stage_pass("speaker_relabel")
    stage_pass("conversation_exports")
    assert counter.spawns == 6                      # 2 replicas x 3 loads


def test_a_batch_that_ends_at_refinement_does_not_hand_the_llm_on(monkeypatch):
    counter, svc, pipe, stage_pass = _lifecycle(monkeypatch, keep=True)
    stage_pass("refinement", continues=False)
    assert not svc.is_resident() and counter.spawns == 2


def test_without_a_relabel_service_refinement_still_releases(monkeypatch):
    counter, svc, pipe, stage_pass = _lifecycle(monkeypatch, keep=True, relabel=False)
    stage_pass("refinement")
    assert not svc.is_resident()


def test_the_batch_releases_a_handed_on_llm_when_relabel_never_ran(monkeypatch):
    counter, svc, pipe, stage_pass = _lifecycle(monkeypatch, keep=True)
    stage_pass("refinement")
    assert svc.is_resident()
    args = types.SimpleNamespace(stop_after=None, keep_models=False)
    stages = ("refinement",)
    # Nothing else in the batch: only refinement is listed, and its files ran.
    class _P:
        refinement_svc = svc
        _release_llm_before = pipe._release_llm_before

        def run(self, args, config, path):
            pass

    B.run_batch_by_stage(_P(), types.SimpleNamespace(
        stop_after=None, keep_models=False, step_llm_refinement=True),
        {}, [], stages=())
    assert not svc.is_resident()


def test_the_batch_marks_whether_it_goes_on_past_a_stage():
    seen = {}

    class _P:
        def run(self, args, config, path):
            seen[args.stop_after] = args.batch_continues

    args = types.SimpleNamespace(stop_after=None, step_refinement=True)
    B.run_batch_by_stage(_P(), args, {}, ["f"], stages=("refinement",))
    assert seen == {"refinement": True}
    args = types.SimpleNamespace(stop_after="refinement", step_refinement=True)
    seen.clear()
    B.run_batch_by_stage(_P(), args, {}, ["f"], stages=("refinement",))
    assert seen == {"refinement": False}
