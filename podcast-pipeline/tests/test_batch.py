"""Tests for duration-bounded batching and stage-major execution.

Run:  python -m pytest tests/test_batch.py -q     (from podcast-pipeline/)
"""
import os
import sys
import types
import threading
import time
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import batch as B
from utils.batch import PIPELINE_STAGES, plan_batches, run_batch_by_stage

HOUR = 3600.0


def _durations(mapping):
    return patch.object(B, "audio_duration", lambda p: mapping[p])


class FakePipeline:
    """Records (stage, path) in call order."""

    def __init__(self, fail_on=None, fail_stage=None):
        self.calls = []
        self.fail_on = fail_on
        self.fail_stage = fail_stage

    def run(self, args, config, path):
        stage = getattr(args, "stop_after", None)
        self.calls.append((stage, path))
        if path == self.fail_on and stage == self.fail_stage:
            raise RuntimeError("boom")


def _args(**kw):
    a = types.SimpleNamespace(stop_after=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# --- planning ------------------------------------------------------------

def test_batch_total_stays_within_the_limit():
    durs = {c: HOUR for c in "abcdefg"}
    with _durations(durs):
        batches = plan_batches(list(durs), 5.0)
    assert [len(b) for b in batches] == [5, 2]
    for b in batches:
        assert sum(durs[p] for p in b) <= 5 * HOUR + 1e-6


def test_forty_hours_becomes_two_full_pipeline_groups_at_twenty_hours():
    durs = {f"f{i:02d}": HOUR for i in range(40)}
    with _durations(durs):
        batches = plan_batches(list(durs), 20.0)
    assert [len(batch) for batch in batches] == [20, 20]


def test_a_batch_may_reach_the_limit_exactly():
    durs = {"a": 2 * HOUR, "b": 3 * HOUR, "c": HOUR}
    with _durations(durs):
        batches = plan_batches(list(durs), 5.0)
    assert batches == [["a", "b"], ["c"]], "5.00h exactly must not spill into a second batch"


def test_oversized_file_runs_alone_rather_than_being_dropped():
    durs = {"a": HOUR, "big": 9 * HOUR, "b": HOUR}
    with _durations(durs):
        batches = plan_batches(list(durs), 5.0)
    assert ["big"] in batches
    assert sum(len(b) for b in batches) == 3, "no file may be silently skipped"


def test_unreadable_duration_is_isolated():
    durs = {"a": HOUR, "bad": 0.0, "b": HOUR}
    with _durations(durs):
        batches = plan_batches(list(durs), 5.0)
    assert ["bad"] in batches


def test_main_never_bypasses_the_duration_limit_for_stage_major_runs():
    """--by_stage changes ordering, not which files fit in one pass."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "main.py"), encoding="utf-8") as f:
        source = f.read()

    assert "group = plan_batches(todo, max_hours, logger=logger)[0]" in source
    assert "group = list(todo)" not in source
    assert "if args.only_batch is not None or not args.by_stage" not in source


# --- stage-major execution ----------------------------------------------

def test_every_file_finishes_a_stage_before_the_next_stage_starts():
    files = ["f1", "f2", "f3"]
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(
        qwen3omni=True, step_speaker_relabel=True,
        step_word_alignment=True, step_conversation_exports=True), {}, files)

    seen = []
    for stage, _path in pipe.calls:
        if not seen or seen[-1] != stage:
            seen.append(stage)
    assert seen == list(PIPELINE_STAGES), (
        f"stages ran as {seen}; each stage must cover the whole batch before the next"
    )
    for stage in PIPELINE_STAGES:
        assert [p for s, p in pipe.calls if s == stage] == files


def test_all_asr_finishes_before_any_refinement_starts():
    """The GPU hand-off requested for long corpora: never interleave ASR and LLM.

    This is deliberately more explicit than the generic stage-major assertion:
    regressing the stage order must not make ASR start again after the first
    refinement call.
    """
    files = ["f1", "f2", "f3"]
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(), {}, files)

    asr_calls = [i for i, (stage, _) in enumerate(pipe.calls) if stage == "asr"]
    refinement_calls = [i for i, (stage, _) in enumerate(pipe.calls)
                        if stage == "refinement"]
    assert asr_calls and refinement_calls
    assert max(asr_calls) < min(refinement_calls)


def test_captioning_is_skipped_when_its_model_is_off():
    """With qwen3omni off the stage returns without touching the transcripts,
    so the pass only reloads the audio and re-reads four checkpoints."""
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(qwen3omni=False), {}, ["f1"])

    assert "captioning" not in [s for s, _ in pipe.calls]
    # Every always-on stage still runs; optional postprocessing stays opt-in.
    assert [s for s, _ in pipe.calls] == [
        s for s in PIPELINE_STAGES
        if s not in {"captioning", "speaker_relabel", "word_alignment",
                     "conversation_exports"}]


def test_captioning_runs_when_its_model_is_on():
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(qwen3omni=True), {}, ["f1"])

    assert "captioning" in [s for s, _ in pipe.calls]


def test_a_file_that_fails_is_dropped_from_later_stages():
    files = ["f1", "f2", "f3"]
    pipe = FakePipeline(fail_on="f2", fail_stage="diarization")
    failures = run_batch_by_stage(pipe, _args(), {}, files)

    assert [p for p, _ in failures] == ["f2"]
    later = [p for s, p in pipe.calls if s == "separation"]
    assert later == ["f1", "f3"], (
        "a file that died in diarization must not be retried in every later "
        "stage, and must not take its neighbours down"
    )


def test_stop_after_is_respected():
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(stop_after="separation"), {}, ["f1", "f2"])
    stages = {s for s, _ in pipe.calls}
    assert stages == {"music", "diarization", "separation"}, f"ran {stages}"


def test_the_music_stage_gets_its_own_pass():
    """It loads PANNs and a vocal separator; a pass loads them once for the
    batch rather than once inside each file's diarization pass."""
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(stop_after=None), {}, ["f1", "f2"])
    assert [s for s, _ in pipe.calls][:2] == ["music", "music"]


def test_a_run_that_stops_after_music_runs_one_pass():
    """The regression: five later passes each reloaded the audio, re-read the
    checkpoints and returned at the same guard, and the ledger then repeated
    the lot on a second attempt."""
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(stop_after=None, step_diarization=False),
                       {}, ["f1", "f2"])
    assert {s for s, _ in pipe.calls} == {"music"}


class _FakeSeparationPools:
    """Records which close_* methods were called, without touching real pools."""

    def __init__(self):
        self.closed = []

    def close_window_pool(self):
        self.closed.append("close_window_pool")

    def close_prefetch_pool(self):
        self.closed.append("close_prefetch_pool")

    def close_async_pools(self):
        self.closed.append("close_async_pools")


def test_a_run_that_stops_after_diarization_still_closes_the_prefetch_pools():
    """The regression: prefetch_overlap_plan()/close_window_pool() are wired
    into PipelineService.run()'s 'separation' section, which a --stop_after
    diarization run never reaches -- the stage loop breaks right after the
    diarization pass. Without a safety net, the window-build pool and the
    prefetch coordinator pool a diarization-stage prefetch may have started
    are never closed for the rest of the process."""
    pipe = FakePipeline()
    pipe.separation_svc = _FakeSeparationPools()
    run_batch_by_stage(pipe, _args(stop_after="diarization"), {}, ["f1"])
    stages = {s for s, _ in pipe.calls}
    assert stages == {"music", "diarization"}, f"ran {stages}"
    assert set(pipe.separation_svc.closed) == {
        "close_window_pool", "close_prefetch_pool", "close_async_pools"}


def test_the_safety_net_does_not_choke_on_a_pipeline_with_no_separation_svc():
    """FakePipeline (and PipelineService.__new__ in other tests) may have no
    separation_svc at all -- the safety net must be a no-op, not a crash."""
    run_batch_by_stage(FakePipeline(), _args(stop_after="diarization"), {}, ["f1"])


def test_a_stage_switched_off_still_gets_its_pass():
    """Only the load-bearing steps end the run. A pass whose own stage is off
    still carries the pipeline from the previous stage to the next."""
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(stop_after=None, step_separation=False),
                       {}, ["f1"])
    stages = {s for s, _ in pipe.calls}
    assert "separation" in stages
    assert None in stages, "the run must still reach the end"


def test_caller_args_are_not_mutated():
    args = _args(stop_after=None)
    run_batch_by_stage(FakePipeline(), args, {}, ["f1"])
    assert args.stop_after is None, "stop_after must be set on a copy, not the caller's args"


def test_diarization_runs_two_files_concurrently_when_pool_is_enabled():
    class ConcurrentPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self.inside = 0
            self.max_inside = 0
            self.lock = threading.Lock()

        def parallel_stage_view(self, stage):
            return self

        def run(self, args, config, path):
            with self.lock:
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
            time.sleep(0.03)
            with self.lock:
                self.inside -= 1

    perf = {
        "enabled": True,
        "stages": {"diarization": {"workers": 2}},
    }
    pipe = ConcurrentPipeline()
    run_batch_by_stage(
        pipe, _args(performance_config=perf, dia3=False), {},
        ["f1", "f2", "f3"], stages=("diarization",))

    assert pipe.max_inside == 2


def test_separation_runs_two_files_concurrently_when_pool_is_enabled():
    class ConcurrentPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self.inside = 0
            self.max_inside = 0
            self.lock = threading.Lock()

        def parallel_stage_view(self, stage):
            return self

        def run(self, args, config, path):
            with self.lock:
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
            time.sleep(0.03)
            with self.lock:
                self.inside -= 1

    perf = {
        "enabled": True,
        "stages": {"separation": {"max_workers": 2}},
    }
    pipe = ConcurrentPipeline()
    run_batch_by_stage(
        pipe, _args(performance_config=perf, separator="sidon"), {},
        ["f1", "f2", "f3"], stages=("separation",))

    assert pipe.max_inside == 2


def test_music_runs_two_files_concurrently_when_cross_file_overlap_is_on():
    """SSLAM classifying file N+1 while BS-RoFormer removes music from file N
    only helps if the batch pass actually lets two files be in flight."""
    class ConcurrentPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self.inside = 0
            self.max_inside = 0
            self.lock = threading.Lock()

        def parallel_stage_view(self, stage):
            return self

        def run(self, args, config, path):
            with self.lock:
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
            time.sleep(0.03)
            with self.lock:
                self.inside -= 1

    perf = {
        "enabled": True,
        "stages": {"music": {"cross_file_overlap": True}},
    }
    pipe = ConcurrentPipeline()
    run_batch_by_stage(
        pipe, _args(performance_config=perf), {},
        ["f1", "f2", "f3"], stages=("music",))

    assert pipe.max_inside == 2


def test_music_runs_two_files_concurrently_when_max_separator_workers_is_two():
    """A 2-instance BS-RoFormer pool (one per GPU) is exactly the condition
    under which two files' own strip_music_spans() calls sharing the same
    checkout queue (see music_service.py) benefit from running at once --
    same as cross_file_overlap, just via two instances of one model instead
    of two model types on two cards."""
    class ConcurrentPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self.inside = 0
            self.max_inside = 0
            self.lock = threading.Lock()

        def parallel_stage_view(self, stage):
            return self

        def run(self, args, config, path):
            with self.lock:
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
            time.sleep(0.03)
            with self.lock:
                self.inside -= 1

    perf = {
        "enabled": True,
        "stages": {"music": {"cross_file_overlap": False, "max_separator_workers": 2}},
    }
    pipe = ConcurrentPipeline()
    run_batch_by_stage(
        pipe, _args(performance_config=perf), {},
        ["f1", "f2", "f3"], stages=("music",))

    assert pipe.max_inside == 2


def test_music_stays_sequential_when_cross_file_overlap_is_off():
    """The flag is opt-in: a profile that never set it must see no change."""
    class ConcurrentPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self.inside = 0
            self.max_inside = 0
            self.lock = threading.Lock()

        def parallel_stage_view(self, stage):
            return self

        def run(self, args, config, path):
            with self.lock:
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
            time.sleep(0.01)
            with self.lock:
                self.inside -= 1

    perf = {"enabled": True, "stages": {"music": {}}}
    pipe = ConcurrentPipeline()
    run_batch_by_stage(
        pipe, _args(performance_config=perf), {},
        ["f1", "f2", "f3"], stages=("music",))

    assert pipe.max_inside == 1


def test_the_next_stage_waits_for_a_slow_diarization_postprocess_future():
    """The drain barrier: 'separation' must not start for ANY file until
    every diarization postprocess future from the previous pass has
    resolved, even though run_one() already returned for that file."""
    release = threading.Event()

    class DeferredPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self._pending_diar_jobs = {}
            self._pending_diar_lock = threading.Lock()

        def run(self, args, config, path):
            stage = getattr(args, "stop_after", None)
            self.calls.append((stage, path))
            if stage == "diarization":
                from concurrent.futures import ThreadPoolExecutor

                def _slow():
                    release.wait(timeout=5)
                    return "done"

                future = ThreadPoolExecutor(max_workers=1).submit(_slow)
                self._pending_diar_jobs[path] = future

    pipe = DeferredPipeline()
    result_holder = {}

    def _run():
        result_holder["failures"] = run_batch_by_stage(
            pipe, _args(stop_after="separation"), {}, ["f1"])

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=0.3)
    assert thread.is_alive(), "separation must not start before the drain releases it"
    assert "separation" not in {s for s, _ in pipe.calls}

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert "separation" in {s for s, _ in pipe.calls}


def test_a_failing_diarization_postprocess_future_is_reported_and_excludes_the_file():
    class DeferredFailPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self._pending_diar_jobs = {}
            self._pending_diar_lock = threading.Lock()

        def run(self, args, config, path):
            stage = getattr(args, "stop_after", None)
            self.calls.append((stage, path))
            if stage == "diarization" and path == "bad":
                from concurrent.futures import ThreadPoolExecutor

                def _boom():
                    raise RuntimeError("no segments")

                self._pending_diar_jobs[path] = ThreadPoolExecutor(max_workers=1).submit(_boom)

    pipe = DeferredFailPipeline()
    failures = run_batch_by_stage(pipe, _args(), {}, ["good", "bad"])

    assert [p for p, _ in failures] == ["bad"]
    later = [p for s, p in pipe.calls if s == "separation"]
    assert later == ["good"], "the file whose postprocess failed must not reach separation"


def test_the_safety_net_also_closes_the_diarization_postprocess_pool():
    class _FakeDiarPool:
        def __init__(self):
            self.closed = False

        def close_postprocess_pool(self):
            self.closed = True

    pipe = FakePipeline()
    pipe.diarization_svc = _FakeDiarPool()
    run_batch_by_stage(pipe, _args(stop_after="diarization"), {}, ["f1"])
    assert pipe.diarization_svc.closed


def test_the_safety_net_also_closes_the_music_async_pools():
    class _FakeMusicPool:
        def __init__(self):
            self.closed = False

        def close_async_pools(self):
            self.closed = True

    pipe = FakePipeline()
    pipe.music_svc = _FakeMusicPool()
    run_batch_by_stage(pipe, _args(stop_after="music"), {}, ["f1"])
    assert pipe.music_svc.closed
