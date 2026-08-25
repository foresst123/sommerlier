"""Tests for duration-bounded batching and stage-major execution.

Run:  python -m pytest tests/test_batch.py -q     (from podcast-pipeline/)
"""
import os
import sys
import types
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


# --- stage-major execution ----------------------------------------------

def test_every_file_finishes_a_stage_before_the_next_stage_starts():
    files = ["f1", "f2", "f3"]
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(qwen3omni=True), {}, files)

    seen = []
    for stage, _path in pipe.calls:
        if not seen or seen[-1] != stage:
            seen.append(stage)
    assert seen == list(PIPELINE_STAGES), (
        f"stages ran as {seen}; each stage must cover the whole batch before the next"
    )
    for stage in PIPELINE_STAGES:
        assert [p for s, p in pipe.calls if s == stage] == files


def test_captioning_is_skipped_when_its_model_is_off():
    """With qwen3omni off the stage returns without touching the transcripts,
    so the pass only reloads the audio and re-reads four checkpoints."""
    pipe = FakePipeline()
    run_batch_by_stage(pipe, _args(qwen3omni=False), {}, ["f1"])

    assert "captioning" not in [s for s, _ in pipe.calls]
    # Every other stage still runs.
    assert [s for s, _ in pipe.calls] == [
        s for s in PIPELINE_STAGES if s != "captioning"]


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
    assert stages == {"diarization", "separation"}, f"ran {stages}"


def test_caller_args_are_not_mutated():
    args = _args(stop_after=None)
    run_batch_by_stage(FakePipeline(), args, {}, ["f1"])
    assert args.stop_after is None, "stop_after must be set on a copy, not the caller's args"
