"""ASR intake decoupled from the vote: a file's thread is freed once its clips are
queued, the vote and checkpoint commit run on their own pool, and RAM stays
bounded. Fake models only -- no GPU."""

import os
import sys
import threading
import time
from concurrent.futures import Future

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.asr_scheduler import ByteBudget, FileTicket
from services.asr_service import ASRService
from tests.test_asr_cross_file import (
    FakePho, FakePool, FakeQwenClient, FakeWhisper, _audio, _segments, _summary)
from utils.asr_progress import AsrProgress
from utils.batch import _drain_pending_asr


class GatedWhisper(FakeWhisper):
    """Whisper that holds every batch until `gate` opens (a very slow lane)."""

    def __init__(self, gate):
        self.gate = gate

    def transcribe_batch(self, audios, vads, language="vi", batch_size=None, callback=None):
        assert self.gate.wait(20)
        return super().transcribe_batch(audios, vads, language, batch_size, callback)


def _service(whisper=None, workers=None, **cfg):
    performance = {"cross_file": True, "async_vote": True, "shared_batch_size": 16,
                   "boost_batch_size": 48, **cfg}
    return ASRService(
        whisper=whisper or FakeWhisper(), phowhisper=FakePho(), qwen3=FakeQwenClient(),
        performance_config=performance, batch_size=8, edge_pad=0.0,
        asr_workers=workers,
        asr_placement={"qwen3": 0, "whisper": 1, "phowhisper": 1})


def _wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


def test_async_transcripts_match_the_synchronous_paths():
    legacy = ASRService(
        whisper=FakeWhisper(), phowhisper=FakePho(), qwen3=FakeQwenClient(),
        performance_config={"cross_file": False}, batch_size=8, edge_pad=0.0,
    ).process(_segments(4), _audio(0.2))

    svc = _service()
    svc.begin_cross_file_stage(1)
    try:
        future = svc.process_async(_segments(4), _audio(0.2))
        svc.settle_file()
        got = future.result(10)
    finally:
        svc.end_cross_file_stage()
    assert len(got) == 4
    assert _summary(got) == _summary(legacy)


def test_async_is_off_unless_the_switch_and_a_cross_file_stage_are_on():
    svc = _service()
    assert svc.async_vote_enabled is False              # no stage open
    svc.begin_cross_file_stage(1)
    assert svc.async_vote_enabled is True
    svc.end_cross_file_stage()
    off = _service(async_vote=False)
    off.begin_cross_file_stage(1)
    assert off.async_vote_enabled is False
    off.end_cross_file_stage()


def test_a_very_slow_lane_does_not_stop_the_fast_lanes_taking_the_next_files():
    gate = threading.Event()
    svc = _service(whisper=GatedWhisper(gate))
    svc.begin_cross_file_stage(4)
    futures = []
    try:
        started = time.monotonic()
        for length in (2.0, 3.0, 4.0, 5.0):
            # The same thread submits all four: none of them waits for Whisper.
            futures.append(svc.process_async(_segments(2, length), _audio()))
            svc.settle_file()
        assert time.monotonic() - started < 5
        assert not any(f.done() for f in futures)
        lanes = svc._scheduler.lanes
        assert _wait_for(lambda: lanes["phowhisper"].done == 8 and lanes["qwen3"].done == 8)
        assert lanes["whisper"].done == 0 and not any(f.done() for f in futures)
        gate.set()
        results = [f.result(10) for f in futures]
    finally:
        gate.set()
        svc.end_cross_file_stage()
    for length, transcripts in zip((2.0, 3.0, 4.0, 5.0), results):
        tag = int(length * 16000)
        assert [t.index for t in transcripts] == ["00000", "00001"]
        assert all(t.text_whisper.startswith(f"w{tag} ") for t in transcripts)
        assert all(t.text_qwen3.startswith(f"q{tag} ") for t in transcripts)


def test_many_files_keep_every_segment_once_and_in_order():
    svc = _service()
    lengths = [2.0 + 0.5 * i for i in range(8)]
    svc.begin_cross_file_stage(len(lengths))
    try:
        futures = []
        for length in lengths:
            futures.append(svc.process_async(_segments(5, length), _audio()))
            svc.settle_file()
        outputs = [f.result(10) for f in futures]
    finally:
        svc.end_cross_file_stage()
    for length, transcripts in zip(lengths, outputs):
        assert [t.index for t in transcripts] == [f"{i:05d}" for i in range(5)]
        assert {t.text_whisper.split()[0] for t in transcripts} == {f"w{int(length * 16000)}"}


def test_results_are_keyed_by_file_segment_and_model():
    table = ASRService._keyed_results(
        "file-1", ["a", "b"], [("w1", "vi", []), ("w2", "vi", [])], ["p1", "p2"], ["q1", "q2"])
    assert table[("file-1", "b", "whisper")][0] == "w2"
    assert table[("file-1", "a", "phowhisper")] == "p1"
    assert table[("file-1", "b", "qwen3")] == "q2"
    assert len(table) == 6


def test_prepared_bytes_are_bounded_and_a_big_file_still_runs_alone():
    budget = ByteBudget(100)
    budget.acquire(60)
    entered = threading.Event()

    def second():
        budget.acquire(60)          # 120 > 100: waits for the first
        entered.set()

    thread = threading.Thread(target=second)
    thread.start()
    assert not entered.wait(0.2)
    budget.release(60)
    assert entered.wait(2)
    thread.join(2)
    budget.release(60)
    solo = ByteBudget(10)
    solo.acquire(1000)              # bigger than the cap, nothing else held: runs
    assert solo.used == 1000
    assert ByteBudget(0).acquire(10**9) is None


def test_intake_blocks_on_the_byte_budget_until_the_lanes_release_a_file():
    gate = threading.Event()
    # 30 s of audio is ~1.9 MB at 16 kHz float32 for the two 2-second clips: cap at 1 MB.
    svc = _service(whisper=GatedWhisper(gate), max_prepared_mb=1)
    svc.begin_cross_file_stage(2)
    submitted = []
    try:
        first = svc.process_async(_segments(2, 20.0), _audio())
        svc.settle_file()
        second_started = threading.Event()

        def later():
            second_started.set()
            submitted.append(svc.process_async(_segments(2, 20.0), _audio()))
            svc.settle_file()

        thread = threading.Thread(target=later)
        thread.start()
        assert second_started.wait(2)
        time.sleep(0.3)
        assert submitted == [], "the second file must wait: the budget is full"
        gate.set()
        thread.join(10)
        assert first.result(10) and submitted[0].result(10)
        peak = svc._budget.peak
    finally:
        gate.set()
        svc.end_cross_file_stage()
    assert peak <= 2 * 20 * 16000 * 4 * 2 // 2 + 1        # one file's clips, never two


def test_a_lane_that_cannot_start_fails_the_future_and_frees_the_budget():
    pool = FakePool(workers=1, ready_error="no GPU")
    svc = _service(workers={"phowhisper": pool})
    svc.begin_cross_file_stage(1)
    try:
        future = svc.process_async(_segments(3), _audio())
        svc.settle_file()
        with pytest.raises(RuntimeError, match="phowhisper.*no GPU"):
            future.result(10)
        assert svc._budget.used == 0
        assert svc._scheduler.progress.snapshot()["files"]["file-1"] == "failed"
    finally:
        svc.end_cross_file_stage()


def test_a_vote_error_fails_that_file_only():
    svc = _service()
    real = svc._vote
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("bad vote")
        return real(*args, **kwargs)

    svc._vote = flaky
    svc.begin_cross_file_stage(2)
    try:
        a = svc.process_async(_segments(2, 2.0), _audio())
        svc.settle_file()
        with pytest.raises(ValueError, match="bad vote"):
            a.result(10)
        b = svc.process_async(_segments(2, 3.0), _audio())
        svc.settle_file()
        assert len(b.result(10)) == 2
    finally:
        svc.end_cross_file_stage()


def test_a_file_with_no_usable_segment_resolves_empty_and_still_settles():
    svc = _service()
    svc.begin_cross_file_stage(1)
    try:
        assert svc.process_async([], _audio()).result(2) == []
        svc.settle_file()
        assert svc._scheduler is None or svc._scheduler.lanes
    finally:
        svc.end_cross_file_stage()


def test_outside_a_cross_file_stage_process_async_runs_the_synchronous_path():
    svc = _service()
    future = svc.process_async(_segments(2), _audio())
    assert len(future.result(10)) == 2


def test_ending_the_stage_stops_the_vote_pool():
    svc = _service()
    svc.begin_cross_file_stage(1)
    svc.process_async(_segments(2), _audio()).result(10)
    svc.settle_file()
    svc.end_cross_file_stage()
    assert not [t for t in threading.enumerate()
                if t.name.startswith(("asr-vote", "asr-")) and t.is_alive()]


# --- ticket ---------------------------------------------------------------------

def test_when_settled_fires_once_on_success_and_once_on_failure():
    ticket = FileTicket("a", {"x": 2})
    seen = []
    ticket.when_settled(seen.append)
    ticket._deliver("x", 0, 1)
    assert seen == []
    ticket._deliver("x", 1, 2)
    assert seen == [None]
    ticket._fail("x", "late")            # too late to matter
    assert seen == [None]

    failing = FileTicket("b", {"x": 1})
    errors = []
    failing.when_settled(errors.append)
    failing._fail("x", "no GPU")
    failing._deliver("x", 0, 1)          # a straggler must not call back again
    assert len(errors) == 1 and "no GPU" in str(errors[0])
    late = []
    failing.when_settled(late.append)
    assert len(late) == 1 and late[0] is errors[0]


# --- progress -------------------------------------------------------------------

def test_progress_reports_a_failed_file_only_when_there_is_one():
    progress = AsrProgress(None, interval=0)
    progress.expect_files(2)
    progress.queued("a", "qwen3", 1)
    progress.queued("b", "qwen3", 1)
    assert "failed" not in progress.render()[0]
    progress.voted("a")
    progress.failed("b")
    head = progress.render()[0]
    assert "voted 1/2" in head and "failed 1" in head


# --- pipeline hooks -------------------------------------------------------------

class _Ckpt:
    def __init__(self):
        self.saved = {}

    def save(self, name, value):
        self.saved[name] = value


class _Out:
    def __init__(self):
        self.calls = []

    def write_asr(self, transcripts):
        self.calls.append(("asr", len(transcripts)))

    def write_manifest(self, data):
        self.calls.append(("manifest", data["stopped_after"]))


def _pipeline(asr):
    from services.pipeline_service import PipelineService
    pipeline = PipelineService.__new__(PipelineService)
    pipeline.asr_svc = asr
    pipeline._pending_asr_jobs = {}
    pipeline._pending_asr_lock = threading.RLock()
    return pipeline


def test_the_commit_lands_before_the_drain_returns():
    from types import SimpleNamespace
    svc = _service()
    pipeline = _pipeline(svc)
    args = SimpleNamespace(stop_after="asr")
    ckpt, out = _Ckpt(), _Out()
    svc.begin_cross_file_stage(1)
    try:
        assert pipeline._asr_async_ok(args) is True
        assert pipeline._asr_async_ok(SimpleNamespace(stop_after=None)) is False
        pipeline._submit_asr_async("/x/a.wav", _segments(3), _audio(), ckpt, out)
        svc.settle_file()
        failures = {}
        _drain_pending_asr(pipeline, failures)
    finally:
        svc.end_cross_file_stage()
    assert failures == {} and pipeline._pending_asr_jobs == {}
    assert len(ckpt.saved["asr"]) == 3
    assert out.calls == [("asr", 3), ("manifest", "asr")]


def test_a_failed_file_is_reported_by_the_drain_not_swallowed():
    pool = FakePool(workers=1, ready_error="no GPU")
    svc = _service(workers={"phowhisper": pool})
    pipeline = _pipeline(svc)
    ckpt, out = _Ckpt(), _Out()
    svc.begin_cross_file_stage(1)
    try:
        pipeline._submit_asr_async("/x/a.wav", _segments(3), _audio(), ckpt, out)
        svc.settle_file()
        failures = {}
        _drain_pending_asr(pipeline, failures)
    finally:
        svc.end_cross_file_stage()
    assert "asr:" in failures["/x/a.wav"] and "no GPU" in failures["/x/a.wav"]
    assert ckpt.saved == {} and out.calls == []
