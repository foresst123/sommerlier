"""ASR models consume a queue that spans files; finished models are released and
the slowest remaining one is helped (bigger batch on the same GPU, replica on the
freed GPU). Everything here uses fake model callables -- no GPU."""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.asr_scheduler import AsrScheduler, Lane, LaneWorker, choose_rebalance


def wait_until(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _tag(name):
    return lambda payloads, size: [f"{name}:{p}" for p in payloads]


def _scheduler(lanes, **kwargs):
    kwargs.setdefault("boost_batch", 48)
    kwargs.setdefault("log_interval", 0)
    return AsrScheduler(lanes, **kwargs)


def _lane(name, gpu, run_batch=None, batch=16, release=None, **kwargs):
    worker = LaneWorker(name, gpu, run_batch or _tag(name), batch, release=release)
    return Lane(name, gpu, worker, empty="", **kwargs)


# --- core: queues, batching, tickets ------------------------------------------

def test_a_fast_model_moves_on_to_the_next_file_without_waiting_for_a_slow_one():
    stamps = {"fast": [], "slow": []}

    def fast(payloads, size):
        stamps["fast"].append(time.monotonic())
        return [f"f:{p}" for p in payloads]

    def slow(payloads, size):
        time.sleep(0.3)
        stamps["slow"].append(time.monotonic())
        return [f"s:{p}" for p in payloads]

    sched = _scheduler([_lane("fast", 0, fast, batch=2), _lane("slow", 1, slow, batch=2)])
    sched.expect_files(2)
    first = sched.submit("a", {"fast": [1, 2], "slow": [1, 2]})
    second = sched.submit("b", {"fast": [3, 4], "slow": [3, 4]})
    try:
        r_second, r_first = second.wait(5), first.wait(5)
    finally:
        sched.shutdown()

    assert stamps["fast"][-1] < stamps["slow"][0]      # no per-file barrier
    assert r_first["slow"] == ["s:1", "s:2"] and r_second["fast"] == ["f:3", "f:4"]


def test_results_keep_submission_order_and_batches_respect_the_batch_size():
    sizes = []

    def record(payloads, size):
        sizes.append((len(payloads), size))
        return [p * 10 for p in payloads]

    sched = _scheduler([_lane("m", 0, record, batch=3)])
    sched.expect_files(2)
    a = sched.submit("a", {"m": [1, 2, 3, 4, 5]})
    b = sched.submit("b", {"m": [6, 7]})
    try:
        assert a.wait(5)["m"] == [10, 20, 30, 40, 50]
        assert b.wait(5)["m"] == [60, 70]
    finally:
        sched.shutdown()
    assert all(count <= size == 3 for count, size in sizes)
    assert sum(count for count, _ in sizes) == 7


def test_a_failing_batch_gives_empty_results_and_still_completes_the_ticket():
    def boom(payloads, size):
        raise RuntimeError("model died")

    sched = _scheduler([_lane("m", 0, boom, batch=4), _lane("ok", 1)])
    sched.expect_files(1)
    ticket = sched.submit("a", {"m": [1, 2], "ok": [1]})
    try:
        results = ticket.wait(5)
    finally:
        sched.shutdown()
    assert results["m"] == ["", ""] and results["ok"] == ["ok:1"]


def test_a_wrong_length_result_is_padded_with_the_empty_value():
    sched = _scheduler([_lane("m", 0, lambda payloads, size: ["only-one"], batch=4)])
    sched.expect_files(1)
    ticket = sched.submit("a", {"m": [1, 2, 3]})
    try:
        assert ticket.wait(5)["m"] == ["only-one", "", ""]
    finally:
        sched.shutdown()


def test_shutdown_stops_the_workers_and_rejects_new_work():
    sched = _scheduler([_lane("m", 0)])
    sched.expect_files(1)
    sched.submit("a", {"m": [1]}).wait(5)
    sched.shutdown()
    with pytest.raises(RuntimeError):
        sched.submit("b", {"m": [1]})


# --- rebalancing -------------------------------------------------------------

def test_a_finished_model_is_released_and_the_same_gpu_peer_gets_the_boosted_batch():
    released, gate = [], threading.Event()

    def slow(payloads, size):
        gate.wait(5)
        return [f"s:{p}" for p in payloads]

    fast = _lane("fast", 1, release=lambda: released.append("fast"), batch=16)
    slow_lane = _lane("slow", 1, slow, batch=16)
    sched = _scheduler([fast, slow_lane], boost_batch=48)
    sched.expect_files(1)
    ticket = sched.submit("a", {"fast": [1, 2], "slow": list(range(40))})
    try:
        assert wait_until(lambda: released == ["fast"])
        assert wait_until(lambda: slow_lane.primary.batch_size == 48)
        assert fast.primary.batch_size == 16
        gate.set()
        assert len(ticket.wait(5)["slow"]) == 40
    finally:
        gate.set()
        sched.shutdown()


def test_a_model_on_another_gpu_gets_a_replica_on_the_freed_gpu_at_the_boost_batch():
    gate, made, released = threading.Event(), [], []
    primary_started, fast_gate = threading.Event(), threading.Event()

    def slow_primary(payloads, size):
        primary_started.set()
        gate.wait(5)
        return [f"primary:{p}" for p in payloads]

    def fast_run(payloads, size):
        fast_gate.wait(5)           # finishes only after the primary has its batch
        return [f"fast:{p}" for p in payloads]

    def factory(gpu, batch):
        made.append((gpu, batch))
        return LaneWorker("slow", gpu, _tag("replica"), batch,
                          release=lambda: released.append("replica"))

    fast = _lane("fast", 0, fast_run, release=lambda: released.append("fast"))
    slow_lane = _lane("slow", 1, slow_primary, batch=4, replica_factory=factory)
    sched = _scheduler([fast, slow_lane], boost_batch=48)
    sched.expect_files(1)
    ticket = sched.submit("a", {"fast": [1], "slow": list(range(30))})
    try:
        assert primary_started.wait(3)
        fast_gate.set()
        assert wait_until(lambda: made == [(0, 48)])
        assert wait_until(lambda: "replica" not in released and slow_lane.done >= 26)
        gate.set()
        results = ticket.wait(5)["slow"]
    finally:
        gate.set()
        fast_gate.set()
        sched.shutdown()

    assert len(results) == 30
    assert sum(1 for r in results if r.startswith("replica:")) >= 26
    assert results[:4] == [f"primary:{i}" for i in range(4)]
    assert released.count("replica") == 1 and "fast" in released


def test_a_small_backlog_is_not_worth_a_replica_but_the_finished_model_is_still_released():
    gate, made, released = threading.Event(), [], []

    def slow(payloads, size):
        gate.wait(5)
        return [f"s:{p}" for p in payloads]

    fast = _lane("fast", 0, release=lambda: released.append("fast"))
    slow_lane = _lane("slow", 1, slow, batch=4,
                      replica_factory=lambda gpu, batch: made.append((gpu, batch)))
    sched = _scheduler([fast, slow_lane], config={"replica_min_pending_jobs": 12})
    sched.expect_files(1)
    ticket = sched.submit("a", {"fast": [1], "slow": [1, 2, 3, 4, 5, 6]})
    try:
        assert wait_until(lambda: released == ["fast"])
        time.sleep(0.1)
        assert made == []
        gate.set()
        ticket.wait(5)
    finally:
        gate.set()
        sched.shutdown()


def test_nothing_is_released_or_rebalanced_while_more_files_may_still_arrive():
    released = []
    fast = _lane("fast", 0, release=lambda: released.append("fast"))
    slow_lane = _lane("slow", 1)
    sched = _scheduler([fast, slow_lane])
    sched.expect_files(2)
    first = sched.submit("a", {"fast": [1], "slow": [1]})
    try:
        first.wait(5)
        time.sleep(0.15)
        assert released == []                # a second file may still come
        sched.add_intake(1)                  # ...and now it will not
        assert wait_until(lambda: released == ["fast"])
    finally:
        sched.shutdown()


def test_a_replica_that_cannot_start_leaves_the_run_on_the_remaining_workers():
    gate = threading.Event()

    def slow(payloads, size):
        gate.wait(5)
        return [f"s:{p}" for p in payloads]

    def broken_factory(gpu, batch):
        raise RuntimeError("no VRAM")

    fast = _lane("fast", 0)
    slow_lane = _lane("slow", 1, slow, batch=4, replica_factory=broken_factory)
    sched = _scheduler([fast, slow_lane])
    sched.expect_files(1)
    ticket = sched.submit("a", {"fast": [1], "slow": list(range(30))})
    try:
        time.sleep(0.15)
        gate.set()
        assert len(ticket.wait(5)["slow"]) == 30
    finally:
        gate.set()
        sched.shutdown()


# --- the decision, on its own -------------------------------------------------

class _Stub:
    """Just the fields choose_rebalance reads."""

    def __init__(self, name, gpu, remaining, rate=0.0, boostable=True, factory=None):
        self.name, self.gpu = name, gpu
        self._remaining, self._rate = remaining, rate
        self.boostable, self.replica_factory = boostable, factory
        self.replica_started = False

    def remaining(self):
        return self._remaining

    def rate(self):
        return self._rate

    def serving(self):
        return True


CFG = {"replica_load_seconds": 30.0, "replica_speed_ratio": 0.5,
       "replica_min_gain_seconds": 15.0, "replica_min_pending_jobs": 12}


def test_no_remaining_work_means_no_action():
    assert choose_rebalance(_Stub("a", 0, 0), [], CFG) == ("none",)


def test_the_slowest_remaining_model_is_the_one_with_the_longest_eta():
    finished = _Stub("done", 1, 0)
    quick = _Stub("quick", 1, 100, rate=100.0)       # 1 s left
    tail = _Stub("tail", 1, 100, rate=1.0)           # 100 s left
    assert choose_rebalance(finished, [quick, tail], CFG) == ("boost", tail)


def test_a_same_gpu_model_that_cannot_boost_is_left_alone():
    finished = _Stub("done", 1, 0)
    assert choose_rebalance(finished, [_Stub("peer", 1, 50, boostable=False)], CFG) == ("none",)


def test_a_replica_is_chosen_only_when_the_forecast_gain_clears_the_threshold():
    finished = _Stub("done", 0, 0)
    big = _Stub("big", 1, 6000, rate=1.0, factory=lambda gpu, batch: None)
    assert choose_rebalance(finished, [big], CFG) == ("replica", big, 0)
    small = _Stub("small", 1, 20, rate=1.0, factory=lambda gpu, batch: None)
    assert choose_rebalance(finished, [small], CFG) == ("none",)


def test_an_unmeasured_model_gets_a_replica_only_with_enough_jobs_left():
    finished = _Stub("done", 0, 0)
    many = _Stub("many", 1, 30, factory=lambda gpu, batch: None)
    few = _Stub("few", 1, 5, factory=lambda gpu, batch: None)
    assert choose_rebalance(finished, [many], CFG) == ("replica", many, 0)
    assert choose_rebalance(finished, [few], CFG) == ("none",)


# --- progress and completion hooks ------------------------------------------------

def test_a_ticket_reports_completion_once_every_lane_has_delivered():
    from services.asr_scheduler import FileTicket
    ticket = FileTicket("a", {"x": 2, "y": 1})
    calls = []
    ticket.when_complete(lambda: calls.append("done"))

    ticket._deliver("x", 0, 1)
    ticket._deliver("x", 1, 2)
    assert calls == []
    ticket._deliver("y", 0, 3)
    assert calls == ["done"]


def test_a_callback_registered_after_completion_runs_at_once():
    from services.asr_scheduler import FileTicket
    ticket = FileTicket("a", {"x": 1})
    ticket._deliver("x", 0, 1)
    calls = []
    ticket.when_complete(lambda: calls.append(1))
    assert calls == [1]


def test_the_scheduler_feeds_the_progress_report_and_writes_a_final_line():
    import logging
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("sched-progress-test")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(Handler())

    sched = _scheduler([_lane("m", 0, batch=2), _lane("n", 1, batch=2)], logger=logger)
    sched.expect_files(2)
    a = sched.submit("a", {"m": [1, 2, 3], "n": [1, 2, 3]})
    b = sched.submit("b", {"m": [4], "n": [4]})
    a.wait(5), b.wait(5)
    sched.shutdown()

    snap = sched.progress.snapshot()
    assert snap["lanes"]["m"]["done"] == snap["lanes"]["m"]["total"] == 4
    assert snap["lanes"]["n"]["done"] == snap["lanes"]["n"]["total"] == 4
    assert set(snap["files"].values()) == {"lanes_done"}
    final = [m for m in records if m.startswith("[ASR]")][-2:]
    assert "files 2/2 queued" in final[0] and "m 4/4" in final[1] and "n 4/4" in final[1]
    assert all("\r" not in m for m in records)


# --- lanes that wait for their own worker to be ready --------------------------------

def _gated_lane(name, gpu, gate, batch=2, error=None, **kwargs):
    def ready():
        gate.wait(5)
        if error:
            raise RuntimeError(error)

    worker = LaneWorker(name, gpu, _tag(name), batch)
    worker.ready = ready
    return Lane(name, gpu, worker, empty="", **kwargs)


def test_a_lane_whose_worker_is_ready_runs_while_another_is_still_loading():
    open_gate, closed_gate = threading.Event(), threading.Event()
    open_gate.set()
    sched = _scheduler([_gated_lane("fast", 0, open_gate),
                        _gated_lane("slow", 1, closed_gate)])
    sched.expect_files(1)
    ticket = sched.submit("a", {"fast": [1, 2], "slow": [1, 2]})
    try:
        assert wait_until(lambda: sched.progress.snapshot()["lanes"]["fast"]["done"] == 2)
        assert sched.progress.snapshot()["lanes"]["slow"]["done"] == 0
        assert sched.progress.snapshot()["lanes"]["slow"]["state"] == "loading"
        closed_gate.set()
        results = ticket.wait(5)
    finally:
        closed_gate.set()
        sched.shutdown()

    assert results["fast"] == ["fast:1", "fast:2"] and results["slow"] == ["slow:1", "slow:2"]
    assert sched.progress.snapshot()["lanes"]["slow"]["state"] != "loading"


def test_a_worker_that_cannot_start_fails_the_files_that_need_it_loudly():
    gate = threading.Event()
    gate.set()
    sched = _scheduler([_gated_lane("m", 0, gate, error="model missing")])
    sched.expect_files(2)
    first = sched.submit("a", {"m": [1, 2]})
    try:
        with pytest.raises(RuntimeError, match="m.*model missing"):
            first.wait(5)
        late = sched.submit("b", {"m": [3]})          # the lane is dead: fail at once
        with pytest.raises(RuntimeError, match="model missing"):
            late.wait(5)
    finally:
        sched.shutdown()


def test_a_lane_keeps_going_when_only_one_of_its_workers_fails_to_start():
    gate = threading.Event()
    gate.set()
    good = LaneWorker("m", 0, _tag("m"), 2)
    good.ready = gate.wait
    bad = LaneWorker("m", 1, _tag("m"), 2)

    def broken():
        raise RuntimeError("gpu 1 is full")

    bad.ready = broken
    lane = Lane("m", 0, good, empty="")
    lane.workers.append(bad)
    sched = _scheduler([lane])
    sched.expect_files(1)
    ticket = sched.submit("a", {"m": [1, 2, 3, 4]})
    try:
        assert ticket.wait(5)["m"] == ["m:1", "m:2", "m:3", "m:4"]
    finally:
        sched.shutdown()


# --- a replica must not be opened for a lane that has not started, or outlive the stage ---

def test_no_replica_is_opened_for_a_lane_whose_own_worker_is_still_loading():
    gate = threading.Event()               # never opened: the primary is "loading"
    loading = _gated_lane("q", 0, gate, replica_factory=lambda gpu, batch: None)
    finished = _lane("done", 1)
    loading.queue.extend(range(60))        # a lot of work waiting, speed unknown

    action = choose_rebalance(finished, [loading], {"replica_min_pending_jobs": 12})

    assert action == ("none",)


def test_a_lane_that_is_up_can_still_get_a_replica():
    ready = threading.Event()
    ready.set()
    lane = _gated_lane("q", 0, ready, replica_factory=lambda gpu, batch: None)
    lane.primary.became_ready = True
    lane.queue.extend(range(60))
    action = choose_rebalance(_lane("done", 1), [lane], {"replica_min_pending_jobs": 12})
    assert action[0] == "replica"


def test_shutdown_waits_for_a_replica_that_is_still_starting_and_releases_it():
    started, released = threading.Event(), []

    def slow_factory(gpu, batch):
        started.set()
        time.sleep(0.4)                    # loading a model
        worker = LaneWorker("m", gpu, _tag("m"), batch, release=lambda: released.append(1))
        return worker

    def slow_batch(payloads, size):
        time.sleep(0.05)                   # the lane that is still busy when `f` ends
        return [f"m:{p}" for p in payloads]

    lane = _lane("m", 0, slow_batch, batch=1, replica_factory=slow_factory)
    finisher = _lane("f", 1, batch=1)
    sched = _scheduler([lane, finisher])
    sched.expect_files(1)
    ticket = sched.submit("a", {"f": [1], "m": list(range(80))})
    # The `f` lane finishes at once and (speed unknown, plenty waiting) opens a replica.
    assert started.wait(3)
    sched.shutdown()

    assert released == [1], "the stage ended while a replica was still loading onto a GPU"


# --- a model that runs on more than one GPU ---------------------------------------

def test_a_lane_knows_every_gpu_it_runs_on():
    assert _lane("x", 1).gpus == {1}
    assert _lane("x", 1, gpus={1, 0}).gpus == {0, 1}


def test_a_model_that_also_runs_on_the_freed_gpu_is_boosted_and_gets_no_replica():
    finished = _Stub("qwen3", 0, 0)
    finished.gpus = {0}
    pho = _Stub("phowhisper", 1, 6000, rate=1.0, factory=lambda gpu, batch: None)
    pho.gpus = {1, 0}                 # one process per card; GPU 0 was just freed
    assert choose_rebalance(finished, [pho], CFG) == ("boost", pho)


def test_a_model_on_none_of_the_freed_gpus_still_gets_a_replica_there():
    finished = _Stub("qwen3", 0, 0)
    finished.gpus = {0}
    pho = _Stub("phowhisper", 1, 6000, rate=1.0, factory=lambda gpu, batch: None)
    pho.gpus = {1}
    assert choose_rebalance(finished, [pho], CFG) == ("replica", pho, 0)


def test_a_two_gpu_model_gets_the_bigger_batch_and_no_extra_process_when_a_gpu_frees():
    started, gate = [], threading.Event()

    def slow(payloads, size):
        gate.wait(5)
        return [f"s:{p}" for p in payloads]

    fast = _lane("fast", 0, batch=16)                                 # finishes on GPU 0
    slow_lane = _lane("slow", 1, slow, batch=16, gpus={1, 0},
                      replica_factory=lambda gpu, batch: started.append((gpu, batch)))
    sched = _scheduler([fast, slow_lane], boost_batch=48)
    sched.expect_files(1)
    ticket = sched.submit("a", {"fast": [1, 2], "slow": list(range(40))})
    try:
        assert wait_until(lambda: slow_lane.primary.batch_size == 48)
        assert started == [] and not slow_lane.replica_started
        gate.set()
        assert len(ticket.wait(5)["slow"]) == 40
    finally:
        gate.set()
        sched.shutdown()
