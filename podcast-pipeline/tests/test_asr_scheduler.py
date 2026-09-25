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
