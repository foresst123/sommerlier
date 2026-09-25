"""Cross-file scheduler for the ASR models.

Each model is a *lane* with its own FIFO queue that every file in flight feeds,
so a model that is done with one file goes straight on to the next instead of
waiting at a per-file barrier for the slowest model. Once no more files will
arrive and a lane runs dry, its VRAM is released and the slowest remaining lane
is helped: a bigger batch when it shares the GPU, a replica on the freed GPU
when it does not.
"""

import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional


class LaneWorker:
    """One running consumer of a lane: a model instance on a GPU.

    ``run_batch(payloads, batch_size) -> list`` returns one result per payload;
    ``release()`` frees the model's VRAM. ``batch_size`` may be raised while the
    lane is running -- the next pull uses the new value.
    """

    def __init__(self, name, gpu, run_batch: Callable, batch_size: int,
                 release: Optional[Callable] = None):
        self.name = name
        self.gpu = gpu
        self.run_batch = run_batch
        self.batch_size = max(1, int(batch_size))
        self.release = release
        self.released = False


class Lane:
    """The jobs for one model, shared by every worker (primary and replicas)."""

    def __init__(self, name, gpu, primary: LaneWorker, empty,
                 replica_factory: Optional[Callable] = None, boostable: bool = True):
        self.name = name
        self.gpu = gpu
        self.primary = primary
        self.workers: List[LaneWorker] = [primary]
        self.empty = empty
        self.replica_factory = replica_factory
        self.boostable = boostable
        self.replica_started = False
        self.queue: deque = deque()
        self.inflight = 0
        self.done = 0
        self.total = 0
        self.busy_seconds = 0.0
        self.finished = False

    def remaining(self) -> int:
        return len(self.queue) + self.inflight

    def rate(self) -> float:
        """Jobs per second for the whole lane, measured on busy time only."""
        if self.done <= 0 or self.busy_seconds <= 0:
            return 0.0
        active = sum(1 for worker in self.workers if not worker.released) or 1
        return (self.done / self.busy_seconds) * active


class _Job:
    __slots__ = ("ticket", "position", "payload")

    def __init__(self, ticket, position, payload):
        self.ticket, self.position, self.payload = ticket, position, payload


class FileTicket:
    """What one file waits on: a result slot for every payload it submitted."""

    def __init__(self, file_id, counts: Dict[str, int]):
        self.file_id = file_id
        self._results = {lane: [None] * count for lane, count in counts.items()}
        self._pending = sum(counts.values())
        self._lock = threading.Lock()
        self._event = threading.Event()
        if self._pending == 0:
            self._event.set()

    def _deliver(self, lane, position, value):
        with self._lock:
            self._results[lane][position] = value
            self._pending -= 1
            if self._pending == 0:
                self._event.set()

    def wait(self, timeout=None) -> Dict[str, list]:
        if not self._event.wait(timeout):
            raise TimeoutError(f"ASR results for {self.file_id} did not arrive")
        return self._results


def _eta(lane) -> float:
    remaining = lane.remaining()
    if remaining <= 0:
        return 0.0
    rate = lane.rate()
    return remaining / rate if rate > 0 else float("inf")


def choose_rebalance(finished, remaining, cfg) -> tuple:
    """Decide how to help the slowest remaining lane after ``finished`` ended.

    Returns ("none",), ("boost", lane) or ("replica", lane, gpu). The slowest
    lane has the longest ETA; lanes not measured yet rank by their job count.
    """
    if not remaining:
        return ("none",)
    slowest = max(remaining, key=lambda lane: (_eta(lane), lane.remaining()))
    if slowest.gpu == finished.gpu:
        return ("boost", slowest) if slowest.boostable else ("none",)
    if slowest.replica_factory is None or slowest.replica_started:
        return ("none",)

    rate = slowest.rate()
    if rate <= 0:
        # Speed unknown: only start a replica when a lot of work is waiting.
        worth = slowest.remaining() >= int(cfg.get("replica_min_pending_jobs", 12))
    else:
        from services.asr_service import estimate_replica_gain
        peer = max((_eta(lane) for lane in remaining if lane is not slowest), default=0.0)
        gain, _, _ = estimate_replica_gain(
            slowest.remaining(), rate,
            float(cfg.get("replica_load_seconds", 30.0)),
            rate * float(cfg.get("replica_speed_ratio", 0.5)), peer)
        worth = gain > 0 and gain >= float(cfg.get("replica_min_gain_seconds", 15.0))
    return ("replica", slowest, finished.gpu) if worth else ("none",)


class AsrScheduler:
    def __init__(self, lanes: List[Lane], boost_batch: int = 48, config=None,
                 logger=None, monitor=None, log_interval: float = 30.0):
        self.lanes: Dict[str, Lane] = {lane.name: lane for lane in lanes}
        self.boost_batch = max(1, int(boost_batch))
        self.config = dict(config or {})
        self.logger = logger
        self.monitor = monitor
        self._log_interval = float(log_interval)
        self._cond = threading.Condition()
        self._expected: Optional[int] = None
        self._intake = 0
        self._closed = False
        self._started = False
        self._threads: List[threading.Thread] = []
        self._helpers: List[threading.Thread] = []
        self._stop_log = threading.Event()

    # -- intake ---------------------------------------------------------------
    def expect_files(self, total: int):
        with self._cond:
            self._expected = int(total)
            self._maybe_finish_locked()

    def add_intake(self, count: int = 1):
        """Count files that will submit nothing (checkpointed, failed, empty)."""
        with self._cond:
            self._intake += int(count)
            self._maybe_finish_locked()

    def submit(self, file_id, payloads_by_lane: Dict[str, list]) -> FileTicket:
        counts = {name: len(items) for name, items in payloads_by_lane.items()
                  if name in self.lanes}
        ticket = FileTicket(file_id, counts)
        with self._cond:
            if self._closed:
                raise RuntimeError("the ASR scheduler has been shut down")
            self._start_locked()
            for name, items in payloads_by_lane.items():
                lane = self.lanes.get(name)
                if lane is None:
                    continue
                for position, payload in enumerate(items):
                    lane.queue.append(_Job(ticket, position, payload))
                lane.total += len(items)
            self._intake += 1
            self._maybe_finish_locked()
            self._cond.notify_all()
        return ticket

    # -- workers --------------------------------------------------------------
    def _start_locked(self):
        if self._started:
            return
        self._started = True
        for lane in self.lanes.values():
            for worker in lane.workers:
                self._spawn_locked(lane, worker)
        if self._log_interval > 0:
            thread = threading.Thread(target=self._progress_loop, daemon=True,
                                      name="asr-progress")
            thread.start()
            self._helpers.append(thread)

    def _spawn_locked(self, lane, worker):
        thread = threading.Thread(target=self._run_worker, args=(lane, worker),
                                  daemon=True, name=f"asr-{lane.name}-{worker.gpu}")
        thread.start()
        self._threads.append(thread)

    def _run_worker(self, lane: Lane, worker: LaneWorker):
        while True:
            with self._cond:
                while not lane.queue:
                    if self._closed or lane.finished:
                        return
                    self._cond.wait()
                size = worker.batch_size
                jobs = [lane.queue.popleft()
                        for _ in range(min(size, len(lane.queue)))]
                lane.inflight += len(jobs)
            began = time.monotonic()
            try:
                results = list(worker.run_batch([job.payload for job in jobs], size))
            except Exception as exc:
                self._log("error", f"[ASR scheduler] {lane.name} batch failed "
                                   f"({type(exc).__name__}: {exc}); leaving it empty")
                results = []
            if len(results) < len(jobs):
                results.extend([lane.empty] * (len(jobs) - len(results)))
            for job, value in zip(jobs, results):
                job.ticket._deliver(lane.name, job.position, value)
            elapsed = time.monotonic() - began
            with self._cond:
                lane.inflight -= len(jobs)
                lane.done += len(jobs)
                lane.busy_seconds += elapsed
                self._maybe_finish_locked()
                self._cond.notify_all()

    # -- finishing and rebalancing ---------------------------------------------
    def _maybe_finish_locked(self):
        if self._expected is None or self._intake < self._expected:
            return
        for lane in self.lanes.values():
            if not lane.finished and not lane.queue and lane.inflight == 0:
                lane.finished = True
                thread = threading.Thread(target=self._rebalance, args=(lane,),
                                          daemon=True, name=f"asr-rebalance-{lane.name}")
                thread.start()
                self._helpers.append(thread)
                self._cond.notify_all()

    def _release_workers(self, lane: Lane, workers):
        for worker in list(workers):
            if worker.released:
                continue
            worker.released = True
            if worker.release is None:
                continue
            try:
                worker.release()
            except Exception as exc:
                self._log("warning", f"[ASR scheduler] releasing {lane.name} failed: {exc}")

    def _rebalance(self, finished: Lane):
        try:
            self._log("info", f"[ASR scheduler] {finished.name} finished "
                              f"({finished.done} jobs); releasing its VRAM")
            self._record("asr_lane_finished", lane=finished.name, jobs=finished.done)
            self._release_workers(finished, finished.workers)
            with self._cond:
                remaining = [lane for lane in self.lanes.values()
                             if not lane.finished and lane.remaining() > 0]
                action = choose_rebalance(finished, remaining, self.config)
            if action[0] == "boost":
                self._boost(action[1])
            elif action[0] == "replica":
                self._start_replica(action[1], action[2])
        except Exception as exc:
            self._log("warning", f"[ASR scheduler] rebalance after {finished.name} "
                                 f"failed: {type(exc).__name__}: {exc}")

    def _boost(self, lane: Lane):
        with self._cond:
            for worker in lane.workers:
                worker.batch_size = max(worker.batch_size, self.boost_batch)
        self._log("info", f"[ASR scheduler] boosting {lane.name} batch to "
                          f"{self.boost_batch} (GPU {lane.gpu} is now free of its peer)")
        self._record("asr_batch_boosted", lane=lane.name, batch=self.boost_batch)

    def _start_replica(self, lane: Lane, gpu):
        self._log("info", f"[ASR scheduler] starting a {lane.name} replica on GPU "
                          f"{gpu} (batch {self.boost_batch}); {lane.remaining()} job(s) left")
        self._record("asr_replica_loading", lane=lane.name, gpu=gpu)
        try:
            worker = lane.replica_factory(gpu, self.boost_batch)
        except Exception as exc:
            self._log("warning", f"[ASR scheduler] {lane.name} replica could not start: "
                                 f"{type(exc).__name__}: {exc}")
            self._record("asr_replica_failed", lane=lane.name, error=str(exc))
            return
        with self._cond:
            if lane.finished or self._closed:
                too_late = True
            else:
                too_late = False
                lane.workers.append(worker)
                lane.replica_started = True
                self._spawn_locked(lane, worker)
                self._cond.notify_all()
        if too_late:
            self._release_workers(lane, [worker])
        else:
            self._record("asr_replica_started", lane=lane.name, gpu=gpu)

    # -- reporting and shutdown ---------------------------------------------------
    def _progress_loop(self):
        while not self._stop_log.wait(self._log_interval):
            with self._cond:
                parts = [f"{lane.name} {lane.done}/{lane.total} "
                         f"({len([w for w in lane.workers if not w.released])}w)"
                         for lane in self.lanes.values()]
            self._log("info", "[ASR scheduler] " + " | ".join(parts))

    def _log(self, level, message):
        if self.logger:
            getattr(self.logger, level)(message)

    def _record(self, event, **fields):
        if self.monitor:
            self.monitor.record(event, **fields)

    def shutdown(self):
        with self._cond:
            self._closed = True
            for lane in self.lanes.values():
                while lane.queue:
                    job = lane.queue.popleft()
                    job.ticket._deliver(lane.name, job.position, lane.empty)
            self._cond.notify_all()
            threads = list(self._threads) + list(self._helpers)
        self._stop_log.set()
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=30)
        for lane in self.lanes.values():
            self._release_workers(lane, lane.workers[1:])
