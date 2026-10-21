"""Progress of the ASR stage, reported from one place.

Progress used to come from several writers -- a `\\r` line per file, a tqdm bar per
file, and a scheduler line whose total grew as files arrived -- so with files in
flight the console showed them written over each other. Here the state lives in one
object and one reporter writes it as ordinary log records, so nothing is overwritten
and every number only moves forward.

A lane's total is the number of clips queued so far. While files are still to arrive
it is marked with `+`: it is a floor, not the final count.
"""

import threading
import time


class AsrProgress:
    def __init__(self, logger=None, interval: float = 15.0, clock=time.monotonic,
                 label=None):
        self.logger = logger
        self._prefix = "[ASR]" + (f" {label}" if label else "")
        self.interval = float(interval)
        self._clock = clock
        self._lock = threading.Lock()
        self._expected = None
        self._files = {}          # file_id -> "queued" | "lanes_done" | "voting" | "voted" | "failed"
        self._lanes = {}          # lane -> counters
        self._last_signature = None
        self._stop = threading.Event()
        self._thread = None

    # -- state ------------------------------------------------------------------
    def expect_files(self, total):
        with self._lock:
            self._expected = int(total)

    def _lane(self, name):
        lane = self._lanes.get(name)
        if lane is None:
            lane = self._lanes[name] = {"total": 0, "done": 0, "state": "waiting",
                                        "since": None, "started_at": None}
        return lane

    def queued(self, file_id, lane, count):
        with self._lock:
            self._files.setdefault(file_id, "queued")
            self._lane(lane)["total"] += int(count)

    def done(self, lane, count=1):
        with self._lock:
            entry = self._lanes.get(lane)
            if entry is not None:
                entry["done"] = min(entry["total"], entry["done"] + int(count))

    def lane_loading(self, lane):
        with self._lock:
            entry = self._lane(lane)
            entry["state"], entry["since"] = "loading", self._clock()

    def lane_ready(self, lane):
        with self._lock:
            entry = self._lane(lane)
            if entry["state"] == "loading":
                entry["state"] = "waiting"

    def lane_started(self, lane):
        with self._lock:
            entry = self._lane(lane)
            if entry["started_at"] is None:
                entry["started_at"] = self._clock()
            entry["state"] = "running"

    def lane_finished(self, lane):
        with self._lock:
            entry = self._lane(lane)
            if entry["state"] != "failed":
                entry["state"] = "finished"
            if entry.get("finished_at") is None:
                entry["finished_at"] = self._clock()

    def lane_failed(self, lane):
        with self._lock:
            self._lane(lane)["state"] = "failed"

    def _set_file(self, file_id, stage):
        with self._lock:
            self._files[file_id] = stage

    def lanes_done(self, file_id):
        self._set_file(file_id, "lanes_done")

    def voting(self, file_id):
        self._set_file(file_id, "voting")

    def voted(self, file_id):
        self._set_file(file_id, "voted")

    def failed(self, file_id):
        """The file ended without a vote (a lane it needs never came up, or the vote broke)."""
        self._set_file(file_id, "failed")

    def snapshot(self):
        with self._lock:
            return {"expected": self._expected, "files": dict(self._files),
                    "lanes": {name: {key: lane[key] for key in ("total", "done", "state")}
                              for name, lane in self._lanes.items()}}

    # -- text -------------------------------------------------------------------
    def render(self):
        with self._lock:
            now = self._clock()
            queued = len(self._files)
            stages = list(self._files.values())
            growing = self._expected is not None and queued < self._expected
            expected = self._expected if self._expected is not None else queued
            waiting = stages.count("queued") + stages.count("lanes_done")
            voting = stages.count("voting")
            voted = stages.count("voted")
            head = (f"{self._prefix} files {queued}/{expected} queued | waiting for lanes "
                    f"{waiting} | voting {voting} | voted {voted}/{expected}")
            failed = stages.count("failed")
            if failed:
                head += f" | failed {failed}"
            parts = [self._lane_text(name, lane, now, growing)
                     for name, lane in self._lanes.items()]
        return [head, f"{self._prefix} " + (" | ".join(parts) if parts else "no models yet")]

    @staticmethod
    def _lane_text(name, lane, now, growing):
        total, done = lane["total"], lane["done"]
        counts = f"{done}/{total}" + ("+" if growing else "")
        if lane["state"] == "loading":
            return f"{name} loading {int(now - lane['since'])}s ({counts})"
        details = []
        if total and not growing:
            details.append(f"{100 * done // total}%")
        elif total:
            details.append(f"{100 * done // total}% so far")
        started = lane["started_at"]
        # A finished lane's rate is frozen at the moment it finished; measured against
        # the clock it would keep falling for as long as the stage goes on.
        until = lane.get("finished_at") or now
        if started is not None and done > 0 and until > started:
            rate = done / (until - started)
            details.append(f"{rate:.1f}/s")
            if not growing and done < total:
                details.append(f"~{int((total - done) / rate)}s")
        tail = {"finished": " finished", "failed": " FAILED"}.get(lane["state"], "")
        return f"{name} {counts}" + (f" ({', '.join(details)})" if details else "") + tail

    # -- reporting ------------------------------------------------------------------
    def _signature(self):
        with self._lock:
            return (self._expected, tuple(sorted(self._files.items())),
                    tuple((name, lane["total"], lane["done"], lane["state"])
                          for name, lane in self._lanes.items()))

    def report(self, force=False):
        """Write the current state, unless nothing moved since the last report.

        A lane that is still loading is reported every time so its seconds tick.
        """
        signature = self._signature()
        loading = any(state == "loading" for _n, _t, _d, state in signature[2])
        if not force and not loading and signature == self._last_signature:
            return
        self._last_signature = signature
        if self.logger:
            for line in self.render():
                self.logger.info(line)

    def start(self):
        if self.interval <= 0 or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="asr-progress")
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.report()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.report(force=True)
