import itertools
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class WorkerPoolService:
    """Lifecycle and dispatch for independent line-JSON worker processes."""

    def __init__(self, services, name=None):
        self.services = list(services)
        if not self.services:
            raise ValueError("worker pool requires at least one service")
        self.name = name or self.services[0].name
        self._counter = itertools.count()
        self._pick_lock = threading.Lock()
        self._available = queue.Queue()
        for service in self.services:
            self._available.put(service)
        self._stats_lock = threading.Lock()
        self._reset_stats()

    def _reset_stats(self):
        with self._stats_lock:
            self._busy = [0.0] * len(self.services)
            self._calls = [0] * len(self.services)
            self._lease_wait = 0.0
            self._first_start = None
            self._last_end = None

    def profile(self):
        """How busy each worker was between the first and the last request."""
        with self._stats_lock:
            wall = ((self._last_end - self._first_start)
                    if self._first_start is not None else 0.0)
            return {"wall_seconds": wall, "calls": sum(self._calls),
                    "lease_wait_seconds": self._lease_wait,
                    "workers": [{"busy_seconds": b, "calls": c}
                                for b, c in zip(self._busy, self._calls)]}

    @property
    def process(self):
        """Compatibility handle for code paths that use only one process."""
        return self.services[0].process

    @property
    def processes(self):
        return [service.process for service in self.services
                if service.process is not None]

    def spawn(self):
        self._reset_stats()
        for service in self.services:
            service.spawn()

    def wait_ready(self):
        # Models load concurrently after spawn(); only readiness waits here.
        with ThreadPoolExecutor(max_workers=len(self.services)) as executor:
            futures = [executor.submit(service.wait_ready)
                       for service in self.services]
            for future in futures:
                future.result()

    def stop(self):
        for service in self.services:
            service.stop()

    def next_service(self):
        with self._pick_lock:
            service = self.services[next(self._counter) % len(self.services)]
        if service.process is None:
            service.start()
        return service

    def request(self, payload, *, response_id=None):
        # Lease an idle process instead of blindly selecting round-robin. With
        # more callers than workers, the next request waits for whichever GPU
        # finishes first rather than queueing behind a still-busy process.
        asked = time.perf_counter()
        service = self._available.get()
        got = time.perf_counter()
        try:
            if service.process is None:
                service.start()
            return service.request(payload, response_id=response_id)
        finally:
            done = time.perf_counter()
            index = self.services.index(service)
            with self._stats_lock:
                self._lease_wait += got - asked
                self._busy[index] += done - got
                self._calls[index] += 1
                if self._first_start is None:
                    self._first_start = got
                self._last_end = done
            self._available.put(service)
