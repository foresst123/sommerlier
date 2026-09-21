import itertools
import queue
import threading
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

    @property
    def process(self):
        """Compatibility handle for code paths that use only one process."""
        return self.services[0].process

    @property
    def processes(self):
        return [service.process for service in self.services
                if service.process is not None]

    def spawn(self):
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
        service = self._available.get()
        try:
            if service.process is None:
                service.start()
            return service.request(payload, response_id=response_id)
        finally:
            self._available.put(service)
