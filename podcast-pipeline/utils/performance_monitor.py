"""Low-overhead resource and scheduler telemetry for long pipeline runs."""

import json
import os
import resource
import threading
import time
from collections import defaultdict


class PerformanceMonitor:
    def __init__(self, output_dir, interval_seconds=1.0, enabled=True,
                 logger=None):
        self.enabled = bool(enabled)
        self.output_dir = output_dir
        self.interval = max(0.25, float(interval_seconds))
        self.logger = logger
        self.started_at = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._event_file = None
        self._sample_file = None
        self._counts = defaultdict(int)
        self._stage_seconds = defaultdict(float)
        self._peak_rss_mb = 0.0
        self._min_gpu_free = {}

    def start(self):
        if not self.enabled or self._thread is not None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self._event_file = open(os.path.join(
            self.output_dir, "scheduler_events.jsonl"), "a", encoding="utf-8")
        self._sample_file = open(os.path.join(
            self.output_dir, "resource_samples.jsonl"), "a", encoding="utf-8")
        self.started_at = time.time()
        self.record("monitor_started", pid=os.getpid())
        self._thread = threading.Thread(
            target=self._sample_loop, name="performance-monitor", daemon=True)
        self._thread.start()

    def record(self, event, **fields):
        if not self.enabled or self._event_file is None:
            return
        payload = {"time": time.time(), "event": event, **fields}
        with self._lock:
            self._event_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._event_file.flush()
            self._counts[event] += 1

    def stage_finished(self, stage, seconds, files, failures):
        self._stage_seconds[str(stage)] += float(seconds)
        self.record("stage_finished", stage=str(stage), seconds=float(seconds),
                    files=int(files), failures=int(failures))

    def _sample(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        rss_mb = rss / (1024.0 * 1024.0) if sys_platform_is_macos() else rss / 1024.0
        self._peak_rss_mb = max(self._peak_rss_mb, rss_mb)
        sample = {"time": time.time(), "rss_mb": rss_mb,
                  "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                  "gpus": []}
        try:
            import torch
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    with torch.cuda.device(index):
                        free, total = torch.cuda.mem_get_info()
                    free_gib = free / (1024 ** 3)
                    self._min_gpu_free[index] = min(
                        self._min_gpu_free.get(index, free_gib), free_gib)
                    sample["gpus"].append({
                        "index": index, "free_gib": free_gib,
                        "total_gib": total / (1024 ** 3),
                    })
        except Exception as exc:
            sample["gpu_error"] = f"{type(exc).__name__}: {exc}"
        return sample

    def _sample_loop(self):
        while not self._stop.wait(self.interval):
            sample = self._sample()
            with self._lock:
                self._sample_file.write(
                    json.dumps(sample, ensure_ascii=False) + "\n")
                self._sample_file.flush()

    def stop(self):
        if not self.enabled or self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 2))
        elapsed = time.time() - self.started_at
        self.record("monitor_stopped", elapsed_seconds=elapsed)
        summary = {
            "schema_version": 1,
            "elapsed_seconds": elapsed,
            "peak_main_process_rss_mb": self._peak_rss_mb,
            "minimum_gpu_free_gib": {str(k): v for k, v in self._min_gpu_free.items()},
            "event_counts": dict(self._counts),
            "stage_seconds": dict(self._stage_seconds),
        }
        path = os.path.join(self.output_dir, "performance_summary.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        for handle in (self._event_file, self._sample_file):
            if handle:
                handle.close()
        self._event_file = self._sample_file = None
        self._thread = None


def sys_platform_is_macos():
    import sys
    return sys.platform == "darwin"
