"""Low-overhead resource and scheduler telemetry for long pipeline runs."""

import json
import os
import resource
import subprocess
import threading
import time
from collections import defaultdict


class PerformanceMonitor:
    def __init__(self, output_dir, interval_seconds=1.0, enabled=True,
                 logger=None, process_interval_seconds=5.0):
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
        # Per-process VRAM: device totals alone cannot say who held the memory
        # when a card filled up. nvidia-smi is a subprocess, so it runs on its
        # own, slower cadence rather than every sample.
        self.process_interval = max(1.0, float(process_interval_seconds))
        self._last_process_sample = 0.0
        self._process_names = {os.getpid(): "main"}
        self._gpu_index_by_uuid = None
        self._process_sampling = True
        self._peak_process_mib = defaultdict(dict)

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

    def register_process(self, pid, name):
        """Name a PID so its VRAM is attributed to a worker, not a number."""
        if pid:
            with self._lock:
                self._process_names[int(pid)] = str(name)

    def _nvidia_smi(self, *query):
        out = subprocess.run(
            ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
        return [[cell.strip() for cell in line.split(",")]
                for line in out.splitlines() if line.strip()]

    def _sample_processes(self):
        """VRAM per compute process, keyed by the physical GPU index.

        Physical, not torch-local: each worker runs under its own
        CUDA_VISIBLE_DEVICES, so "cuda:0" means a different card per process.
        Inside a container nvidia-smi may report host PIDs that match nothing
        registered here; those are kept as "pid:N" rather than dropped.
        """
        if self._gpu_index_by_uuid is None:
            self._gpu_index_by_uuid = {
                uuid: int(index)
                for index, uuid in self._nvidia_smi("--query-gpu=index,uuid")}
        rows = self._nvidia_smi("--query-compute-apps=pid,gpu_uuid,used_memory")
        with self._lock:
            names = dict(self._process_names)
        processes = []
        for pid, uuid, used in rows:
            try:
                pid, used = int(pid), float(used)
            except ValueError:
                continue
            gpu = self._gpu_index_by_uuid.get(uuid, uuid)
            name = names.get(pid, f"pid:{pid}")
            processes.append({"pid": pid, "name": name, "gpu": gpu,
                              "used_mib": used})
            key = f"{name}({pid})"
            self._peak_process_mib[key][str(gpu)] = max(
                self._peak_process_mib[key].get(str(gpu), 0.0), used)
        return processes

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
        now = time.time()
        if self._process_sampling and now - self._last_process_sample >= self.process_interval:
            self._last_process_sample = now
            try:
                sample["processes"] = self._sample_processes()
            except Exception as exc:
                # No nvidia-smi (CPU box, macOS) or a driver that refuses the
                # query: stop asking instead of paying a failed spawn per tick.
                self._process_sampling = False
                self.record("process_sampling_disabled",
                            error=f"{type(exc).__name__}: {exc}")
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
            "peak_process_vram_mib": {k: dict(v) for k, v in self._peak_process_mib.items()},
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
