"""The performance file: step timings, worker load, separation internals, and the report
built from them. Everything goes to files; none of it reaches the console log."""

import collections
import copy
import json
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.separation_backends import SidonBackend
from services.worker_pool_service import WorkerPoolService
from utils import profiling
from utils.performance_monitor import PerformanceMonitor
from utils.performance_report import build_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- the report ---------------------------------------------------------------

def _events():
    t = 1000.0
    return [
        {"time": t + 100, "event": "stage_finished", "stage": "separation",
         "seconds": 97.0, "files": 2, "failures": 0},
        {"time": t + 40, "event": "stage_finished", "stage": "music", "seconds": 40.0,
         "files": 2, "failures": 0},
        {"time": t + 5, "event": "worker_loading", "worker": "sidon"},
        {"time": t + 34, "event": "worker_ready", "worker": "sidon"},
        {"event": "span", "time": t + 90, "name": "file_stage", "stage": "separation",
         "file": "a.mp3", "seconds": 90.0},
        {"event": "span", "time": t + 90, "name": "separation.process_overlaps",
         "stage": "separation", "file": "a.mp3", "seconds": 62.0},
        {"event": "span", "time": t + 90, "name": "release_worker", "stage": "separation",
         "worker": "sidon", "seconds": 3.6},
        {"event": "worker_profile", "time": t + 99, "worker": "sidon", "calls": 57,
         "wall_seconds": 62.0, "lease_wait_seconds": 1.5,
         "workers": [{"busy_seconds": 20.0, "calls": 30}, {"busy_seconds": 10.0, "calls": 27}]},
        {"event": "separation_profile", "time": t + 90, "file": "a.mp3", "values": {
            "windows": 57, "consumer": 61.8, "raw_wait": 5.7, "post": 54.0, "other": 2.1,
            "gpu_calls": 57, "gpu_queue": 0.0, "gpu_run": 200.0, "sidon_calls": 57,
            "sidon_total": 100.0, "sidon_roundtrip": 98.0, "sidon_worker": 90.0,
            "sidon_infer": 80.0, "wespeaker": 30.0, "wespeaker_calls": 228,
            "embedding_batch_attempts": 20, "embedding_batches": 19,
            "embedding_batched_items": 70, "embedding_batch_fallbacks": 1,
            "embedding_single_calls": 5}},
    ]


def test_the_report_has_the_stage_table_with_percentages_in_pipeline_order():
    text = build_report(_events(), [], 120.0)
    assert text.index("music") < text.index("separation")
    assert "70.8%" in text and "29.2%" in text       # 97/137 and 40/137... order-independent check below
    assert "total in stages: 137.0s" in text


def test_the_report_lists_per_file_steps_workers_and_separation_internals():
    text = build_report(_events(), [], 120.0)
    assert "a.mp3" in text and "separation.process_overlaps" in text
    assert "start -> ready" in text and "29.0" in text            # sidon load: 34 - 5
    assert "worker 0: busy 20.0s" in text and "waited 1.5s" in text
    assert "speaker assignment" in text and "87.4%" in text        # 54.0 / 61.8
    assert "GPU inference 1.40s" in text                           # 80 / 57
    assert "19/20 batches accepted" in text and "70 items batched" in text


def test_the_report_attributes_gpu_and_cpu_use_to_the_stage_it_happened_in():
    samples = [{"time": 1000.0 + i, "cpu_pct": 50.0, "gpu_util": {0: 10.0, 1: 90.0}}
               for i in range(1, 30)]
    text = build_report(_events(), samples, 120.0)
    assert "GPU0 avg/peak" in text and "10% / 10%" in text and "90% / 90%" in text


def test_an_empty_run_still_produces_a_report():
    assert "PIPELINE PERFORMANCE REPORT" in build_report([], [], 1.0)


# --- the monitor writes it as a file -------------------------------------------

def test_the_monitor_writes_the_report_to_its_own_file(tmp_path):
    monitor = PerformanceMonitor(str(tmp_path), interval_seconds=0.25, enabled=True)
    monitor.start()
    monitor.record_span("asr.process", 1.5, stage="asr", file="b.mp3", thread="t", depth=0)
    monitor.stage_finished("asr", 2.0, 1, 0)
    monitor.stop()
    report = (tmp_path / "performance_report.txt").read_text()
    assert "asr.process" in report and "1. STAGES" in report
    events = [json.loads(line) for line in
              (tmp_path / "scheduler_events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "span" and e["name"] == "asr.process" for e in events)


# --- timing wrappers on classes -------------------------------------------------

@pytest.fixture
def fake_module(monkeypatch):
    module = types.ModuleType("fake_profiled")

    class Service:
        def __init__(self, tag):
            self.tag = tag

        def work(self, x):
            return f"{self.tag}:{x}"

        def outer(self):
            return self.work(1)

        def boom(self):
            raise ValueError("bad")

        @staticmethod
        def static(x):
            return x * 2

    def free_function(x):
        return x + 1

    module.Service, module.free_function = Service, free_function
    monkeypatch.setitem(sys.modules, "fake_profiled", module)
    monkeypatch.setattr(profiling, "TARGETS", [
        ("fake_profiled", "Service", "work", "svc.work"),
        ("fake_profiled", "Service", "outer", "svc.outer"),
        ("fake_profiled", "Service", "boom", "svc.boom"),
        ("fake_profiled", "Service", "static", "svc.static"),
        ("fake_profiled", None, "free_function", "free"),
        ("fake_profiled", "Service", "missing", "nope"),        # skipped, not an error
        ("no_such_module_anywhere", "X", "y", "z"),             # skipped, not an error
    ])
    yield module
    profiling.uninstall()


class Recorder:
    def __init__(self):
        self.spans = []

    def record_span(self, name, seconds, **fields):
        self.spans.append({"name": name, "seconds": seconds, **fields})


def test_spans_carry_the_stage_and_file_and_nest(fake_module):
    recorder = Recorder()
    profiling.install(recorder)
    with profiling.file_stage("asr", "/data/x.mp3"):
        assert fake_module.Service("a").outer() == "a:1"
    names = [s["name"] for s in recorder.spans]
    assert names == ["svc.work", "svc.outer", "file_stage"]
    work, outer, whole = recorder.spans
    assert work["depth"] == 1 and outer["depth"] == 0
    assert work["stage"] == "asr" and work["file"] == "x.mp3" and whole["file"] == "x.mp3"


def test_a_copied_instance_still_runs_its_own_method(fake_module):
    # SeparationService is copied per file; a wrapper stored on the instance would
    # keep calling the original object.
    profiling.install(Recorder())
    original = fake_module.Service("orig")
    clone = copy.copy(original)
    clone.tag = "clone"
    assert clone.work(2) == "clone:2" and original.work(2) == "orig:2"


def test_static_methods_module_functions_and_errors(fake_module):
    recorder = Recorder()
    profiling.install(recorder)
    assert fake_module.Service.static(4) == 8 and fake_module.free_function(1) == 2
    with pytest.raises(ValueError):
        fake_module.Service("a").boom()
    by_name = {s["name"]: s for s in recorder.spans}
    assert set(by_name) == {"svc.static", "free", "svc.boom"}
    assert by_name["svc.boom"].get("error") is True and "error" not in by_name["free"]


def test_uninstall_restores_everything_and_nothing_is_recorded_without_a_monitor(fake_module):
    original_work = fake_module.Service.work
    recorder = Recorder()
    profiling.install(recorder)
    assert fake_module.Service.work is not original_work
    profiling.uninstall()
    assert fake_module.Service.work is original_work
    fake_module.Service("a").work(1)
    assert recorder.spans == []


def test_threads_keep_their_own_stage_and_file(fake_module):
    recorder = Recorder()
    profiling.install(recorder)
    gate = threading.Barrier(2, timeout=5)

    def run(stage, path):
        with profiling.file_stage(stage, path):
            gate.wait()
            fake_module.Service("t").work(0)

    threads = [threading.Thread(target=run, args=("asr", "a.mp3")),
               threading.Thread(target=run, args=("separation", "b.mp3"))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    work = {(s["stage"], s["file"]) for s in recorder.spans if s["name"] == "svc.work"}
    assert work == {("asr", "a.mp3"), ("separation", "b.mp3")}


def test_every_default_target_exists_in_the_code():
    # A renamed method would silently stop being timed.
    import importlib
    missing = []
    for module_name, class_name, attribute, _ in profiling.TARGETS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue                    # heavy dependency absent in this environment
        owner = getattr(module, class_name, None) if class_name else module
        if owner is None or not hasattr(owner, attribute):
            missing.append((module_name, class_name, attribute))
    assert not missing, missing


def test_the_batch_loop_and_main_switch_the_profiling_on():
    batch = open(os.path.join(ROOT, "utils", "batch.py"), encoding="utf-8").read()
    assert "profiling.file_stage(label, path)" in batch
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "profiling.install(performance_monitor)" in main
    assert "profiling.uninstall()" in main


# --- worker pool load -------------------------------------------------------------

class SlowService:
    name = "slow"
    process = object()

    def __init__(self, delay):
        self.delay = delay

    def request(self, payload, *, response_id=None):
        time.sleep(self.delay)
        return {"id": response_id}

    def spawn(self):
        pass


def test_the_pool_reports_busy_time_per_worker_and_the_wait_for_an_idle_one():
    pool = WorkerPoolService([SlowService(0.05), SlowService(0.05)], name="p")
    threads = [threading.Thread(target=pool.request, args=({"n": i},)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    profile = pool.profile()
    assert profile["calls"] == 4 and [w["calls"] for w in profile["workers"]] == [2, 2]
    assert all(w["busy_seconds"] >= 0.09 for w in profile["workers"])
    assert profile["lease_wait_seconds"] >= 0.04           # the 3rd/4th call waited
    assert profile["wall_seconds"] >= 0.09


def test_spawning_the_pool_again_starts_its_counters_over():
    pool = WorkerPoolService([SlowService(0.01)], name="p")
    pool.request({})
    pool.spawn()
    assert pool.profile()["calls"] == 0


# --- Sidon call breakdown -----------------------------------------------------------

def test_a_sidon_call_adds_its_parts_to_the_shared_counter(tmp_path):
    class Process:
        def request(self, payload, *, response_id=None):
            np.save(str(tmp_path / "t1.npy"), np.zeros(10, np.float32))
            np.save(str(tmp_path / "t2.npy"), np.zeros(10, np.float32))
            time.sleep(0.02)
            return {"id": response_id, "track_1_path": str(tmp_path / "t1.npy"),
                    "track_2_path": str(tmp_path / "t2.npy"), "target_sr": 24000,
                    "infer_seconds": 0.5, "worker_seconds": 0.75}

    backend = SidonBackend(process=Process(), temp_dir=str(tmp_path))
    backend.timing, backend.timing_lock = collections.Counter(), threading.Lock()
    backend.separate(np.zeros(1600, np.float32), 16000)
    t = backend.timing
    assert t["sidon_calls"] == 1 and t["sidon_infer"] == 0.5 and t["sidon_worker"] == 0.75
    assert t["sidon_roundtrip"] >= 0.02 and t["sidon_total"] >= t["sidon_roundtrip"]


def test_a_sidon_call_without_a_timing_sink_is_unchanged(tmp_path):
    class Process:
        def request(self, payload, *, response_id=None):
            np.save(str(tmp_path / "t1.npy"), np.zeros(10, np.float32))
            np.save(str(tmp_path / "t2.npy"), np.zeros(10, np.float32))
            return {"id": response_id, "track_1_path": str(tmp_path / "t1.npy"),
                    "track_2_path": str(tmp_path / "t2.npy"), "target_sr": 24000}

    out = SidonBackend(process=Process(), temp_dir=str(tmp_path)).separate(
        np.zeros(1600, np.float32), 16000)
    assert out[2] == 24000


def test_the_sidon_worker_reports_its_own_time():
    source = open(os.path.join(ROOT, "sidon_worker.py"), encoding="utf-8").read()
    assert '"infer_seconds"' in source and '"worker_seconds"' in source
