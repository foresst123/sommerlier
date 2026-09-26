"""Stage passes, replica decisions and VRAM attribution on a 2x T4 run.

Each test pins something the Kaggle run of 2026-09-20 got wrong: the
music_removal pass ran ASR and the LLM for every file and ran out of memory,
the Qwen replica loaded twice for a 3-second and a zero-second gain, pass 2
revived the ASR worker in the music stage, and the summary called a recovered
file failed.

Run:  python -m pytest tests/test_stage_scheduling.py -q   (from podcast-pipeline/)
"""
import os
import re
import sys
import threading
import time
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from services.asr_service import ASRService, estimate_replica_gain
from services.pipeline_service import PipelineService
from utils.batch import PIPELINE_STAGES, split_final_failures
from utils.performance_monitor import PerformanceMonitor


def _source(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


# --- every batch pass stops where its name says -----------------------------

def test_every_batch_pass_has_a_stop_point_in_run():
    """A pass with no stop point runs everything after it, file by file.

    music_removal had none, so it ran ASR -> LLM -> export per file: the LLM of
    file 1 and the ASR of file 2 met on GPU1 and ran it out of memory, and the
    later asr and refinement passes only read checkpoints.
    """
    src = _source("services/pipeline_service.py")
    for stage in PIPELINE_STAGES:
        if stage is None:
            continue
        assert f'getattr(args, "stop_after", None) == "{stage}"' in src, (
            f"the '{stage}' pass has no stop point in run(), so it would run "
            "every later step as well")


def test_music_removal_stops_before_asr():
    src = _source("services/pipeline_service.py")
    stop = src.index('getattr(args, "stop_after", None) == "music_removal"')
    assert stop < src.index("# 5. Nhận dạng lời nói"), (
        "the music_removal pass must return before ASR starts")


def test_window_prefetch_fires_only_on_the_diarization_stop_and_only_when_separation_runs():
    """Window building is pure CPU and does not need Sidon loaded, so it can
    start the moment this file's diarization result exists -- while DiariZen
    is still busy with the rest of the batch and before Sidon's worker has
    even started. It must fire on exactly the one pass that stops after
    diarization, not on a later pass re-entering that section of run() to
    load the checkpoint on its way to another stage."""
    src = _source("services/pipeline_service.py")
    stop = src.index('getattr(args, "stop_after", None) == "diarization"')
    block = src[stop:src.index("return None", stop)]
    assert "prefetch_overlap_plan" in block, (
        "the diarization pass must kick off window building while it still "
        "has the rest of the batch to run on GPU")
    assert 'self.step_enabled(args, "separation")' in block, (
        "prefetching must not run when separation itself is switched off")


def test_the_real_separation_call_passes_its_audio_path_to_the_prefetch_cache():
    src = _source("services/pipeline_service.py")
    call = re.search(r"self\.separation_svc\.process_overlaps\([^)]*\)", src, re.S).group(0)
    assert "audio_path=audio_path" in call, (
        "without the path, process_overlaps can never find what was prefetched for this file")


def test_diarization_postprocessing_is_deferred_only_when_stopping_after_diarization():
    """The fresh-compute branch must call diarize_raw() unconditionally, but
    only defer diarize_postprocess() to the background pool inside the
    stop_after == "diarization" block -- a file-major run or the final pass
    needs the finished DiarizationResult before it can continue within the
    same call, so it must stay synchronous there."""
    src = _source("services/pipeline_service.py")
    assert "self.diarization_svc.diarize_raw(" in src
    stop = src.index('getattr(args, "stop_after", None) == "diarization"')
    deferred_block = src[stop:src.index("return None", stop)]
    assert "self.diarization_svc.submit_postprocess(" in deferred_block
    assert "then=_finish" in deferred_block, (
        "the tail must run inside the future the drain waits on, not in a done-callback")
    assert "add_done_callback" not in deferred_block
    after_deferred_block = src[src.index("return None", stop):]
    # The synchronous fallback (for every OTHER call to this section) must
    # still call diarize_postprocess directly, outside the deferred block.
    assert "self.diarization_svc.diarize_postprocess(" in after_deferred_block[:800]


def test_the_diarization_tail_checkpoints_then_prefetches_then_writes_audit_clips():
    src = _source("services/pipeline_service.py")
    stop = src.index('getattr(args, "stop_after", None) == "diarization"')
    deferred_block = src[stop:src.index("return None", stop)]
    checkpoint_at = deferred_block.index('checkpoint.save("diarization"')
    write_at = deferred_block.index("stage_out.write_diarization(")
    prefetch_at = deferred_block.index("self.separation_svc.prefetch_overlap_plan(")
    assert checkpoint_at < prefetch_at < write_at, (
        "must checkpoint, then start the overlap-plan prefetch, then write the "
        "audit clips nothing downstream depends on")


def test_pending_diar_jobs_is_shared_across_parallel_stage_view_copies():
    """parallel_stage_view() does copy.copy(self); _pending_diar_jobs must be
    created once in __init__ (like _model_load_lock) so every concurrent
    file's view shares the same dict, not one each."""
    src = _source("services/pipeline_service.py")
    init_src = src[src.index("def __init__"):src.index("def parallel_stage_view")]
    assert "self._pending_diar_jobs" in init_src
    assert "self._pending_diar_lock" in init_src


def test_closing_the_prefetch_pool_is_deferred_like_the_window_pool():
    """A prefetch started during 'diarization' is only consumed during
    'separation'; close_prefetch_pool() must be deferred to that SAME stage
    boundary as close_window_pool(), not fired eagerly, or an in-flight
    prefetch would be torn down before the separation pass ever reads it."""
    src = _source("services/pipeline_service.py")
    window_pool_defer = src.index("self.separation_svc.close_window_pool()")
    prefetch_defer = src.index("self.separation_svc.close_prefetch_pool()")
    between = src[window_pool_defer:prefetch_defer]
    assert between.count("_defer_or_run(") == 1, (
        "close_prefetch_pool must be the very next _defer_or_run() call after "
        "close_window_pool(), in the same deferred-cleanup block")


@pytest.mark.parametrize("stop, stage, expected", [
    (None, "asr", True),
    ("music", "diarization", False),
    ("music_removal", "separation", True),
    ("music_removal", "asr", False),
    ("asr", "asr", True),
    ("captioning", "asr", True),
    ("something_unknown", "asr", True),
])
def test_a_pass_reaches_only_the_stages_before_its_stop(stop, stage, expected):
    args = types.SimpleNamespace(stop_after=stop)
    assert PipelineService._reaches(args, stage) is expected


def test_a_worker_is_revived_only_by_a_pass_that_reaches_its_stage():
    """Pass 2's music stage started the ASR worker: its client survived pass 1,
    and nothing asked whether this pass would ever get to ASR."""
    src = _source("services/pipeline_service.py")
    for worker, stage in (("diarizen", "diarization"), ("qwen3", "asr"),
                          ("sidon", "separation")):
        call = re.search(rf'.*_rebind_worker\(args, "{worker}".*', src).group(0)
        guard = src.splitlines()[src[:src.index(call)].count("\n") - 1]
        assert f'self._reaches(args, "{stage}")' in guard, (
            f"{worker} can be started by a pass that never reaches {stage}")


# --- the replica opens only when it shortens the stage ----------------------

def test_a_backlog_the_running_worker_clears_during_the_load_gains_nothing():
    """Pass 2: 22 jobs at 1.36/s are gone before a 30s load finishes."""
    gain, t_old, t_new = estimate_replica_gain(22, 1.36, 30.0, 1.36)
    assert gain == 0.0
    assert t_new == t_old


def test_the_measured_throughput_of_pass_one_does_not_justify_a_replica():
    """Pass 1: two workers together did 1.44 jobs/s against 1.36 alone."""
    gain, _, _ = estimate_replica_gain(117, 1.36, 30.8, 1.36 * 0.06)
    assert gain < 15.0


def test_the_first_guess_would_still_open_one_before_anything_is_measured():
    gain, t_old, t_new = estimate_replica_gain(117, 1.36, 30.8, 1.36 * 0.5)
    assert gain > 15.0
    assert t_new < t_old


def test_nothing_is_gained_while_another_model_is_the_tail():
    """PhoWhisper finished last in pass 2; a faster Qwen ends nothing sooner."""
    _, t_old, _ = estimate_replica_gain(117, 1.36, 30.8, 1.36)
    gain, _, _ = estimate_replica_gain(117, 1.36, 30.8, 1.36, peer_remaining=t_old + 10)
    assert gain == 0.0


def test_the_gain_is_capped_by_the_other_models_remaining_time():
    gain, t_old, t_new = estimate_replica_gain(117, 1.36, 30.8, 1.36, peer_remaining=70.0)
    assert gain == pytest.approx(max(t_old, 70.0) - max(t_new, 70.0))
    assert 0 < gain < t_old - t_new


def test_a_replica_that_ran_teaches_the_next_decision():
    svc = ASRService()
    svc._learn_from_replica(load_seconds=30.8, rate_alone=1.36,
                            jobs_together=75, seconds_together=52.0)
    assert svc._replica_load_seconds == pytest.approx(30.8)
    assert svc._replica_speed_ratio == pytest.approx(75 / 52.0 / 1.36 - 1.0)
    assert svc._replica_speed_ratio < 0.1


class _SlowClient:
    def __init__(self, per_job):
        self.per_job = per_job
        self.calls = 0

    def transcribe_batch(self, jobs, language="vi"):
        self.calls += 1
        time.sleep(self.per_job * len(jobs))
        return [f"text-{job_id}" for job_id, _ in jobs]


class _FakeReplicaService:
    def __init__(self):
        self.process = None
        self.spawns = 0
        self.stops = 0

    def spawn(self):
        self.spawns += 1
        self.process = types.SimpleNamespace(pid=4242)

    def wait_ready(self):
        pass

    def stop(self):
        self.stops += 1
        self.process = None


class _Events:
    def __init__(self):
        self.events = []
        self.names = {}

    def record(self, event, **fields):
        self.events.append((event, fields))

    def register_process(self, pid, name):
        self.names[pid] = name

    def named(self, event):
        return [fields for name, fields in self.events if name == event]


def _asr(replica, events, **cfg):
    svc = ASRService(qwen3=_SlowClient(0.01), qwen3_replica_service=replica,
                     performance_config=cfg, performance_monitor=events,
                     batch_size=1)
    return svc


def _run(svc, tmp_path, jobs, peer):
    released = threading.Event()
    released.set()
    audios = [np.zeros(160, dtype=np.float32) for _ in range(jobs)]
    return svc._run_qwen3_batch(audios, [f"{i:05d}" for i in range(jobs)],
                                str(tmp_path), replica_event=released,
                                peer_remaining=lambda: peer)


def test_no_replica_is_loaded_while_the_other_model_is_the_tail(tmp_path):
    replica, events = _FakeReplicaService(), _Events()
    svc = _asr(replica, events, replica_min_pending_jobs=1,
               replica_min_gain_seconds=0.0, replica_load_seconds=0.01,
               replica_speed_ratio=1.0)

    results = _run(svc, tmp_path, 40, peer=1000.0)

    assert replica.spawns == 0
    assert all(results)
    assert [f["reason"] for f in events.named("asr_replica_skipped")] == ["not_the_tail"]


def test_a_replica_opens_when_it_is_forecast_to_end_the_stage_sooner(tmp_path, monkeypatch):
    import models.qwen3_asr as qwen3_asr
    monkeypatch.setattr(qwen3_asr, "Qwen3ASRClient", lambda process: _SlowClient(0.01))
    replica, events = _FakeReplicaService(), _Events()
    svc = _asr(replica, events, replica_min_pending_jobs=1,
               replica_min_gain_seconds=0.05, replica_load_seconds=0.01,
               replica_speed_ratio=1.0)

    results = _run(svc, tmp_path, 80, peer=0.0)

    assert replica.spawns == 1
    assert replica.stops == 1, "the replica must not outlive its batch"
    assert results == [f"text-{i:05d}" for i in range(80)], "every job answered once, in order"
    assert events.named("asr_replica_loading")[0]["gain"] >= 0.05
    assert events.names == {4242: "qwen3_replica"}
    assert events.named("asr_replica_measured"), "the run must feed the next decision"
    assert svc._replica_speed_ratio is not None


# --- the monitor can say who held the memory --------------------------------

def test_vram_is_attributed_to_the_worker_that_holds_it(tmp_path, monkeypatch):
    monitor = PerformanceMonitor(str(tmp_path), enabled=False)
    monitor.register_process(101, "qwen3")
    answers = {
        "--query-gpu=index,uuid": [["0", "GPU-a"], ["1", "GPU-b"]],
        "--query-compute-apps=pid,gpu_uuid,used_memory": [
            ["101", "GPU-b", "6300"], ["202", "GPU-b", "5100"],
            [str(os.getpid()), "GPU-a", "3700"]],
    }
    monkeypatch.setattr(monitor, "_nvidia_smi", lambda query: answers[query])

    rows = monitor._sample_processes()

    by_name = {row["name"]: (row["gpu"], row["used_mib"]) for row in rows}
    assert by_name == {"qwen3": (1, 6300.0), "pid:202": (1, 5100.0),
                       "main": (0, 3700.0)}
    assert monitor._peak_process_mib["qwen3(101)"] == {"1": 6300.0}


def test_a_box_without_nvidia_smi_stops_asking(tmp_path, monkeypatch):
    monitor = PerformanceMonitor(str(tmp_path), enabled=False)

    def missing(*query):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(monitor, "_nvidia_smi", missing)
    sample = monitor._sample()
    assert "processes" not in sample
    assert monitor._process_sampling is False


def test_a_pool_reports_one_name_per_process():
    events = _Events()
    pipeline = PipelineService.__new__(PipelineService)
    pipeline.performance_monitor = events
    pool = types.SimpleNamespace(processes=[types.SimpleNamespace(pid=7),
                                            types.SimpleNamespace(pid=8)])

    assert pipeline._register_worker_pids("sidon", pool) == [7, 8]
    assert events.names == {7: "sidon[0]", 8: "sidon[1]"}


# --- a stage hands the next one an empty card -------------------------------

def test_closing_a_stage_reclaims_vram_and_records_what_is_left():
    """"Unloaded X from VRAM" only drops a reference; the allocator can hold
    the memory for another minute, which is long enough for the next stage to
    ask for 82% of each card and miss."""
    events = _Events()
    pipeline = PipelineService.__new__(PipelineService)
    pipeline.logger = None
    pipeline.model_loader = None
    pipeline.worker_services = {}
    pipeline.performance_monitor = events
    pipeline.begin_stage_scope()

    pipeline.end_stage_scope()

    reclaimed = events.named("stage_vram_reclaimed")
    import torch
    if torch.cuda.is_available():
        assert reclaimed and reclaimed[0]["free_gib"], "free VRAM must be recorded"
    else:
        assert reclaimed == [], "nothing to record without a GPU"


def test_the_reclaim_runs_after_the_deferred_releases_not_before():
    """Reclaiming before the unloads would measure the memory still held."""
    src = _source("services/pipeline_service.py")
    scope = src[src.index("def end_stage_scope"):src.index("def _reclaim_vram")]
    assert scope.index("Deferred cleanup callback failed") < scope.index("self._reclaim_vram()")


def test_a_stage_that_blows_up_still_reclaims():
    src = _source("utils/batch.py")
    assert "finally:" in src[:src.index("if end:\n                end()")], (
        "end_stage_scope must run from a finally, or a failed stage keeps its models")


# --- the summary reports what is still broken -------------------------------

def test_a_file_recovered_by_a_later_pass_is_not_reported_failed():
    failures = [("a.mp3", "music_removal: OOM")]
    still, recovered = split_final_failures(failures, is_done=lambda p: p == "a.mp3")
    assert still == []
    assert recovered == [("a.mp3", "music_removal: OOM")]


def test_a_file_that_failed_twice_is_reported_once_with_its_last_error():
    failures = [("b.mp3", "first"), ("b.mp3", "second")]
    still, recovered = split_final_failures(failures, is_done=lambda p: False)
    assert still == [("b.mp3", "second")]
    assert recovered == []


def test_music_service_is_constructed_with_a_performance_config():
    src = _source("main.py")
    start = src.index("music_svc = MusicService(")
    call = src[start:src.index(")", start)]
    assert "performance_config" in call, (
        "MusicService must receive performance_config the same way "
        "SeparationService/ASRService already do, or the async runtime "
        "can never turn on in production")
