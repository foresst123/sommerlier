"""Lazy window streams and the byte admission budget of the separation stage.
Run in podcast-pipeline: python -m pytest tests/test_separation_admission.py -q"""
import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import services.separation_service as sep_module
from services.separation_service import SeparationService
from tests.test_separation_prefetch import (
    _AsyncModel, _StalledDownstream, _audio, _many_overlaps, _dialogue)
from utils.window_pool import ByteBudget, resolve_admission_bytes, window_cost_bytes


def _cfg(**extra):
    return {"enabled": True, "gpu_workers": 1, "postprocess_workers": 1,
            "gpu_prefetch_per_worker": 0, "ordered_postprocess": True, **extra}


def _run(**extra):
    svc = SeparationService(_AsyncModel(), logger=None, performance_config=_cfg(**extra))
    try:
        out = svc.process_overlaps(_many_overlaps(6), _audio(80.0), overlap_threshold=0.1)
    finally:
        svc.close_async_pools()
    return svc, out


def test_byte_budget_is_a_ledger_that_never_stalls_an_empty_pipeline():
    budget = ByteBudget(100)
    assert budget.fits(500)                 # empty: an oversized item still goes
    budget.charge(60)
    assert budget.fits(40) and not budget.fits(41)
    budget.release(60)
    assert budget.in_flight == 0
    assert ByteBudget(0).fits(10 ** 12)     # 0 = unlimited


def test_admission_bytes_off_explicit_and_derived():
    assert resolve_admission_bytes(-1) == 0
    assert resolve_admission_bytes(12345) == 12345
    derived = resolve_admission_bytes(0, 0.5)
    assert derived >= 0 and derived == int(derived)


def test_window_cost_counts_audio_and_is_zero_for_a_failed_build():
    class Built:
        audio = np.zeros(1000, dtype=np.float32)
    assert window_cost_bytes(Built()) >= 4000
    assert window_cost_bytes(None) == 0


def test_lazy_plan_keeps_only_a_factory_and_yields_the_same_windows():
    eager = SeparationService(_AsyncModel(), logger=None, performance_config=_cfg())
    lazy = SeparationService(_AsyncModel(), logger=None,
                             performance_config=_cfg(lazy_windows=True))
    audio = _audio(80.0)
    plan_e = eager._build_overlap_plan(_many_overlaps(6), audio, 0.1)
    plan_l = lazy._build_overlap_plan(_many_overlaps(6), audio, 0.1)
    assert plan_l.jobs == [] and plan_l.job_factory is not None
    streamed = list(plan_l.job_factory())
    assert len(streamed) == len(plan_e.jobs) > 1
    for (job_e, out_e), (job_l, out_l) in zip(plan_e.jobs, streamed):
        assert job_e[:2] == job_l[:2]
        assert np.array_equal(out_e[0].audio, out_l[0].audio)
        assert out_e[0].core == out_l[0].core


def test_lazy_windows_and_a_tiny_budget_change_nothing_in_the_result():
    _s, plain = _run()
    _s, lazy = _run(lazy_windows=True)
    _s, bounded = _run(lazy_windows=True, admission_bytes=1)
    for other in (lazy, bounded):
        assert [(s.index, s.bss_spans) for s in other] == [
            (s.index, s.bss_spans) for s in plain]
        for a, b in zip(plain, other):
            assert np.array_equal(a.audio, b.audio)


def _raw_done_while_stuck(expect=1, **extra):
    model = _StalledDownstream()
    svc = SeparationService(model, logger=None, performance_config=_cfg(**extra))
    result = {}
    worker = threading.Thread(target=lambda: result.update(
        out=svc.process_overlaps(_many_overlaps(8), _audio(120.0), overlap_threshold=0.1)))
    worker.start()
    deadline = time.time() + 20
    while model.raw_done < expect and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)                 # room for anything the budget should have stopped
    ahead = model.raw_done
    model.release.set()
    worker.join(30)
    svc.close_async_pools()
    return svc, ahead, result["out"]


def test_the_byte_budget_pauses_intake_without_dropping_a_window():
    svc, ahead, out = _raw_done_while_stuck(admission_bytes=1)
    assert ahead == 1                                    # backpressure held the rest
    assert svc.stats["admission_waits"] >= 1
    _svc, _ahead, reference = _raw_done_while_stuck(8)
    assert [(s.index, s.bss_spans) for s in out] == [
        (s.index, s.bss_spans) for s in reference]       # every window still ran


def test_without_a_budget_the_count_limit_of_zero_still_means_no_limit():
    _svc, ahead, _out = _raw_done_while_stuck(8, admission_bytes=-1)
    assert ahead == 8


def test_a_generous_budget_never_pauses_intake():
    svc, ahead, _out = _raw_done_while_stuck(8, admission_bytes=10 ** 12)
    assert ahead == 8 and svc.stats["admission_waits"] == 0


def test_closing_a_lazy_stream_early_releases_the_file_shared_memory(monkeypatch):
    class FakeFile:
        closed = False

        def build_all(self, groups):
            for _ in groups:
                yield None, "window_error", "x", []

        def close(self):
            FakeFile.closed = True

    class FakePool:
        def open_file(self, *a, **k):
            return FakeFile()

    monkeypatch.setattr(sep_module, "_resolve_pool_size", lambda: 2)
    svc = SeparationService(_AsyncModel(), logger=None,
                            performance_config=_cfg(lazy_windows=True))
    svc._window_pool_state["pool"] = FakePool()
    plan = svc._build_overlap_plan(_many_overlaps(6), _audio(80.0), 0.1)
    assert not FakeFile.closed                           # nothing opened until pulled
    stream = plan.job_factory()
    next(stream)
    assert not FakeFile.closed
    stream.close()
    assert FakeFile.closed


def test_thread_budget_shares_the_cores_and_never_starves_a_consumer():
    from utils.cpu_plan import separation_thread_budget
    plan = separation_thread_budget(24, sidon_workers=6, assignment_workers=4,
                                    assignment_worker_threads=4)
    assert plan["assignment_worker_threads"] * 4 <= int((24 - 1 - 6) * 0.6)
    assert plan["window_pool_workers"] >= 1
    assert plan["assignment_worker_threads"] >= 1
    tiny = separation_thread_budget(2, 6, 4, 4)
    assert tiny["window_pool_workers"] >= 1 and tiny["assignment_worker_threads"] >= 1
    no_assign = separation_thread_budget(24, 6, 0, 4)
    assert no_assign["assignment_total_threads"] == 0


def test_cpu_budget_only_caps_the_pool_when_switched_on(monkeypatch):
    monkeypatch.setattr(sep_module, "usable_cores", lambda: 24)
    monkeypatch.setattr(sep_module, "_BSS_WINDOW_WORKERS_ENV", None)
    on = SeparationService(_AsyncModel(), logger=None, performance_config=_cfg(
        cpu_budget=True, gpu_workers=6, max_workers=2, assignment_workers_per_gpu=2,
        assignment_worker_threads=4))
    off = SeparationService(_AsyncModel(), logger=None, performance_config=_cfg())
    assert on._budgeted_pool_size(22) < 22
    assert off._budgeted_pool_size(22) == 22
    assert on._budgeted_pool_size(0) == 0
