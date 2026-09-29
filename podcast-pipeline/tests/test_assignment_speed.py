"""Speaker assignment after each Sidon window ran on 4 CPU threads, one embedding at
a time, in the ordered consumer -- so it, not the Sidon workers, set the pace
(~1.8 s per window at 2 and at 6 workers). The four probe embeddings are independent,
so they can run at once, and the ONNX thread count is configurable."""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bss_model import BssSeparator
from utils import performance_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _separator(score_workers):
    sep = object.__new__(BssSeparator)
    sep.score_workers = score_workers
    from concurrent.futures import ThreadPoolExecutor
    sep._score_pool = (ThreadPoolExecutor(max_workers=score_workers)
                       if score_workers > 1 else None)
    return sep


def test_scores_come_back_in_task_order_whether_or_not_they_run_together():
    tasks = [lambda i=i: (time.sleep(0.02 * (4 - i)), i)[1] for i in range(4)]
    assert _separator(1)._run_scores(tasks) == [0, 1, 2, 3]
    assert _separator(4)._run_scores(tasks) == [0, 1, 2, 3]


def test_the_four_scores_really_run_at_the_same_time():
    barrier = threading.Barrier(4, timeout=2)

    def wait_for_the_others():
        barrier.wait()          # only passes if all four run concurrently
        return True

    assert _separator(4)._run_scores([wait_for_the_others] * 4) == [True] * 4


def test_the_settings_are_valid_performance_keys_with_safe_defaults():
    schema = performance_config._STAGES["separation"]
    assert schema["assignment_threads"][1] == 0        # 0 = the process budget
    assert schema["assignment_parallel"][1] == 1       # off unless a profile asks


def test_the_a100_profile_turns_them_on_and_keeps_the_cpu_for_assignment():
    cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    stage = cfg["environments"]["a100"]["performance"]["stages"]["separation"]
    assert stage["postprocess_device"] == "cpu"
    assert stage["assignment_threads"] > 4 and stage["assignment_parallel"] > 1


def test_ticks_add_up_per_phase_and_count_the_calls():
    import collections
    sep = object.__new__(BssSeparator)
    sep.timing = collections.Counter()
    sep._timing_lock = threading.Lock()
    started = time.perf_counter() - 0.5
    sep._tick("wespeaker", started)
    sep._tick("wespeaker", started)
    assert sep.timing["wespeaker_calls"] == 2 and sep.timing["wespeaker"] >= 1.0
