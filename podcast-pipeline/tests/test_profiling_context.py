"""A span recorded on a pool thread must still say which stage and file it belongs to.

The report's ASR / refinement / conversation breakdown keeps only spans that carry a stage,
and the parts that matter most run off the file thread: the ASR vote on its own pool, LLM
requests on the replica threads and the window pool. Without the stage they were dropped
from the breakdown without any sign that they had been.
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import profiling
from utils.performance_report import _focused_stages


class Monitor:
    def __init__(self):
        self.spans, self.events = [], []
        self._lock = threading.Lock()

    def record_span(self, name, seconds, **fields):
        with self._lock:
            self.spans.append({"event": "span", "name": name, "seconds": seconds, **fields})

    def record(self, event, **fields):
        with self._lock:
            self.events.append({"event": event, **fields})


def _timed(name):
    return profiling._wrap(lambda: None, name)


def _install(monitor):
    previous = profiling._monitor
    profiling._monitor = monitor
    return previous


def test_a_bound_function_carries_the_submitters_stage_and_file_to_the_pool_thread():
    monitor = Monitor()
    previous = _install(monitor)
    try:
        unbound, bound = _timed("asr.vote"), _timed("asr.vote")
        with profiling.file_stage("asr", "/data/talk.mp3"):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(unbound).result()
                pool.submit(profiling.bind(bound)).result()
    finally:
        profiling._monitor = previous

    votes = [s for s in monitor.spans if s["name"] == "asr.vote"]
    assert [s["stage"] for s in votes] == [None, "asr"]      # what bind() fixes
    assert votes[1]["file"] == "talk.mp3"


def test_the_pool_thread_is_left_as_it_was_after_the_bound_call():
    monitor = Monitor()
    previous = _install(monitor)
    try:
        with profiling.file_stage("asr", "/data/talk.mp3"):
            bound = profiling.bind(lambda: None)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(bound).result()
            leftover = pool.submit(lambda: (profiling.current_file(),
                                            profiling._context().stage)).result()
    finally:
        profiling._monitor = previous
    assert leftover == (None, None)


def test_the_focused_breakdown_now_counts_work_that_ran_on_pool_threads():
    monitor = Monitor()
    previous = _install(monitor)
    try:
        with profiling.file_stage("asr", "/data/talk.mp3"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                pool.submit(profiling.bind(_timed("asr.vote"))).result()
                pool.submit(_timed("asr.vote")).result()          # not bound: no stage
    finally:
        profiling._monitor = previous

    events = monitor.spans + [{"event": "stage_finished", "stage": "asr", "seconds": 10.0}]
    text = "\n".join(_focused_stages(events))
    assert "asr" in text and "asr.vote" in text
    assert "| 1 " in text or " 1 " in text          # only the bound call was attributed


def test_the_sites_that_hand_work_to_a_pool_bind_it():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def source(*parts):
        return open(os.path.join(root, *parts), encoding="utf-8").read()

    assert "profiling.bind(run_vote)" in source("services", "asr_service.py")
    assert "profiling.bind(_ask_range)" in source("utils", "llm_batches.py")
    refinement = source("services", "diarization_refinement_service.py")
    assert "profiling.bind(self.generate_texts)" in refinement
