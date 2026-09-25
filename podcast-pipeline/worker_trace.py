"""TEMPORARY start-up/step tracing for the vLLM workers.

Delete this file and the `worker_trace` lines in qwen3_worker.py and
whisper_vllm_worker.py once the start-up hang and the bottleneck are understood.

Importing it turns vLLM's log level up to DEBUG. `step` prints a timestamped
line on stdout while the model loads (the pipeline logs those as
"[<name> worker] ..."); `watch` makes Python dump every thread's stack every
`interval` seconds, so a hang shows the line it is stuck on; `done` stops the
dumps and must run before the worker starts answering requests, because after
"ready" stdout carries only JSON replies. `note` goes to stderr for use while
serving requests.
"""

import faulthandler
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

_T0 = time.monotonic()


def _stamp(message):
    return f"[TRACE +{time.monotonic() - _T0:7.1f}s pid={os.getpid()}] {message}"


def step(message):
    print(_stamp(message), flush=True)


def note(message):
    print(_stamp(message), file=sys.stderr, flush=True)


def _dump_from_a_thread(interval, minutes=15):
    """Print every thread's stack to stderr each `interval` s for `minutes` min."""
    def loop():
        for _ in range(int(minutes * 60 / max(interval, 1))):
            time.sleep(interval)
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    import threading
    threading.Thread(target=loop, daemon=True, name="trace-dump").start()


def watch(interval=45):
    faulthandler.dump_traceback_later(interval, repeat=True, file=sys.stdout)
    # The engine runs in child processes (vLLM's EngineCore), which is where a
    # silent start-up hangs. Make them dump their own stacks to stderr too: the
    # sitecustomize in trace_site/ does it for spawned children, the fork hook
    # below for forked ones.
    os.environ["SOMMELIER_TRACE_CHILD"] = str(interval)
    site = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_site")
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if site not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([site] + parts)
    if hasattr(os, "register_at_fork"):
        # NOT dump_traceback_later here: a forked child inherits the parent's
        # "watchdog is running" state without the watchdog thread, and calling
        # it waits for that thread forever -- the engine process hung at birth.
        os.register_at_fork(after_in_child=lambda: _dump_from_a_thread(interval))


def done():
    faulthandler.cancel_dump_traceback_later()
