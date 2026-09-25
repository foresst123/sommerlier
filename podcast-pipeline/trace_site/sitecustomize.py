"""TEMPORARY -- part of worker_trace.py; delete this folder with it.

worker_trace puts this folder on PYTHONPATH of the vLLM worker, so every Python
process the worker starts (vLLM's EngineCore, compile workers) runs this at
start-up. It makes the process print all its threads' stacks to stderr every
`SOMMELIER_TRACE_CHILD` seconds for the first 15 minutes -- enough to see which
line a silent start-up is stuck on.
"""
import faulthandler
import os
import sys
import threading
import time

_interval = os.environ.get("SOMMELIER_TRACE_CHILD")


def _loop(interval):
    for _ in range(int(900 / max(interval, 1))):
        time.sleep(interval)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


if _interval:
    try:
        threading.Thread(target=_loop, args=(float(_interval),), daemon=True,
                         name="trace-dump").start()
    except Exception:
        pass
