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

_interval = os.environ.get("SOMMELIER_TRACE_CHILD")
if _interval:
    try:
        faulthandler.dump_traceback_later(float(_interval), repeat=True, file=sys.stderr)
        _stop = threading.Timer(900, faulthandler.cancel_dump_traceback_later)
        _stop.daemon = True
        _stop.start()
    except Exception:
        pass
