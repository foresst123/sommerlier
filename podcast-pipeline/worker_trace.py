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


def watch(interval=45):
    faulthandler.dump_traceback_later(interval, repeat=True, file=sys.stdout)


def done():
    faulthandler.cancel_dump_traceback_later()
