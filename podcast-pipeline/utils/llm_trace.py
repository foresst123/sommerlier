"""Write every Qwen call (input and raw output) next to the audio file it belongs to.

<output_dir>/llm_trace/<stage>.jsonl holds one line per message of a call; the system
prompt goes once into <stage>.system_prompts.json, keyed by its sha1. Files are appended
to, so a re-run or a retry adds lines. A write never raises: a trace must not break a run.

The Trace is thread-local. Take it on the thread that calls the LLM (`trace_to`, then
`bind` when work moves to another thread) and pass it to `record_call` explicitly.
"""

import contextlib
import functools
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

RUN_ID = uuid.uuid4().hex[:8]

_local = threading.local()
_locks = {}
_locks_guard = threading.Lock()
_known_prompts = {}      # path -> {sha1: prompt}, so a prompt is written once


@dataclass(frozen=True)
class Trace:
    stage: str
    output_dir: str


def current() -> Optional[Trace]:
    return getattr(_local, "trace", None)


@contextlib.contextmanager
def activate(trace: Optional[Trace]):
    """Make `trace` (possibly None) the current one for this thread inside the block."""
    previous = current()
    _local.trace = trace
    try:
        yield trace
    finally:
        _local.trace = previous


@contextlib.contextmanager
def trace_to(stage: str, output_dir: str, enabled: bool = True):
    trace = Trace(stage, str(output_dir)) if enabled else None
    with activate(trace):
        yield trace


def bind(function):
    """Run `function` under the Trace of the thread that calls bind()."""
    trace = current()

    @functools.wraps(function)
    def bound(*args, **kwargs):
        with activate(trace):
            return function(*args, **kwargs)
    return bound


def _lock_for(path):
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


def _dumps(value, **kwargs):
    """JSON text that always encodes to UTF-8; a lone surrogate falls back to \\u escapes."""
    text = json.dumps(value, ensure_ascii=False, **kwargs)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(value, ensure_ascii=True, **kwargs)
    return text


def _load_prompts(path):
    """Prompts already on disk; an unreadable file is kept aside as .bad-<time>."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            known = json.load(handle)
        if isinstance(known, dict):
            return known
    except (OSError, ValueError):
        pass
    os.replace(path, f"{path}.bad-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:4]}")
    return {}


def _store_prompt(path, sha, system_prompt):
    """Add the prompt to the file (atomic replace). The cache follows only a good write."""
    known = _known_prompts.get(path) if os.path.isfile(path) else None
    if known is None:
        known = _load_prompts(path)
    if sha in known:
        _known_prompts[path] = known
        return
    updated = dict(known)
    updated[sha] = system_prompt
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(_dumps(updated, indent=2))
    os.replace(temporary, path)
    _known_prompts[path] = updated


def record_call(trace, *, system_prompt, user_messages, outputs, details, ok, error,
                thinking, max_new_tokens, duration_s, backend, replica=None,
                call_id=None) -> None:
    """Append one line per message. `details[i]` = finish_reason / token counts or None."""
    if trace is None:
        return
    try:
        folder = os.path.join(trace.output_dir, "llm_trace")
        os.makedirs(folder, exist_ok=True)
        jsonl = os.path.join(folder, f"{trace.stage}.jsonl")
        prompts = os.path.join(folder, f"{trace.stage}.system_prompts.json")
        call_id = call_id or uuid.uuid4().hex[:8]
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        file_name = os.path.basename(os.path.normpath(trace.output_dir))
        count = len(user_messages)
        with _lock_for(jsonl):
            sha = hashlib.sha1((system_prompt or "").encode(
                "utf-8", "surrogatepass")).hexdigest()
            try:
                _store_prompt(prompts, sha, system_prompt or "")
            except Exception as exc:  # noqa: BLE001 -- the call lines matter more
                logger.warning("llm_trace: could not write %s: %s", prompts, exc)
            lines = []
            for index, message in enumerate(user_messages):
                output = outputs[index] if outputs and index < len(outputs) else None
                detail = details[index] if details and index < len(details) else None
                detail = detail or {}
                lines.append(_dumps({
                    "run_id": RUN_ID, "ts": stamp, "stage": trace.stage,
                    "file": file_name, "call_id": call_id, "index": index,
                    "n_messages": count, "backend": backend, "replica": replica,
                    "thinking": bool(thinking), "max_new_tokens": max_new_tokens,
                    "system_prompt_sha1": sha, "input": message, "output": output,
                    "finish_reason": detail.get("finish_reason"),
                    "prompt_tokens": detail.get("prompt_tokens"),
                    "completion_tokens": detail.get("completion_tokens"),
                    "ok": bool(ok), "error": error,
                    "duration_s": round(float(duration_s), 3),
                }))
            with open(jsonl, "a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
    except Exception as exc:  # noqa: BLE001 -- tracing must never break the pipeline
        logger.warning("llm_trace: could not write %s trace: %s",
                       getattr(trace, "stage", "?"), exc)
