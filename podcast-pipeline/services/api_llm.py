"""A chat-completions API standing in for the resident local LLM in the LLM passes.

Refinement, speaker relabel, transcript review, persona profile and conversation selection
all run on one object (`DiarizationRefinementService`); the passes after refinement talk to it
through a small surface:
`ensure_loaded`, `count_tokens`, `generate_texts`, `batch_size`, `max_batch_tokens`,
`model_name`, `parallel_windows`, `replica_count`. `ApiLLM` offers the same surface
over any OpenAI-compatible endpoint, so those passes run unchanged: same prompts, same
windows, same parsing, same checkpoints and output layout.

Finished work is never redone. Each pass keys its checkpoint on `llm.model_name`, so
`model_name` here is the NAME OF THE LOCAL MODEL the checkpoints were made with, not the
API model: files already reviewed or selected are loaded from their checkpoints and only
files still to do reach the API. The model actually called is `api_model` and is written
in every llm_trace line (`backend: "api"`).

The key is read from the environment (`api_key_env`) and is never logged or stored.

Refinement uses it too: `DiarizationRefinementService.use_remote(api_llm)` makes the
refinement service hand its requests to this object instead of starting the local worker.
"""

import hashlib
import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

from utils import llm_trace

DEFAULT_BASE_URL = "https://api.openai.com/v1"
BACKENDS = ("local", "api")
# The LLM stages that can run on the API. Refinement sends one request per segment, so it
# dominates the token bill; running it locally and only the rest on the API cuts most of it.
STAGES = ("refinement", "relabel", "review", "persona", "selection")
_CONFIG_KEYS = frozenset({
    "model", "checkpoint_model_name", "base_url", "api_key_env", "concurrency",
    "timeout_seconds", "max_retries", "reasoning_effort", "completion_floor", "stages",
    "prompt_cache", "prompt_cache_retention", "cache_max_shards"})
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_LOG_EVERY_CALLS = 25
# OpenAI routes a request to a cache by its prefix and `prompt_cache_key`; a prefix+key pair
# should stay under about 15 requests a minute or the overflow lands on cold machines.
_CACHE_RPM_PER_KEY = 12
_CACHE_WINDOW_SECONDS = 60.0


class ApiError(RuntimeError):
    """One request failed for good (retries used up, or a status that retrying cannot fix)."""


def _default_transport(url: str, body: bytes, headers: dict, timeout: float) -> Tuple[int, bytes, dict]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() or b"", dict(exc.headers or {})


class ApiLLM:
    """Same surface as the local LLM service, backed by a chat-completions endpoint."""

    def __init__(self, model: str, *, checkpoint_model_name: str,
                 base_url: str = DEFAULT_BASE_URL, api_key_env: str = "OPENAI_API_KEY",
                 concurrency: int = 8, timeout_seconds: float = 120.0, max_retries: int = 3,
                 reasoning_effort: Optional[str] = None, completion_floor: Optional[int] = None,
                 prompt_cache: Optional[bool] = None, prompt_cache_retention: Optional[str] = None,
                 cache_max_shards: int = 32, logger=None,
                 transport: Optional[Callable] = None, sleep: Callable = time.sleep,
                 clock: Callable = time.monotonic):
        if not model:
            raise ValueError("api_llm.model is required (the id of the API model to call)")
        if not checkpoint_model_name:
            raise ValueError("api_llm needs the local model name its checkpoints were made with")
        self.api_model = str(model)
        self.stages = frozenset(STAGES)          # build_pass_llm narrows this to what was asked for
        self.model_name = str(checkpoint_model_name)      # checkpoint identity, see module docstring
        self.base_url = str(base_url).rstrip("/")
        self.api_key_env = str(api_key_env)
        self.concurrency = max(1, int(concurrency))
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self.reasoning_effort = reasoning_effort or None
        # A reasoning model spends part of max_completion_tokens thinking; a small cap can leave
        # no room for the answer. With reasoning on, never ask for less than this.
        self.completion_floor = (int(completion_floor) if completion_floor is not None
                                 else (1024 if self.reasoning_effort else 0))
        self.logger = logger
        self._transport = transport or _default_transport
        self._sleep = sleep
        # The passes size their calls from these: one message per call, many calls at once.
        self.batch_size = 1
        self.max_batch_tokens = 0
        self.parallel_windows = True
        self.replica_count = self.concurrency
        # Prompt caching: the long system prompt comes first in every request, so a repeated
        # prefix is cached by the provider at a fraction of the input price. `prompt_cache_key`
        # steers requests that share a prefix to the same cache; it is sent only to the OpenAI
        # endpoint (another gateway may reject an unknown field) and dropped if the server says no.
        self.prompt_cache = ((self.base_url == DEFAULT_BASE_URL) if prompt_cache is None
                             else bool(prompt_cache))
        self.prompt_cache_retention = prompt_cache_retention or None
        self.cache_max_shards = max(1, int(cache_max_shards))
        self._clock = clock
        self._cache_seen = {}            # prefix hash -> request times in the last minute
        self._cache_turn = {}            # prefix hash -> requests so far, to spread them over keys
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.concurrency)   # requests in flight, across every caller
        self.stats = {"calls": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cached_tokens": 0}

    # -- the surface the passes use ------------------------------------------------
    def ensure_loaded(self) -> bool:
        """Nothing to load; usable when the key is set."""
        return bool(os.environ.get(self.api_key_env))

    def token_usage(self) -> dict:
        """Tokens billed so far, in the shape the refinement progress line reads."""
        with self._lock:
            return {"prompt_tokens": self.stats["prompt_tokens"],
                    "completion_tokens": self.stats["completion_tokens"]}

    def count_tokens(self, text: str) -> int:
        """Same estimate the local service uses when its tokenizer is in another process."""
        return max(1, int(len(text) / 3.5))

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None, thinking=False):
        """(succeeded, texts), in the order of `user_messages`; (False, []) when one request failed."""
        messages = list(user_messages)
        trace = llm_trace.current()
        started = time.perf_counter()
        texts: List[Optional[str]] = [None] * len(messages)
        details: List[Optional[dict]] = [None] * len(messages)
        error = None
        try:
            if len(messages) <= 1 or self.concurrency <= 1:
                for i, message in enumerate(messages):
                    texts[i], details[i] = self._chat(system_prompt, message, max_new_tokens, thinking)
            else:
                with ThreadPoolExecutor(max_workers=min(self.concurrency, len(messages)),
                                        thread_name_prefix="api-llm") as pool:
                    futures = [pool.submit(self._chat, system_prompt, m, max_new_tokens, thinking)
                               for m in messages]
                    for i, future in enumerate(futures):
                        texts[i], details[i] = future.result()
            ok = True
        except ApiError as exc:
            ok, error = False, str(exc)
            if self.logger:
                self.logger.warning(f"[api-llm] {error}")
        finally:
            llm_trace.record_call(
                trace, system_prompt=system_prompt, user_messages=messages,
                outputs=texts if error is None else None,
                details=details if error is None else None, ok=error is None, error=error,
                thinking=bool(thinking), max_new_tokens=max_new_tokens,
                duration_s=time.perf_counter() - started, backend="api")
        return (True, texts) if ok else (False, [])

    # -- one request ---------------------------------------------------------------
    def _cache_key(self, system_prompt: str) -> str:
        """Same prefix, same key; a busy prefix is spread over a few keys to stay under the rate."""
        sha = hashlib.sha1((system_prompt or "").encode("utf-8", "surrogatepass")).hexdigest()[:10]
        now = self._clock()
        with self._lock:
            seen = self._cache_seen.setdefault(sha, deque())
            while seen and now - seen[0] > _CACHE_WINDOW_SECONDS:
                seen.popleft()
            seen.append(now)
            shards = min(self.cache_max_shards, max(1, math.ceil(len(seen) / _CACHE_RPM_PER_KEY)))
            turn = self._cache_turn.get(sha, 0)
            self._cache_turn[sha] = turn + 1
        return f"sommelier-{sha}-{turn % shards}"

    def _body(self, system_prompt, user_message, max_new_tokens) -> bytes:
        body = {"model": self.api_model,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_message}],
                "max_completion_tokens": max(int(max_new_tokens), self.completion_floor)}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.prompt_cache:
            body["prompt_cache_key"] = self._cache_key(system_prompt)
            if self.prompt_cache_retention:
                body["prompt_cache_retention"] = self.prompt_cache_retention
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def _chat(self, system_prompt, user_message, max_new_tokens, thinking) -> Tuple[str, dict]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ApiError(f"environment variable {self.api_key_env} is not set")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = self._body(system_prompt, user_message, max_new_tokens)
        last = "no attempt made"
        for attempt in range(self.max_retries + 1):
            try:
                with self._slots:
                    status, raw, response_headers = self._transport(url, body, headers, self.timeout_seconds)
            except Exception as exc:                      # network error, timeout
                status, raw, response_headers = None, b"", {}
                last = f"{type(exc).__name__}"
            else:
                if status == 200:
                    return self._parse(raw)
                last = f"HTTP {status}: {self._message(raw)}".replace(key, "***")
                if status == 400 and self.prompt_cache and "prompt_cache" in last:
                    # This endpoint or model does not take the caching fields: stop sending them.
                    self.prompt_cache = False
                    if self.logger:
                        self.logger.warning("[api-llm] prompt caching fields rejected; sending without them")
                    return self._chat(system_prompt, user_message, max_new_tokens, thinking)
                if status not in _RETRY_STATUS:
                    self._count(failed=True)
                    raise ApiError(last)
            if attempt < self.max_retries:
                self._sleep(self._delay(attempt, response_headers))
        self._count(failed=True)
        raise ApiError(f"gave up after {self.max_retries + 1} attempt(s): {last}")

    @staticmethod
    def _delay(attempt: int, headers: dict) -> float:
        try:
            retry_after = float({k.lower(): v for k, v in headers.items()}.get("retry-after", ""))
        except (TypeError, ValueError):
            retry_after = 0.0
        return max(retry_after, min(30.0, 2.0 ** attempt) + random.uniform(0, 0.5))

    @staticmethod
    def _message(raw: bytes) -> str:
        try:
            error = json.loads(raw.decode("utf-8", "replace")).get("error")
            text = error.get("message") if isinstance(error, dict) else str(error)
        except (ValueError, AttributeError):
            text = raw.decode("utf-8", "replace")
        return " ".join(str(text or "").split())[:200]

    def _parse(self, raw: bytes) -> Tuple[str, dict]:
        try:
            data = json.loads(raw.decode("utf-8"))
            choice = data["choices"][0]
            text = choice["message"].get("content")
        except (ValueError, KeyError, IndexError, AttributeError, TypeError):
            self._count(failed=True)
            raise ApiError("the reply is not a chat completion")
        usage = data.get("usage") or {}
        detail = {"finish_reason": choice.get("finish_reason"),
                  "prompt_tokens": usage.get("prompt_tokens"),
                  "completion_tokens": usage.get("completion_tokens")}
        if not text or not str(text).strip():
            self._count(failed=True)
            raise ApiError(f"empty reply (finish_reason={detail['finish_reason']})")
        self._count(usage=usage)
        return str(text), detail

    def _count(self, usage: Optional[dict] = None, failed: bool = False) -> None:
        with self._lock:
            self.stats["calls"] += 1
            if failed:
                self.stats["failures"] += 1
            if usage:
                self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                details = usage.get("prompt_tokens_details") or {}
                self.stats["cached_tokens"] += int(details.get("cached_tokens") or 0)
            calls, snapshot = self.stats["calls"], dict(self.stats)
        if self.logger and calls % _LOG_EVERY_CALLS == 0:
            self.logger.info(
                f"[api-llm] {self.api_model}: {snapshot['calls']} call(s), {snapshot['failures']} failed, "
                f"{snapshot['prompt_tokens']} prompt ({snapshot['cached_tokens']} cached, "
                f"{100 * snapshot['cached_tokens'] // max(1, snapshot['prompt_tokens'])}%) "
                f"+ {snapshot['completion_tokens']} completion tokens")


def resolve_backend(args, env_profile) -> str:
    """`--llm_backend`, then `models.llm_backend` in the profile, then local."""
    backend = (getattr(args, "llm_backend", None)
               or (env_profile.get("models", {}) or {}).get("llm_backend") or "local")
    if backend not in BACKENDS:
        raise ValueError(f"llm_backend must be one of {', '.join(BACKENDS)}, not {backend!r}")
    return backend


def resolve_stages(args, config) -> frozenset:
    """`--api_stages` (comma list), then `models.api_llm.stages`, then every stage."""
    raw = getattr(args, "api_stages", None) or config.get("stages")
    if not raw:
        return frozenset(STAGES)
    names = [n.strip() for n in (raw.split(",") if isinstance(raw, str) else raw) if str(n).strip()]
    bad = sorted(set(names) - set(STAGES))
    if bad:
        raise ValueError(f"unknown api stage(s): {', '.join(bad)}; choose from {', '.join(STAGES)}")
    return frozenset(names)


def stage_llm(pass_llm, local_llm, stage: str):
    """The LLM object `stage` should use: the API one only when that stage was asked for."""
    if pass_llm is local_llm:
        return local_llm
    return pass_llm if stage in pass_llm.stages else local_llm


def build_pass_llm(args, env_profile, local_llm, logger=None):
    """The object the LLM passes talk to: `local_llm` itself unless the API backend is chosen.

    Raises at startup when the API backend is chosen but cannot be used, rather than letting
    every pass quietly skip its work later.
    """
    if resolve_backend(args, env_profile) == "local":
        return local_llm
    config = dict((env_profile.get("models", {}) or {}).get("api_llm", {}) or {})
    unknown = sorted(set(config) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) in models.api_llm: {', '.join(unknown)}")
    stages = resolve_stages(args, config)
    config.pop("stages", None)
    model = getattr(args, "api_model", None) or config.pop("model", None)
    config.pop("model", None)
    checkpoint_name = config.pop("checkpoint_model_name", None) or getattr(local_llm, "model_name", None)
    llm = ApiLLM(model, checkpoint_model_name=checkpoint_name, logger=logger, **config)
    llm.stages = stages
    if not llm.ensure_loaded():
        raise RuntimeError(
            f"llm_backend is 'api' but the environment variable {llm.api_key_env} is not set")
    if logger:
        local = [n for n in STAGES if n not in stages]
        logger.info(
            f"LLM stages {', '.join(n for n in STAGES if n in stages)} call the API model "
            f"{llm.api_model} at {llm.base_url}"
            + (f"; {', '.join(local)} stay on the local model" if local else "")
            + f". Checkpoints stay under the name {llm.model_name}, "
            f"so work already done is reused and only unfinished files are sent")
    return llm
