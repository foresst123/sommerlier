# Music (BS-RoFormer) Async Post-Processing — Design Spec

**Date:** 2026-09-23
**Status:** Draft, pending user review

## Problem

`MusicService.strip_music_spans()` (the only path in production, since
`config.json` keeps `music_scope: "spans"`) already releases a BS-RoFormer
checkout right after its raw GPU call (`ad52c37`, merged into
`solid-architecture`) instead of holding it through post-processing. But the
thread that does the raw call is the **same thread** that then runs
`postprocess_separated()`/`separate_span_postprocess()` (pure CPU) — just
without holding the model lock anymore. That orchestration thread is one of
`ThreadPoolExecutor(max_workers=min(len(pool_models), len(indexed_jobs)))`,
created fresh **inside every `strip_music_spans()` call** (i.e. once per
file) and torn down at the end of that call — not a pool that persists across
the whole "music" stage-major pass.

So a freed BS-RoFormer instance is only picked up immediately by a **new**
raw call if some other thread is already waiting on the shared checkout
queue at that moment — either another orchestration thread from the *same*
file (only exists when that file has more spans than model instances) or a
concurrently-running *second file*'s own orchestration threads (`music`'s
`cross_file_overlap`/2-file concurrency). A file with few spans (1-2, common)
running without a concurrent second file has no such thread waiting, so the
model can sit idle while the one thread that just freed it is busy doing
CPU-only post-processing instead of immediately being available to accept
new GPU work from elsewhere.

This is the same class of problem diarization's `diarize_raw`/
`diarize_postprocess` split (2026-09-23) just fixed, and separation's
`_async_runtime()`/`gpu_executor`/`post_executor` already solved for Sidon.

## Goal

Give BS-RoFormer the same persistent, stage-lifetime GPU/post split Sidon
already has, so a freed model instance is immediately available to any
pending raw job, not gated on whichever thread happened to free it also
finishing its own CPU work first.

## Non-goals

- `strip_full_recording()` stays untouched. `music_scope` is `"spans"` in
  both profiles (a separate, already-known, not-yet-fixed regression); this
  spec targets `strip_music_spans()` only, per the explicit constraint
  carried over from the dual-GPU music design (`feat/music-dual-gpu-pipeline`).
- No change to `_checkout_queue()`'s semantics (one job per model instance at
  a time) or to `BSRoformerRemover.separate_raw`/`separate_span_raw`/
  `postprocess_separated`/`separate_span_postprocess` themselves.
- No change to any stage besides "music".

## Architecture

Mirror `SeparationService._async_state`/`_async_runtime()`/
`close_async_pools()` (`services/separation_service.py:223-318`) on
`MusicService`:

```python
# MusicService.__init__
self.performance_config = dict(performance_config or {})
self._async_state = {
    "gpu_executor": None, "post_executor": None,
    "lock": threading.Lock(),
}

def _async_runtime(self):
    """Return shared GPU/post executors, or None to use the old inline path."""
    cfg = self.performance_config
    if not cfg.get("enabled", False) or not cfg.get("ordered_postprocess", True):
        return None
    model = self.bs_roformer
    pool_models = getattr(model, "models", None) or ([model] if model else [])
    if not pool_models or not all(
            callable(getattr(m, "separate_raw", None)) for m in pool_models):
        return None
    state = self._async_state
    with state["lock"]:
        if state["gpu_executor"] is None:
            state["gpu_executor"] = ThreadPoolExecutor(
                max_workers=len(pool_models), thread_name_prefix="music-gpu")
            post_workers = max(1, int(cfg.get("postprocess_workers", 2)))
            state["post_executor"] = ThreadPoolExecutor(
                max_workers=post_workers, thread_name_prefix="music-post")
        return state["gpu_executor"], state["post_executor"]

def close_async_pools(self):
    state = getattr(self, "_async_state", None)
    if state is None:
        return
    with state["lock"]:
        gpu_executor, state["gpu_executor"] = state["gpu_executor"], None
        post_executor, state["post_executor"] = state["post_executor"], None
    if gpu_executor is not None:
        gpu_executor.shutdown(wait=True, cancel_futures=False)
    if post_executor is not None:
        post_executor.shutdown(wait=True, cancel_futures=False)
```

`gpu_executor`'s size is `len(pool_models)` (one thread per physical GPU
instance) rather than an independently-tunable number: the checkout queue
already gates "one raw job per model instance at a time", so more GPU-pool
threads than model instances would only mean extra threads blocked on
`available.get()`, buying nothing. `post_executor`'s size is the new
`postprocess_workers` config key, independent and tunable (CPU-only work has
no such 1-per-model ceiling).

Since `parallel_stage_view("music")` only shallow-copies `MusicService`
(`copy.copy(self.music_svc)` in `pipeline_service.py`), `_async_state` — a
mutable dict, not rebound per copy — is shared across every concurrent
file's view, exactly like `_checkout` already is. The pools genuinely span
every file in the "music" stage-major pass, not one pair per file.

## `strip_music_spans()` change

Split `separate_job()` into a raw step and a post step, keeping the existing
legacy fallback (`hasattr(model, "separate_raw")` false → old, single-call,
checkout-held-throughout path, needed for test doubles/plugins without the
split — same contract `ad52c37` already established):

```python
def _raw_step(item):
    ordinal, (start, end) = item
    i, j = max(0, int(start * sr)), min(total, int(end * sr))
    if j - i < sr // 2:
        return ordinal, i, j, None, None, False, None
    reference = np.asarray(waveform[i:j], dtype=np.float32).copy()
    model = available.get()
    if not hasattr(model, "separate_raw"):
        try:
            vocals = None
            used_hi_res = False
            if source_path is not None:
                separate_span = getattr(model, "separate_span", None)
                if separate_span is not None:
                    vocals = separate_span(source_path, start, end, sr, reference)
                    used_hi_res = vocals is not None
            if vocals is None:
                vocals = model.separate_segment(reference, sr)
        finally:
            available.put(model)
        return ordinal, i, j, reference, ("legacy", vocals), used_hi_res, model

    used_hi_res = False
    raw_result = None
    try:
        if source_path is not None:
            separate_span_raw = getattr(model, "separate_span_raw", None)
            if separate_span_raw is not None:
                raw_result = separate_span_raw(source_path, start, end)
                used_hi_res = raw_result is not None
        if raw_result is None:
            raw_result = model.separate_raw(reference, sr)
    finally:
        available.put(model)
    return ordinal, i, j, reference, ("raw", raw_result), used_hi_res, model

def _post_step(step):
    ordinal, i, j, reference, payload, used_hi_res, model = step
    if reference is None:
        return _finish_job(ordinal, i, j, None, False)
    kind, data = payload
    if kind == "legacy":
        return _finish_job(ordinal, i, j, data, used_hi_res)
    if used_hi_res:
        vocals = model.separate_span_postprocess(data, reference, sr)
    else:
        vocals = model.postprocess_separated(data, reference, sr)
        if vocals is None:
            vocals = reference
    return _finish_job(ordinal, i, j, vocals, used_hi_res)
```

Note `_post_step` calls `model.postprocess_separated(...)` on the **same**
model reference `_raw_step` used, purely for convenience — the method is
stateless CPU math and does not need the checkout (already released by the
time `_post_step` runs), so which pool instance's method object is invoked
is irrelevant to correctness.

Orchestration (`separated` must end up holding every job's
`(ordinal, start_sample, vocals, used_hi_res)` tuple; **order does not
matter** — the existing caller already does `for ... in sorted(separated):`
by `ordinal`):

```python
indexed_jobs = list(enumerate(jobs))
async_runtime = self._async_runtime()
if async_runtime is not None and pool_models and len(indexed_jobs) > 1:
    gpu_executor, post_executor = async_runtime
    raw_futures = [gpu_executor.submit(_raw_step, item) for item in indexed_jobs]
    post_futures = [post_executor.submit(_post_step, f.result()) for f in raw_futures]
    separated = [f.result() for f in post_futures]
elif pool_models and len(indexed_jobs) > 1:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(pool_models), len(indexed_jobs))) as executor:
        separated = list(executor.map(lambda item: _post_step(_raw_step(item)), indexed_jobs))
else:
    separated = [_post_step(_raw_step(item)) for item in indexed_jobs]
```

The middle list comprehension (`[post_executor.submit(_post_step, f.result()) ...]`)
blocks on each `raw_future.result()` in submission order on the calling
thread, but since `gpu_executor` already runs up to `len(pool_models)` raw
jobs concurrently regardless of when `.result()` is called, this does not
serialise the GPU work itself — it only paces how quickly post-processing
gets *queued*, which is cheap. The `async_runtime is None` branch (tests
without `performance_config`, or the config disabled) is byte-for-byte
today's existing two paths, so every existing test that does not set
`performance_config` is unaffected.

## Config

Add to both `kaggle` and `a100` in `config.json`, under
`performance.stages.music`:
```json
"ordered_postprocess": true,
"postprocess_workers": 2
```

Wire `performance_config` into `MusicService`'s constructor call in
`main.py`, mirroring `SeparationService`'s:
```python
music_svc = MusicService(
    model_loader=model_loader,
    logger=logger,
    performance_config={
        **perf_cfg["stages"]["music"],
        "enabled": perf_cfg["enabled"],
    },
)
```

## Cleanup

Extend `utils/batch.py`'s existing end-of-run safety net (already closing
`separation_svc`'s 3 pools and `diarization_svc.close_postprocess_pool()`)
with one more line: `music_svc.close_async_pools()` if present and callable.

## Testing plan

- `tests/test_music_full_strip.py` (already has fakes for the raw/post
  split, e.g. `_SlowPostFake`): add a test proving that with
  `performance_config={"enabled": True, "ordered_postprocess": True}`, a
  second span's raw call starts before the first span's post-processing
  (blocked on a `threading.Event`) finishes — proof of the actual fix,
  single file, no cross-file overlap needed.
- A regression test that with `performance_config` unset or
  `{"enabled": False}`, behaviour is byte-for-byte the existing inline path
  (same tests that exist today must keep passing unmodified).
- A test that the legacy (`hasattr(model, "separate_raw")` false) path still
  works when `_async_runtime()` is active (async_runtime becomes None in
  this case anyway, since the gate checks `separate_raw` on every pool
  model — confirm this explicitly).
- `tests/test_batch.py`: extend the safety-net test to also assert
  `music_svc.close_async_pools()` is called.

## Summary of touched files

- `services/music_service.py` — `_async_state`/`_async_runtime`/
  `close_async_pools`, `_raw_step`/`_post_step` split in
  `strip_music_spans()`.
- `main.py` — wire `performance_config` into `MusicService(...)`.
- `config.json` — `ordered_postprocess`/`postprocess_workers` under
  `performance.stages.music`, both profiles.
- `utils/batch.py` — extend the safety net.
- Tests as listed above.
