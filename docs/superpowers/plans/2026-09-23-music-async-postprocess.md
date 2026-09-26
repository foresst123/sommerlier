# Music Async Post-Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give BS-RoFormer's `strip_music_spans()` the same persistent, stage-lifetime GPU/post executor split Sidon already has, so a freed model instance is picked up by the next pending raw job immediately, not gated on whichever thread freed it also finishing its own CPU post-processing first.

**Architecture:** `MusicService` gains `_async_state`/`_async_runtime()`/`close_async_pools()`, mirroring `SeparationService`. `strip_music_spans()`'s per-job work splits into `_raw_step()` (GPU, releases checkout immediately) and `_post_step()` (CPU), each submitted to its own persistent `ThreadPoolExecutor` when the async runtime is available; otherwise falls back to today's exact behaviour.

**Tech Stack:** Python, `concurrent.futures.ThreadPoolExecutor`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-23-music-async-postprocess-design.md`

## Global Constraints

- `strip_full_recording()` is untouched. Only `strip_music_spans()` changes.
- When `performance_config` is unset or `enabled`/`ordered_postprocess` is false, or the model(s) lack `separate_raw`, behaviour must be byte-for-byte identical to today (every existing test that does not pass `performance_config` must keep passing unmodified).
- `gpu_executor`'s worker count is `len(pool_models)`, never independently configurable.
- No stage besides "music" changes.
- Use `threading.Event` for any test proving one thread waits on / is unblocked by another (`condition-based-waiting`, never `sleep()` as the actual assertion mechanism).

---

### Task 1: `_async_state`/`_async_runtime`/`close_async_pools` + the raw/post split in `strip_music_spans`

**Files:**
- Modify: `services/music_service.py`
- Test: `tests/test_music_full_strip.py`

**Interfaces:**
- Consumes: `BSRoformerRemover.separate_raw`/`separate_span_raw`/`postprocess_separated`/`separate_span_postprocess` (already exist, unchanged).
- Produces: `MusicService.__init__(..., performance_config=None)`; `MusicService._async_runtime() -> (ThreadPoolExecutor, ThreadPoolExecutor) | None`; `MusicService.close_async_pools() -> None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_music_full_strip.py` (after `test_a_model_without_the_split_still_works_with_the_checkout_held_throughout`):

```python
class _PoolFake:
    """Looks like a BSRoformerPool: exposes `.models` so _async_runtime()'s
    gate (getattr(model, "models", None)) treats it as a pool."""
    def __init__(self, model):
        self.models = [model]


class _TrackedModel:
    """separate_raw() is instant and counts its own calls; the FIRST
    postprocess_separated() call blocks on `gate` until released, so a test
    can observe whether a second job's raw call ran while it was blocked."""

    def __init__(self, gate):
        self.gate = gate
        self._lock = threading.Lock()
        self.raw_count = 0
        self.first_post_started = threading.Event()
        self.second_raw_started = threading.Event()

    def separate_raw(self, audio, sr):
        with self._lock:
            self.raw_count += 1
            count = self.raw_count
        if count == 2:
            self.second_raw_started.set()
        return (np.asarray(audio, dtype=np.float32), sr, False)

    def postprocess_separated(self, raw, audio_array, sample_rate):
        if not self.first_post_started.is_set():
            self.first_post_started.set()
            self.gate.wait(timeout=5)
        out, _out_sr, _stereo_in = raw
        return np.asarray(out, dtype=np.float32)


def test_a_second_raw_call_starts_while_the_first_jobs_postprocess_is_still_blocked():
    """The whole point of a persistent gpu/post split: with the async runtime
    enabled, the model instance the first span just freed must be picked up
    by a SECOND span's raw call immediately, without waiting for the first
    span's post-processing (now on a different thread) to finish."""
    gate = threading.Event()
    model = _TrackedModel(gate)
    pool = _PoolFake(model)
    svc = MusicService(pool, logger=None,
                       performance_config={"enabled": True, "ordered_postprocess": True})
    audio = _audio(seconds=20)
    music_map = MusicMap([(2.0, 3.0, MUSIC), (10.0, 11.0, MUSIC)])

    result_holder = {}

    def run():
        result_holder["patches"] = svc.strip_music_spans(audio, music_map, source_path=None)

    t = threading.Thread(target=run)
    t.start()
    assert model.first_post_started.wait(timeout=2), \
        "the first job's postprocess must have started"
    assert model.second_raw_started.wait(timeout=2), (
        "the second span's raw call never ran while the first job's "
        "postprocess was still blocked -- the GPU instance was not freed "
        "for new work")

    gate.set()
    t.join(timeout=5)
    svc.close_async_pools()
    assert result_holder["patches"], "both spans should still produce patches"


def test_async_runtime_is_off_when_a_model_in_the_pool_lacks_the_split():
    """A pool holding even one old-style model (no separate_raw) must not
    turn the async path on -- there is no raw step to schedule for it."""
    class _OldStyleFake:
        def separate_segment(self, audio, sr):
            return (np.asarray(audio) * 0.5).astype(np.float32)

    pool = _PoolFake(_OldStyleFake())
    svc = MusicService(pool, logger=None,
                       performance_config={"enabled": True, "ordered_postprocess": True})
    assert svc._async_runtime() is None


def test_async_runtime_is_off_when_performance_config_is_not_enabled():
    model = _TrackedModel(threading.Event())
    pool = _PoolFake(model)
    svc = MusicService(pool, logger=None)  # no performance_config at all
    assert svc._async_runtime() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_music_full_strip.py -k "second_raw_call or async_runtime_is_off" -q`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'performance_config'`, and `AttributeError: 'MusicService' object has no attribute '_async_runtime'`).

- [ ] **Step 3: Implement**

In `services/music_service.py`, add the import and update `__init__`:

```python
from concurrent.futures import ThreadPoolExecutor
```

```python
def __init__(self, bs_roformer_model=None, logger=None, model_loader=None,
            performance_config=None):
    self._bs_roformer = bs_roformer_model
    self.model_loader = model_loader
    self.logger = logger
    self._checkout = {"lock": threading.Lock(), "key": None, "queue": None}
    # Runs postprocess_separated()/separate_span_postprocess() in the
    # background during strip_music_spans(), so a BS-RoFormer instance a
    # raw call just freed is picked up by the next pending raw job
    # immediately, not gated on that same thread finishing its own CPU
    # work first. Mirrors SeparationService._async_state. Shared (not
    # per-file) the same way self._checkout already is --
    # parallel_stage_view("music") only shallow-copies MusicService.
    self.performance_config = dict(performance_config or {})
    self._async_state = {
        "gpu_executor": None, "post_executor": None,
        "lock": threading.Lock(),
    }
```

Add near `_checkout_queue`:

```python
def _async_runtime(self):
    """Return shared (gpu_executor, post_executor), or None to use the
    plain inline path (performance off, or a model in the pool predates
    the raw/post split)."""
    cfg = self.performance_config
    if not cfg.get("enabled", False) or not cfg.get("ordered_postprocess", True):
        return None
    model = self.bs_roformer
    if model is None:
        return None
    pool_models = getattr(model, "models", None) or [model]
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
    """Shut down the background pools. Safe to call with nothing ever
    submitted, and safe to call twice."""
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

Now restructure `strip_music_spans()`'s job execution. Replace the existing `separate_job(item)` function and the `indexed_jobs`/execution block (from `def separate_job(item):` through `separated = [separate_job(item) for item in indexed_jobs]`) with:

```python
def _raw_step(item):
    ordinal, (start, end) = item
    i, j = max(0, int(start * sr)), min(total, int(end * sr))
    if j - i < sr // 2:
        # Under half a second there is not enough for the separator to
        # work with, and the seams would cost more than the bed does.
        return ordinal, i, j, None, None, False, None
    reference = np.asarray(waveform[i:j], dtype=np.float32).copy()
    model = available.get()

    if not hasattr(model, "separate_raw"):
        # Predates the raw/post split (a test double, or a custom
        # separator plugin) -- run the old, single-call interface with
        # the checkout held for the whole job, exactly as before the
        # split.
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

    # The raw/post split is available: release the instance right after
    # its GPU work finishes, instead of holding it through CPU-only
    # postprocessing too -- see _checkout_queue's own docstring for why
    # this matters across concurrent files sharing this same queue.
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
            # separate_segment()'s own contract: stay mixture rather
            # than go silent, since silence would enter the dataset
            # labelled as speech.
            vocals = reference
    return _finish_job(ordinal, i, j, vocals, used_hi_res)

indexed_jobs = list(enumerate(jobs))
async_runtime = self._async_runtime()
if async_runtime is not None and pool_models and len(indexed_jobs) > 1:
    gpu_executor, post_executor = async_runtime
    raw_futures = [gpu_executor.submit(_raw_step, item) for item in indexed_jobs]
    post_futures = [post_executor.submit(_post_step, f.result()) for f in raw_futures]
    separated = [f.result() for f in post_futures]
elif pool_models and len(indexed_jobs) > 1:
    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=min(len(pool_models), len(indexed_jobs))) as executor:
        separated = list(executor.map(lambda item: _post_step(_raw_step(item)), indexed_jobs))
else:
    separated = [_post_step(_raw_step(item)) for item in indexed_jobs]
```

`separated`'s tuples are `(ordinal, start_sample, vocals, used_hi_res)`, matching what `_finish_job` already returns and what the loop right after this block (`for _ordinal, start_sample, vocals, used_hi_res in sorted(separated):`) already expects -- unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_music_full_strip.py -q`
Expected: PASS, all tests (the 3 new ones and every pre-existing one, including `test_a_span_releases_its_checkout_before_postprocessing_finishes` and `test_a_model_without_the_split_still_works_with_the_checkout_held_throughout`, unaffected since they never pass `performance_config`).

- [ ] **Step 5: Run the broader music test suite for regressions**

Run: `python -m pytest tests/test_music_separator.py tests/test_excise.py tests/test_performance_gating.py -q`
Expected: same pass/fail counts as the pre-Task-1 baseline.

- [ ] **Step 6: Commit**

```bash
git add services/music_service.py tests/test_music_full_strip.py
git commit -m "feat(music): async GPU/post split for strip_music_spans's BS-RoFormer calls

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Wire `performance_config` into `MusicService` and add config keys

**Files:**
- Modify: `main.py`, `config.json`

**Interfaces:**
- Consumes: `MusicService.__init__`'s new `performance_config` parameter (Task 1).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_scheduling.py` (or a suitable existing source-string test file):

```python
def test_music_service_is_constructed_with_a_performance_config():
    src = _source("main.py")
    call = src[src.index("music_svc = MusicService("):src.index(")", src.index("music_svc = MusicService("))]
    assert "performance_config" in call, (
        "MusicService must receive performance_config the same way "
        "SeparationService/ASRService already do, or the async runtime "
        "can never turn on in production")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stage_scheduling.py -k music_service_is_constructed -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `main.py`, change:
```python
music_svc = MusicService(
    model_loader=model_loader,
    logger=logger
)
```
to:
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
(matches `separation_svc`'s construction immediately above it -- same `perf_cfg` variable already in scope there).

In `config.json`, under `environments.kaggle.performance.stages.music` and
`environments.a100.performance.stages.music`, add:
```json
"ordered_postprocess": true,
"postprocess_workers": 2
```
(alongside the existing `cross_file_overlap`/`max_separator_workers`/`tagger_workers` keys in that same block).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_stage_scheduling.py -q`
Expected: PASS, all tests.

- [ ] **Step 5: Validate config.json is still valid JSON and run the config-driven tests**

Run: `python -c "import json; json.load(open('config.json'))" && python -m pytest tests/test_config_driven.py -q`
Expected: JSON loads without error; same pass/fail counts as the pre-Task-2 baseline for `test_config_driven.py` (this file has pre-existing unrelated failures -- do not chase those).

- [ ] **Step 6: Commit**

```bash
git add main.py config.json tests/test_stage_scheduling.py
git commit -m "feat(music): wire performance_config into MusicService, add ordered_postprocess config

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Close the pools at the end of the batch

**Files:**
- Modify: `utils/batch.py`
- Test: `tests/test_batch.py`

**Interfaces:**
- Consumes: `MusicService.close_async_pools()` (Task 1).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_batch.py`:

```python
def test_the_safety_net_also_closes_the_music_async_pools():
    class _FakeMusicPool:
        def __init__(self):
            self.closed = False

        def close_async_pools(self):
            self.closed = True

    pipe = FakePipeline()
    pipe.music_svc = _FakeMusicPool()
    run_batch_by_stage(pipe, _args(stop_after="music"), {}, ["f1"])
    assert pipe.music_svc.closed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_batch.py -k music_async_pools -q`
Expected: FAIL (`AssertionError: assert False`).

- [ ] **Step 3: Implement**

In `utils/batch.py`, extend the end-of-run safety net (right after the `diarization_svc.close_postprocess_pool()` block added for diarization):

```python
    music_svc = getattr(pipeline, "music_svc", None)
    close_music_pools = getattr(music_svc, "close_async_pools", None)
    if callable(close_music_pools):
        close_music_pools()

    return list(failures.items())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_batch.py -q`
Expected: PASS, all tests.

- [ ] **Step 5: Run the full suite for the overall regression count**

Run: `python -m pytest tests -q`
Expected: same known-failing count as the current baseline (100 failed), only new tests added on top, zero new failures among pre-existing tests.

- [ ] **Step 6: Commit**

```bash
git add utils/batch.py tests/test_batch.py
git commit -m "fix(batch): close the music async pools at the end of a run

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Push

- [ ] **Step 1:** `git push origin solid-architecture`
- [ ] **Step 2:** Verify: `git fetch origin solid-architecture && git rev-parse solid-architecture origin/solid-architecture` -- both SHAs must match.
