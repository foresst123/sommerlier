# Diarization Async Post-Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the DiariZen GPU worker sitting idle while its own thread does the CPU-only post-processing tail (VAD/merge/split/write) of the file it just finished, by splitting diarization into a raw GPU call and a background CPU post-processing call, deferred only during the diarization stage-major pass.

**Architecture:** `DiarizationService.run_diarization()` splits into `diarize_raw()` (GPU) and `diarize_postprocess()` (CPU), joined by a persistent per-service `ThreadPoolExecutor` (`submit_postprocess()`/`close_postprocess_pool()`), mirroring `SeparationService._async_runtime()`. `PipelineService.run()` defers `diarize_postprocess()` + checkpoint/write/prefetch to that pool only when `stop_after == "diarization"`, tracking the future so `utils/batch.py`'s `run_batch_by_stage()` can drain it (barrier) before the next stage starts and route a failure into the normal `failures` dict.

**Tech Stack:** Python, `concurrent.futures.ThreadPoolExecutor`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-23-diarization-async-postprocess-design.md`

## Global Constraints

- Only the `stop_after == "diarization"` branch of `PipelineService.run()` changes behaviour. File-major runs and the final checkpoint-restoring pass must be byte-for-byte the same control flow as today (just calling `diarize_postprocess(diarize_raw(...))` instead of the old monolithic method).
- Separation is out of scope — do not touch `separation_service.py`'s `_async_runtime`/`process_overlaps`, it already has this pattern.
- No stage other than diarization changes.
- A postprocess failure must end up in `run_batch_by_stage`'s returned `failures` dict with the format `f"diarization: {type(e).__name__}: {e}"`, and the file must be excluded from every later stage's `pending` list — exactly like a synchronous failure is today.
- Follow existing test conventions in this codebase: `tests/test_stage_scheduling.py`'s source-string assertions (`_source("services/pipeline_service.py")`, substring/`index()` checks) are how this codebase already tests `run()`'s control flow around `stop_after`/`prefetch_overlap_plan` — new control-flow tests for `run()` use the same style, not a fully-constructed `PipelineService.run()` call. Use `condition-based-waiting` (a `threading.Event`, never `sleep()`) for any test proving one thread waits on another.

---

### Task 1: Split `run_diarization` into `diarize_raw`/`diarize_postprocess`, add the post-processing pool

**Files:**
- Modify: `services/diarization_service.py`
- Test: `tests/test_diarization_service.py` (new file)

**Interfaces:**
- Produces: `DiarizationService.diarize_raw(chunks, audio, args) -> _RawDiarization` (module-level dataclass, fields `combined_df: pd.DataFrame`, `method: str`); `DiarizationService.diarize_postprocess(raw: _RawDiarization, audio, args) -> DiarizationResult`; `DiarizationService.submit_postprocess(raw, audio, args) -> concurrent.futures.Future[DiarizationResult]`; `DiarizationService.close_postprocess_pool() -> None`. `run_diarization()` keeps its existing signature and becomes `return self.diarize_postprocess(self.diarize_raw(chunks, audio, args), audio, args)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diarization_service.py`:

```python
"""DiarizationService's raw/post split -- see
docs/superpowers/specs/2026-09-23-diarization-async-postprocess-design.md

Run:  python -m pytest tests/test_diarization_service.py -q   (from podcast-pipeline/)
"""
import os
import sys
import types

import numpy as np
import pytest
from pyannote.core import Annotation, Segment as PyannoteSegment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.audio import AudioData
from services.diarization_service import DiarizationService


class _FakeDiarizer:
    """Returns a fixed pyannote Annotation, ignoring speaker-bound kwargs."""

    def __init__(self, tracks):
        self._tracks = tracks  # [(start, end, speaker), ...]
        self.calls = 0

    def diarize(self, audio_input, **kwargs):
        self.calls += 1
        annotation = Annotation()
        for start, end, speaker in self._tracks:
            annotation[PyannoteSegment(start, end)] = speaker
        return annotation


def _audio(duration=20.0, sr=16000):
    return AudioData(
        waveform=np.zeros(int(duration * sr), dtype=np.float32),
        sample_rate=sr, name="test", audio_segment=None, duration=duration)


def _args(**kw):
    a = types.SimpleNamespace(dia3=False, vad=False, merge_gap=0.5,
                              max_segment_length=30.0, bridge_gap=3.0)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _service(tracks):
    svc = DiarizationService(diarizer=_FakeDiarizer(tracks))
    return svc


def test_diarize_raw_does_not_run_any_cpu_postprocessing():
    """The GPU-only half must not touch filter/VAD/merge/split at all."""
    import services.diarization_service as mod
    patched = ("filter_diarizer_noise", "cut_by_speaker_label",
               "bridge_interrupted_speaker_turns", "split_long_segments")
    originals = {name: getattr(mod, name) for name in patched}
    calls = []

    def _make_tracker(name, original):
        def _tracker(*a, **kw):
            calls.append(name)
            return original(*a, **kw)
        return _tracker

    for name in patched:
        setattr(mod, name, _make_tracker(name, originals[name]))
    try:
        svc = _service([(0.0, 2.0, "A"), (1.5, 3.0, "B")])
        raw = svc.diarize_raw([], _audio(), _args())
    finally:
        for name in patched:
            setattr(mod, name, originals[name])
    assert calls == [], f"diarize_raw must be GPU-only, but it ran: {calls}"
    assert raw.method == "diarizen"
    assert len(raw.combined_df) == 2


def test_diarize_postprocess_reproduces_run_diarization_output():
    """diarize_postprocess(diarize_raw(...)) must equal today's run_diarization()."""
    tracks = [(0.0, 2.0, "A"), (1.5, 3.0, "B"), (5.0, 5.05, "A")]
    svc_a = _service(tracks)
    svc_b = _service(tracks)
    audio = _audio()
    args = _args()

    raw = svc_a.diarize_raw([], audio, args)
    via_split = svc_a.diarize_postprocess(raw, audio, args)
    via_monolith = svc_b.run_diarization([], audio, args)

    assert [ (s.start, s.end, s.speaker) for s in via_split.segments ] == \
           [ (s.start, s.end, s.speaker) for s in via_monolith.segments ]
    assert via_split.num_speakers == via_monolith.num_speakers
    assert via_split.method == via_monolith.method == "diarizen"


def test_submit_postprocess_runs_in_the_background_and_resolves():
    svc = _service([(0.0, 2.0, "A"), (1.5, 3.0, "B")])
    audio = _audio()
    args = _args()
    raw = svc.diarize_raw([], audio, args)

    future = svc.submit_postprocess(raw, audio, args)
    result = future.result(timeout=5)
    assert len(result.segments) >= 1
    svc.close_postprocess_pool()


def test_close_postprocess_pool_is_a_safe_no_op_before_any_submit_and_when_called_twice():
    svc = _service([])
    svc.close_postprocess_pool()
    svc.close_postprocess_pool()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_diarization_service.py -q`
Expected: FAIL (`AttributeError: 'DiarizationService' object has no attribute 'diarize_raw'`, etc. -- the methods do not exist yet).

- [ ] **Step 3: Implement the split**

In `services/diarization_service.py`, add the imports and dataclass near the top (after the existing imports, before `ENABLE_GHOST_MERGE = False`):

```python
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class _RawDiarization:
    """The GPU half's output: the raw per-track dataframe, nothing filtered,
    merged or split yet. Consumed by diarize_postprocess()."""
    combined_df: "pd.DataFrame"
    method: str
```

Replace the body of `run_diarization` (from `is_diarizen = not getattr(args, "dia3", False)` through `self._log_segment_stats("raw", df_to_list(combined_df))`, i.e. everything up to but NOT including the `# Drop clustering-jitter blips...` comment) by extracting it into `diarize_raw`, and turn the rest into `diarize_postprocess`. Concretely:

```python
def diarize_raw(self, chunks: List[DiarizationChunk], audio: AudioData, args: Any) -> "_RawDiarization":
    """GPU-only half of diarization: the model call and building the raw
    per-track dataframe. No filtering, merging or splitting happens here --
    see diarize_postprocess(). Safe to call from the thread that will hand
    the GPU worker straight back to the next file; postprocessing can then
    run in the background (see submit_postprocess)."""
    is_diarizen = not getattr(args, "dia3", False)
    import pandas as pd
    import torch

    audio_input = {
        "waveform": torch.from_numpy(audio.waveform).unsqueeze(0),
        "sample_rate": audio.sample_rate
    }

    if self.logger:
        self.logger.info(f"Running diarization on the full audio file natively (Duration: {audio.duration:.2f}s)...")

    num_speakers = self.diarizer_config.get("num_speakers")
    min_speakers = self.diarizer_config.get("min_speakers")
    max_speakers = self.diarizer_config.get("max_speakers")

    if is_diarizen:
        if self.logger:
            self.logger.info("Diarizing with DiariZen (speaker bounds applied via worker config)")
        diar_out = self.diarizer.diarize(audio_input)
    else:
        if self.logger:
            self.logger.info(
                f"Diarizing with pyannote, speaker bounds num={num_speakers} "
                f"min={min_speakers} max={max_speakers}"
            )
        diar_out = self.diarizer.diarize(
            audio_input,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    data = []
    annotation = (
        diar_out.speaker_diarization
        if hasattr(diar_out, "speaker_diarization")
        else diar_out
    )

    if annotation is not None:
        try:
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                data.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
        except Exception as e:
            if self.logger: self.logger.error(f"Error iterating diarization result: {e}")

    combined_df = pd.DataFrame(data) if data else pd.DataFrame(columns=["start", "end", "speaker"])
    combined_df = combined_df.sort_values("start").reset_index(drop=True)
    self._log_segment_stats("raw", df_to_list(combined_df))

    return _RawDiarization(combined_df=combined_df, method="diarizen" if is_diarizen else "pyannote")

def diarize_postprocess(self, raw: "_RawDiarization", audio: AudioData, args: Any) -> DiarizationResult:
    """CPU-only half: filter, seam-split, VAD, merge, bridge, length-split,
    and build the final DiarizationResult. No GPU call happens here, so this
    is safe to run on a background thread (see submit_postprocess) while the
    GPU worker diarize_raw() used is already free for the next file."""
    combined_df = raw.combined_df
    is_diarizen = raw.method == "diarizen"

    # Drop clustering-jitter blips before anything else touches the list.
    # Must run first, on the RAW diarizer output: filtering after merging
    # (the old approach, inside cut_by_speaker_label) judged a short
    # fragment by whether it happened to sit next to something it could
    # merge into, and silently dropped genuine short overlap evidence
    # otherwise. This keeps any fragment with a foreign-speaker overlap
    # regardless of length, and only drops isolated sub-200ms noise.
    #
    # 200ms is still below the shortest real overlap measured on this
    # corpus (0.24s), so a genuine backchannel survives this filter
    # either way (via the foreign-overlap exemption or simply being long
    # enough) -- raised from the original 150ms for a wider noise margin.
    raw_for_filter = df_to_list(combined_df)
    filtered_list = filter_diarizer_noise(raw_for_filter)
    dropped = len(raw_for_filter) - len(filtered_list)
    if self.logger and dropped:
        self.logger.info(
            f"Dropped {dropped} segment(s) under 200ms as diarizer noise "
            "(any with a foreign-speaker overlap was kept regardless of length)"
        )
    combined_df = pd.DataFrame(filtered_list)
    self._log_segment_stats("post-noise-filter", filtered_list)

    # ... (everything from "Cut at the joins first" through the end of the
    # current run_diarization body, UNCHANGED -- copy verbatim from the
    # current method body starting at `timeline = getattr(self, "timeline", None)`
    # through `return DiarizationResult(...)`.)

def run_diarization(self, chunks: List[DiarizationChunk], audio: AudioData, args: Any) -> DiarizationResult:
    """Run diarization model on the full audio natively.

    Thin wrapper kept for callers that want the whole thing synchronously
    (file-major runs, tests). See diarize_raw()/diarize_postprocess() for
    the split this composes -- PipelineService.run() calls those directly
    so it can defer the CPU half during the diarization stage-major pass."""
    return self.diarize_postprocess(self.diarize_raw(chunks, audio, args), audio, args)
```

Then, in `DiarizationService.__init__`, add:
```python
# Runs diarize_postprocess() in the background during the diarization
# stage-major pass, so the GPU worker diarize_raw() just freed can take
# the next file immediately instead of waiting for this file's CPU-only
# filter/VAD/merge/split/write tail. Mirrors SeparationService._async_state.
self._postprocess_state = {"executor": None, "lock": threading.Lock()}
```

And these two methods (near `run_diarization`):
```python
def _postprocess_pool(self):
    state = self._postprocess_state
    with state["lock"]:
        if state["executor"] is None:
            workers = int((self.diarizer_config or {}).get("postprocess_workers", 2))
            state["executor"] = ThreadPoolExecutor(
                max_workers=max(1, workers), thread_name_prefix="diar-post")
        return state["executor"]

def submit_postprocess(self, raw: "_RawDiarization", audio: AudioData, args: Any):
    """Run diarize_postprocess() on the shared background pool. Shared (not
    per-file) the same way SeparationService's gpu/post executors are --
    parallel_stage_view("diarization") only shallow-copies DiarizationService,
    so every concurrent file's view submits to the same pool."""
    return self._postprocess_pool().submit(self.diarize_postprocess, raw, audio, args)

def close_postprocess_pool(self):
    """Shut down the background pool. Safe to call with nothing ever
    submitted, and safe to call twice."""
    state = getattr(self, "_postprocess_state", None)
    if state is None:
        return
    with state["lock"]:
        executor, state["executor"] = state["executor"], None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_diarization_service.py -q`
Expected: PASS (4 tests). Note: the monkeypatch-tracking test above patches module-level names in `services.diarization_service` -- if `diarize_postprocess` calls the filter/merge functions via their imported module-level names (as `run_diarization` already does today, e.g. `filter_diarizer_noise(...)`), the patch takes effect; no code change needed for this to work, since Step 3 is a verbatim relocation of existing calls.

- [ ] **Step 5: Run the full existing diarization-adjacent suite**

Run: `python -m pytest tests/test_overlap_preservation.py tests/test_diarization_service.py -q`
Expected: PASS, no regressions (these test the same `utils.segment_utils` functions `diarize_postprocess` now calls, unchanged).

- [ ] **Step 6: Commit**

```bash
git add services/diarization_service.py tests/test_diarization_service.py
git commit -m "refactor(diarization): split run_diarization into diarize_raw/diarize_postprocess

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Defer post-processing during the diarization stage-major pass

**Files:**
- Modify: `services/pipeline_service.py`
- Test: `tests/test_stage_scheduling.py`

**Interfaces:**
- Consumes: `DiarizationService.diarize_raw`/`diarize_postprocess`/`submit_postprocess` from Task 1.
- Produces: `PipelineService._pending_diar_jobs: dict[str, Future]`, `PipelineService._pending_diar_lock: threading.Lock` (set in `__init__`, alongside `_model_load_lock`). These are read by Task 3's drain step in `utils/batch.py`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stage_scheduling.py` (near the other `_source("services/pipeline_service.py")` tests):

```python
def test_diarization_postprocessing_is_deferred_only_when_stopping_after_diarization():
    """The fresh-compute branch must call diarize_raw() unconditionally, but
    only defer diarize_postprocess() to the background pool inside the
    stop_after == "diarization" block -- a file-major run or the final pass
    needs the finished DiarizationResult before it can continue within the
    same call, so it must stay synchronous there."""
    src = _source("services/pipeline_service.py")
    assert "self.diarization_svc.diarize_raw(" in src
    stop = src.index('getattr(args, "stop_after", None) == "diarization"')
    deferred_block = src[stop:src.index("return None", stop)]
    assert "self.diarization_svc.submit_postprocess(" in deferred_block
    assert "add_done_callback" in deferred_block
    after_deferred_block = src[src.index("return None", stop):]
    # The synchronous fallback (for every OTHER call to this section) must
    # still call diarize_postprocess directly, outside the deferred block.
    assert "self.diarization_svc.diarize_postprocess(" in after_deferred_block[:800]


def test_the_diarization_done_callback_writes_checkpoint_and_output_before_prefetching():
    src = _source("services/pipeline_service.py")
    stop = src.index('getattr(args, "stop_after", None) == "diarization"')
    deferred_block = src[stop:src.index("return None", stop)]
    checkpoint_at = deferred_block.index('checkpoint.save("diarization"')
    write_at = deferred_block.index("stage_out.write_diarization(")
    prefetch_at = deferred_block.index("self.separation_svc.prefetch_overlap_plan(")
    assert checkpoint_at < write_at < prefetch_at, (
        "must checkpoint, then write stage-out, then prefetch -- in that order")


def test_pending_diar_jobs_is_shared_across_parallel_stage_view_copies():
    """parallel_stage_view() does copy.copy(self); _pending_diar_jobs must be
    created once in __init__ (like _model_load_lock) so every concurrent
    file's view shares the same dict, not one each."""
    src = _source("services/pipeline_service.py")
    init_src = src[src.index("def __init__"):src.index("def parallel_stage_view")]
    assert "self._pending_diar_jobs" in init_src
    assert "self._pending_diar_lock" in init_src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stage_scheduling.py -k diar -q`
Expected: FAIL (the new calls/attributes do not exist yet).

- [ ] **Step 3: Implement**

In `PipelineService.__init__`, right after `self._model_load_lock = threading.RLock()`:
```python
# Futures from a deferred diarize_postprocess() (see run()'s "diarization"
# stop-point). Created once here, not per file, so every
# parallel_stage_view("diarization") copy (a shallow copy.copy()) shares the
# same dict -- utils.batch.run_batch_by_stage drains it after the
# diarization stage's file loop, before "separation" can start reading
# checkpoints that might not be written yet.
self._pending_diar_jobs = {}
self._pending_diar_lock = threading.RLock()
```

In `run()`, replace the diarization section (the `else:` branch that freshly computes diarization, currently ending with `checkpoint.save("diarization", diarization_result)` / `computed.add("diarization")`, PLUS the unconditional `self._free(args, "diarizer", "vad")` / `self._release_worker(args, "diarizen")` lines that currently sit right after the whole if/elif/else block, PLUS the `if getattr(args, "stop_after", None) == "diarization":` block) with:

```python
else:
    self._load("base")
    self._load("diarization")
    chunks, _ = self.diarization_svc.prepare_chunks(audio_data)
    raw = self.diarization_svc.diarize_raw(chunks, audio_data, args)
    self._free(args, "diarizer", "vad")
    self._release_worker(args, "diarizen")

    if getattr(args, "stop_after", None) == "diarization":
        # Window building is pure CPU (utils/window_pool.py) and does not
        # need Sidon loaded, so it can run now, in the background, while
        # DiariZen is still busy with the rest of this stage-major pass and
        # before the separation stage has even started the Sidon worker.
        # Deferring diarize_postprocess() too means the GPU worker
        # diarize_raw() just released is free for the NEXT file immediately,
        # instead of waiting for this file's CPU-only filter/VAD/merge/
        # split/write tail. See services/diarization_service.py:submit_postprocess
        # and utils/batch.py's drain step, which blocks "separation" from
        # starting until every such future here has resolved.
        future = self.diarization_svc.submit_postprocess(raw, audio_data, args)
        with self._pending_diar_lock:
            self._pending_diar_jobs[audio_path] = future

        def _finish(fut, audio_path=audio_path, checkpoint=checkpoint,
                    stage_out=stage_out, audio_data=audio_data, args=args):
            try:
                result = fut.result()
            except Exception:
                # Surfaced to run_batch_by_stage's `failures` by the drain
                # step, which calls future.result() again on this same
                # future -- deliberately not handled here.
                return
            checkpoint.save("diarization", result)
            stage_out.write_diarization(
                result.segments,
                total_dur=audio_data.duration,
                raw_segments=getattr(result, "raw_segments", None),
                audio=audio_data.waveform,
                sample_rate=audio_data.sample_rate,
            )
            if self.step_enabled(args, "separation"):
                self.separation_svc.prefetch_overlap_plan(
                    result.segments, audio_data, audio_path)

        future.add_done_callback(_finish)
        if self.logger:
            self.logger.info(
                "Stopping pipeline after diarization as requested by "
                "--stop_after (post-processing continues in the background).")
        stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                  "stopped_after": "diarization"})
        return None

    diarization_result = self.diarization_svc.diarize_postprocess(raw, audio_data, args)
    if self.logger: self.logger.info(f"[DEBUG] Diarization returned {len(diarization_result.segments)} segments via {diarization_result.method}")

    if not diarization_result.segments:
        raise RuntimeError(
            f"Diarization ({diarization_result.method}) produced no segments for "
            f"{audio_path}. Check the worker log above for the underlying error; "
            "continuing would write an empty transcript."
        )

    checkpoint.save("diarization", diarization_result)
    computed.add("diarization")
```

Leave the `if "diarization" in computed: stage_out.write_diarization(...)` block right after this section (unchanged) -- it still handles the write for the synchronous (non-deferred) fresh-compute path, exactly as before. Note the OLD unconditional `self._free(args, "diarizer", "vad")` / `self._release_worker(args, "diarizen")` lines that used to sit after that block are now deleted (they moved inside the `else:` branch above, right after `diarize_raw`); do not leave a duplicate copy in the old spot.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_stage_scheduling.py -q`
Expected: PASS, including the 3 new tests and every pre-existing one in this file (the substring positions the old tests check -- `prefetch_overlap_plan`, `step_enabled(args, "separation")`, `close_prefetch_pool` deferral -- all still hold, since those calls are still textually inside the same `if ... stop_after ... "diarization"` block, just one level deeper inside `_finish`).

- [ ] **Step 5: Run the broader suite for regressions**

Run: `python -m pytest tests/test_worker_lifecycle.py tests/test_lazy_model_loading.py tests/test_output_layout.py tests/test_performance_gating.py -q`
Expected: same pass/fail counts as the pre-Task-2 baseline (these exercise `_free`/`_release_worker`/`_load` directly, not `run()`'s exact source layout, so the relocation in Step 3 should not affect them).

- [ ] **Step 6: Commit**

```bash
git add services/pipeline_service.py tests/test_stage_scheduling.py
git commit -m "feat(pipeline): defer diarization post-processing during the diarization stop-point

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Drain the pending futures before the next stage, and close the pool at the end

**Files:**
- Modify: `utils/batch.py`
- Test: `tests/test_batch.py`

**Interfaces:**
- Consumes: `pipeline._pending_diar_jobs` / `pipeline._pending_diar_lock` from Task 2; `pipeline.diarization_svc.close_postprocess_pool()` from Task 1.
- Produces: `utils.batch._drain_pending_diarization(pipeline, failures) -> None` (module-level helper).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_batch.py`:

```python
def test_the_next_stage_waits_for_a_slow_diarization_postprocess_future():
    """The drain barrier: 'separation' must not start for ANY file until
    every diarization postprocess future from the previous pass has
    resolved, even though run_one() already returned for that file."""
    release = threading.Event()

    class DeferredPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self._pending_diar_jobs = {}
            self._pending_diar_lock = threading.Lock()

        def run(self, args, config, path):
            stage = getattr(args, "stop_after", None)
            self.calls.append((stage, path))
            if stage == "diarization":
                from concurrent.futures import ThreadPoolExecutor
                def _slow():
                    release.wait(timeout=5)
                    return "done"
                future = ThreadPoolExecutor(max_workers=1).submit(_slow)
                self._pending_diar_jobs[path] = future

    pipe = DeferredPipeline()
    result_holder = {}

    def _run():
        result_holder["failures"] = run_batch_by_stage(
            pipe, _args(stop_after="separation"), {}, ["f1"])

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=0.3)
    assert thread.is_alive(), "separation must not start before the drain releases it"
    assert "separation" not in {s for s, _ in pipe.calls}

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert "separation" in {s for s, _ in pipe.calls}


def test_a_failing_diarization_postprocess_future_is_reported_and_excludes_the_file():
    class DeferredFailPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self._pending_diar_jobs = {}
            self._pending_diar_lock = threading.Lock()

        def run(self, args, config, path):
            stage = getattr(args, "stop_after", None)
            self.calls.append((stage, path))
            if stage == "diarization" and path == "bad":
                from concurrent.futures import ThreadPoolExecutor
                def _boom():
                    raise RuntimeError("no segments")
                self._pending_diar_jobs[path] = ThreadPoolExecutor(max_workers=1).submit(_boom)

    pipe = DeferredFailPipeline()
    failures = run_batch_by_stage(pipe, _args(), {}, ["good", "bad"])

    assert [p for p, _ in failures] == ["bad"]
    later = [p for s, p in pipe.calls if s == "separation"]
    assert later == ["good"], "the file whose postprocess failed must not reach separation"


def test_the_safety_net_also_closes_the_diarization_postprocess_pool():
    class _FakeDiarPool:
        def __init__(self):
            self.closed = False
        def close_postprocess_pool(self):
            self.closed = True

    pipe = FakePipeline()
    pipe.diarization_svc = _FakeDiarPool()
    run_batch_by_stage(pipe, _args(stop_after="diarization"), {}, ["f1"])
    assert pipe.diarization_svc.closed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_batch.py -k "drain or postprocess_pool or excludes_the_file" -q`
Expected: FAIL. The first test currently fails because nothing ever drains `_pending_diar_jobs`, so `separation` starts immediately (the thread will NOT still be alive at the 0.3s check -- `assert thread.is_alive()` fails). The second fails because "bad" is not excluded. The third fails because `close_postprocess_pool` is never called.

- [ ] **Step 3: Implement**

In `utils/batch.py`, add near the top-level helpers (after `_stage_index`, or anywhere at module scope):

```python
def _drain_pending_diarization(pipeline, failures):
    """Block until every deferred diarize_postprocess() future has resolved,
    routing a failure into `failures` exactly like a synchronous one.

    A no-op for any stage but 'diarization' (pipeline._pending_diar_jobs is
    only ever populated by PipelineService.run()'s diarization stop-point --
    see services/pipeline_service.py). Must run before the stage loop is
    allowed to move on, so 'separation' never reads a checkpoint that a
    still-running background thread has not written yet.
    """
    pending = getattr(pipeline, "_pending_diar_jobs", None)
    if not pending:
        return
    for path, future in list(pending.items()):
        try:
            future.result()
        except Exception as e:
            failures[path] = f"diarization: {type(e).__name__}: {e}"
        finally:
            pending.pop(path, None)
```

In `run_batch_by_stage`, call it right after the `if parallelism > 1 and len(pending) > 1: ... else: ...` block (still inside the outer `try`, before the `finally`):

```python
        try:
            parallelism = _stage_parallelism(stage_args, stage)

            def run_one(i, path):
                ...

            if parallelism > 1 and len(pending) > 1:
                ...
            else:
                ...

            _drain_pending_diarization(pipeline, failures)
        finally:
            if end:
                end()
            ...
```

Extend the end-of-function safety net (already closing `separation_svc`'s pools) with one more service:

```python
    separation_svc = getattr(pipeline, "separation_svc", None)
    for close_name in ("close_window_pool", "close_prefetch_pool", "close_async_pools"):
        close = getattr(separation_svc, close_name, None)
        if callable(close):
            close()

    diarization_svc = getattr(pipeline, "diarization_svc", None)
    close_postprocess = getattr(diarization_svc, "close_postprocess_pool", None)
    if callable(close_postprocess):
        close_postprocess()

    return list(failures.items())
```

Add `import threading` at the top of `utils/batch.py` if not already present (check first -- `test_batch.py` already imports `threading`, but `batch.py` itself currently only imports `os` at module scope; `run_batch_by_stage` does local `import copy` / `import os` inside the function body, so add `import threading` the same way, as a local import inside `run_batch_by_stage`, to match the existing style rather than adding a new top-of-file import).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_batch.py -q`
Expected: PASS, all tests including the 3 new ones and every pre-existing test in this file (in particular `test_a_run_that_stops_after_diarization_still_closes_the_prefetch_pools` and `test_the_safety_net_does_not_choke_on_a_pipeline_with_no_separation_svc`, which must keep passing unchanged).

- [ ] **Step 5: Run the full suite for the overall regression count**

Run: `python -m pytest tests -q`
Expected: same 100 failed / (879 + number of new tests) passed as the current baseline, i.e. only new tests added, zero new failures among pre-existing tests.

- [ ] **Step 6: Commit**

```bash
git add utils/batch.py tests/test_batch.py
git commit -m "fix(batch): drain diarization postprocess futures before the next stage

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Push

- [ ] **Step 1:** `git push origin solid-architecture`
- [ ] **Step 2:** Verify: `git fetch origin solid-architecture && git rev-parse solid-architecture origin/solid-architecture` -- both SHAs must match.
