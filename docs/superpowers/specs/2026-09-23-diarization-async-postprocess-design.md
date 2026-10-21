# Diarization Async Post-Processing — Design Spec

**Date:** 2026-09-23
**Status:** Draft, pending user review

## Problem

During the diarization stage-major pass (`--by_stage`, `stop_after == "diarization"`),
`config.json`'s `performance.stages.diarization.workers` is `2` for both `kaggle`
and `a100`: two files are meant to be in flight at once, each leased its own
DiariZen worker process out of `WorkerPoolService`.

`DiarizationService.run_diarization()` is one synchronous call per file that does
both halves of the work on the thread `run_batch_by_stage`'s `ThreadPoolExecutor`
handed it:

1. The GPU call itself (`self.diarizer.diarize(audio_input)`, routed through
   `WorkerPoolService.request()`).
2. Everything after: `filter_diarizer_noise`, `split_at_seams`, VAD,
   `cut_by_speaker_label`, `bridge_interrupted_speaker_turns`,
   `split_long_segments`, building `Segment`/`DiarizationResult`, then (in
   `PipelineService.run()`) `checkpoint.save()` and `stage_out.write_diarization()`
   (which measurably costs several seconds writing per-segment clips to disk).

Step 2 is pure CPU/disk work. While a thread is inside it, the DiariZen worker
process that thread just finished with sits completely idle — `run_batch_by_stage`
does not hand that thread (and therefore that worker) a new file until the
**entire** `run()` call for the current file returns, which is only after step 2
finishes. With two threads and two worker processes, this halves realised GPU
throughput on the diarization pass any time step 2 is non-trivial.

This was found by systematic-debugging investigation (see prior session): the
originally-suspected "logic bug forcing DiariZen to wait" does not exist as a
lock or blocking call anywhere in `worker_pool_service.py`,
`base_worker_service.py`, or `diarizen_worker_service.py` — the idle GPU time is
a real architectural consequence of doing step 1 and step 2 on the same thread
before the batch loop will dispatch the next file.

## Goal

Optimise GPU throughput on the diarization stage-major pass: keep a DiariZen
worker process available for the next file's `diarize()` call as soon as this
file's own GPU call returns, instead of waiting for this file's CPU
post-processing tail.

## Non-goals

- **Separation is explicitly out of scope — it already has this.**
  `SeparationService._async_runtime()` / `gpu_executor` / `post_executor`
  (`services/separation_service.py:280-318`) already split Sidon's raw GPU call
  from its CPU post-processing across two persistent, batch-lifetime
  `ThreadPoolExecutor`s, gated by `performance.enabled` and
  `stages.separation.ordered_postprocess`. Both `kaggle` and `a100` already have
  `ordered_postprocess: true` and `postprocess_workers: 2` in `config.json`.
  `fork_for_file()` does not reset `_async_state`, so these executors are
  already shared, and already overlap, across concurrent files. There is
  nothing to design or build here.
- **The other five stages (music, asr, captioning, refinement, speaker_relabel,
  word_alignment, conversation_exports) are explicitly out of scope**, because
  they do not share diarization/separation's shape (one file per thread via
  `run_batch_by_stage`'s per-stage `ThreadPoolExecutor`, one GPU call followed
  by a real CPU tail, before the *next file* can be dispatched):
  - `music` (BS-RoFormer) already has its own raw/post split, on the
    unmerged `feat/music-dual-gpu-pipeline` branch — not part of this spec.
  - `captioning` (`CaptionService.add_captions`) is a plain per-segment loop
    calling the model; there is no separate CPU post-processing phase to pull
    out.
  - `refinement` already has its own, more specialised concurrency:
    `RefinementPipelinePool` pipelines two GPU layer-partitions internally via
    its own two-stage queue (`stage_zero_queue`/`stage_one_queue`) inside a
    single `generate()` call, and `diarization_refinement_service.py`'s
    `refine()` already parallelises with its own `ThreadPoolExecutor`.
  - `asr` already uses its own internal `ThreadPoolExecutor`s
    (`max_workers=2` and `max_workers=3` at two call sites).
  - `speaker_relabel`, `word_alignment`, `conversation_exports` all run as one
    corpus-wide call over every file's transcripts at once
    (`relabel(segments)`, `align(transcripts, ...)`, `run(transcripts, ...)`),
    not as one call per file dispatched through `run_batch_by_stage`'s
    per-stage thread pool — "the next file is waiting on this thread" does not
    apply to them the way it does to diarization/separation.

  If any of these turn out to want similar treatment later, it needs its own,
  separate investigation and spec — the shapes above are different enough that
  reusing this design blind would be fabricated, not engineered.

## Architecture

Split `DiarizationService.run_diarization()` into two methods, mirroring the
already-proven Sidon pattern:

- **`diarize_raw(chunks, audio, args) -> _RawDiarization`** — GPU-only. Builds
  `audio_input`, calls `self.diarizer.diarize(...)` (DiariZen or pyannote
  branch, unchanged), extracts `annotation`, builds `combined_df`, logs the
  `"raw"` segment-stats line. Returns a small dataclass:
  ```python
  @dataclass
  class _RawDiarization:
      combined_df: "pd.DataFrame"
      method: str  # "diarizen" or "pyannote"
  ```
- **`diarize_postprocess(raw: _RawDiarization, audio, args) -> DiarizationResult`**
  — CPU-only. Everything `run_diarization` currently does starting from
  `raw_for_filter = df_to_list(combined_df)`: noise filter, seam split, VAD,
  merge, bridge, length split, building `Segment`s, and the final
  `DiarizationResult`. No GPU call. Behaviourally identical output to today's
  `run_diarization()` for the same raw input — this is a pure extraction, not
  a behaviour change.
- **`run_diarization()` becomes a thin wrapper**: `diarize_postprocess(diarize_raw(chunks, audio, args), audio, args)`,
  kept for any caller that still wants the whole thing synchronously (tests,
  file-major runs — see below).

`DiarizationService` gains a persistent post-processing pool, named and shaped
exactly like `SeparationService`'s `_async_state`/`_async_runtime()`/
`close_async_pools()`:

```python
# __init__
self._postprocess_state = {"executor": None, "lock": threading.Lock()}

def _postprocess_pool(self):
    state = self._postprocess_state
    with state["lock"]:
        if state["executor"] is None:
            workers = int((self.diarizer_config or {}).get("postprocess_workers", 2))
            state["executor"] = ThreadPoolExecutor(
                max_workers=max(1, workers), thread_name_prefix="diar-post")
        return state["executor"]

def submit_postprocess(self, raw, audio, args):
    return self._postprocess_pool().submit(self.diarize_postprocess, raw, audio, args)

def close_postprocess_pool(self):
    state = getattr(self, "_postprocess_state", None)
    if state is None:
        return
    with state["lock"]:
        executor, state["executor"] = state["executor"], None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=False)
```

`postprocess_workers` is read from `diarizer_config` (the same config block
`DiarizationService` already receives — `models.diarizen`), defaulting to `2`.
Both `kaggle` and `a100` profiles in `config.json` get `"postprocess_workers": 2`
added to `models.diarizen` explicitly rather than relying on the default, so
the value is visible and tunable next to `batch_size`/`clustering_method`.

Since `DiarizationService` instances are shared (shallow `copy.copy()`, not
`fork_for_file()`-style deep isolation) across `parallel_stage_view("diarization")`
copies — the same pattern `_vad_lock` already relies on — `_postprocess_state`'s
executor and lock are shared across every concurrent file's view exactly like
`SeparationService._async_state` is, so the pool genuinely overlaps files
rather than existing once per file.

## Control flow change in `PipelineService.run()`

Only the **fresh-compute branch** (not loaded from checkpoint, not skipped)
changes, and only for the exact case that already gates `prefetch_overlap_plan`
today: `getattr(args, "stop_after", None) == "diarization"`. This is precise
because `run_batch_by_stage` sets `stage_args.stop_after = stage` for every
stage-major pass, so this condition is true if and only if this call is part
of the diarization stage-major sweep — the one case where "the next thing
that happens" is a *different file's* `diarize_raw()`, not this file's own
separation step within the same call. A file-major run or the final
checkpoint-restoring pass never sets it, so they are unaffected and keep
today's fully-synchronous behaviour.

```python
else:
    self._load("base")
    self._load("diarization")
    chunks, _ = self.diarization_svc.prepare_chunks(audio_data)
    raw = self.diarization_svc.diarize_raw(chunks, audio_data, args)
    self._free(args, "diarizer", "vad")
    self._release_worker(args, "diarizen")

    if getattr(args, "stop_after", None) == "diarization":
        future = self.diarization_svc.submit_postprocess(raw, audio_data, args)
        with self._pending_diar_lock:
            self._pending_diar_jobs[audio_path] = future

        def _finish(fut, audio_path=audio_path, checkpoint=checkpoint,
                    stage_out=stage_out, audio_data=audio_data, args=args):
            try:
                result = fut.result()
            except Exception:
                return  # surfaced to `failures` by the drain step below, once
            checkpoint.save("diarization", result)
            stage_out.write_diarization(
                result.segments, total_dur=audio_data.duration,
                raw_segments=getattr(result, "raw_segments", None),
                audio=audio_data.waveform, sample_rate=audio_data.sample_rate)
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
    if self.logger:
        self.logger.info(f"[DEBUG] Diarization returned {len(diarization_result.segments)} "
                         f"segments via {diarization_result.method}")
    if not diarization_result.segments:
        raise RuntimeError(...)  # unchanged
    checkpoint.save("diarization", diarization_result)
    computed.add("diarization")
```

The old unconditional `self._free(...)`/`self._release_worker(...)` calls that
sat *after* the whole if/elif/else block are removed — they are now inside the
fresh-compute branch, right after `diarize_raw()`, which is the only branch
that ever loaded anything to free. Both are no-ops when called on a
branch that never loaded a worker, so this is a pure relocation, not a
behaviour change for the checkpoint-loaded or diarization-disabled branches.

`PipelineService.__init__` gains, alongside `_model_load_lock`:
```python
self._pending_diar_jobs = {}
self._pending_diar_lock = threading.Lock()
```
Shared across `parallel_stage_view()` copies the same way `_model_load_lock`
already is (shallow `copy.copy()` does not rebind them).

## The stage-boundary drain (correctness barrier)

`run_batch_by_stage` (`utils/batch.py`) must not let the "separation" pass
start reading `checkpoint.exists("diarization")` for a file whose
post-processing future is still running. Immediately after the per-file loop
of *every* stage pass (cheap no-op for any stage but diarization), before that
stage's `finally` block:

```python
def _drain_pending_diarization(pipeline, failures):
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

This blocks until every outstanding file's post-processing has actually
finished (success or failure) before the loop is allowed to move to the next
stage — exactly the same guarantee the current fully-synchronous code gets for
free, restored explicitly here. A failure is recorded into the same
`failures` dict, with the same `"<label>: <ExceptionType>: <message>"` format
already used for every other stage failure, so it is excluded from every
later stage's `pending` list exactly like a synchronous failure is today
(`test_a_file_that_fails_is_dropped_from_later_stages`already covers this
contract; the new test only adds an async source for it).

`_finish()` (in `pipeline_service.py`) deliberately does **not** pop from
`_pending_diar_jobs` itself and does not re-raise — only the drain step reads
and removes entries, so a fast-resolving future can never disappear from the
dict before the drain gets a chance to observe (and report) its outcome.

The existing end-of-run safety net in `run_batch_by_stage` (the one that
already closes `separation_svc`'s `close_window_pool`/`close_prefetch_pool`/
`close_async_pools` for a run that stops before "separation") is extended with
one more line, closing `diarization_svc.close_postprocess_pool()` the same
defensive way (`getattr(..., None)`, callable check, no-op if absent).

## Failure handling / edge cases

- **Postprocess raises** (e.g. "produced no segments"): captured by the drain
  into `failures[path]`, excluded from later stages — matches today's
  synchronous `raise RuntimeError(...)` path exactly in observable outcome,
  just surfaced one step later (at the drain, not inline in `run_one()`).
- **`--stop_after diarization` with separation disabled**
  (`step_enabled(args, "separation")` false): `_finish()` still checkpoints
  and writes the diarization output, just skips `prefetch_overlap_plan()` —
  unchanged from today's guard.
- **File-major / full runs** (`stop_after` not `"diarization"`): fully
  synchronous, byte-for-byte the same code path as today, just calling
  `diarize_postprocess(diarize_raw(...))` instead of the old monolithic
  `run_diarization()`. No new concurrency, no new failure surface.
- **`keep_models=True`**: `_free`/`_release_worker` are no-ops as they are
  today; unaffected by the relocation.

## Testing plan

- `tests/test_diarization_service.py` (or wherever `DiarizationService` is
  tested today):
  - `diarize_raw()` does not call `filter_diarizer_noise`/VAD/merge/split
    (monkeypatch and assert not called).
  - `diarize_postprocess(diarize_raw(...))` produces the same segments as
    today's `run_diarization()` for one fixed fake-diarizer output —
    behavioural-equivalence regression test.
  - `submit_postprocess()` returns a `Future` resolving to a
    `DiarizationResult`; `close_postprocess_pool()` shuts down cleanly and is
    a safe no-op when called twice or with nothing ever submitted.
- `tests/test_pipeline_service.py`:
  - With `stop_after="diarization"`, `run()` returns before a
    `submit_postprocess` future (blocked on a `threading.Event`, not
    `sleep()`, per `condition-based-waiting`) resolves.
  - Once the future resolves, `checkpoint.save`, `stage_out.write_diarization`,
    and (when separation is enabled) `prefetch_overlap_plan` are each called
    exactly once, with the expected arguments.
  - Without `stop_after="diarization"`, behaviour is unchanged: separation's
    section of the same `run()` call still receives a populated
    `diarization_result`.
- `tests/test_batch.py`:
  - Two-file test proving overlap: file 2's `diarize_raw` (fake) call starts
    before file 1's deferred postprocess (fake, gated on an `Event`)
    completes — mirrors `test_a_span_releases_its_checkout_before_postprocessing_finishes`.
  - A failing postprocess future ends up in `run_batch_by_stage`'s returned
    failures list, and the file is absent from the next stage's file list —
    mirrors `test_a_file_that_fails_is_dropped_from_later_stages`.
  - The drain blocks stage transition: the "separation" pass's first call
    does not happen until a slow (`Event`-gated) postprocess future resolves.
  - The final safety net also closes `diarization_svc.close_postprocess_pool()`
    for a run that stops before it would otherwise be closed, and remains a
    no-op when `diarization_svc` is absent.

## Summary of touched files

- `services/diarization_service.py` — split `run_diarization` into
  `diarize_raw`/`diarize_postprocess`, add `_postprocess_state`/
  `_postprocess_pool`/`submit_postprocess`/`close_postprocess_pool`, add the
  `_RawDiarization` dataclass.
- `services/pipeline_service.py` — restructure the diarization section of
  `run()` per above; add `_pending_diar_jobs`/`_pending_diar_lock` to
  `__init__`.
- `utils/batch.py` — add `_drain_pending_diarization()`, call it after every
  stage pass's file loop; extend the existing end-of-run safety net to also
  close `diarization_svc.close_postprocess_pool()`.
- `config.json` — add `"postprocess_workers": 2` to `models.diarizen` for both
  `kaggle` and `a100`.
- Tests as listed above.
