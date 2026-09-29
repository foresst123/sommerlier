# Music stage: dual-GPU BS-RoFormer pool with early checkout release — design

## Problem

The music stage (SSLAM tagging + BS-RoFormer vocal-stripping) is slow. Both
shipped profiles (`kaggle`, `a100`) currently set `music.cross_file_overlap:
true`, `music.max_separator_workers: 1`. Per `resolve_music_devices()`
(`utils/performance_config.py:221-253`), `cross_file_overlap` always wins
over `max_separator_workers` when both could apply, so today: SSLAM (one
instance) stays on `device_1`; BS-RoFormer's *sole* instance moves entirely
to `device_2`. Two files' music-stage `run()` calls are allowed to overlap
(`utils/batch.py`'s `_stage_parallelism` returns `2` for stage `"music"`
when `cross_file_overlap` is on), so file N+1's SSLAM tagging can run while
file N is still stripping — but there is still only **one** BS-RoFormer
instance in existence, so only one file's vocal-stripping (the expensive
GPU step, the one both profiles' `music_scope: "full"` runs over the whole
recording) can happen at a time. `device_1` sits mostly idle during that
step.

## Goal

Use both GPUs for the actual vocal-stripping work, and stop a GPU sitting
idle while a finished file's CPU-only post-processing (resample, mono-fold,
length-match) is still running.

## Decisions made during brainstorming (see conversation for the reasoning)

- **SSLAM stays a single, shared instance** (unchanged from today). It is
  the fast, small model; duplicating it was considered and rejected as
  unnecessary VRAM/complexity for no real throughput gain — it is not the
  bottleneck.
- **Two BS-RoFormer instances, one per GPU**, via the existing
  `max_separator_workers` pool machinery already built into
  `resolve_music_devices()`/`ModelLoader.load_music_models()`
  (`services/model_loader.py:154-204`) — no new pool-construction code
  needed, just a config value change and removing `cross_file_overlap`'s
  precedence over it.
- **Concurrency stays capped at 2 files at once** in the music stage-major
  pass (matching today's `_stage_parallelism` cap). A deeper, queue-driven
  pipeline allowing more files in flight was considered (more theoretical
  GPU utilization) and explicitly rejected: it requires restructuring how
  `run_batch_by_stage` drives the music stage from "N concurrent whole-file
  `run()` calls" to a true multi-stage queue, which is a much larger,
  riskier change for benefit that was not judged worth it (YAGNI).
- **No separate post-processing thread pool.** Considered (a literal "third
  pool" mirroring the user's original phrasing) and rejected once traced
  through: with the 2-file cap already chosen, post-processing for file A
  and file B already runs concurrently on their own two worker threads (one
  each, from `_stage_parallelism`'s `ThreadPoolExecutor(max_workers=2)`).
  Adding a dedicated executor would only additionally help if more than 2
  files' post-processing needed to run at once — which reopens the "deeper
  pipeline" question just rejected. The one thing worth doing is releasing
  a BS-RoFormer instance back to its checkout queue as soon as its GPU work
  is done, instead of holding it through post-processing — that alone lets
  a third (queued) file grab a now-free GPU instance sooner, without any
  new pool.
- **Correction (post-approval):** both profiles' `music_scope` is actually
  `"spans"` today, not `"full"` — a known, separate, not-yet-fixed regression
  (commit `0148e4e`). The user chose to keep `spans` as-is for this branch
  ("chỉ dùng span và mô hình hiện tại") rather than fix `music_scope`
  alongside this feature. **Scope is therefore `MusicService.strip_music_spans`
  → its `separate_job` closure, not `strip_full_recording`**, which is left
  completely untouched (still dead code under the current, unfixed
  `music_scope`). `strip_music_spans` already parallelizes multiple spans
  within one file across the pool via its own `ThreadPoolExecutor`; the same
  early-checkout-release idea applies inside `separate_job`, and the real
  benefit shows up across files (two files' own `strip_music_spans()` calls
  sharing the same checkout queue — see `_checkout_queue`'s own docstring),
  not within one file's job executor (which already sizes itself to
  `min(pool_size, job_count)` and so never contends with itself).

## Architecture

`BSRoformerRemover._run()` (`models/bs_roformer.py`) currently does, in one
call: write the input to a temp wav, run the GPU separator, read the output
stem back from disk, mono-fold/resample/length-match it against the input,
and clean up temp files. It splits cleanly into a GPU-only half and a
CPU-only half, because none of the mono-fold/resample/length-match work
(`_to_mono`, `_resample`, `_match_length` — plain numpy functions, verified
by reading them) touches the model or the device at all.

- `separate_raw(audio_array, sample_rate) -> (out, out_sr, stereo_in) | None`:
  everything through reading the raw output stem back from disk, and all
  temp-file cleanup (self-contained — no state survives this call).
- `postprocess_separated(raw, audio_array, sample_rate) -> np.ndarray | None`:
  the mono-fold/resample/length-match math on the tuple `separate_raw`
  returned. Takes no model/device access, so it is safe to run on any
  thread, including one that no longer holds the checkout for the instance
  that produced `raw`.
- `_run()` becomes a thin wrapper: `postprocess_separated(separate_raw(...), ...)`.
  Its existing callers (`separate_full`, `separate_segment`, and
  `separate_span`'s own post-processing) are unchanged.

`MusicService.strip_full_recording()` (`services/music_service.py`) checks
out a BS-RoFormer instance from the existing `_checkout_queue` exactly as it
does today. With the split available, when the pool has more than one
instance, it calls `separate_raw()`, immediately returns the instance to the
queue (`available.put(model)`), and only then calls `postprocess_separated()`
on that same file's own worker thread. A single-instance pool keeps calling
`_run()` exactly as before — no behavior change when there is nothing to
pipeline against.

`utils/batch.py`'s `_stage_parallelism()` gates 2-file concurrency for stage
`"music"` on `cross_file_overlap` alone today; it gains an `or` on
`max_separator_workers >= 2`, since a 2-instance pool is exactly the
condition under which running 2 files concurrently is worth it.

`config.json`: both `kaggle` and `a100` change
`music.max_separator_workers: 1 -> 2`, `music.cross_file_overlap: true ->
false` (the two settings conflict; `cross_file_overlap` no longer describes
the wanted behavior — a single instance pipelined across two files' stages
— once a genuine 2-instance pool is in play).

No new configuration keys, no new thread pools, no changes to
`services/model_loader.py` (its `max_separator_workers > 1` branch already
builds the 2-device pool correctly) or to `separate_span`/`strip_music_spans`.

## Data flow

1. The `"music"` stage-major pass runs 2 files concurrently (per the
   `_stage_parallelism` change).
2. Each file's worker thread: SSLAM tags it (one shared instance, serialized
   by the existing `_tagger_lock`, released as soon as tagging finishes —
   unchanged) → `strip_full_recording`.
3. `strip_full_recording` checks out one of the two BS-RoFormer instances →
   `separate_raw()` runs the GPU inference → the instance is returned to the
   queue immediately → `postprocess_separated()` runs the CPU-only
   finishing work on that same thread → the result patches the waveform,
   exactly as `strip_full_recording` does today.
4. A third file, once its own SSLAM tagging is done and it reaches
   `strip_full_recording`, only waits for whichever GPU instance finishes
   its `separate_raw()` call first — it does not wait for that file's
   post-processing to finish too. This is the actual time saved.

## Error handling

`separate_raw()` returns `None` on any failure (bad output, exception during
inference) — the same conditions `_run()` already treats as failure today,
just detected one call earlier. `postprocess_separated(None, ...)` returns
`None` immediately. `strip_full_recording`'s existing failure path (log via
`_unusable_vocals`, keep the recording unmodified, return `[]`) is
unchanged; no new failure mode is introduced, and the instance is still
returned to the queue via the same `finally`-guarded checkout regardless of
whether `separate_raw` succeeded or failed.

## Testing

- Equivalence: calling `separate_raw()` then `postprocess_separated()` on a
  test fixture produces the identical array to calling `_run()` directly,
  reusing the existing fake/test doubles in `tests/test_music_separator.py`
  / `tests/test_music_full_strip.py`.
- Early-release: a test with two fake BS-RoFormer instances and an
  artificially slow `postprocess_separated()` (e.g. a threading `Event`)
  confirms a second caller can check out an instance while the first
  caller's post-processing is still in flight — i.e., the checkout queue
  actually gets the instance back before post-processing completes, not
  after.
- `_stage_parallelism`: a new case alongside the existing
  `cross_file_overlap` one — `max_separator_workers=2, cross_file_overlap=False`
  → expect `2`.
- Full regression run of `tests/test_music_separator.py`,
  `tests/test_music_full_strip.py`, `tests/test_batch.py` after the change,
  compared against this branch's own pre-change baseline (not the
  cross-branch 100-failure baseline tracked on `solid-architecture`, since
  this branch forked from it before any of those were fixed here).
