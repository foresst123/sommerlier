# Separation window prefetch — spec

## Problem

`podcast-pipeline` runs stage-major (`by_stage: true` in both `kaggle` and `a100`
profiles): `utils/batch.py:run_batch_by_stage` finishes the `"diarization"`
pass for every file in the batch before starting the `"separation"` pass for
any file. Building the overlap-separation windows (`WindowPlanner`, driven
through `utils/window_pool.py`'s process pool) is pure CPU/numpy work — it
needs only `segments` (diarization's own output), the processed waveform,
`music_map` and the cut timeline, none of which touch a GPU. But today that
work only starts inside `SeparationService.process_overlaps`, which is only
called from the `"separation"` stage-major pass. So the CPU cores that would
build windows sit idle for the whole `"diarization"` pass (GPU-bound, via the
DiariZen worker), and only start once every file in the batch has finished
diarizing — even though, for any one file, everything window-building needs
is ready the moment that file's own diarization result exists.

## Goal

As soon as a file's diarization result is available (i.e., right where
`PipelineService.run()` returns for `--stop_after diarization`), start
building that file's overlap-separation windows in the background — using
only CPU, via the existing `utils/window_pool.py` pool — so that work runs
concurrently with DiariZen working through the rest of the batch and with
Sidon's own worker still starting up. When the `"separation"` stage-major
pass later reaches this file, `process_overlaps` finds the windows already
built and skips straight to calling the separator.

## Decisions

- **No checkpoint.** A built window plan is cheap and deterministic to
  rebuild (from data that is *already* checkpointed: `diarization`,
  `music_patches`/`timeline`). Losing it to a rare process crash costs a few
  seconds of CPU, not lost GPU work or lost correctness — the same reasoning
  the codebase already applies to `conversation_exports` judging (not
  checkpointed) and to `SeparationService._window_pool_state` (in-process
  only). A `WindowPlan.window.audio` array is real decoded audio (~1–1.5MB
  per window); pickling that to disk per file would be a real, avoidable
  cost with no correctness upside.
- **No RAM cap.** However many files finish diarization before Sidon starts
  consuming them, all of their prefetched plans (including their window
  audio) stay resident until consumed. No backpressure/limit is added in
  this pass.
- **Isolation via the existing `fork_for_file()`.** The codebase already has
  a mechanism for giving one file its own isolated `SeparationService`
  state while sharing the process pool: `parallel_stage_view()` +
  `fork_for_file()`, used today when the `"separation"` stage itself
  processes more than one file concurrently. Prefetching reuses it exactly:
  the background build runs on a fresh `fork_for_file()` clone, never on the
  shared instance, so it cannot race with `self.music_map` /
  `self.timeline` being overwritten for the next file, or with
  `self._same_speaker_pairs`-style shared mutable state (which this plan
  also removes, since it was already unsafe under `fork_for_file()`-style
  concurrency in spirit, only accidentally safe today because grouping and
  consumption happen back-to-back on the same instance).
- **Stats stay single-owner.** Whatever runs inside plan-building
  (`self.stats["jobs"]`, `self.stats["pairs"]`, `self.overlap_durations`)
  must land on the *real* instance handling the file's actual separation,
  not on a throwaway clone. The plan-building step returns these as plain
  data; only `process_overlaps`, running on `self`, applies them.
- **Pool lifetime spans two stage scopes.** A plan is *produced* during the
  `"diarization"` stage-major pass and *consumed* during the `"separation"`
  pass. The background coordinator pool must therefore close at the end of
  the `"separation"` stage's scope (next to `close_window_pool()` /
  `close_async_pools()`), never the `"diarization"` stage's own
  `end_stage_scope()` — closing it there would cancel work still in flight.
- **Trigger fires exactly once per file.** `PipelineService.run()` is
  re-entered once per stage-major pass for the same file, and the
  diarization section of `run()` executes on *every* one of those passes
  (loading the checkpoint on later passes). The prefetch call must be gated
  on `getattr(args, "stop_after", None) == "diarization"`, which is true for
  exactly one pass.
- **Stale prefetch on retry.** If a file fails after its diarization
  checkpoint exists (e.g., in `"separation"` or later), `main.py`'s
  `_discard_partial` wipes its checkpoint directory, including
  `diarization`. Its next attempt recomputes diarization from scratch; any
  plan already prefetched for the failed attempt must be dropped so a retry
  cannot be handed a plan for an attempt that no longer exists. (Given
  fixed args/config within one `main.py` run, a stale plan would in practice
  be numerically identical to a fresh one — this is cheap insurance, not a
  known bug.)
- **File-major mode is unaffected.** `--stop_after` is never set outside
  `run_batch_by_stage`, so the prefetch trigger simply never fires there;
  `process_overlaps` behaves exactly as it does today (no `audio_path`
  passed in that mode, or a miss if it is).

## Non-goals

- No change to what gets separated, how windows are laid out, or the
  overlap policy. This is a scheduling change only — behavior-preserving
  when there is no prefetch hit; behavior-preserving in speech/QC/stats
  terms when there is one.
- No new checkpoint format.
- No concurrency cap / RAM budget (explicitly deferred; see Decisions).
