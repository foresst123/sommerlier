# ASR Cross-File Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop ASR models from idling at a per-file barrier. Each ASR model consumes work from a queue that spans several files; when a model runs out of work its VRAM is released and the slowest remaining model is helped (a bigger batch on the same GPU, or a fresh replica on the freed GPU).

**Architecture:** `ASRService.process()` keeps its data preparation and ROVER voting but hands the three model calls to a shared `AsrScheduler`. The scheduler owns one *lane* per model (Whisper, PhoWhisper, Qwen3), each with a FIFO job queue that is filled by every file in flight. Worker threads pull batches from the lanes. The stage loop runs `files_in_flight` files concurrently through the existing `parallel_stage_view` mechanism, so while one file waits for its slowest model, the others keep the lanes fed. When a lane is drained and no more files will arrive, the scheduler frees that model's VRAM and rebalances.

**Tech Stack:** Python threads + `threading.Condition`, existing `WorkerProcessService`, pytest with fake models (no GPU).

**Spec:** Conversation of 2026-09-25. Rules from the user:
1. Qwen3-ASR on GPU 0; PhoWhisper and Whisper share GPU 1, each with batch size 16.
2. Models keep working across files; nothing waits for the other models' current file.
3. When a model finishes: release its VRAM. If the slowest remaining model is on the **same GPU**, raise that model's batch size to 48. If it is on a **different GPU**, start a replica of that slowest model on the freed GPU with batch size 48 from the start.

## Global Constraints

- Default behaviour unchanged: with `performance.stages.asr.cross_file` false the existing per-file thread-pool path in `ASRService.process` runs exactly as before (including the per-file Qwen replica logic).
- Cross-file mode must produce the **same transcripts** as the legacy path for the same inputs (parity test).
- A failure inside one lane batch yields empty results for that batch (never an exception into `process`), matching the legacy fallback contract.
- Boost/replica batch size and the shared batch size are configuration (`shared_batch_size` 16, `boost_batch_size` 48), not literals.
- Same interpreter and existing worker classes; no new virtualenv.
- Commits end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `podcast-pipeline/services/asr_scheduler.py` (create) | `LaneWorker`, `Lane`, `FileTicket`, `AsrScheduler`, pure `choose_rebalance` |
| `podcast-pipeline/services/asr_service.py` (modify) | Build lanes from loaded models, adapters per model, `process()` cross-file path, `begin_cross_file_stage` / `settle_file` / `end_cross_file_stage` |
| `podcast-pipeline/utils/performance_config.py` (modify) | New `asr` keys; `resolve_asr_placement` |
| `podcast-pipeline/utils/batch.py` (modify) | `_stage_parallelism` for `asr`; stage hooks |
| `podcast-pipeline/services/pipeline_service.py` (modify) | `parallel_stage_view("asr")` when cross-file; release scheduler at `end_stage_scope` |
| `podcast-pipeline/services/model_loader.py` (modify) | Place PhoWhisper / Whisper by `asr_placement` |
| `podcast-pipeline/main.py` (modify) | Placement, replica factories, pass to `ASRService` |
| `podcast-pipeline/qwen3_worker.py`, `services/qwen3_worker_service.py` (modify) | `--batch-size` override so a replica starts at 48 |
| `podcast-pipeline/config.json` (modify, `a100`) | Placement, batch sizes, `cross_file`, `files_in_flight` |
| `tests/test_asr_scheduler.py`, `tests/test_asr_cross_file.py`, `tests/test_asr_placement.py` (create) | Fakes-only tests |

---

### Task 1: `AsrScheduler` core (queues, batching, tickets, no barrier)

**Files:** Create `services/asr_scheduler.py`; Test `tests/test_asr_scheduler.py`.

**Interfaces:**
- `LaneWorker(name, gpu, run_batch, batch_size, release=None)` where `run_batch(payloads: list, batch_size: int) -> list` (same length) and `release()` frees the model's VRAM.
- `Lane(name, gpu, primary: LaneWorker, empty, replica_factory=None, boostable=True)` with `empty` the result used when a batch fails; `replica_factory(gpu, batch_size) -> LaneWorker`.
- `AsrScheduler(lanes, shared_batch, boost_batch, config=None, logger=None, monitor=None)`; `expect_files(n)`, `add_intake(n=1)`, `submit(file_id, payloads_by_lane) -> FileTicket`, `shutdown()`. `FileTicket.wait(timeout=None) -> {lane_name: [results]}`.
- Tests: results arrive in submission order per lane; a fast lane finishes file 2 before a slow lane finishes file 1 (no barrier); a batch never exceeds the worker's `batch_size`; `run_batch` raising gives `empty` results and still completes the ticket; `shutdown()` joins threads.

### Task 2: Rebalancing (release, boost, replica)

**Files:** Modify `services/asr_scheduler.py`; Test `tests/test_asr_scheduler.py`.

**Interfaces:**
- `choose_rebalance(finished, remaining, now, cfg) -> ("none",) | ("boost", lane) | ("replica", lane, gpu)` pure and unit-tested. Slowest = largest `remaining_jobs / rate`; a lane with no rate yet ranks by remaining jobs.
- Same GPU as the finished lane and slowest is `boostable` → boost. Different GPU and slowest has a `replica_factory` and `estimate_replica_gain(...) >= replica_min_gain_seconds` → replica on the freed GPU. Otherwise none.
- A lane finishes when intake is closed (`settled >= expected`), its queue is empty and nothing is in flight. Its `release()` runs in every case; rebalancing never runs while intake is open.
- Tests: same-GPU boost sets every worker of the slowest lane to `boost_batch`; cross-GPU replica calls `replica_factory(freed_gpu, boost_batch)` and the replica's results are merged in order; small backlog skips the replica but still releases; intake not closed means no release or rebalance; replicas are released at `shutdown`.

### Task 3: Placement and configuration

**Files:** Modify `utils/performance_config.py`; Test `tests/test_asr_placement.py`.

**Interfaces:**
- New `stages.asr` keys: `cross_file` (bool, False), `files_in_flight` (int, 3, 1..8), `shared_batch_size` (int, 16), `boost_batch_size` (int, 48), `qwen3_gpu` (str, "gpu_2"), `whisper_gpu` (str, "gpu_2"), `phowhisper_gpu` (str, "gpu_1") — the defaults reproduce today's layout.
- `resolve_asr_placement(asr_cfg, gpu_1, gpu_2) -> {"qwen3": int, "whisper": int, "phowhisper": int}`; a value other than `gpu_1`/`gpu_2` raises `ValueError`.
- Tests: defaults equal the legacy layout; the new a100 layout resolves; invalid value raises; resolved dict contains the new keys.

### Task 4: `ASRService` cross-file path and stage hooks

**Files:** Modify `services/asr_service.py`, `utils/batch.py`, `services/pipeline_service.py`; Test `tests/test_asr_cross_file.py`.

**Interfaces:**
- `ASRService(..., replica_factories=None, asr_workers=None)`; `cross_file_enabled` property; `begin_cross_file_stage(total_files)`, `settle_file()` (counts a file that never submitted, thread-local flag), `end_cross_file_stage()`.
- `_stage_parallelism(args, "asr")` returns `files_in_flight` when `cross_file`; the stage loop calls `begin_cross_file_stage(len(pending))` before and `settle_file()` in `run_one`'s `finally`; `end_stage_scope` for `asr` calls `end_cross_file_stage()`.
- `parallel_stage_view("asr")` returns a per-file view only when cross-file is enabled (the existing test that expects `self` for `asr` keeps passing).
- Tests: **parity** — fake Whisper/PhoWhisper/Qwen3 produce identical transcripts through the legacy path and the cross-file path; a file that never submits (checkpointed) still lets intake close; VRAM release runs only when the user did not ask for `--keep_models`.

### Task 5: Wire placement, replicas and the a100 profile

**Files:** Modify `main.py`, `services/model_loader.py`, `qwen3_worker.py`, `services/qwen3_worker_service.py`, `config.json`.

- Qwen3 worker on `placement["qwen3"]`, Whisper worker on `placement["whisper"]`, PhoWhisper / CT2 Whisper devices from `asr_placement`.
- Replica factories: Qwen3 (new `Qwen3WorkerService` on the freed GPU with `--batch-size 48`), Whisper vLLM (new `WhisperVLLMWorkerService`), PhoWhisper (new `PhoWhisperASR` on the freed GPU).
- a100: `qwen3_gpu: gpu_1`, `whisper_gpu: gpu_2`, `phowhisper_gpu: gpu_2`, `cross_file: true`, `files_in_flight: 3`, `models.whisper.batch_size` and `models.phowhisper.batch_size` 16.
- Run the related existing tests and compare against a clean HEAD worktree.

### Task 6: Verify on the real machine (manual)

Expect in the log: `[ASR scheduler]` progress lines, `lane whisper finished, releasing`, `boost phowhisper batch 16 -> 48` or `starting qwen3 replica on GPU 1 (batch 48)`, transcripts identical in shape to before, ASR stage wall time lower than the previous 225 min. If a replica fails to start, the run must continue on the remaining workers.

---

## Self-Review

- **Spec coverage:** placement (Task 3, 5), batch 16 (Task 3, 5), no per-file barrier (Task 1, 4), release on finish (Task 2), same-GPU boost to 48 (Task 2), cross-GPU replica at 48 (Task 2, 5).
- **Assumptions to confirm on hardware:** two vLLM engines and PhoWhisper fit as sized; a Whisper replica on the freed GPU starts in reasonable time; the CTranslate2 PhoWhisper accepts a runtime batch size of 48 (its `transcribe_batch` takes `batch_size`).
- **Not covered:** vLLM engines cannot grow their KV budget at runtime, so "boost" for a vLLM lane only raises the client request size, not the engine's memory.
