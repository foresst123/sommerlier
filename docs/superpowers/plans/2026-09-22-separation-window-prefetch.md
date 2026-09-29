# Separation Window Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a file's overlap-separation windows on a background thread the moment its diarization result exists (during the `"diarization"` stage-major pass), so that CPU-only work overlaps with DiariZen still working through the rest of the batch, instead of waiting for the whole batch to finish diarizing before the `"separation"` pass starts building any windows.

**Architecture:** Extract the CPU-only portion of `SeparationService.process_overlaps` (pairs → enrollments → job grouping → materialized window plans, no GPU call) into `_build_overlap_plan`, returning a plain-data `_OverlapPlan`. Add a small in-process cache keyed by audio path: `prefetch_overlap_plan` builds a plan on a `fork_for_file()` clone via a background `ThreadPoolExecutor`; `process_overlaps` consults the cache first and falls back to building inline on a miss or failure. `PipelineService.run()` triggers the prefetch exactly once per file, right where it returns for `--stop_after diarization`, and closes the prefetch pool at the same stage boundary as the existing window-build pool.

**Tech Stack:** Python 3, `concurrent.futures.ThreadPoolExecutor` (new coordinator pool) + the existing `utils/window_pool.py` `ProcessPoolExecutor`-backed `WindowBuildPool`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-22-separation-window-prefetch.md` — read it first; it has the reasoning behind every constraint below (no checkpoint, no RAM cap, why `fork_for_file()` is reused instead of new isolation, why stats must be applied by `process_overlaps` and not by plan-building itself, why the pool closes with the separation stage and not the diarization stage).

## Global Constraints

- No RAM budget or cap on how many files' plans may be prefetched and resident at once (spec: "No RAM cap").
- No new checkpoint format; prefetched plans live only in process memory for the life of one `main.py` run (spec: "No checkpoint").
- Every existing test in `tests/test_separation_logic.py`, `tests/test_separation_recovery.py`, `tests/test_ghost_speakers.py`, `tests/test_separation_window.py`, `tests/test_separation_backends.py`, and `tests/test_stage_scheduling.py` must still pass, unmodified in intent (some call sites need mechanical signature updates — see Task 1).
- `main.py` calls `argparse.parse_args()` at import time and cannot be imported by a test process; verify `main.py` changes by reading its source text (`open(...).read()`), the same style `tests/test_stage_scheduling.py` already uses via its `_source()` helper.
- The prefetch trigger must fire on exactly one stage-major pass per file (`stop_after == "diarization"`), never on a later pass that re-enters the diarization section of `run()` while loading it from checkpoint.

---

### Task 1: `_group_jobs` returns same-speaker pairs instead of mutating `self`

**Files:**
- Modify: `podcast-pipeline/services/separation_service.py:797-818` (`_group_jobs`)
- Modify: `podcast-pipeline/services/separation_service.py` (the one call site inside `process_overlaps`, currently around line 1134-1176)
- Modify: `podcast-pipeline/tests/test_separation_window.py:30-34` (`setup()` helper) and `:246` (a second direct call)
- Modify: `podcast-pipeline/tests/test_ghost_speakers.py:142-143` (a third direct call)
- Test: `podcast-pipeline/tests/test_separation_logic.py`

**Interfaces:**
- Produces: `SeparationService._group_jobs(self, pairs: list) -> tuple[list, list]` — `(jobs, same_speaker_pairs)`, where `jobs` is the existing `[(speaker_a, speaker_b, plist), ...]` shape and `same_speaker_pairs` is the list `_group_jobs` used to stash on `self._same_speaker_pairs`. Task 2's `_build_overlap_plan` calls it this new way.

Today `_group_jobs` writes `self._same_speaker_pairs = [...]` as a side effect and its one caller reads that attribute right after. That is an instance-level mutable field two files sharing one `SeparationService` could race on the moment plan-building can run on a background clone for one file while the real `process_overlaps` runs for another (Task 3). Fixing it first, standalone, keeps every later task free of that hazard.

- [ ] **Step 1: Write the failing test**

Add to `podcast-pipeline/tests/test_separation_logic.py`, near the top after the existing imports (add `from algorithms.diarization.overlap import detect_overlapping_segments` to the import block):

```python
def test_group_jobs_returns_same_speaker_pairs_instead_of_mutating_self():
    """_group_jobs used to stash same-speaker pairs on self._same_speaker_pairs.
    Once plan-building can run on a background clone for one file while the
    real process_overlaps runs for another (see test_separation_prefetch.py),
    a shared instance attribute like that is a race. Returning the pairs
    directly removes the shared mutable state instead of isolating it."""
    segs = [
        Segment(index="00001", start=0.0, end=15.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=14.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00003", start=32.0, end=40.0, speaker="SPEAKER_01"),
        Segment(index="00004", start=39.0, end=48.0, speaker="SPEAKER_02"),
    ]
    seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "index": s.index}
                 for s in segs]
    pairs = detect_overlapping_segments(seg_dicts, overlap_threshold=0.0)

    svc = SeparationService(FakeTSE(), logger=None)
    jobs, same_speaker = svc._group_jobs(pairs)

    assert len(same_speaker) == 1, "the two SPEAKER_00 segments overlap each other"
    assert same_speaker[0]["seg1"]["speaker"] == same_speaker[0]["seg2"]["speaker"] == "SPEAKER_00"
    assert len(jobs) == 1
    a, b, plist = jobs[0]
    assert {a, b} == {"SPEAKER_01", "SPEAKER_02"}
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `podcast-pipeline/`): `python -m pytest tests/test_separation_logic.py::test_group_jobs_returns_same_speaker_pairs_instead_of_mutating_self -v`
Expected: FAIL with `ValueError: not enough values to unpack (expected 2, got 1)` — `_group_jobs` today returns a plain list with one `(speaker_a, speaker_b, plist)` entry, not a 2-tuple.

- [ ] **Step 3: Write the minimal implementation**

In `podcast-pipeline/services/separation_service.py`, replace `_group_jobs` (currently lines 797-818):

```python
    def _group_jobs(self, pairs):
        """Chỉ nhóm overlap giao/chạm nhau của cùng hai người; giữ mọi nguồn.
        Trả (jobs, same_speaker_pairs) thay vì ghi self._same_speaker_pairs --
        việc dựng kế hoạch có thể chạy trên một bản sao nền cho file này trong
        khi self đang xử lý thật cho file khác (xem prefetch_overlap_plan),
        nên không có thuộc tính instance nào dùng chung an toàn ở đây."""
        buckets = {}
        same_speaker_pairs = []
        for p in pairs:
            key = tuple(sorted({p["seg1"]["speaker"], p["seg2"]["speaker"]}))
            if len(key) != 2:
                same_speaker_pairs.append(p)
                continue
            buckets.setdefault(key, []).append(p)
        jobs = []
        for (a, b), plist in sorted(buckets.items()):
            current, end = [], None
            for p in sorted(plist, key=lambda p: (p["overlap_start"], p["overlap_end"])):
                if current and p["overlap_start"] > end:
                    jobs.append((a, b, current))
                    current = []
                current.append(p)
                end = max(end, p["overlap_end"]) if end is not None else p["overlap_end"]
            if current:
                jobs.append((a, b, current))
        return sorted(jobs, key=lambda job: job[2][0]["overlap_start"]), same_speaker_pairs
```

Then update the one call site inside `process_overlaps`. Find:

```python
        self._same_speaker_pairs = []
        # Tách hết, không lọc theo overlap_threshold. WindowPlanner giữ nguyên
        # core dù rất ngắn rồi lấy context có giới hạn và padding sạch để bù.
        queue = list(self._group_jobs(pairs))
```

Replace with:

```python
        # Tách hết, không lọc theo overlap_threshold. WindowPlanner giữ nguyên
        # core dù rất ngắn rồi lấy context có giới hạn và padding sạch để bù.
        queue, same_speaker_pairs = self._group_jobs(pairs)
```

A little further down, find:

```python
        by_index = {e.index: e for e in speech}
        for p in self._same_speaker_pairs:
```

Replace with:

```python
        by_index = {e.index: e for e in speech}
        for p in same_speaker_pairs:
```

Two other test files call `_group_jobs` directly and assume the old plain-list return. Fix both:

In `podcast-pipeline/tests/test_separation_window.py`, the `setup()` helper (lines 30-34):

```python
def setup(rows, **kwargs):
    segs = segments(rows)
    pairs = detect_overlapping_segments([s.__dict__ for s in segs], overlap_threshold=0)
    planner = WindowPlanner(segs, pairs, waveform(), SR, **kwargs)
    svc = SeparationService()
    return planner, svc._group_jobs(pairs)
```

Change the last line to:

```python
    return planner, svc._group_jobs(pairs)[0]
```

(Every one of its 15 callers destructures as `planner, jobs = setup(...)` and only this one line needs to change.)

Further down in the same file, around line 246:

```python
    jobs = SeparationService()._group_jobs(pairs)
```

Change to:

```python
    jobs, _ = SeparationService()._group_jobs(pairs)
```

In `podcast-pipeline/tests/test_ghost_speakers.py`, around line 142-143:

```python
    fused = [job[2][0] for job in sep.SeparationService()._group_jobs(
        [pair(5.0, 5.2), pair(1.0, 1.2)])]
```

Change to:

```python
    fused = [job[2][0] for job in sep.SeparationService()._group_jobs(
        [pair(5.0, 5.2), pair(1.0, 1.2)])[0]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_separation_logic.py tests/test_separation_window.py tests/test_ghost_speakers.py tests/test_separation_recovery.py tests/test_separation_backends.py -q`
Expected: PASS, including the new test and every previously-existing test in all five files.

- [ ] **Step 5: Commit**

```bash
git add services/separation_service.py tests/test_separation_logic.py tests/test_separation_window.py tests/test_ghost_speakers.py
git commit -m "refactor(separation): _group_jobs returns same-speaker pairs instead of mutating self"
```

---

### Task 2: Extract `_build_overlap_plan` — every CPU-only step of `process_overlaps`

**Files:**
- Modify: `podcast-pipeline/services/separation_service.py` (imports; new `_OverlapPlan`; new `_build_overlap_plan`; shrink `process_overlaps`)
- Test: `podcast-pipeline/tests/test_separation_logic.py`

**Interfaces:**
- Consumes: `SeparationService._group_jobs(pairs) -> (jobs, same_speaker_pairs)` (Task 1).
- Produces: `SeparationService._build_overlap_plan(self, segments: List[Segment], audio: AudioData, overlap_threshold: float = 0.1) -> _OverlapPlan`, and the type `_OverlapPlan` with fields `speech: list`, `pairs: list`, `enrollments: dict`, `seg_by_index: dict`, `jobs: list`, `stats_jobs: int`, `stats_pairs: int`, `overlap_durations: list`. Task 3's `_take_prefetched_plan` returns this exact type; Task 3's `prefetch_overlap_plan` calls this exact method.

This is a pure Extract Method: no behavior changes for a normal (non-prefetched) call. `process_overlaps` calls `_build_overlap_plan` for everything before the first GPU call, then continues exactly as before. The one deliberate side effect: `expanded_window_iter()`'s generator is fully drained into a list *inside* `_build_overlap_plan` (`jobs = list(expanded_window_iter())`) instead of being consumed lazily during the GPU loop, and `file_windows.close()` moves to right after that drain instead of the end of `process_overlaps`. Both changes are required so a background prefetch call (Task 3) can produce a plain-data result — and are safe, because by the time the list is fully materialized every pool job for this file is done and its shared memory is no longer needed.

- [ ] **Step 1: Write the failing test**

Add to `podcast-pipeline/tests/test_separation_logic.py`:

```python
def test_build_overlap_plan_returns_one_materialized_job_for_the_backchannel():
    """_build_overlap_plan carries every CPU-only step of process_overlaps
    through the materialized window list, with no GPU call. Extracted so a
    plan can be built ahead of time on a fork_for_file() clone (see
    test_separation_prefetch.py) while the real process_overlaps still
    consumes it exactly the same way it consumes one built inline."""
    svc = SeparationService(FakeTSE(), logger=None)
    plan = svc._build_overlap_plan(_dialogue(), _audio(), overlap_threshold=0.1)
    assert len(plan.pairs) == 1
    assert plan.stats_jobs == 1 and plan.stats_pairs == 1
    assert len(plan.jobs) == 1
    subjob, outcome = plan.jobs[0]
    assert outcome[0] is not None, "the window must already be built, not just planned"
    assert len(plan.speech) == len(_dialogue())
    assert fake_calls_is_empty := (svc.stats["jobs"] == 0), (
        "building a plan must not touch self.stats -- only process_overlaps "
        "may, once it knows the plan is actually being used")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_separation_logic.py::test_build_overlap_plan_returns_one_materialized_job_for_the_backchannel -v`
Expected: FAIL with `AttributeError: 'SeparationService' object has no attribute '_build_overlap_plan'`.

- [ ] **Step 3: Write the minimal implementation**

In `podcast-pipeline/services/separation_service.py`, add `dataclass` to the imports at the top of the file. Find:

```python
import collections
import copy
import json
import os
import threading
import time as _time
import traceback
from concurrent.futures import ThreadPoolExecutor
```

Replace with:

```python
import collections
import copy
import json
import os
import threading
import time as _time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
```

Now find the start of `process_overlaps`:

```python
    def process_overlaps(self, segments: List[Segment], audio: AudioData, overlap_threshold: float = 0.1) -> List[SpeechSegment]:
        if not self.bss_model:
            # passthrough gắn mixture vào từng segment. Trả SpeechSegment rỗng
            # audio ở đây làm checkpoint, bước xuất clip và dual-channel đều
            # nhận về giá trị None mà không có dấu hiệu nào là separation đã
            # không chạy.
            if self.logger:
                self.logger.warning(
                    "[TSE] no separator is loaded; every overlap stays raw mixture")
            return self.passthrough(segments, audio)

        if self.logger:
            self.logger.info("Processing overlaps with blind source separation")

        seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "index": s.index} for s in segments]
        # Phát hiện mọi giao dương; ngưỡng chạy model chỉ áp dụng sau khi nhóm.
        pairs = detect_overlapping_segments(seg_dicts, overlap_threshold=0.0, logger=self.logger)

        # Bỏ qua micro-overlap (< 60ms): thường là khoảng lặng ở ranh giới
        # segment, không đáng tách và dễ gây insufficient_evidence khi retry.
       
        micro = [p for p in pairs if p["overlap_end"] - p["overlap_start"] < MIN_OVERLAP_SECONDS]
        if micro and self.logger:
            self.logger.info(
                f"[TSE] skipping {len(micro)} micro-overlap(s) < {MIN_OVERLAP_SECONDS*1000:.0f}ms")
        pairs = [p for p in pairs if p["overlap_end"] - p["overlap_start"] >= MIN_OVERLAP_SECONDS]

        # Tạo danh sách các SpeechSegment từ danh sách segments
        speech = [SpeechSegment(**s.__dict__) for s in segments]
        sr = audio.sample_rate
        waveform = audio.waveform
        total_dur = len(waveform) / sr
        for e in speech:
            e.audio = waveform[round(e.start * sr):round(e.end * sr)].copy()

        if not pairs:
            if self.logger:
                self.logger.info(
                    f"[TSE] no overlap >= {overlap_threshold}s among {len(segments)} segments"
                )
            return speech

        enrollments = self.mine_enrollments(segments, audio)
        seg_by_index = {s.index: s for s in speech}

        self._same_speaker_pairs = []
        # Tách hết, không lọc theo overlap_threshold. WindowPlanner giữ nguyên
        # core dù rất ngắn rồi lấy context có giới hạn và padding sạch để bù.
        queue, same_speaker_pairs = self._group_jobs(pairs)
        below = []  # giữ để không vỡ bss_spans tracking
        buildable = []
        for spk_a, spk_b, plist in queue:
            self.stats["jobs"] += 1
            self.stats["pairs"] += len(plist)
            for p in plist:
                self.overlap_durations.append(p["overlap_end"] - p["overlap_start"])
            targets = self._splice_pairs(plist)
            buildable.append((spk_a, spk_b, plist, targets))
```

Replace that whole block (from `def process_overlaps` through the `buildable.append` loop above) with:

```python
    def process_overlaps(self, segments: List[Segment], audio: AudioData,
                         overlap_threshold: float = 0.1,
                         audio_path: Optional[str] = None) -> List[SpeechSegment]:
        if not self.bss_model:
            # passthrough gắn mixture vào từng segment. Trả SpeechSegment rỗng
            # audio ở đây làm checkpoint, bước xuất clip và dual-channel đều
            # nhận về giá trị None mà không có dấu hiệu nào là separation đã
            # không chạy.
            if self.logger:
                self.logger.warning(
                    "[TSE] no separator is loaded; every overlap stays raw mixture")
            return self.passthrough(segments, audio)

        if self.logger:
            self.logger.info("Processing overlaps with blind source separation")

        sr = audio.sample_rate
        waveform = audio.waveform

        # A prefetch started while this file's diarization pass returned (see
        # PipelineService.run()) may already have built every window on a
        # background thread. On a miss (no prefetch, file-major mode, or the
        # background build raised) this builds the plan right now, exactly as
        # process_overlaps always has.
        plan = self._take_prefetched_plan(audio_path) if audio_path else None
        if plan is None:
            plan = self._build_overlap_plan(segments, audio, overlap_threshold)

        speech = plan.speech
        if not plan.pairs:
            return speech

        enrollments = plan.enrollments
        seg_by_index = plan.seg_by_index
        # Stats belong to whichever instance is actually handling this file's
        # separation -- never to a fork_for_file() clone a prefetch ran on --
        # so they are applied here, once, regardless of where the plan came from.
        self.stats["jobs"] += plan.stats_jobs
        self.stats["pairs"] += plan.stats_pairs
        self.overlap_durations.extend(plan.overlap_durations)

        pairs = plan.pairs
        buildable = None  # no longer built here; see _build_overlap_plan
```

Now find the pool-building section right after (still inside `process_overlaps` today):

```python
        # same_speaker: hai segment cùng nhãn chồng nhau — không có gì để tách.
```

Everything from that comment through the end of the `expanded_window_iter` function definition (i.e., through this block, currently ending just before `pending_retries = collections.deque()`):

```python
        # same_speaker: hai segment cùng nhãn chồng nhau — không có gì để tách.
        # Mở rộng vùng base ra hai bên (±2s) rồi giữ nguyên mixture.
        # Không gọi model, không cần speaker thứ hai, không thay đổi audio.
        # Chỉ ghi vào bss_spans với sim=-2 (passthrough marker) để downstream
        # biết vùng này đã được xem xét, không phải bỏ sót.
        by_index = {e.index: e for e in speech}
        for p in same_speaker_pairs:
            ... (through the end of expanded_window_iter's definition and
            `window_iter = window_iter()` / `file_windows = self._window_pool.open_file(...)` etc.)
```

... is being **moved**, not deleted. Delete the whole block from `# same_speaker: hai segment cùng nhãn...` through the closing of `def expanded_window_iter(): ... return job, (None, reason, detail or "no_core_targets", actions)` (i.e., everything Task 2 removes ends right before the line `pending_retries = collections.deque()`, which stays in `process_overlaps`).

Immediately before `def process_overlaps` (i.e., as a new method placed right above it), add:

```python
@dataclass
class _OverlapPlan:
    """Every CPU-only step of process_overlaps, materialized ahead of any GPU
    call: which windows to build and what each one produced. Safe to build on
    a fork_for_file() clone from a background thread (see prefetch_overlap_plan)."""
    speech: list
    pairs: list
    enrollments: dict
    seg_by_index: dict
    jobs: list                # the flattened (subjob, outcome) sequence expanded_window_iter() used to yield
    stats_jobs: int
    stats_pairs: int
    overlap_durations: list


class SeparationService:
    ...

    def _build_overlap_plan(self, segments: List[Segment], audio: AudioData,
                            overlap_threshold: float = 0.1) -> "_OverlapPlan":
        """Every CPU-only step of process_overlaps: detect pairs, mine
        enrollments, group jobs, and materialize every window plan. No GPU
        call happens here -- self.bss_model is never read -- so this is safe
        to run on a fork_for_file() clone, from a background thread, for a
        file whose diarization just finished while this SeparationService's
        real instance may be handling a different file's actual separation."""
        seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "index": s.index}
                     for s in segments]
        pairs = detect_overlapping_segments(seg_dicts, overlap_threshold=0.0, logger=self.logger)

        micro = [p for p in pairs if p["overlap_end"] - p["overlap_start"] < MIN_OVERLAP_SECONDS]
        if micro and self.logger:
            self.logger.info(
                f"[TSE] skipping {len(micro)} micro-overlap(s) < {MIN_OVERLAP_SECONDS*1000:.0f}ms")
        pairs = [p for p in pairs if p["overlap_end"] - p["overlap_start"] >= MIN_OVERLAP_SECONDS]

        speech = [SpeechSegment(**s.__dict__) for s in segments]
        sr = audio.sample_rate
        waveform = audio.waveform
        total_dur = len(waveform) / sr
        for e in speech:
            e.audio = waveform[round(e.start * sr):round(e.end * sr)].copy()

        if not pairs:
            if self.logger:
                self.logger.info(
                    f"[TSE] no overlap >= {overlap_threshold}s among {len(segments)} segments"
                )
            return _OverlapPlan(speech=speech, pairs=[], enrollments={}, seg_by_index={},
                                jobs=[], stats_jobs=0, stats_pairs=0, overlap_durations=[])

        enrollments = self.mine_enrollments(segments, audio)
        seg_by_index = {s.index: s for s in speech}

        stats_jobs = 0
        stats_pairs = 0
        overlap_durations = []
        queue, same_speaker_pairs = self._group_jobs(pairs)
        buildable = []
        for spk_a, spk_b, plist in queue:
            stats_jobs += 1
            stats_pairs += len(plist)
            for p in plist:
                overlap_durations.append(p["overlap_end"] - p["overlap_start"])
            targets = self._splice_pairs(plist)
            buildable.append((spk_a, spk_b, plist, targets))

        # same_speaker: hai segment cùng nhãn chồng nhau — không có gì để tách.
        # Mở rộng vùng base ra hai bên (±2s) rồi giữ nguyên mixture.
        # Không gọi model, không cần speaker thứ hai, không thay đổi audio.
        # Chỉ ghi vào bss_spans với sim=-2 (passthrough marker) để downstream
        # biết vùng này đã được xem xét, không phải bỏ sót.
        by_index = {e.index: e for e in speech}
        for p in same_speaker_pairs:
            lo, hi = p["overlap_start"], p["overlap_end"]
            pad = 2.0
            lo_ext = max(0.0, lo - pad)
            hi_ext = min(total_dur, hi + pad)
            for side in ("seg1", "seg2"):
                enh = by_index.get(p[side].get("index"))
                if enh is None:
                    continue
                if any(not (hi_ext <= a or lo_ext >= b) for a, b, _ in enh.bss_spans):
                    continue
                dst = round(lo_ext * sr) - round(enh.start * sr)
                limit = round(hi_ext * sr) - round(lo_ext * sr)
                if dst < 0 or dst + limit > len(enh.audio):
                    continue
                enh.bss_spans.append((lo_ext, lo_ext + limit / sr, -2.0))
            if self.logger:
                self.logger.info(
                    f"[TSE] same_speaker {lo:.2f}-{hi:.2f}s: "
                    f"base extended ±{pad}s, mixture kept")
        if self.logger:
            self.logger.info(
                f"[TSE] {len(pairs)} overlap pairs -> {len(queue)} separation jobs "
                "(missing enrollment uses best-effort/complement assignment)"
            )

        file_windows = None
        if buildable and _worth_pooling(len(buildable)):
            pool_size = _resolve_pool_size()
            if pool_size > 0:
                try:
                    state = self._window_pool_state
                    with state["lock"]:
                        if state["pool"] is None:
                            state["pool"] = WindowBuildPool(
                                n_workers=pool_size,
                                max_pending=min(BSS_WINDOW_MAX_PENDING,
                                                max(1, pool_size * 2)))
                            if self.logger:
                                self.logger.info(
                                    f"[TSE] window pool started with {pool_size} worker "
                                    "process(es) (persists for the rest of this batch)")
                        self._window_pool = state["pool"]
                    file_windows = self._window_pool.open_file(
                        segments, pairs, waveform, sr, music_map=self.music_map,
                        seams=self.seams(), context_seconds=BSS_STITCH_EDGE_PAD,
                        max_context_seconds=BSS_STITCH_EDGE_MAX,
                        padding_min_seconds=BSS_PADDING_MIN_PER_SPEAKER,
                        search_seconds=BSS_STITCH_SEARCH, use_vad=False)
                    if self.logger:
                        self.logger.info("[TSE] building windows for this file in parallel")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"[TSE] window pool failed for this file ({type(e).__name__}: {e}); "
                            "falling back to sequential build")
                    file_windows = None

        if file_windows is not None:
            window_iter = file_windows.build_all([plist for _a, _b, plist, _t in buildable])
        else:
            planner = WindowPlanner(
                segments, pairs, waveform, sr, music_map=self.music_map,
                seams=self.seams(), vad=getattr(self.bss_model, "_vad", None),
                context_seconds=BSS_STITCH_EDGE_PAD,
                max_context_seconds=BSS_STITCH_EDGE_MAX,
                padding_min_seconds=BSS_PADDING_MIN_PER_SPEAKER,
                search_seconds=BSS_STITCH_SEARCH)

            def window_iter():
                for _a, _b, plist, _t in buildable:
                    try:
                        r = planner.build_many(plist)
                        yield r, planner.reason, planner.detail, list(planner.actions)
                    except Exception as exc:
                        actions = list(getattr(planner, "actions", []))
                        actions.append({
                            "step": len(actions) + 1,
                            "action": "window_builder_exception",
                            "detail": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        })
                        yield None, "window_error", actions[-1]["detail"], actions
            window_iter = window_iter()

        def expanded_window_iter():
            for job, outcome in zip(buildable, window_iter):
                spk_a, spk_b, plist, targets = job
                plans, reason, detail, actions = outcome
                if plans is None:
                    yield job, (None, reason, detail, actions)
                    continue

                emitted = False
                for plan in plans:
                    core_lo, core_hi = plan.core_source_samples
                    if core_hi <= core_lo:
                        clipped_targets = list(targets)
                    else:
                        lo_seconds, hi_seconds = core_lo / sr, core_hi / sr
                        clipped_targets = [
                            (sd, max(lo, lo_seconds), min(hi, hi_seconds))
                            for sd, lo, hi in targets
                            if min(hi, hi_seconds) > max(lo, lo_seconds)
                        ]
                    if not clipped_targets:
                        continue
                    emitted = True
                    subjob = (spk_a, spk_b, plist, clipped_targets)
                    yield subjob, (
                        plan.window, plan.reason, plan.detail, plan.actions
                    )

                if not emitted:
                    yield job, (None, reason, detail or "no_core_targets", actions)

        # Drained eagerly: by the time this list exists, every pool job for
        # this file has finished, so file_windows' shared memory can close
        # now instead of waiting for the (separate) GPU consumption loop.
        jobs = list(expanded_window_iter())
        if file_windows is not None:
            file_windows.close()

        return _OverlapPlan(
            speech=speech, pairs=pairs, enrollments=enrollments, seg_by_index=seg_by_index,
            jobs=jobs, stats_jobs=stats_jobs, stats_pairs=stats_pairs,
            overlap_durations=overlap_durations)
```

Finally, back inside `process_overlaps`, right after the `buildable = None` line you added earlier, splice in a tiny generator wrapper and then keep everything from `pending_retries = collections.deque()` onward completely unchanged:

```python
        pairs = plan.pairs
        buildable = None  # no longer built here; see _build_overlap_plan

        recovery_planner = WindowPlanner(
            segments, pairs, waveform, sr, music_map=self.music_map,
            seams=self.seams(), vad=getattr(self.bss_model, "_vad", None),
            context_seconds=BSS_STITCH_EDGE_PAD,
            max_context_seconds=BSS_STITCH_EDGE_MAX,
            padding_min_seconds=BSS_PADDING_MIN_PER_SPEAKER,
            search_seconds=BSS_STITCH_SEARCH)

        def expanded_window_iter():
            return iter(plan.jobs)

        pending_retries = collections.deque()
        previous_outputs = {}
        # ... everything from here to `return speech` at the end of the
        # method is UNCHANGED -- do not touch retry_failed, processing_iter,
        # scheduled_processing_iter, the big `for ... in
        # scheduled_processing_iter():` loop, or the trailing
        # `self._finalize_failure_artifacts(speech, sr)` /
        # `self._report_stats()` / `return speech`.
```

The original code already declared `recovery_planner` right there (it is unchanged, just now built from `plan.pairs` instead of the local `pairs` it used to compute inline — same values). The only structural edit in this second half of the method is replacing the old `expanded_window_iter` generator function (which built windows lazily) with the two-line version above (which just iterates the already-built list) — `processing_iter()` and `scheduled_processing_iter()` both already call it as `iter(expanded_window_iter())`, so neither needs to change.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_separation_logic.py tests/test_separation_recovery.py tests/test_ghost_speakers.py tests/test_separation_window.py tests/test_separation_backends.py -q`
Expected: PASS — every test in all five files, including the two new ones from Task 1 and Task 2. This is the regression gate for the whole extraction: it means `process_overlaps` produces identical output whether or not the CPU-only work is a separate method call.

- [ ] **Step 5: Commit**

```bash
git add services/separation_service.py tests/test_separation_logic.py
git commit -m "refactor(separation): extract _build_overlap_plan from process_overlaps"
```

---

### Task 3: Prefetch cache on `SeparationService`

**Files:**
- Modify: `podcast-pipeline/services/separation_service.py` (`__init__`, `fork_for_file`, new methods)
- Create: `podcast-pipeline/tests/test_separation_prefetch.py`

**Interfaces:**
- Consumes: `SeparationService._build_overlap_plan(segments, audio, overlap_threshold=0.1) -> _OverlapPlan` (Task 2), `SeparationService.fork_for_file(self) -> SeparationService` (already exists, unchanged).
- Produces: `SeparationService.prefetch_overlap_plan(self, segments, audio, audio_path: str) -> None`, `SeparationService._take_prefetched_plan(self, audio_path: str) -> Optional[_OverlapPlan]` (already consumed by `process_overlaps`, wired in Task 2's Step 3), `SeparationService.drop_prefetched_plan(self, audio_path: str) -> None`, `SeparationService.close_prefetch_pool(self) -> None`. Task 4 calls `prefetch_overlap_plan` and `close_prefetch_pool`; Task 5 calls `drop_prefetched_plan`.

- [ ] **Step 1: Write the failing test**

Create `podcast-pipeline/tests/test_separation_prefetch.py`:

```python
"""Kiểm thử cache dựng cửa sổ trước (prefetch), tách khỏi test_separation_logic.py
vì đây là mối quan tâm khác: bộ nhớ đệm/luồng nền, không phải chính sách overlap.
Chạy trong podcast-pipeline: python -m pytest tests/test_separation_prefetch.py -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.audio import AudioData
from schemas.segment import Segment
from services.separation_service import SeparationService

SR = 24000


class FakeTSE:
    def __init__(self, sim_a=0.6, sim_b=0.6):
        self.sim_a, self.sim_b = sim_a, sim_b
        self.calls = []

    def separate_two_speakers(self, mixture_audio, enroll_A, enroll_B, sample_rate,
                              id_A, id_B, probe_A=None, probe_B=None, core_range=None):
        self.calls.append(len(mixture_audio))
        return (np.full(len(mixture_audio), 0.5, dtype=np.float32),
                np.full(len(mixture_audio), -0.5, dtype=np.float32),
                self.sim_a, self.sim_b,
                {"anchor_self": 0.6, "anchor_other": 0.1, "other_rms": 0.5})


def _audio(duration=60.0):
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 0.05, int(duration * SR)).astype(np.float32)
    wave[np.arange(len(wave)) % (SR // 2) < int(0.08 * SR)] = 0
    return AudioData(waveform=wave, sample_rate=SR, name="test", audio_segment=None,
                     duration=duration)


def _dialogue():
    return [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=14.0, end=14.4, speaker="SPEAKER_01"),
        Segment(index="00003", start=32.0, end=40.0, speaker="SPEAKER_01"),
        Segment(index="00004", start=42.0, end=48.0, speaker="SPEAKER_00"),
    ]


def test_prefetching_produces_the_same_result_as_building_inline():
    """The point of prefetching is that it changes nothing observable --
    only when the CPU work happens, not what it produces."""
    plain = SeparationService(FakeTSE(), logger=None)
    plain_out = plain.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)

    prefetched = SeparationService(FakeTSE(), logger=None)
    prefetched.prefetch_overlap_plan(_dialogue(), _audio(), "b.mp3")
    prefetched_out = prefetched.process_overlaps(
        _dialogue(), _audio(), overlap_threshold=0.1, audio_path="b.mp3")

    assert len(plain_out) == len(prefetched_out)
    for a, b in zip(plain_out, prefetched_out):
        assert a.index == b.index
        assert a.bss == b.bss
        assert a.bss_spans == b.bss_spans
    assert prefetched.stats["pairs"] == plain.stats["pairs"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_separation_prefetch.py::test_prefetching_produces_the_same_result_as_building_inline -v`
Expected: FAIL with `AttributeError: 'SeparationService' object has no attribute 'prefetch_overlap_plan'`.

- [ ] **Step 3: Write the minimal implementation**

In `podcast-pipeline/services/separation_service.py`, in `SeparationService.__init__`, find:

```python
        self._window_pool = None
        self._window_pool_state = {
            "pool": None,
            "lock": threading.Lock(),
        }
```

Add right after it:

```python
        # Overlap plans built ahead of time, on a background thread, the
        # moment a file's diarization result exists (see
        # prefetch_overlap_plan). Keyed by audio path; shared across every
        # fork_for_file() clone the same way _window_pool_state is, since a
        # plan is a finished snapshot -- there is no per-file mutable state
        # left in it for two files to race on.
        self._prefetch_cache = {}
        self._prefetch_lock = threading.Lock()
        self._prefetch_executor = None
```

In `fork_for_file`, find the comment at the end of the method:

```python
        # _window_pool_state is intentionally shared: it owns stateless CPU
        # workers, while each open_file call owns separate shared memory.
        return clone
```

Replace with:

```python
        # _window_pool_state, _prefetch_cache/_prefetch_lock/_prefetch_executor
        # are intentionally shared (copy.copy() above already does this,
        # since these are mutable container objects copied by reference):
        # they own stateless CPU workers and finished plan snapshots, not
        # anything a clone needs its own copy of.
        return clone
```

Now add the four new methods. A convenient anchor is right after `close_window_pool` (which ends with `if pool is not None: pool.close()`); add these immediately below it:

```python
    def _prefetch_pool(self) -> ThreadPoolExecutor:
        """The shared coordinator pool for background overlap-plan building.

        Small on purpose: each submitted task mostly waits on the process
        pool in utils/window_pool.py (or runs the sequential planner
        directly, for a file too small to pool) rather than doing CPU work
        of its own."""
        with self._prefetch_lock:
            if self._prefetch_executor is None:
                self._prefetch_executor = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="overlap-prefetch")
            return self._prefetch_executor

    def prefetch_overlap_plan(self, segments: List[Segment], audio: AudioData,
                              audio_path: str) -> None:
        """Start building this file's overlap windows now, on a background
        thread, using only CPU. Call this the moment a file's diarization is
        ready (see PipelineService.run(), right where it returns for
        --stop_after diarization), so the work runs while DiariZen is still
        busy with later files and before Sidon has even started loading.

        Safe to call from the pipeline's main thread: fork_for_file() takes a
        shallow copy of self synchronously, right here, which snapshots
        self.music_map/self.timeline as they are for THIS file, before the
        caller moves on and mutates them for the next one."""
        if not segments or audio_path in self._prefetch_cache:
            return
        clone = self.fork_for_file()
        future = self._prefetch_pool().submit(clone._build_overlap_plan, segments, audio)
        with self._prefetch_lock:
            self._prefetch_cache[audio_path] = future

    def _take_prefetched_plan(self, audio_path: str) -> Optional["_OverlapPlan"]:
        """The plan prefetch_overlap_plan started for this path, or None.

        Blocks until that background build finishes if it has not already --
        by separation-stage time this is normally instant, since the file's
        diarization finished a whole stage-major pass earlier. A prefetch
        that raised is logged and treated as a miss: process_overlaps then
        builds the plan itself, exactly as it would with no prefetch at all."""
        with self._prefetch_lock:
            future = self._prefetch_cache.pop(audio_path, None)
        if future is None:
            return None
        try:
            return future.result()
        except Exception as exc:
            if self.logger:
                self.logger.warning(
                    f"[TSE] prefetch for {os.path.basename(audio_path)} failed "
                    f"({type(exc).__name__}: {exc}); building its windows now")
            return None

    def drop_prefetched_plan(self, audio_path: str) -> None:
        """Discard a prefetch that will never be consumed -- the file failed
        and its checkpoint was wiped, so a retry must not be handed a plan
        built for the attempt that no longer exists. Safe to call for a path
        with nothing cached."""
        with self._prefetch_lock:
            self._prefetch_cache.pop(audio_path, None)

    def close_prefetch_pool(self) -> None:
        """Shut down the background coordinator pool, once, at the end of
        the 'separation' stage's batch-wide scope -- not the 'diarization'
        stage's, since a prefetch started there is only consumed once the
        separation pass reaches this file. Mirrors close_window_pool()."""
        with self._prefetch_lock:
            executor = self._prefetch_executor
            self._prefetch_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_separation_prefetch.py -q`
Expected: PASS.

- [ ] **Step 5: Add the fallback and lifecycle tests, then run them**

Append to `podcast-pipeline/tests/test_separation_prefetch.py`:

```python
def test_a_failed_prefetch_falls_back_to_building_now(monkeypatch):
    """A background build can fail for reasons that have nothing to do with
    the file (a transient pool error). process_overlaps must not propagate
    that -- it must build the plan itself, the same way it would with no
    prefetch at all."""
    calls = {"n": 0}
    original = SeparationService._build_overlap_plan

    def flaky(self, segments, audio, overlap_threshold=0.1):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return original(self, segments, audio, overlap_threshold)

    monkeypatch.setattr(SeparationService, "_build_overlap_plan", flaky)

    svc = SeparationService(FakeTSE(), logger=None)
    svc.prefetch_overlap_plan(_dialogue(), _audio(), "a.mp3")
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1, audio_path="a.mp3")

    assert calls["n"] == 2, "the prefetch attempt failed once, then process_overlaps built it itself"
    assert len(out) == len(_dialogue())
    assert any(s.bss for s in out), "the fallback build must still separate the overlap"


def test_drop_prefetched_plan_discards_an_unused_prefetch():
    svc = SeparationService(FakeTSE(), logger=None)
    svc.prefetch_overlap_plan(_dialogue(), _audio(), "d.mp3")
    svc.drop_prefetched_plan("d.mp3")
    assert "d.mp3" not in svc._prefetch_cache
    svc.drop_prefetched_plan("never-existed.mp3")  # must not raise


def test_close_prefetch_pool_is_idempotent():
    svc = SeparationService(FakeTSE(), logger=None)
    svc.prefetch_overlap_plan(_dialogue(), _audio(), "c.mp3")
    svc._take_prefetched_plan("c.mp3")  # drain it so the executor did real work
    svc.close_prefetch_pool()
    assert svc._prefetch_executor is None
    svc.close_prefetch_pool()  # must not raise when called twice
```

Run: `python -m pytest tests/test_separation_prefetch.py -q`
Expected: PASS, all five tests in the file.

- [ ] **Step 6: Commit**

```bash
git add services/separation_service.py tests/test_separation_prefetch.py
git commit -m "feat(separation): prefetch overlap-window plans on a background thread"
```

---

### Task 4: Wire the prefetch into `PipelineService`

**Files:**
- Modify: `podcast-pipeline/services/pipeline_service.py` (the diarization stop point; the real `process_overlaps` call site; the deferred-close block after separation)
- Modify: `podcast-pipeline/tests/test_stage_scheduling.py`

**Interfaces:**
- Consumes: `SeparationService.prefetch_overlap_plan(segments, audio, audio_path)`, `SeparationService.process_overlaps(segments, audio, overlap_threshold=0.1, audio_path=None)`, `SeparationService.close_prefetch_pool()` (all from Tasks 2-3).

- [ ] **Step 1: Write the failing tests**

Add to `podcast-pipeline/tests/test_stage_scheduling.py`:

```python
def test_window_prefetch_fires_only_on_the_diarization_stop_and_only_when_separation_runs():
    """Window building is pure CPU and does not need Sidon loaded, so it can
    start the moment this file's diarization result exists -- while DiariZen
    is still busy with the rest of the batch and before Sidon's worker has
    even started. It must fire on exactly the one pass that stops after
    diarization, not on a later pass re-entering that section of run() to
    load the checkpoint on its way to another stage."""
    src = _source("services/pipeline_service.py")
    stop = src.index('getattr(args, "stop_after", None) == "diarization"')
    block = src[stop:src.index("return None", stop)]
    assert "prefetch_overlap_plan" in block, (
        "the diarization pass must kick off window building while it still "
        "has the rest of the batch to run on GPU")
    assert 'self.step_enabled(args, "separation")' in block, (
        "prefetching must not run when separation itself is switched off")


def test_the_real_separation_call_passes_its_audio_path_to_the_prefetch_cache():
    src = _source("services/pipeline_service.py")
    call = re.search(r"self\.separation_svc\.process_overlaps\([^)]*\)", src, re.S).group(0)
    assert "audio_path=audio_path" in call, (
        "without the path, process_overlaps can never find what was prefetched for this file")


def test_closing_the_prefetch_pool_is_deferred_like_the_window_pool():
    """A prefetch started during 'diarization' is only consumed during
    'separation'; close_prefetch_pool() must be deferred to that SAME stage
    boundary as close_window_pool(), not fired eagerly, or an in-flight
    prefetch would be torn down before the separation pass ever reads it."""
    src = _source("services/pipeline_service.py")
    window_pool_defer = src.index("self.separation_svc.close_window_pool()")
    prefetch_defer = src.index("self.separation_svc.close_prefetch_pool()")
    between = src[window_pool_defer:prefetch_defer]
    assert between.count("_defer_or_run(") == 1, (
        "close_prefetch_pool must be the very next _defer_or_run() call after "
        "close_window_pool(), in the same deferred-cleanup block")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stage_scheduling.py::test_window_prefetch_fires_only_on_the_diarization_stop_and_only_when_separation_runs tests/test_stage_scheduling.py::test_the_real_separation_call_passes_its_audio_path_to_the_prefetch_cache tests/test_stage_scheduling.py::test_closing_the_prefetch_pool_is_deferred_like_the_window_pool -v`
Expected: FAIL — the first two with `AssertionError` (the text is not in `pipeline_service.py` yet), the third with `ValueError: substring not found` (`close_prefetch_pool()` does not exist yet).

- [ ] **Step 3: Write the minimal implementation**

In `podcast-pipeline/services/pipeline_service.py`, find the diarization stop point:

```python
        if getattr(args, "stop_after", None) == "diarization":
            if self.logger: self.logger.info("Stopping pipeline after diarization as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "diarization"})
            return None
```

Replace with:

```python
        if getattr(args, "stop_after", None) == "diarization":
            # This is the one pass that sets stop_after == "diarization" --
            # a later pass re-enters this section of run() only to load the
            # checkpoint on its way to another stage -- so this fires exactly
            # once per file. Window building is pure CPU (utils/window_pool.py)
            # and does not need Sidon loaded, so it can run now, in the
            # background, while DiariZen is still busy with the rest of this
            # stage-major pass and before the separation stage has even
            # started the Sidon worker. See
            # services/separation_service.py:prefetch_overlap_plan.
            if diarization_result is not None and self.step_enabled(args, "separation"):
                self.separation_svc.prefetch_overlap_plan(
                    diarization_result.segments, audio_data, audio_path)
            if self.logger: self.logger.info("Stopping pipeline after diarization as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "diarization"})
            return None
```

Find the real separation call:

```python
            speech_segments = self.separation_svc.process_overlaps(diarization_result.segments, audio_data)
```

Replace with:

```python
            speech_segments = self.separation_svc.process_overlaps(
                diarization_result.segments, audio_data, audio_path=audio_path)
```

Find the deferred cleanup block right after the separation section:

```python
        self._defer_or_run(
            lambda: self.separation_svc.close_window_pool()
            if self.separation_svc else None)
        self._defer_or_run(
            lambda: self.separation_svc.close_async_pools()
            if self.separation_svc
            and hasattr(self.separation_svc, "close_async_pools") else None)
```

Replace with:

```python
        self._defer_or_run(
            lambda: self.separation_svc.close_window_pool()
            if self.separation_svc else None)
        self._defer_or_run(
            lambda: self.separation_svc.close_prefetch_pool()
            if self.separation_svc
            and hasattr(self.separation_svc, "close_prefetch_pool") else None)
        self._defer_or_run(
            lambda: self.separation_svc.close_async_pools()
            if self.separation_svc
            and hasattr(self.separation_svc, "close_async_pools") else None)
```

(`close_prefetch_pool` is inserted directly after `close_window_pool` so `test_closing_the_prefetch_pool_is_deferred_like_the_window_pool`'s adjacency check holds.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_stage_scheduling.py -q`
Expected: PASS — every test in the file, including the three new ones.

- [ ] **Step 5: Commit**

```bash
git add services/pipeline_service.py tests/test_stage_scheduling.py
git commit -m "feat(pipeline): trigger overlap-window prefetch at the diarization stop point"
```

---

### Task 5: Drop a stale prefetch when a file's checkpoint is discarded

**Files:**
- Modify: `podcast-pipeline/main.py` (`_discard_partial`)
- Create: `podcast-pipeline/tests/test_main_discard.py`

**Interfaces:**
- Consumes: `SeparationService.drop_prefetched_plan(audio_path)` (Task 3).

`main.py` calls `argparse.parse_args()` at import time, so it cannot be imported in a test process (no existing test does). This task's test reads `main.py`'s source text instead, the same style `tests/test_stage_scheduling.py` uses for `pipeline_service.py`.

- [ ] **Step 1: Write the failing test**

Create `podcast-pipeline/tests/test_main_discard.py`:

```python
"""main.py parses sys.argv at import time, so it cannot be imported here --
these tests read its source text instead. See tests/test_stage_scheduling.py
for the same style applied to pipeline_service.py."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_discard_partial_drops_a_stale_prefetched_plan():
    """A discarded file's diarization checkpoint is gone, so its next
    attempt recomputes diarization from scratch. Any window plan already
    prefetched for the attempt that just failed must be dropped, or a retry
    could be handed a plan built before the checkpoint it depended on was
    wiped."""
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    body = src[src.index("def _discard_partial"):src.index("def main()")]
    assert "drop_prefetched_plan" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_main_discard.py -v`
Expected: FAIL with `AssertionError`.

- [ ] **Step 3: Write the minimal implementation**

In `podcast-pipeline/main.py`, inside `_discard_partial`, find:

```python
    ledger.discard_partial_output(*targets, logger=logger)

    # Release any worker subprocess that may be holding GPU memory after the
    # failure. _release_worker is a no-op when the worker is already gone.
    if pipeline is not None:
        for worker_name in ("diarizen", "sidon", "qwen3"):
```

Replace with:

```python
    ledger.discard_partial_output(*targets, logger=logger)

    # This file's diarization checkpoint is gone, so its next attempt
    # recomputes diarization from scratch. Drop any window plan already
    # prefetched for the attempt that just failed, so a retry cannot be
    # handed a plan built for a checkpoint that no longer exists.
    if pipeline is not None:
        separation_svc = getattr(pipeline, "separation_svc", None)
        drop = getattr(separation_svc, "drop_prefetched_plan", None)
        if callable(drop):
            drop(audio_path)

    # Release any worker subprocess that may be holding GPU memory after the
    # failure. _release_worker is a no-op when the worker is already gone.
    if pipeline is not None:
        for worker_name in ("diarizen", "sidon", "qwen3"):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_main_discard.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full regression suite**

Run: `python -m pytest tests -q`
Expected: PASS across the whole suite (this plan touched `services/separation_service.py`, `services/pipeline_service.py`, and `main.py`; this is the final check that nothing else in the tree assumed the old shapes).

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main_discard.py
git commit -m "fix(main): drop a stale window-plan prefetch when a file's checkpoint is discarded"
```

## Self-Review

**Spec coverage:**
- "As soon as a file's diarization result is available... start building windows in the background" → Task 4 (trigger at the diarization stop point).
- "process_overlaps finds the windows already built and skips straight to calling the separator" → Task 2 (`_build_overlap_plan`/`_OverlapPlan`) + Task 3 (`_take_prefetched_plan` wired into `process_overlaps`).
- "No checkpoint" → no task writes a new checkpoint; plans live only in `_prefetch_cache` (Task 3).
- "No RAM cap" → `prefetch_overlap_plan`/`_prefetch_cache` have no size limit (Task 3).
- "Isolation via fork_for_file()" → Task 3's `prefetch_overlap_plan` (and the thread-safety prerequisite in Task 1).
- "Stats stay single-owner" → Task 2's `process_overlaps` applies `plan.stats_jobs`/`stats_pairs`/`overlap_durations`, never `_build_overlap_plan` itself.
- "Pool lifetime spans two stage scopes" → Task 4's `close_prefetch_pool()` deferred next to `close_window_pool()`, not in the diarization stage's own scope.
- "Trigger fires exactly once per file" → Task 4, gated on `getattr(args, "stop_after", None) == "diarization"`.
- "Stale prefetch on retry" → Task 5.
- "File-major mode is unaffected" → the trigger is inside the `stop_after == "diarization"` branch, which file-major mode never sets; `process_overlaps`'s `audio_path` parameter defaults to `None`.

**Placeholder scan:** no TBD/TODO, no "add error handling", every step has real code or a real shell command.

**Type consistency:** `_OverlapPlan` fields (`speech`, `pairs`, `enrollments`, `seg_by_index`, `jobs`, `stats_jobs`, `stats_pairs`, `overlap_durations`) are the same across Task 2 (defines them) and Task 3 (reads `plan.jobs` implicitly via `_take_prefetched_plan`'s return type, matches `process_overlaps`'s usage in Task 2 Step 3). `_group_jobs`'s new `(jobs, same_speaker_pairs)` return (Task 1) matches its one call site inside `_build_overlap_plan` (Task 2) and its three test call sites (Task 1). `prefetch_overlap_plan(segments, audio, audio_path)` / `process_overlaps(segments, audio, overlap_threshold=0.1, audio_path=None)` signatures match between Task 3's definition and Task 4's call sites.
