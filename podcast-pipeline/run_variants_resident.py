#!/usr/bin/env python3
"""Run every noise-filter variant in ONE process, with the models loaded once and kept.

tools/run_variants.py starts one main.py per variant, so each variant loads the diarizer, Sidon,
the Roformers and Qwen3 again and sees one GPU. This file imports main.py once, lets main() build
the services once (--keep_models), and swaps main's `run_batch_by_stage` for a scheduler that calls
the real one once per variant (same input files, other denoise points / job_id / save_path).

Only what main() reads from `args` per call can differ between variants: the three denoise points,
`steps.*` (as step_<name>) and `pipeline.*` keys that are not read while the services are built.
A variant that needs anything else (models.*, outputs.*, music_separator, extra_args, ...) is NOT run:
status "skipped" with the reason; use tools/run_variants.py for it. Output layout and
variants_manifest.json are the same as the old runner's, so the compare / lanes pages work unchanged.

    python run_variants_resident.py --audio-dir DIR --variants tools/variants_placement.json \\
        --out-root OUT --stop-after asr --resume          # --dry-run prints the plan and the argv
"""

import argparse
import contextlib
import copy
import importlib.util
import json
import os
import re
import shlex
import socket
import sys
import tempfile
import uuid
import time
from datetime import datetime
from pathlib import Path

try:
    import fcntl
except ImportError:  # not POSIX: the manifest lock degrades to nothing
    fcntl = None

ROOT = Path(__file__).resolve().parent
TOOL = ROOT / "tools" / "run_variants.py"
RESIDENT_DIR = "_resident"
JOB_ID = "exp-resident"

# --layout split: the ASR models on gpu_2, everything else on gpu_1. Edit here to change it.
# Keys are `--set` paths into the profile; all of them live in utils/performance_config.py.
# split leaves the speaker-assignment workers on both cards (main.py starts them on gpu_1 and gpu_2): stages run
# one after another, so they never compete with ASR, they only hold a little VRAM on gpu_2.
# split_strict also turns them off, which keeps gpu_2 to the ASR models alone but assigns speakers in the main process.
LAYOUT_PRESETS = {
    "default": {},
    "split": {
        "performance.stages.asr.qwen3_gpu": "gpu_2",
        "performance.stages.asr.whisper_gpu": "gpu_2",
        "performance.stages.asr.phowhisper_gpu": "gpu_2",
        "performance.stages.asr.phowhisper_workers": 1,       # one copy, on phowhisper_gpu
        "performance.stages.asr.dynamic_replicas": False,     # no Qwen3 replica on the other card
        "performance.stages.asr.max_replicas_per_model": 1,   # cross_file scheduler: no replica either
        "performance.stages.diarization.placement": "single",
        "performance.stages.diarization.workers": 1,
        "performance.stages.separation.max_workers": 1,
        "performance.stages.music.max_separator_workers": 1,
        "performance.stages.music.tagger_workers": 1,
        "performance.stages.music.cross_file_overlap": False,
    },
}
LAYOUT_PRESETS["split_strict"] = {**LAYOUT_PRESETS["split"],
                                  "performance.stages.separation.assignment_workers_per_gpu": 0}
# --music off: what is forced on every variant, and the cache that keeps these runs apart from music-on ones.
MUSIC_OFF_SETS = ("steps.music_analysis=false", "steps.music_removal=false")
NO_MUSIC_CACHE_DIR = "cache_nomusic"
# --stop-after values that end before refinement; later ones put the LLM on both cards (main.py pins that).
STOPS_BEFORE_REFINEMENT = frozenset({"input_quality", "music", "denoise", "diarization", "denoise_pre_separation",
                                     "separation_plan", "separation", "music_removal", "denoise_pre_asr", "asr",
                                     "captioning"})

# Copy of PipelineService._STOP_ORDER (importing that needs torch); a test compares it with the source text.
STAGE_ORDER = ("input_quality", "music", "denoise", "diarization",
               "denoise_pre_separation", "separation_plan", "separation", "music_removal",
               "denoise_pre_asr", "asr", "captioning", "refinement", "speaker_relabel",
               "transcript_review", "persona_profile", "conversation_selection",
               "word_alignment", "conversation_exports")
# (variant key of a denoise point, stop point at which that point has run)
POINT_STAGE = (("chain", "denoise"), ("chain_pre_separation", "denoise_pre_separation"),
               ("chain_pre_asr", "denoise_pre_asr"))

# The processes of the line, in order: each hands its stage's files on by .ready_<stage>[.<stem>] markers. A stage
# with nothing to do for a variant is skipped (see is_noop). "all" (--stop-after all) is the last process: no stop_after
# for main.py, it runs everything after conversation_selection.
LINE = ("denoise", "diarization", "denoise_pre_separation", "separation_plan", "separation", "denoise_pre_asr",
        "asr", "conversation_selection")
FULL_STOPS = (None, "none", "all")
MUSIC_KEYS = ("steps.music_analysis", "steps.music_removal")
NOOP_NOTE = "nothing to do at this point"

# pipeline.* keys read while main() builds the services: they cannot change per variant.
BUILD_TIME_PIPELINE_KEYS = frozenset({
    "ASRMoE", "bss", "separator", "lang", "prefetch_workers", "dia3", "keep_models", "by_stage",
    "asr_models", "diarization_backend", "job_id", "cache_dir", "gpu_1", "gpu_2", "performance",
    "denoise", "denoise_pre_separation", "denoise_pre_asr", "stop_after"})
# steps that decide at build time whether a worker / model is created at all.
BUILD_GATED_STEPS = frozenset({"diarization", "separation", "asr", "transcript_review", "persona_profile"})

OVERLAP_MESSAGE = ("--mode overlap: not implemented: the pipeline shares per-stage state "
                   "(timeline, noise_track, music_map, worker scopes), see doc/chay_nhieu_variant_mot_process.md")
NEEDS_OWN_PROCESS = "set key {key} needs its own process (use tools/run_variants.py)"

_rv = None


def rv():
    """tools/run_variants.py, loaded on first use (it is a script, tools/ is not a package)."""
    global _rv
    if _rv is None:
        spec = importlib.util.spec_from_file_location("run_variants_tool", TOOL)
        _rv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_rv)
    return _rv


# ------------------------------------------------------------------ per-variant settings
def unsupported_reason(variant):
    """Why this variant cannot share the resident process, or None. Never ignores a key."""
    for key in variant.get("set") or {}:
        if key.startswith("steps.") and key[len("steps."):]:
            continue
        if key.startswith("pipeline.") and key[len("pipeline."):] not in BUILD_TIME_PIPELINE_KEYS \
                and key[len("pipeline."):]:
            continue
        return NEEDS_OWN_PROCESS.format(key=key)
    if variant.get("music_separator"):
        return NEEDS_OWN_PROCESS.format(key="music_separator")
    if variant.get("extra_args"):
        return NEEDS_OWN_PROCESS.format(key="extra_args")
    return None


def variant_overrides(variant, stop_after=None, out_root=None):
    """The args fields a variant changes: denoise points, job_id, steps and pipeline keys."""
    over = {"denoise": variant.get("spec") or "none",
            "denoise_pre_separation": variant.get("spec_pre_separation") or "none",
            "denoise_pre_asr": variant.get("spec_pre_asr") or "none",
            "job_id": f"exp-{variant['id']}",
            "stop_after": None if stop_after in (None, "", "none", "all") else stop_after}
    if out_root is not None:
        over["save_path"] = str(Path(out_root) / variant["id"])
    for key, val in (variant.get("set") or {}).items():
        if key.startswith("steps."):
            over["step_" + key[len("steps."):]] = bool(val)
        elif key.startswith("pipeline."):
            over[key[len("pipeline."):]] = val
    return over


def gate_problem(variant, config, env):
    """A step this variant turns ON that the profile built no worker for, or None."""
    profile = ((config or {}).get("environments") or {}).get(env, {})
    steps = profile.get("steps") or {}
    for key, val in (variant.get("set") or {}).items():
        name = key[len("steps."):] if key.startswith("steps.") else None
        if name in BUILD_GATED_STEPS and val and not steps.get(name, True):
            return (f"set key {key} turns on a stage the profile did not build at startup "
                    "(use tools/run_variants.py)")
    return None


def stage_rank(stop):
    """Position of a stop point; None / "none" runs everything, an unknown one sorts before all."""
    if stop in FULL_STOPS:
        return len(STAGE_ORDER)
    return STAGE_ORDER.index(stop) if stop in STAGE_ORDER else -1


def reached_variant(variant, stop):
    """Shallow copy of the variant without the denoise chains that only run after `stop`."""
    out = dict(variant)
    for key, stage in POINT_STAGE:
        if stage_rank(stop) < stage_rank(stage):
            out[key] = []
    return out


def music_active(variant, music="off"):
    """Whether the music stage does real work for this variant (analysis or removal switched on)."""
    sets = variant.get("set") or {}
    vals = [sets[k] for k in MUSIC_KEYS if k in sets]
    if music == "off":                      # --music off forces both off; only the variant's own set can undo it
        return any(vals)
    return not (len(vals) == len(MUSIC_KEYS) and not any(vals))


def is_noop(variant, stage, music="off"):
    """Stage of the line that has nothing to do for this variant (the others are always real work)."""
    if stage == "denoise":
        return not variant.get("chain") and not music_active(variant, music)
    if stage == "denoise_pre_separation":
        return not variant.get("chain_pre_separation")
    if stage == "denoise_pre_asr":
        return not variant.get("chain_pre_asr")
    return False


def noop_stages(variant, music="off"):
    return [st for st in LINE if is_noop(variant, st, music)]


def effective_stage(variant, stage, music="off"):
    """Nearest stage at or before `stage` that is real work for the variant; None = the whole line up to it is a no-op."""
    if stage not in LINE:
        return stage
    for st in reversed(LINE[:LINE.index(stage) + 1]):
        if not is_noop(variant, st, music):
            return st
    return None


def plan_ownership(variants, skipped, shard):
    """Ids this process owns: plan entries (after skips) whose index % N == K-1."""
    k, n = shard
    eligible = [v["id"] for v in variants if v["id"] not in skipped and not unsupported_reason(v)]
    return {vid for i, vid in enumerate(eligible) if i % n == k - 1}


# Speed knobs (workers, placement) never change what a variant produces, and the processes of the line differ
# in them (the separation process has its own Sidon worker count): they must not make one process's .ready_<stage>
# look stale to the next, nor invalidate a finished variant.
HASH_IGNORED_SETS = ("performance.",)


def variant_hash(variant, args, files, stop_after=None):
    stop = args.stop_after if stop_after is None else stop_after
    sig = {"id": variant["id"], "env": args.env,
           "overrides": variant_overrides(variant, stop),
           "global_set": [g for g in (args.global_set or []) if not g.startswith(HASH_IGNORED_SETS)]}
    return rv().run_hash([json.dumps(sig, sort_keys=True)], files)


def reset_between_variants(pipeline, paths=()):
    """Drop what a variant could leave on the shared services for the next one (see the doc)."""
    sep = getattr(pipeline, "separation_svc", None)
    for path in paths:
        drop = getattr(sep, "drop_prefetched_plan", None)
        if callable(drop):
            drop(path)
    cache = getattr(sep, "_prefetch_cache", None)       # plans keyed by the shared input path
    if isinstance(cache, dict):
        lock = getattr(sep, "_prefetch_lock", None)
        if lock is not None:
            with lock:
                cache.clear()
        else:
            cache.clear()
    for name in ("_pending_diar_jobs", "_pending_asr_jobs"):
        pending = getattr(pipeline, name, None)
        if isinstance(pending, dict):
            pending.clear()
    if getattr(pipeline, "noise_track", None) is not None:
        pipeline.noise_track = None
    timeline = getattr(pipeline, "timeline", None)
    if timeline is not None:
        try:
            pipeline.timeline = type(timeline)()
        except Exception:
            pass


# --------------------------------------------------------------------------- runner
class ResidentRunner:
    def __init__(self, args, variants, files, run_batch=None, skipped=None, say=None,
                 sleep=None, clock=None):
        self.args, self.variants, self.files = args, variants, files
        self.run_batch = run_batch
        self.say = say or (lambda msg: print(msg, flush=True))
        self.sleep, self.clock = sleep or time.sleep, clock or time.time
        self.out_root = Path(args.out_root)
        self.stems = [f.stem for f in files]
        self.reasons = dict(skipped or {})
        self.status = {vid: "skipped" for vid in self.reasons}
        self.reports = {}
        self.reference = rv().build_reference(self.stems, args.reference_dir, args.reference)
        self.seen, self.failures, self.crashed = {}, {}, {}
        self.elapsed, self.finalized = {}, set()
        self.gated = False
        self._prefetch_checked = False
        self.discard_partial = None      # main._discard_partial, set by run_sequential
        self.signal = getattr(args, "signal", None)
        self.claim = bool(getattr(args, "claim", False))
        self.reclaim_after = float(getattr(args, "reclaim_after", 0.0) or 0.0)
        # what this copy hands over: its --signal, or (a claim copy without one) its stop point, so claims and
        # per-file markers always have a stage name; the .done name below stays on --signal alone
        self.mark = self.signal or (args.stop_after if self.claim else None)
        self.wait_signal = getattr(args, "wait_signal", None)
        self.poll_seconds = float(getattr(args, "poll_seconds", 15.0))
        self.wait_timeout = float(getattr(args, "wait_timeout", 0.0))
        self.owned = plan_ownership(variants, set(self.reasons), getattr(args, "shard", (1, 1)))
        self.per_file = bool(getattr(args, "per_file", False))
        self.music = getattr(args, "music", "off")
        self.notes = {}                  # short reasons shown in the manifest for variants that are fine
        # A producer's resume marker is its own, so it never makes the consumer think the variant is done.
        self.done_name = f"{rv().DONE_MARKER}.{args.stop_after}" if self.signal else rv().DONE_MARKER

    # -- bookkeeping
    @contextlib.contextmanager
    def _manifest_lock(self):
        self.out_root.mkdir(parents=True, exist_ok=True)
        with open(self.out_root / ".manifest.lock", "a+") as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            yield                         # closing the file releases the lock

    def _manifest(self):
        """Read-modify-write of variants_manifest.json: our entries replace ours, the rest is kept."""
        r, out, stop = rv(), self.out_root, self.args.stop_after
        mine = [v for v in self.variants if v["id"] in self.owned or self.status.get(v["id"]) == "skipped"]
        with self._manifest_lock():
            try:
                prior = json.loads((out / r.MANIFEST_NAME).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prior = {}
            old = {e["id"]: e for e in prior.get("variants") or [] if isinstance(e, dict) and "id" in e}
            # write_manifest only writes OUR_ROOT/variants_manifest.json: build the entries in a scratch dir
            # whose variant dirs link to the real ones, so out_dir and files point at the real output.
            with tempfile.TemporaryDirectory(dir=out, prefix=".manifest_") as scratch:
                for v in mine:
                    os.symlink((out / v["id"]).resolve(), Path(scratch) / v["id"])
                r.write_manifest(scratch, mine, set(self.stems), self.status, self.reference,
                                 {**self.notes, **self.reasons}, self.reports)
                new = json.loads((Path(scratch) / r.MANIFEST_NAME).read_text(encoding="utf-8"))
            for e in new["variants"]:
                e["stop_after"] = stop
                prev = old.get(e["id"])
                keep = prev is not None and (stage_rank(prev.get("stop_after")) > stage_rank(stop)
                                             or e["status"] == "missing")
                if not keep:
                    old[e["id"]] = e
            order = [v["id"] for v in self.variants if v["id"] in old]
            order += [i for i in old if i not in order]
            new["variants"] = [old[i] for i in order]
            r.write_json_atomic(out / r.MANIFEST_NAME, new)

    def runnable(self):
        return [v for v in self.variants if v["id"] in self.owned and v["id"] not in self.reasons
                and v["id"] not in self.finalized]

    def _assess(self, variant):
        """Read the denoise.json of every point that ran before the stop; "degraded" or "ok"."""
        r = rv()
        variant = reached_variant(variant, self.args.stop_after)
        report = r.variant_report(variant, r.discover_dirs(self.out_root / variant["id"],
                                                           set(self.stems)), self.stems)
        counts = r.report_counts(report, len(self.stems))
        self.reports[variant["id"]] = {"denoise_report": report, **counts}
        if counts["suspect_files"]:
            self.say(f"[{variant['id']}] WARNING: output identical to input in "
                     f"{counts['suspect_files']}/{counts['total_files']} file(s) (suspect_noop)")
        if counts["degraded_files"]:
            self.say(f"[{variant['id']}] DEGRADED: {r.degraded_reason(report, counts)}")
            return "degraded"
        return "ok"

    # -- handoff markers between processes
    def _signal_path(self, vid, kind, stage, stem=None):
        return self.out_root / vid / (f".{kind}_{stage}" + (f".{stem}" if stem else ""))

    def _write_signal(self, v, ok, reason=None, stem=None):
        """Producer: .ready_<stage>[.<stem>] (hash of the settings the consumer recomputes) or .failed_<stage>[.<stem>]."""
        stage, vid = self.mark, v["id"]
        digest = variant_hash(v, self.args, self.files, stop_after=stage)
        ready = self._signal_path(vid, "ready", stage, stem)
        failed = self._signal_path(vid, "failed", stage, stem)
        ready.parent.mkdir(parents=True, exist_ok=True)
        if ok:
            failed.unlink(missing_ok=True)
            ready.write_text(json.dumps({"hash": digest, "stop_after": stage,
                                         "time": datetime.now().isoformat(timespec="seconds")}))
        else:
            ready.unlink(missing_ok=True)
            failed.write_text(json.dumps({"reason": reason or "failed", "hash": digest}))

    def _write_signals(self, v, ok, reason=None):
        for stem in (self.stems if self.per_file else [None]):
            self._write_signal(v, ok, reason, stem)

    def _marker_state(self, v, stage, stem=None):
        """("ready", None), ("failed", reason) or (None, None): what the producer of `stage` left for v (or one file)."""
        vid = v["id"]
        digest = variant_hash(v, self.args, self.files, stop_after=stage)
        for kind in ("failed", "ready"):
            try:
                data = json.loads(self._signal_path(vid, kind, stage, stem).read_text())
            except (OSError, ValueError):
                continue
            if kind == "failed" and data.get("hash") in (None, digest):
                return "failed", str(data.get("reason") or "failed")
            if kind == "ready" and data.get("hash") == digest:   # a stale hash is no signal at all
                return "ready", None
        return None, None

    def _signal_state(self, v, stem=None):
        """Consumer: ("ready", None), ("failed", reason) or (None, None) for what the producer left.

        The stage looked at is the nearest one before this process that is real work for the variant; when the
        whole line up to here is a no-op the variant is ready at once.
        """
        stage = effective_stage(v, self.wait_signal, self.music)
        if stage is None:
            return "ready", None
        return self._marker_state(v, stage, stem)

    # -- claims: copies of one stage share the files; the copy that creates a file's claim works on it
    def _claim_path(self, vid, stem):
        return self.out_root / vid / f".claim_{self.args.stop_after}.{stem}"

    def _take_claim(self, vid, stem, say):
        """Create the claim atomically (O_EXCL). True when this copy owns the file; another copy's claim gives False.
        A claim older than --reclaim-after minutes with no ready/failed marker is taken over (logged)."""
        path = self._claim_path(vid, stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in (0, 1):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    age = self.clock() - path.stat().st_mtime
                except OSError:
                    return False
                if attempt or self.reclaim_after <= 0 or age < self.reclaim_after * 60:
                    return False
                holder = _claim_holder(path)
                path.unlink(missing_ok=True)
                say(f"[{vid}] {stem}: claim by {holder} is {age / 60:.0f} min old with no result; taking it over")
                continue
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                                    "time": datetime.now().isoformat(timespec="seconds")}))
            return True
        return False

    def _adopt(self, v, unit, state, reason, say):
        """Another copy settled this file (its ready or failed marker is there): count it here, run nothing."""
        vid, stem = v["id"], Path(unit[0]).stem
        self.seen.setdefault(vid, set()).add(stem)
        if state == "failed":
            self.failures.setdefault(vid, []).append((unit[0], f"[{vid}] {reason}"))
        say(f"[{vid}] {Path(unit[0]).name}: {state} by another copy; not run here")
        if set(self.stems) <= self.seen[vid]:
            self._finish_variant(v)

    def _write_total_ready(self, v):
        """The stage's own .ready_<stage> once EVERY file is ready (identical content from every copy that gets here)."""
        if not all(self._marker_state(v, self.mark, st)[0] == "ready" for st in self.stems):
            return
        path = self._signal_path(v["id"], "ready", self.mark)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps({"hash": variant_hash(v, self.args, self.files, stop_after=self.mark),
                                   "stop_after": self.mark, "time": datetime.now().isoformat(timespec="seconds")}))
        os.replace(tmp, path)

    def prepare(self):
        """Resume check per variant (our own markers, not main's ledger); first manifest."""
        r = rv()
        self.out_root.mkdir(parents=True, exist_ok=True)
        for v in self.variants:
            vid = v["id"]
            if vid in self.reasons:
                continue
            why = unsupported_reason(v)
            if why:
                self.reasons[vid], self.status[vid] = why, "skipped"
                self.say(f"[{vid}] SKIPPED: {why}")
                continue
            if vid not in self.owned:
                continue
            marker = self.out_root / vid / self.done_name
            if self.args.resume and marker.is_file():
                try:
                    if json.loads(marker.read_text()).get("hash") == variant_hash(v, self.args, self.files):
                        self.say(f"[{vid}] up to date, skipped")
                        self.status[vid] = self._assess(v)
                        self.finalized.add(vid)
                        if self.mark and (self.status[vid] == "ok" or self.args.accept_degraded):
                            self._write_signals(v, True)
                except (ValueError, OSError):
                    pass
        self._manifest()

    def _finish_variant(self, v):
        vid = v["id"]
        fails, crash = self.failures.get(vid, []), self.crashed.get(vid)
        missing = set(self.stems) - self.seen.get(vid, set())
        if crash:
            state, self.reasons[vid] = "failed", crash
        elif fails:
            state, self.reasons[vid] = "failed", f"{len(fails)} file(s) failed; first: {fails[0][1]}"
        elif missing:
            state, self.reasons[vid] = "failed", f"{len(missing)} file(s) never reached"
        else:
            state = self._assess(v)
        kept = state == "ok" or (state == "degraded" and self.args.accept_degraded)
        if kept:
            (self.out_root / vid).mkdir(parents=True, exist_ok=True)     # a no-op stage ran nothing, so no dir yet
            (self.out_root / vid / self.done_name).write_text(json.dumps(
                {"hash": variant_hash(v, self.args, self.files), "exit": 0, "status": state}))
        if self.mark:
            why = self.reasons.get(vid) or "degraded: the filter fell back to unfiltered audio"
            if not self.per_file:
                self._write_signal(v, kept, why)
            elif state == "degraded" and not kept:    # per-file: ready markers went out as each file finished
                for st in self.stems:                 # the filter did not run: nothing of this variant is handed on
                    if not (self.claim and self._marker_state(v, self.mark, st)[0] == "ready"):  # not another copy's
                        self._write_signal(v, False, why, st)
        if self.claim and self.per_file and kept and self.mark:
            self._write_total_ready(v)
        self.status[vid] = state
        self.finalized.add(vid)
        self.say(f"[{vid}] {state.upper() if state != 'ok' else 'ok'} "
                 f"({self.elapsed.get(vid, 0):.0f}s)")
        self._manifest()

    def finish(self):
        """After main() returned (or died): settle variants whose files were not all covered."""
        for v in self.runnable():
            if v["id"] in self.seen or v["id"] in self.crashed:
                self._finish_variant(v)
        self._manifest()
        return self.status

    # -- the replacement for main.run_batch_by_stage
    def _discard_failed(self, pipeline, vargs, found, logger, crashed=False):
        """Clear a failed file's checkpoint and output of THIS variant, as main does for its own run.

        main's own failure handling only sees main's args (job exp-resident), so it would leave
        cache/exp-<id>_<stem> and <out>/<id>/<stem> behind and the next --resume would reload the stages
        that finished in the failed attempt.
        """
        ledger = getattr(pipeline, "progress_ledger", None)
        if crashed or not found or self.discard_partial is None or ledger is None:
            return
        for path, _err in found:
            try:
                self.discard_partial(ledger, vargs, path, logger, pipeline)
            except Exception as exc:
                self.say(f"[{vargs.job_id}] could not clear partial output of {Path(path).name}: {exc}")

    def own_noop(self, v):
        """True when this process's own stage has nothing to do for v: it only hands the variant over."""
        return self.args.stop_after in LINE and is_noop(v, self.args.stop_after, self.music)

    def _units(self, v, group):
        """What to hand to run_batch for v: the whole group, or (per-file) one file at a time, small first."""
        if not self.per_file:
            return [list(group)]
        todo = [p for p in group if Path(p).stem not in self.seen.get(v["id"], set())]

        def size(p):
            try:
                return (Path(p).stat().st_size, Path(p).name)
            except OSError:
                return (0, Path(p).name)
        return [[p] for p in sorted(todo, key=size)]

    def _stem_of(self, unit):
        return Path(unit[0]).stem if self.per_file else None

    def _run_variant(self, v, pipeline, args, config, paths, logger, stages):
        """One variant over `paths` (a whole group, or one file when per-file). Returns [(path, error)] for main."""
        vid = v["id"]
        say = logger.info if logger is not None else self.say
        noop = self.own_noop(v)
        stem = self._stem_of(paths)
        if vid not in self.seen:           # first time this process starts it: drop what an earlier run left
            (self.out_root / vid / self.done_name).unlink(missing_ok=True)
            if self.mark and not self.per_file:
                for kind in ("ready", "failed"):
                    self._signal_path(vid, kind, self.mark).unlink(missing_ok=True)
        if self.mark and self.per_file:
            for kind in ("ready", "failed"):
                self._signal_path(vid, kind, self.mark, stem).unlink(missing_ok=True)
        vargs = copy.copy(args)
        for key, val in variant_overrides(v, self.args.stop_after, self.out_root).items():
            setattr(vargs, key, val)
        grp = f" group={v['group']}" if v.get("group") else ""
        t0 = time.time()
        crashed_here = False
        if noop:
            say(f"[{vid}] {self.args.stop_after}: {NOOP_NOTE}; handing over {len(paths)} file(s)")
            found = []
            self.notes[vid] = NOOP_NOTE
        else:
            say(f"[{vid}] start{grp}: {len(paths)} file(s), denoise={vargs.denoise} "
                f"pre_separation={vargs.denoise_pre_separation} pre_asr={vargs.denoise_pre_asr}")
            try:
                found = self.run_batch(pipeline, vargs, config, paths, logger=logger) \
                    if stages is None else \
                    self.run_batch(pipeline, vargs, config, paths, logger=logger, stages=stages)
                found = list(found or [])
            except Exception as exc:  # one variant must never stop the others
                err = f"{type(exc).__name__}: {exc}"
                found = [(p, f"[{vid}] {err}") for p in paths]
                crashed_here = True
                if not self.per_file:    # per-file: only this file is lost
                    self.crashed[vid] = err
                say(f"[{vid}] runner error: {err}")
            finally:
                reset_between_variants(pipeline, paths)
        self._discard_failed(pipeline, vargs, found, logger, crashed=crashed_here or vid in self.crashed)
        self.elapsed[vid] = self.elapsed.get(vid, 0) + time.time() - t0
        self.failures.setdefault(vid, []).extend(found)
        self.seen.setdefault(vid, set()).update(Path(p).stem for p in paths)
        if self.mark and self.per_file:
            if found:
                self._write_signal(v, False, str(found[0][1]), stem)
            else:
                self._write_signal(v, True, stem=stem)
        say(f"[{vid}] end of this {'file' if self.per_file else 'group'} after {time.time() - t0:.0f}s "
            f"({len(found)} failure(s))")
        if self.crashed.get(vid) or set(self.stems) <= self.seen[vid]:
            self._finish_variant(v)
        return [(p, e if str(e).startswith(f"[{vid}]") else f"[{vid}] {e}") for p, e in found]

    def _unit_failed(self, v, unit, reason, say):
        """The upstream process failed (or never signalled) for this unit: record it without running."""
        vid = v["id"]
        text = reason
        if not self.per_file:
            self.crashed[vid] = text
            say(f"[{vid}] {text}")
            self._finish_variant(v)
            return [(p, f"[{vid}] {text}") for p in unit]
        say(f"[{vid}] {Path(unit[0]).name}: {text}")
        self.failures.setdefault(vid, []).extend((p, f"[{vid}] {text}") for p in unit)
        self.seen.setdefault(vid, set()).update(Path(p).stem for p in unit)
        if self.mark:
            self._write_signal(v, False, text, self._stem_of(unit))
        if set(self.stems) <= self.seen[vid]:
            self._finish_variant(v)
        return [(p, f"[{vid}] {text}") for p in unit]

    def _consume(self, pipeline, args, config, group, logger, stages, say):
        """Run units as their upstream stage signals them (or, with --claim, as this copy claims them), not in plan
        order; wait while none is ready. A file another copy holds is waited for, never failed here."""
        failures, announced = [], False
        idle_since = self.clock()
        while True:
            pending = [(v, u) for v in self.runnable() if v["id"] not in self.crashed
                       for u in self._units(v, group)]
            if not pending:
                break
            ready, dead, adopted = [], [], False
            for v, unit in pending:
                stem = self._stem_of(unit)
                if self.claim:
                    own, why = self._marker_state(v, self.mark, stem)
                    if own is not None:
                        self._adopt(v, unit, own, why, say)
                        adopted = True
                        continue
                state, why = self._signal_state(v, stem) if self.wait_signal else ("ready", None)
                if state == "failed":
                    dead.append((v, unit, why))
                elif state == "ready":
                    ready.append((v, unit))
            if dead:
                for v, unit, why in dead:
                    if v["id"] not in self.finalized:
                        failures += self._unit_failed(v, unit, f"producer failed: {why}", say)
                idle_since = self.clock()
                continue
            pick = None
            for v, unit in ready:
                if not self.claim or self._take_claim(v["id"], self._stem_of(unit), say):
                    pick = (v, unit)
                    break
            if pick is not None:
                v, unit = pick
                eff = effective_stage(v, self.wait_signal, self.music)
                say(f"[{v['id']}] ready" + (f" (.ready_{eff})" if eff else " (nothing upstream)")
                    + (f" {Path(unit[0]).name}" if self.per_file else "") + "; starting")
                failures += self._run_variant(v, pipeline, args, config, unit, logger, stages)
                idle_since, announced = self.clock(), False
                continue
            if adopted:
                idle_since = self.clock()
                continue
            if self.wait_timeout and self.clock() - idle_since >= self.wait_timeout * 60:
                for v, unit in pending:
                    if v["id"] in self.finalized:
                        continue
                    if any(unit is u for _, u in ready):     # upstream ready, another copy holds it: not ours to fail
                        say(f"[{v['id']}] {Path(unit[0]).name}: still held by another copy; gave up waiting")
                        continue
                    say(f"[{v['id']}] producer never signalled (waited {self.wait_timeout:g} min)")
                    failures += self._unit_failed(v, unit, "producer never signalled", say)
                break
            if not announced:
                names = sorted({v["id"] for v, _ in pending})
                say("waiting for the upstream .ready_ marker or another copy's claim, of: " + ", ".join(names))
                announced = True
            self.sleep(self.poll_seconds)
        return failures

    def _skip_wasted_prefetch(self, pipeline, say):
        """Diarization starts building the overlap windows of separation (a 6-process pool, CPU) as soon as a file's
        diarization is ready, and the stage cannot end before that finishes. A process that stops before
        separation never uses those windows -- separation runs in another process -- so the work only burns CPU and
        delays the hand-over. Turn it off there."""
        if stage_rank(self.args.stop_after) >= stage_rank("separation"):
            return
        sep = getattr(pipeline, "separation_svc", None)
        if sep is not None and hasattr(sep, "prefetch_overlap_plan"):
            sep.prefetch_overlap_plan = lambda *a, **k: None
            say(f"[line] separation window prefetch is off in this process (--stop-after {self.args.stop_after}: "
                "separation runs in another process)")

    def scheduler(self, pipeline, args, config, group, logger=None, stages=None):
        say = logger.info if logger is not None else self.say
        if not self._prefetch_checked:
            self._prefetch_checked = True
            self._skip_wasted_prefetch(pipeline, say)
        if not self.gated:
            for v in self.runnable():
                why = gate_problem(v, config, getattr(args, "env", self.args.env))
                if why:
                    self.reasons[v["id"]], self.status[v["id"]] = why, "skipped"
                    self.say(f"[{v['id']}] SKIPPED: {why}")
            self.gated = True
        if self.wait_signal or self.claim:
            return self._consume(pipeline, args, config, group, logger, stages, say)
        failures = []
        for v in self.runnable():
            if v["id"] in self.crashed:
                continue
            for unit in self._units(v, group):
                failures += self._run_variant(v, pipeline, args, config, unit, logger, stages)
                if v["id"] in self.crashed:
                    break
        return failures


# ------------------------------------------------------------------------ plan / argv
def layout_warnings(args):
    """What --layout split cannot promise, said before the run starts."""
    out = []
    if not str(args.layout).startswith("split"):
        return out
    if int(args.gpu_1) != 0:
        out.append("denoise backends use cuda:0 (device 'auto'), which is gpu_2 here: keep --gpu-1 0 so they stay "
                   "off the ASR card")
    if args.stop_after not in STOPS_BEFORE_REFINEMENT:
        out.append("--stop-after goes past asr: main.py puts the refinement LLM on both cards, so it shares gpu_2 "
                   "with the resident ASR models")
    return out


def layout_sets(layout):
    return [f"{k}={json.dumps(v)}" for k, v in LAYOUT_PRESETS[layout].items()]


def build_argv(args, input_dir):
    """The sys.argv main.py gets (argv[0] included)."""
    argv = [str(ROOT / "main.py"),
            "--audio_dir", str(input_dir), "--save_path", str(Path(args.out_root) / RESIDENT_DIR),
            "--env", args.env, *(["--stop_after", args.stop_after] if args.stop_after != "all" else []),
            "--job_id", JOB_ID,
            "--by_stage", "--keep_models", "--no_review_page",
            "--gpu_1", str(args.gpu_1), "--gpu_2", str(args.gpu_2)]
    if args.cache_dir:
        argv += ["--cache_dir", str(args.cache_dir)]
    if args.skip_bootstrap:
        argv += ["--skip_bootstrap"]
    if getattr(args, "no_stage_audio", False):
        argv += ["--no_stage_audio"]
    # A comparison run is not a dataset export; layout, then the user's own sets (last wins).
    for item in ['outputs.clean_two_channel_dir=""', *layout_sets(args.layout), *args.global_set]:
        argv += ["--set", item]
    return argv


def is_music_off_copy(variant):
    """A variant that only turns the music steps off: with --music off it is the baseline again."""
    chains = [variant.get(k) for k in ("chain", "chain_pre_separation", "chain_pre_asr")]
    sets = variant.get("set") or {}
    return (not any(chains) and bool(sets)
            and set(sets) <= {"steps.music_analysis", "steps.music_removal"}
            and not any(sets.values()) and not variant.get("music_separator") and not variant.get("extra_args"))


def resolve_plan(args):
    r = rv()
    variants = r.filter_groups(r.load_variants(args.variants), args.groups.split(","))
    variants = r.pick_variants(variants, args.only_ids.split(","))
    files = r.collect_files(args.audio_dir, args.files)
    skipped = r.check_availability(variants) if args.only_available else {}
    if args.music == "off":
        for v in variants:
            if is_music_off_copy(v):
                skipped.setdefault(v["id"], "same as the baseline when --music off (no music step runs anywhere)")
    return variants, files, skipped


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run all noise-filter variants in one process (models resident).")
    p.add_argument("--audio-dir")
    p.add_argument("--files", nargs="*")
    p.add_argument("--variants", help="variants JSON (default: baseline only)")
    p.add_argument("--out-root", required=True)
    p.add_argument("--env", default="a100")
    p.add_argument("--stop-after", default="asr",
                   help="a STAGE, 'all' (the last process: main.py gets no stop_after, it runs to the end) or 'none'")
    p.add_argument("--cache-dir")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only-ids", default="", help="comma list of variant ids (baseline always runs)")
    p.add_argument("--groups", default="", help="comma list of variant groups (baseline always runs)")
    p.add_argument("--only-available", action="store_true",
                   help="skip (with the reason) variants whose chain has an unusable backend")
    p.add_argument("--dry-run", action="store_true", help="print the plan, argv and overrides; run nothing")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="global_set",
                   help="forwarded to main.py --set (profile override), after the layout preset")
    p.add_argument("--gpu-1", type=int, default=0, dest="gpu_1")
    p.add_argument("--gpu-2", type=int, default=1, dest="gpu_2")
    p.add_argument("--layout", choices=sorted(LAYOUT_PRESETS), default="split",
                   help="split: Qwen3 alone on gpu_2, the rest on gpu_1; default: the profile decides")
    p.add_argument("--music", choices=["off", "profile"], default="off",
                   help="off (default): no music analysis and no music removal for any variant, so every filter "
                        "sees the recording as it is; profile: whatever the profile says (a100 removes music "
                        "from the whole file with BS-RoFormer)")
    p.add_argument("--mode", choices=["sequential", "overlap"], default="sequential")
    p.add_argument("--accept-degraded", action="store_true",
                   help="write .done for a degraded variant too (so --resume keeps it)")
    p.add_argument("--strict", action="store_true", help="exit 3 when any variant is degraded or failed")
    p.add_argument("--no-stage-audio", action="store_true", dest="no_stage_audio",
                   help="forward --no_stage_audio to main.py: stage folders keep their JSON, drop their WAVs")
    p.add_argument("--skip-bootstrap", action="store_true", help="forward --skip_bootstrap to main.py")
    p.add_argument("--reference-dir")
    p.add_argument("--reference", nargs="*", metavar="STEM=PATH")
    p.add_argument("--shard", default="1/1", metavar="K/N",
                   help="this process owns plan entries with index %% N == K-1 (1-based K)")
    p.add_argument("--signal", metavar="STAGE",
                   help="producer: write OUT/<id>/.ready_STAGE when a variant is done (STAGE = --stop-after)")
    p.add_argument("--wait-signal", metavar="STAGE", dest="wait_signal",
                   help="consumer: run a variant only once its producer wrote .ready_STAGE (earlier than --stop-after)")
    p.add_argument("--poll-seconds", type=float, default=15.0, dest="poll_seconds")
    p.add_argument("--wait-timeout", type=float, default=0.0, dest="wait_timeout", metavar="MINUTES",
                   help="consumer: give up on variants nobody signalled after this idle time (0 = never)")
    p.add_argument("--per-file", action="store_true", dest="per_file",
                   help="hand over file by file: one run_batch call per (variant, file), small files first, markers "
                        ".ready_<stage>.<stem>; a long file no longer holds back the short ones")
    p.add_argument("--claim", action="store_true",
                   help="with --per-file: several copies of this stage (same out-root, other GPU or job) share the "
                        "files: a copy works on a file only after it creates .claim_<stage>.<stem> first")
    p.add_argument("--reclaim-after", type=float, default=0.0, dest="reclaim_after", metavar="MINUTES",
                   help="--claim: take over a claim older than this with no ready/failed marker (0 = never); "
                        "set it above the longest file run")
    p.add_argument("--minutes-per-file-run", type=float, default=20.0, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.mode == "overlap":
        raise SystemExit(OVERLAP_MESSAGE)
    if not args.audio_dir and not args.files:
        p.error("one of --audio-dir or --files is required")
    m = re.fullmatch(r"(\d+)/(\d+)", args.shard.strip())
    if not m or not 1 <= int(m.group(1)) <= int(m.group(2)):
        p.error(f"--shard expects K/N with 1 <= K <= N, got {args.shard!r}")
    args.shard = (int(m.group(1)), int(m.group(2)))
    if args.stop_after not in ("none", "all") and args.stop_after not in STAGE_ORDER:
        p.error(f"--stop-after {args.stop_after!r} is not a stage; known: {', '.join(STAGE_ORDER)}, none, all")
    if args.signal and args.signal != args.stop_after:
        p.error(f"--signal {args.signal!r} must equal --stop-after {args.stop_after!r}")
    if args.wait_signal and (args.wait_signal not in STAGE_ORDER
                             or stage_rank(args.wait_signal) >= stage_rank(args.stop_after)):
        p.error(f"--wait-signal {args.wait_signal!r} must be a stage earlier than --stop-after {args.stop_after!r}")
    if args.poll_seconds <= 0 or args.wait_timeout < 0:
        p.error("--poll-seconds must be > 0 and --wait-timeout >= 0")
    if args.claim and not args.per_file:
        p.error("--claim needs --per-file (copies share files one at a time)")
    if args.reclaim_after < 0:
        p.error("--reclaim-after must be >= 0")
    args.out_root = str(Path(args.out_root).resolve())
    if args.music == "off":
        args.global_set = list(MUSIC_OFF_SETS) + list(args.global_set)
        # Diarization / separation checkpoints are not keyed by the audio they were computed on, so a cache
        # written with music removal on must never be read by a run with it off.
        args.cache_dir = args.cache_dir or NO_MUSIC_CACHE_DIR
    return args


def _claim_holder(path):
    try:
        data = json.loads(path.read_text())
        return f"{data.get('host', '?')} pid {data.get('pid', '?')}"
    except (OSError, ValueError):
        return "an unknown copy"


def role_of(args):
    if args.signal and args.wait_signal:
        return "producer+consumer"
    return "producer" if args.signal else "consumer" if args.wait_signal else "all"


def print_dry_run(args, variants, files, skipped):
    rv().print_plan(variants, files, skipped, args)
    owned = plan_ownership(variants, skipped, args.shard)
    print(f"\nrole: {role_of(args)}   shard: {args.shard[0]}/{args.shard[1]}   "
          f"signal: {args.signal or '-'}   wait-signal: {args.wait_signal or '-'}"
          + (f" (poll {args.poll_seconds:g}s, timeout {args.wait_timeout:g} min)" if args.wait_signal else "")
          + f"   per-file: {'yes' if args.per_file else 'no'}"
          + ("   claim" if args.claim else "") + (f" (reclaim after {args.reclaim_after:g} min)" if args.reclaim_after else ""))
    print("this process owns: " + (", ".join(v["id"] for v in variants if v["id"] in owned) or "nothing"))
    input_dir = Path(args.out_root) / RESIDENT_DIR / "input_<timestamp>"
    print("\nmain.py argv (one process, models kept):")
    print("  " + shlex.join(build_argv(args, input_dir)))
    print("\nper-variant args overrides (applied on a copy of main's args before each pass):")
    for v in variants:
        why = skipped.get(v["id"]) or unsupported_reason(v)
        if why:
            print(f"  [{v['id']}] SKIPPED: {why}")
            continue
        if v["id"] not in owned:
            continue
        print(f"  [{v['id']}] " + json.dumps(variant_overrides(v, args.stop_after, args.out_root),
                                              ensure_ascii=False))
        idle = noop_stages(v, args.music)
        print(f"      no-op stages: {', '.join(idle) if idle else '-'}"
              + ("   (this process only hands it over)" if args.stop_after in idle else "")
              + (f"   waits for: {effective_stage(v, args.wait_signal, args.music) or 'nothing (ready at once)'}"
                 if args.wait_signal else ""))


def child_gpu_map(visible):
    """Index a child asks for -> the id this process really owns, from its own CUDA_VISIBLE_DEVICES list.

    A process that sees only card 1 calls it cuda:0, and the pipeline's worker services then start every
    worker with CUDA_VISIBLE_DEVICES=0 -- the job's card 0, not this process's card. CUDA's own rule is that a
    child's index is relative to the parent's visible list; this restores it for the children we spawn.
    """
    ids = [x.strip() for x in str(visible or "").split(",") if x.strip()]
    return ids


def install_child_gpu_mapping(visible=None):
    """Make subprocess.Popen map a digit CUDA_VISIBLE_DEVICES in `env` through this process's visible list.

    Returns a function that undoes it. Does nothing when this process has no CUDA_VISIBLE_DEVICES of its own
    (it then sees every card and the pipeline's indexes already mean what they say).
    """
    import subprocess
    ids = child_gpu_map(os.environ.get("CUDA_VISIBLE_DEVICES") if visible is None else visible)
    if not ids or not all(x.isdigit() for x in ids):
        return lambda: None
    real = subprocess.Popen

    class MappedPopen(real):
        def __init__(self, *a, **kw):
            env = kw.get("env")
            want = None if env is None else str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()
            if want and want.isdigit() and int(want) < len(ids):
                env = dict(env)
                env["CUDA_VISIBLE_DEVICES"] = ids[int(want)]
                kw["env"] = env
            super().__init__(*a, **kw)

    subprocess.Popen = MappedPopen
    print(f"[gpu] this process owns CUDA_VISIBLE_DEVICES={','.join(ids)}: a worker that asks for cuda index i "
          f"runs on {', '.join(f'{i}->{d}' for i, d in enumerate(ids))}", flush=True)

    def undo():
        subprocess.Popen = real
    return undo


def run_sequential(args, variants, files, skipped):
    """Variants one after another inside the resident process; each is a stage-by-stage pass."""
    runner = ResidentRunner(args, variants, files, skipped=skipped)
    runner.prepare()
    todo = runner.runnable()
    if not todo:
        print("nothing to run: every variant is skipped or up to date", flush=True)
        return runner
    out = Path(args.out_root)
    input_dir = out / RESIDENT_DIR / f"input_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    rv().link_inputs(input_dir, files)
    os.chdir(ROOT)                         # main.py resolves config.json against the cwd
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.argv = build_argv(args, input_dir)
    import main as main_mod                # module level only parses args and loads the profile
    from utils.batch import run_batch_by_stage as original
    runner.run_batch = original
    runner.discard_partial = main_mod._discard_partial
    for line in layout_warnings(args):
        print(f"WARNING (--layout {args.layout}): {line}", flush=True)
    main_mod.run_batch_by_stage = runner.scheduler
    undo_gpu_map = install_child_gpu_mapping()
    try:
        main_mod.main()
    finally:
        undo_gpu_map()
        main_mod.run_batch_by_stage = original
        runner.finish()
    return runner


MODES = {"sequential": run_sequential}


def main(argv=None):
    args = parse_args(argv)
    variants, files, skipped = resolve_plan(args)
    if args.dry_run:
        print_dry_run(args, variants, files, skipped)
        return 0
    for vid, why in skipped.items():
        print(f"[{vid}] SKIPPED (not available here): {why}", flush=True)
    runner = MODES[args.mode](args, variants, files, skipped)
    status = runner.status
    print(json.dumps(status))
    rv().print_degraded_summary(runner.reports, status)
    print(f"manifest: {Path(args.out_root) / rv().MANIFEST_NAME}")
    if args.strict and any(s in ("degraded", "failed") for s in status.values()):
        return 3
    return 0 if all(s in ("ok", "skipped", "degraded") for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
