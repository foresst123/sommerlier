"""Grouping audio files into runs bounded by total duration."""

import os

from utils.steps import step_enabled


def audio_duration(path: str) -> float:
    """Length of `path` in seconds, or 0.0 if it cannot be read.

    Reads the container header rather than decoding, so a five-hour batch is
    planned in milliseconds.
    """
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.frames and info.samplerate:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        # soundfile cannot read mp3 on every build; mutagen reads the header.
        from mutagen import File as MutagenFile
        m = MutagenFile(path)
        if m is not None and getattr(m, "info", None) is not None:
            return float(m.info.length)
    except Exception:
        pass
    try:
        from pydub.utils import mediainfo
        return float(mediainfo(path).get("duration") or 0.0)
    except Exception:
        return 0.0


def find_audio_files(directory: str, extensions) -> list:
    """Audio files directly under `directory`, sorted by name."""
    exts = tuple(e.lower() for e in extensions)
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(
        os.path.join(directory, n) for n in names
        if n.lower().endswith(exts) and os.path.isfile(os.path.join(directory, n))
    )


def find_name_collisions(paths) -> dict:
    """Files that share a basename, keyed by that name.

    Output directories are named after the audio's stem, so `talk.mp3` and
    `talk.wav` in the same folder resolve to one directory and the second run
    overwrites the first. Detected up front because the loss is silent: the
    pipeline reports success for both.
    """
    import collections
    by_stem = collections.defaultdict(list)
    for p in paths:
        by_stem[os.path.splitext(os.path.basename(p))[0]].append(p)
    return {stem: group for stem, group in by_stem.items() if len(group) > 1}


def plan_batches(paths, max_hours: float, logger=None):
    """Split `paths` into runs whose total duration stays under `max_hours`.

    Files keep their given order, so a run is a contiguous slice and the output
    layout does not depend on how the batches fall. A file longer than the limit
    becomes a batch of its own rather than being skipped -- refusing to process
    it would be a worse answer than exceeding the bound once, deliberately.

    Files whose duration cannot be read are placed in their own batch too: an
    unknown length cannot be added to a budget honestly.
    """
    limit = max(0.0, float(max_hours)) * 3600.0
    batches, current, current_total = [], [], 0.0

    for path in paths:
        dur = audio_duration(path)

        if dur <= 0.0:
            if logger:
                logger.warning(
                    f"Could not read duration of {os.path.basename(path)}; "
                    "running it on its own")
            if current:
                batches.append(current)
                current, current_total = [], 0.0
            batches.append([path])
            continue

        if limit > 0 and dur > limit:
            if logger:
                logger.warning(
                    f"{os.path.basename(path)} is {dur / 3600:.2f}h, longer than the "
                    f"{max_hours}h limit; running it on its own")
            if current:
                batches.append(current)
                current, current_total = [], 0.0
            batches.append([path])
            continue

        if limit > 0 and current and current_total + dur > limit:
            batches.append(current)
            current, current_total = [], 0.0

        current.append(path)
        current_total += dur

    if current:
        batches.append(current)
    return batches


def split_final_failures(failures, is_done):
    """Separate files that stayed failed from those a later pass recovered.

    `failures` accumulates across passes, so a file that failed in pass 1 and
    finished in pass 2 is still in it. Reporting that list as-is printed
    "1/2 file(s) failed" right under "Corpus complete: 2 done". A file that
    failed twice appears twice; only its last error is kept.
    """
    last = {}
    for path, err in failures:
        last[path] = err
    still = [(p, e) for p, e in last.items() if not is_done(p)]
    recovered = [(p, e) for p, e in last.items() if is_done(p)]
    return still, recovered


# Stage order must match the pipeline's own sequence; `None` means "run to the
# end", which covers refinement and export.
#
# "music" leads because it is a stage like any other: it loads PANNs and a
# vocal separator, and running it as its own pass loads them once for the whole
# batch instead of once inside each file's diarization pass. It is also the
# stage a run keeps when everything after it is switched off, so it has to be
# reachable on its own.
PIPELINE_STAGES = ("music", "diarization", "separation", "music_removal", "asr", "captioning", None)


def label_of(stage):
    return stage or "refinement+export"


def run_batch_by_stage(pipeline, args, config, batch, logger=None, stages=PIPELINE_STAGES):
    """Run one stage across every file before moving to the next stage.

    File-major order reloads each model once per file. Stage-major loads the
    diarizer once, runs all five files, frees it, then loads the separator. The
    pipeline already checkpoints per stage and per file, so each pass resumes
    from the previous pass's output rather than recomputing.

    Returns a list of (path, error) for files that failed.
    """
    import copy
    import os

    failures = {}
    original_stop = getattr(args, "stop_after", None)

    for stage in stages:
        if stage is not None and original_stop is not None:
            # The caller asked to stop early; do not run past their request.
            if _stage_index(stage) > _stage_index(original_stop):
                break

        label = label_of(stage)

        # Skip this pass entirely when the stage is switched off in the config.
        # The None stage covers refinement+export; skip it only if both are off.
        _stage_step_map = {
            "music": "music_analysis",
            "diarization": "diarization",
            "separation": "separation",
            "music_removal": "music_removal_fallback",
            "asr": "asr",
            "captioning": "captioning",
        }
        if stage is not None:
            step_name = _stage_step_map.get(stage, stage)
            if not step_enabled(args, step_name):
                if logger:
                    logger.info(f"Stage '{label}' is off in the profile; skipping batch pass")
                continue
        elif stage is None:
            if not step_enabled(args, "refinement") and not step_enabled(args, "export"):
                if logger:
                    logger.info("Both refinement and export are off; skipping batch pass")
                continue
        pending = [p for p in batch if p not in failures]
        if not pending:
            break
        if logger:
            logger.info(f"=== Stage '{label}': {len(pending)} file(s) ===")
        monitor = getattr(pipeline, "performance_monitor", None)
        stage_started = __import__("time").time()
        if monitor:
            monitor.record("stage_started", stage=label, files=len(pending))

        stage_args = copy.copy(args)
        stage_args.stop_after = stage

        # Hold model releases until every file has passed through this stage.
        # Without this the first file frees the diarizer that the second file
        # is about to use, which turns stage-major back into file-major with
        # extra steps.
        begin = getattr(pipeline, "begin_stage_scope", None)
        end = getattr(pipeline, "end_stage_scope", None)
        if begin:
            begin()

        # finally, not a plain call after the loop: an open scope swallows every
        # later release, so a stage that dies outside the per-file try would
        # leave the models it finished with resident for the rest of the run.
        try:
            parallelism = _stage_parallelism(stage_args, stage)

            def run_one(i, path):
                if logger:
                    logger.info(
                        f"[{label} {i}/{len(pending)}] {os.path.basename(path)}")
                stage_pipeline = (
                    pipeline.parallel_stage_view(stage)
                    if parallelism > 1 and hasattr(pipeline, "parallel_stage_view")
                    else pipeline)
                stage_pipeline.run(copy.copy(stage_args), config, path)

            if parallelism > 1 and len(pending) > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(
                        max_workers=min(parallelism, len(pending)),
                        thread_name_prefix=f"stage-{stage}") as executor:
                    submitted = [
                        (path, executor.submit(run_one, i, path))
                        for i, path in enumerate(pending, start=1)
                    ]
                    for path, future in submitted:
                        try:
                            future.result()
                        except Exception as e:
                            if logger:
                                logger.error(
                                    f"Failed on {path} during {label}: "
                                    f"{type(e).__name__}: {e}")
                            failures[path] = (
                                f"{label}: {type(e).__name__}: {e}")
            else:
                for i, path in enumerate(pending, start=1):
                    try:
                        run_one(i, path)
                    except Exception as e:
                        # A failed file is omitted from every later stage but
                        # does not stop its neighbours.
                        if logger:
                            logger.error(
                                f"Failed on {path} during {label}: "
                                f"{type(e).__name__}: {e}")
                        failures[path] = f"{label}: {type(e).__name__}: {e}"
        finally:
            if end:
                end()
            if monitor:
                monitor.stage_finished(
                    label, __import__("time").time() - stage_started,
                    len(pending), sum(1 for path in pending if path in failures))

        if stage is not None and original_stop == stage:
            break

    return list(failures.items())


def _stage_parallelism(args, stage) -> int:
    """Concurrent files allowed for stages with independent GPU workers."""
    perf = getattr(args, "performance_config", None) or {}
    if not perf.get("enabled", False):
        return 1
    stages = perf.get("stages", {})
    if stage == "diarization" and not getattr(args, "dia3", False):
        return max(1, min(2, int(
            stages.get("diarization", {}).get("workers", 1))))
    if stage == "separation":
        separator = (getattr(args, "separator", None)
                     or os.environ.get("BSS_SEPARATOR", "sidon"))
        if str(separator).strip().lower() == "sidon":
            return max(1, min(2, int(
                stages.get("separation", {}).get("max_workers", 1))))
    if stage == "music" and stages.get("music", {}).get("cross_file_overlap", False):
        # SSLAM and BS-RoFormer are two model TYPES on two GPUs, not multiple
        # instances of one -- a third file gains nothing further, since the
        # first two already keep both cards busy (file N removing music while
        # file N+1 is classified).
        return 2
    return 1


def _stage_index(stage) -> int:
    try:
        return PIPELINE_STAGES.index(stage)
    except ValueError:
        return len(PIPELINE_STAGES)
