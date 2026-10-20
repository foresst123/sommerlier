"""Start a folder over: forget what earlier runs did to its files, then run them again.

Redoing a run used to mean deleting by hand three things per file: the checkpoint
cache, the folder under _final, and the file's line in the progress ledger. With
a few hundred files that is slow and easy to get wrong in one place (a ledger entry
left behind means the file is skipped; a cache left behind means the old result is
loaded instead of recomputed).

With `pipeline.rerun` on in the profile (or --rerun on the command line) main.py calls
clear_for_rerun() once at startup, before the first pass. It only touches files of
this run, never the audio, and leaves a file alone when it cannot be sure the delete
is safe or complete.
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor


def rerun_enabled(args) -> bool:
    """Off unless the profile or the command line turns it on (the command line wins)."""
    return bool(getattr(args, "rerun", None))


def rerun_targets(audio_path, cache_dir, job_id, output_dir):
    """The two per-file folders a rerun deletes: the checkpoint cache and the output."""
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    return [os.path.join(cache_dir, f"{job_id}_{stem}"), output_dir]


def _contains(parent, child):
    try:
        return os.path.commonpath([parent, child]) == parent
    except ValueError:                      # different drives
        return False


def unsafe_reason(target, audio_path):
    """Why `target` must not be deleted, or None.

    A custom --save_path can make the output folder the audio's own folder (audio at
    /data/ep/ep.wav with save_path /data resolves to /data/ep), and deleting that
    would delete the recording. The check is on both the path as given and the
    resolved one, because the audio may be a symlink.
    """
    if not target:
        return "empty path"
    real = os.path.realpath(target)
    if os.path.dirname(real) == real:
        return "filesystem root"
    for audio in (os.path.abspath(audio_path), os.path.realpath(audio_path)):
        if _contains(real, audio) or _contains(os.path.abspath(target), audio):
            return "holds the audio file"
    return None


def _remove(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def clear_for_rerun(paths, cache_dir, job_id, output_dir_of, ledger,
                    claims=None, logger=None, workers=8):
    """Delete cache and output of each file in `paths`, then forget them in the ledger.

    Returns {"cleared": [...], "skipped": [(path, reason), ...]}. A file is forgotten
    only when everything it owns was removed, so a failed delete leaves it exactly as
    it was rather than half reset. Files another pipeline holds a claim on are skipped.
    """
    held = set()
    if claims is not None:
        try:
            held = set(claims.claimed_by_others(paths))
        except Exception:
            held = set()

    plan, skipped = [], []
    for path in paths:
        if path in held:
            skipped.append((path, "being processed by another pipeline"))
            continue
        try:
            targets = rerun_targets(path, cache_dir, job_id, output_dir_of(path))
        except Exception as exc:
            skipped.append((path, f"could not work out its output folder: {exc}"))
            continue
        bad = next((r for r in (unsafe_reason(t, path) for t in targets) if r), None)
        if bad:
            skipped.append((path, f"refused to delete: {bad}"))
            continue
        plan.append((path, [t for t in targets if os.path.lexists(t)]))

    def work(item):
        path, targets = item
        try:
            for target in targets:
                _remove(target)
            return path, None
        except Exception as exc:
            return path, f"{type(exc).__name__}: {exc}"

    cleared = []
    if plan:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(plan)))) as pool:
            for path, error in pool.map(work, plan):
                if error:
                    skipped.append((path, f"delete failed: {error}"))
                else:
                    cleared.append(path)

    forgotten = ledger.forget(cleared) if cleared else 0

    if logger:
        logger.info(f"Rerun: cleared the cache and output of {len(cleared)} file(s) "
                    f"and dropped {forgotten} ledger entr{'y' if forgotten == 1 else 'ies'}")
        for path, why in skipped:
            logger.warning(f"Rerun: left {os.path.basename(path)} alone ({why})")
    return {"cleared": cleared, "skipped": skipped}
