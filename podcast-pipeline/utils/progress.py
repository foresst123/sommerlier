"""A ledger of which audio files a corpus run has finished.

Three ways to track progress were possible: delete each file when it is done,
move it to a `done/` directory, or keep a list. A list is the one that does not
touch the input. Deleting is unrecoverable if a later stage turns out to be
wrong; moving breaks any path recorded in an earlier run's output and races
with whatever is copying new files in. A ledger beside the corpus leaves the
audio exactly where it was and can be deleted to force a full re-run.

Files can be added to the input directory while a run is in progress: the
directory is re-scanned between passes, so anything that appeared since the
last scan simply joins the next one.
"""

import json
import os
import tempfile
import time

LEDGER_NAME = "_sommelier_progress.json"


class ProgressLedger:
    """Records completed and failed files for one input directory."""

    def __init__(self, directory: str, name: str = LEDGER_NAME, logger=None):
        self.path = os.path.join(directory, name)
        self.logger = logger
        self.done = {}       # relative path -> {finished, seconds}
        self.failed = {}     # relative path -> {error, attempts, last_tried}
        self.directory = directory
        self._load()

    # -- persistence ---------------------------------------------------
    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.done = data.get("done", {}) or {}
            self.failed = data.get("failed", {}) or {}
        except Exception as e:
            # A corrupt ledger must not stop the run: the worst case is
            # reprocessing files, which is wasteful but not wrong.
            if self.logger:
                self.logger.warning(
                    f"Could not read {self.path} ({e}); starting a fresh ledger")

    def save(self):
        """Write atomically: a run killed mid-write must not lose the ledger."""
        payload = {
            "directory": self.directory,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "done": self.done,
            "failed": self.failed,
        }
        d = os.path.dirname(self.path) or "."
        try:
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, prefix=".progress-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not write progress ledger: {e}")

    # -- queries -------------------------------------------------------
    def _key(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.directory)
        except ValueError:
            return os.path.basename(path)

    def is_done(self, path: str) -> bool:
        return self._key(path) in self.done

    def attempts(self, path: str) -> int:
        return int(self.failed.get(self._key(path), {}).get("attempts", 0))

    def pending(self, paths, max_attempts: int = 2):
        """Files still to process, skipping finished ones and repeat failures.

        A file that has already failed `max_attempts` times is left out: it is
        almost always a broken input, and retrying it every pass would stall a
        corpus run on the same file forever.
        """
        out = []
        for p in paths:
            if self.is_done(p):
                continue
            if max_attempts and self.attempts(p) >= max_attempts:
                continue
            out.append(p)
        return out

    # -- updates -------------------------------------------------------
    def mark_done(self, path: str, seconds: float = None):
        key = self._key(path)
        entry = {"finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if seconds is not None:
            entry["seconds"] = round(float(seconds), 1)
        self.done[key] = entry
        self.failed.pop(key, None)

    def mark_failed(self, path: str, error: str):
        key = self._key(path)
        prior = self.failed.get(key, {})
        self.failed[key] = {
            "error": str(error)[:500],
            "attempts": int(prior.get("attempts", 0)) + 1,
            "last_tried": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def discard_partial_output(self, *paths, logger=None):
        """Delete the half-finished artifacts of a failed file.

        A file that failed leaves a checkpoint and a stage directory behind. The
        checkpoint is the dangerous half: the next run would load the stages
        that did complete and skip straight past them, so a file that failed in
        refinement would be retried with the same broken state and fail the same
        way. Clearing both means a retry starts clean.

        Missing paths are fine -- the point is to end up with nothing there.
        """
        import shutil

        removed = []
        for path in paths:
            if not path or not os.path.exists(path):
                continue
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed.append(path)
            except Exception as e:
                if logger:
                    logger.warning(f"Could not remove {path}: {e}")
        if removed and logger:
            logger.info(
                f"Cleared {len(removed)} partial artifact(s) so the retry starts clean")
        return removed

    # -- reporting -----------------------------------------------------
    def summary(self, total_seen: int = None) -> str:
        parts = [f"{len(self.done)} done"]
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if total_seen is not None:
            remaining = total_seen - len(self.done) - len(self.failed)
            if remaining > 0:
                parts.append(f"{remaining} remaining")
        return ", ".join(parts)
