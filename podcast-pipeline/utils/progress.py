"""A ledger of which audio files a corpus run has finished.

Three ways to track progress were possible: delete each file when it is done,
move it to a `done/` directory, or keep a list. A list is the one that does not
touch the input. Deleting is unrecoverable if a later stage turns out to be
wrong; moving breaks any path recorded in an earlier run's output and races
with whatever is copying new files in. A ledger beside the corpus leaves the
audio exactly where it was and can be deleted to force a full re-run.

The caller freezes the input snapshot at startup. Files added while a run is
in progress are therefore picked up on the next invocation, keeping the
duration budget for the current run stable.
"""

import json
import os
import re
import tempfile
import threading
import time

LEDGER_NAME = "_sommelier_progress.json"

# Class name of services.dnsmos_gate.AudioQualityRejected. Kept as text because a
# failure reaches the ledger as the string "<stage>: <ExcType>: <message>"; a test
# pins this to the real class so a rename cannot silently turn rejections back
# into retried failures.
REJECTION_TYPE = "AudioQualityRejected"
_REJECTION = re.compile(rf"(?:^|[\s:]){REJECTION_TYPE}:")


def is_input_rejection(error) -> bool:
    """True when `error` says the input audio itself was judged too poor to use."""
    return bool(_REJECTION.search(str(error or "")))


class ProgressLedger:
    """Records completed, failed and rejected files for one input directory.

    A failed file may succeed on a retry (an OOM, a worker that did not start). A
    rejected file was judged unusable by the input quality gate; running it again
    can only give the same verdict, so it is never offered again.
    """

    def __init__(self, directory: str, name: str = LEDGER_NAME, logger=None,
                 persist: bool = True):
        self.path = os.path.join(directory, name)
        # False keeps the ledger in memory only: nothing is read or written, so a
        # one-off run of a single file neither skips itself nor leaves a record.
        self.persist = persist
        self.logger = logger
        self.done = {}       # relative path -> {finished, seconds}
        self.failed = {}     # relative path -> {error, attempts, last_tried}
        self.rejected = {}   # relative path -> {reason, rejected_at}
        # relative path -> the input-quality (DNSMOS) report, kept here rather than only
        # in the cache so clearing the cache does not force every file to be scored again.
        self.quality = {}
        self._lock = threading.Lock()
        self.directory = directory
        self._load()

    # -- persistence ---------------------------------------------------
    def _load(self):
        if not self.persist or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.done = data.get("done", {}) or {}
            self.failed = data.get("failed", {}) or {}
            self.rejected = data.get("rejected", {}) or {}
            self.quality = data.get("quality", {}) or {}
        except Exception as e:
            # A corrupt ledger must not stop the run: the worst case is
            # reprocessing files, which is wasteful but not wrong.
            if self.logger:
                self.logger.warning(
                    f"Could not read {self.path} ({e}); starting a fresh ledger")

    @staticmethod
    def _read_disk(path):
        """The ledger on disk as four dicts, or None when missing or unreadable."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return tuple((data.get(k) or {}) for k in
                     ("done", "failed", "rejected", "quality"))

    def _merge(self, disk):
        """Fold another process's view (`disk`) into memory. Memory wins on equal keys.

        done wins over failed/rejected: a file one process finished is finished,
        whatever another process saw before. Attempt counts only grow.
        """
        d_done, d_failed, d_rejected, d_quality = disk
        done = {**d_done, **self.done}
        failed = {**d_failed, **self.failed}
        for key, entry in d_failed.items():
            mine = self.failed.get(key)
            if mine is not None:
                try:
                    top = max(int(entry.get("attempts", 0)),
                              int(mine.get("attempts", 0)))
                    failed[key] = {**mine, "attempts": top}
                except (TypeError, ValueError, AttributeError):
                    pass
        rejected = {**d_rejected, **self.rejected}
        for key in done:
            failed.pop(key, None)
            rejected.pop(key, None)
        self.done, self.failed, self.rejected = done, failed, rejected
        self.quality = {**d_quality, **self.quality}

    def refresh(self):
        """Merge the on-disk ledger into memory without writing.

        Lets `pending()` see what other pipelines sharing this directory finished.
        """
        if not self.persist:
            return
        with self._lock:
            disk = self._read_disk(self.path)
            if disk is not None:
                self._merge(disk)

    def save(self):
        """Merge with the disk, then write atomically.

        Several pipelines may share one input directory, so overwriting the file
        from memory would drop what the others recorded. The read-merge-write runs
        under an flock on a sibling lock file. A run killed mid-write must not lose
        the ledger, hence the atomic replace.
        """
        if not self.persist:
            return
        d = os.path.dirname(self.path) or "."
        lock_fd = None
        try:
            os.makedirs(d, exist_ok=True)
            try:
                import fcntl
                lock_fd = os.open(os.path.join(d, ".progress.lock"),
                                  os.O_CREAT | os.O_RDWR, 0o644)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except (ImportError, OSError):
                lock_fd = None  # no flock (Windows): best-effort merge
            disk = self._read_disk(self.path)
            if disk is not None:
                self._merge(disk)
            payload = {
                "directory": self.directory,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "done": self.done,
                "failed": self.failed,
                "rejected": self.rejected,
                "quality": self.quality,
            }
            fd, tmp = tempfile.mkstemp(dir=d, prefix=".progress-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not write progress ledger: {e}")
        finally:
            if lock_fd is not None:
                try:
                    os.close(lock_fd)  # closing releases the flock
                except OSError:
                    pass

    # -- queries -------------------------------------------------------
    def _key(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.directory)
        except ValueError:
            return os.path.basename(path)

    def is_done(self, path: str) -> bool:
        return self._key(path) in self.done

    def is_rejected(self, path: str) -> bool:
        return self._key(path) in self.rejected

    def get_quality(self, path: str):
        return self.quality.get(self._key(path))

    def set_quality(self, path: str, report: dict):
        """Remember a file's input-quality report and write the ledger now.

        Written at once, not at the end of the file: scoring is the slow part and the
        run may stop before the file finishes.
        """
        with self._lock:
            self.quality[self._key(path)] = dict(report)
            self.save()

    def attempts(self, path: str) -> int:
        return int(self.failed.get(self._key(path), {}).get("attempts", 0))

    def pending(self, paths, max_attempts: int = 2, exclude=()):
        """Files still to process, skipping finished ones and repeat failures.

        A file that has already failed `max_attempts` times is left out: it is
        almost always a broken input, and retrying it every pass would stall a
        corpus run on the same file forever. A rejected file is always left out,
        whatever its attempt count. `exclude` holds paths already handled by this
        invocation: a run that stops early (--stop_after) never marks a file done,
        so without it the same group would be offered forever.
        """
        skip = set(exclude)
        out = []
        for p in paths:
            if p in skip or self.is_done(p) or self.is_rejected(p):
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
        self.rejected.pop(key, None)

    def mark_failed(self, path: str, error: str):
        key = self._key(path)
        prior = self.failed.get(key, {})
        self.failed[key] = {
            "error": str(error)[:500],
            "attempts": int(prior.get("attempts", 0)) + 1,
            "last_tried": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def mark_rejected(self, path: str, reason: str):
        key = self._key(path)
        self.rejected[key] = {
            "reason": str(reason)[:500],
            "rejected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.failed.pop(key, None)

    def record_failure(self, path: str, error: str) -> str:
        """Record why `path` did not finish; returns "rejected" or "failed".

        An input-quality rejection is final, anything else may be retried.
        """
        if is_input_rejection(error):
            self.mark_rejected(path, error)
            return "rejected"
        self.mark_failed(path, error)
        return "failed"

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
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejected")
        if total_seen is not None:
            remaining = (total_seen - len(self.done) - len(self.failed)
                         - len(self.rejected))
            if remaining > 0:
                parts.append(f"{remaining} remaining")
        return ", ".join(parts)
