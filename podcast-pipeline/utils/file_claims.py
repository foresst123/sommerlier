"""Per-file claims so several pipelines can share one audio directory.

Each claim is a small JSON file in `<directory>/_sommelier_claims/`, created with
O_CREAT|O_EXCL so exactly one process wins. A claim is stale when its owner (same
host) is dead, or when nobody has touched it for `ttl` seconds; a live owner keeps
its claims fresh from a heartbeat thread. A stale claim is taken over by renaming
it to a unique name first: rename succeeds for one process only, so concurrent
takeovers cannot delete a claim someone else just created.

Limits: O_EXCL is unreliable on very old NFS; there the heartbeat/ttl is what
recovers a wrongly shared or abandoned claim. Across hosts only the ttl applies.
"""

import atexit
import contextlib
import hashlib
import json
import os
import re
import socket
import threading
import time
import uuid

CLAIMS_DIRNAME = "_sommelier_claims"
DEFAULT_TTL = 600.0
TTL_ENV = "SOMMELIER_CLAIM_TTL"


def _resolve_ttl(ttl):
    if ttl is not None:
        return float(ttl)
    try:
        return float(os.environ.get(TTL_ENV, DEFAULT_TTL))
    except ValueError:
        return DEFAULT_TTL


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return True  # unknown: assume alive, never steal a live claim
    return True


class FileClaims:
    def __init__(self, directory: str, logger=None, enabled: bool = True,
                 ttl=None, job_id=None):
        self.directory = directory
        self.logger = logger
        self.enabled = enabled
        self.ttl = _resolve_ttl(ttl)
        self.job_id = job_id
        self.token = uuid.uuid4().hex
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.claims_dir = os.path.join(directory, CLAIMS_DIRNAME)
        self._held = {}  # audio path -> claim file path
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._atexit = False

    # -- naming --------------------------------------------------------
    def _claim_path(self, path: str) -> str:
        try:
            rel = os.path.relpath(path, self.directory)
        except ValueError:
            rel = os.path.basename(path)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", rel)[:80]
        digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
        return os.path.join(self.claims_dir, f"{safe}.{digest}.claim")

    def _read(self, claim_path):
        try:
            with open(claim_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return None
        except Exception:
            return {}  # unreadable: judged by mtime only

    def _is_stale(self, claim_path) -> bool:
        try:
            age = time.time() - os.stat(claim_path).st_mtime
        except FileNotFoundError:
            return False
        if age > self.ttl:
            return True
        data = self._read(claim_path) or {}
        if data.get("host") == self.host and data.get("pid") is not None:
            return not _pid_alive(data["pid"])
        return False

    # -- claiming ------------------------------------------------------
    def _create(self, claim_path, path) -> bool:
        body = json.dumps({
            "path": path, "pid": self.pid, "host": self.host,
            "token": self.token, "job_id": self.job_id,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }).encode("utf-8")
        try:
            fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        try:
            os.write(fd, body)
        finally:
            os.close(fd)
        return True

    def _won(self, path, claim_path) -> bool:
        with self._lock:
            self._held[path] = claim_path
        return True

    @contextlib.contextmanager
    def _dir_lock(self):
        fd = None
        try:
            import fcntl
            fd = os.open(os.path.join(self.claims_dir, ".takeover.lock"),
                         os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass  # no flock: rename alone arbitrates
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)

    def try_claim(self, path: str) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            if path in self._held:
                return True
        claim_path = self._claim_path(path)
        try:
            os.makedirs(self.claims_dir, exist_ok=True)
            if self._create(claim_path, path):
                return self._won(path, claim_path)
            if not self._is_stale(claim_path):
                return False
            # Takeover: serialized per host by a directory lock, so a late loser
            # cannot rename away the fresh claim the winner just made.
            with self._dir_lock():
                if self._create(claim_path, path):
                    return self._won(path, claim_path)
                if not self._is_stale(claim_path):
                    return False
                grave = f"{claim_path}.stale-{uuid.uuid4().hex}"
                try:
                    os.rename(claim_path, grave)  # one process wins this
                except FileNotFoundError:
                    return False
                try:
                    os.remove(grave)
                except OSError:
                    pass
                if self.logger:
                    self.logger.info(
                        f"Took over a stale claim on {os.path.basename(path)}")
                if self._create(claim_path, path):
                    return self._won(path, claim_path)
            return False
        except OSError as e:
            if self.logger:
                self.logger.warning(f"Could not claim {path}: {e}")
            return False

    def release(self, path: str):
        if not self.enabled:
            return
        with self._lock:
            claim_path = self._held.pop(path, None)
        if claim_path is None:
            return
        data = self._read(claim_path)
        if data is not None and data.get("token") == self.token:
            try:
                os.remove(claim_path)
            except OSError:
                pass

    def release_all(self):
        if not self.enabled:
            return
        with self._lock:
            paths = list(self._held)
        for p in paths:
            self.release(p)

    # -- queries -------------------------------------------------------
    def held(self) -> set:
        with self._lock:
            return set(self._held)

    def is_claimed_by_other(self, path: str) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if path in self._held:
                return False
        claim_path = self._claim_path(path)
        data = self._read(claim_path)
        if data is None:
            return False
        if data.get("token") == self.token:
            return False
        return not self._is_stale(claim_path)

    def claimed_by_others(self, paths) -> set:
        return {p for p in paths if self.is_claimed_by_other(p)}

    # -- heartbeat -----------------------------------------------------
    def _beat(self):
        with self._lock:
            items = list(self._held.items())
        for _, claim_path in items:
            try:
                os.utime(claim_path, None)
            except OSError:
                pass

    def _run(self):
        interval = max(self.ttl / 10.0, 0.05)
        while not self._stop.wait(interval):
            self._beat()

    def start(self):
        if not self.enabled or self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="file-claims-heartbeat")
        self._thread.start()
        if not self._atexit:
            atexit.register(self.release_all)
            self._atexit = True
        return self

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        self.release_all()
        return False


def claim_group(ledger, claims, run_snapshot, attempted, max_hours, logger=None,
                planner=None, max_replans: int = 3):
    """Pick the next group of files and claim exactly those.

    Returns (group, others_hold_count). `group` is empty when nothing is left to
    run; then `others_hold_count` says how many pending files other pipelines
    hold (0 means the work is simply finished). Only the group about to run is
    claimed, so the other pipelines keep the rest.

    When every remaining file is held by another pipeline this returns an empty
    group and the caller stops; if that pipeline dies, run again to pick up what
    is left (its stale claims are then taken over).
    """
    if planner is None:
        from utils.batch import plan_batches as planner
    ledger.refresh()
    todo = ledger.pending(run_snapshot, exclude=attempted)
    others = claims.claimed_by_others(todo)
    candidates = [p for p in todo if p not in others]
    lost, finished = set(), set()
    for _ in range(max_replans + 1):
        if not candidates:
            break
        group = planner(candidates, max_hours, logger=logger)[0]
        got = []
        for p in group:
            if claims.try_claim(p):
                got.append(p)
            else:
                lost.add(p)
        if got:
            # Another pipeline may have finished and released these between our
            # refresh and our claim. It saves before releasing, so re-reading now
            # shows it; hand back whatever is no longer pending.
            ledger.refresh()
            keep = set(ledger.pending(got, exclude=attempted))
            for p in got:
                if p not in keep:
                    claims.release(p)
                    finished.add(p)
            got = [p for p in got if p in keep]
            if got:
                return got, len(others | lost)
        candidates = [p for p in candidates if p not in lost and p not in finished]
    return [], len(others | lost)
