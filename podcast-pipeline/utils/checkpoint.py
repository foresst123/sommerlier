import os
import pickle
import json
import hashlib
import tempfile
import time
from typing import Any, Optional


class _DigestWriter:
    """Binary writer that computes the checkpoint digest as bytes are written."""

    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()

    def write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.digest.update(data)
        return self.handle.write(data)

    def flush(self):
        return self.handle.flush()

    def fileno(self):
        return self.handle.fileno()


def _json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

def _verify_digest_on_load() -> bool:
    """Full sha256 of every checkpoint on every load is off: a stage-major batch
    re-enters run() once per stage per file, so it re-read and re-hashed each
    checkpoint many times. Size + mtime still catch a truncated or edited file, and
    writes are atomic. Set CHECKPOINT_VERIFY_DIGEST=1 to hash on load again."""
    return os.environ.get("CHECKPOINT_VERIFY_DIGEST", "") == "1"


class CheckpointManager:
    def __init__(self, cache_dir: str, job_id: str):
        self.job_dir = os.path.join(cache_dir, job_id)
        self.namespaces = {}
        os.makedirs(self.job_dir, exist_ok=True)
    
    def _get_stage_path(self, stage: str, fmt: str) -> str:
        stage_dir = os.path.join(self.job_dir, stage)
        # Phiên bản thuật toán có thư mục riêng; giữ nguyên checkpoint cũ
        # nhưng không dùng nhầm đầu ra của chính sách dựng cửa sổ trước đó.
        if stage in self.namespaces:
            stage_dir = os.path.join(stage_dir, self.namespaces[stage])
        os.makedirs(stage_dir, exist_ok=True)
        return os.path.join(stage_dir, f"result.{fmt}")

    def save(self, stage: str, data: Any, fmt: str = "pkl"):
        """Lưu atomically; file đích chỉ xuất hiện sau khi serialize thành công."""
        path = self._get_stage_path(stage, fmt)
        directory = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(prefix=".result-", suffix=f".{fmt}.tmp",
                                        dir=directory)
        try:
            if fmt not in ("pkl", "json"):
                os.close(fd)
                raise ValueError(f"Unsupported format: {fmt}")
            with os.fdopen(fd, "wb") as raw:
                writer = _DigestWriter(raw)
                if fmt == "pkl":
                    pickle.dump(data, writer)
                else:
                    writer.write(_json_bytes(data))
                writer.flush()
                os.fsync(writer.fileno())
                digest_hex = writer.digest.hexdigest()
            os.replace(tmp_path, path)
            self._write_manifest(path, stage, fmt, digest_hex)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    def load(self, stage: str, fmt: str = "pkl") -> Optional[Any]:
        """Đọc kết quả trung gian nếu đã lưu."""
        path = self._get_stage_path(stage, fmt)
        if not self._is_committed(path, verify_digest=_verify_digest_on_load()):
            return None
            
        if fmt == "pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif fmt == "json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def load_siblings(self, stage: str, fmt: str = "pkl") -> list:
        """Committed payloads of `stage` kept under every OTHER namespace folder, newest first.

        A changed setting or prompt version gets a folder of its own, so the answers made
        before the change are still on disk; this reads them (for replaying model replies).
        """
        base = os.path.join(self.job_dir, stage)
        current = self.namespaces.get(stage)
        found = []
        if os.path.isdir(base):
            for name in os.listdir(base):
                path = os.path.join(base, name, f"result.{fmt}")
                if name != current and os.path.isfile(path) and self._is_committed(path, verify_digest=False):
                    found.append((os.path.getmtime(path), path))
        out = []
        for _, path in sorted(found, reverse=True):
            try:
                with open(path, "rb") as handle:
                    out.append(pickle.load(handle) if fmt == "pkl" else json.load(handle))
            except Exception:
                continue
        return out

    def load_if_committed(self, stage: str, fmt: str = "pkl") -> Optional[Any]:
        """Load one checkpoint after one validation pass.

        Callers commonly used ``exists(); load()``.  That verified and hashed
        a large pickle twice before deserializing it.  A missing or corrupt
        checkpoint returns ``None`` just like ``load``; stage payloads in this
        pipeline never use ``None`` as a successful result.
        """
        return self.load(stage, fmt=fmt)

    def save_array(self, stage: str, array) -> None:
        """Atomically persist a lossless NumPy array beside stage checkpoints."""
        import numpy as np

        path = self._get_stage_path(stage, "npy")
        directory = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(prefix=".result-", suffix=".npy.tmp",
                                        dir=directory)
        try:
            with os.fdopen(fd, "wb") as raw:
                writer = _DigestWriter(raw)
                np.save(writer, np.asarray(array), allow_pickle=False)
                writer.flush()
                os.fsync(writer.fileno())
                digest_hex = writer.digest.hexdigest()
            os.replace(tmp_path, path)
            self._write_manifest(path, stage, "npy", digest_hex)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def load_array(self, stage: str):
        """Load a validated, read-only mmap of a cached array, if present.

        The manifest size check catches interrupted or truncated writes. The
        digest is recorded at commit time and full hash verification remains
        available through ``load``; avoiding a second full read here is what
        makes a multi-stage audio cache useful for long recordings.
        """
        import numpy as np

        path = self._get_stage_path(stage, "npy")
        if not self._is_committed(path, verify_digest=False):
            return None
        try:
            return np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            return None

    def _write_manifest(self, path: str, stage: str, fmt: str,
                        digest_hex: str) -> None:
        manifest = {
            "schema_version": 1,
            "stage": stage,
            "format": fmt,
            "size": os.path.getsize(path),
            "sha256": digest_hex,
            "mtime_ns": os.stat(path).st_mtime_ns,
            "committed_at": time.time(),
        }
        directory = os.path.dirname(path)
        manifest_path = path + ".manifest.json"
        mfd, manifest_tmp = tempfile.mkstemp(
            prefix=".manifest-", suffix=".json.tmp", dir=directory)
        try:
            with os.fdopen(mfd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(manifest_tmp, manifest_path)
        finally:
            if os.path.exists(manifest_tmp):
                os.unlink(manifest_tmp)
        
    def exists(self, stage: str, fmt: str = "pkl", verify_digest: bool = True) -> bool:
        """Kiểm tra một bước đã có checkpoint trong đúng không gian lưu chưa."""
        path = self._get_stage_path(stage, fmt)
        return self._is_committed(path, verify_digest=verify_digest)

    @staticmethod
    def _is_committed(path: str, verify_digest: bool = True) -> bool:
        if not os.path.exists(path):
            return False
        manifest_path = path + ".manifest.json"
        if not os.path.exists(manifest_path):
            # Read checkpoints written before atomic manifests were introduced.
            return True
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            if int(manifest.get("size", -1)) != os.path.getsize(path):
                return False
            if not verify_digest:
                recorded_mtime = manifest.get("mtime_ns")
                if recorded_mtime is not None:
                    return int(recorded_mtime) == os.stat(path).st_mtime_ns
                return True
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest() == manifest.get("sha256")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
