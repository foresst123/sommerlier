import os
import pickle
import json
import hashlib
import tempfile
import time
from typing import Any, Optional

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
            if fmt == "pkl":
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(data, f)
                    f.flush()
                    os.fsync(f.fileno())
            elif fmt == "json":
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
            else:
                os.close(fd)
                raise ValueError(f"Unsupported format: {fmt}")
            os.replace(tmp_path, path)
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest = {
                "schema_version": 1,
                "stage": stage,
                "format": fmt,
                "size": os.path.getsize(path),
                "sha256": digest.hexdigest(),
                "committed_at": time.time(),
            }
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
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    def load(self, stage: str, fmt: str = "pkl") -> Optional[Any]:
        """Đọc kết quả trung gian nếu đã lưu."""
        path = self._get_stage_path(stage, fmt)
        if not self._is_committed(path):
            return None
            
        if fmt == "pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif fmt == "json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
        
    def exists(self, stage: str, fmt: str = "pkl") -> bool:
        """Kiểm tra một bước đã có checkpoint trong đúng không gian lưu chưa."""
        path = self._get_stage_path(stage, fmt)
        return self._is_committed(path)

    @staticmethod
    def _is_committed(path: str) -> bool:
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
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest() == manifest.get("sha256")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
