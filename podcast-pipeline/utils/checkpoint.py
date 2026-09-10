import os
import pickle
import json
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
        """Lưu kết quả trung gian xuống đĩa."""
        path = self._get_stage_path(stage, fmt)
        if fmt == "pkl":
            with open(path, "wb") as f:
                pickle.dump(data, f)
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
            
    def load(self, stage: str, fmt: str = "pkl") -> Optional[Any]:
        """Đọc kết quả trung gian nếu đã lưu."""
        path = self._get_stage_path(stage, fmt)
        if not os.path.exists(path):
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
        return os.path.exists(path)
