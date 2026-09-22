"""Client IPC cho refinement_worker subprocess.

Thay thế việc load AutoModelForCausalLM trực tiếp trong main process.
Worker chạy trong refinement_env có transformers>=5.2, tránh xung đột
với transformers==4.53 của main env (pyannote, whisperx, DiariZen).

Cách dùng:
    Đặt trong config.json:
        "worker_envs": {"refinement": "/path/to/refinement_env/bin/python"}
    Hoặc env var:
        export REFINEMENT_PYTHON=/path/to/refinement_env/bin/python

    Nếu không tìm thấy refinement_env → fallback về load trực tiếp
    (hành vi cũ, vẫn cảnh báo về xung đột).
"""

import json
import threading
import uuid
from services.base_worker_service import BaseWorkerService


class RefinementWorkerService(BaseWorkerService):
    """Subprocess wrapper cho refinement LLM worker."""

    def __init__(self, python_env_path: str, worker_script_path: str,
                 model_name: str, dtype: str = "bfloat16",
                 device_map: str = "auto", gpu_ids: str = "0,1",
                 gpu_memory_fraction: float = 0.90,
                 logger=None):
        worker_args = [
            "--model", model_name,
            "--dtype", dtype,
            "--device-map", device_map,
            "--gpu-ids", gpu_ids,
            "--gpu-memory-fraction", str(gpu_memory_fraction),
        ]
        super().__init__(
            name="refinement",
            python_bin=python_env_path,
            script=worker_script_path,
            extra_args=worker_args,
            logger=logger,
        )
        self._lock = threading.Lock()
        self.model_name = model_name

    # ── IPC ────────────────────────────────────────────────────────────────

    def generate_texts(self, system_prompt: str, user_messages: list,
                       max_new_tokens: int = 512,
                       thinking: bool = False) -> tuple[bool, list]:
        """Gửi batch generate request, trả về (ok, texts).

        Interface giống DiarizationRefinementService.generate_texts để
        caller không cần biết đang dùng subprocess hay direct load.
        """
        if not user_messages or not self.process:
            return False, []

        req_id = str(uuid.uuid4())[:8]
        req = {
            "cmd": "generate",
            "id": req_id,
            "system_prompt": system_prompt,
            "user_messages": user_messages,
            "max_new_tokens": max_new_tokens,
            "thinking": thinking,
        }

        try:
            with self._lock:
                self.process.stdin.write(json.dumps(req) + "\n")
                self.process.stdin.flush()
                line = self.process.stdout.readline()

            if not line:
                if self.logger:
                    self.logger.warning("[refinement_worker] empty response")
                return False, []

            resp = json.loads(line.strip())
            if resp.get("ok"):
                return True, resp.get("texts", [])
            else:
                if self.logger:
                    self.logger.warning(
                        f"[refinement_worker] generate failed: "
                        f"{resp.get('error', 'unknown')}")
                return False, []

        except Exception as e:
            if self.logger:
                self.logger.error(f"[refinement_worker] IPC error: {e}")
            return False, []

    def ping(self) -> bool:
        """Kiểm tra worker còn sống không."""
        if not self.process:
            return False
        try:
            with self._lock:
                self.process.stdin.write(json.dumps({"cmd": "ping"}) + "\n")
                self.process.stdin.flush()
                line = self.process.stdout.readline()
            return json.loads(line.strip()).get("status") == "ok"
        except Exception:
            return False
