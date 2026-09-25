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
from concurrent.futures import ThreadPoolExecutor
from services.base_worker_service import WorkerProcessService


class RefinementWorkerService(WorkerProcessService):
    """Subprocess wrapper cho refinement LLM worker."""

    def __init__(self, python_env_path: str, worker_script_path: str,
                 model_name: str, dtype: str = "bfloat16",
                 device_map: str = "auto", gpu_ids: str = "0,1",
                 gpu_memory_fraction: float = 0.90,
                 backend: str = "transformers", device_id=None,
                 tensor_parallel_size: int = 1, max_model_len: int = 0,
                 enable_prefix_caching: bool = True,
                 logger=None):
        worker_args = [
            "--model", model_name,
            "--dtype", dtype,
            "--device-map", device_map,
            "--gpu-ids", gpu_ids,
            "--gpu-memory-fraction", str(gpu_memory_fraction),
            "--backend", backend,
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--max-model-len", str(max_model_len),
        ]
        if not enable_prefix_caching:
            worker_args.append("--disable-prefix-caching")
        super().__init__(
            name="refinement",
            python_bin=python_env_path,
            worker_script=worker_script_path,
            extra_args=worker_args,
            device_id=device_id,
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


class RefinementWorkerPoolService:
    """Two independent vLLM replicas with stable output ordering."""

    def __init__(self, services, logger=None):
        self.services = list(services)
        if not self.services:
            raise ValueError("refinement worker pool requires at least one service")
        self.name = "refinement"
        self.logger = logger
        self.model_name = self.services[0].model_name

    @property
    def process(self):
        return self.services[0].process

    @property
    def processes(self):
        return [service.process for service in self.services
                if service.process is not None]

    def start(self):
        for service in self.services:
            service.spawn()
        with ThreadPoolExecutor(max_workers=len(self.services)) as executor:
            futures = [executor.submit(service.wait_ready)
                       for service in self.services]
            for future in futures:
                future.result()

    def stop(self):
        for service in self.services:
            service.stop()

    def ping(self):
        return all(service.ping() for service in self.services)

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       thinking=False):
        if not user_messages:
            return True, []
        count = min(len(self.services), len(user_messages))
        assignments = [[] for _ in range(count)]
        for index, message in enumerate(user_messages):
            assignments[index % count].append((index, message))

        def run(worker_index):
            indexed = assignments[worker_index]
            ok, texts = self.services[worker_index].generate_texts(
                system_prompt, [message for _, message in indexed],
                max_new_tokens=max_new_tokens, thinking=thinking)
            return ok, indexed, texts

        ordered = [""] * len(user_messages)
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [executor.submit(run, index) for index in range(count)]
            for future in futures:
                ok, indexed, texts = future.result()
                if not ok or len(texts) != len(indexed):
                    return False, []
                for (original_index, _), text in zip(indexed, texts):
                    ordered[original_index] = text
        return True, ordered
