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
import queue
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from services.base_worker_service import WorkerProcessService


class RefinementWorkerService(WorkerProcessService):
    """Subprocess wrapper cho refinement LLM worker."""

    ready_requires_json = True   # vLLM logs to stdout; see base_worker_service

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
            isolate_library_path=(backend == "vllm"),
        )
        self._lock = threading.Lock()
        self.model_name = model_name
        # Tokens the engine really processed, from the worker's own counts.
        self._usage_lock = threading.Lock()
        self._usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    @property
    def usage(self) -> dict:
        with self._usage_lock:
            return dict(self._usage)

    def _add_usage(self, reported, requests: int) -> None:
        if not isinstance(reported, dict):
            return
        with self._usage_lock:
            self._usage["requests"] += int(requests)
            for key in ("prompt_tokens", "completion_tokens"):
                self._usage[key] += int(reported.get(key, 0) or 0)

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
                self._add_usage(resp.get("usage"), len(user_messages))
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
    """Independent vLLM replicas that pull work from one shared queue.

    Every replica has a thread that takes the next job as soon as its engine is
    free, so a replica that finishes early is never left waiting for the other one
    and several callers can have jobs in flight at once. `generate_texts` keeps its
    old contract (one call, replies in request order) on top of that queue.
    """

    def __init__(self, services, logger=None):
        self.services = list(services)
        if not self.services:
            raise ValueError("refinement worker pool requires at least one service")
        self.name = "refinement"
        self.logger = logger
        self.model_name = self.services[0].model_name
        self._jobs = queue.Queue()
        self._threads = []
        self._threads_lock = threading.Lock()

    @property
    def process(self):
        return self.services[0].process

    @property
    def processes(self):
        return [service.process for service in self.services
                if service.process is not None]

    @property
    def usage(self) -> dict:
        total = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for service in self.services:
            for key, value in (getattr(service, "usage", None) or {}).items():
                total[key] = total.get(key, 0) + int(value)
        return total

    def start(self):
        for service in self.services:
            service.spawn()
        with ThreadPoolExecutor(max_workers=len(self.services)) as executor:
            futures = [executor.submit(service.wait_ready)
                       for service in self.services]
            for future in futures:
                future.result()
        self._ensure_threads()

    def stop(self):
        self._stop_threads()
        for service in self.services:
            service.stop()

    def ping(self):
        return all(service.ping() for service in self.services)

    # -- shared queue ----------------------------------------------------------
    def _ensure_threads(self):
        with self._threads_lock:
            if self._threads:
                return
            for index, service in enumerate(self.services):
                thread = threading.Thread(
                    target=self._replica_loop, args=(index, service), daemon=True,
                    name=f"refinement-replica-{index}")
                self._threads.append(thread)
                thread.start()

    def _stop_threads(self):
        with self._threads_lock:
            threads, self._threads = self._threads, []
        for _ in threads:
            self._jobs.put(None)
        for thread in threads:
            thread.join(timeout=5)

    @staticmethod
    def _alive(service) -> bool:
        process = getattr(service, "process", None)
        poll = getattr(process, "poll", None)
        return process is not None and (not callable(poll) or poll() is None)

    def _replica_loop(self, index, service):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            future, request, tried = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                ok, texts = service.generate_texts(
                    request["system_prompt"], request["user_messages"],
                    max_new_tokens=request["max_new_tokens"],
                    thinking=request["thinking"])
            except Exception:
                ok, texts = False, []
            tried = tried | {index}
            if not ok and not self._alive(service):
                # A replica whose engine died leaves the pool; the job goes to one
                # that is still up, once, before it is reported as failed.
                others = [i for i, other in enumerate(self.services)
                          if i not in tried and self._alive(other)]
                if others:
                    retry = Future()
                    self._jobs.put((retry, request, tried))
                    retry.add_done_callback(
                        lambda done, outer=future: outer.set_result(done.result()))
                else:
                    future.set_result((False, []))
                return
            future.set_result((ok, texts))

    def submit(self, system_prompt, user_messages, max_new_tokens=512,
               thinking=False) -> Future:
        """Queue one request; the Future resolves to (ok, texts)."""
        future = Future()
        if not user_messages:
            future.set_result((True, []))
            return future
        self._ensure_threads()
        request = {"system_prompt": system_prompt,
                   "user_messages": list(user_messages),
                   "max_new_tokens": max_new_tokens, "thinking": thinking}
        self._jobs.put((future, request, frozenset()))
        return future

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       thinking=False):
        if not user_messages:
            return True, []
        count = min(len(self.services), len(user_messages))
        assignments = [[] for _ in range(count)]
        for index, message in enumerate(user_messages):
            assignments[index % count].append((index, message))

        futures = [self.submit(system_prompt, [message for _, message in indexed],
                               max_new_tokens=max_new_tokens, thinking=thinking)
                   for indexed in assignments]
        ordered = [""] * len(user_messages)
        failed = False
        for indexed, future in zip(assignments, futures):
            ok, texts = future.result()
            if not ok or len(texts) != len(indexed):
                failed = True
                continue
            for (original_index, _), text in zip(indexed, texts):
                ordered[original_index] = text
        return (False, []) if failed else (True, ordered)
