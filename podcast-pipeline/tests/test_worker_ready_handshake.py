"""A vLLM worker was declared ready while its engine was still compiling: with DEBUG
logging vLLM dumps its environment variables, one of which is VLLM_ENGINE_READY_TIMEOUT_S,
and the generic 'a line containing ready' fallback took it for the handshake. vLLM also
logs to stdout by default, the same pipe as the JSON replies."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.base_worker_service import WorkerProcessService
from services.qwen3_worker_service import Qwen3WorkerService
from services.refinement_worker_service import RefinementWorkerService
from services.whisper_vllm_worker_service import WhisperVLLMWorkerService

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DUMP_LINE = "(EngineCore pid=1) DEBUG 09-25 15:16:22 [compilation/backends.py:1111]  'VLLM_ENGINE_READY_TIMEOUT_S': 600,"


def _service(cls):
    return cls.__new__(cls)


def test_the_generic_service_still_accepts_a_bare_ready_marker():
    service = _service(WorkerProcessService)
    assert service.is_ready_line("READY") is True
    assert service.is_ready_line('{"status": "ready"}') is True
    assert service.is_ready_line('{"status": "loading"}') is False


def test_the_vllm_workers_accept_only_the_json_handshake():
    for cls in (Qwen3WorkerService, WhisperVLLMWorkerService, RefinementWorkerService):
        service = _service(cls)
        assert service.is_ready_line('{"status": "ready", "device": "cuda:0"}') is True, cls
        assert service.is_ready_line(ENV_DUMP_LINE) is False, cls
        assert service.is_ready_line("Engine is ready to serve") is False, cls
        assert service.is_ready_line('{"status": "loading"}') is False, cls


def test_a_json_error_is_still_reported_by_the_strict_workers():
    import pytest
    service = _service(Qwen3WorkerService)
    service.name = "Qwen3-ASR"
    with pytest.raises(RuntimeError, match="boom"):
        service.is_ready_line('{"status": "error", "message": "boom"}')


def test_vllm_logs_go_to_stderr_so_stdout_carries_only_the_protocol():
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "VLLM_LOGGING_STREAM"}
    out = subprocess.run(
        [sys.executable, "-c",
         "import worker_vllm_env, os; print(os.environ.get('VLLM_LOGGING_STREAM'))"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == "ext://sys.stderr"


def test_only_error_lines_from_worker_stderr_reach_the_log():
    import logging
    from services.base_worker_service import WorkerProcessService

    class Capture(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.DEBUG)
            self.records = []

        def emit(self, record):
            self.records.append(record)

    logger = logging.getLogger("stderr-filter-test")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = Capture()
    logger.addHandler(handler)
    service = WorkerProcessService("w", "/usr/bin/python3", "x.py", logger=logger)

    import io
    service._drain_stderr(io.StringIO(
        "INFO 09-25 15:26:52 Graph capturing finished in 35 secs\n"
        "Capturing CUDA graphs:  75%|███\n"
        "Traceback (most recent call last):\n"
        "TypeError: Unsupported audio input type\n"))

    logged = [r.getMessage() for r in handler.records]
    assert len(logged) == 2 and all("Traceback" in m or "TypeError" in m for m in logged)
    assert "Graph capturing" in service.stderr_tail(10)   # still kept for a failed start
