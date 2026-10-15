"""Workers in another virtualenv must not load this process's CUDA libraries, and
a worker that dies at start-up must say why."""

import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.base_worker_service import (
    WorkerProcessService, foreign_library_paths_removed)


def test_only_paths_from_other_environments_are_dropped():
    ld = ":".join([
        "/envs/main/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/usr/local/cuda/lib64",
        "/envs/vllm/lib/python3.12/site-packages/nvidia/cublas/lib",
        "/envs/main/lib/python3.12/site-packages/torch/lib",
    ])
    kept = foreign_library_paths_removed(ld, "/envs/vllm/bin/python")
    assert kept == ":".join([
        "/usr/local/cuda/lib64",
        "/envs/vllm/lib/python3.12/site-packages/nvidia/cublas/lib"])


def test_nothing_left_means_no_variable_at_all():
    ld = "/envs/main/lib/python3.12/site-packages/nvidia/cudnn/lib"
    assert foreign_library_paths_removed(ld, "/envs/vllm/bin/python") == ""
    assert foreign_library_paths_removed("", "/envs/vllm/bin/python") == ""
    assert foreign_library_paths_removed(None, "/envs/vllm/bin/python") == ""


def _service(tmp_path, body, **kwargs):
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(body))
    return WorkerProcessService(
        name="Fake", python_bin=sys.executable, worker_script=str(script),
        ready_timeout=20, **kwargs)


RECORD_ENV = """
    import json, os, sys
    open(os.environ["OUT_FILE"], "w").write(os.environ.get("LD_LIBRARY_PATH", "<unset>"))
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        pass
"""


def _env_seen_by_worker(tmp_path, monkeypatch, isolate):
    other = "/somewhere/else_env/lib/python3.12/site-packages/nvidia/cublas/lib"
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{other}:/usr/local/cuda/lib64")
    monkeypatch.setenv("OUT_FILE", str(tmp_path / "seen.txt"))
    service = _service(tmp_path, RECORD_ENV, isolate_library_path=isolate)
    service.start()
    service.stop()
    return (tmp_path / "seen.txt").read_text()


def test_an_isolated_worker_does_not_inherit_another_environments_libraries(
        tmp_path, monkeypatch):
    assert _env_seen_by_worker(tmp_path, monkeypatch, True) == "/usr/local/cuda/lib64"


def test_by_default_the_library_path_is_inherited_unchanged(tmp_path, monkeypatch):
    seen = _env_seen_by_worker(tmp_path, monkeypatch, False)
    assert "else_env" in seen and "/usr/local/cuda/lib64" in seen


def test_a_start_up_failure_reports_more_than_the_last_ten_stderr_lines(tmp_path):
    service = _service(tmp_path, """
        import sys
        for n in range(30):
            print(f"trace-line-{n:02d}", file=sys.stderr, flush=True)
        sys.exit(1)
    """)
    try:
        service.start()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("the worker should not have become ready")
    assert "trace-line-29" in message
    assert "trace-line-05" in message          # beyond the old 10-line tail


def test_an_unexpected_ready_error_still_reaps_the_worker(tmp_path):
    """A protocol/parser error must be as transactional as a reported OOM.

    _fail_start handles the expected exit/error paths itself.  This exercises
    the formerly-uncovered exception path around readiness parsing, where a
    live worker (and a vLLM engine child in production) was left behind.
    """
    service = _service(tmp_path, """
        import json, sys
        print(json.dumps({"status": "ready"}), flush=True)
        for line in sys.stdin:
            pass
    """)

    def broken_protocol(_line):
        raise ValueError("broken ready message")

    service.is_ready_line = broken_protocol
    try:
        service.start()
    except ValueError as exc:
        assert "broken ready message" in str(exc)
    else:
        raise AssertionError("readiness parsing should have failed")

    assert service.process is None, "failed startup left its process group alive"


# --- which workers are isolated ------------------------------------------------

def test_vllm_workers_do_not_inherit_the_main_environments_libraries():
    from services.qwen3_worker_service import Qwen3WorkerService
    from services.refinement_worker_service import RefinementWorkerService
    from services.whisper_vllm_worker_service import WhisperVLLMWorkerService

    assert WhisperVLLMWorkerService("/e/vllm/bin/python", "w.py").isolate_library_path
    assert RefinementWorkerService(
        "/e/vllm/bin/python", "r.py", "m", backend="vllm").isolate_library_path
    assert not RefinementWorkerService(
        "/e/ref/bin/python", "r.py", "m").isolate_library_path
    assert not Qwen3WorkerService("/e/q/bin/python", "q.py").isolate_library_path
    assert Qwen3WorkerService(
        "/e/vllm/bin/python", "q.py", isolate_library_path=True).isolate_library_path


def test_the_vllm_workers_report_the_real_import_error():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qwen = open(os.path.join(root, "qwen3_worker.py"), encoding="utf-8").read()
    assert "import vllm  # noqa: F401" in qwen
    assert qwen.index("import vllm  # noqa") < qwen.index("from qwen_asr import Qwen3ASRModel")
    for name in ("qwen3_worker.py", "whisper_vllm_worker.py", "refinement_worker.py"):
        source = open(os.path.join(root, name), encoding="utf-8").read()
        assert "describe_exception" in source, name


DIE_ON_REQUEST = """
    import json, sys
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        print("Unable to load any of {libcudnn_ops.so.9}", file=sys.stderr, flush=True)
        sys.exit(134)
"""


def test_a_worker_that_dies_on_a_request_raises_worker_died_with_its_stderr(tmp_path):
    import pytest
    from services.base_worker_service import WorkerDiedError

    service = _service(tmp_path, DIE_ON_REQUEST)
    service.start()
    try:
        with pytest.raises(WorkerDiedError, match="libcudnn_ops"):
            service.request({"cmd": "ping"})
        with pytest.raises(WorkerDiedError):
            service.request({"cmd": "ping"})       # stays dead, still the same error type
    finally:
        service.stop()


def _fake_env(tmp_path, packages):
    root = tmp_path / "env"
    (root / "bin").mkdir(parents=True)
    python = root / "bin" / "python"
    python.symlink_to(sys.executable)
    site = root / "lib" / "python3.12" / "site-packages"
    for name in packages:
        (site / "nvidia" / name / "lib").mkdir(parents=True)
    return str(python), str(site)


def test_the_cuda_libraries_of_the_workers_own_environment_are_found(tmp_path):
    from services.base_worker_service import own_cuda_library_paths

    python, site = _fake_env(tmp_path, ["cudnn"])
    assert own_cuda_library_paths(python, ("cudnn", "cublas")) == [
        os.path.join(site, "nvidia", "cudnn", "lib")]          # missing cublas is skipped


def test_a_worker_asking_for_cuda_libraries_gets_them_on_its_library_path(
        tmp_path, monkeypatch):
    python, site = _fake_env(tmp_path, ["cudnn", "cublas"])
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
    monkeypatch.setenv("OUT_FILE", str(tmp_path / "seen.txt"))
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(RECORD_ENV))
    service = WorkerProcessService(
        name="Fake", python_bin=python, worker_script=str(script), ready_timeout=20,
        cuda_library_packages=("cudnn", "cublas"))
    service.start()
    service.stop()
    assert (tmp_path / "seen.txt").read_text().split(":") == [
        os.path.join(site, "nvidia", "cudnn", "lib"),
        os.path.join(site, "nvidia", "cublas", "lib"),
        "/usr/local/cuda/lib64"]


def test_the_phowhisper_worker_asks_for_cudnn_and_cublas():
    from services.phowhisper_worker_service import PhoWhisperWorkerService

    service = PhoWhisperWorkerService(sys.executable, "phowhisper_worker.py")
    assert set(service.cuda_library_packages) == {"cudnn", "cublas"}
