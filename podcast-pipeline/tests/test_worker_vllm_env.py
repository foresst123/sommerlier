"""vLLM workers must not JIT-compile FlashInfer's sampler: the DGX has no CUDA
toolkit (no nvcc), and the engine died in its first profiling run because of it."""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKERS = ("qwen3_worker.py", "whisper_vllm_worker.py", "refinement_worker.py")


def _value_after_import(env_value):
    env = {k: v for k, v in os.environ.items() if k != "VLLM_USE_FLASHINFER_SAMPLER"}
    if env_value is not None:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = env_value
    result = subprocess.run(
        [sys.executable, "-c",
         "import worker_vllm_env, os; print(os.environ.get('VLLM_USE_FLASHINFER_SAMPLER'))"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_the_flashinfer_sampler_is_off_unless_someone_asks_for_it():
    assert _value_after_import(None) == "0"


def test_an_explicit_choice_is_left_alone():
    assert _value_after_import("1") == "1"


def test_every_vllm_worker_sets_it_before_vllm_can_be_imported():
    for name in WORKERS:
        source = open(os.path.join(ROOT, name), encoding="utf-8").read()
        assert "import worker_vllm_env" in source, name
        first_vllm = re.search(r"^\s*(from vllm|import vllm)", source, re.MULTILINE)
        assert first_vllm is None or (
            source.index("import worker_vllm_env") < first_vllm.start()), name


def _env_after_import(overrides=None):
    keys = ("VLLM_HOST_IP", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME")
    env = {k: v for k, v in os.environ.items() if k not in keys}
    env.update(overrides or {})
    result = subprocess.run(
        [sys.executable, "-c",
         "import worker_vllm_env, os; print(*[os.environ.get(k) for k in "
         "('VLLM_HOST_IP','GLOO_SOCKET_IFNAME','NCCL_SOCKET_IFNAME')])"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return result.stdout.split()


def test_the_engine_talks_over_loopback_unless_told_otherwise():
    # The engine hung on tcp://<host ip> on the A100 host; one GPU needs no network.
    assert _env_after_import() == ["127.0.0.1", "lo", "lo"]
    assert _env_after_import({"VLLM_HOST_IP": "10.0.0.5"})[0] == "10.0.0.5"


def test_the_qwen3_profile_bounds_the_context_so_the_kv_cache_fits():
    import json
    cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    qwen = cfg["environments"]["a100"]["models"]["qwen3"]
    assert qwen["backend"] == "vllm" and 0 < qwen["max_model_len"] <= 16384



def test_worker_trace_makes_child_processes_dump_their_stacks():
    # EngineCore is a child of the worker; its silent start-up is what we need to see.
    code = ("import os, worker_trace; worker_trace.watch(7); worker_trace.done();"
            "print(os.environ['SOMMELIER_TRACE_CHILD'], os.environ['PYTHONPATH'])")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    assert out[0] == "7" and out[1].split(os.pathsep)[0].endswith("trace_site")
    assert os.path.isfile(os.path.join(ROOT, "trace_site", "sitecustomize.py"))


def test_a_forked_child_does_not_hang_while_the_parent_is_being_traced():
    # vLLM forks its EngineCore. The first version of worker_trace re-armed
    # faulthandler in the child and the child hung before running a line.
    code = (
        "import os, sys, time, worker_trace\n"
        "worker_trace.watch(1)\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os._exit(0)\n"
        "end = time.time() + 5\n"
        "while time.time() < end:\n"
        "    if os.waitpid(pid, os.WNOHANG)[0]:\n"
        "        print('exited'); sys.exit(0)\n"
        "    time.sleep(0.05)\n"
        "os.kill(pid, 9); print('hung')\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=30).stdout
    assert "exited" in out and "hung" not in out


def test_the_engine_process_is_spawned_not_forked_unless_someone_asks():
    # A forked EngineCore hung in its first OpenMP torch op (InputBatch.__init__).
    env = {k: v for k, v in os.environ.items() if k != "VLLM_WORKER_MULTIPROC_METHOD"}
    out = subprocess.run(
        [sys.executable, "-c",
         "import worker_vllm_env, os; print(os.environ.get('VLLM_WORKER_MULTIPROC_METHOD'))"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == "spawn"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    out = subprocess.run(
        [sys.executable, "-c",
         "import worker_vllm_env, os; print(os.environ.get('VLLM_WORKER_MULTIPROC_METHOD'))"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == "fork"


def test_a_spawned_engine_can_reimport_every_vllm_worker_module():
    # spawn re-imports __main__ in the child: each worker must guard its entry point.
    for name in WORKERS:
        source = open(os.path.join(ROOT, name), encoding="utf-8").read()
        assert 'if __name__ == "__main__":' in source, name
