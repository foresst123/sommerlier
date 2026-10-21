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


def test_the_vllm_asr_workers_pass_a_fixed_kv_cache_so_profiling_cannot_trip_on_neighbours():
    # vLLM asserts that nobody freed GPU memory while it profiled; PhoWhisper workers
    # share the cards, so the engines are given a fixed KV cache instead.
    import json
    for name in ("qwen3_worker.py", "whisper_vllm_worker.py"):
        source = open(os.path.join(ROOT, name), encoding="utf-8").read()
        assert 'kwargs["kv_cache_memory_bytes"]' in source, name
    cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    models = cfg["environments"]["a100"]["models"]
    for key in ("qwen3", "whisper"):
        assert int(models[key]["kv_cache_memory_bytes"]) > 0, key
