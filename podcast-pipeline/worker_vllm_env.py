"""Import this in a vLLM worker before vllm is imported.

vLLM samples tokens with FlashInfer's top-k/top-p kernel by default, and
FlashInfer builds that kernel with nvcc the first time it runs -- during the
engine's start-up profiling. The A100 host has no CUDA toolkit
("Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist"),
so the engine never came up. Turning the FlashInfer sampler off makes vLLM use
its PyTorch sampler. An explicit VLLM_USE_FLASHINFER_SAMPLER in the
environment still wins.

The engine also starts its own process and joins it over tcp://<host IP>. On the
A100 host that address was not reachable from the worker, so the engine hung
right after "FlashInfer ... disabled" with the GPU idle and no error. A single
GPU only needs loopback, so the host IP and the gloo/nccl interface default to
it; set the variables yourself to override.

vLLM starts its engine process with fork by default. Forking a process that has
already run an OpenMP parallel region leaves the child with a thread pool whose
threads do not exist, and the child's first parallel torch op waits for them
forever. On the A100 host the engine hung exactly like that, in
InputBatch.__init__ (a torch.zeros of 8 MB, py-spy'd by worker_trace), with the
GPU idle and 5% CPU. Spawning a fresh interpreter instead does not inherit that
state; the cost is a few seconds of imports.
"""

import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# vLLM logs to stdout by default -- the pipe that carries the JSON replies. The
# client reads one line per reply and json.loads it, so a log line landing between
# request and reply broke the protocol. stderr is drained into the pipeline log.
os.environ.setdefault("VLLM_LOGGING_STREAM", "ext://sys.stderr")
