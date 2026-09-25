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
"""

import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
