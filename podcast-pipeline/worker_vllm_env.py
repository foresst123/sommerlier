"""Import this in a vLLM worker before vllm is imported.

vLLM samples tokens with FlashInfer's top-k/top-p kernel by default, and
FlashInfer builds that kernel with nvcc the first time it runs -- during the
engine's start-up profiling. The A100 host has no CUDA toolkit
("Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist"),
so the engine never came up. Turning the FlashInfer sampler off makes vLLM use
its PyTorch sampler. An explicit VLLM_USE_FLASHINFER_SAMPLER in the
environment still wins.
"""

import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
