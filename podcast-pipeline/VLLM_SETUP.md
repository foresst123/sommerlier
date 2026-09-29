# vLLM runtime for the A100 profile

The `a100` profile uses one isolated Python 3.12 environment for all three
vLLM-backed models:

- `Qwen/Qwen3-ASR-1.7B`
- `openai/whisper-large-v3`
- `Qwen/Qwen3.5-9B`

Keep the main pipeline environment unchanged. It contains the older
Transformers/WhisperX/Pyannote stack, while current vLLM and Qwen3.5 require a
newer, tightly coupled Torch/Transformers stack.

```bash
cd podcast-pipeline
bash scripts/create_vllm_env.sh /home/lamkd2/sommelier_envs/vllm_env
export VLLM_PYTHON=/home/lamkd2/sommelier_envs/vllm_env/bin/python
```

The variable must be exported in every shell/job that launches `main.py`.
Alternatively set the same absolute interpreter path under
`environments.a100.worker_envs.vllm` in `config.json`.

Before a full run, verify imports and GPU visibility:

```bash
$VLLM_PYTHON -c 'import qwen_asr, vllm; print(vllm.__version__)'
CUDA_VISIBLE_DEVICES=0 $VLLM_PYTHON -c 'import torch; print(torch.cuda.get_device_name(0))'
```

The two ASR vLLM engines share the configured ASR GPU at 40% each. During
refinement, ASR workers are released and two Qwen3.5 replicas are started, one
per A100; each refinement batch is distributed across both replicas while
preserving request order.
