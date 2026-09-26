#!/usr/bin/env python3
"""Refinement LLM worker for either Transformers or vLLM.

Mục đích: tách refinement LLM (Qwen3.5/3.8) ra khỏi main env để tránh
xung đột transformers 4.53 (cần cho pyannote/whisperx) vs 5.x (cần cho
Qwen3.5). Worker này chạy trong refinement_env với transformers>=5.2.

Cài venv:
    python3 -m venv /path/to/refinement_env
    refinement_env/bin/pip install torch transformers>=5.2.0 accelerate

Chạy thủ công:
    REFINEMENT_PYTHON=/path/to/refinement_env/bin/python
    CUDA_VISIBLE_DEVICES=0,1 $REFINEMENT_PYTHON refinement_worker.py \\
        --model Qwen/Qwen3.8-27B --dtype bfloat16

Protocol (stdin/stdout, line-delimited JSON):
    Yêu cầu:
        {"cmd": "ping"}
        {"cmd": "quit"}
        {"cmd": "generate",
         "id": "req-001",
         "system_prompt": "...",
         "user_messages": ["...", "..."],
         "max_new_tokens": 512,
         "thinking": false}
    Phản hồi:
        {"status": "ok"}                   # ping
        {"status": "shutdown"}             # quit
        {"id": "req-001",
         "texts": ["...", "..."],
         "usage": {"prompt_tokens": 0, "completion_tokens": 0},
         "ok": true}                       # generate thành công
        {"id": "req-001", "ok": false,
         "error": "OOM or similar"}        # generate thất bại
        {"status": "ready", "model": "..."} # sau khi load xong
"""

import sys
import json
import os
import argparse
import warnings

import worker_vllm_env  # noqa: F401  (sets VLLM_* defaults before vllm loads)
from utils.llm_sampling import hf_generate_kwargs, vllm_sampling_kwargs

warnings.filterwarnings("ignore")

import torch


def _patch_torch_load():
    """Tránh lỗi weights_only trên checkpoint cũ."""
    if not hasattr(torch.serialization, "add_safe_globals"):
        return
    original = torch.load
    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)
    torch.load = _patched


def load_model(model_name: str, dtype_str: str, device_map: str,
               gpu_ids: list, gpu_memory_fraction: float,
               backend: str = "transformers", tensor_parallel_size: int = 1,
               max_model_len: int = 0, enable_prefix_caching: bool = True):

    if backend == "vllm":
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is missing; install podcast-pipeline/requirements-vllm.txt "
                "in the VLLM_PYTHON environment") from exc
        print(json.dumps({"status": "loading", "model": model_name,
                          "backend": backend}), flush=True)
        kwargs = {
            "model": model_name,
            "dtype": dtype_str,
            "tensor_parallel_size": max(1, int(tensor_parallel_size)),
            "gpu_memory_utilization": min(
                0.95, max(0.1, float(gpu_memory_fraction))),
            "enable_prefix_caching": bool(enable_prefix_caching),
            "trust_remote_code": True,
        }
        if max_model_len:
            kwargs["max_model_len"] = int(max_model_len)
        model = LLM(**kwargs)
        tokenizer = model.get_tokenizer()
        print(json.dumps({"status": "ready", "model": model_name,
                          "backend": backend,
                          "tensor_parallel_size": kwargs["tensor_parallel_size"]}),
              flush=True)
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if not gpu_ids:
        gpu_ids = [0]

    device_count = torch.cuda.device_count()

    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= device_count:
            raise RuntimeError(
                f"Invalid GPU id {gpu_id}; worker sees "
                f"{device_count} GPU(s)"
            )

    gpu_memory_fraction = min(
        0.95,
        max(0.1, float(gpu_memory_fraction))
    )

    dtype = (
        torch.bfloat16
        if dtype_str == "bfloat16"
        else torch.float16
    )

    print(json.dumps({
        "status": "loading",
        "model": model_name
    }), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device_map == "pipeline" and len(gpu_ids) >= 2:

        max_memory = {
            i: int(
                torch.cuda.get_device_properties(i).total_memory
                * gpu_memory_fraction
            )
            for i in gpu_ids
        }

        # Hiện tại là balanced sharding, chưa phải pipeline parallel thật.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="balanced",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )

    elif device_map == "auto":

        max_memory = {
            i: int(
                torch.cuda.get_device_properties(i).total_memory
                * gpu_memory_fraction
            )
            for i in gpu_ids
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="balanced",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )

    else:
        device = f"cuda:{gpu_ids[0]}"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)

    model.eval()

    print(json.dumps({
        "status": "ready",
        "model": model_name,
        "device_map": getattr(model, "hf_device_map", None),
    }), flush=True)

    return model, tokenizer


def _entry_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return model.device


def _render_prompts(tokenizer, system_prompt, user_messages, thinking):
    texts = []
    for msg in user_messages:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": msg}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=bool(thinking))
        except TypeError:
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": msg}],
                tokenize=False, add_generation_prompt=True)
        texts.append(text)
    return texts


def generate(model, tokenizer, system_prompt: str, user_messages: list,
             max_new_tokens: int = 512, thinking: bool = False,
             backend: str = "transformers") -> list:
    """Chạy một batch chat requests, trả về list[str]."""
    return generate_with_usage(model, tokenizer, system_prompt, user_messages,
                               max_new_tokens, thinking, backend)[0]


def generate_with_usage(model, tokenizer, system_prompt: str, user_messages: list,
                        max_new_tokens: int = 512, thinking: bool = False,
                        backend: str = "transformers"):
    """(texts, usage): usage counts the tokens the engine really processed."""
    if not user_messages:
        return [], {"prompt_tokens": 0, "completion_tokens": 0}

    # Tokenize
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    texts = _render_prompts(tokenizer, system_prompt, user_messages, thinking)

    if backend == "vllm":
        from vllm import SamplingParams
        kwargs = vllm_sampling_kwargs(thinking, max_new_tokens)
        detection = kwargs.pop("repetition_detection", None)
        if detection:
            try:
                from vllm.sampling_params import RepetitionDetectionParams
                kwargs["repetition_detection"] = RepetitionDetectionParams(**detection)
            except ImportError:
                pass   # an older vLLM: the token limit is the only stop
        outputs = model.generate(
            texts, sampling_params=SamplingParams(**kwargs), use_tqdm=False)
        usage = {
            "prompt_tokens": sum(len(o.prompt_token_ids or ()) for o in outputs),
            "completion_tokens": sum(len(o.outputs[0].token_ids or ()) for o in outputs),
            "repetition_stops": sum(
                1 for o in outputs
                if getattr(o.outputs[0], "finish_reason", None) == "repetition"),
        }
        return [output.outputs[0].text for output in outputs], usage

    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs = inputs.to(_entry_device(model))

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **hf_generate_kwargs(thinking),
        )

    # Cắt phần prompt
    prompt_len = inputs.input_ids.size(1)
    generated = generated[:, prompt_len:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    usage = {
        "prompt_tokens": int(inputs.attention_mask.sum()),
        # Padding after the first end token is not generated text.
        "completion_tokens": int((generated != tokenizer.pad_token_id).sum()),
    }
    return decoded, usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get(
        "SOMMELIER_LLM", "Qwen/Qwen3-8B-Instruct"))
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--device-map", default="auto",
                        choices=["auto", "pipeline", "single"])
    parser.add_argument("--gpu-ids", default="0,1",
                        help="Comma-separated GPU indices (CUDA_VISIBLE_DEVICES remaps)")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.90)
    parser.add_argument("--backend", choices=["transformers", "vllm"],
                        default="transformers")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]

    _patch_torch_load()

    try:
        model, tokenizer = load_model(
            args.model, args.dtype, args.device_map,
            gpu_ids, args.gpu_memory_fraction, args.backend,
            args.tensor_parallel_size, args.max_model_len,
            not args.disable_prefix_caching)
    except Exception as e:
        from worker_errors import describe_exception
        print(json.dumps({"status": "error", "error": describe_exception(e)}),
              flush=True)
        sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "invalid JSON"}), flush=True)
            continue

        cmd = req.get("cmd", "generate")

        if cmd == "quit":
            print(json.dumps({"status": "shutdown"}), flush=True)
            break

        if cmd == "ping":
            print(json.dumps({"status": "ok"}), flush=True)
            continue

        # cmd == "generate"
        req_id = req.get("id", "")
        try:
            texts, usage = generate_with_usage(
                model, tokenizer,
                system_prompt=req.get("system_prompt", ""),
                user_messages=req.get("user_messages", []),
                max_new_tokens=int(req.get("max_new_tokens", 512)),
                thinking=bool(req.get("thinking", False)),
                backend=args.backend,
            )
            print(json.dumps({"id": req_id, "ok": True, "texts": texts,
                              "usage": usage}), flush=True)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            print(json.dumps({"id": req_id, "ok": False,
                               "error": f"OOM: {e}"}), flush=True)
        except Exception as e:
            print(json.dumps({"id": req_id, "ok": False,
                               "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
