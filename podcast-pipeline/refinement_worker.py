#!/usr/bin/env python3
"""Refinement LLM Worker — chạy trong venv riêng có transformers>=5.2.

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
               gpu_ids: list, gpu_memory_fraction: float):
    """Load model và tokenizer, trả về (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    print(json.dumps({"status": "loading", "model": model_name}), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device_map == "pipeline" and len(gpu_ids) >= 2:
        # Chia đôi layers: GPU 0 phần đầu, GPU 1 phần sau
        # refinement_pipeline_pool.py làm điều này bằng cách split layers.
        # Ở đây ta dùng device_map="balanced" để đơn giản hơn.
        max_memory = {
            i: int(torch.cuda.get_device_properties(i).total_memory
                   * gpu_memory_fraction)
            for i in gpu_ids
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="balanced",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
    elif device_map == "auto" or len(gpu_ids) > 1:
        max_memory = {
            i: int(torch.cuda.get_device_properties(i).total_memory
                   * gpu_memory_fraction)
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
        device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cuda:0"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)

    model.eval()
    print(json.dumps({"status": "ready", "model": model_name}), flush=True)
    return model, tokenizer


def _entry_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return model.device


def generate(model, tokenizer, system_prompt: str, user_messages: list,
             max_new_tokens: int = 512, thinking: bool = False) -> list:
    """Chạy một batch chat requests, trả về list[str]."""
    if not user_messages:
        return []

    # Tokenize
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    texts = []
    for msg in user_messages:
        try:
            t = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": msg}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(thinking),
            )
        except TypeError:
            # transformers cũ không có enable_thinking
            t = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
        texts.append(t)

    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs = inputs.to(_entry_device(model))

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Cắt phần prompt
    generated = generated[:, inputs.input_ids.size(1):]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return decoded


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
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]

    _patch_torch_load()

    try:
        model, tokenizer = load_model(
            args.model, args.dtype, args.device_map,
            gpu_ids, args.gpu_memory_fraction)
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}), flush=True)
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
            texts = generate(
                model, tokenizer,
                system_prompt=req.get("system_prompt", ""),
                user_messages=req.get("user_messages", []),
                max_new_tokens=int(req.get("max_new_tokens", 512)),
                thinking=bool(req.get("thinking", False)),
            )
            print(json.dumps({"id": req_id, "ok": True, "texts": texts}),
                  flush=True)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            print(json.dumps({"id": req_id, "ok": False,
                               "error": f"OOM: {e}"}), flush=True)
        except Exception as e:
            print(json.dumps({"id": req_id, "ok": False,
                               "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
