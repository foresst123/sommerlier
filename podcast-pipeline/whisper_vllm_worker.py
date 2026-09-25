#!/usr/bin/env python3
"""Offline vLLM worker for Whisper large-v3.

The worker keeps vLLM out of the pipeline's main Python environment and speaks
the same line-delimited JSON protocol as the other model workers.
"""

import argparse
import json
import os
import sys

import numpy as np

from worker_errors import describe_exception


def _profile(config_path, env_name):
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    return (config.get("environments", {}).get(env_name, {})
            .get("models", {}).get("whisper", {}))


def _language_token(language):
    value = str(language or "").strip().lower()
    aliases = {"vietnamese": "vi", "english": "en"}
    return aliases.get(value, value or "vi")


def load_model(config_path, env_name):
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM audio support is missing; install podcast-pipeline/"
            "requirements-vllm.txt in the VLLM_PYTHON environment") from exc

    cfg = _profile(config_path, env_name)
    model_name = cfg.get("model_name") or cfg.get("model_size", "large-v3")
    if "/" not in model_name:
        model_name = f"openai/whisper-{model_name}"
    kwargs = {
        "model": model_name,
        "dtype": cfg.get("torch_dtype", "bfloat16"),
        "gpu_memory_utilization": float(cfg.get(
            "gpu_memory_utilization", 0.40)),
        "limit_mm_per_prompt": {"audio": 1},
    }
    if cfg.get("max_model_len"):
        kwargs["max_model_len"] = int(cfg["max_model_len"])
    print(json.dumps({"status": "loading", "model": model_name,
                      "backend": "vllm"}), flush=True)
    llm = LLM(**kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=int(cfg.get("max_new_tokens", 256)),
    )
    print(json.dumps({"status": "ready", "model": model_name,
                      "backend": "vllm"}), flush=True)
    return llm, sampling


def transcribe_batch(llm, sampling, jobs, language):
    token = _language_token(language)
    prompt = f"<|startoftranscript|><|{token}|><|transcribe|><|notimestamps|>"
    inputs = []
    for job in jobs:
        audio = np.load(job["audio_path"]).astype(np.float32, copy=False)
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"audio": (audio, 16000)},
        })
    outputs = llm.generate(inputs, sampling_params=sampling, use_tqdm=False)
    return [{
        "id": str(job["id"]),
        "text": output.outputs[0].text.strip(),
        "language": token,
    } for job, output in zip(jobs, outputs)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--env", default="a100")
    args = parser.parse_args()
    try:
        llm, sampling = load_model(args.config, args.env)
    except Exception as exc:
        print(json.dumps({"status": "error", "message": describe_exception(exc)}),
              flush=True)
        raise

    for line in sys.stdin:
        try:
            request = json.loads(line)
            cmd = request.get("cmd", "transcribe")
            if cmd == "quit":
                print(json.dumps({"status": "shutdown"}), flush=True)
                break
            if cmd == "ping":
                print(json.dumps({"status": "ok"}), flush=True)
                continue
            jobs = request.get("jobs") or [{
                "id": request.get("id", "0"),
                "audio_path": request.get("audio_path", ""),
            }]
            missing = [job.get("audio_path", "") for job in jobs
                       if not job.get("audio_path")
                       or not os.path.exists(job.get("audio_path", ""))]
            if missing:
                raise FileNotFoundError(f"audio file not found: {missing[0]}")
            results = transcribe_batch(
                llm, sampling, jobs, request.get("language", "vi"))
            if cmd == "transcribe_batch":
                print(json.dumps({"results": results}), flush=True)
            else:
                print(json.dumps(results[0]), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
