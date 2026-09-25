#!/usr/bin/env python3
"""
Qwen3-ASR Worker — runs in an isolated qwen3_env or vllm_env.

Usage:
    CUDA_VISIBLE_DEVICES=1 /path/to/qwen3_env/bin/python qwen3_worker.py

Protocol (stdin/stdout, line-delimited):
    Input:  JSON line  {"audio_path": "/tmp/seg_xxx.wav", "language": "vi"}
    Output: JSON line  {"text": "transcribed text"}
    
    Send  {"cmd": "quit"}  to shut down the worker.
    Send  {"cmd": "ping"}  to check if the worker is alive → returns {"status": "ok"}
"""

import sys
import json
import os
import numpy as np
import torch
import soundfile as sf
import argparse

import worker_vllm_env  # noqa: F401  (sets VLLM_* defaults before vllm loads)
from worker_errors import describe_exception

import warnings
warnings.filterwarnings("ignore")

if hasattr(torch, "torch_version") and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

def _model_config(config_path=None, env_name="kaggle"):
    qwen_cfg = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            qwen_cfg = cfg.get("environments", {}).get(env_name, {}).get("models", {}).get("qwen3", {})
        except Exception as e:
            print(json.dumps({"error": f"Failed to parse config: {str(e)}"}), flush=True)
    return qwen_cfg


def load_model(config_path=None, env_name="kaggle", batch_size=None):
    """Load the configured Transformers or vLLM Qwen3-ASR backend."""
    qwen_cfg = dict(_model_config(config_path, env_name))
    if batch_size:
        qwen_cfg["batch_size"] = int(batch_size)
    backend = str(qwen_cfg.get("backend", "transformers")).lower()
    default_model = ("Qwen/Qwen3-ASR-1.7B" if backend == "vllm"
                     else "Qwen/Qwen3-ASR-1.7B-hf")
    model_name = qwen_cfg.get("model_name", default_model)
    print(json.dumps({"status": "loading", "model": model_name,
                      "backend": backend}), flush=True)

    if backend == "vllm":
        try:
            # qwen-asr swallows the real reason vllm fails to import and reports
            # a generic "vLLM is not available"; import it here first so the
            # actual error (a missing library, a version clash) is what surfaces.
            import vllm  # noqa: F401
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "vllm / qwen-asr cannot be imported in this environment "
                "(see requirements-vllm.txt): " + describe_exception(exc)) from exc
        kwargs = {
            "model": model_name,
            "max_inference_batch_size": int(qwen_cfg.get("batch_size", 16)),
            "max_new_tokens": int(qwen_cfg.get("max_new_tokens", 256)),
            "gpu_memory_utilization": float(qwen_cfg.get(
                "gpu_memory_utilization", 0.40)),
            "dtype": qwen_cfg.get("torch_dtype", "bfloat16"),
        }
        if qwen_cfg.get("max_model_len"):
            kwargs["max_model_len"] = int(qwen_cfg["max_model_len"])
        if qwen_cfg.get("kv_cache_memory_bytes"):
            # A fixed KV cache skips vLLM's memory profiling, which asserts that no other
            # process frees GPU memory while it runs. PhoWhisper workers share these
            # cards and free memory between batches, so profiling failed at start-up
            # ("Error in memory profiling"). gpu_memory_utilization then no longer applies.
            kwargs["kv_cache_memory_bytes"] = int(qwen_cfg["kv_cache_memory_bytes"])
        model = Qwen3ASRModel.LLM(**kwargs)
        print(json.dumps({"status": "ready", "device": "cuda:0",
                          "backend": backend}), flush=True)
        return model, None, None, backend

    from transformers import AutoProcessor, AutoModelForMultimodalLM
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES remaps the physical GPU

    dtype_str = qwen_cfg.get("torch_dtype", "float16")
    use_bf16 = dtype_str == "bfloat16" or os.environ.get("SOMMELIER_USE_BF16") == "1"
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_flash_attention = qwen_cfg.get("use_flash_attention", False)

    processor = AutoProcessor.from_pretrained(model_name)
    model_kwargs = {
        "device_map": {"": device},
        "torch_dtype": dtype
    }
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Ép kiểu toàn bộ model một cách triệt để
    model.to(dtype)
    
    model.eval()

    print(json.dumps({"status": "ready", "device": str(device),
                      "backend": backend}), flush=True)
    return model, processor, device, backend


def _read_audio(audio_path):
    """Load a request's audio as float32 mono at 16 kHz.

    The pipeline already holds 16 kHz float32 in memory, so it hands over a .npy
    dump and skips the WAV encode/decode round trip entirely.
    """
    if audio_path.endswith(".npy"):
        return np.load(audio_path).astype("float32"), 16000
    return sf.read(audio_path, dtype="float32")


def _qwen_language(language):
    aliases = {
        "vi": "Vietnamese", "vietnamese": "Vietnamese",
        "en": "English", "english": "English",
    }
    return aliases.get(str(language or "").strip().lower())


def transcribe(model, processor, device, audio_path, language="vi",
               backend="transformers"):
    """Run Qwen3-ASR inference on an audio file."""
    try:
        audio_data, sr = _read_audio(audio_path)
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

        if backend == "vllm":
            result = model.transcribe(
                audio=(audio_data, 16000),
                language=_qwen_language(language),
            )[0]
            return result.text.strip()

        conversation = [
            {"role": "system", "content": "You are a highly accurate Vietnamese ASR system. Transcribe the audio precisely. Maintain natural punctuation and capitalization. Ignore background noise, music, and do not hallucinate content if the audio is silent or unintelligible."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "dummy"},
                {"type": "text", "text": "Transcription in Vietnamese."},
            ]}
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        dtype = model.dtype
        inputs = processor(text=text, audio=audio_data, return_tensors="pt", sampling_rate=16000).to(device, dtype)

        with torch.no_grad():
            # Tối ưu hóa Decoding: Temperature=0.0 để giảm ảo giác, no_repeat_ngram_size=3 chặn lặp từ
            gen_ids = model.generate(
                **inputs, 
                max_new_tokens=256, 
                do_sample=False, 
                temperature=0.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            gen_ids = gen_ids[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            if "<asr_text>" in response:
                response = response.split("<asr_text>")[-1]

        return response.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def transcribe_batch(model, processor, device, jobs, language="vi",
                     backend="transformers"):
    """Run one padded generation batch, with a compatibility fallback."""
    if not jobs:
        return []
    try:
        audio_arrays = []
        prompts = []
        for job in jobs:
            audio_data, sr = _read_audio(job["audio_path"])
            if sr != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data, orig_sr=sr, target_sr=16000)
            audio_arrays.append(audio_data)
        if backend == "vllm":
            # qwen_asr takes a path/URL or a (waveform, sample_rate) pair; a bare
            # ndarray raises "Unsupported audio input type".
            clips = [(array, 16000) for array in audio_arrays]
            forced_language = _qwen_language(language)
            outputs = model.transcribe(
                audio=clips,
                language=[forced_language] * len(clips),
            )
            return [{"id": str(job["id"]), "text": output.text.strip()}
                    for job, output in zip(jobs, outputs)]

        for job in jobs:
            conversation = [
                {"role": "system", "content": "You are a highly accurate Vietnamese ASR system. Transcribe the audio precisely. Maintain natural punctuation and capitalization. Ignore background noise, music, and do not hallucinate content if the audio is silent or unintelligible."},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "dummy"},
                    {"type": "text", "text": "Transcription in Vietnamese."},
                ]},
            ]
            prompts.append(processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False))
        inputs = processor(text=prompts, audio=audio_arrays, padding=True,
                           return_tensors="pt", sampling_rate=16000).to(
                               device, model.dtype)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=256, do_sample=False, temperature=0.0,
                repetition_penalty=1.2, no_repeat_ngram_size=3)
        generated = generated[:, inputs.input_ids.size(1):]
        decoded = processor.batch_decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        results = []
        for job, response in zip(jobs, decoded):
            if "<asr_text>" in response:
                response = response.split("<asr_text>")[-1]
            results.append({"id": str(job["id"]), "text": response.strip()})
        return results
    except Exception as exc:
        # Transformers/Qwen processor APIs have changed between releases.
        # Keep correctness and report the fallback rather than losing a batch.
        print(f"[Qwen3Worker] batch fallback: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return [{"id": str(job["id"]),
                 "text": transcribe(model, processor, device,
                                    job["audio_path"], language, backend)}
                for job in jobs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--env", default="kaggle")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override models.qwen3.batch_size (used by replicas).")
    args = parser.parse_args()

    try:
        model, processor, device, backend = load_model(
            args.config, args.env, args.batch_size)
    except Exception as exc:
        print(json.dumps({"status": "error", "message": describe_exception(exc)}),
              flush=True)
        raise

    # Read commands from stdin, one JSON per line
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "invalid JSON"}), flush=True)
            continue

        cmd = request.get("cmd", "transcribe")

        if cmd == "quit":
            print(json.dumps({"status": "shutdown"}), flush=True)
            break
        elif cmd == "ping":
            print(json.dumps({"status": "ok"}), flush=True)
            continue

        if cmd == "transcribe_batch":
            jobs = request.get("jobs") or []
            missing = [job.get("audio_path", "") for job in jobs
                       if not job.get("audio_path")
                       or not os.path.exists(job.get("audio_path", ""))]
            if missing:
                print(json.dumps({"error": f"audio file not found: {missing[0]}"}),
                      flush=True)
            else:
                print(json.dumps({"results": transcribe_batch(
                    model, processor, device, jobs,
                    request.get("language", "vi"), backend)}), flush=True)
            continue

        # Transcribe
        audio_path = request.get("audio_path", "")
        language = request.get("language", "vi")

        if not audio_path or not os.path.exists(audio_path):
            print(json.dumps({"error": f"audio file not found: {audio_path}"}), flush=True)
            continue

        text = transcribe(model, processor, device, audio_path, language, backend)
        print(json.dumps({"text": text}), flush=True)


if __name__ == "__main__":
    main()
