#!/usr/bin/env python3
"""
Qwen3-ASR Worker — runs in a separate environment (qwen3_env) on GPU 1.

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


def load_model():
    """Load Qwen3-ASR model and processor."""
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    model_name = "Qwen/Qwen3-ASR-1.7B-hf"
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES remaps physical GPU 1 → cuda:0

    print(json.dumps({"status": "loading", "model": model_name}), flush=True)

    import os
    use_bf16 = os.environ.get("SOMMELIER_USE_BF16") == "1"
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        device_map={"": device},
        torch_dtype=dtype
    )
    
    # Ép kiểu toàn bộ model một cách triệt để
    model.to(dtype)
    
    model.eval()

    print(json.dumps({"status": "ready", "device": str(device)}), flush=True)
    return model, processor, device


def transcribe(model, processor, device, audio_path, language="vi"):
    """Run Qwen3-ASR inference on an audio file."""
    try:
        audio_data, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "dummy"},
                {"type": "text", "text": f"Transcribe the audio in Vietnamese."},
            ]}
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        use_bf16 = os.environ.get("SOMMELIER_USE_BF16") == "1"
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        inputs = processor(text=text, audio=audio_data, return_tensors="pt", sampling_rate=16000).to(device, dtype)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.2)
            gen_ids = gen_ids[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            if "<asr_text>" in response:
                response = response.split("<asr_text>")[-1]

        return response.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def main():
    model, processor, device = load_model()

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

        # Transcribe
        audio_path = request.get("audio_path", "")
        language = request.get("language", "vi")

        if not audio_path or not os.path.exists(audio_path):
            print(json.dumps({"error": f"audio file not found: {audio_path}"}), flush=True)
            continue

        text = transcribe(model, processor, device, audio_path, language)
        print(json.dumps({"text": text}), flush=True)


if __name__ == "__main__":
    main()
