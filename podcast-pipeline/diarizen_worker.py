#!/usr/bin/env python3
"""
DiariZen Worker — runs in a separate environment (diarizen_env) on GPU 1.

Usage:
    CUDA_VISIBLE_DEVICES=1 /path/to/diarizen_env/bin/python diarizen_worker.py

Protocol (stdin/stdout, line-delimited):
    Input:  JSON line  {"audio_path": "/tmp/seg_xxx.wav"}
    Output: JSON line  {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}, ...]}
    
    Send  {"cmd": "quit"}  to shut down the worker.
    Send  {"cmd": "ping"}  to check if the worker is alive → returns {"status": "ok"}
"""

import sys
import json
import os
import torch
import soundfile as sf
import warnings
import argparse

warnings.filterwarnings("ignore")

def load_model(config_path=None, env_name="kaggle"):
    """Load DiariZen WavLM-Large s80-md-v2 with custom config."""
    from diarizen.pipelines.inference import DiariZenPipeline
    
    device = torch.device("cuda:0") # CUDA_VISIBLE_DEVICES remaps physical GPU → cuda:0

    print(json.dumps({"status": "loading", "model": "BUT-FIT/diarizen-wavlm-large-s80-md-v2"}), flush=True)

    config_parse = None
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            diar_cfg = cfg.get("environments", {}).get(env_name, {}).get("models", {}).get("diarizen", {})
            if diar_cfg:
                config_parse = {
                    "inference": {"args": diar_cfg},
                    "clustering": {"args": {"method": diar_cfg.get("clustering_method", "VBxClustering"), **diar_cfg}}
                }
        except Exception as e:
            print(json.dumps({"error": f"Failed to parse config: {str(e)}"}), flush=True)

    pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    if config_parse:
        pipeline.instantiate(config_parse)
    pipeline.to(device)

    print(json.dumps({"status": "ready", "device": str(device)}), flush=True)
    return pipeline, device


def diarize(pipeline, audio_path):
    """Run DiariZen inference on an audio file."""
    try:
        # Pass the audio path directly to the pipeline. 
        # Pyannote/DiariZen handles loading and resampling internally.
        diar_out = pipeline(audio_path)
        
        annotation = (
            diar_out.speaker_diarization
            if hasattr(diar_out, "speaker_diarization")
            else diar_out
        )
        
        segments = []
        if annotation is not None:
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker)
                })
                
        return segments
    except Exception as e:
        return f"[ERROR] {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--env", default="kaggle")
    args = parser.parse_args()

    try:
        pipeline, device = load_model(args.config, args.env)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {str(e)}"}), flush=True)
        sys.exit(1)

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

        cmd = request.get("cmd", "diarize")

        if cmd == "quit":
            print(json.dumps({"status": "shutdown"}), flush=True)
            break
        elif cmd == "ping":
            print(json.dumps({"status": "ok"}), flush=True)
            continue

        # Diarize
        audio_path = request.get("audio_path", "")

        if not audio_path or not os.path.exists(audio_path):
            print(json.dumps({"error": f"audio file not found: {audio_path}"}), flush=True)
            continue

        segments = diarize(pipeline, audio_path)
        if isinstance(segments, str) and segments.startswith("[ERROR]"):
            print(json.dumps({"error": segments}), flush=True)
        else:
            print(json.dumps({"segments": segments}), flush=True)

if __name__ == "__main__":
    main()
