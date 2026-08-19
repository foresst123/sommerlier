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

# PyTorch 2.6+ defaults to weights_only=True, which breaks Pyannote/DiariZen models containing TorchVersion
if hasattr(torch, "torch_version") and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    
# Alternatively, we can monkey-patch torch.load to always use weights_only=False
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

def load_model(config_path=None, env_name="kaggle"):
    """Load DiariZen WavLM-Large s80-md-v2 with custom config."""
    from diarizen.pipelines.inference import DiariZenPipeline
    
    device = torch.device("cuda:0") # CUDA_VISIBLE_DEVICES remaps physical GPU → cuda:0

    print(json.dumps({"status": "loading", "model": "BUT-FIT/diarizen-wavlm-large-s80-md-v2"}), flush=True)

    pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            diar_cfg = cfg.get("environments", {}).get(env_name, {}).get("models", {}).get("diarizen", {})
            
            # Apply batch size dynamically if present
            if "batch_size" in diar_cfg:
                bs = int(diar_cfg["batch_size"])
                if hasattr(pipeline, 'segmentation_batch_size'):
                    pipeline.segmentation_batch_size = bs
                if hasattr(pipeline, 'embedding_batch_size'):
                    pipeline.embedding_batch_size = bs
                    
            # Overwrite other internal config variables if needed without calling instantiate()
            if hasattr(pipeline, '_segmentation') and hasattr(pipeline._segmentation, 'step'):
                pipeline._segmentation.step = diar_cfg.get("segmentation_step", 0.1)
                
        except Exception as e:
            print(json.dumps({"error": f"Failed to apply config overrides: {str(e)}"}), flush=True)

    pipeline.to(device)

    # Monkey-patch get_segmentations to emit JSON progress updates
    import types
    original_get_segmentations = pipeline.get_segmentations
    def patched_get_segmentations(self_obj, file, hook=None, soft=False):
        def custom_hook(step_name, step_details, completed=None, total=None):
            if completed is not None and total is not None:
                # To avoid spamming, only print every 10 chunks or the last chunk
                if completed == total or completed % 10 == 0:
                    print(json.dumps({"progress": f"DiariZen Progress: {completed}/{total} chunks"}), flush=True)
        return original_get_segmentations(file, hook=custom_hook, soft=soft)
    
    pipeline.get_segmentations = types.MethodType(patched_get_segmentations, pipeline)

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
