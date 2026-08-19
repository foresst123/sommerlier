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

# This worker runs under its own virtualenv, so the pipeline package is not on
# the path by default.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.torch_compat import install_torch_load_shim

install_torch_load_shim()

def _load_diarizen_config(config_path, env_name):
    """Read the diarizen block out of the environment profile."""
    if not (config_path and os.path.exists(config_path)):
        return {}
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        return cfg.get("environments", {}).get(env_name, {}).get("models", {}).get("diarizen", {})
    except Exception as e:
        print(json.dumps({"error": f"Failed to read config: {str(e)}"}), flush=True)
        return {}


def _apply_diarizen_config(pipeline, diar_cfg):
    """Apply the config keys DiariZen actually honours, reporting the rest.

    Keys the pipeline cannot express are echoed back as warnings instead of
    being silently dropped, so tuning a value that has no effect is visible.
    """
    applied, ignored = [], []

    if "batch_size" in diar_cfg:
        bs = int(diar_cfg["batch_size"])
        for attr in ("segmentation_batch_size", "embedding_batch_size"):
            if hasattr(pipeline, attr):
                setattr(pipeline, attr, bs)
                applied.append(f"{attr}={bs}")
            else:
                ignored.append(f"batch_size (pipeline has no {attr})")

    if "segmentation_step" in diar_cfg:
        step = float(diar_cfg["segmentation_step"])
        if hasattr(pipeline, "_segmentation") and hasattr(pipeline._segmentation, "step"):
            pipeline._segmentation.step = step
            applied.append(f"segmentation_step={step}")
        else:
            ignored.append("segmentation_step (pipeline._segmentation.step absent)")

    if "seg_duration" in diar_cfg:
        # seg_duration is consumed by from_pretrained/instantiate; mutating the
        # already-built sliding window here would desync the segmentation model.
        ignored.append("seg_duration (constructor-time only, cannot be set post-load)")

    # Clustering knobs live on pipeline.clustering when the chosen clusterer
    # exposes them; names differ between DiariZen releases, so probe rather than
    # assume, and say so when the probe fails.
    clustering = getattr(pipeline, "clustering", None)
    clustering_map = {
        "ahc_threshold": ("threshold",),
        "apply_median_filtering": ("apply_median_filtering",),
        "min_speakers": ("min_num_speakers", "min_speakers"),
        "max_speakers": ("max_num_speakers", "max_speakers"),
    }
    for cfg_key, attr_names in clustering_map.items():
        if cfg_key not in diar_cfg:
            continue
        value = diar_cfg[cfg_key]
        target = next((a for a in attr_names if clustering is not None and hasattr(clustering, a)), None)
        if target:
            setattr(clustering, target, value)
            applied.append(f"clustering.{target}={value}")
        else:
            ignored.append(f"{cfg_key} (no matching attribute on pipeline.clustering)")

    if "clustering_method" in diar_cfg:
        requested = str(diar_cfg["clustering_method"])
        actual = type(clustering).__name__ if clustering is not None else "unknown"
        if requested.lower() not in actual.lower():
            ignored.append(
                f"clustering_method={requested} (pipeline was built with {actual}; "
                "the clusterer is fixed at from_pretrained time)"
            )
        else:
            applied.append(f"clustering_method={actual}")

    if applied:
        print(json.dumps({"config_applied": applied}), flush=True)
    if ignored:
        print(json.dumps({"config_ignored": ignored}), flush=True)


def load_model(config_path=None, env_name="kaggle"):
    """Load DiariZen WavLM-Large s80-md-v2 with custom config."""
    from diarizen.pipelines.inference import DiariZenPipeline

    device = torch.device("cuda:0") # CUDA_VISIBLE_DEVICES remaps physical GPU → cuda:0

    print(json.dumps({"status": "loading", "model": "BUT-FIT/diarizen-wavlm-large-s80-md-v2"}), flush=True)

    pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")

    diar_cfg = _load_diarizen_config(config_path, env_name)
    if diar_cfg:
        try:
            _apply_diarizen_config(pipeline, diar_cfg)
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


def diarize(pipeline, audio_path, speaker_bounds=None):
    """Run DiariZen inference on an audio file."""
    try:
        # Pass the audio path directly to the pipeline.
        # Pyannote/DiariZen handles loading and resampling internally.
        # Speaker bounds are pipeline call arguments, not constructor ones; older
        # DiariZen builds reject them, so fall back to an unbounded call.
        kwargs = {k: v for k, v in (speaker_bounds or {}).items() if v is not None}
        try:
            diar_out = pipeline(audio_path, **kwargs) if kwargs else pipeline(audio_path)
        except TypeError as e:
            if kwargs:
                print(json.dumps({
                    "warning": f"Pipeline rejected speaker bounds {sorted(kwargs)}: {e}. "
                               "Running unbounded."
                }), flush=True)
                diar_out = pipeline(audio_path)
            else:
                raise
        
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

        speaker_bounds = {
            "num_speakers": request.get("num_speakers"),
            "min_speakers": request.get("min_speakers"),
            "max_speakers": request.get("max_speakers"),
        }
        segments = diarize(pipeline, audio_path, speaker_bounds)
        if isinstance(segments, str) and segments.startswith("[ERROR]"):
            print(json.dumps({"error": segments}), flush=True)
        else:
            print(json.dumps({"segments": segments}), flush=True)

if __name__ == "__main__":
    main()
