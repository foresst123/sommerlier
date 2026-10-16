#!/usr/bin/env python3
"""
Nemotron-3-Diarization Worker -- runs in its own environment (NeMo) on one GPU.

Usage:
    CUDA_VISIBLE_DEVICES=1 /path/to/nemotron_env/bin/python nemotron_worker.py

Protocol (stdin/stdout, line-delimited), identical to diarizen_worker.py:
    Input:  JSON line  {"audio_path": "/tmp/seg_xxx.wav"}
    Output: JSON line  {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}, ...]}

    Send  {"cmd": "quit"}  to shut down, {"cmd": "ping"} for {"status": "ok"}.

UNVERIFIED: written without a GPU or NeMo; see doc/nemotron_diarization_experiment.md.
"""

import argparse
import json
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

MODEL_ID = "nvidia/Nemotron-3-Diarization"


def _load_nemotron_config(config_path, env_name):
    """Read models.nemotron (optional tuning keys) out of the environment profile."""
    if not (config_path and os.path.exists(config_path)):
        return {}
    try:
        with open(config_path, encoding="utf-8") as handle:
            profile = json.load(handle).get("environments", {}).get(env_name, {})
        return profile.get("models", {}).get("nemotron", {})
    except Exception as e:
        print(json.dumps({"error": f"Failed to read config: {e}"}), flush=True)
        return {}


def load_model(config_path=None, env_name="kaggle"):
    # Imported here so --help and the conversion helpers work without NeMo.
    from nemo.collections.asr.models import SortformerEncLabelModel

    print(json.dumps({"status": "loading", "model": MODEL_ID}), flush=True)
    # CUDA_VISIBLE_DEVICES remaps the assigned GPU to cuda:0.
    model = SortformerEncLabelModel.from_pretrained(MODEL_ID)
    import torch
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    cfg = _load_nemotron_config(config_path, env_name)
    # Streaming parameters are in 80 ms frames (model card); only set on request.
    if "chunk_len" in cfg:
        model.sortformer_modules.chunk_len = int(cfg["chunk_len"])
        model._check_streaming_parameters()

    print(json.dumps({"status": "ready", "config_applied": sorted(cfg)}), flush=True)
    return model, cfg


def diarize(model, audio_path, cfg):
    """Run inference on one file and return pipeline segments or an [ERROR] string."""
    # utils/nemotron_segments.py is loaded by path: putting podcast-pipeline on
    # sys.path would shadow NeMo's own 'utils' package (same reason as the
    # DiariZen worker's duplicated torch shim).
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "utils", "nemotron_segments.py")

    try:
        spec = importlib.util.spec_from_file_location("nemotron_segments", path)
        convert = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(convert)
        predicted = model.diarize(audio=[audio_path], batch_size=1)
        return convert.to_pipeline_segments(
            predicted[0],
            merge_gap=float(cfg.get("merge_gap", 0.0)),
            min_duration=float(cfg.get("min_duration", 0.0)),
        )
    except Exception as e:
        cause = e.__cause__ or e.__context__ or e
        return (f"[ERROR] {type(cause).__name__}: {cause}\n"
                + "".join(traceback.format_exception(type(cause), cause, cause.__traceback__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--env", default="kaggle")
    args = parser.parse_args()

    try:
        model, cfg = load_model(args.config, args.env)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {e}"}), flush=True)
        sys.exit(1)

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
        if cmd == "ping":
            print(json.dumps({"status": "ok"}), flush=True)
            continue

        audio_path = request.get("audio_path", "")
        if not audio_path or not os.path.exists(audio_path):
            print(json.dumps({"error": f"audio file not found: {audio_path}"}), flush=True)
            continue

        # num/min/max_speakers are ignored: the model has a fixed 8-slot output.
        segments = diarize(model, audio_path, cfg)
        if isinstance(segments, str) and segments.startswith("[ERROR]"):
            print(json.dumps({"error": segments}), flush=True)
        else:
            print(json.dumps({"segments": segments}), flush=True)


if __name__ == "__main__":
    main()
