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


def install_torch_load_shim():
    """Force torch.load back to weights_only=False for pickled checkpoints.

    Duplicated from utils/torch_compat.py on purpose: this worker runs in its
    own virtualenv, and putting podcast-pipeline on sys.path to share the helper
    would shadow any 'utils'/'models' package DiariZen depends on.
    """
    if hasattr(torch, "torch_version") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

    original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _patched_load


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
        ratio = float(diar_cfg["segmentation_step"])
        segmentation = getattr(pipeline, "_segmentation", None)
        if segmentation is not None and hasattr(segmentation, "step"):
            # pyannote's Inference.step is a hop in SECONDS (its own default is
            # 0.1 * duration), while DiariZen's config expresses it as a ratio of
            # the window. Writing the ratio straight in gives a 0.05s hop across
            # a 16s window -- 99.7% overlap and ~32x the chunks for no gain.
            window = float(getattr(segmentation, "duration", 0.0) or 0.0)
            if window > 0.0 and 0.0 < ratio <= 1.0:
                step_seconds = ratio * window
                segmentation.step = step_seconds
                applied.append(
                    f"segmentation_step={ratio} -> {step_seconds:.2f}s hop over {window:.0f}s window"
                )
            else:
                # Either the window is unknown or the value is already seconds.
                segmentation.step = ratio
                applied.append(f"segmentation_step={ratio}s (absolute)")
        else:
            ignored.append("segmentation_step (pipeline._segmentation.step absent)")

    if "seg_duration" in diar_cfg:
        # seg_duration is consumed by from_pretrained/instantiate; mutating the
        # already-built sliding window here would desync the segmentation model.
        ignored.append("seg_duration (constructor-time only, cannot be set post-load)")

    # DiariZenPipeline.__init__ copies these out of its config onto itself
    # (self.min_speakers / self.max_speakers feed the clustering call, and
    # self.apply_median_filtering gates the median filter over segmentations),
    # so they are set on the pipeline, not on pipeline.clustering.
    clustering = getattr(pipeline, "clustering", None)
    pipeline_attrs = ("min_speakers", "max_speakers", "apply_median_filtering")
    for cfg_key in pipeline_attrs:
        if cfg_key not in diar_cfg:
            continue
        value = diar_cfg[cfg_key]
        if hasattr(pipeline, cfg_key):
            setattr(pipeline, cfg_key, value)
            applied.append(f"{cfg_key}={value}")
        else:
            ignored.append(f"{cfg_key} (pipeline has no such attribute)")

    # ahc_threshold lives on the clustering object itself.
    if "ahc_threshold" in diar_cfg:
        value = diar_cfg["ahc_threshold"]
        attr = next(
            (n for n in ("ahc_threshold", "threshold")
             if clustering is not None and hasattr(clustering, n)),
            None,
        )
        if attr is not None:
            setattr(clustering, attr, value)
            applied.append(f"clustering.{attr}={value}")
        else:
            ignored.append("ahc_threshold (no matching attribute on pipeline.clustering)")

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

    # A too-small hop is not an error, just ruinously slow, so say so up front
    # rather than letting the user discover it from the chunk counter.
    segmentation = getattr(pipeline, "_segmentation", None)
    step = float(getattr(segmentation, "step", 0.0) or 0.0)
    window = float(getattr(segmentation, "duration", 0.0) or 0.0)
    if step > 0.0 and window > 0.0 and step < 0.05 * window:
        print(json.dumps({
            "warning": f"segmentation hop {step:.2f}s over a {window:.0f}s window is "
                       f"{100 * (1 - step / window):.1f}% overlap; expect very slow "
                       "segmentation. Values around 0.1 (10%) are typical."
        }), flush=True)

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
        # DiariZen takes speaker bounds through its clustering config, not as
        # call arguments, and rejects them with assorted exception types, so a
        # failed attempt must never cost us the diarization itself.
        kwargs = {k: v for k, v in (speaker_bounds or {}).items() if v is not None}
        if kwargs:
            try:
                diar_out = pipeline(audio_path, **kwargs)
            except Exception as e:
                print(json.dumps({
                    "warning": f"Pipeline rejected speaker bounds {sorted(kwargs)} "
                               f"({type(e).__name__}: {e}); running unbounded."
                }), flush=True)
                diar_out = pipeline(audio_path)
        else:
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
