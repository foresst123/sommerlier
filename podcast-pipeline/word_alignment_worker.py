"""Forced-alignment worker: one Wav2Vec2 alignment model in its own process.

Speaks line-JSON on stdio (see services/base_worker_service.py) and moves audio
as one packed .npy per request. WhisperX's alignment does its CTC trellis and
backtracking in Python on the CPU, which holds the GIL, so several batches only
run in parallel when they run in separate processes. Each worker keeps its own
small model on one GPU and the pool hands it whichever batch is next.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.word_alignment_service import align_prepared  # noqa: E402


def handle_request(model, metadata, device, interpolate_method, req: dict) -> dict:
    req_id = req.get("id", "unknown")
    try:
        audio = np.load(req["audio_path"])
        prepared, cursor = [], 0
        for item in req["items"]:
            length = int(item["n"])
            prepared.append({
                "index": item["index"], "start": item["start"], "end": item["end"],
                "text": item["text"], "audio": audio[cursor:cursor + length],
            })
            cursor += length
        return {"id": req_id, "words_by_index": align_prepared(
            prepared, model, metadata, device, interpolate_method)}
    except Exception as exc:
        return {"id": req_id, "error": f"{type(exc).__name__}: {exc}"}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="vi")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--interpolate", default="nearest")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def serve():
    args = _parse_args()
    # Several workers share the CPU: cap each one's math threads before torch loads.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    # Library code may print to stdout; keep the real stdout for protocol only.
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def emit(message):
        print(json.dumps(message), file=protocol, flush=True)

    try:
        import torch
        torch.set_num_threads(max(1, args.threads))
        import whisperx
        model, metadata = whisperx.load_align_model(
            language_code=args.language, device=args.device,
            model_name=args.model_name, model_dir=args.model_dir,
            model_cache_only=args.cache_only)
    except Exception as exc:
        emit({"status": "error", "message": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)
    emit({"status": "ready", "device": args.device})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        emit(handle_request(model, metadata, args.device, args.interpolate, req))


if __name__ == "__main__":
    serve()
