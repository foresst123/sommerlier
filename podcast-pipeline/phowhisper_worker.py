#!/usr/bin/env python3
"""PhoWhisper (CTranslate2) in its own process.

Run inside the main process, PhoWhisper shares one interpreter with everything else
the pipeline does -- ROVER, progress reporting, result collection -- and one card
with whatever else the scheduler put there. As a worker it can be started next to
the other ASR models, given a GPU of its own, and run as several copies that pull
from one queue when it is the slowest model.

Line-JSON on stdio (see services/base_worker_service.py). A batch travels as one
.npy holding the clips end to end, plus their lengths:
  {"cmd": "transcribe_batch", "id", "audio_path", "lengths": [...], "ids": [...],
   "batch_size"}  ->  {"id", "results": [{"id", "text"}, ...]}
  {"cmd": "ping"} -> {"status": "ok"}     {"cmd": "quit"} -> {"status": "shutdown"}
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_model(config_path, env_name):
    import torch
    from models.phowhisper import PhoWhisperASR
    from utils.asr_model_config import phowhisper_kwargs

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    models_cfg = config.get("environments", {}).get(env_name, {}).get("models", {})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return PhoWhisperASR(device=device, **phowhisper_kwargs(models_cfg))


class _StderrLog:
    """The model reports OOM retries through a logger; stderr is what the parent
    forwards (lines that say "out of memory" reach its log as warnings)."""

    @staticmethod
    def warning(message):
        print(message, file=sys.stderr, flush=True)


def handle_request(model, req: dict) -> dict:
    req_id = req.get("id", "unknown")
    try:
        cmd = req.get("cmd")
        if cmd != "transcribe_batch":
            return {"id": req_id, "error": f"unknown command: {cmd}"}
        path = req.get("audio_path", "")
        if not path or not os.path.exists(path):
            return {"id": req_id, "error": f"audio file not found: {path}"}
        lengths = [int(n) for n in req.get("lengths") or []]
        ids = list(req.get("ids") or [])
        carrier = np.load(path)
        if len(ids) != len(lengths) or sum(lengths) != len(carrier):
            return {"id": req_id, "error": (
                f"batch does not add up: {len(ids)} ids, {len(lengths)} lengths "
                f"summing to {sum(lengths)} samples, carrier has {len(carrier)}")}
        clips, offset = [], 0
        for length in lengths:
            clips.append(carrier[offset:offset + length])
            offset += length
        texts = model.transcribe_batch(clips, batch_size=req.get("batch_size"),
                                       logger=_StderrLog())
        return {"id": req_id,
                "results": [{"id": str(i), "text": t} for i, t in zip(ids, texts)]}
    except Exception as exc:
        return {"id": req_id, "error": f"{type(exc).__name__}: {exc}"}


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--env", default="a100")
    args = parser.parse_args()

    # Library code prints to stdout; keep the real one for the protocol only.
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def emit(message):
        print(json.dumps(message), file=protocol, flush=True)

    try:
        model = build_model(args.config, args.env)
    except Exception as exc:
        emit({"status": "error", "message": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)
    emit({"status": "ready", "model": "PhoWhisper-large-ct2"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        cmd = req.get("cmd")
        if cmd == "quit":
            emit({"status": "shutdown"})
            break
        if cmd == "ping":
            emit({"status": "ok"})
            continue
        emit(handle_request(model, req))


if __name__ == "__main__":
    serve()
