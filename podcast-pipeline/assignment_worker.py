"""Speaker-assignment worker: Silero VAD and WeSpeaker in their own process.

Scoring a separated window needs four probes cleaned by Silero VAD and embedded by
WeSpeaker ResNet293. Run inside the main process they share one interpreter (and the
VAD one lock), so they queue up behind each other and behind the ordered consumer
that waits for them. Here each worker owns a copy of both models on one GPU and the
pool (services/worker_pool_service.py) hands it whichever probe is next, so probes
from several windows and several files are processed at once.

Line-JSON on stdio (see services/base_worker_service.py); audio moves as .npy files.
  {"cmd": "embed",       "audio_path", "sr"}                     -> {"embedding": [...]}
  {"cmd": "probe",       "audio_path", "sr", "out_path", ...}    -> {"probe_path": path | null}
  {"cmd": "probe_embed", "audio_path", "sr", ...}                -> {"embedding": [...] | null}
`...` = floor_db, min_voiced_sec, abs_floor_rms of BssSeparator._probe_from_segment.
The logic is BssSeparator's own, so results match the in-process path.
"""

import argparse
import json
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_separator(device, threads=None):
    """A BssSeparator with just the two models this worker needs."""
    import torch
    from models.bss_model import BssSeparator
    from models.silero_vad import SileroVAD
    from models.wespeaker_embedding import WeSpeakerONNXEmbedder

    sep = object.__new__(BssSeparator)
    sep.device = torch.device(device)
    sep._assignment = None
    sep.speaker_embedder = WeSpeakerONNXEmbedder(device=device, threads=threads)
    sep._vad_lock = threading.Lock()
    sep._vad = None
    try:
        sep._vad = SileroVAD(device=sep.device)
    except Exception as exc:                       # same fallback as BssSeparator
        print(f"[assignment_worker] Silero VAD unavailable ({exc}); energy gate",
              file=sys.stderr, flush=True)
    return sep


def handle_request(sep, req: dict) -> dict:
    req_id = req.get("id", "unknown")
    try:
        cmd = req.get("cmd")
        audio = np.load(req["audio_path"])
        sr = int(req["sr"])
        if cmd == "embed":
            return {"id": req_id,
                    "embedding": sep._get_embedding(audio, sr).detach().cpu().tolist()}
        if cmd not in ("probe", "probe_embed"):
            return {"id": req_id, "error": f"unknown command: {cmd}"}
        options = {key: req[key] for key in ("floor_db", "min_voiced_sec", "abs_floor_rms")
                   if req.get(key) is not None}
        probe = sep._probe_from_segment(audio, sr, **options)
        if cmd == "probe":
            if probe is None:
                return {"id": req_id, "probe_path": None}
            np.save(req["out_path"], np.asarray(probe, dtype=np.float32))
            return {"id": req_id, "probe_path": req["out_path"]}
        if probe is None:
            return {"id": req_id, "embedding": None}
        return {"id": req_id,
                "embedding": sep._get_embedding(probe, sr).detach().cpu().tolist()}
    except Exception as exc:
        return {"id": req_id, "error": f"{type(exc).__name__}: {exc}"}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def serve():
    args = _parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    # Library code prints to stdout; keep the real one for protocol only.
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def emit(message):
        print(json.dumps(message), file=protocol, flush=True)

    try:
        sep = build_separator(args.device, args.threads)
        # The ONNX session is built on first use; do it now so the first real
        # request is not the slow one.
        sep._get_embedding(np.zeros(16000, dtype=np.float32), 16000)
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
        emit(handle_request(sep, req))


if __name__ == "__main__":
    serve()
