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
  {"cmd": "embed_batch" | "probe_embed_batch", "audio_path", "sr", "lengths", ...}
      -> {"results": [{"embedding": [...] | null} | {"error": ...}], "batch_stats": ...}
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


CUDA_FALLBACK_NOTE = "CUDA was requested but WeSpeaker runs on the CPU"


def describe_embedder(sep, device, threads) -> str:
    """One line saying where WeSpeaker really runs, and how long one embed takes.

    ONNX Runtime falls back to the CPU without failing when CUDA cannot load, and
    nothing else reports it, so a slow speaker-assignment stage looked the same
    either way.
    """
    import time
    embedder = sep.speaker_embedder
    providers = embedder.active_providers()
    audio = np.zeros(3 * 16000, dtype=np.float32)
    started = time.perf_counter()
    embedder.embed(audio, 16000)
    millis = (time.perf_counter() - started) * 1000.0
    on_gpu = bool(providers) and providers[0] == "CUDAExecutionProvider"
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    line = (f"[assignment_worker] WeSpeaker requested={device} "
            f"running_on={'GPU' if on_gpu else 'CPU'} providers={providers} "
            f"embed(3s)={millis:.1f}ms threads={threads} usable_cpus={cpus}")
    if str(device).startswith("cuda") and not on_gpu:
        line += f" -- {CUDA_FALLBACK_NOTE}"
    return line


def handle_request(sep, req: dict) -> dict:
    req_id = req.get("id", "unknown")
    try:
        cmd = req.get("cmd")
        audio = np.load(req["audio_path"])
        sr = int(req["sr"])
        if cmd in ("embed_batch", "probe_embed_batch"):
            return _handle_batch(sep, req, audio, sr)
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


def _handle_batch(sep, req, audio, sr):
    lengths = req.get("lengths", [])
    if (audio.ndim != 1 or not lengths or len(lengths) > 32
            or any(type(n) is not int or n < 0 for n in lengths)
            or sum(lengths) != len(audio)):
        raise ValueError("invalid packed assignment batch lengths")
    options = {key: req[key] for key in ("floor_db", "min_voiced_sec", "abs_floor_rms")
               if req.get(key) is not None}
    results = [{"embedding": None} for _ in lengths]
    probes, indices, offset = [], [], 0
    for index, length in enumerate(lengths):
        segment = audio[offset:offset + length]
        offset += length
        try:
            probe = (sep._probe_from_segment(segment, sr, **options)
                     if req["cmd"] == "probe_embed_batch" else segment)
            if probe is not None:
                probes.append(probe)
                indices.append(index)
        except Exception as exc:
            results[index] = {"error": f"{type(exc).__name__}: {exc}"}
    before = dict(sep.speaker_embedder.batch_stats)
    if probes:
        values = sep._get_embeddings(probes, sr)
        if len(values) != len(indices):
            raise RuntimeError("assignment embedding result count mismatch")
        for index, value in zip(indices, values):
            results[index] = ({"error": f"{type(value).__name__}: {value}"}
                              if isinstance(value, Exception) else
                              {"embedding": value.detach().cpu().tolist()})
    stats = {key: value - before.get(key, 0)
             for key, value in sep.speaker_embedder.batch_stats.items()}
    return {"id": req.get("id", "unknown"), "results": results, "batch_stats": stats}


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
        report = describe_embedder(sep, args.device, args.threads)
    except Exception as exc:
        emit({"status": "error", "message": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)
    # The parent logs this; stderr lines without an error word never reach its log.
    emit({"status": "ready", "device": args.device, "info": report,
          "warning": CUDA_FALLBACK_NOTE in report})

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
