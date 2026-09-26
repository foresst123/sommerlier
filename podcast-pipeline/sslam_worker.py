"""SSLAM tagger worker: one process, one GPU, one sweep at a time.

Line-JSON on stdio (see services/base_worker_service.py); audio and scores move as
files, so a request stays a few bytes however long the recording is:

  {"id", "cmd": "tag", "audio_path": "<x>.npy", "sample_rate": 24000}
      -> {"id", "scores_path": "<x>_scores.npz", "fps": 2.0, "worker_seconds": 12.3}
         | {"id", "error": "..."}

Why a process of its own rather than a thread: each worker gets its own CUDA
context and its own interpreter, so N of them sweep N recordings truly in parallel
-- the resampling and mel features are CPU work that threads would take turns on
under one GIL -- and the sweeps of every file can spread over both cards with no
lock between them. The parent leases an idle worker per request
(services/worker_pool_service.py), so a file waits for whichever worker frees up first.

`--device` is the device as this process sees it: the parent masks
CUDA_VISIBLE_DEVICES to one physical card, so that card is always cuda:0 here.
"""
import argparse
import json
import sys
import time
import traceback


def handle(detector, req: dict) -> dict:
    """Answer one request. Never raises: a bad recording must not end the worker."""
    import numpy as np

    req_id = req.get("id", "unknown")
    cmd = req.get("cmd", "tag")
    if cmd != "tag":
        return {"id": req_id, "error": f"unknown command {cmd!r}"}
    audio_path = req.get("audio_path")
    if not audio_path:
        return {"id": req_id, "error": "Missing audio_path"}
    started = time.perf_counter()
    try:
        audio = np.load(audio_path)
        scores, fps = detector.tag_framewise(audio, int(req.get("sample_rate", 16000)))
        stem = audio_path[:-4] if audio_path.endswith(".npy") else audio_path
        out_path = stem + "_scores.npz"
        np.savez(out_path, **{name: np.asarray(curve) for name, curve in scores.items()})
        return {"id": req_id, "scores_path": out_path, "fps": float(fps),
                "worker_seconds": time.perf_counter() - started}
    except Exception as exc:
        return {"id": req_id, "error": f"{type(exc).__name__}: {exc}"}


def serve(detector, stdin=None, stdout=None) -> None:
    """Answer requests, one line in and one line out, until stdin closes."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    while True:
        try:
            line = stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            reply = handle(detector, json.loads(line))
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()
        except Exception:
            traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    try:
        from models.sslam import SSLAMDetector
        detector = SSLAMDetector(device=args.device)
        # Loaded now, so that "ready" means a sweep can start and the first request
        # does not pay for the checkpoint.
        detector._load()
    except Exception as exc:
        print(json.dumps({"status": "error", "message": f"{type(exc).__name__}: {exc}"}),
              flush=True)
        sys.exit(1)
    print(json.dumps({"status": "ready", "device": args.device}), flush=True)
    serve(detector)


if __name__ == "__main__":
    main()
