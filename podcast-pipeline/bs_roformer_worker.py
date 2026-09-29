"""BS-RoFormer worker: one separator instance in its own process.

Speaks line-JSON on stdio (see services/base_worker_service.py) and moves audio
as .npy files, like the Sidon worker. Each GPU's separator gets its own
process, so its copy of PyTorch's process-global SDPA backend flags cannot be
corrupted by another separator: audio-separator wraps attention in
sdpa_kernel(), and two threads of one process can leave it stuck (see
utils/sdpa_guard.py).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class _StderrLogger:
    """Diagnostics go to stderr, which the parent drains; stdout is protocol only."""

    def _emit(self, level, message):
        print(f"[BSRoformerWorker:{level}] {message}", file=sys.stderr, flush=True)

    def debug(self, message, **_):
        pass

    def info(self, message, **_):
        self._emit("info", message)

    def warning(self, message, **_):
        self._emit("warning", message)

    def error(self, message, **_):
        self._emit("error", message)


def handle_request(remover, req: dict) -> dict:
    req_id = req.get("id", "unknown")
    audio_path = req.get("audio_path")
    if not audio_path:
        return {"id": req_id, "error": "Missing audio_path"}
    try:
        audio = np.load(audio_path)
        raw = remover.separate_raw(audio, int(req.get("sample_rate", 44100)))
        if raw is None:
            return {"id": req_id, "result": None}
        out, out_sr, stereo_in = raw
        out_path = os.path.splitext(audio_path)[0] + "_out.npy"
        np.save(out_path, np.asarray(out, dtype=np.float32))
        return {"id": req_id, "out_path": out_path,
                "out_sr": int(out_sr), "stereo_in": bool(stereo_in)}
    except Exception as exc:
        return {"id": req_id, "error": f"{type(exc).__name__}: {exc}"}


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="")
    parser.add_argument("--kwargs", default="{}")
    args = parser.parse_args()

    # Library code (audio-separator prints "A100 GPU detected...") writes to
    # stdout; keep the real stdout for protocol lines only.
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def emit(message):
        print(json.dumps(message), file=protocol, flush=True)

    try:
        from models.bs_roformer import BSRoformerRemover
        remover = BSRoformerRemover(
            device=args.device or None, logger=_StderrLogger(), **json.loads(args.kwargs))
        remover._get_model()
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
        emit(handle_request(remover, req))


if __name__ == "__main__":
    serve()
