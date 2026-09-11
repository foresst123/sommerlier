#!/usr/bin/env python3
"""Enhance identical selected windows for each matrix prefix, without rerunning stages."""

import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from services.json_worker_service import JsonWorkerService
from tools.matrix_axes import POLISH_SUFFIX, STAGE_DIR
from tools.matrix_io import read_json, write_json
from tools.metrics.audio import read_audio, selected_windows
from utils.worker_env import resolve_worker_python


def polish_jobs(matrix, windows, backends):
    manifest = read_json(matrix / "manifest.json")
    if not manifest:
        raise ValueError("Matrix manifest is missing")
    seen = set()
    for cell in manifest["cells"]:
        backend = cell["axes"]["F"]
        if backend not in backends:
            continue
        for stem in manifest["inputs"]:
            directory = matrix / "out" / cell["jobs"]["separation"] / stem / STAGE_DIR["separation"] / "audio/raw/separated"
            for mix in selected_windows(directory, windows):
                for track in ("A", "B"):
                    path = Path(str(mix).removesuffix("_mix.wav") + f"_track{track}.wav")
                    key = (backend, str(path))
                    if key not in seen:
                        yield backend, path
                        seen.add(key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--env", default="a100")
    parser.add_argument("--backend", choices=("resemble", "audiosr"), action="append")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    matrix = args.matrix.resolve()
    config = read_json(args.config)
    profile = config.get("environments", {}).get(args.env, {})
    state_path = matrix / "polish_state.json"
    state = read_json(state_path, {})
    failed = 0
    jobs = list(polish_jobs(matrix, args.windows, args.backend or ("resemble", "audiosr")))
    for backend in args.backend or ("resemble", "audiosr"):
        worker = None
        try:
            for name, source in jobs:
                if name != backend:
                    continue
                target = source.with_name(source.stem + POLISH_SUFFIX[backend] + ".wav")
                key = str(target.relative_to(matrix))
                if target.is_file():
                    continue
                state[key] = dict(status="running")
                write_json(state_path, state)
                try:
                    if worker is None:
                        python = resolve_worker_python(backend, config, profile)
                        worker = JsonWorkerService(backend, python, str(ROOT / "tools/polish_worker.py"),
                            extra_args=["--backend", backend], device_id=args.gpu)
                    audio, sr = read_audio(source)
                    with tempfile.TemporaryDirectory(prefix="sommelier-polish-") as directory:
                        inp, out = Path(directory) / "in.npy", Path(directory) / "out.npy"
                        np.save(inp, audio, allow_pickle=False)
                        result = worker.request({"in": str(inp), "out": str(out), "sr": sr}, args.timeout)
                        enhanced = np.load(out, allow_pickle=False)
                    import soundfile as sf
                    temporary = target.with_suffix(".partial.wav")
                    sf.write(temporary, enhanced, result["sr"], subtype="FLOAT")
                    if result["sr"] != 24000:
                        import librosa
                        at24 = librosa.resample(enhanced, orig_sr=result["sr"], target_sr=24000)
                        comparison = target.with_name(target.stem + ".24k.wav")
                        sf.write(comparison, at24, 24000, subtype="FLOAT")
                    temporary.replace(target)
                    state[key] = dict(status="done", native_sr=result["sr"], seed=42)
                except Exception as exc:
                    state[key] = dict(status="failed", error=f"{type(exc).__name__}: {exc}")
                    failed += 1
                    if worker:
                        worker.stop()
                        worker = None
                write_json(state_path, state)
                print(f"{key}: {state[key]['status']}", flush=True)
        finally:
            if worker:
                worker.stop()
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
