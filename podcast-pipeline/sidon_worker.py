import os
import sys
import json

# Phải set trước khi torch / numpy / torchaudio được import.
CPU_THREADS = int(os.environ.get("SIDON_CPU_THREADS", "8"))

os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("TORCH_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

# pyrefly: ignore [missing-import]
import torch
import numpy as np
import traceback
import argparse
import torchaudio.functional as F_audio

try:
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Có thể đã bị set trước đó trong một vài runtime; bỏ qua để worker vẫn chạy.
    pass

# PyTorch 2.6+ defaults to weights_only=True, which can break loading checkpoints containing TorchVersion
if hasattr(torch, "torch_version") and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

_original_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)

torch.load = _patched_load


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--env", type=str, default="kaggle")
    args = parser.parse_args()

    device = torch.device(args.device)

    # In ra để kiểm tra worker thật sự nhận bao nhiêu CPU thread.
    print(json.dumps({
        "status": "cpu_threads",
        "sidon_cpu_threads": CPU_THREADS,
        "torch_num_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        "torch_env_threads": os.environ.get("TORCH_NUM_THREADS"),
        "device": str(device),
    }), flush=True)

    num_steps = 60
    try:
        with open(args.config, encoding="utf-8") as fh:
            _cfg = json.load(fh)
        _sidon = (_cfg.get("environments", {}).get(args.env, {})
                      .get("models", {}).get("sidon", {}))
        num_steps = int(_sidon.get("num_steps", num_steps))
    except Exception as e:
        print(json.dumps({
            "warning": f"could not read sidon config: {e}; using num_steps={num_steps}"
        }), flush=True)

    # Fix SpeechBrain 1.1.0 logger bug where it tries to setLevel("20")
    import logging
    logging.addLevelName(10, "10")
    logging.addLevelName(20, "20")
    logging.addLevelName(30, "30")
    logging.addLevelName(40, "40")
    logging.addLevelName(50, "50")

    try:
        from sidon_infer import run_separation_chunked, load_models
    except ImportError as e:
        print(json.dumps({
            "status": "error",
            "message": f"Sidon not installed or import failed: {str(e)}"
        }), flush=True)
        sys.exit(1)

    try:
        load_models(device)
        print(json.dumps({
            "status": "ready",
            "num_steps": num_steps,
            "sidon_cpu_threads": CPU_THREADS,
            "torch_num_threads": torch.get_num_threads(),
            "device": str(device),
        }), flush=True)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e)
        }), flush=True)
        sys.exit(1)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            req = json.loads(line)
            req_id = req.get("id", "unknown")
            audio_path = req.get("audio_path")
            sample_rate = req.get("sample_rate", 16000)

            if not audio_path:
                print(json.dumps({
                    "id": req_id,
                    "error": "Missing audio_path"
                }), flush=True)
                continue

            try:
                audio_np = np.load(audio_path)
                mix_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)

                with torch.inference_mode():
                    est_sources, out_sr = run_separation_chunked(
                        wav=mix_tensor,
                        sample_rate=sample_rate,
                        num_steps=num_steps,
                        device=device
                    )

                if est_sources.ndim == 2 and est_sources.shape[0] == 2:
                    track_1 = est_sources[0].detach().cpu().numpy()
                    track_2 = est_sources[1].detach().cpu().numpy()

                    raw_peak = max(np.abs(track_1).max(), np.abs(track_2).max())
                    print(
                        f"[SidonWorker] Raw output peak amplitude = {raw_peak:.6f}",
                        file=sys.stderr,
                        flush=True
                    )
                else:
                    raise ValueError(f"Unexpected DialogueSidon output shape: {est_sources.shape}")

                out_path_1 = audio_path.replace(".npy", "_t1.npy")
                out_path_2 = audio_path.replace(".npy", "_t2.npy")

                np.save(out_path_1, track_1)
                np.save(out_path_2, track_2)

                print(json.dumps({
                    "id": req_id,
                    "track_1_path": out_path_1,
                    "track_2_path": out_path_2,
                    "target_sr": int(out_sr)
                }), flush=True)

            except Exception as e:
                print(json.dumps({
                    "id": req_id,
                    "error": str(e)
                }), flush=True)

        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    serve()