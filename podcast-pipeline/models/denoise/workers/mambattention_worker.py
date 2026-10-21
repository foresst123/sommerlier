"""MambAttention worker: runs in the separate torch 2.2.2 + mamba_ssm environment (see scripts/denoise_mambattention.md).

    python mambattention_worker.py --in X.npy --out Y.npy --sr 16000 --cfg-json PATH

cfg keys: repo_dir (clone of NikolaiKyhne/MambAttention), checkpoint, config_yaml, chunk_s, overlap_s,
device, seed. Follows MambAttention/inference.py: whole-file norm_factor = sqrt(N / sum(x^2)),
compressed mag/phase STFT (n_fft 400, hop 100, win 400, centre), model, inverse STFT, / norm_factor.
The input is already at 16 kHz. UNTESTED against the real model (needs CUDA + mamba_ssm).
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def _load_chunking():
    """Load models/denoise/chunking.py by path: upstream also has a top-level `models` package."""
    spec = importlib.util.spec_from_file_location("_denoise_chunking", os.path.join(os.path.dirname(HERE), "chunking.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fail(msg: str):
    print(f"mambattention_worker: {msg}", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, required=True)
    ap.add_argument("--cfg-json", required=True)
    args = ap.parse_args()
    with open(args.cfg_json) as f:
        cfg = json.load(f)
    if args.sr != 16000:
        _fail(f"expects 16000 Hz input, got {args.sr}")
    chunking = _load_chunking()

    repo = cfg.get("repo_dir") or os.environ.get("MAMBATTENTION_DIR") or ""
    if not os.path.isfile(os.path.join(repo, "models", "generator.py")):
        _fail(f"repo_dir {repo!r} is not a MambAttention clone (models/generator.py missing)")
    ckpt = cfg.get("checkpoint") or os.path.join(repo, "checkpoints", "seed4321_DNS.pth")
    yaml_path = cfg.get("config_yaml") or os.path.join(repo, "checkpoints", "MambAttention_seed3441_VB-DemandEx.yaml")
    # our project root must not shadow upstream's `models` package
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != PROJECT_ROOT]
    sys.path.insert(0, repo)

    import torch
    import yaml
    from models.generator import MambAttention
    from models.stfts import mag_phase_stft, mag_phase_istft

    x = np.load(args.inp, allow_pickle=False).astype(np.float32)
    if x.ndim != 1:
        _fail(f"input must be 1-D, got {x.shape}")
    if not torch.cuda.is_available() and str(cfg.get("device", "auto")) in ("auto", "cuda"):
        _fail("CUDA is not available (upstream does not support CPU)")
    dev = torch.device("cuda" if cfg.get("device", "auto") == "auto" else cfg["device"])
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open(yaml_path) as f:
        mc = yaml.safe_load(f)
    n_fft, hop, win = (mc["stft_cfg"][k] for k in ("n_fft", "hop_size", "win_size"))
    comp = mc["model_cfg"]["compress_factor"]
    if int(mc["stft_cfg"]["sampling_rate"]) != 16000:
        _fail("config sampling_rate is not 16000")
    model = MambAttention(mc).to(dev)
    try:
        state = torch.load(ckpt, map_location=dev, weights_only=True)
    except Exception:
        state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state["generator"])
    model.eval()

    energy = float(np.sum(x.astype(np.float64) ** 2))
    if energy <= 0.0:
        y = x.copy()
    else:
        norm = float(np.sqrt(len(x) / energy))  # one factor for the whole file so chunks keep matching levels
        min_len = 2 * n_fft

        def run(chunk: np.ndarray) -> np.ndarray:
            n = len(chunk)
            if n < min_len:
                chunk = np.concatenate([chunk, np.zeros(min_len - n, dtype=np.float32)])
            with torch.no_grad():
                w = torch.from_numpy(np.ascontiguousarray(chunk * np.float32(norm))).unsqueeze(0).to(dev)
                amp, pha, _ = mag_phase_stft(w, n_fft, hop, win, comp)
                amp_g, pha_g, _ = model(amp, pha)
                out = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, comp) / norm
            return out.squeeze(0).float().cpu().numpy()[:n]

        chunk = int(float(cfg.get("chunk_s", 4.0)) * 16000)
        ov = int(float(cfg.get("overlap_s", 0.5)) * 16000)
        y = run(x) if chunk <= 0 else chunking.chunk_apply(x, run, chunk, ov)
    y = chunking.fit_length(np.asarray(y, dtype=np.float32), len(x))
    lag, score = chunking.estimate_delay(x, y, max_lag=320, probe=10 * 16000)
    print(f"mambattention_worker: measured lag {lag} samples (score {score:.2f})", file=sys.stderr)
    np.save(args.out, np.ascontiguousarray(y, dtype=np.float32))


if __name__ == "__main__":
    main()
