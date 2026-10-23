"""RE-USE worker: runs in the separate environment that has mamba_ssm (see scripts/denoise_re_use.md).

    python re_use_worker.py --in X.npy --out Y.npy --sr 24000 --cfg-json PATH

cfg keys: repo_dir (the downloaded Hugging Face folder: models/, utils/, recipes/, model.safetensors),
recipe (yaml, default recipes/USEMamba_30x1_..._early_001.yaml), hf_repo (used only when repo_dir has no
model.safetensors), chunk_s, hop_portion, device, seed.

Follows the vendor's inference.py / inference_chunk.py: the STFT sizes in the recipe are for 8 kHz and are
scaled by sr / 8000, no level normalisation, compressed mag/phase STFT (centre), model, the "strange sweep
artifact" fix, inverse STFT. Long input is cut as in inference_chunk.py: Hann-weighted chunks, hop =
hop_portion * chunk, the last chunk aligned to the end. UNTESTED against the real model (needs CUDA +
mamba_ssm; the model has no CPU path upstream).
"""
import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DEFAULT_RECIPE = os.path.join("recipes", "USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml")
MIN_SR, MAX_SR = 8000, 48000


def make_even(value) -> int:
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def scaled_stft(n_fft: int, hop: int, win: int, base_sr: int, sr: int):
    """The recipe's STFT sizes (given for base_sr) at the file's own sample rate, as upstream does."""
    return tuple(make_even(v * sr // base_sr) for v in (n_fft, hop, win))


def overlap_add(x: np.ndarray, run, chunk: int, hop_portion: float) -> np.ndarray:
    """Upstream's chunk scheme. `run(seg)` returns an array of the same length as `seg`.

    One call when chunk <= 0 or the signal fits in one chunk (upstream does the same: its single chunk is
    the whole signal and the window weights cancel in the division). Otherwise Hann-weighted chunks every
    `hop_portion * chunk` samples, the last one taken from the end, summed and divided by the summed window.
    """
    n = len(x)
    if chunk <= 0 or n <= chunk:
        return np.asarray(run(x), dtype=np.float32)
    hop = max(1, int(hop_portion * chunk))
    win = np.maximum(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(chunk) / chunk), 1e-7)  # periodic Hann, clamped
    out = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    num = max(1, math.ceil((n - chunk) / hop) + 1)
    for i in range(num):
        lo = i * hop if i < num - 1 else n - chunk
        seg = x[lo:lo + chunk]
        y = np.asarray(run(seg), dtype=np.float64)
        if len(y) != len(seg):
            raise ValueError(f"run() returned {len(y)} samples for a {len(seg)}-sample chunk")
        w = win[:len(y)]
        out[lo:lo + len(y)] += y * w
        wsum[lo:lo + len(y)] += w
    return (out / wsum).astype(np.float32)


def _load_chunking():
    """Load models/denoise/chunking.py by path: upstream also has a top-level `models` package."""
    spec = importlib.util.spec_from_file_location("_denoise_chunking", os.path.join(os.path.dirname(HERE), "chunking.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fail(msg: str):
    print(f"re_use_worker: {msg}", file=sys.stderr)
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
    sr = int(args.sr)
    if not MIN_SR <= sr <= MAX_SR:
        _fail(f"supports {MIN_SR}-{MAX_SR} Hz input, got {sr}")

    repo = cfg.get("repo_dir") or os.environ.get("REUSE_DIR") or ""
    if not os.path.isfile(os.path.join(repo, "models", "generator_SEMamba_time_d4.py")):
        _fail(f"repo_dir {repo!r} is not a RE-USE download (models/generator_SEMamba_time_d4.py missing)")
    recipe = cfg.get("recipe") or os.path.join(repo, DEFAULT_RECIPE)
    if not os.path.isfile(recipe):
        _fail(f"recipe yaml not found: {recipe}")
    local_weights = os.path.isfile(os.path.join(repo, "model.safetensors"))
    source = repo if local_weights else cfg.get("hf_repo")
    if not source:
        _fail(f"no model.safetensors in {repo!r} and no hf_repo configured")
    chunking = _load_chunking()
    # our project root must not shadow upstream's `models` and `utils` packages
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != PROJECT_ROOT]
    sys.path.insert(0, repo)

    import torch
    import torch.nn as nn
    from models.generator_SEMamba_time_d4 import SEMamba
    from models.stfts import mag_phase_istft, mag_phase_stft
    from utils.util import load_config, pad_or_trim_to_match

    x = np.load(args.inp, allow_pickle=False).astype(np.float32)
    if x.ndim != 1:
        _fail(f"input must be 1-D, got {x.shape}")
    if not torch.cuda.is_available():
        _fail("CUDA is not available (upstream does not support CPU)")
    dev = torch.device("cuda" if str(cfg.get("device", "auto")) == "auto" else cfg["device"])
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    mc = load_config(recipe)
    n_fft, hop, win = scaled_stft(mc["stft_cfg"]["n_fft"], mc["stft_cfg"]["hop_size"], mc["stft_cfg"]["win_size"],
                                  int(mc["stft_cfg"]["sampling_rate"]), sr)
    comp = mc["model_cfg"]["compress_factor"]
    model = SEMamba.from_pretrained(source, cfg=mc).to(dev)
    model.eval()
    relu = nn.ReLU()
    min_len = 2 * n_fft

    def run(seg: np.ndarray) -> np.ndarray:
        n = len(seg)
        if n < min_len:
            seg = np.concatenate([seg, np.zeros(min_len - n, dtype=np.float32)])
        with torch.no_grad():
            w = torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0).to(dev)
            mag, pha, _ = mag_phase_stft(w, n_fft=n_fft, hop_size=hop, win_size=win, compress_factor=comp,
                                         center=True, addeps=False)
            amp_g, pha_g, _ = model(mag, pha)
            # upstream's fix for the "strange sweep artifact": silence frames that are mostly zero
            m = torch.expm1(relu(amp_g))
            zero_portion = torch.sum(m == 0, 1) / m.shape[1]
            amp_g[:, :, (zero_portion > 0.5)[0]] = 0
            out = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, comp)
            out = pad_or_trim_to_match(w.detach(), out, pad_value=1e-8)
        return out.squeeze(0).float().cpu().numpy()[:n]

    chunk = int(float(cfg.get("chunk_s", 5.0)) * sr)
    y = overlap_add(x, run, chunk, float(cfg.get("hop_portion", 0.5)))
    y = chunking.fit_length(np.asarray(y, dtype=np.float32), len(x))
    lag, score = chunking.estimate_delay(x, y, max_lag=max(1, int(0.02 * sr)), probe=10 * sr)
    print(f"re_use_worker: measured lag {lag} samples (score {score:.2f})", file=sys.stderr)
    np.save(args.out, np.ascontiguousarray(y, dtype=np.float32))


if __name__ == "__main__":
    main()
