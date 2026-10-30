#!/usr/bin/env python3
"""Vocals stem of a 16 kHz mono WAV with Hybrid Transformer Demucs (htdemucs). Experiment code, not the pipeline.

    <demucs python> tools/demucs_vocals.py --in chunk.wav --out chunk_vocals.wav [--model htdemucs] [--device cuda|cpu]

HT Demucs is a MUSIC source separator (drums, bass, other, vocals). Its vocals stem is not a speech denoiser: it
was trained on songs. Here it only feeds the EMBEDDING branch of arm K (tools/windowed_vbx_stage.py --embed-audio);
the segmentation probabilities still come from the original audio.

Steps, as the demucs CLI does them: the input is resampled to the model rate (44100 Hz) and duplicated to stereo,
normalised with ref = wav.mean(0), wav = (wav - ref.mean()) / ref.std(); apply_model(split=True, overlap=0.25)
runs the model; the vocals source is de-normalised, averaged to mono, resampled back to 16 kHz and fitted to the
exact input length. The output is a 32-bit float WAV.

The numpy helpers (to_stereo, mono_mix, normalise, denormalise, fit_length) import no torch or demucs, so they are
tested without them. demucs and torch are imported inside main(). UNVERIFIED: not run here (demucs not installed).
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence, Tuple

import numpy as np

SR = 16000
MODEL_DEFAULT = "htdemucs"
VOCALS = "vocals"


def to_stereo(mono: np.ndarray) -> np.ndarray:
    """(T,) -> (2, T): the same channel twice."""
    x = np.asarray(mono, dtype=np.float32).reshape(-1)
    return np.stack([x, x])


def mono_mix(x: np.ndarray) -> np.ndarray:
    """(C, T) -> (T,) by averaging the channels; (T,) stays as it is."""
    a = np.asarray(x, dtype=np.float32)
    return a.reshape(-1) if a.ndim == 1 else a.mean(axis=0)


def normalise(wav: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """The demucs CLI's normalisation of a (C, T) mix: (wav - mean) / std, with mean and std of wav.mean(0).

    Returns (normalised, mean, std). A silent input has std 0; it is kept at 1 so the output stays zeros, not NaN.
    """
    a = np.asarray(wav, dtype=np.float32)
    ref = a.mean(axis=0)
    mean = float(ref.mean())
    std = float(ref.std()) or 1.0
    return (a - mean) / std, mean, std


def denormalise(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Undo normalise(): x * std + mean."""
    return np.asarray(x, dtype=np.float32) * std + mean


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """(T,) -> exactly n samples: trimmed at the end or padded with zeros."""
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(a) >= n:
        return a[:n].copy()
    return np.concatenate([a, np.zeros(n - len(a), dtype=np.float32)])


def load_mono_16k(path: str) -> np.ndarray:
    import soundfile as sf
    wave, sr = sf.read(path, dtype="float32", always_2d=False)   # soundfile gives (frames, channels) for stereo
    if sr != SR:
        raise SystemExit(f"demucs_vocals expects {SR} Hz input, got {sr} Hz: {path}")
    return wave.mean(axis=1) if getattr(wave, "ndim", 1) == 2 else np.asarray(wave, dtype=np.float32)


def separate_vocals(mono16: np.ndarray, model, device: str, source_index: int, model_sr: int, apply_model,
                    resample) -> np.ndarray:
    """(T,) at 16 kHz -> (T,) vocals at 16 kHz, same length. model, apply_model, resample come from main()."""
    import torch

    n = len(mono16)
    up = resample(torch.from_numpy(to_stereo(mono16)), SR, model_sr)          # (2, T_model)
    up = up.numpy()
    norm, mean, std = normalise(up)
    with torch.no_grad():
        out = apply_model(model, torch.from_numpy(norm)[None], device=device, split=True, overlap=0.25,
                          progress=False)
    stem = out[0, source_index].detach().cpu().numpy()                        # (2, T_model)
    stem = denormalise(stem, mean, std)
    back = resample(torch.from_numpy(np.ascontiguousarray(mono_mix(stem))[None]), model_sr, SR)
    return fit_length(back.numpy().reshape(-1), n)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="16 kHz mono WAV")
    ap.add_argument("--out", required=True, help="vocals WAV to write (32-bit float, 16 kHz, same length)")
    ap.add_argument("--model", default=MODEL_DEFAULT, help="demucs pretrained model name (default htdemucs)")
    ap.add_argument("--device", help="cuda or cpu (default: cuda when torch sees a GPU)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        import torch
        import torchaudio.functional as taf
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:
        print(f"demucs_vocals needs torch, torchaudio and the demucs package in this env: {exc}", file=sys.stderr)
        return 2

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(a.model)
    model.eval()
    model.to(device)
    if VOCALS not in model.sources:
        print(f"model {a.model} has no {VOCALS!r} source: {model.sources}", file=sys.stderr)
        return 2
    index = list(model.sources).index(VOCALS)

    def resample(x, src_sr, dst_sr):
        return taf.resample(x, src_sr, dst_sr)

    mono = load_mono_16k(a.src)
    vocals = separate_vocals(mono, model, device, index, int(model.samplerate), apply_model, resample)
    import soundfile as sf
    sf.write(a.out, vocals.astype(np.float32), SR, subtype="FLOAT")
    print(f"wrote {a.out}: {len(vocals)} samples, model {a.model} on {device}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
