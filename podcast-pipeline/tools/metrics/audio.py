"""Audio loading and sidecar units shared by metrics and polish."""

import json
from pathlib import Path

import numpy as np


def read_audio(path, sr=None):
    import soundfile as sf
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    if sr and rate != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=rate, target_sr=sr)
        rate = sr
    return audio, rate


def sidecar(mix):
    path = Path(str(mix).removesuffix("_mix.wav") + ".json")
    return json.loads(path.read_text()) if path.is_file() else None


def probe_spans(meta, track, sr, length):
    if meta is None:
        return None
    speaker = meta["speakers"][0 if track == "A" else 1]
    scale = sr / meta["sr"] if meta.get("probe_units", "samples") == "samples" else sr
    spans = []
    for lo, hi in meta.get("probes", {}).get(str(speaker), []):
        a, b = max(0, round(lo * scale)), min(length, round(hi * scale))
        if b > a:
            spans.append((a, b))
    return spans


def selected_windows(directory, windows=4):
    candidates = sorted(Path(directory).glob("*_mix.wav"))
    def rank(path):
        meta = sidecar(path)
        if not meta:
            return (0, path.name)
        core = meta["core"]
        duration = (core[1] - core[0]) / (meta["sr"] if meta.get("core_units", "samples") == "samples" else 1)
        return (-duration, path.name)
    ranked = sorted(candidates, key=rank)
    return ranked[:windows] if windows else ranked


def slices(audio, spans):
    return [audio[a:b] for a, b in spans if b > a]
