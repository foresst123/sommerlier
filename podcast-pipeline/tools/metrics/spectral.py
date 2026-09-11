"""Phase-insensitive spectral shape and relative frame energy."""

import numpy as np
from scipy.signal import welch

from tools.metrics.audio import slices


def frame_levels(parts, sr):
    width = max(1, round(sr * 0.02))
    values = []
    for part in parts:
        usable = len(part) // width * width
        if usable:
            frames = part[:usable].reshape(-1, width).astype(np.float64)
            values.extend(np.sqrt(np.mean(frames ** 2, axis=1)))
    if not values:
        return None
    rms = np.asarray(values)
    peak = float(rms.max())
    if peak < 1e-12:
        return {"floor_db": -120.0, "gated_pct": 100.0}
    db = 20 * np.log10(np.maximum(rms / peak, 1e-6))
    return {"floor_db": float(np.percentile(db, 5)),
            "gated_pct": float(np.mean(db < -50) * 100)}


def band_shape(parts, sr):
    nfft = 4096
    sums = np.zeros(nfft // 2 + 1)
    total = 0
    for part in parts:
        if len(part) < 64:
            continue
        frequencies, power = welch(part, sr, nperseg=min(nfft, len(part)), nfft=nfft)
        sums += power * len(part)
        total += len(part)
    if total == 0 or not np.any(sums):
        return None
    power = sums / total
    centers = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                        1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000])
    bands = [float(power[(frequencies >= c / 2 ** (1/6)) &
                         (frequencies < c * 2 ** (1/6))].sum())
             for c in centers if c <= min(8000, sr / 2)]
    db = 10 * np.log10(np.maximum(bands, 1e-20))
    return db - db.mean()


def score(reference, output, sr, spans=None):
    length = min(len(reference), len(output))
    spans = [(0, length)] if spans is None else spans
    ref_parts, out_parts = slices(reference, spans), slices(output, spans)
    ref_levels, out_levels = frame_levels(ref_parts, sr), frame_levels(out_parts, sr)
    left, right = band_shape(ref_parts, sr), band_shape(out_parts, sr)
    result = {"tilt_db": float(np.mean(np.abs(right - left)))
              if left is not None and right is not None else None,
              "floor_db": None, "gated_pct": None,
              "mix_floor_db": None, "mix_gated_pct": None}
    if out_levels:
        result.update(out_levels)
    if ref_levels:
        result.update({f"mix_{k}": v for k, v in ref_levels.items()})
    return result
