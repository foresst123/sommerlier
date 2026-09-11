"""Pitch comparison on independently voiced, same-timeline probe spans."""

import numpy as np


def score(reference, output, sr, spans):
    import librosa
    cents, agreements = [], []
    for lo, hi in spans or ():
        if hi - lo < max(2048, int(sr * 0.15)):
            continue
        kwargs = dict(sr=sr, fmin=70, fmax=400, hop_length=round(sr * 0.01))
        f_ref, v_ref, _ = librosa.pyin(reference[lo:hi], **kwargs)
        f_out, v_out, _ = librosa.pyin(output[lo:hi], **kwargs)
        n = min(len(f_ref), len(f_out))
        both = v_ref[:n] & v_out[:n] & np.isfinite(f_ref[:n]) & np.isfinite(f_out[:n])
        union = v_ref[:n] | v_out[:n]
        # Silence agreeing with silence must not hide lost voiced endings.
        agreements.extend((v_ref[:n] == v_out[:n])[union])
        cents.extend(np.abs(1200 * np.log2(f_out[:n][both] / f_ref[:n][both])))
    return {"median_abs_cents": float(np.median(cents)) if cents else None,
            "pct_over_50_cents": float(np.mean(np.asarray(cents) > 50) * 100) if cents else None,
            "voiced_agreement": float(np.mean(agreements)) if agreements else None,
            "voiced_frames": len(cents)}
