"""Audio helpers shared by the denoise backends: chunked overlap-add, exact-length
resampling and constant-delay estimation. numpy/scipy only (no torch)."""
from math import gcd
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.signal import correlate, resample_poly


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad the tail of a 1-D array to exactly n samples."""
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=x.dtype)])


# ---------------------------------------------------------------- resampling

def resampled_length(n: int, sr_in: int, sr_out: int) -> int:
    """Length scipy's resample_poly returns for n samples (ceil of n * sr_out / sr_in)."""
    g = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    return -(-n * up // down)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample by the exact rational ratio sr_out / sr_in. Always returns a new float32 array."""
    sr_in, sr_out = int(round(sr_in)), int(round(sr_out))
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError(f"sample rates must be positive, got {sr_in} -> {sr_out}")
    if sr_in == sr_out or len(x) == 0:
        return np.array(x, dtype=np.float32, copy=True)
    g = gcd(sr_in, sr_out)
    y = resample_poly(np.asarray(x, dtype=np.float32), sr_out // g, sr_in // g)
    return np.ascontiguousarray(y, dtype=np.float32)


def resample_exact(x: np.ndarray, sr_in: int, sr_out: int, n_out: int) -> np.ndarray:
    """resample_to, then pad/trim by the resampler's rounding to exactly n_out samples."""
    return fit_length(resample_to(x, sr_in, sr_out), n_out)


# ------------------------------------------------------------ chunk + overlap

def chunk_spans(n: int, chunk: int, overlap: int) -> list:
    """(start, end) spans covering [0, n), none longer than `chunk`. The spans have equal length and
    neighbours overlap by about `overlap` (never less than overlap - 1), so there is no tiny tail chunk."""
    if chunk <= 0 or overlap < 0 or 4 * overlap > chunk:
        raise ValueError(f"need 0 <= overlap <= chunk / 4, got chunk={chunk}, overlap={overlap}")
    if n <= chunk:
        return [(0, n)]
    k = -(-(n - overlap) // (chunk - overlap))  # number of spans
    length = -(-(n + (k - 1) * overlap) // k)
    starts = [(i * (n - length)) // (k - 1) for i in range(k)]
    return [(s, s + length) for s in starts]


def chunk_apply(x: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], chunk: int, overlap: int) -> np.ndarray:
    """Apply fn chunk-wise and overlap-add with a squared-sine/cosine crossfade.

    fn takes and returns a 1-D array; a result of the wrong length is padded/trimmed to the
    chunk length. The fade pair sums to exactly 1, so fn=identity reconstructs x.
    """
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    spans = chunk_spans(n, chunk, overlap)
    out = np.zeros(n, dtype=np.float32)
    prev_end = 0
    for i, (s, e) in enumerate(spans):
        y = fit_length(np.asarray(fn(x[s:e]), dtype=np.float32), e - s)
        w = np.ones(e - s, dtype=np.float32)
        if i > 0:
            ov = prev_end - s
            if ov > 0:
                w[:ov] = np.sin(0.5 * np.pi * (np.arange(ov) + 0.5) / ov) ** 2
        if i < len(spans) - 1:
            ov = e - spans[i + 1][0]
            if ov > 0:
                w[e - s - ov:] *= np.cos(0.5 * np.pi * (np.arange(ov) + 0.5) / ov) ** 2
        out[s:e] += y * w
        prev_end = e
    return out


# ------------------------------------------------------------- delay handling

def estimate_delay(ref: np.ndarray, out: np.ndarray, max_lag: int, probe: int = 0) -> Tuple[int, float]:
    """Constant lag of `out` relative to `ref` by cross-correlation on the loudest probe region.

    Returns (lag, score): lag > 0 means out[i + lag] ~ ref[i] (out is late); score is the
    normalised correlation at the peak. (0, 0.0) when the signal is too short or silent.
    """
    n = min(len(ref), len(out))
    if probe <= 0:
        probe = n
    probe = min(probe, n - 2 * max_lag)
    if max_lag <= 0 or probe < max(16, max_lag):
        return 0, 0.0
    # loudest probe-sized block whose +-max_lag neighbourhood fits inside the signal
    ref64 = np.asarray(ref[:n], dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(ref64 ** 2)])
    starts = np.arange(max_lag, n - probe - max_lag + 1, max(1, probe // 2))
    if len(starts) == 0:
        return 0, 0.0
    a = int(starts[np.argmax(csum[starts + probe] - csum[starts])])
    r = ref64[a:a + probe]
    o = np.asarray(out[a - max_lag:a + probe + max_lag], dtype=np.float64)
    r_energy = float(np.sum(r ** 2))
    if r_energy <= 1e-12:
        return 0, 0.0
    corr = correlate(o, r, mode="valid", method="fft")  # index k <-> lag k - max_lag
    k = int(np.argmax(corr))
    o_win = o[k:k + probe]
    denom = np.sqrt(r_energy * float(np.sum(o_win ** 2)))
    score = float(corr[k] / denom) if denom > 0 else 0.0
    return k - max_lag, score


def compensate_delay(out: np.ndarray, lag: int) -> np.ndarray:
    """Shift out earlier by lag samples (later if negative), keeping the length."""
    if lag == 0:
        return out
    n = len(out)
    lag = int(np.clip(lag, -n, n))
    res = np.zeros_like(out)
    if lag > 0:
        res[:n - lag] = out[lag:]
    else:
        res[-lag:] = out[:n + lag]
    return res


def align_delay(ref: np.ndarray, out: np.ndarray, max_lag: int, probe: int = 0,
                min_score: float = 0.3) -> Tuple[np.ndarray, int]:
    """estimate_delay + compensate_delay; leaves out untouched when the peak is not convincing."""
    lag, score = estimate_delay(ref, out, max_lag, probe)
    if lag == 0 or score < min_score:
        return out, 0
    return compensate_delay(out, lag), lag
