"""Per-variant metrics: audio, diarization, separation, ASR."""

import os
from typing import Dict, List, Optional

import numpy as np

from .io_stage import read_wav
from .speaker_align import HOP, label_grid
from .text_align import tokens

FRAME_S = 0.025
GATE_DB = -60.0
QUIET_DB = -50.0
LAG_WINDOW_S = 60.0
MAX_LAG_S = 1.0
HF_CUTOFF_HZ = 4000.0


def frame_levels_db(x: np.ndarray, sr: int) -> np.ndarray:
    """RMS level in dBFS of consecutive 25 ms frames."""
    n = max(1, int(round(FRAME_S * sr)))
    k = len(x) // n
    if k == 0:
        return np.zeros(0)
    frames = x[:k * n].astype(np.float64).reshape(k, n)
    return 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)


def estimate_lag(raw: np.ndarray, var: np.ndarray, sr: int) -> int:
    """Samples by which `var` is delayed relative to `raw` (negative: early).

    Cross-correlation over the first 60 s, searched within +-1 s.
    """
    from scipy.signal import fftconvolve
    n = int(min(len(raw), len(var), LAG_WINDOW_S * sr))
    if n < 2:
        return 0
    a, b = raw[:n].astype(np.float64), var[:n].astype(np.float64)
    corr = fftconvolve(b, a[::-1], mode="full")  # index k <-> lag k-(n-1)
    m = int(MAX_LAG_S * sr)
    lo, hi = max(0, n - 1 - m), min(len(corr), n + m)
    return int(np.argmax(corr[lo:hi]) + lo - (n - 1))


def aligned_corr(raw: np.ndarray, var: np.ndarray, lag: int) -> Optional[float]:
    """Pearson correlation of the two signals after removing `lag`."""
    if lag >= 0:
        a, b = raw, var[lag:]
    else:
        a, b = raw[-lag:], var
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def hf_energy_ratio(x: np.ndarray, sr: int, cutoff: float = HF_CUTOFF_HZ) -> Optional[float]:
    """Spectral energy above `cutoff` Hz over total energy (chunked FFT)."""
    if len(x) == 0:
        return None
    size = 1 << 15
    freqs = np.fft.rfftfreq(size, 1.0 / sr)
    hi = freqs > cutoff
    tot = hf = 0.0
    for i in range(0, len(x), size):
        p = np.abs(np.fft.rfft(x[i:i + size].astype(np.float64), n=size)) ** 2
        tot += p.sum()
        hf += p[hi].sum()
    return float(hf / tot) if tot > 0 else None


def _ecapa_cos(raw_path: str, var_path: str) -> Optional[float]:
    """Cosine of speaker embeddings of the first 30 s, raw vs variant.

    Uses the repo's WeSpeaker embedder, and only when WESPEAKER_MODEL_PATH
    points at a local file, so nothing is ever downloaded. None otherwise.
    """
    path = os.environ.get("WESPEAKER_MODEL_PATH")
    if not path or not os.path.isfile(os.path.expanduser(path)):
        return None
    try:
        import torch
        from models.wespeaker_embedding import WeSpeakerONNXEmbedder
        emb = WeSpeakerONNXEmbedder("cpu")
        vecs = []
        for p in (raw_path, var_path):
            x, sr = read_wav(p)
            vecs.append(emb.embed(x[:int(30 * sr)], sr).float())
        return float(torch.nn.functional.cosine_similarity(vecs[0], vecs[1], dim=0))
    except Exception:
        return None


def audio_metrics(raw_path: Optional[str], var_path: Optional[str],
                  same_as_raw: bool = False, embed: bool = True) -> Dict[str, Optional[float]]:
    keys = ("noise_floor_db", "gated_frame_pct", "lag_samples", "corr_with_raw",
            "hf_energy_ratio", "length_equal", "ecapa_cos")
    out: Dict[str, Optional[float]] = {k: None for k in keys}
    var, sr = read_wav(var_path)
    if var is None:
        return out
    lv = frame_levels_db(var, sr)
    if len(lv):
        out["noise_floor_db"] = float(np.percentile(lv, 10))
        out["gated_frame_pct"] = float(100.0 * np.mean(lv < GATE_DB))
    out["hf_energy_ratio"] = hf_energy_ratio(var, sr)
    if same_as_raw:
        out.update(lag_samples=0, corr_with_raw=1.0, length_equal=True, ecapa_cos=1.0)
        return out
    raw, rsr = read_wav(raw_path)
    if raw is None:
        return out
    out["length_equal"] = bool(len(raw) == len(var) and rsr == sr)
    if rsr == sr:
        lag = estimate_lag(raw, var, sr)
        out["lag_samples"] = lag
        out["corr_with_raw"] = aligned_corr(raw, var, lag)
        if embed:
            out["ecapa_cos"] = _ecapa_cos(raw_path, var_path)
    return out


def diarization_metrics(segs: Optional[List[dict]]) -> Dict[str, Optional[float]]:
    keys = ("n_speakers", "n_segments", "overlap_seconds", "median_segment_s",
            "short_segments_lt_0_5s")
    if segs is None:
        return {k: None for k in keys}
    durs = [s["end"] - s["start"] for s in segs]
    end = max([s["end"] for s in segs], default=0.0)
    grid = label_grid(segs, int(np.ceil(end / HOP)) + 1)
    n_active = sum(a.astype(np.int16) for a in grid.values()) if grid else np.zeros(1)
    return {
        "n_speakers": len(grid),
        "n_segments": len(segs),
        "overlap_seconds": float(np.count_nonzero(n_active >= 2) * HOP),
        "median_segment_s": float(np.median(durs)) if durs else None,
        "short_segments_lt_0_5s": sum(d < 0.5 for d in durs),
    }


def separation_metrics(sep: Optional[dict]) -> Dict[str, Optional[float]]:
    keys = ("overlap_spans", "segments_separated", "failed_spans", "mean_similarity")
    if not sep:
        return {k: None for k in keys}
    stats = sep.get("stats") or {}
    segs = sep.get("segments")
    spliced, failed = stats.get("spans_spliced"), stats.get("spans_failed")
    separated = stats.get("segments_separated")
    sims: List[float] = []
    if segs is not None:
        spliced = sum(len(s["bss_spans"]) for s in segs)
        failed = sum(len(s["bss_failed_spans"]) for s in segs)
        separated = sum(1 for s in segs if s["bss"])
        # bss_spans rows are (start, end, similarity); negative means unknown
        sims = [sp[2] for s in segs for sp in s["bss_spans"]
                if len(sp) > 2 and sp[2] is not None and sp[2] >= 0]
    return {
        "overlap_spans": None if spliced is None and failed is None else (spliced or 0) + (failed or 0),
        "segments_separated": separated,
        "failed_spans": failed,
        "mean_similarity": float(np.mean(sims)) if sims else None,
    }


def asr_metrics(segs: Optional[List[dict]], var_audio_path: Optional[str],
                wer_vs_baseline=None, wer_vs_reference=None) -> Dict[str, Optional[float]]:
    keys = ("n_words", "n_segments", "empty_segments", "words_per_sec", "wer_vs_baseline",
            "wer_vs_reference", "words_in_quiet_frames")
    out: Dict[str, Optional[float]] = {k: None for k in keys}
    if segs is None:
        return out
    n_words = sum(len(tokens(s["text"])) for s in segs)
    speech = sum(s["end"] - s["start"] for s in segs)
    out.update(
        n_words=n_words, n_segments=len(segs),
        empty_segments=sum(1 for s in segs if not tokens(s["text"])),
        words_per_sec=n_words / speech if speech > 0 else None,
        wer_vs_baseline=wer_vs_baseline, wer_vs_reference=wer_vs_reference)
    words = [w for s in segs for w in (s.get("words") or [])]
    if words:
        x, sr = read_wav(var_audio_path)
        if x is not None:
            lv = frame_levels_db(x, sr)
            quiet = 0
            for w in words:
                a = max(0, int(w["start"] / FRAME_S))
                b = min(len(lv), max(a + 1, int(np.ceil(w["end"] / FRAME_S))))
                if b > a and np.all(lv[a:b] < QUIET_DB):
                    quiet += 1
            out["words_in_quiet_frames"] = quiet
    return out
