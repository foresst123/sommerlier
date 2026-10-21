"""Optional learned MOS (Uni-VERSA-Ext 5-metric) for the comparison toolchain.

A distilled mimic of DNSMOS/UTMOS/NISQA/SCOREQ plus a human-MOS head, 16 kHz only.
Informational delta, never a gate (see SCHEMA.md). torch/transformers are imported
only inside the loader; every failure returns None.

Enable with: UNIVERSA_MODEL (model.pt), UNIVERSA_CONFIG (config.yaml) and, unless
urgent2026_sqa is pip-installed, UNIVERSA_REPO (clone of urgent2026_challenge_track2).
"""

import math
import os
import sys
from typing import Callable, Dict, Optional

import numpy as np

KEYS = ("mos", "dnsmos_ovrl", "scoreq", "utmos", "nisqa_mos")
MODEL_SR = 16000
CHUNK_S = 20.0
_CACHE: Dict[tuple, Callable] = {}


def _paths(cfg: Optional[dict]):
    cfg = cfg or {}
    get = lambda k, env: os.path.expanduser(cfg.get(k) or os.environ.get(env) or "")
    return get("repo", "UNIVERSA_REPO"), get("model", "UNIVERSA_MODEL"), get("config", "UNIVERSA_CONFIG")


def load_predictor(cfg: Optional[dict] = None) -> Optional[Callable]:
    """predict(float32 1-D 16 kHz) -> {key: float}, or None if unconfigured/unavailable."""
    repo, model_p, config_p = _paths(cfg)
    if not (os.path.isfile(model_p) and os.path.isfile(config_p)):
        return None
    key = (repo, model_p, config_p)
    if key in _CACHE:
        return _CACHE[key]
    try:
        if repo and os.path.isdir(repo) and repo not in sys.path:
            sys.path.insert(0, repo)
        import torch
        from urgent2026_sqa.infer import infer_single, load_model
        model, mcfg = load_model(model_p, config_p)

        def predict(x: np.ndarray) -> Dict[str, float]:
            return infer_single(model, mcfg, torch.from_numpy(x[None, :].astype(np.float32)), MODEL_SR)

        _CACHE[key] = predict
        return predict
    except Exception:
        return None


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == MODEL_SR:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    g = math.gcd(int(sr), MODEL_SR)
    return resample_poly(x.astype(np.float64), MODEL_SR // g, int(sr) // g).astype(np.float32)


def score_chunks(x: np.ndarray, predict: Callable, chunk_s: float = CHUNK_S,
                 sr: int = MODEL_SR) -> Optional[Dict[str, float]]:
    """Length-weighted mean of predict() over near-equal chunks of about chunk_s seconds."""
    n = len(x)
    if n == 0:
        return None
    parts = max(1, int(round(n / (chunk_s * sr))))
    bounds = np.linspace(0, n, parts + 1).astype(int)
    sums = {k: 0.0 for k in KEYS}
    total = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        res = predict(x[a:b])
        for k in KEYS:
            v = float(res[k])
            if not math.isfinite(v):
                return None
            sums[k] += v * (b - a)
        total += b - a
    return {k: s / total for k, s in sums.items()} if total else None


def score_wav(path_or_array, sr: Optional[int] = None, cfg: Optional[dict] = None
              ) -> Optional[Dict[str, float]]:
    """{mos, dnsmos_ovrl, scoreq, utmos, nisqa_mos} (about 1-5) or None. Never raises."""
    try:
        predict = load_predictor(cfg)
        if predict is None:
            return None
        if isinstance(path_or_array, (str, os.PathLike)):
            from .io_stage import read_wav
            x, sr = read_wav(os.fspath(path_or_array))
            if x is None:
                return None
        else:
            x = np.asarray(path_or_array, dtype=np.float32)
            if x.ndim > 1:
                x = x.mean(axis=int(np.argmin(x.shape)))  # channel axis is the short one
        if not sr:
            return None
        chunk_s = float((cfg or {}).get("chunk_s") or CHUNK_S)
        return score_chunks(to_16k(x, int(sr)), predict, chunk_s)
    except Exception:
        return None
