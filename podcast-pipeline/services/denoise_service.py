"""Optional noise filter between music removal and diarization.

A chain such as "dfn3>mf2se48" runs its backends in order on the post-music
waveform. Pseudo-steps such as "blend50" (models/denoise/pseudo.py) are handled
here: they mix the chain input back into the current audio. The result must be sample-aligned with the input: later stages map
timestamps back through the music timeline, so a filter that changes the length
(or shifts the audio) would silently corrupt every segment time. A chain that
fails any check is dropped whole and the input waveform kept -- a variant with
half a chain applied would be a different variant under the same name.
"""

import time

import numpy as np

NONE_SPECS = ("", "none")
FRAME_SEC = 0.025
MAX_LAG_SEC = 0.1          # largest shift the lag estimate looks for
LAG_WINDOW_SEC = 60.0      # cross-correlate at most this much audio
SPEECH_DB = -45.0          # a 90th-percentile frame level above this means speech
SILENT_DB = -80.0          # an output below this where the input had speech is dead


def _load_api():
    """models.denoise imported on first use, so the pipeline runs without it."""
    from models.denoise import chain_id, get_backend, parse_chain
    return get_backend, parse_chain, chain_id


def is_none(spec) -> bool:
    return spec is None or str(spec).strip().lower() in NONE_SPECS


def parse_spec(spec) -> list:
    """Backend names in `spec`; [] for none, without importing models.denoise."""
    if is_none(spec):
        return []
    return list(_load_api()[1](spec))


def chain_id_of(spec) -> str:
    if is_none(spec):
        return "none"
    return _load_api()[2](spec)


def frame_levels_db(wav: np.ndarray, sr: int) -> np.ndarray:
    """RMS level in dBFS of consecutive 25 ms frames."""
    n = max(1, int(round(FRAME_SEC * sr)))
    frames = len(wav) // n
    if frames == 0:
        return np.zeros(0, dtype=np.float64)
    x = np.asarray(wav[:frames * n], dtype=np.float64).reshape(frames, n)
    rms = np.sqrt(np.mean(x * x, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-6))


def noise_floor_db(wav: np.ndarray, sr: int):
    """10th percentile of the frame levels: the level of the quiet parts."""
    levels = frame_levels_db(wav, sr)
    return float(np.percentile(levels, 10)) if len(levels) else None


def estimate_lag(ref: np.ndarray, out: np.ndarray, sr: int):
    """(lag in samples, correlation at that lag). Positive lag: `out` is late."""
    n = min(len(ref), len(out), int(LAG_WINDOW_SEC * sr))
    if n < 2:
        return 0, None
    a = np.asarray(ref[:n], dtype=np.float64)
    b = np.asarray(out[:n], dtype=np.float64)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 0.0:
        return 0, None
    size = 1 << (2 * n - 1).bit_length()
    # xcorr[k] = sum_t a[t] * b[t + k]; k > 0 means b is a delayed copy of a.
    spec = np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size))
    xc = np.fft.irfft(spec, size)
    max_lag = min(int(MAX_LAG_SEC * sr), n - 1)
    window = np.concatenate([xc[size - max_lag:], xc[:max_lag + 1]])
    best = int(np.argmax(window))
    return best - max_lag, float(window[best] / norm)


def _check(inp: np.ndarray, out: np.ndarray, sr: int, name: str) -> str:
    """Why `out` cannot replace `inp`, or "" when it can."""
    if out is None or getattr(out, "ndim", 0) != 1:
        return f"{name}: output is not a mono 1-D array"
    if len(out) != len(inp):
        return f"{name}: output has {len(out)} samples, input {len(inp)}"
    if not np.all(np.isfinite(out)):
        return f"{name}: output has NaN or Inf"
    in_levels = frame_levels_db(inp, sr)
    if len(in_levels) and np.percentile(in_levels, 90) > SPEECH_DB:
        out_levels = frame_levels_db(out, sr)
        if not len(out_levels) or np.percentile(out_levels, 90) < SILENT_DB:
            return f"{name}: output is silent where the input has speech"
    return ""


class DenoiseService:
    """Applies a denoise chain to AudioData, never raising for a model failure."""

    def apply(self, audio_data, chain_spec, cfg=None, logger=None) -> dict:
        """Filter audio_data.waveform in place and return the denoise.json dict."""
        if not isinstance(cfg, dict):  # a profile may hold a plain string under "denoise"
            cfg = None
        inp = np.ascontiguousarray(audio_data.waveform, dtype=np.float32)
        sr = int(audio_data.sample_rate)
        info = {"chain": [], "chain_id": "none", "sample_rate": sr,
                "samples": int(len(inp)), "lag_samples": 0,
                "noise_floor_db_before": noise_floor_db(inp, sr),
                "noise_floor_db_after": None, "corr_with_input": None,
                "errors": [], "steps": [], "blend": None}
        out = inp  # inp is also the chain input that blend steps mix back in
        step_in = inp  # what the latest backend step was given; dry steps mix this back in
        try:
            from models.denoise.pseudo import blend, parse_pseudo, validate_chain_tokens
            get_backend, parse_chain, chain_id = _load_api()
            chain = list(parse_chain(chain_spec))
            info["chain"] = chain
            info["chain_id"] = chain_id(chain)
            validate_chain_tokens(chain)
        except Exception as exc:
            info["errors"].append(f"denoise unavailable: {exc}")
            chain = []
        for name in chain:
            backend = None
            started = time.time()
            pseudo = parse_pseudo(name)
            try:
                if pseudo is not None:
                    kind, pct = pseudo
                    cand = blend(out, inp if kind == "blend" else step_in, pct)
                    problem = _check(inp, cand, sr, name)
                    if problem:
                        raise ValueError(problem)
                    out = cand
                    if kind == "blend":
                        info["blend"] = {"percent": pct}
                    else:
                        info.setdefault("dry", []).append({"percent": pct})
                    info["steps"].append({"token": name, "kind": kind,
                                          "seconds": round(time.time() - started, 3)})
                    continue
                backend = get_backend(name, cfg, logger)
                backend.load()
                step_in = out
                if logger:
                    logger.info(f"[denoise] {name} on {len(inp) / sr:.1f}s of audio")
                cand = np.asarray(backend.enhance(out, sr), dtype=np.float32)
                problem = _check(inp, cand, sr, name)
                if problem:
                    raise ValueError(problem)
                out = cand
                info["steps"].append({"token": name, "kind": "backend",
                                      "seconds": round(time.time() - started, 3)})
            except Exception as exc:
                info["errors"].append(f"{name}: {exc}" if name not in str(exc) else str(exc))
                out = inp
                break
            finally:
                if backend is not None:
                    try:
                        backend.unload()
                    except Exception as exc:
                        if logger:
                            logger.warning(f"[denoise] could not unload {name}: {exc}")
        if info["errors"]:
            if logger:
                logger.warning("[denoise] keeping the unfiltered audio: "
                               + "; ".join(info["errors"]))
            out = inp
        lag, corr = estimate_lag(inp, out, sr)
        info["lag_samples"] = int(lag)
        info["corr_with_input"] = corr
        info["noise_floor_db_after"] = noise_floor_db(out, sr)
        audio_data.waveform = out
        return info
