"""Pseudo-steps in a denoise chain: tokens the service handles itself, not backends.

    blendNN   NN in 0..100: out = (1 - NN/100) * current + (NN/100) * chain_input
    dryNN     NN in 0..100: out = (1 - NN/100) * current + (NN/100) * step_input
    hpNN      NN in 10..200 Hz: zero-phase 2nd-order Butterworth high-pass, run forward and backward so
              the response is squared (4th-order slope, no delay). hp60 clears rumble and mains hum and
              costs about 1.9 dB at 85 Hz, the fundamental of a low male voice; use hp50 to keep more of it

chain_input is the waveform entering the whole chain (the raw audio of the stage), also
when the blend sits between two backends. step_input is the waveform that entered the nearest
backend before the token: in `a>blend25>b>dry50` the dry50 mixes back what b was given (a's
output blended with the raw audio), not the raw audio, so a second stage can be made lighter
without bringing back the noise the first stage removed. Pure functions, no heavy imports.
"""
import re
from typing import List, Optional, Tuple

import numpy as np

_BLEND = re.compile(r"blend(\d{1,3})")
_DRY = re.compile(r"dry(\d{1,3})")
_HP = re.compile(r"hp(\d{2,3})")
HP_MIN_HZ, HP_MAX_HZ, HP_ORDER = 10, 200, 2


def parse_pseudo(token) -> Optional[Tuple[str, int]]:
    """("blend", NN), ("dry", NN) or ("hp", Hz) for a valid pseudo-step; None for anything else (incl. blend101, hp5)."""
    t = str(token).strip().lower()
    m = _HP.fullmatch(t)
    if m:
        return ("hp", int(m.group(1))) if HP_MIN_HZ <= int(m.group(1)) <= HP_MAX_HZ else None
    for kind, rx in (("blend", _BLEND), ("dry", _DRY)):
        m = rx.fullmatch(t)
        if m:
            return (kind, int(m.group(1))) if int(m.group(1)) <= 100 else None
    return None


def is_pseudo(token) -> bool:
    return parse_pseudo(token) is not None


def backend_tokens(tokens) -> List[str]:
    """Tokens that need a real backend; pseudo-steps are always available."""
    return [t for t in tokens if not is_pseudo(t)]


def describe(token) -> str:
    p = parse_pseudo(token)
    if p is None:
        return f"backend {token}"
    if p[0] == "hp":
        return f"high-pass at {p[1]} Hz"
    if p[0] == "dry":
        return f"dry: {100 - p[1]}% filtered + {p[1]}% input of the preceding backend"
    return f"blend: {100 - p[1]}% filtered + {p[1]}% chain input"


def validate_chain_tokens(tokens) -> None:
    """ValueError if a blend/dry leads the chain, or a chain without backends has a blend/dry.

    A chain of hp tokens only (e.g. ["hp60"]) is valid: a high-pass filters the input itself,
    so it needs no backend. A blend or dry with no backend has nothing to mix against.
    """
    tokens = list(tokens)
    if not tokens:
        return
    if is_pseudo(tokens[0]) and parse_pseudo(tokens[0])[0] != "hp":
        raise ValueError(f"pseudo-step {tokens[0]!r} cannot be first in a denoise chain: "
                         "there is no filtered audio to blend yet")
    if not backend_tokens(tokens) and any(parse_pseudo(t)[0] != "hp" for t in tokens):
        raise ValueError("a denoise chain with blend or dry steps needs at least one backend")
    seen_backend = False
    for t in tokens:
        if not is_pseudo(t):
            seen_backend = True
        elif parse_pseudo(t)[0] == "dry" and not seen_backend:
            raise ValueError(f"pseudo-step {t!r} needs a backend step before it")


def highpass(x: np.ndarray, sr: int, cutoff_hz: float, order: int = HP_ORDER) -> np.ndarray:
    """Zero-phase Butterworth high-pass (same length, no delay); plain forward filter if x is too short to pad."""
    if not 0 < cutoff_hz < sr / 2:
        raise ValueError(f"highpass: cutoff {cutoff_hz} Hz must be inside (0, {sr / 2})")
    from scipy.signal import butter, sosfilt, sosfiltfilt
    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    x64 = np.asarray(x, dtype=np.float64)
    if len(x64) > 3 * (2 * len(sos) + 1):
        return sosfiltfilt(sos, x64).astype(np.float32)
    return sosfilt(sos, x64).astype(np.float32)


def blend(current: np.ndarray, chain_input: np.ndarray, percent: int) -> np.ndarray:
    """float32 mix; exact current at 0 and exact chain_input at 100; equal lengths required."""
    if len(current) != len(chain_input):
        raise ValueError(f"blend: lengths differ ({len(current)} vs {len(chain_input)})")
    cur = np.asarray(current, dtype=np.float32)
    raw = np.asarray(chain_input, dtype=np.float32)
    if percent <= 0:
        return cur.copy()
    if percent >= 100:
        return raw.copy()
    w = np.float32(percent / 100.0)
    return ((np.float32(1.0) - w) * cur + w * raw).astype(np.float32)
