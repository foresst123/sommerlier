"""Pseudo-steps in a denoise chain: tokens the service handles itself, not backends.

    blendNN   NN in 0..100: out = (1 - NN/100) * current + (NN/100) * chain_input

chain_input is the waveform entering the whole chain (the raw audio of the stage), also
when the blend sits between two backends. Pure functions, no heavy imports.
"""
import re
from typing import List, Optional, Tuple

import numpy as np

_BLEND = re.compile(r"blend(\d{1,3})")


def parse_pseudo(token) -> Optional[Tuple[str, int]]:
    """("blend", NN) for a valid blend token; None for anything else (incl. blend101)."""
    m = _BLEND.fullmatch(str(token).strip().lower())
    if not m or int(m.group(1)) > 100:
        return None
    return "blend", int(m.group(1))


def is_pseudo(token) -> bool:
    return parse_pseudo(token) is not None


def backend_tokens(tokens) -> List[str]:
    """Tokens that need a real backend; pseudo-steps are always available."""
    return [t for t in tokens if not is_pseudo(t)]


def describe(token) -> str:
    p = parse_pseudo(token)
    if p is None:
        return f"backend {token}"
    return f"blend: {100 - p[1]}% filtered + {p[1]}% chain input"


def validate_chain_tokens(tokens) -> None:
    """ValueError if a pseudo-step leads the chain or the chain has no backend at all."""
    tokens = list(tokens)
    if not tokens:
        return
    if is_pseudo(tokens[0]):
        raise ValueError(f"pseudo-step {tokens[0]!r} cannot be first in a denoise chain: "
                         "there is no filtered audio to blend yet")
    if not backend_tokens(tokens):
        raise ValueError("a denoise chain needs at least one backend, not only pseudo-steps")


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
