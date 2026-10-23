"""Optional noise-filter backends for the pipeline's denoise stage.

    from models.denoise import get_backend, parse_chain, chain_id, available_backends

Importing this package never imports torch/df/clearvoice; each backend imports its heavy
dependency inside load().
"""
import re
from typing import Dict, List, Optional, Tuple

from .base import DenoiseBackend, DenoiseError
from .dfn3 import DeepFilterNet3Backend
from .identity import FakeHighPassBackend, IdentityBackend
from .mf2se48 import MossFormer2SE48Backend
from .mpsenet import MPSENetBackend, MPSENetDNSBackend
from .dbs_tiger import DBSTigerBackend
from .mambattention import MambAttentionBackend
from .bsrnn_urgent import BSRNN26Backend
from .re_use import ReUseBackend, ReUseLaughterBackend
from .audio_separator import ALIASES as _AS_ALIASES, AudioSeparatorBackend

__all__ = ["DenoiseBackend", "DenoiseError", "get_backend", "parse_chain", "chain_id",
           "available_backends", "BACKENDS"]

BACKENDS = {cls.name: cls for cls in (IdentityBackend, FakeHighPassBackend, DeepFilterNet3Backend,
                                      MossFormer2SE48Backend, MPSENetBackend, MPSENetDNSBackend,
                                      DBSTigerBackend, MambAttentionBackend, BSRNN26Backend,
                                      ReUseBackend, ReUseLaughterBackend,
                                      AudioSeparatorBackend, *_AS_ALIASES)}

_TOKEN = re.compile(r"[a-z0-9_]+")


def parse_chain(spec) -> List[str]:
    """"dfn3>mf2se48" -> ["dfn3", "mf2se48"]; None, "" and "none" -> []. A list/tuple is validated the same way.

    Raises ValueError on empty or malformed tokens ("a>>b", "a>", "none>a"). Names are not checked
    against the registry here; get_backend does that.
    """
    if spec is None:
        return []
    tokens = list(spec) if isinstance(spec, (list, tuple)) else str(spec).split(">")
    tokens = [str(t).strip().lower() for t in tokens]
    if tokens == [""] or tokens == ["none"]:
        return []
    for t in tokens:
        if not t:
            raise ValueError(f"empty step in denoise chain {spec!r}")
        if t == "none":
            raise ValueError(f"'none' cannot be combined with other steps in denoise chain {spec!r}")
        if not _TOKEN.fullmatch(t):
            raise ValueError(f"invalid step {t!r} in denoise chain {spec!r} (use letters, digits, underscore)")
    return tokens


def chain_id(spec) -> str:
    """Filename-safe canonical id: [] -> "none", ["dfn3", "mf2se48"] -> "dfn3+mf2se48"."""
    return "+".join(parse_chain(spec)) or "none"


def _merge_cfg(cfg: Optional[dict], key: str) -> dict:
    cfg = dict(cfg or {})
    merged = {k: v for k, v in cfg.items() if k not in BACKENDS}
    if isinstance(cfg.get(key), dict):
        merged.update(cfg[key])
    return merged


def get_backend(name: str, cfg: Optional[dict] = None, logger=None) -> DenoiseBackend:
    """Instantiate a backend (not loaded yet). KeyError for unknown names.

    cfg may mix shared keys with a per-backend sub-dict: {"chunk_s": 20, "dfn3": {"chunk_s": 60}}.
    """
    key = str(name).strip().lower()
    cls = BACKENDS.get(key)
    if cls is None:
        raise KeyError(f"unknown denoise backend {name!r}; known: {', '.join(sorted(BACKENDS))}")
    return cls(_merge_cfg(cfg, key), logger)


def available_backends(cfg: Optional[dict] = None) -> Dict[str, Tuple[bool, str]]:
    """name -> (usable, reason). Never raises and never imports heavy dependencies."""
    out = {}
    for name, cls in BACKENDS.items():
        try:
            out[name] = cls.check_available(_merge_cfg(cfg, name))
        except Exception as exc:  # a broken environment must not break the listing
            out[name] = (False, f"availability check failed: {exc}")
    return out
