"""Keep PyTorch's process-global SDPA backend flags from drifting between stages.

``torch.nn.attention.sdpa_kernel`` saves the *current* flags on entry and puts
them back on exit. When two threads (e.g. one BS-RoFormer worker per GPU) enter
and leave it interleaved, the second exit "restores" the first thread's
narrowed set and the process is left with, say, math/mem-efficient disabled.
A later fp32 model (Wav2Vec2 word alignment) then fails with "No available
kernel". Snapshot the flags once, and put them back at each stage boundary.
"""

_NAMES = ("flash", "mem_efficient", "math", "cudnn")


def capture_baseline():
    """Current SDPA backend flags, or an empty dict if torch is unavailable."""
    try:
        from torch.backends import cuda
        return {
            "flash": bool(cuda.flash_sdp_enabled()),
            "mem_efficient": bool(cuda.mem_efficient_sdp_enabled()),
            "math": bool(cuda.math_sdp_enabled()),
            "cudnn": bool(cuda.cudnn_sdp_enabled()),
        }
    except Exception:
        return {}


def restore_if_changed(baseline, logger=None, where: str = "") -> bool:
    """Reset flags that differ from ``baseline``; return True if any changed."""
    if not baseline:
        return False
    current = capture_baseline()
    changed = {n: (baseline[n], current[n]) for n in _NAMES
               if n in current and current[n] != baseline[n]}
    if not changed:
        return False
    from torch.backends import cuda
    setters = {
        "flash": cuda.enable_flash_sdp, "mem_efficient": cuda.enable_mem_efficient_sdp,
        "math": cuda.enable_math_sdp, "cudnn": cuda.enable_cudnn_sdp,
    }
    for name in changed:
        setters[name](baseline[name])
    if logger:
        detail = ", ".join(f"{n} {now}->{was}" for n, (was, now) in changed.items())
        logger.warning(
            f"[sdpa] backend flags drifted before {where or 'this point'} ({detail}); "
            "restored. Likely interleaved sdpa_kernel() contexts from concurrent threads.")
    return True
