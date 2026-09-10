"""Compatibility shims for PyTorch behaviour that differs across versions.

Used by the main pipeline. The worker scripts each carry their own copy of the
shim because they run in separate virtualenvs, where importing from here would
require putting podcast-pipeline on sys.path and shadowing common package names
like 'utils' and 'models'.
"""

_PATCHED = False


def install_torch_load_shim():
    """Force ``torch.load`` back to ``weights_only=False``.

    PyTorch 2.6 flipped the default to ``weights_only=True``, which rejects the
    pickled checkpoints several models in this pipeline ship (pyannote, DiariZen,
    Silero and several exported graphs carry non-tensor objects). Call this before
    importing any model code; repeated calls are harmless.
    """
    global _PATCHED
    if _PATCHED:
        return

    import torch

    if hasattr(torch, "torch_version") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

    original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _patched_load
    _PATCHED = True
