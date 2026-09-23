"""SDPA backend flags are process-global; interleaved sdpa_kernel() contexts can
leave them stuck, so the pipeline restores them at every stage boundary."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from utils.sdpa_guard import capture_baseline, restore_if_changed  # noqa: E402


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, **kw):
        self.warnings.append(msg)


def _flags():
    cuda = torch.backends.cuda
    return {
        "flash": cuda.flash_sdp_enabled(), "mem_efficient": cuda.mem_efficient_sdp_enabled(),
        "math": cuda.math_sdp_enabled(), "cudnn": cuda.cudnn_sdp_enabled(),
    }


@pytest.fixture
def clean_flags():
    saved = _flags()
    yield
    cuda = torch.backends.cuda
    cuda.enable_flash_sdp(saved["flash"])
    cuda.enable_mem_efficient_sdp(saved["mem_efficient"])
    cuda.enable_math_sdp(saved["math"])
    cuda.enable_cudnn_sdp(saved["cudnn"])


def test_interleaved_sdpa_kernel_contexts_leave_flags_stuck_and_guard_restores(clean_flags):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    baseline = capture_baseline()
    first = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION])
    second = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    second.__exit__(None, None, None)
    assert _flags() != baseline  # the race: exiting both does not restore the start state

    logger = _Logger()
    assert restore_if_changed(baseline, logger, "test") is True

    assert _flags() == baseline
    assert len(logger.warnings) == 1
    assert "math" in logger.warnings[0] and "test" in logger.warnings[0]


def test_guard_is_silent_when_flags_are_unchanged(clean_flags):
    baseline = capture_baseline()
    logger = _Logger()
    assert restore_if_changed(baseline, logger, "test") is False
    assert logger.warnings == []
