"""Swapping the separator without touching the assignment and QC around it.

`TargetSpeakerExtractor` produces two tracks and then decides which belongs to
whom. Only the first is model-specific, and these pin the seam between them --
including the part that matters most: a target-conditioned separator was told
whose voice to extract, so the assignment must not run at all.

Run:  python -m pytest tests/test_separation_backends.py -q   (from podcast-pipeline/)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.separation_backends import (
    USEF_ENROLL_SAMPLES,
    USEF_MIX_SAMPLES,
    USEF_SR,
    UsefOnnxBackend,
    make_backend,
)


# --- choosing a backend -----------------------------------------------------

def test_usef_is_the_default():
    assert make_backend(None, process=None, temp_dir="/tmp").name == "usef"


def test_names_are_case_and_space_insensitive():
    assert make_backend("  USEF ").name == "usef"


def test_an_unknown_name_fails_rather_than_falling_back():
    """A typo that quietly kept the old model would make two runs look
    comparable when they are not."""
    with pytest.raises(ValueError, match="unknown separator"):
        make_backend("tfgridnet-large")


def test_blind_and_targeted_backends_declare_themselves():
    assert make_backend("usef").ordered is True


# --- enrollment reaches USEF in the one shape its graph accepts -------------

def _clips(seconds, count=1, sr=24000):
    return [np.full(int(sr * seconds), 0.1, dtype=np.float32)] * count


def test_a_short_enrollment_is_padded_to_eight_seconds():
    out = UsefOnnxBackend._prepare_enrollment(_clips(3.0), 24000)
    assert len(out) == USEF_ENROLL_SAMPLES


def test_a_long_enrollment_is_trimmed():
    out = UsefOnnxBackend._prepare_enrollment(_clips(20.0), 24000)
    assert len(out) == USEF_ENROLL_SAMPLES


def test_several_mined_clips_are_joined_before_trimming():
    """A speaker whose clean audio is spread over short spans still fills the
    window; taking only the first clip would leave it mostly padding."""
    out = UsefOnnxBackend._prepare_enrollment(_clips(2.0, count=4), 24000)
    assert len(out) == USEF_ENROLL_SAMPLES
    assert np.abs(out[:USEF_ENROLL_SAMPLES]).mean() > 0.05


def test_a_missing_enrollment_is_an_error_not_a_silent_pass():
    with pytest.raises(ValueError, match="enrollment"):
        UsefOnnxBackend._prepare_enrollment(None, 24000)
    with pytest.raises(ValueError, match="empty"):
        UsefOnnxBackend._prepare_enrollment([], 24000)


# --- the fixed 2s window ----------------------------------------------------

class _FakeSession:
    """Stands in for the ONNX graph, asserting the shapes it would require."""

    def __init__(self):
        self.windows = []

    def get_inputs(self):
        return [type("I", (), {"name": "mixture"}), type("I", (), {"name": "enrollment"})]

    def run(self, _outputs, feeds):
        mixture = feeds["mixture"]
        assert mixture.shape == (1, USEF_MIX_SAMPLES), mixture.shape
        assert feeds["enrollment"].shape == (1, USEF_ENROLL_SAMPLES)
        self.windows.append(mixture.shape)
        return [mixture * 2.0]


def _backend_with_fake():
    backend = UsefOnnxBackend()
    backend._session = _FakeSession()
    backend._input_names = ["mixture", "enrollment"]
    return backend


@pytest.mark.parametrize("seconds,expected_windows", [(0.4, 1), (2.0, 1), (5.0, 3), (20.0, 10)])
def test_audio_is_chunked_into_two_second_windows(seconds, expected_windows):
    """TF-GridNet's export bakes its unfold constants in, so the window cannot
    be negotiated: longer audio is chunked and concatenated."""
    backend = _backend_with_fake()
    samples = int(USEF_SR * seconds)
    mixture = np.random.default_rng(0).standard_normal(samples).astype(np.float32)

    out = backend._extract_one(mixture, np.zeros(USEF_ENROLL_SAMPLES, dtype=np.float32))

    assert len(backend._session.windows) == expected_windows
    assert len(out) == samples, "the tail must not be padded into the result"


def test_a_partial_final_window_is_padded_in_but_trimmed_out():
    backend = _backend_with_fake()
    samples = USEF_MIX_SAMPLES + 100
    out = backend._extract_one(np.ones(samples, dtype=np.float32),
                               np.zeros(USEF_ENROLL_SAMPLES, dtype=np.float32))
    assert len(out) == samples
    assert out[-1] != 0.0, "real audio, not the zero padding, should survive"


def test_the_declared_rate_is_returned_so_the_caller_can_resample():
    """USEF runs at 8 kHz while the pipeline carries 16 kHz, so the backend
    returns its rate and the caller resamples back rather than assuming."""
    backend = _backend_with_fake()
    enroll = [np.full(USEF_SR, 0.1, dtype=np.float32)]
    _a, _b, out_sr = backend.separate(np.zeros(USEF_SR, dtype=np.float32), USEF_SR,
                                      enroll_A=enroll, enroll_B=enroll)
    assert out_sr == USEF_SR


def test_a_targeted_backend_runs_once_per_speaker():
    backend = _backend_with_fake()
    enroll = [np.full(USEF_SR, 0.1, dtype=np.float32)]
    backend.separate(np.zeros(USEF_MIX_SAMPLES, dtype=np.float32), USEF_SR,
                     enroll_A=enroll, enroll_B=enroll)
    assert len(backend._session.windows) == 2, "one pass per enrollment"


# --- the seam in TargetSpeakerExtractor -------------------------------------

def test_the_extractor_skips_assignment_for_a_targeted_backend():
    """The measured reason this matters: similarity here sits at p50 0.58 where
    natural speech scores 0.70-0.90, so every assignment decision is made on
    thin evidence. A targeted separator removes the decision entirely."""
    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models", "tse_model.py"), encoding="utf-8").read()
    assert 'if getattr(self.backend, "ordered", False):' in source
    assert "self.backend.separate(" in source


def test_both_profiles_declare_a_separator():
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for name, profile in config["environments"].items():
        assert profile["models"]["tse"]["separator"] == "usef", name


def test_only_the_target_conditioned_backend_ships():
    """DialogueSidon was removed. It separated blind, which meant ECAPA had to
    decide which track was whom -- a decision that sat at p50 0.58 similarity
    where natural speech scores 0.70-0.90, and which three separate repair
    paths existed to cope with. USEF is told who to extract, so the question
    does not arise."""
    from models.separation_backends import BACKENDS
    assert set(BACKENDS) == {"usef"}


def test_every_shipped_backend_knows_its_track_order():
    """`ordered` is what the caller branches on. A backend that left it False
    would silently re-enable the assignment and repair paths."""
    from models.separation_backends import BACKENDS
    for name, cls in BACKENDS.items():
        assert cls.ordered is True, name


def test_an_unknown_separator_is_refused_not_quietly_replaced():
    """A typo that fell back to a default would make two runs look comparable
    when they are not."""
    import pytest as _pytest
    from models.separation_backends import make_backend
    with _pytest.raises(ValueError):
        make_backend("sidon")
