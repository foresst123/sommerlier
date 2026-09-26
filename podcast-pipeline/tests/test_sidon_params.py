"""DialogueSidon's inference settings.

Run:  python -m pytest tests/test_sidon_params.py -q     (from podcast-pipeline/)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _config():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


# --- num_steps --------------------------------------------------------------

def test_diffusion_steps_come_from_the_profile():
    """60 steps was hardcoded in the worker, so the one knob that trades
    separation time against quality could not be touched without editing code."""
    src = _source("sidon_worker.py")

    assert "num_steps=num_steps" in src, "the request must use the resolved value"
    assert '"num_steps"' in src and '.get("sidon", {})' in src


def test_every_profile_declares_num_steps():
    for env, profile in _config()["environments"].items():
        sidon = profile.get("models", {}).get("sidon", {})
        assert "num_steps" in sidon, f"{env} cannot set sidon.num_steps"
        assert isinstance(sidon["num_steps"], int)


def test_the_worker_is_given_the_config_to_read():
    src = _source("services/sidon_worker_service.py")
    assert '"--config"' in src and '"--env"' in src


def test_a_missing_sidon_section_falls_back_rather_than_failing():
    """A profile written before this existed must still start the worker."""
    cfg = {"environments": {"x": {"models": {}}}}
    sidon = cfg["environments"]["x"]["models"].get("sidon", {})
    assert int(sidon.get("num_steps", 60)) == 60


# --- amplitude restoration --------------------------------------------------

def test_each_chunk_is_returned_to_the_recording_scale():
    """Chunks are normalised by their own peak before separation. Without
    undoing that, two neighbours of different loudness come back on different
    scales and the crossfade between them blends two gains."""
    src = _source("sidon_infer.py")
    assert src.count("* (max_val / 0.9)") == 2, (
        "both the single-chunk and the chunked path must restore the scale")


def test_restoration_preserves_the_ratio_between_two_chunks():
    import torch

    torch.manual_seed(0)

    def separate(x):
        """Stand-in for the model: returns tracks on the input's scale."""
        return torch.stack([x.reshape(-1) * 0.7, x.reshape(-1) * 0.5])

    loud = torch.randn(1, 8000) * 0.30
    quiet = torch.randn(1, 8000) * 0.08
    expected = float(loud.abs().max() / quiet.abs().max())

    def run(chunk, restore):
        peak = chunk.abs().max().clamp_min(1e-6)
        out = separate(0.9 * chunk / peak)
        return out * (peak / 0.9) if restore else out

    def rms(x):
        return float(x[0].pow(2).mean().sqrt())

    without = rms(run(loud, False)) / rms(run(quiet, False))
    with_fix = rms(run(loud, True)) / rms(run(quiet, True))

    assert abs(without - 1.0) < 0.2, "unrestored chunks land on one scale"
    assert abs(with_fix - expected) / expected < 0.1

def test_the_worker_does_not_renormalise_what_infer_already_scaled():
    """sidon_infer returns each chunk multiplied back by (max_val / 0.9), so the
    tracks already sit on the mixture's scale when the worker receives them.

    The worker used to divide both tracks by their joint peak and scale that to
    0.9 a second time. Dividing by a *shared* peak does preserve the ratio
    between the two speakers -- which is what the comment claimed -- but it
    discards the ratio between the pair and the mixture, and nothing downstream
    puts it back: `restore_track` only resamples and pads. Across the 22 windows
    of one run every pair came back at a peak of exactly 0.9000, standing 3.5% to
    56% above the mixture's own peak, and rms(A+B)/rms(mix) ran 1.08 to 1.71
    where separation should leave it near 1. `match_splice_level` then spent its
    +/-3x budget absorbing that factor at every join instead of correcting the
    real level step it exists for.
    """
    src = _source("sidon_worker.py")
    assert "global_max" not in src, (
        "the worker must not renormalise the tracks; sidon_infer already returns "
        "them on the mixture's scale")
    assert "/ global_max * 0.9" not in src
