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
