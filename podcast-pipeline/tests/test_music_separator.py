"""Swapping the vocal-isolation model without MusicService noticing.

Demucs and BS-RoFormer answer the same three questions, so the choice is a
profile setting. What these pin is the seam: the interface both must satisfy,
and the two failure paths where a wrong answer would enter the dataset silently.

Worth recording alongside: on this corpus music removal ran on 1 of 932
segments in one file and 0 of 200 in the other -- there is 30s of music in 50
minutes. The swap matters for a corpus with more music, not for this one.

Run:  python -m pytest tests/test_music_separator.py -q   (from podcast-pipeline/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bs_roformer import DEFAULT_MODEL, BSRoformerRemover


def _remover(**kw):
    return BSRoformerRemover(device=None, logger=None, **kw)


# --- the interface MusicService relies on -----------------------------------

def test_it_answers_the_same_three_questions_as_demucs():
    from models.demucs import DemucsRemover
    for name in ("separate_full", "separate_segment", "unload"):
        assert hasattr(BSRoformerRemover, name), name
        assert hasattr(DemucsRemover, name), name


def test_demucs_tuning_knobs_are_accepted_and_ignored():
    """The same profile block drives either model, so the loader does not have
    to branch on which keys apply."""
    assert _remover(segment=10, overlap=0.1).model_filename == DEFAULT_MODEL


def test_the_checkpoint_is_named_not_left_to_the_library_default():
    """audio-separator's default model changes between releases; a run has to
    be reproducible from the profile alone."""
    assert DEFAULT_MODEL.endswith(".ckpt")
    assert "bs_roformer" in DEFAULT_MODEL


def test_the_checkpoint_can_be_overridden():
    assert _remover(model_filename="other.ckpt").model_filename == "other.ckpt"


# --- the two paths where a wrong answer would be silent ---------------------

def test_a_failed_segment_keeps_the_mixture_rather_than_going_silent():
    """Silence here would enter the dataset labelled as speech."""
    remover = _remover()
    remover._run = lambda audio, sr: None          # separation unavailable
    audio = np.full(1000, 0.25, dtype=np.float32)
    assert np.array_equal(remover.separate_segment(audio, 24000), audio)


def test_a_failed_full_pass_returns_none_so_the_caller_can_fall_back():
    remover = _remover()
    remover._run = lambda audio, sr: None
    assert remover.separate_full(np.zeros(1000, dtype=np.float32), 24000) is None


def test_empty_audio_is_refused_before_the_model_loads():
    """_get_model would download weights; an empty array must not trigger that."""
    remover = _remover()
    remover._get_model = lambda: (_ for _ in ()).throw(AssertionError("must not load"))
    assert remover._run(np.array([], dtype=np.float32), 24000) is None


# --- stem selection ---------------------------------------------------------

def test_the_vocal_stem_is_picked_by_name_not_by_position():
    """Order is undocumented, and taking the instrumental stem would be a
    total, silent failure -- speech replaced by backing track."""
    import inspect
    source = inspect.getsource(BSRoformerRemover._run)
    assert '"vocal" in os.path.basename(p).lower()' in source


def test_unload_is_safe_before_anything_was_loaded():
    remover = _remover()
    remover.unload()
    assert remover._separator is None


# --- the profile switch -----------------------------------------------------

def test_both_profiles_name_a_music_separator():
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for name, profile in config["environments"].items():
        assert profile["models"]["demucs"]["model"] in ("demucs", "bs_roformer"), name


def test_the_loader_strips_model_before_passing_kwargs():
    """`model` names the class; the rest of the block is constructor arguments.
    Forwarding it would reach DemucsRemover as an argument it does not take."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "services", "model_loader.py"), encoding="utf-8").read()
    assert 'demucs_cfg.pop("model", None)' in source
    assert "demucs_cfg = dict(" in source, "the profile dict must be copied before popping"
