"""Turning stages on and off from the profile.

`steps` replaces reasoning about which flag gates which stage. Each key is one
stage; false skips it. The old flags still work for anything the profile does
not mention, so a config written before this exists behaves exactly as it did.

Run:  python -m pytest tests/test_steps.py -q     (from podcast-pipeline/)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.pipeline_service import PipelineService

STEPS = ("music_analysis", "music_removal", "music_removal_fallback",
         "cut_singing", "diarization", "separation", "asr", "captioning",
         "refinement", "export")


def _args(**kw):
    return argparse.Namespace(**kw)


# --- the switch --------------------------------------------------------------

def test_a_step_set_true_runs():
    assert PipelineService.step_enabled(_args(step_separation=True), "separation")


def test_a_step_set_false_is_skipped():
    assert not PipelineService.step_enabled(_args(step_separation=False), "separation")


def test_the_profile_wins_over_the_flag_it_replaced():
    """`tse` used to gate separation; `steps` is the newer, clearer answer."""
    args = _args(tse=True, step_separation=False)
    assert not PipelineService.step_enabled(args, "separation")


def test_an_unlisted_step_falls_back_to_its_old_flag():
    """A config written before `steps` existed must behave as it always did."""
    assert not PipelineService.step_enabled(_args(tse=False), "separation")
    assert PipelineService.step_enabled(_args(tse=True), "separation")
    assert not PipelineService.step_enabled(_args(qwen3omni=False), "captioning")


def test_a_step_with_neither_runs():
    """Silence is not a reason to skip: an unknown step is on."""
    assert PipelineService.step_enabled(_args(), "diarization")
    assert PipelineService.step_enabled(_args(), "export")


def test_one_flag_can_gate_two_steps_independently():
    """`panns` gated both the sweep and the removal; they are separable now."""
    args = _args(panns=True, step_music_removal=False)
    assert PipelineService.step_enabled(args, "music_analysis")
    assert not PipelineService.step_enabled(args, "music_removal")


# --- the profile -------------------------------------------------------------

def test_both_profiles_list_every_step():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for name, profile in config["environments"].items():
        steps = profile.get("steps")
        assert steps is not None, f"{name} has no steps block"
        for step in STEPS:
            assert step in steps, f"{name} is missing '{step}'"
            assert isinstance(steps[step], bool), f"{name}.{step} is not a boolean"


def test_steps_are_kept_apart_from_the_tuning_values():
    """`pipeline` mixes switches with thresholds; `steps` is only switches, so
    a stage's name can never collide with a number that shares it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for name, profile in config["environments"].items():
        assert set(profile["steps"]) & set(profile["pipeline"]) == set(), name


# --- skipping a load-bearing stage ends the run ------------------------------

def test_the_service_stops_rather_than_running_without_diarization():
    """Nothing downstream has an input without it, so a missing stage must end
    the run rather than produce a file with a hole where a stage was."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "services", "pipeline_service.py"),
                  encoding="utf-8").read()
    assert 'for _required in ("diarization", "asr", "export"):' in source
    assert '"stopped_before": _required' in source


def test_separation_off_still_produces_segments_of_the_same_shape():
    """ASR and the exporters read enhanced_audio either way; only the overlap
    replacement is missing."""
    import numpy as np

    from schemas.audio import AudioData
    from schemas.segment import Segment
    import services.separation_service as sep

    service = sep.TargetExtractionService.__new__(sep.TargetExtractionService)
    segments = [Segment(index="00001", start=0.0, end=1.0, speaker="1"),
                Segment(index="00002", start=1.0, end=2.0, speaker="2")]
    audio = AudioData(name="t", waveform=np.ones(24000 * 3, dtype=np.float32),
                      sample_rate=24000, duration=3.0, audio_segment=None)

    out = service.passthrough(segments, audio)
    assert len(out) == 2
    assert all(s.enhanced_audio is not None and len(s.enhanced_audio) for s in out)
    assert all(not s.tse for s in out)


def test_the_per_segment_pass_is_off_by_default():
    """It predates the waveform pass and does the same job a stage later, one
    segment at a time -- redundant with a map, coarser without one."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for name, profile in config["environments"].items():
        assert profile["steps"]["music_removal_fallback"] is False, name


def test_the_fallback_is_skipped_for_either_reason():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "services", "pipeline_service.py"),
                  encoding="utf-8").read()
    assert 'not self.step_enabled(args, "music_removal_fallback")' in source
    assert "turned off in the profile" in source


# --- the loader has to reach the same verdict --------------------------------

def _stub_model_modules():
    """Import services.model_loader without the model stack behind it.

    It imports eleven model wrappers at module level, each pulling a framework
    that has no bearing on which switch decides whether a model loads.
    """
    import types

    import importlib.util

    # Only what is genuinely absent. A stub left in sys.modules for something
    # real -- librosa, pandas -- would be inherited by every test that runs
    # after this one, and they would fail somewhere unrelated.
    for name in ("faster_whisper", "whisperx", "whisperx.audio", "whisperx.asr",
                 "panns_inference", "pyannote", "pyannote.audio", "librosa",
                 "onnxruntime", "soundfile", "pandas"):
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue
        except (ImportError, ValueError):
            pass
        sys.modules[name] = types.ModuleType(name)
    for name in ("models.whisper", "models.whisper_wrapper", "models.phowhisper",
                 "models.silero_vad", "models.pyannote", "models.diarizen_model",
                 "models.pyannote_embedding", "models.tse_model", "models.panns",
                 "models.demucs", "models.qwen3_omni", "models.qwen3_asr"):
        module = types.ModuleType(name)
        for attr in ("WhisperASR", "PhoWhisperASR", "SileroVAD", "PyannoteDiarizer",
                     "DiariZenDiarizer", "PyannoteEmbedder", "TargetSpeakerExtractor",
                     "PANNSDetector", "DemucsRemover", "Qwen3OmniCaptioner",
                     "Qwen3ASRClient", "load_asr_model"):
            setattr(module, attr, type(attr, (), {"__init__": lambda self, *a, **k: None}))
        sys.modules.setdefault(name, module)

    from services.model_loader import ModelLoader
    return ModelLoader


def _loader(monkeypatch, **flags):
    ModelLoader = _stub_model_modules()

    # Patched on the loader module rather than left to the stubs: by the time
    # the whole suite runs, models.panns may already be imported for real, and
    # the real detector would try to build a 312MB checkpoint.
    import services.model_loader as ml
    monkeypatch.setattr(ml, "PANNSDetector", lambda *a, **k: object())

    loader = ModelLoader.__new__(ModelLoader)
    loader.args = _args(**flags)
    loader.models = {}
    loader.logger = None
    loader.device_1 = "cpu"
    loader.device_2 = "cpu"
    loader.config = {"environments": {"kaggle": {"models": {"demucs": {}}}}}
    return loader


def test_the_profile_alone_is_enough_to_load_the_tagger(monkeypatch):
    """`steps.music_analysis: true` with no --panns used to load nothing.

    The stage then ran, found no detector, and reported an empty music map --
    a run that says it swept the audio and did not.
    """
    loader = _loader(monkeypatch, env="kaggle", step_music_analysis=True)
    loader.load_panns()
    assert "panns" in loader.models


def test_turning_the_step_off_leaves_the_tagger_unloaded(monkeypatch):
    loader = _loader(monkeypatch, env="kaggle", panns=True, step_music_analysis=False)
    loader.load_panns()
    assert "panns" not in loader.models


def test_the_old_flag_still_loads_it(monkeypatch):
    loader = _loader(monkeypatch, env="kaggle", panns=True)
    loader.load_panns()
    assert "panns" in loader.models
