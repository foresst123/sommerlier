"""Models are loaded when their stage runs, not all at once up front.

Run:  python -m pytest tests/test_lazy_model_loading.py -q     (from podcast-pipeline/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


# --- main.py no longer front-loads ------------------------------------------

def test_main_builds_the_loader_without_loading_anything():
    """Loading every model before the first stage ran meant peak VRAM was the
    sum of all of them -- DiariZen 5.15GB plus Sidon 2.62GB plus PhoWhisper,
    Whisper, Demucs and the captioner -- rather than the largest pair. A run
    resuming from a checkpoint paid for models it then never called."""
    src = _source("main.py")
    assert "ModelLoader(config, args" in src, "the loader is still constructed"
    for loader in ("load_base_models", "load_diarization_models",
                   "load_separation_models", "load_music_models",
                   "load_asr_models", "load_caption_model"):
        assert f"model_loader.{loader}(" not in src, (
            f"main.py must not call {loader}; PipelineService loads it when "
            "that stage runs")


def test_services_receive_the_loader_not_a_model():
    """A model captured at construction is None for every stage that has not
    loaded yet -- and stays None after it does."""
    src = _source("main.py")
    assert "model_loader.get(" not in src, (
        "services must take model_loader= and resolve models on use")


# --- PipelineService loads per stage ----------------------------------------

def test_every_stage_loads_its_own_models():
    src = _source("services/pipeline_service.py")
    for group in ("base", "diarization", "separation", "music", "asr", "caption"):
        assert f'self._load("{group}")' in src, f"stage {group} never loads"


def test_loading_sits_inside_the_compute_branch():
    """A stage that reads its checkpoint must not load the model it will not
    call, which is the whole point of resuming."""
    src = _source("services/pipeline_service.py")
    for group, marker in (("diarization", 'checkpoint.exists("diarization")'),
                          ("separation", 'checkpoint.exists("separation")'),
                          ("asr", 'checkpoint.exists("asr")')):
        load_at = src.index(f'self._load("{group}")')
        guard_at = src.index(marker)
        assert guard_at < load_at, (
            f"{group} loads before its checkpoint is checked")


# --- the loaders are safe to call repeatedly --------------------------------

def test_loaders_are_idempotent():
    """Under stage-major execution run() is re-entered once per file, so each
    loader is reached again with its models already resident."""
    src = _source("services/model_loader.py")
    for guard in ('if "vad" in self.models',
                  'if "diarizer" in self.models',
                  'if "separator" in self.models',
                  'if "panns" in self.models',
                  'if "phowhisper" in self.models',
                  'if "captioner" in self.models'):
        assert guard in src, f"missing early return: {guard}"


# Importing ModelLoader pulls in faster_whisper, which is not installed in the
# test environment, so the idempotence guard is asserted from the source above
# rather than exercised here.


# --- services resolve late --------------------------------------------------

def test_a_service_sees_a_model_loaded_after_it_was_built():
    """This is the failure the properties exist to prevent: construct the
    service first, load second, and the service must still see the model."""
    from services.music_service import MusicService

    class Loader:
        def __init__(self):
            self.models = {}

        def get(self, name):
            return self.models.get(name)

    loader = Loader()
    svc = MusicService(model_loader=loader)
    assert svc.panns is None, "nothing is loaded yet"

    loader.models["panns"] = object()
    assert svc.panns is loader.models["panns"], (
        "the service captured None instead of resolving on use")


def test_an_explicit_model_still_wins():
    """Passing a model directly is how the tests and any older caller work."""
    from services.music_service import MusicService

    sentinel = object()
    svc = MusicService(panns_model=sentinel, model_loader=None)
    assert svc.panns is sentinel
