"""The performance path must stay opt-in, honest and out of the baseline's way.

Each test here pins a defect that was live in the working tree and would come
back silently: a config loader that lost its body, two threads sharing one
separator's temp file, a GPU declared free while the weights were still on it.
"""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diarizen_worker
from services.music_service import MusicService
from utils import performance_config


# --- the diarizen worker still reads its own config ------------------------

def test_the_diarizen_block_survives_the_round_trip(tmp_path):
    """A loader that returns None drops every threshold without saying so."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"environments": {"kaggle": {"models": {
        "diarizen": {"ahc_threshold": 0.71, "batch_size": 8}}}}}))

    loaded = diarizen_worker._load_diarizen_config(str(config), "kaggle")

    assert loaded == {"ahc_threshold": 0.71, "batch_size": 8}


def test_a_missing_config_is_an_empty_block_not_a_none(tmp_path):
    assert diarizen_worker._load_diarizen_config(
        str(tmp_path / "absent.json"), "kaggle") == {}


def test_placement_is_off_unless_the_profile_asks_for_it():
    assert diarizen_worker._place_components(object(), {})["split"] is False


# --- one separator, one caller ---------------------------------------------

class _CountingSeparator:
    """Stands in for BSRoformerRemover, which writes a fixed in.wav."""

    def __init__(self):
        self.inside = 0
        self.max_inside = 0
        self._lock = threading.Lock()

    def separate_segment(self, reference, sr):
        with self._lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        # Proportional to the span, which is what desynchronises the two
        # threads. A constant hold keeps them in lockstep and hides the bug.
        time.sleep(len(reference) / sr * 0.05)
        with self._lock:
            self.inside -= 1
        return reference


class _Pool:
    def __init__(self, models):
        self.models = models


class _Span:
    def __init__(self, spans):
        self.spans = spans


def test_a_separator_is_never_driven_by_two_threads_at_once():
    """Picking a model by job index does not pin it to one thread."""
    import numpy as np
    from schemas.audio import AudioData
    from utils.music_map import MUSIC

    sr = 16000
    # Uneven spans: equal ones would hide the interleaving that breaks the
    # index-based assignment.
    lengths = [1.0, 0.6, 1.4, 0.55, 1.2, 0.5]
    spans, cursor = [], 0.0
    for length in lengths:
        spans.append((cursor, cursor + length, MUSIC))
        cursor += length + 0.1

    models = [_CountingSeparator(), _CountingSeparator()]
    service = MusicService(model_loader=None, logger=None)
    service.bs_roformer = _Pool(models)
    waveform = np.zeros(int(cursor * sr) + sr, dtype=np.float32)
    audio = AudioData(waveform=waveform, sample_rate=sr, name="t.wav",
                      audio_segment=None, duration=len(waveform) / sr)

    service.strip_music_spans(audio, _Span(spans))

    for model in models:
        assert model.max_inside <= 1, (
            "two threads entered one separator; they share its work directory "
            "and its fixed in.wav")


def test_two_files_sharing_one_music_service_do_not_double_book_a_separator():
    """A checkout queue rebuilt on every call only protects ONE call.

    Cross-file overlap means two files can each be inside their own
    strip_music_spans() call at the same time, on the ONE MusicService the
    whole run shares. If each call built its own queue from the same
    underlying instances, both would believe they held a separator
    exclusively and could drive it concurrently -- the same defect as above,
    triggered by two files instead of two threads inside one file.
    """
    import numpy as np
    from schemas.audio import AudioData
    from utils.music_map import MUSIC

    sr = 16000
    service = MusicService(model_loader=None, logger=None)
    service.bs_roformer = _Pool([_CountingSeparator()])  # exactly one instance

    def run_one_file(duration):
        spans = [(0.0, duration, MUSIC)]
        waveform = np.zeros(int(duration * sr) + sr, dtype=np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr, name="t.wav",
                          audio_segment=None, duration=len(waveform) / sr)
        service.strip_music_spans(audio, _Span(spans))

    a = threading.Thread(target=run_one_file, args=(1.4,))
    b = threading.Thread(target=run_one_file, args=(1.2,))
    a.start(); b.start()
    a.join(); b.join()

    model = service.bs_roformer.models[0]
    assert model.max_inside <= 1, (
        "two files drove the same BS-RoFormer instance at once; the checkout "
        "queue must be shared across strip_music_spans() calls, not rebuilt "
        "inside each one")


def test_the_checkout_queue_is_rebuilt_when_the_model_set_changes():
    """A stale queue from an unloaded batch pass must not outlive its models."""
    service = MusicService(model_loader=None, logger=None)
    first = [object()]
    second = [object()]

    q1 = service._checkout_queue(first)
    q1.get()  # drain it, simulating a job in flight when the pass ended
    q2 = service._checkout_queue(second)

    assert q2 is not q1
    assert q2.get() is second[0]


# --- the config is validated and reported ----------------------------------

def test_the_shipped_profiles_resolve_without_a_single_complaint():
    """Every profile must validate cleanly, whether the path is on or off.

    A key the validator rejects falls back to its default, so a profile that
    limps through with warnings is running settings nobody chose.
    """
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "config.json").read_text())
    seen = 0
    for name, profile in config.get("environments", {}).items():
        if "performance" not in profile:
            continue
        seen += 1
        resolved = performance_config.resolve(profile)
        assert resolved["_problems"] == [], f"{name}: {resolved['_problems']}"
        assert isinstance(resolved["enabled"], bool)
    assert seen, "no profile carries a performance block any more"


@pytest.mark.parametrize("profile, expected", [
    ({"performance": {"enabld": True}}, "not a known setting"),
    ({"performance": {"stages": {"nope": {}}}}, "not a known stage"),
    ({"performance": {"ram_soft_fraction": "x"}}, "not valid"),
    ({"performance": {"stages": {"refinement": {"placement": "wat"}}}}, "must be one of"),
])
def test_a_bad_key_is_reported_rather_than_swallowed(profile, expected):
    problems = performance_config.resolve(profile)["_problems"]
    assert any(expected in problem for problem in problems), problems


def test_the_override_only_reports_when_it_changes_something():
    off = {"performance": {"enabled": False}}
    assert performance_config.resolve(off, enabled_override=True)["enabled"] is True
    assert performance_config.resolve(off, enabled_override=None)["enabled"] is False


# --- the fingerprint keeps the two paths apart -----------------------------

def test_a_disabled_run_keeps_the_baseline_directory():
    assert performance_config.fingerprint(
        performance_config.resolve({"performance": {"enabled": False}})) == "off"


def test_a_semantic_change_moves_the_output_elsewhere():
    def token(**asr):
        return performance_config.fingerprint(performance_config.resolve(
            {"performance": {"enabled": True, "stages": {"asr": asr}}}))

    assert token(true_batching=True) != token(true_batching=False)


def test_pure_scheduling_settings_do_not_invalidate_a_checkpoint():
    """Worker counts and telemetry describe how, not what."""
    def token(**top):
        return performance_config.fingerprint(performance_config.resolve(
            {"performance": dict(enabled=True, **top)}))

    assert token(max_pending_jobs=32) == token(max_pending_jobs=64)
    assert token(telemetry_interval_seconds=1.0) == token(telemetry_interval_seconds=5.0)


# --- the per-file copies still share one checkout queue --------------------

def test_per_file_copies_of_the_music_service_share_one_checkout_queue():
    """parallel_stage_view hands every file copy.copy(music_svc).

    A queue kept in an attribute that _checkout_queue REBINDS is per-copy the
    moment the copies exist: each file builds its own, and two files book the
    same separator again -- with a lock in place and the single-instance test
    above still green. The state has to be a dict the copies share.
    """
    import copy

    import numpy as np
    from schemas.audio import AudioData
    from utils.music_map import MUSIC

    sr = 16000
    base = MusicService(model_loader=None, logger=None)
    base.bs_roformer = _Pool([_CountingSeparator()])
    # Copied BEFORE any queue exists: that is the order the batch pass uses.
    left, right = copy.copy(base), copy.copy(base)

    def run_one_file(service, duration):
        waveform = np.zeros(int(duration * sr) + sr, dtype=np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr, name="t.wav",
                          audio_segment=None, duration=len(waveform) / sr)
        service.strip_music_spans(audio, _Span([(0.0, duration, MUSIC)]))

    a = threading.Thread(target=run_one_file, args=(left, 1.4))
    b = threading.Thread(target=run_one_file, args=(right, 1.2))
    a.start(); b.start()
    a.join(); b.join()

    assert left._checkout is right._checkout
    assert base.bs_roformer.models[0].max_inside <= 1, (
        "two per-file copies drove one BS-RoFormer at once; the queue is "
        "being rebuilt per copy")


def test_a_view_is_isolated_where_it_must_be_and_shared_where_it_must_be():
    from services.pipeline_service import PipelineService

    music = MusicService(model_loader=None, logger=None)
    pipeline = PipelineService(*(None,) * 8)
    pipeline.music_svc = music
    pipeline.noise_track = "the noise of file A"

    view = pipeline.parallel_stage_view("music")

    assert view is not pipeline
    assert view.noise_track is None, "file B would read file A's noise curve"
    assert view.music_svc is not music, "each file gets its own bs_roformer slot"
    assert view.music_svc._checkout is music._checkout, "but ONE checkout queue"
    assert view._tagger_lock is pipeline._tagger_lock, "and one file in SSLAM"
    assert pipeline.parallel_stage_view("asr") is pipeline


def test_sslam_runs_under_its_lock_and_the_model_load_does_not():
    """The loader has its own lock. Taking the tagger lock first and then the
    loader's, in one file, while another does the reverse, would deadlock."""
    src = (Path(__file__).resolve().parents[1]
           / "services" / "pipeline_service.py").read_text(encoding="utf-8")
    load_at = src.index('self._load("tagger")')
    lock_at = src.index('with (tagger_lock if tagger_lock is not None')
    build_at = src.index("build_maps(", lock_at)

    assert load_at < lock_at < build_at


# --- the loader builds each model once, however many files ask -------------

def test_two_files_loading_the_tagger_together_build_it_once(monkeypatch):
    """Idempotent is not the same as safe to call twice at once: both callers
    saw the tagger missing, both built one, and the second overwrote the first
    in the dict -- so the first was never unloaded and kept its VRAM."""
    import importlib
    import types

    built = []

    class SlowTagger:
        def __init__(self, device=None):
            time.sleep(0.05)  # wide enough for the second caller to arrive
            built.append(device)

    stubs = {
        "models.whisper_wrapper": {"WhisperASR": object},
        "models.phowhisper": {"PhoWhisperASR": object},
        "models.silero_vad": {"SileroVAD": object},
        "models.pyannote": {"PyannoteDiarizer": object},
        "models.diarizen_model": {"DiariZenDiarizer": object},
        "models.bss_model": {"BssSeparator": object},
        "models.sslam": {"SSLAMDetector": SlowTagger},
        "models.qwen3_omni": {"Qwen3OmniCaptioner": object},
        "models.qwen3_asr": {"Qwen3ASRClient": object},
        "services.qwen3_worker_service": {"Qwen3WorkerService": object},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("services.model_loader", None)
    try:
        loader_module = importlib.import_module("services.model_loader")
        args = types.SimpleNamespace(gpu_1=0, gpu_2=1, env="kaggle")
        loader = loader_module.ModelLoader({}, args)

        threads = [threading.Thread(target=loader.load_tagger) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        # The stubs above are gone after this test; a module built against
        # them must not be left behind for the next one to import.
        sys.modules.pop("services.model_loader", None)

    assert len(built) == 1
    assert list(loader.models) == ["tagger"]


# --- SSLAM and BS-RoFormer land on different cards -------------------------

class _Log:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


def test_the_default_keeps_bs_roformer_beside_sslam():
    """Without the flag nothing overlaps files, so nothing gains from moving."""
    assert performance_config.resolve_music_devices(
        "cuda:0", "cuda:1", perf_enabled=True) == ["cuda:0"]


def test_cross_file_overlap_moves_the_sole_instance_off_sslams_card():
    """SSLAM stays on device_1; the one BS-RoFormer goes to device_2 so file
    N's removal and file N+1's classification do not fight for one card."""
    assert performance_config.resolve_music_devices(
        "cuda:0", "cuda:1", perf_enabled=True,
        cross_file_overlap=True) == ["cuda:1"]


def test_a_second_instance_per_file_still_uses_both_cards():
    assert performance_config.resolve_music_devices(
        "cuda:0", "cuda:1", perf_enabled=True,
        max_separator_workers=2) == ["cuda:0", "cuda:1"]


def test_asking_for_both_overlap_modes_picks_one_and_says_so():
    """Both want the second card for a different job. Overlap wins, because
    without it no file ever runs beside another, and the choice is reported
    rather than silently made."""
    log = _Log()

    devices = performance_config.resolve_music_devices(
        "cuda:0", "cuda:1", perf_enabled=True, max_separator_workers=2,
        cross_file_overlap=True, logger=log)

    assert devices == ["cuda:1"]
    assert len(log.warnings) == 1
    assert "cross_file_overlap" in log.warnings[0]
    assert "max_separator_workers" in log.warnings[0]


@pytest.mark.parametrize("kwargs", [
    {"cross_file_overlap": True},
    {"max_separator_workers": 2},
    {"cross_file_overlap": True, "max_separator_workers": 2},
])
def test_a_single_card_or_a_disabled_profile_never_moves_anything(kwargs):
    """One GPU has no second card to move to, and a profile with the
    performance path off must behave exactly as the baseline did."""
    assert performance_config.resolve_music_devices(
        "cuda:0", "cuda:0", perf_enabled=True, **kwargs) == ["cuda:0"]
    assert performance_config.resolve_music_devices(
        "cuda:0", "cuda:1", perf_enabled=False, **kwargs) == ["cuda:0"]


def test_the_shipped_profiles_do_not_ask_for_both_music_modes():
    """A profile asking for both would warn on every run and quietly ignore
    one of its own settings."""
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "config.json").read_text())
    seen = 0
    for name, profile in config["environments"].items():
        music = profile.get("performance", {}).get("stages", {}).get("music")
        if music is None:
            continue
        seen += 1
        both = (music.get("cross_file_overlap")
                and music.get("max_separator_workers", 1) > 1)
        assert not both, f"{name} asks for cross_file_overlap AND max_separator_workers>1"
    assert seen, "no profile carries a music performance block any more"
