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
