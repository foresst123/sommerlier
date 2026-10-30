"""The assistant's clean track: a denoise chain on the assistant only, written beside the originals.

Fake backends and synthetic audio only. The profile checks read config.json.

Run:  python -m pytest tests/test_assistant_clean_channel.py -q     (from podcast-pipeline/)
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sf = pytest.importorskip("soundfile")

from services import denoise_service  # noqa: E402
from services.conversation_export_service import ConversationExportService  # noqa: E402
from test_fullduplex_output import (  # noqa: E402
    A, SR, FakeSeparation, FinderLLM, _only, _quiet, _region, _talk, _wave)
from test_config_driven import _resolve  # noqa: E402
from utils.excise import TimelineMap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HalfBackend:
    """Halves the level; 'boom' raises like a model that ran out of memory."""

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def load(self):
        pass

    def unload(self):
        pass

    def enhance(self, wav, sr):
        if self.behaviour == "boom":
            raise RuntimeError("out of memory")
        return (wav * 0.5).astype(np.float32)


@pytest.fixture
def fake_filter(monkeypatch):
    from models.denoise import BACKENDS, chain_id, parse_chain

    def get_backend(name, cfg=None, logger=None):
        if name not in ("half", "boom"):
            raise KeyError(name)
        return HalfBackend(name)

    monkeypatch.setattr(denoise_service, "_load_api",
                        lambda: (get_backend, parse_chain, chain_id))
    # The service checks chain names against the registry, so the fakes are listed there too.
    monkeypatch.setitem(BACKENDS, "half", HalfBackend)
    monkeypatch.setitem(BACKENDS, "boom", HalfBackend)


def _export(tmp_path, **settings):
    """One two-channel excerpt: the assistant is speaker B, held at -0.25 by the fake separator."""
    segs = _talk()
    speech = [SimpleNamespace(speaker=A, start=0.0, end=1.0, audio=None,
                              bss_spans=[], bss_failed_spans=[])]
    settings.setdefault("min_score", 0)
    svc = ConversationExportService(FinderLLM([_region(segs)]), **settings)
    return svc.run(segs, timeline=TimelineMap(), noise=_quiet(), music_map=None,
                   waveform=_wave(), sample_rate=SR, out_dir=str(tmp_path), base_name="ep 01",
                   speech_segments=speech, separation_service=FakeSeparation(0.25, -0.25))


# --- export output -------------------------------------------------------------------

def test_no_chain_writes_exactly_the_old_files(tmp_path):
    result = _export(tmp_path)
    folder, doc = _only(tmp_path)
    assert sorted(p.name for p in folder.iterdir() if "clean" in p.name) == []
    assert "assistant_clean" not in doc
    assert "assistant_clean" not in result.exports[0]["files"]
    assert "stereo_2ch_clean" not in result.exports[0]["files"]
    assert (folder / "stereo_2ch.wav").exists()


def test_a_chain_writes_a_filtered_assistant_and_a_stereo_with_the_user_unchanged(tmp_path, fake_filter):
    result = _export(tmp_path, assistant_clean_chain="half")
    folder, doc = _only(tmp_path)
    assistant, sr = sf.read(str(folder / "assistant.wav"), dtype="float32")
    clean, sr_clean = sf.read(str(folder / "assistant_clean.wav"), dtype="float32")
    assert sr_clean == sr and len(clean) == len(assistant)
    mid = len(clean) // 2
    assert clean[mid] == pytest.approx(assistant[mid] * 0.5, abs=2e-4)
    assert not np.array_equal(clean, assistant)

    stereo, _ = sf.read(str(folder / "stereo_2ch.wav"), dtype="float32", always_2d=True)
    stereo_clean, _ = sf.read(str(folder / "stereo_2ch_clean.wav"), dtype="float32", always_2d=True)
    assert stereo_clean.shape == stereo.shape
    # Assistant is in the left ear by default, so the user is the right channel.
    assert np.array_equal(stereo_clean[:, 1], stereo[:, 1])
    assert np.array_equal(stereo_clean[:, 0], clean)

    record = doc["assistant_clean"]
    assert record["written"] is True and record["errors"] == [] and record["chain"] == "half"
    files = result.exports[0]["files"]
    assert files["assistant_clean"] == f"{result.exports[0]['folder']}/assistant_clean.wav"
    assert files["stereo_2ch_clean"] == f"{result.exports[0]['folder']}/stereo_2ch_clean.wav"


def test_a_failed_filter_writes_no_clean_files_and_says_why(tmp_path, fake_filter):
    result = _export(tmp_path, assistant_clean_chain="boom")
    folder, doc = _only(tmp_path)
    assert not (folder / "assistant_clean.wav").exists()
    assert not (folder / "stereo_2ch_clean.wav").exists()
    assert (folder / "assistant.wav").exists() and (folder / "stereo_2ch.wav").exists()
    record = doc["assistant_clean"]
    assert record["written"] is False and record["reason"] == "filter_failed"
    assert record["errors"] and "out of memory" in record["errors"][0]
    assert "assistant_clean" not in result.exports[0]["files"]


def test_a_malformed_chain_fails_when_the_service_is_built(fake_filter):
    with pytest.raises(ValueError):
        ConversationExportService(FinderLLM([]), assistant_clean_chain="half>>half")


def test_a_pseudo_step_cannot_lead_the_clean_chain(fake_filter):
    with pytest.raises(ValueError):
        ConversationExportService(FinderLLM([]), assistant_clean_chain="blend50>half")


def test_an_unknown_backend_name_fails_when_the_service_is_built():
    with pytest.raises(ValueError, match="nonexist"):
        ConversationExportService(FinderLLM([]),
                                  assistant_clean_chain="rf_vocals_resurrection>nonexist")
    ConversationExportService(FinderLLM([]), assistant_clean_chain="rf_vocals_resurrection>blend50")


# --- profiles -----------------------------------------------------------------------

def _config():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)["environments"]


def test_a100_filters_once_before_diarization_and_runs_no_music_step():
    a100 = _config()["a100"]
    assert a100["pipeline"]["denoise"] == "wpe>rf_crowd>blend50"
    assert a100["pipeline"]["denoise_pre_separation"] == "none"
    assert a100["pipeline"]["denoise_pre_asr"] == "none"
    assert a100["steps"]["music_analysis"] is False
    assert a100["steps"]["music_removal"] is False
    assert a100["steps"]["cut_music"] is False
    assert a100["pipeline"]["music"] is False


def test_a100_clean_chain_is_configured_and_valid():
    from models.denoise import BACKENDS, parse_chain
    from models.denoise.pseudo import validate_chain_tokens
    exports = _config()["a100"]["conversation_exports"]
    chain = exports["assistant_clean_chain"]
    # No blend on the two RoFormer steps; MP-SENet keeps 80% of its output plus 20% of what it was given
    # (dry20, not the raw audio); hp60 last clears rumble and hum that dry20 mixes back in
    assert chain == "rf_crowd>rf_vocals_resurrection>mpsenet_dns>dry20>hp60"
    assert exports["user_clean_chain"] == chain
    tokens = parse_chain(chain)
    validate_chain_tokens(tokens)
    assert all(t in BACKENDS or t.startswith(("blend", "dry", "hp")) for t in tokens)
    for spec in (_config()["a100"]["pipeline"]["denoise"], chain):
        validate_chain_tokens(parse_chain(spec))


def test_kaggle_is_unchanged_by_the_a100_switch():
    kaggle = _config()["kaggle"]
    assert kaggle["pipeline"]["denoise"] == "none"
    assert kaggle["steps"]["music_removal"] is True
    assert "conversation_exports" not in kaggle


def test_a100_resolves_to_the_new_settings_through_main():
    args = _resolve(["--audio", "x.mp3", "--env", "a100"])
    assert args["denoise"] == "wpe>rf_crowd>blend50"
    assert args["step_music_removal"] is False
    assert args["step_music_analysis"] is False
    assert args["step_cut_music"] is False
    assert args["music"] is False
