"""Both channels of the clean pair filtered: user_clean_chain beside assistant_clean_chain.

Fake backends and synthetic audio only (the helpers of test_assistant_clean_channel).

Run:  python -m pytest tests/test_user_clean_channel.py -q     (from podcast-pipeline/)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sf = pytest.importorskip("soundfile")

from services.conversation_export_service import ConversationExportService  # noqa: E402
from test_assistant_clean_channel import _export, fake_filter  # noqa: E402,F401
from test_fullduplex_output import FinderLLM, _only  # noqa: E402


def test_both_chains_filter_both_channels_of_the_clean_pair(tmp_path, fake_filter):
    result = _export(tmp_path, assistant_clean_chain="half", user_clean_chain="half")
    folder, doc = _only(tmp_path)
    user, _ = sf.read(str(folder / "user.wav"), dtype="float32")
    user_clean, _ = sf.read(str(folder / "user_clean.wav"), dtype="float32")
    assistant_clean, _ = sf.read(str(folder / "assistant_clean.wav"), dtype="float32")
    mid = len(user) // 2
    assert user_clean[mid] == pytest.approx(user[mid] * 0.5, abs=2e-4)
    stereo_clean, _ = sf.read(str(folder / "stereo_2ch_clean.wav"), dtype="float32", always_2d=True)
    assert np.array_equal(stereo_clean[:, 0], assistant_clean)       # assistant is the left ear by default
    assert np.array_equal(stereo_clean[:, 1], user_clean)
    record = doc["assistant_clean"]
    assert record["written"] is True and record["user"]["written"] is True
    assert result.exports[0]["files"]["user_clean"].endswith("/user_clean.wav")
    assert doc["verification"]["ok"] is True
    assert doc["verification"]["checks"]["stereo_clean_user_channel_is_user"] is True


def test_without_a_user_chain_the_user_channel_stays_as_it_was(tmp_path, fake_filter):
    _export(tmp_path, assistant_clean_chain="half")
    folder, doc = _only(tmp_path)
    assert not (folder / "user_clean.wav").exists()
    assert "user" not in doc["assistant_clean"]
    user, _ = sf.read(str(folder / "user.wav"), dtype="float32")
    stereo_clean, _ = sf.read(str(folder / "stereo_2ch_clean.wav"), dtype="float32", always_2d=True)
    assert np.array_equal(stereo_clean[:, 1], user)


def test_a_failed_user_filter_writes_no_clean_pair_at_all(tmp_path, fake_filter):
    """One unfiltered channel in a pair called clean would be a lie: nothing clean is written."""
    _export(tmp_path, assistant_clean_chain="half", user_clean_chain="boom")
    folder, doc = _only(tmp_path)
    for name in ("assistant_clean.wav", "user_clean.wav", "stereo_2ch_clean.wav"):
        assert not (folder / name).exists()
    assert (folder / "stereo_2ch.wav").exists()
    record = doc["assistant_clean"]
    assert record["written"] is False and record["reason"] == "user_filter_failed"
    assert record["user"]["errors"]


def test_the_postprocess_runs_on_each_channel_when_asked(tmp_path, fake_filter, monkeypatch):
    from utils import ambience_bed
    calls = []

    def fake(clean, before, sr, params, islands=None):
        calls.append(len(clean))
        return clean * 0.5, {"islands": 1, "restored": 0, "level_adjusted": 0, "zeroed_fraction": 0.0}

    monkeypatch.setattr(ambience_bed, "postprocess_assistant", fake)
    _export(tmp_path, assistant_clean_chain="half", user_clean_chain="half", clean_postprocess=True)
    folder, doc = _only(tmp_path)
    assert len(calls) == 2
    assert doc["assistant_clean"]["postprocess"]["islands"] == 1
    assert doc["assistant_clean"]["user"]["postprocess"]["islands"] == 1


def test_no_postprocess_unless_asked(tmp_path, fake_filter, monkeypatch):
    from utils import ambience_bed
    monkeypatch.setattr(ambience_bed, "postprocess_assistant",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    _export(tmp_path, assistant_clean_chain="half", user_clean_chain="half")
    folder, doc = _only(tmp_path)
    assert "postprocess" not in doc["assistant_clean"]


def test_a_malformed_user_chain_fails_when_the_service_is_built(fake_filter):
    with pytest.raises(ValueError):
        ConversationExportService(FinderLLM([]), user_clean_chain="half>>half")
    with pytest.raises(ValueError, match="user_clean_chain"):
        ConversationExportService(FinderLLM([]), user_clean_chain="rf_vocals_resurrection>nonexist")
