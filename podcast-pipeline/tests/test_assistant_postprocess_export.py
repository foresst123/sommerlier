"""The assistant channel after the filter: the postprocess hook in _clean_assistant, and the quality numbers of _verify.

utils.ambience_bed.postprocess_assistant is replaced by fakes here (its real body is not written yet). The filter is the
HalfBackend of test_assistant_clean_channel.py: the filtered track is the input times 0.5.

Run:  python -m pytest tests/test_assistant_postprocess_export.py -q     (from podcast-pipeline/)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sf = pytest.importorskip("soundfile")

import utils.ambience_bed as ambience_bed  # noqa: E402
from services.conversation_export_service import ConversationExportService  # noqa: E402
from test_assistant_clean_channel import HalfBackend, fake_filter  # noqa: E402,F401
from test_fullduplex_output import SR, FinderLLM  # noqa: E402

REPORT = {"islands": 2, "restored": 1, "level_adjusted": 1, "zeroed_fraction": 0.3}


def _assistant(seconds=2.0):
    """A sentence, then 0.5 s of digital zero, then a sentence: the shape the export hands the chain."""
    n = int(seconds * SR)
    track = np.zeros(n, dtype=np.float32)
    track[:n // 4] = 0.2 * np.where(np.arange(n // 4) % 2 == 0, 1.0, -1.0)
    track[3 * n // 4:] = 0.2 * np.where(np.arange(n - 3 * n // 4) % 2 == 0, 1.0, -1.0)
    return track


def _service(chain="half", **settings):
    return ConversationExportService(FinderLLM([]), assistant_clean_chain=chain, **settings)


def _spy(monkeypatch, behaviour=None):
    """Replaces postprocess_assistant; records every call."""
    calls = []

    def fake(clean, before, sr, params, islands=None):
        calls.append({"clean": np.array(clean), "before": np.array(before), "sr": sr, "params": params})
        if behaviour is not None:
            raise behaviour
        return (np.full(len(clean), 0.1, dtype=np.float32), dict(REPORT))

    monkeypatch.setattr(ambience_bed, "postprocess_assistant", fake)
    return calls


# --- _clean_assistant ------------------------------------------------------------------

def test_mode_none_does_not_call_the_postprocess_and_writes_the_old_clean_pair(tmp_path, fake_filter, monkeypatch):
    calls = _spy(monkeypatch)
    svc = _service()                                   # user_ambience defaults to mode "none"
    assistant = _assistant()
    user = np.full(len(assistant), 0.05, dtype=np.float32)
    record, files = svc._clean_assistant(str(tmp_path), assistant, user, SR)

    assert calls == []
    assert "postprocess" not in record
    assert record["written"] is True and record["errors"] == []
    assert files == {"assistant_clean": "assistant_clean.wav", "stereo_2ch_clean": "stereo_2ch_clean.wav"}
    clean, _ = sf.read(str(tmp_path / "assistant_clean.wav"), dtype="float32")
    assert np.allclose(clean, assistant * 0.5, atol=2e-4)


def test_raw_mask_hands_the_filtered_track_and_the_pre_filter_track_to_the_postprocess(tmp_path, fake_filter,
                                                                                    monkeypatch):
    calls = _spy(monkeypatch)
    svc = _service(user_ambience={"mode": "raw_mask"})
    assistant = _assistant()
    user = np.full(len(assistant), 0.05, dtype=np.float32)
    record, files = svc._clean_assistant(str(tmp_path), assistant, user, SR)

    assert len(calls) == 1
    call = calls[0]
    assert np.allclose(call["clean"], assistant * 0.5, atol=2e-4)   # what came out of the chain
    assert np.array_equal(call["before"], assistant)                # what went in: the zeros between sentences
    assert call["sr"] == SR
    assert call["params"] is svc.user_ambience and call["params"].mode == "raw_mask"

    # The file carries the postprocessed audio, not the filter's output.
    written, _ = sf.read(str(tmp_path / "assistant_clean.wav"), dtype="float32")
    assert np.allclose(written, 0.1, atol=1e-4)
    stereo, _ = sf.read(str(tmp_path / "stereo_2ch_clean.wav"), dtype="float32", always_2d=True)
    assert np.allclose(stereo[:, 0], written, atol=1e-4)            # assistant on the left (the default)
    assert np.allclose(stereo[:, 1], user, atol=1e-4)

    assert record["postprocess"] == REPORT
    assert record["written"] is True and "error" not in record["postprocess"]
    assert files == {"assistant_clean": "assistant_clean.wav", "stereo_2ch_clean": "stereo_2ch_clean.wav"}


def test_a_failing_postprocess_keeps_the_filtered_track_and_records_the_error(tmp_path, fake_filter, monkeypatch):
    _spy(monkeypatch, behaviour=RuntimeError("boom in the expander"))
    svc = _service(user_ambience={"mode": "raw_mask"})
    assistant = _assistant()
    user = np.full(len(assistant), 0.05, dtype=np.float32)
    record, files = svc._clean_assistant(str(tmp_path), assistant, user, SR)

    assert record["written"] is True
    assert "boom in the expander" in record["postprocess"]["error"]
    assert files == {"assistant_clean": "assistant_clean.wav", "stereo_2ch_clean": "stereo_2ch_clean.wav"}
    clean, _ = sf.read(str(tmp_path / "assistant_clean.wav"), dtype="float32")
    assert np.allclose(clean, assistant * 0.5, atol=2e-4)           # the unprocessed filter output
    stereo, _ = sf.read(str(tmp_path / "stereo_2ch_clean.wav"), dtype="float32", always_2d=True)
    assert np.allclose(stereo[:, 0], clean, atol=1e-4)


def test_a_postprocess_that_changes_the_length_is_an_error_not_a_bad_file(tmp_path, fake_filter, monkeypatch):
    def short(clean, before, sr, params, islands=None):
        return np.zeros(len(clean) - 10, dtype=np.float32), {}

    monkeypatch.setattr(ambience_bed, "postprocess_assistant", short)
    svc = _service(user_ambience={"mode": "raw_mask"})
    assistant = _assistant()
    record, _ = svc._clean_assistant(str(tmp_path), assistant, assistant, SR)

    assert "error" in record["postprocess"]
    clean, _ = sf.read(str(tmp_path / "assistant_clean.wav"), dtype="float32")
    assert len(clean) == len(assistant)
    assert np.allclose(clean, assistant * 0.5, atol=2e-4)


def test_the_unfinished_real_postprocess_falls_back_without_breaking_the_export(tmp_path, fake_filter):
    # No monkeypatch: the module's body may still be NotImplementedError. The export must go on either way.
    svc = _service(user_ambience={"mode": "raw_mask"})
    assistant = _assistant()
    record, _ = svc._clean_assistant(str(tmp_path), assistant, assistant, SR)

    assert record["written"] is True
    if "error" in record.get("postprocess", {}):
        assert (tmp_path / "assistant_clean.wav").exists()


# --- _verify quality -------------------------------------------------------------------

def _pair(tmp_path, user, assistant):
    n = len(user)
    mixture = np.clip(user + assistant, -1, 1).astype(np.float32)
    for name, track in (("mixture", mixture), ("user", user), ("assistant", assistant)):
        sf.write(str(tmp_path / f"{name}.wav"), track, SR, subtype="PCM_16")
    files = {"mixture": "mixture.wav", "user": "user.wav", "assistant": "assistant.wav"}
    return files, n


def _alternating(n, amp):
    return (amp * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)).astype(np.float32)


def test_verify_reports_the_longest_zero_run_of_the_user_file(tmp_path):
    n = SR                                              # one second
    user = _alternating(n, 0.1)
    user[SR // 4: SR // 4 + SR // 2] = 0.0             # 0.5 s of digital zero
    assistant = _alternating(n, 0.2)
    files, n = _pair(tmp_path, user, assistant)
    svc = _service(chain="none")
    result = svc._verify(str(tmp_path), files, SR, n, "left")

    assert result["ok"] is True
    assert result["quality"]["longest_zero_run_s"] == pytest.approx(0.5)
    assert "quality" not in result["checks"]


def test_verify_quality_does_not_change_ok(tmp_path):
    svc = _service(chain="none")
    n = SR
    clean_user = _alternating(n, 0.1)
    zero_user = _alternating(n, 0.1)
    zero_user[:SR // 2] = 0.0
    assistant = _alternating(n, 0.2)

    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    files, n = _pair(dir_a, clean_user, assistant)
    a = svc._verify(str(dir_a), files, SR, n, "left")
    files, n = _pair(dir_b, zero_user, assistant)
    b = svc._verify(str(dir_b), files, SR, n, "left")

    assert a["quality"]["longest_zero_run_s"] == 0.0
    assert b["quality"]["longest_zero_run_s"] == pytest.approx(0.5)
    assert a["ok"] is True and b["ok"] is True
    assert a["checks"] == b["checks"]


def test_verify_floor_and_zero_share_of_the_assistant(tmp_path):
    n = SR
    user = _alternating(n, 0.1)
    assistant = _alternating(n, 0.2)
    assistant[:SR // 2] = 0.0                           # half of the frames are digital zero
    files, n = _pair(tmp_path, user, assistant)
    quality = _service(chain="none")._verify(str(tmp_path), files, SR, n, "left")["quality"]

    assert quality["assistant_zero_frame_share"] == pytest.approx(0.5, abs=0.02)
    assert quality["assistant_floor_dbfs"] == pytest.approx(-120.0)  # the clamp: the 10th percentile is a zero frame


def test_verify_floor_of_a_continuous_assistant_is_its_level(tmp_path):
    n = SR
    files, n = _pair(tmp_path, _alternating(n, 0.1), _alternating(n, 0.2))
    quality = _service(chain="none")._verify(str(tmp_path), files, SR, n, "left")["quality"]

    assert quality["assistant_zero_frame_share"] == 0.0
    assert quality["assistant_floor_dbfs"] == pytest.approx(20 * np.log10(0.2), abs=0.1)


def test_a_failing_quality_drops_the_key_and_keeps_the_verdict(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no numbers today")

    monkeypatch.setattr(ConversationExportService, "_quality_numbers", staticmethod(broken))
    n = SR
    files, n = _pair(tmp_path, _alternating(n, 0.1), _alternating(n, 0.2))
    result = _service(chain="none")._verify(str(tmp_path), files, SR, n, "left")

    assert "quality" not in result
    assert result["ok"] is True


# --- _verify: the clean pair ------------------------------------------------------------

CLEAN_CHECKS = ("stereo_clean_assistant_channel_is_assistant_clean", "stereo_clean_user_channel_is_user")


def _clean_pair(tmp_path, user, assistant_clean, assistant_channel="left"):
    """The raw mixture/user/assistant plus the clean assistant and the clean stereo pair, as _clean_assistant writes."""
    files, n = _pair(tmp_path, user, assistant_clean)
    pair = (assistant_clean, user) if assistant_channel == "left" else (user, assistant_clean)
    sf.write(str(tmp_path / "assistant_clean.wav"), assistant_clean, SR, subtype="PCM_16")
    sf.write(str(tmp_path / "stereo_2ch_clean.wav"), np.column_stack(pair), SR, subtype="PCM_16")
    files.update({"assistant_clean": "assistant_clean.wav", "stereo_2ch_clean": "stereo_2ch_clean.wav"})
    return files, n


def test_a_right_clean_pair_passes_both_clean_checks(tmp_path):
    user = _alternating(SR, 0.1)
    assistant_clean = _alternating(SR, 0.05)
    files, n = _clean_pair(tmp_path, user, assistant_clean)
    result = _service(chain="none")._verify(str(tmp_path), files, SR, n, "left")

    assert result["checks"][CLEAN_CHECKS[0]] is True
    assert result["checks"][CLEAN_CHECKS[1]] is True
    assert result["ok"] is True


@pytest.mark.parametrize("assistant_channel", ["left", "right"])
def test_a_clean_pair_with_the_assistant_channel_changed_by_one_sample_fails_that_check(tmp_path, assistant_channel):
    user = _alternating(SR, 0.1)
    assistant_clean = _alternating(SR, 0.05)
    files, n = _clean_pair(tmp_path, user, assistant_clean, assistant_channel)
    stereo, _ = sf.read(str(tmp_path / "stereo_2ch_clean.wav"), dtype="int16", always_2d=True)
    mine = 0 if assistant_channel == "left" else 1
    stereo[SR // 2, mine] += 1                          # one 16-bit step in the assistant's ear
    sf.write(str(tmp_path / "stereo_2ch_clean.wav"), stereo, SR, subtype="PCM_16")
    result = _service(chain="none")._verify(str(tmp_path), files, SR, n, assistant_channel)

    assert result["checks"][CLEAN_CHECKS[0]] is False
    assert result["checks"][CLEAN_CHECKS[1]] is True
    assert result["ok"] is False


def test_a_clean_pair_with_the_user_channel_changed_fails_the_user_check(tmp_path):
    user = _alternating(SR, 0.1)
    assistant_clean = _alternating(SR, 0.05)
    files, n = _clean_pair(tmp_path, user, assistant_clean)
    stereo, _ = sf.read(str(tmp_path / "stereo_2ch_clean.wav"), dtype="int16", always_2d=True)
    stereo[SR // 3, 1] -= 1                             # one 16-bit step in the user's ear
    sf.write(str(tmp_path / "stereo_2ch_clean.wav"), stereo, SR, subtype="PCM_16")
    result = _service(chain="none")._verify(str(tmp_path), files, SR, n, "left")

    assert result["checks"][CLEAN_CHECKS[0]] is True
    assert result["checks"][CLEAN_CHECKS[1]] is False
    assert result["ok"] is False


def test_without_the_clean_files_the_clean_checks_are_not_there(tmp_path):
    files, n = _pair(tmp_path, _alternating(SR, 0.1), _alternating(SR, 0.2))
    result = _service(chain="none")._verify(str(tmp_path), files, SR, n, "left")

    assert not any(name in result["checks"] for name in CLEAN_CHECKS)


# --- _verify quality: the level step of the user file -----------------------------------

def test_the_user_level_step_is_the_largest_jump_between_50_ms_windows(tmp_path):
    # 0.5 s at 2048 / 32768 (-24 dBFS), then 0.5 s at 20480 / 32768 (-4 dBFS): a 20 dB step at 0.5 s.
    user = np.concatenate([_alternating(SR // 2, 2048 / 32768.0), _alternating(SR // 2, 20480 / 32768.0)])
    files, n = _pair(tmp_path, user, _alternating(SR, 0.2))
    quality = _service(chain="none")._verify(str(tmp_path), files, SR, n, "left")["quality"]

    assert quality["user_max_rms_step_db"] == pytest.approx(20.0, abs=0.01)


def test_the_user_level_step_skips_digital_silence_and_is_none_without_two_levels(tmp_path):
    # Loud, digital zero, then loud at the same level: the silence is skipped, so the step is 0.
    # 2400 + 3200 + 2400 samples: whole 50 ms windows (400 samples) on each side of the silence.
    user = np.concatenate([_alternating(2400, 2048 / 32768.0), np.zeros(3200, dtype=np.float32),
                           _alternating(2400, 2048 / 32768.0)])
    files, n = _pair(tmp_path, user, _alternating(SR, 0.2))
    quality = _service(chain="none")._verify(str(tmp_path), files, SR, n, "left")["quality"]
    assert quality["user_max_rms_step_db"] == pytest.approx(0.0, abs=0.01)

    silent = tmp_path / "silent"
    silent.mkdir()
    files, n = _pair(silent, np.zeros(SR, dtype=np.float32), _alternating(SR, 0.2))
    quality = _service(chain="none")._verify(str(silent), files, SR, n, "left")["quality"]
    assert quality["user_max_rms_step_db"] is None
