"""The review page is the last step of a pass and fills in whatever exists.

A run that stops after diarization has segments and clips but no transcript.
The page is still the one tools/make_review_page.py writes; the ASR fields and
the processed audio are simply empty. These tests pin that, and the switch.
"""
import importlib.util
import json
import os
import sys
import types
import wave

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _review_module():
    path = os.path.join(ROOT, "tools", "make_review_page.py")
    spec = importlib.util.spec_from_file_location("make_review_page_final", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wav(path, seconds=0.2, rate=8000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


def _diarized_dir(tmp_path, name="ep01"):
    out = tmp_path / name
    stage = out / "02_diarization"
    stage.mkdir(parents=True)
    segments = [
        {"index": "00001", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"index": "00002", "start": 0.8, "end": 2.0, "speaker": "SPEAKER_01"},
    ]
    (stage / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
    _wav(str(stage / "audio" / "processed" / "00001_SPEAKER_00_0.000-1.000.wav"))
    _wav(str(stage / "audio" / "processed" / "00002_SPEAKER_01_0.800-2.000.wav"))
    return out


def _rows_in(page_text):
    marker = "const ROWS = "
    start = page_text.index(marker) + len(marker) if marker in page_text else None
    return start


def test_a_diarization_only_run_still_gets_the_page(tmp_path):
    mod = _review_module()
    out = _diarized_dir(tmp_path)
    info = {}
    page = mod.build_review_page(str(out), source="diarization", info=info)
    text = open(page, encoding="utf-8").read()
    assert os.path.basename(page) == "ep01_review.html"      # the same file name as before
    assert info["kind"] == "diarization"
    assert "ep01 (diarization only)" in text
    assert "SPEAKER_00" in text and "SPEAKER_01" in text
    assert text.count("data:audio/x-wav;base64,") + text.count("data:audio/wav;base64,") >= 2


def test_empty_asr_fields_do_not_break_the_rows(tmp_path):
    mod = _review_module()
    out = _diarized_dir(tmp_path)
    name, segments, _, kind = mod._load_source(str(out), "diarization")
    original, processed = mod._audio_index(str(out), name, kind)
    rows = mod._rows(segments, original, processed, [10 ** 9])
    assert kind == "diarization" and len(rows) == 2
    assert rows[0]["final"] == "" and rows[0]["whisper"] == ""
    assert rows[0]["audio_src"] and rows[0]["audio_out"] is None   # no processed audio yet


def test_a_stale_transcript_is_not_shown_for_a_run_that_did_not_export(tmp_path):
    mod = _review_module()
    out = _diarized_dir(tmp_path)
    (out / "ep01.json").write_text(
        json.dumps({"segments": [{"index": "9", "start": 0, "end": 1, "speaker": "X",
                                  "text": "old"}]}), encoding="utf-8")
    assert mod._load_source(str(out), "auto")[3] == "transcript"
    assert mod._load_source(str(out), "diarization")[3] == "diarization"


def test_transcript_requested_but_missing_falls_back_to_diarization(tmp_path):
    mod = _review_module()
    out = _diarized_dir(tmp_path)
    assert mod._load_source(str(out), "transcript")[3] == "diarization"


def test_nothing_on_disk_is_a_clear_exit(tmp_path):
    mod = _review_module()
    empty = tmp_path / "ep02"
    empty.mkdir()
    with pytest.raises(SystemExit):
        mod.build_review_page(str(empty), source="auto")


def test_an_unknown_source_is_refused(tmp_path):
    mod = _review_module()
    with pytest.raises(ValueError):
        mod._load_source(str(tmp_path), "everything")


def test_the_manifest_names_the_page(tmp_path):
    mod = _review_module()
    out = _diarized_dir(tmp_path)
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({"stages": {}, "final": {"json": "ep01.json"}}),
                        encoding="utf-8")
    assert mod.record_in_manifest(str(out), str(out / "ep01_review.html"), "diarization")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["final"] == {"json": "ep01.json", "review": "ep01_review.html"}
    assert data["review_page"] == {"file": "ep01_review.html", "kind": "diarization"}


def test_no_manifest_is_not_an_error(tmp_path):
    mod = _review_module()
    out = _diarized_dir(tmp_path)
    assert mod.record_in_manifest(str(out), "x.html", "diarization") is False


def test_the_page_switch_defaults_on_and_the_command_line_can_force_it():
    from utils.steps import review_page_enabled
    ns = types.SimpleNamespace
    assert review_page_enabled(ns()) is True                  # never set: on
    assert review_page_enabled(ns(review_page=None)) is True  # profile silent: on
    assert review_page_enabled(ns(review_page=True)) is True
    assert review_page_enabled(ns(review_page=False)) is False
