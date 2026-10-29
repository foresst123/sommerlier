import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import compare_arms as C  # noqa: E402


def seg(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


REF = [seg(0, 5, "A"), seg(5, 10, "B")]


def test_same_turns_under_swapped_labels_score_zero():
    hyp = [seg(0, 5, "x"), seg(5, 10, "y")]
    r = C.compare(REF, hyp, 10.0, collar=0.0)
    assert (r["miss"], r["fa"], r["conf"], r["der"]) == (0.0, 0.0, 0.0, 0.0)
    assert r["ref_speakers"] == r["arm_speakers"] == 2
    assert r["speaker_recall"] == {"A": 1.0, "B": 1.0}


def test_speech_the_arm_lacks_is_a_miss_and_speech_it_adds_is_a_false_alarm():
    r = C.compare(REF, [seg(0, 5, "A"), seg(5, 8, "B")], 10.0, collar=0.0)
    assert r["miss"] == pytest.approx(20.0) and r["fa"] == 0.0 and r["arm_s"] == 8.0
    assert r["speaker_recall"]["B"] == pytest.approx(0.6)
    r = C.compare(REF, REF + [seg(10, 11, "C")], 12.0, collar=0.0)
    assert r["fa"] == pytest.approx(10.0) and r["miss"] == 0.0 and r["arm_speakers"] == 3


def test_two_reference_speakers_merged_into_one_label_are_confusion():
    r = C.compare(REF, [seg(0, 10, "A")], 10.0, collar=0.0)
    assert r["conf"] == pytest.approx(50.0) and r["miss"] == 0.0 and r["fa"] == 0.0
    assert r["arm_speakers"] == 1 and r["ref_speakers"] == 2


def test_the_collar_leaves_out_frames_around_the_reference_edges():
    hyp = [seg(0, 5.2, "A"), seg(5.2, 10, "B")]               # the turn change is 0.2 s late
    assert C.compare(REF, hyp, 10.0, collar=0.0)["conf"] == pytest.approx(2.0)
    assert C.compare(REF, hyp, 10.0, collar=0.25)["conf"] == 0.0


def test_short_reference_segments_are_checked_for_speech_and_for_the_right_speaker():
    ref = [seg(0, 5, "A"), seg(3.0, 3.5, "B")]                # a half-second backchannel inside A's turn
    r = C.compare(ref, [seg(0, 5, "x")], 6.0, collar=0.0)
    assert r["short_n"] == 1 and r["short_any"] == 100.0 and r["short_spk"] == 0.0
    r = C.compare(ref, [seg(0, 5, "x"), seg(3.0, 3.5, "y")], 6.0, collar=0.0)
    assert r["short_spk"] == 100.0


def test_run_reads_every_sample_folder_and_prints_a_table(tmp_path, capsys):
    for name in ("01_a", "02_b"):
        d = tmp_path / name
        d.mkdir()
        (d / "segments_diarizen_auto.json").write_text(json.dumps({"segments": REF}), encoding="utf-8")
        (d / "segments_nemotron_dz.json").write_text(
            json.dumps({"segments": [seg(0, 5, "x"), seg(5, 8, "y")]}), encoding="utf-8")
    (tmp_path / "03_no_ref").mkdir()
    (tmp_path / "03_no_ref" / "segments_nemotron_dz.json").write_text("{}", encoding="utf-8")
    rows = C.run(str(tmp_path), "diarizen_auto", None, 0.0)
    assert [r["sample"] for r in rows] == ["01_a", "02_b"] and rows[0]["arm"] == "nemotron_dz"
    csv_path = tmp_path / "t.csv"
    assert C.main([str(tmp_path), "--collar", "0", "--csv", str(csv_path)]) == 0
    out = capsys.readouterr().out
    assert "reference: diarizen_auto" in out and "== 01_a" in out and "speaker recall" in out
    assert csv_path.read_text(encoding="utf-8").startswith("sample,arm")
    assert C.main([str(tmp_path / "03_no_ref"), "--ref", "nothing"]) == 1


def test_postprocess_merges_the_raw_arms_but_not_the_reference_or_the_pipeline_arms(tmp_path):
    ref = [seg(0, 5, "A"), seg(5.3, 10, "B")]
    raw = [seg(0, 2, "x"), seg(2.3, 5, "x"), seg(5.3, 10, "y")]          # a 0.3 s gap inside x's turn
    d = tmp_path / "01_a"
    d.mkdir()
    for name, rows in (("diarizen_auto", ref), ("nemotron_dz", raw), ("nemotron", raw)):
        (d / f"segments_{name}.json").write_text(json.dumps({"segments": rows}), encoding="utf-8")
    plain = {r["arm"]: r for r in C.run(str(tmp_path), "diarizen_auto", None, 0.0)}
    merged = {r["arm"]: r for r in C.run(str(tmp_path), "diarizen_auto", None, 0.0, postprocess=True)}
    assert plain["nemotron_dz"]["miss"] > 0.0                             # the gap counts as missed speech
    assert merged["nemotron_dz"]["miss"] == 0.0                           # merged like the pipeline would
    assert merged["nemotron"]["miss"] == plain["nemotron"]["miss"] > 0.0   # arm A is left as it is
    assert C.pipeline_postprocess([seg(0, 2, "x"), seg(2.3, 4, "x")]) == [seg(0.0, 4.0, "x")]
