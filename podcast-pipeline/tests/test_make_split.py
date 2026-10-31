import json
import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from tools import make_split as MS  # noqa: E402
from tools import analyze_arms_detail as AD  # noqa: E402


def test_split_is_deterministic_disjoint_and_covers_everything():
    names = [f"01_Src_{i:02d}" for i in range(28)]
    a = MS.make_split(names, seed=0)
    assert a == MS.make_split(names, seed=0)
    assert set(a["A"]) | set(a["B"]) == set(names)
    assert not set(a["A"]) & set(a["B"])
    assert len(a["A"]) == len(a["B"]) == 14
    assert MS.make_split(names, seed=1) != a          # another seed gives another split


def test_odd_count_gives_the_extra_sample_to_A():
    s = MS.make_split([f"s{i}" for i in range(5)], seed=3)
    assert len(s["A"]) == 3 and len(s["B"]) == 2


def test_too_few_samples_is_an_error():
    with pytest.raises(ValueError):
        MS.make_split(["only"], seed=0)


def test_load_split_rejects_overlap(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"A": ["x", "y"], "B": ["y"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        MS.load_split(str(bad))


def write_wav(path, seconds=6):
    t = np.arange(int(16000 * seconds)) / 16000
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def make_sample(root, name):
    d = root / name
    d.mkdir(parents=True)
    write_wav(d / "clip.wav")
    labels = {"segments": [{"start": 0.0, "end": 4.0, "speaker": "A"}], "tolerance_s": 0.25}
    (d / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (d / "segments_nemotron_dz.json").write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 4.0, "speaker": "0"}]}), encoding="utf-8")


def test_labelled_samples_lists_only_folders_with_labels(tmp_path):
    make_sample(tmp_path, "01_a")
    (tmp_path / "02_nolabel").mkdir()
    assert MS.labelled_samples(str(tmp_path)) == ["01_a"]


def test_analysis_on_one_half_only_reads_that_half(tmp_path, capsys):
    root = tmp_path / "arms"
    for name in ["01_a", "01_b", "01_c", "01_d"]:
        make_sample(root, name)
    split = MS.make_split(MS.labelled_samples(str(root)), seed=0)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"seed": 0, **split}), encoding="utf-8")

    rc = AD.main([str(root), "--arms", "nemotron_dz", "--split", str(split_path), "--half", "B",
                  "--boot", "50", "--top", "2", "--out", str(tmp_path / "outB")])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"Mẫu: {len(split['B'])}" in out
    with open(tmp_path / "outB" / "per_sample.csv", encoding="utf-8") as fh:
        reported = {line.split(",")[0] for line in fh.read().splitlines()[1:]}
    assert reported == set(split["B"])                  # half A never reaches the report


def test_analysis_refuses_split_without_half(tmp_path):
    with pytest.raises(SystemExit):
        AD.main([str(tmp_path), "--arms", "nemotron_dz", "--split", str(tmp_path / "x.json")])
