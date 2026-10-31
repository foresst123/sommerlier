import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from tools import compare_plda_runs as C  # noqa: E402


def make_root(root, arm_segments):
    """arm_segments: {sample: [(start, end, speaker), ...]} for arm nemotron_dz; labels are fixed."""
    labels = {"segments": [{"start": 0.0, "end": 4.0, "speaker": "A"},
                           {"start": 5.0, "end": 9.0, "speaker": "B"}], "tolerance_s": 0.25}
    for sample, segs in arm_segments.items():
        d = root / sample
        d.mkdir(parents=True)
        with open(d / "labels.json", "w", encoding="utf-8") as fh:
            json.dump(labels, fh)
        with open(d / "segments_nemotron_dz.json", "w", encoding="utf-8") as fh:
            json.dump({"segments": [{"start": s, "end": e, "speaker": spk} for s, e, spk in segs]}, fh)
        (d / "clip.wav").write_bytes(b"")          # clip_duration falls back to the segment ends


PERFECT = [(0.0, 4.0, "0"), (5.0, 9.0, "1")]
SWAPPED_ONE_SIDE = [(0.0, 4.0, "0")]               # misses speaker B entirely: DER = 50 %


def test_identical_runs_pass(tmp_path):
    base, cand = tmp_path / "base", tmp_path / "cand"
    make_root(base, {"01_a": PERFECT, "01_b": PERFECT})
    make_root(cand, {"01_a": PERFECT, "01_b": PERFECT})
    rc = C.main([str(base), str(cand), "--arm", "nemotron_dz"])
    assert rc == 0


def test_worse_candidate_fails_and_reports_mean(tmp_path, capsys):
    base, cand = tmp_path / "base", tmp_path / "cand"
    make_root(base, {"01_a": PERFECT, "01_b": PERFECT})
    make_root(cand, {"01_a": PERFECT, "01_b": SWAPPED_ONE_SIDE})
    rc = C.main([str(base), str(cand), "--arm", "nemotron_dz", "--csv", str(tmp_path / "out.csv")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "KHÔNG ĐẠT" in out
    assert (tmp_path / "out.csv").read_text(encoding="utf-8").count("\n") == 3   # header + 2 samples


def test_samples_only_on_one_side_are_reported_not_compared(tmp_path, capsys):
    base, cand = tmp_path / "base", tmp_path / "cand"
    make_root(base, {"01_a": PERFECT, "01_only_base": PERFECT})
    make_root(cand, {"01_a": PERFECT})
    rc = C.main([str(base), str(cand), "--arm", "nemotron_dz"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "01_only_base" in out and "mẫu chung" in out


def test_no_common_sample_is_an_error(tmp_path):
    base, cand = tmp_path / "base", tmp_path / "cand"
    make_root(base, {"01_a": PERFECT})
    make_root(cand, {"01_b": PERFECT})
    assert C.main([str(base), str(cand), "--arm", "nemotron_dz"]) == 1


def test_compare_pure_function():
    res = C.compare({"a": 10.0, "b": 20.0}, {"a": 11.0, "b": 19.0})
    assert res["n"] == 2 and res["mean"] == pytest.approx(0.0)
    assert res["max_abs"] == pytest.approx(1.0)
    assert res["n_worse_over_1"] == 0 and res["n_better_over_1"] == 0
