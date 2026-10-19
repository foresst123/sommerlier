import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import collect_overlap_samples as cos  # noqa: E402

sf = pytest.importorskip("soundfile")


def test_context_window_slides_at_edges():
    assert cos.context_window(10, 12, 60, 300) == (0.0, 60.0)
    assert cos.context_window(100, 102, 60, 300) == (71.0, 131.0)
    assert cos.context_window(290, 295, 60, 300) == (240.0, 300.0)
    assert cos.context_window(5, 100, 60, 300) == (5, 100)
    assert cos.context_window(1, 2, 60, 30) == (0.0, 30.0)


def test_cross_speaker_overlaps_ignores_same_speaker():
    segs = [{"start": 0, "end": 3, "speaker": "A"}, {"start": 2, "end": 5, "speaker": "B"},
            {"start": 4, "end": 6, "speaker": "B"}, {"start": 8, "end": 9, "speaker": "A"},
            {"start": 8.5, "end": 12, "speaker": "A"}]
    assert cos.cross_speaker_overlaps(segs) == [(2, 3)]


def test_clean_clips_exclude_margin_and_music():
    segs = [{"start": 0, "end": 10, "speaker": "A"}, {"start": 4, "end": 5, "speaker": "B"},
            {"start": 20, "end": 25, "speaker": "A", "has_music": True}]
    got = cos.clean_clips(segs, "A", 0.5, 1.0)
    assert got == [(0, 3.5), (5.5, 10)]


def _recording(root, name, n_overlaps):
    rec = root / name
    (rec / "01_music").mkdir(parents=True)
    (rec / "03_separation").mkdir()
    sr, total = 8000, 400
    sf.write(str(rec / "01_music" / "after_music.wav"), np.random.randn(sr * total).astype("float32") * 0.1, sr)
    segs = [{"index": "00000", "start": 0.0, "end": 30.0, "speaker": "A"},
            {"index": "00001", "start": 31.0, "end": 60.0, "speaker": "B"}]
    for i in range(n_overlaps):
        t = 70 + i * 70
        d = [0.2, 1.0, 3.0, 7.0][i % 4]
        segs += [{"index": f"a{i}", "start": t, "end": t + 10, "speaker": "A"},
                 {"index": f"b{i}", "start": t + 10 - d, "end": t + 20, "speaker": "B"}]
    (rec / "03_separation" / "segments.json").write_text(json.dumps(segs))
    return rec


def test_build_covers_bins_and_writes_files(tmp_path):
    _recording(tmp_path / "run" / "_final" / "p", "recA", 4)
    _recording(tmp_path / "run" / "_final" / "p", "recB", 4)
    out = tmp_path / "out"
    idx = cos.build([str(tmp_path / "run")], str(out), n=8, context=60.0)
    assert len(idx) == 8
    assert {s["bin"] for s in idx} == {"short", "medium", "long", "xlong"}
    s = next(s for s in idx if s["bin"] == "medium")
    d = out / s["folder"]
    meta = json.loads((d / "meta.json").read_text())
    assert sf.info(str(d / "context.wav")).duration == pytest.approx(60.0, abs=0.01)
    assert sf.info(str(d / "overlap_original.wav")).duration == pytest.approx(meta["overlap"]["duration"], abs=0.01)
    assert set(meta["references"]) == {"A", "B"}
    assert (d / "speakers" / "A" / "reference.wav").exists()
    assert meta["context"]["start"] <= meta["overlap"]["start"] and meta["overlap"]["end"] <= meta["context"]["end"]
