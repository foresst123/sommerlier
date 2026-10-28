import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import diar_arms_experiment as X  # noqa: E402


def _nemotron_rows():
    # chunk coordinates; the clip is 50-80 s of a chunk that starts at 0
    return [{"start": 10.0, "end": 20.0, "speaker": "C"},      # before the clip: cropped away
            {"start": 50.0, "end": 70.0, "speaker": "A"},      # main talker
            {"start": 72.0, "end": 73.2, "speaker": "B"}]      # backchannel listener, 1.2 s clean


class FakeMain:
    """Stands in for main.py: writes 02_diarization/segments.json where build_cmd says it goes."""

    def __init__(self, table, fail=()):
        self.table = table
        self.fail = set(fail)
        self.calls = []

    def __call__(self, cmd, env):
        self.calls.append(cmd)
        backend = cmd[cmd.index("--diarization_backend") + 1]
        if backend in self.fail:
            return 1
        sets = {}
        for i, tok in enumerate(cmd):
            if tok == "--set":
                k, v = cmd[i + 1].split("=", 1)
                sets[k] = v
        save = Path(cmd[cmd.index("--save_path") + 1])
        stem = Path(cmd[cmd.index("--audio") + 1]).stem
        path = save / stem / "02_diarization" / "segments.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.table(backend, sets)), encoding="utf-8")
        return 0


def _table(backend, sets):
    if backend == "nemotron":
        return _nemotron_rows()
    n = int(sets.get("models.diarizen.max_speakers", 3))
    return [{"start": 50.0 + 5 * i, "end": 54.0 + 5 * i, "speaker": f"S{i % n}"} for i in range(6)]


def _source(path, seconds=100):
    sf.write(str(path), np.zeros(int(seconds * 16000), dtype=np.float32), 16000, subtype="PCM_16")
    return str(path)


def _make_samples(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _source(src / "AAAA1111_talk.wav")
    _source(src / "BBBB2222.wav")
    samples = [
        {"name": "First one", "kind": "Backchannel", "turns": 7, "duration": "0:30", "stem": "AAAA1111",
         "start": 50, "end": 80},
        {"name": "Second one", "kind": "Turns", "turns": 10, "duration": "0:30", "stem": "BBBB2222",
         "start": 50, "end": None},
    ]
    sp = tmp_path / "samples.json"
    sp.write_text(json.dumps(samples), encoding="utf-8")
    return src, sp


def test_k_counts_backchannel_listener_at_one_second(tmp_path):
    segs = X.crop(_nemotron_rows(), 50, 80)
    assert [r["speaker"] for r in segs] == ["A", "B"]
    assert X.estimate_k(segs, 1.0) == 2
    # the listener has 1.2 s of clean speech: under 1.5 s it does not count
    assert X.estimate_k(segs, 1.5) == 1
    assert X.estimate_k([], 1.0) == 1


def test_k_ignores_overlapped_speech(tmp_path):
    # B's 3 s are all inside A's turn, so they are not clean speech for B
    segs = [{"start": 0.0, "end": 10.0, "speaker": "A"}, {"start": 2.0, "end": 5.0, "speaker": "B"}]
    assert X.estimate_k(segs, 1.0) == 1


def test_chunk_window_centres_and_clamps():
    assert X.chunk_window(50, 80) == (0.0, 300.0, 50.0, 80.0)            # no total: lead clamped at 0
    assert X.chunk_window(1000, 1040, 5000) == (880.0, 1180.0, 120.0, 160.0)
    assert X.chunk_window(50, 80, 100) == (0.0, 100.0, 50.0, 80.0)      # short file
    # near the end of the file the chunk moves back so it still lasts CHUNK_S
    assert X.chunk_window(1050, 1090, 1100) == (800.0, 1100.0, 250.0, 290.0)
    with pytest.raises(ValueError):
        X.chunk_window(1050, 1120, 1100)                                 # clip ends after the file
    with pytest.raises(ValueError):
        X.chunk_window(80, 50)


def test_crop_clips_and_shifts():
    segs = [{"start": 0, "end": 5, "speaker": "A"}, {"start": 8, "end": 12, "speaker": "B"},
            {"start": 14, "end": 20, "speaker": "A"}]
    assert X.crop(segs, 10, 16) == [{"start": 0.0, "end": 2.0, "speaker": "B"},
                                    {"start": 4.0, "end": 6.0, "speaker": "A"}]


def test_fuse_renames_boundaries_one_to_one():
    nem = [{"start": 0, "end": 6, "speaker": "A"}, {"start": 6, "end": 10, "speaker": "B"},
           {"start": 10, "end": 20, "speaker": "B"}]
    dz = [{"start": 0, "end": 10, "speaker": "X"}, {"start": 10, "end": 20, "speaker": "Y"},
          {"start": 0, "end": 3, "speaker": "W"}, {"start": 30, "end": 35, "speaker": "Z"}]
    got = [r["speaker"] for r in X.fuse(dz, nem)]
    # Y-B (10 s) first, then X-A (6 s); W's only partner A is taken, Z has none
    assert got == ["A", "B", "x_W", "x_Z"]


def test_stats_counts_short_speech_and_overlap():
    segs = [{"start": 0.0, "end": 0.5, "speaker": "A"}, {"start": 0.0, "end": 4.0, "speaker": "B"},
            {"start": 3.0, "end": 6.0, "speaker": "A"}]
    st = X.stats(segs)
    assert st["speakers"] == 2 and st["short"] == 1
    assert st["speech_s"] == 6.0
    # A 0-0.5 against B 0-4 gives 0.5 s, B 0-4 against A 3-6 gives 1 s; their union is 1.5 s
    assert st["overlap_s"] == 1.5


def test_span_from_sample_then_final_then_suffix(tmp_path):
    base = {"name": "x", "stem": "70MHLeTDFXA", "folder_suffix": 576, "duration": "0:45"}
    assert X.resolve_span(base, None) == (576.0, 621.0, "suffix")
    assert X.resolve_span({**base, "start": 10, "end": None}, None) == (10.0, 55.0, "sample")

    final = tmp_path / "final" / "batch1" / "70MHLeTDFXA_talk_576s"
    final.mkdir(parents=True)
    (final / "conversation.json").write_text(json.dumps(
        {"source_stem": "70MHLeTDFXA_talk", "source_start": 575.2, "source_end": 620.8}), encoding="utf-8")
    assert X.resolve_span(base, str(tmp_path / "final")) == (575.2, 620.8, "final")
    with pytest.raises(ValueError):
        X.resolve_span({"name": "y", "stem": "ZZZ"}, None)


def test_find_source_by_prefix(tmp_path):
    (tmp_path / "AAAA1111_talk.wav").write_bytes(b"")
    (tmp_path / "other.wav").write_bytes(b"")
    assert X.find_source(str(tmp_path), "AAAA1111") == str(tmp_path / "AAAA1111_talk.wav")
    with pytest.raises(FileNotFoundError):
        X.find_source(str(tmp_path), "NOPE")
    (tmp_path / "AAAA1111_part2.wav").write_bytes(b"")
    (tmp_path / "AAAA1111_part3.wav").write_bytes(b"")
    with pytest.raises(ValueError):
        X.find_source(str(tmp_path), "AAAA1111")


def test_run_writes_all_arms_index_and_fusion(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeMain(_table)
    rc = X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--fusion"], runner=fake)
    assert rc == 0

    index = (out / "index.html").read_text(encoding="utf-8")
    assert "First one" in index and "Second one" in index
    assert index.count("lanes.html") >= 2
    lanes = list(out.glob("*/lanes.html"))
    assert len(lanes) == 2 and all("clip.wav" in p.read_text(encoding="utf-8") for p in lanes)

    slug = sorted(p.name for p in out.iterdir() if p.is_dir())[0]
    sdir = out / slug
    clip = sdir / "clip.wav"
    assert clip.is_file()
    assert sf.info(str(clip)).duration == pytest.approx(30.0, abs=0.05)
    for name in ("nemotron", "diarizen_auto", "diarizen_k", "diarizen_1_k", "fusion"):
        assert (sdir / f"segments_{name}.json").is_file(), name

    # K = 2 from the backchannel listener: arm C is pinned to two speakers, auto arm B has three
    seg_c = json.loads((sdir / "segments_diarizen_k.json").read_text())["segments"]
    assert {r["speaker"] for r in seg_c} == {"S0", "S1"}
    seg_b = json.loads((sdir / "segments_diarizen_auto.json").read_text())["segments"]
    assert len({r["speaker"] for r in seg_b}) == 3
    seg_d = [c for c in fake.calls if "models.diarizen.min_speakers=1" in c]
    assert seg_d and "models.diarizen.max_speakers=2" in seg_d[0]

    # crop: the speaker 10-20 s of the chunk lies before the clip and must not show
    nem = json.loads((sdir / "segments_nemotron.json").read_text())["segments"]
    assert [r["speaker"] for r in nem] == ["A", "B"]
    assert nem[0]["start"] == 0.0 and nem[0]["end"] == 20.0

    fus = json.loads((sdir / "segments_fusion.json").read_text())["segments"]
    assert all(r["speaker"] in {"A", "B"} or r["speaker"].startswith("x_") for r in fus)


def test_reuse_skips_finished_arms(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    first = FakeMain(_table)
    X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out)], runner=first)
    assert len(first.calls) == 8   # 4 arms x 2 samples
    again = FakeMain(_table)
    X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--reuse"], runner=again)
    assert again.calls == []
    assert (out / "index.html").is_file()


def test_failed_arm_does_not_stop_others(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeMain(_table, fail={"nemotron"})
    X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out)], runner=fake)
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    assert (sdir / "segments_diarizen_auto.json").is_file()
    assert not (sdir / "segments_diarizen_k.json").exists()     # no K without arm A
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "no K" in index


class FakeWindowed:
    """Stands in for tools/windowed_nemotron_run.py: writes its --out file, or fails."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def __call__(self, cmd, env):
        self.calls.append(cmd)
        if self.fail:
            return 1
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"segments": [
            {"start": 49.0, "end": 52.0, "speaker": "SPEAKER_00"},
            {"start": 60.0, "end": 66.0, "speaker": "SPEAKER_01"}]}), encoding="utf-8")
        return 0


def test_arm_f_is_listed_but_not_a_default_arm():
    assert X.ARMS["F"] == "nemotron_windowed" and "F" in X.ARM_ORDER
    assert "F" not in X.DEFAULT_ARMS.split(",")


def test_arm_f_runs_the_windowed_script_and_needs_no_k(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeWindowed()
    rc = X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--only-arms", "F",
                 "--windowed-python", sys.executable, "--windowed-piece-s", "200", "--windowed-hop-s", "5"],
                runner=fake)
    assert rc == 0
    assert len(fake.calls) == 2                                   # one run per sample
    cmd = fake.calls[0]
    assert cmd[1] == X.WINDOWED_SCRIPT
    assert cmd[cmd.index("--piece-s") + 1] == "200.0" and cmd[cmd.index("--hop-s") + 1] == "5.0"
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    seg = json.loads((sdir / "segments_nemotron_windowed.json").read_text())["segments"]
    assert [r["speaker"] for r in seg] == ["SPEAKER_00", "SPEAKER_01"]
    assert seg[0]["start"] == 0.0 and seg[0]["end"] == 2.0       # cropped to the clip, shifted to 0
    assert not (sdir / "segments_nemotron.json").exists()
    assert "nemotron_windowed" in (out / "index.html").read_text(encoding="utf-8")


def test_failed_arm_f_is_recorded_and_the_run_goes_on(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    rc = X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--only-arms", "F",
                 "--windowed-python", sys.executable], runner=FakeWindowed(fail=True))
    assert rc == 0
    assert "failed (exit 1)" in (out / "index.html").read_text(encoding="utf-8")


def test_main_py_runs_without_a_terminal_on_stdin(monkeypatch):
    # main.py asks for folders at a prompt when stdin is a terminal; the arms must not wait on it.
    import subprocess
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(X.subprocess, "run", fake_run)
    assert X.run_main(["x"], {}) == 0
    assert seen["stdin"] is subprocess.DEVNULL
