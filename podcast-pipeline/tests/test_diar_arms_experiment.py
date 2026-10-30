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


def test_arm_f_is_listed_but_not_a_default_arm():
    assert X.ARMS["F"] == "nemotron_windowed" and "F" in X.ARM_ORDER
    assert "F" not in X.DEFAULT_ARMS.split(",")


class FakeWindowedVbx:
    """Stands in for the two arm G stages: stage 1 writes plan.json, stage 2 writes --out. fail_stage sets one to fail."""

    def __init__(self, fail_stage=None):
        self.fail_stage = fail_stage
        self.calls = []

    def __call__(self, cmd, env):
        self.calls.append(cmd)
        stage = 1 if X.WINDOWED_SCRIPT in cmd else 2
        if stage == self.fail_stage:
            return 1
        if stage == 1:
            probs = Path(cmd[cmd.index("--dump-probs") + 1])
            probs.mkdir(parents=True, exist_ok=True)
            (probs / "plan.json").write_text(json.dumps({"duration": 30.0, "frame_s": 0.08, "pieces": []}),
                                             encoding="utf-8")
        else:
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"segments": [
                {"start": 49.0, "end": 52.0, "speaker": "SPEAKER_00"},
                {"start": 60.0, "end": 66.0, "speaker": "SPEAKER_01"}]}), encoding="utf-8")
        return 0


class Dispatch:
    """One runner for a mixed run: main.py commands go to main, the windowed scripts to windowed."""

    def __init__(self, main, windowed):
        self.main, self.windowed = main, windowed

    def __call__(self, cmd, env):
        if X.WINDOWED_SCRIPT in cmd or X.VBX_STAGE_SCRIPT in cmd:
            return self.windowed(cmd, env)
        return self.main(cmd, env)


def test_arm_g_is_listed_but_not_a_default_arm():
    assert X.ARMS["G"] == "nemotron_windowed_vbx" and "G" in X.ARM_ORDER
    assert "G" not in X.DEFAULT_ARMS.split(",")


def test_arm_g_runs_dump_then_vbx_stage_in_order(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeWindowedVbx()
    rc = X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--only-arms", "G",
                 "--windowed-python", sys.executable, "--python", sys.executable,
                 "--windowed-piece-s", "200", "--windowed-hop-s", "5",
                 "--vbx-ahc-threshold", "0.5", "--vbx-fa", "0.1", "--vbx-fb", "0.9",
                 "--plda-dir", "/plda/here"], runner=fake)
    assert rc == 0
    assert len(fake.calls) == 4                                   # two stages per sample
    for pair in (fake.calls[0:2], fake.calls[2:4]):
        dump, vbx = pair
        assert dump[1] == X.WINDOWED_SCRIPT and "--dump-probs" in dump and "--probs-dir" not in dump
        assert dump[dump.index("--piece-s") + 1] == "200.0" and dump[dump.index("--hop-s") + 1] == "5.0"
        assert vbx[1] == X.VBX_STAGE_SCRIPT
        assert vbx[vbx.index("--probs-dir") + 1] == dump[dump.index("--dump-probs") + 1]
        assert vbx[vbx.index("--ahc-threshold") + 1] == "0.5"
        assert vbx[vbx.index("--fa") + 1] == "0.1" and vbx[vbx.index("--fb") + 1] == "0.9"
        assert vbx[vbx.index("--plda-dir") + 1] == "/plda/here"
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    seg = json.loads((sdir / "segments_nemotron_windowed_vbx.json").read_text())["segments"]
    assert [r["speaker"] for r in seg] == ["SPEAKER_00", "SPEAKER_01"]
    assert seg[0]["start"] == 0.0 and seg[0]["end"] == 2.0       # cropped to the clip, shifted to 0
    assert not (sdir / "segments_nemotron_windowed.json").exists()
    assert "nemotron_windowed_vbx" in (out / "index.html").read_text(encoding="utf-8")


def test_arm_g_stage_one_failure_skips_stage_two_and_others_still_run(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    main = FakeMain(_table)
    windowed = FakeWindowedVbx(fail_stage=1)
    rc = X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--only-arms", "A,G",
                 "--windowed-python", sys.executable, "--python", sys.executable],
                runner=Dispatch(main, windowed))
    assert rc == 0
    assert not any(X.VBX_STAGE_SCRIPT in c for c in windowed.calls)     # stage 2 never ran
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "stage 1 (probabilities) failed (exit 1)" in index
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    assert (sdir / "segments_nemotron.json").is_file()                 # arm A still ran


def test_arm_g_stage_two_failure_is_recorded(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    rc = X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--only-arms", "G",
                 "--windowed-python", sys.executable, "--python", sys.executable],
                runner=FakeWindowedVbx(fail_stage=2))
    assert rc == 0
    assert "stage 2 (VBx) failed (exit 1)" in (out / "index.html").read_text(encoding="utf-8")


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


def test_arm_f_runs_dump_then_the_ahc_stage_with_the_onnx_embedder(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeWindowedVbx()
    rc = X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--only-arms", "F",
                 "--windowed-python", sys.executable, "--python", sys.executable,
                 "--windowed-piece-s", "200", "--windowed-hop-s", "5"], runner=fake)
    assert rc == 0
    assert len(fake.calls) == 4                                   # two stages per sample
    dump, stage2 = fake.calls[0:2]
    assert dump[1] == X.WINDOWED_SCRIPT and "--dump-probs" in dump
    assert dump[dump.index("--piece-s") + 1] == "200.0" and dump[dump.index("--hop-s") + 1] == "5.0"
    assert stage2[1] == X.VBX_STAGE_SCRIPT
    assert stage2[stage2.index("--clustering") + 1] == "ahc"
    assert stage2[stage2.index("--embedder") + 1] == "wespeaker"
    assert "--fa" not in stage2 and "--plda-dir" not in stage2     # the VBx flags are arm G's
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    seg = json.loads((sdir / "segments_nemotron_windowed.json").read_text())["segments"]
    assert [r["speaker"] for r in seg] == ["SPEAKER_00", "SPEAKER_01"]
    assert seg[0]["start"] == 0.0 and seg[0]["end"] == 2.0       # cropped to the clip, shifted to 0
    assert "nemotron_windowed" in (out / "index.html").read_text(encoding="utf-8")


def test_failed_arm_f_is_recorded_and_the_run_goes_on(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    rc = X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--only-arms", "F",
                 "--windowed-python", sys.executable, "--python", sys.executable],
                runner=FakeWindowedVbx(fail_stage=2))
    assert rc == 0
    assert "stage 2 (AHC) failed (exit 1)" in (out / "index.html").read_text(encoding="utf-8")


def test_arm_h_runs_nemotron_in_one_pass_and_needs_no_k(tmp_path):
    assert X.ARMS["H"] == "nemotron_one_pass" and "H" not in X.DEFAULT_ARMS.split(",")
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeMain(_table)
    rc = X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--only-arms", "H"],
                runner=fake)
    assert rc == 0
    cmd = fake.calls[0]
    assert cmd[cmd.index("--diarization_backend") + 1] == "nemotron"
    assert "models.nemotron.legacy_one_pass=true" in cmd
    assert not any(c.startswith("models.diarizen") for c in cmd)
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    assert (sdir / "segments_nemotron_one_pass.json").is_file()
    assert not (sdir / "segments_nemotron.json").exists()            # arm A did not run
    assert "nemotron_one_pass" in (out / "index.html").read_text(encoding="utf-8")


def test_arm_i_is_arm_g_with_dense_windows(tmp_path):
    assert X.ARMS["I"] == "nemotron_dense_vbx" and "I" not in X.DEFAULT_ARMS.split(",")
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeWindowedVbx()
    rc = X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--only-arms", "I",
                 "--windowed-python", sys.executable, "--python", sys.executable,
                 "--windowed-hop-s", "10", "--dense-window-s", "12", "--dense-hop-s", "0.4",
                 "--vbx-fb", "0.3"], runner=fake)
    assert rc == 0
    dump, stage2 = fake.calls[0:2]
    assert dump[dump.index("--window-s") + 1] == "12.0" and dump[dump.index("--hop-s") + 1] == "0.4"
    assert stage2[1] == X.VBX_STAGE_SCRIPT and "--clustering" not in stage2      # VBx is the stage default
    assert stage2[stage2.index("--fb") + 1] == "0.3"
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    assert (sdir / "segments_nemotron_dense_vbx.json").is_file()
    assert "dense" in str(sdir / "runs") or (sdir / "runs" / "nemotron_dense_vbx" / "probs" / "plan.json").is_file()
    # arm G keeps its own sparse settings and has no --window-s
    fake_g = FakeWindowedVbx()
    X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(tmp_path / "out_g"), "--only-arms", "G",
            "--windowed-python", sys.executable, "--python", sys.executable, "--windowed-hop-s", "7"],
           runner=fake_g)
    assert "--window-s" not in fake_g.calls[0] and fake_g.calls[0][fake_g.calls[0].index("--hop-s") + 1] == "7.0"


def test_arm_j_runs_the_diarizen_steps_in_the_diarizen_env(tmp_path):
    assert X.ARMS["J"] == "nemotron_dz" and "J" not in X.DEFAULT_ARMS.split(",")
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeWindowedVbx()
    rc = X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(out), "--only-arms", "J",
                 "--windowed-python", sys.executable, "--python", "/som/python", "--vbx-python", "/dz/python",
                 "--dense-window-s", "16", "--dense-hop-s", "0.8", "--dz-thr", "0.4", "--dz-median-s", "0.3"],
                runner=fake)
    assert rc == 0
    dump, stage2 = fake.calls[0:2]
    assert dump[dump.index("--window-s") + 1] == "16.0" and dump[dump.index("--hop-s") + 1] == "0.8"
    assert stage2[0] == "/dz/python" and stage2[1] == X.VBX_STAGE_SCRIPT
    assert stage2[stage2.index("--clustering") + 1] == "dz"
    assert stage2[stage2.index("--thr") + 1] == "0.4" and stage2[stage2.index("--median-s") + 1] == "0.3"
    assert "--fa" in stage2 and "--fb" in stage2
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    assert (sdir / "segments_nemotron_dz.json").is_file()
    # F keeps the som python; G and I use --vbx-python too
    fake_f = FakeWindowedVbx()
    X.main(["--audio-dir", str(src), "--samples", str(sp), "--out", str(tmp_path / "out_f"), "--only-arms", "F",
            "--windowed-python", sys.executable, "--python", "/som/python", "--vbx-python", "/dz/python"],
           runner=fake_f)
    assert fake_f.calls[1][0] == "/som/python"


class FakeDemucs(FakeWindowedVbx):
    """The two windowed stages as FakeWindowedVbx; the demucs step writes its --out wav (fail_demucs makes it exit 1)."""

    def __init__(self, fail_demucs=False):
        super().__init__()
        self.fail_demucs = fail_demucs

    def __call__(self, cmd, env):
        if X.DEMUCS_SCRIPT in cmd:
            self.calls.append(cmd)
            if self.fail_demucs:
                return 1
            sf.write(cmd[cmd.index("--out") + 1], np.zeros(1600, dtype=np.float32), 16000)
            return 0
        return super().__call__(cmd, env)


def _windowed_args(samples_path, src, out, arms, *extra):
    return ["--audio-dir", str(src), "--samples", str(samples_path), "--out", str(out), "--only-arms", arms,
            "--windowed-python", sys.executable, "--python", sys.executable, *extra]


def _stage2_out(cmd):
    return Path(cmd[cmd.index("--out") + 1]).as_posix()


def test_arm_k_is_listed_and_not_a_default_arm():
    assert X.ARMS["K"] == "nemotron_dz_demucs" and "K" in X.ARM_ORDER
    assert "K" not in X.DEFAULT_ARMS.split(",")


def test_arm_k_runs_stage_one_then_demucs_then_stage_two_with_embed_audio(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeDemucs()
    rc = X.main(_windowed_args(sp, src, out, "K", "--demucs-python", "/demucs/python", "--vbx-python", "/dz/python",
                               "--dz-thr", "0.4", "--dz-median-s", "0.3"), runner=fake)
    assert rc == 0
    assert len(fake.calls) == 6                                    # per sample: stage 1, demucs, stage 2
    for dump, demucs, stage2 in (fake.calls[0:3], fake.calls[3:6]):
        assert dump[1] == X.WINDOWED_SCRIPT and dump[dump.index("--dump-probs") + 1].endswith("runs/nemotron_dz/probs")
        assert demucs[0] == "/demucs/python" and demucs[1] == X.DEMUCS_SCRIPT
        assert demucs[demucs.index("--in") + 1].endswith("chunk.wav")
        vocals = demucs[demucs.index("--out") + 1]
        assert vocals.endswith("chunk_vocals.wav")
        assert stage2[0] == "/dz/python" and stage2[1] == X.VBX_STAGE_SCRIPT
        assert stage2[stage2.index("--probs-dir") + 1] == dump[dump.index("--dump-probs") + 1]
        assert stage2[stage2.index("--embed-audio") + 1] == vocals
        assert stage2[stage2.index("--clustering") + 1] == "dz"
        assert stage2[stage2.index("--thr") + 1] == "0.4" and stage2[stage2.index("--median-s") + 1] == "0.3"
        assert stage2[stage2.index("--fb") + 1] == "0.8"                 # the base arm keeps --vbx-fb
        assert _stage2_out(stage2).endswith("runs/nemotron_dz_demucs/out/segments.json")
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    assert (sdir / "segments_nemotron_dz_demucs.json").is_file()
    assert not (sdir / "segments_nemotron_dz.json").exists()       # arm J did not run
    assert "nemotron_dz_demucs" in (out / "index.html").read_text(encoding="utf-8")


def test_arm_j_stage_two_has_no_embed_audio(tmp_path):
    src, sp = _make_samples(tmp_path)
    fake = FakeWindowedVbx()
    X.main(_windowed_args(sp, src, tmp_path / "out", "J"), runner=fake)
    stage2 = fake.calls[1]
    assert "--embed-audio" not in stage2 and stage2[stage2.index("--clustering") + 1] == "dz"


def test_arm_k_reuses_arm_j_probs_and_skips_stage_one_when_they_exist(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    for name in ("01_First_one", "02_Second_one"):                  # arm J's probs of each sample, from an earlier run
        probs = out / name / "runs" / "nemotron_dz" / "probs"
        probs.mkdir(parents=True)
        (probs / "plan.json").write_text(json.dumps({"duration": 30.0, "frame_s": 0.08, "pieces": []}), encoding="utf-8")
    fake = FakeDemucs()
    assert X.main(_windowed_args(sp, src, out, "K"), runner=fake) == 0
    assert not any(X.WINDOWED_SCRIPT in c for c in fake.calls)     # no stage 1
    assert [X.DEMUCS_SCRIPT in c for c in fake.calls] == [True, False, True, False]


def test_demucs_wav_is_reused_with_reuse_and_rebuilt_without(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    first = FakeDemucs()
    X.main(_windowed_args(sp, src, out, "K"), runner=first)
    assert sum(X.DEMUCS_SCRIPT in c for c in first.calls) == 2      # one per sample
    for seg in out.glob("*/runs/nemotron_dz_demucs/out/segments.json"):
        seg.unlink()                                               # K's result is gone, the vocals wav is not
    again = FakeDemucs()
    assert X.main(_windowed_args(sp, src, out, "K", "--reuse"), runner=again) == 0
    assert not any(X.DEMUCS_SCRIPT in c for c in again.calls)      # reuse: no demucs
    assert not any(X.WINDOWED_SCRIPT in c for c in again.calls)    # and no stage 1
    assert sum(X.VBX_STAGE_SCRIPT in c for c in again.calls) == 2
    plain = FakeDemucs()
    X.main(_windowed_args(sp, src, out, "K"), runner=plain)        # without reuse the step runs again
    assert sum(X.DEMUCS_SCRIPT in c for c in plain.calls) == 2


def test_demucs_failure_is_recorded_and_stage_two_does_not_run(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeDemucs(fail_demucs=True)
    assert X.main(_windowed_args(sp, src, out, "J,K"), runner=fake) == 0
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "arm nemotron_dz_demucs demucs step failed (exit 1)" in index
    assert not any(X.VBX_STAGE_SCRIPT in c and "runs/nemotron_dz_demucs" in " ".join(c) for c in fake.calls)
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    assert (sdir / "segments_nemotron_dz.json").is_file()          # arm J still ran


def test_demucs_python_defaults_to_python(tmp_path):
    src, sp = _make_samples(tmp_path)
    fake = FakeDemucs()
    X.main(_windowed_args(sp, src, tmp_path / "out", "K", "--python", "/som/python"), runner=fake)
    demucs = next(c for c in fake.calls if X.DEMUCS_SCRIPT in c)
    assert demucs[0] == "/som/python"
    assert next(c for c in fake.calls if X.VBX_STAGE_SCRIPT in c)[0] == "/som/python"


def test_fb_sweep_adds_arms_that_reuse_probs_and_vocals(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeDemucs()
    rc = X.main(_windowed_args(sp, src, out, "J,K", "--fb-sweep", "0.2,0.4"), runner=fake)
    assert rc == 0
    per_sample = len(fake.calls) // 2
    assert per_sample == 8                       # stage 1, J, J@0.2, J@0.4, demucs, K, K@0.2, K@0.4
    assert sum(X.WINDOWED_SCRIPT in c for c in fake.calls) == 2    # stage 1 once per sample, never for the sweep
    assert sum(X.DEMUCS_SCRIPT in c for c in fake.calls) == 2      # demucs once per sample, never for the sweep
    calls = fake.calls[0:per_sample]
    stage2 = [c for c in calls if X.VBX_STAGE_SCRIPT in c]
    expected = [("0.8", "runs/nemotron_dz/out/"), ("0.2", "runs/nemotron_dz_fb0.2/out/"),
                ("0.4", "runs/nemotron_dz_fb0.4/out/"), ("0.8", "runs/nemotron_dz_demucs/out/"),
                ("0.2", "runs/nemotron_dz_demucs_fb0.2/out/"), ("0.4", "runs/nemotron_dz_demucs_fb0.4/out/")]
    assert len(stage2) == len(expected)
    for cmd, (fb, where) in zip(stage2, expected):
        assert cmd[cmd.index("--fb") + 1] == fb
        assert where in _stage2_out(cmd)
        assert cmd[cmd.index("--clustering") + 1] == "dz"
        assert ("--embed-audio" in cmd) == ("demucs" in where)     # only the K arms embed the vocals
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    for name in ("nemotron_dz", "nemotron_dz_fb0.2", "nemotron_dz_fb0.4", "nemotron_dz_demucs",
                 "nemotron_dz_demucs_fb0.2", "nemotron_dz_demucs_fb0.4"):
        assert (sdir / f"segments_{name}.json").is_file(), name
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "J nemotron_dz_fb0.4" in index and "K nemotron_dz_demucs_fb0.2" in index


def test_fb_sweep_for_j_alone_needs_no_demucs(tmp_path):
    src, sp = _make_samples(tmp_path)
    fake = FakeDemucs()
    X.main(_windowed_args(sp, src, tmp_path / "out", "J", "--fb-sweep", "0.6"), runner=fake)
    assert not any(X.DEMUCS_SCRIPT in c for c in fake.calls)
    assert sum(X.VBX_STAGE_SCRIPT in c for c in fake.calls) == 4  # J and J@0.6, two samples


@pytest.mark.parametrize("bad", ["0", "-0.4", "0.2,0", "abc", "0.2,,0.4", "nan", "0.2,0.2", " "])
def test_fb_sweep_refuses_bad_values_before_anything_runs(tmp_path, bad):
    src, sp = _make_samples(tmp_path)
    fake = FakeDemucs()
    with pytest.raises(SystemExit):
        X.main(_windowed_args(sp, src, tmp_path / "out", "J,K", "--fb-sweep", bad), runner=fake)
    assert fake.calls == []


def test_fb_sweep_needs_arm_j_or_k(tmp_path):
    src, sp = _make_samples(tmp_path)
    fake = FakeDemucs()
    with pytest.raises(SystemExit):
        X.main(_windowed_args(sp, src, tmp_path / "out", "A", "--fb-sweep", "0.4"), runner=fake)
    assert fake.calls == []


def test_parse_fb_sweep_and_arm_names():
    assert X.parse_fb_sweep("0.2, 0.4") == ["0.2", "0.4"]
    assert X.parse_fb_sweep(None) == []
    assert X.arm_name("J") == "nemotron_dz" and X.arm_name("J@0.4") == "nemotron_dz_fb0.4"
    assert X.arm_name("K@0.2") == "nemotron_dz_demucs_fb0.2"
    assert X.split_key("K@0.2") == ("K", "0.2") and X.split_key("A") == ("A", None)


def _stage2(calls):
    return [c for c in calls if X.VBX_STAGE_SCRIPT in c]


def _without_out_thr_min(cmd):
    """The command as a list with --out, --thr and --min-on-s and their values taken out."""
    drop = {"--out", "--thr", "--min-on-s"}
    res, skip = [], False
    for tok in cmd:
        if skip:
            skip = False
            continue
        if tok in drop:
            skip = True
            continue
        res.append(tok)
    return res


def test_overlap_variants_add_one_stage_two_run_per_base_arm(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeDemucs()
    rc = X.main(_windowed_args(sp, src, out, "J,K", "--overlap-variants", "--demucs-python", "/demucs/python",
                               "--vbx-python", "/dz/python"), runner=fake)
    assert rc == 0
    per_sample = len(fake.calls) // 2
    assert per_sample == 6            # stage 1, J, J+ov, demucs, K, K+ov: no stage 1 or demucs for the variants
    assert sum(X.WINDOWED_SCRIPT in c for c in fake.calls) == 2 and sum(X.DEMUCS_SCRIPT in c for c in fake.calls) == 2
    base_j, ov_j, base_k, ov_k = _stage2(fake.calls[0:per_sample])
    assert _stage2_out(ov_j).endswith("runs/nemotron_dz_ov_thr0.35_on0.2/out/segments.json")
    assert _stage2_out(ov_k).endswith("runs/nemotron_dz_demucs_ov_thr0.35_on0.2/out/segments.json")
    for cmd in (ov_j, ov_k):
        assert cmd[0] == "/dz/python" and cmd[cmd.index("--clustering") + 1] == "dz"
        assert cmd[cmd.index("--thr") + 1] == "0.35" and cmd[cmd.index("--min-on-s") + 1] == "0.2"
        assert cmd[cmd.index("--fb") + 1] == "0.8" and cmd[cmd.index("--median-s") + 1] == "0.22"
        assert cmd[cmd.index("--probs-dir") + 1] == base_j[base_j.index("--probs-dir") + 1]
    assert ov_k[ov_k.index("--embed-audio") + 1] == base_k[base_k.index("--embed-audio") + 1]
    assert "--embed-audio" not in ov_j
    assert base_j[base_j.index("--thr") + 1] == "0.5" and "--min-on-s" not in base_j       # base is unchanged
    assert "--min-on-s" not in base_k and base_k[base_k.index("--thr") + 1] == "0.5"
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    for name in ("nemotron_dz", "nemotron_dz_ov_thr0.35_on0.2", "nemotron_dz_demucs", "nemotron_dz_demucs_ov_thr0.35_on0.2"):
        assert (sdir / f"segments_{name}.json").is_file(), name
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "J nemotron_dz_ov_thr0.35_on0.2" in index and "K nemotron_dz_demucs_ov_thr0.35_on0.2" in index


def test_overlap_variant_command_differs_from_its_base_only_in_thr_and_min_on_s(tmp_path):
    src, sp = _make_samples(tmp_path)
    plain = FakeDemucs()
    X.main(_windowed_args(sp, src, tmp_path / "plain", "J,K"), runner=plain)
    ov = FakeDemucs()
    X.main(_windowed_args(sp, src, tmp_path / "ov", "J,K", "--overlap-variants"), runner=ov)
    fix = lambda cmd: [t.replace(str(tmp_path / "plain"), "<OUT>").replace(str(tmp_path / "ov"), "<OUT>")
                       for t in cmd]
    base_plain = [fix(c) for c in _stage2(plain.calls) if "--min-on-s" not in c]
    base_ov = [fix(c) for c in _stage2(ov.calls) if "--min-on-s" not in c]
    assert base_plain == base_ov and len(base_plain) == 4                       # the base commands did not change
    ov_only = [fix(c) for c in _stage2(ov.calls) if "--min-on-s" in c]
    assert len(ov_only) == 4                                                    # two samples x (J+ov, K+ov)
    for cmd in ov_only:
        stripped = _without_out_thr_min(cmd)
        twins = [b for b in base_ov if _without_out_thr_min(b) == stripped]
        assert len(twins) == 1 and "0.35" in cmd and "0.2" in cmd


def test_overlap_variant_uses_the_ov_flags_and_fb_of_the_base(tmp_path):
    src, sp = _make_samples(tmp_path)
    fake = FakeDemucs()
    X.main(_windowed_args(sp, src, tmp_path / "out", "J", "--overlap-variants", "--ov-thr", "0.3",
                          "--ov-min-on-s", "0.25", "--vbx-fb", "0.6", "--dz-thr", "0.45"), runner=fake)
    base, ov = _stage2(fake.calls[0:3])
    assert ov[ov.index("--thr") + 1] == "0.3" and ov[ov.index("--min-on-s") + 1] == "0.25"
    assert ov[ov.index("--fb") + 1] == "0.6" and base[base.index("--thr") + 1] == "0.45"


def test_overlap_variant_reuses_existing_inputs_and_runs_only_the_variants(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    for name in ("01_First_one", "02_Second_one"):
        sdir = out / name
        probs = sdir / "runs" / "nemotron_dz" / "probs"
        probs.mkdir(parents=True)
        (probs / "plan.json").write_text(json.dumps({"duration": 30.0, "frame_s": 0.08, "pieces": []}), encoding="utf-8")
        for arm in ("nemotron_dz", "nemotron_dz_demucs"):
            (sdir / "runs" / arm / "out").mkdir(parents=True)
            (sdir / "runs" / arm / "out" / "segments.json").write_text('{"segments": []}', encoding="utf-8")
        sf.write(str(sdir / "chunk_vocals.wav"), np.zeros(1600, dtype=np.float32), 16000)
    fake = FakeDemucs()
    rc = X.main(_windowed_args(sp, src, out, "J,K", "--overlap-variants", "--reuse"), runner=fake)
    assert rc == 0
    assert len(fake.calls) == 4                                   # two variants per sample, nothing else
    assert all(X.VBX_STAGE_SCRIPT in c and "_ov_thr0.35_on0.2/" in _stage2_out(c) for c in fake.calls)
    assert all("--min-on-s" in c for c in fake.calls)


def test_overlap_variants_keep_the_fb_sweep_arms_at_the_base_threshold(tmp_path):
    src, sp = _make_samples(tmp_path)
    fake = FakeDemucs()
    rc = X.main(_windowed_args(sp, src, tmp_path / "out", "J,K", "--fb-sweep", "0.2,0.4", "--overlap-variants"),
                runner=fake)
    assert rc == 0
    per_sample = len(fake.calls) // 2
    assert per_sample == 10                 # stage 1, J, J@0.2, J@0.4, J+ov, demucs, K, K@0.2, K@0.4, K+ov
    stage2 = _stage2(fake.calls[0:per_sample])
    assert len(stage2) == 8
    names = [_stage2_out(c).split("runs/")[1].split("/out/")[0] for c in stage2]
    assert names == ["nemotron_dz", "nemotron_dz_fb0.2", "nemotron_dz_fb0.4", "nemotron_dz_ov_thr0.35_on0.2",
                     "nemotron_dz_demucs", "nemotron_dz_demucs_fb0.2", "nemotron_dz_demucs_fb0.4",
                     "nemotron_dz_demucs_ov_thr0.35_on0.2"]
    for cmd, name in zip(stage2, names):
        if "_ov_thr" in name:
            assert cmd[cmd.index("--thr") + 1] == "0.35" and cmd[cmd.index("--min-on-s") + 1] == "0.2"
        else:
            assert cmd[cmd.index("--thr") + 1] == "0.5" and "--min-on-s" not in cmd
    fb_of = {n: c[c.index("--fb") + 1] for c, n in zip(stage2, names)}
    assert fb_of["nemotron_dz_fb0.2"] == "0.2" and fb_of["nemotron_dz_ov_thr0.35_on0.2"] == "0.8"


@pytest.mark.parametrize("flag, bad", [("--ov-thr", "0"), ("--ov-thr", "1"), ("--ov-thr", "1.5"),
                                       ("--ov-thr", "-0.1"), ("--ov-thr", "nan"),
                                       ("--ov-min-on-s", "-0.1"), ("--ov-min-on-s", "nan")])
def test_overlap_variant_values_are_refused_before_anything_runs(tmp_path, flag, bad):
    src, sp = _make_samples(tmp_path)
    fake = FakeDemucs()
    with pytest.raises(SystemExit):
        X.main(_windowed_args(sp, src, tmp_path / "out", "J,K", "--overlap-variants", flag, bad), runner=fake)
    assert fake.calls == []


def test_overlap_variants_need_arm_j_or_k(tmp_path):
    src, sp = _make_samples(tmp_path)
    fake = FakeDemucs()
    with pytest.raises(SystemExit):
        X.main(_windowed_args(sp, src, tmp_path / "out", "A", "--overlap-variants"), runner=fake)
    assert fake.calls == []


def test_overlap_key_names_and_split():
    assert X.arm_name(X.ov_key("J", 0.35, 0.2)) == "nemotron_dz_ov_thr0.35_on0.2"
    assert X.arm_name(X.ov_key("K", 0.3, 0)) == "nemotron_dz_demucs_ov_thr0.3_on0"
    assert X.ov_key("J", 0.35, 0.2) == "J+ov|0.35|0.2" and X.ov_params("J+ov|0.35|0.2") == (0.35, 0.2)
    assert X.split_key("J+ov|0.35|0.2") == ("J", None) and X.is_ov("J+ov|0.35|0.2") and not X.is_ov("J@0.2")
    assert X.arm_name("J") == "nemotron_dz"


def test_parse_grid_takes_comma_lists_and_refuses_bad_ones():
    assert X.parse_grid("0.3, 0.35", "--ov-thr", lambda v: 0 < v < 1, "r") == [0.3, 0.35]
    assert X.parse_grid("0.2", "--ov-min-on-s", lambda v: v >= 0, "r") == [0.2]
    assert X.parse_grid("0", "--ov-min-on-s", lambda v: v >= 0, "r") == [0.0]
    for bad in ("", " ", "0.3,", "0.3,0.3", "0.3,0.30", "x", "1.0,0.5,0.5"):
        with pytest.raises(ValueError):
            X.parse_grid(bad, "--ov-thr", lambda v: 0 < v < 1, "must be in (0, 1)")


def test_ov_grid_caps_the_combinations_per_arm_at_24():
    combos = X.ov_grid([0.3, 0.35, 0.4, 0.45], [0, 0.1, 0.2, 0.3, 0.4, 0.5])     # 4 x 6 = 24
    assert len(combos) == 24 and combos[0] == (0.3, 0) and combos[1] == (0.3, 0.1)   # thr outer
    with pytest.raises(ValueError, match="at most 24"):
        X.ov_grid([0.3, 0.35, 0.4, 0.45, 0.5], [0, 0.1, 0.2, 0.3, 0.4, 0.5])          # 5 x 6 = 30


def test_overlap_grid_runs_one_arm_per_combination_for_j_and_k(tmp_path):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FakeDemucs()
    rc = X.main(_windowed_args(sp, src, out, "J,K", "--overlap-variants", "--ov-thr", "0.3,0.35",
                               "--ov-min-on-s", "0,0.2"), runner=fake)
    assert rc == 0
    per_sample = len(fake.calls) // 2
    assert per_sample == 12                 # stage 1, J, 4 J grid arms, demucs, K, 4 K grid arms
    names = [_stage2_out(c).split("runs/")[1].split("/out/")[0] for c in _stage2(fake.calls[0:per_sample])]
    assert names == ["nemotron_dz", "nemotron_dz_ov_thr0.3_on0", "nemotron_dz_ov_thr0.3_on0.2",
                     "nemotron_dz_ov_thr0.35_on0", "nemotron_dz_ov_thr0.35_on0.2", "nemotron_dz_demucs",
                     "nemotron_dz_demucs_ov_thr0.3_on0", "nemotron_dz_demucs_ov_thr0.3_on0.2",
                     "nemotron_dz_demucs_ov_thr0.35_on0", "nemotron_dz_demucs_ov_thr0.35_on0.2"]
    stage2 = _stage2(fake.calls[0:per_sample])
    grid = [c for c in stage2 if "--min-on-s" in c]
    assert [(c[c.index("--thr") + 1], c[c.index("--min-on-s") + 1]) for c in grid[:4]] == \
        [("0.3", "0"), ("0.3", "0.2"), ("0.35", "0"), ("0.35", "0.2")]
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    assert (sdir / "segments_nemotron_dz_ov_thr0.3_on0.json").is_file()
    assert (sdir / "segments_nemotron_dz_demucs_ov_thr0.35_on0.2.json").is_file()


@pytest.mark.parametrize("args", [["--ov-thr", "0.3,0.3"], ["--ov-thr", "0.3,0.30"],
                                  ["--ov-min-on-s", "0,0"], ["--ov-min-on-s", "0.2,0.2000"],
                                  ["--ov-thr", "0.3,1"], ["--ov-min-on-s", "0,-1"],
                                  ["--ov-thr", "0.2,0.3,0.4,0.45,0.5", "--ov-min-on-s", "0,0.1,0.2,0.3,0.4,0.5"]])
def test_overlap_grid_refusals_run_nothing(tmp_path, args):
    src, sp = _make_samples(tmp_path)
    fake = FakeDemucs()
    with pytest.raises(SystemExit):
        X.main(_windowed_args(sp, src, tmp_path / "out", "J,K", "--overlap-variants", *args), runner=fake)
    assert fake.calls == []


def test_embedder_trial_names_keys_and_parsing():
    assert X.arm_name(X.trial_key("r34lm", 0.5)) == "nemotron_dz_emb_r34lm_c0.5"
    assert X.arm_name(X.trial_key("espnet_spk", 0.4, "K")) == "nemotron_dz_demucs_emb_espnet_spk_c0.4"
    assert X.trial_params("J+emb|espnet_spk|0.6") == ("espnet_spk", 0.6) and X.is_trial("J+emb|r34lm|0.5")
    assert X.split_key("J+emb|r34lm|0.5") == ("J", None)
    assert X.parse_kinds("") == [] and X.parse_kinds("r34lm, espnet_spk") == ["r34lm", "espnet_spk"]
    for bad in ("nope", "r34lm,r34lm"):
        with pytest.raises(ValueError):
            X.parse_kinds(bad)
    assert X.parse_embedder_python(["r34lm=/a/python"], ["r34lm"]) == {"r34lm": "/a/python"}
    for bad in (["espnet_spk=/x"], ["r34lm"], ["r34lm=/a", "r34lm=/b"]):
        with pytest.raises(ValueError):
            X.parse_embedder_python(bad, ["r34lm"])


class FailingKind(FakeDemucs):
    """Stage 2 of the trials of one embedder kind exits 1, like a missing package would."""

    def __init__(self, bad_kind):
        super().__init__()
        self.bad_kind = bad_kind

    def __call__(self, cmd, env):
        if X.VBX_STAGE_SCRIPT in cmd and "--embedder-kind" in cmd:
            self.calls.append(cmd)
            if cmd[cmd.index("--embedder-kind") + 1] == self.bad_kind:
                return 1
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"segments": [{"start": 50.0, "end": 60.0, "speaker": "SPEAKER_00"}]}),
                           encoding="utf-8")
            return 0
        return super().__call__(cmd, env)


def test_embedder_trials_run_stage_two_per_kind_and_cos_and_continue_past_a_failure(tmp_path, capsys):
    src, sp = _make_samples(tmp_path)
    out = tmp_path / "out"
    fake = FailingKind("wespeaker_r293")
    rc = X.main(_windowed_args(sp, src, out, "J", "--embedder-trials", "r34lm,wespeaker_r293",
                               "--cos-thr", "0.4,0.5,0.6", "--embedder-python", "wespeaker_r293=/r293/python",
                               "--vbx-python", "/dz/python", "--simam-dir", "/unused"), runner=fake)
    assert rc == 0
    trials = [c for c in fake.calls if "--embedder-kind" in c]
    assert len(trials) == 2 * 2 * 3                                 # 2 samples x 2 embedders x 3 cos cuts
    for cmd in trials:
        assert cmd[cmd.index("--clustering") + 1] == "dz-ahc"
        assert cmd[cmd.index("--cos-threshold") + 1] in ("0.4", "0.5", "0.6")
        assert cmd[0] == ("/r293/python" if cmd[cmd.index("--embedder-kind") + 1] == "wespeaker_r293" else "/dz/python")
        assert cmd[cmd.index("--embcache-dir") + 1].endswith("embcache")
        assert "--embed-audio" not in cmd and cmd[cmd.index("--audio") + 1].endswith("chunk.wav")
    sdir = sorted(p for p in out.iterdir() if p.is_dir())[0]
    for cos in ("0.4", "0.5", "0.6"):
        assert (sdir / f"segments_nemotron_dz_emb_r34lm_c{cos}.json").is_file()        # the good embedder ran
        assert not (sdir / f"segments_nemotron_dz_emb_wespeaker_r293_c{cos}.json").exists()
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "J nemotron_dz_emb_r34lm_c0.5" in index and "J nemotron_dz_emb_wespeaker_r293_c0.5" in index
    printed = capsys.readouterr().out
    assert "embedder trials failed for 2 (sample, embedder) pairs" in printed
    assert "First one / wespeaker_r293 (cos 0.4, 0.5, 0.6)" in printed
    assert "Second one / wespeaker_r293 (cos 0.4, 0.5, 0.6)" in printed


def test_embedder_trial_refusals_run_nothing(tmp_path):
    src, sp = _make_samples(tmp_path)
    for extra in (["--embedder-trials", "nope"], ["--embedder-trials", "r34lm", "--cos-thr", "0"],
                  ["--embedder-trials", "r34lm", "--cos-thr", "2"], ["--embedder-python", "r34lm=/p"],
                  ["--embedder-trials", "wespeaker_simam100"],                 # needs --simam-onnx or --simam-dir
                  ["--embedder-trials", "wespeaker_simam100", "--simam-onnx", "a", "--simam-dir", "b"]):
        fake = FakeDemucs()
        with pytest.raises(SystemExit):
            X.main(_windowed_args(sp, src, tmp_path / "out", "J", *extra), runner=fake)
        assert fake.calls == [], extra


def test_embedder_trials_default_python_is_the_vbx_python(tmp_path):
    src, sp = _make_samples(tmp_path)
    fake = FakeDemucs()
    X.main(_windowed_args(sp, src, tmp_path / "out", "J", "--embedder-trials", "r34lm", "--cos-thr", "0.5",
                          "--vbx-python", "/dz/python"), runner=fake)
    trials = [c for c in fake.calls if "--embedder-kind" in c]
    assert len(trials) == 2 and all(c[0] == "/dz/python" for c in trials)
    assert all(c[c.index("--cos-threshold") + 1] == "0.5" for c in trials)
