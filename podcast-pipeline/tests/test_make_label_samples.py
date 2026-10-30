import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import diar_arms_experiment as X  # noqa: E402
from tools import make_label_samples as M  # noqa: E402

DUR = 3600.0


def write_export(out_dir: Path, cat: str, folder: str, stem: str, lo: float, hi: float) -> dict:
    """One export excerpt: its folder with conversation.json, and the manifest row that points at it."""
    d = out_dir / cat / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "conversation.json").write_text(
        json.dumps({"source_start": lo, "source_end": hi, "orig_spans": [{"start": lo, "end": hi}]}),
        encoding="utf-8")
    return {"path": f"{cat}/{folder}/stereo_2ch.wav", "category": cat, "source_stem": stem,
            "duration": round(hi - lo, 3), "sample_rate": 16000}


def write_manifest(out_dir: Path, rows, name="manifest.jsonl") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def make_audio(root: Path, stems) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for s in stems:
        (root / f"{s}.mp3").write_bytes(b"")
    return root


def fake_probe(_path):
    return DUR


def fake_find_for(stems):
    known = set(stems)

    def find(stem):
        if stem in known:
            return f"/src/{stem}.mp3"
        raise M.SourceNotFound(stem)
    return find


def tone(n=16000 * 50):
    t = np.arange(n) / 16000.0
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def run(argv_extra, tmp_path, *, probe=fake_probe, decode=None, find=None):
    args = M._parser().parse_args(argv_extra)
    if args.exclude_samples is None:
        args.exclude_samples = [str(tmp_path / "none.json")]
    if not os.path.exists(args.exclude_samples[0]):
        (tmp_path / "none.json").write_text("[]", encoding="utf-8")
    return M.build(args, probe=probe, decode=decode or (lambda p, s, d: tone(int(d * 16000))),
                   find=find or fake_find_for([]))


def base_args(tmp_path, manifest, audio_dirs, extra=()):
    none = tmp_path / "none.json"
    if not none.exists():
        none.write_text("[]", encoding="utf-8")
    argv = ["--manifest", str(manifest), "--out", str(tmp_path / "out.json"),
            "--exclude-samples", str(none)]
    for a in audio_dirs:
        argv += ["--audio-dir", str(a)]
    return argv + list(extra)


@pytest.fixture
def corpus(tmp_path):
    """Three categories, one duplicate recording across two categories, and a random audio folder."""
    out = tmp_path / "data"
    rows = []
    rows.append(write_export(out, "interrupt", "rec_i1_100s", "rec_i1", 100.0, 400.0))
    rows.append(write_export(out, "interrupt", "rec_i2_200s", "rec_i2", 200.0, 400.0))
    rows.append(write_export(out, "interrupt", "rec_i3_300s", "rec_i3", 300.0, 500.0))
    rows.append(write_export(out, "backchannel", "rec_b1_50s", "rec_b1", 50.0, 300.0))
    rows.append(write_export(out, "backchannel", "rec_i1_900s", "rec_i1", 900.0, 1100.0))  # same recording again
    rows.append(write_export(out, "turn_by_turn", "rec_t1_10s", "rec_t1", 10.0, 200.0))
    manifest = write_manifest(out, rows)
    audio = make_audio(tmp_path / "audio", ["rec_i1", "rec_i2", "rec_i3", "rec_b1", "rec_t1",
                                            "rec_r0", "rec_r1", "rec_r2", "rec_r3"])
    stems = ["rec_i1", "rec_i2", "rec_i3", "rec_b1", "rec_t1"]
    return {"manifest": manifest, "audio": audio, "stems": stems, "tmp": tmp_path}


def test_round_robin_one_clip_per_recording(corpus):
    argv = base_args(corpus["tmp"], corpus["manifest"], [corpus["audio"]],
                     ["--n-exports", "6", "--n-random", "0"])
    res = run(argv, corpus["tmp"], find=fake_find_for(corpus["stems"]))
    # Five unique recordings exist (rec_i1 also sits in backchannel and is taken once). Round 1 takes one
    # from each category in order; the rest depends on the seeded shuffle, so only that is asserted.
    kinds = [e["kind"] for e in res["entries"]]
    assert kinds[:3] == ["Interrupt", "Backchannel", "Turn_by_turn"]
    assert len(kinds) == 5
    stems = [e["stem"] for e in res["entries"]]
    assert len(stems) == len(set(stems))               # one clip per recording
    assert sorted(stems) == sorted(corpus["stems"])
    assert res["skips"]["recording-used"] >= 1          # the second rec_i1 row never gets a second clip


def test_same_seed_same_output_other_seed_differs(corpus):
    argv = base_args(corpus["tmp"], corpus["manifest"], [corpus["audio"]],
                     ["--n-exports", "5", "--n-random", "4", "--seed", "3"])
    a = run(argv, corpus["tmp"], find=fake_find_for(corpus["stems"]))["entries"]
    b = run(argv, corpus["tmp"], find=fake_find_for(corpus["stems"]))["entries"]
    assert a == b
    other = list(argv)
    other[other.index("3")] = "4"
    c = run(other, corpus["tmp"], find=fake_find_for(corpus["stems"]))["entries"]
    assert [e["start"] for e in a] != [e["start"] for e in c]


def test_existing_samples_are_excluded(corpus):
    samples = corpus["tmp"] / "samples.json"
    samples.write_text(json.dumps([
        {"name": "x1", "stem": "rec_i1", "start": None},
        {"name": "x2", "stem": "rec_b1_50s", "start": None},
        {"name": "x3", "audio": "/elsewhere/rec_r1.mp3", "stem": None},
    ]), encoding="utf-8")
    argv = base_args(corpus["tmp"], corpus["manifest"], [corpus["audio"]],
                     ["--n-exports", "10", "--n-random", "10"])
    argv[argv.index("--exclude-samples") + 1] = str(samples)
    res = run(argv, corpus["tmp"], find=fake_find_for(corpus["stems"]))
    stems = {e["stem"] for e in res["entries"]}
    assert "rec_i1" not in stems and "rec_b1" not in stems and "rec_r1" not in stems
    assert res["skips"]["excluded"] >= 1
    assert "rec_r0" in stems or "rec_r2" in stems or "rec_r3" in stems


def test_windows_inside_excerpt_and_recording(tmp_path):
    out = tmp_path / "data"
    rows = [write_export(out, "interrupt", "rec_long_100s", "rec_long", 100.0, 400.0),
            write_export(out, "interrupt", "rec_mid_500s", "rec_mid", 500.0, 535.0),
            write_export(out, "backchannel", "rec_short_600s", "rec_short", 600.0, 620.0),
            write_export(out, "turn_by_turn", "rec_edge_350s", "rec_edge", 350.0, 420.0)]
    manifest = write_manifest(out, rows)
    audio = make_audio(tmp_path / "audio", [])
    argv = base_args(tmp_path, manifest, [audio], ["--n-exports", "10", "--n-random", "0", "--seed", "7"])
    stems = ["rec_long", "rec_mid", "rec_short", "rec_edge"]
    res = run(argv, tmp_path, probe=lambda p: 380.0 if "rec_edge" in p else DUR,
              find=fake_find_for(stems))
    by = {e["stem"]: e for e in res["entries"]}
    assert res["skips"]["short-excerpt"] == 1
    assert "rec_short" not in by
    long = by["rec_long"]
    assert 100.0 <= long["start"] <= 350.0 + 1e-6
    assert long["end"] - long["start"] == pytest.approx(50.0, abs=0.01)
    assert long["end"] <= 400.0 + 1e-6
    mid = by["rec_mid"]                                  # 35 s excerpt: whole excerpt, no random window
    assert (mid["start"], mid["end"]) == (500.0, 535.0)
    if "rec_edge" in by:
        assert by["rec_edge"]["end"] <= 380.0
    for e in res["entries"]:
        assert e["start"] >= 0 and e["end"] <= (380.0 if e["stem"] == "rec_edge" else DUR)


def test_speech_gate_rejects_silence_accepts_noise(tmp_path):
    sil = np.zeros(16000 * 50, dtype=np.float32)
    noise = (np.random.default_rng(0).standard_normal(16000 * 50) * 0.05).astype(np.float32)
    assert M.speech_ratio(sil) == 0.0
    assert M.speech_ratio(noise) == 1.0           # about -26 dBFS RMS, above the -45 dBFS floor
    assert M.speech_ratio(tone()) == 1.0
    audio = make_audio(tmp_path / "audio", ["rec_r0", "rec_r1"])
    out = tmp_path / "data"
    manifest = write_manifest(out, [])
    argv = base_args(tmp_path, manifest, [audio], ["--n-random", "2", "--n-exports", "0"])
    silent = run(argv, tmp_path, decode=lambda p, s, d: sil)
    assert silent["entries"] == []
    assert silent["skips"]["random-no-speech"] == 2
    loud = run(argv, tmp_path, decode=lambda p, s, d: noise)
    assert len(loud["entries"]) == 2
    assert all(e["kind"] == "Random" and e["source"] == "random" for e in loud["entries"])
    assert all("speech=" in e["note"] for e in loud["entries"])


def test_missing_source_is_reported_not_fatal(corpus):
    argv = base_args(corpus["tmp"], corpus["manifest"], [corpus["audio"]],
                     ["--n-exports", "6", "--n-random", "0"])
    res = run(argv, corpus["tmp"], find=fake_find_for(["rec_i2", "rec_b1", "rec_t1"]))
    assert "rec_i1" in res["missing"] and "rec_i3" in res["missing"]
    assert res["skips"]["source-not-found"] >= 2
    assert {e["stem"] for e in res["entries"]} == {"rec_i2", "rec_b1", "rec_t1"}


def test_entries_pass_resolve_span_unchanged(corpus):
    argv = base_args(corpus["tmp"], corpus["manifest"], [corpus["audio"]],
                     ["--n-exports", "5", "--n-random", "4", "--clip-s", "55"])
    res = run(argv, corpus["tmp"], find=fake_find_for(corpus["stems"]))
    assert len(res["entries"]) >= 5
    for e in res["entries"]:
        assert X.resolve_span(e, None) == (e["start"], e["end"], "sample")
        assert os.path.isabs(e["audio"])
        assert e["start"] >= 0
        assert e["duration"] == M.fmt_mmss(e["end"] - e["start"])
        assert e["name"] and e["kind"] in {"Interrupt", "Backchannel", "Turn_by_turn", "Random"}


def test_dry_run_writes_nothing(corpus):
    argv = base_args(corpus["tmp"], corpus["manifest"], [corpus["audio"]], ["--dry-run"])
    rc = M.main(argv, probe=fake_probe, decode=lambda p, s, d: tone(int(d * 16000)),
                find=fake_find_for(corpus["stems"]))
    assert rc == 0
    assert not (corpus["tmp"] / "out.json").exists()


def test_no_overwrite_without_force(corpus):
    out = corpus["tmp"] / "out.json"
    out.write_text("keep me", encoding="utf-8")
    argv = base_args(corpus["tmp"], corpus["manifest"], [corpus["audio"]], ["--n-random", "0"])
    rc = M.main(argv, probe=fake_probe, decode=lambda p, s, d: tone(int(d * 16000)),
                find=fake_find_for(corpus["stems"]))
    assert rc == 2
    assert out.read_text(encoding="utf-8") == "keep me"
    rc = M.main(argv + ["--force"], probe=fake_probe, decode=lambda p, s, d: tone(int(d * 16000)),
                find=fake_find_for(corpus["stems"]))
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list) and data


def test_clip_s_out_of_range_is_an_error(corpus):
    argv = base_args(corpus["tmp"], corpus["manifest"], [corpus["audio"]], ["--clip-s", "20"])
    with pytest.raises(SystemExit):
        M.main(argv, probe=fake_probe, decode=lambda p, s, d: tone(), find=fake_find_for([]))


def test_speech_ratio_frame_math():
    x = np.zeros(16000, dtype=np.float32)
    x[:8000] = 0.3                                # first half loud, second half silent
    assert M.speech_ratio(x) == pytest.approx(0.5, abs=0.01)
