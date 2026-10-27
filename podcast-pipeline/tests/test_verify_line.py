import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("verify_line", Path(__file__).resolve().parents[1] / "tools" / "verify_line.py")
vl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vl)

VARIANTS = {"variants": [
    {"chain": "none", "id": "none"},
    {"chain": "none", "id": "raw", "set": {"steps.music_analysis": False, "steps.music_removal": False}},
    {"chain": "identity", "id": "p1"},
    {"chain": "none", "chain_pre_asr": "identity", "id": "p3"},
]}


def build(tmp_path, broken=None):
    audio = tmp_path / "audio"; audio.mkdir()
    (audio / "a.wav").write_bytes(b"x"); (audio / "b.wav").write_bytes(b"x")
    var = tmp_path / "variants.json"; var.write_text(json.dumps(VARIANTS))
    out = tmp_path / "out"
    manifest = []
    for vid, chains in (("none", ()), ("p1", ("01b_denoise",)), ("p3", ("04b_denoise_pre_asr",))):
        manifest.append({"id": vid, "status": "ok"})
        for stem in ("a", "b"):
            d = out / vid / stem
            for rel in ("02_diarization/segments.json", "03_separation/segments.json", "05_asr/transcripts.json"):
                (d / rel).parent.mkdir(parents=True, exist_ok=True)
                (d / rel).write_text(json.dumps([{"start": 0, "end": 1, "speaker": "S0", "text": "x"}]))
            for sub in chains:
                (d / sub).mkdir(parents=True, exist_ok=True)
                (d / sub / "denoise.json").write_text(json.dumps({"errors": [], "corr_with_input": 0.5}))
    (out / "variants_manifest.json").write_text(json.dumps({"variants": manifest}))
    return audio, var, out


def run(audio, var, out, *extra):
    return vl.main(["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(out), *extra])


def test_everything_present_is_complete_and_raw_is_skipped(tmp_path, capsys):
    audio, var, out = build(tmp_path)
    assert run(audio, var, out) == 0
    text = capsys.readouterr().out
    assert "3/3 variants complete for all 2 file(s)" in text and "skipped on purpose: raw" in text


def test_missing_asr_filter_error_and_bad_status_are_reported(tmp_path, capsys):
    audio, var, out = build(tmp_path)
    (out / "none" / "b" / "05_asr" / "transcripts.json").unlink()
    (out / "p1" / "a" / "01b_denoise" / "denoise.json").write_text(json.dumps({"errors": ["boom: out of memory"]}))
    (out / "p3" / "b" / "04b_denoise_pre_asr" / "denoise.json").unlink()
    m = json.loads((out / "variants_manifest.json").read_text())
    m["variants"][2]["status"] = "degraded"
    (out / "variants_manifest.json").write_text(json.dumps(m))
    assert run(audio, var, out) == 1
    text = capsys.readouterr().out
    assert "0/3 variants complete" in text
    assert "b: asr: 05_asr/transcripts.json missing" in text
    assert "a: pre_diarization: filter fell back to the input: boom" in text
    assert "b: pre_asr: 04b_denoise_pre_asr/denoise.json missing" in text
    assert "manifest status is 'degraded'" in text


def test_a_filter_that_changed_nothing_is_a_warning_not_a_failure(tmp_path, capsys):
    audio, var, out = build(tmp_path)
    (out / "p1" / "a" / "01b_denoise" / "denoise.json").write_text(json.dumps({"errors": [], "corr_with_input": 1.0}))
    assert run(audio, var, out) == 0
    assert "warning  p1" in capsys.readouterr().out


def test_music_profile_keeps_raw_in_the_expected_set(tmp_path, capsys):
    audio, var, out = build(tmp_path)
    assert run(audio, var, out, "--music", "profile") == 1
    assert "NOT COMPLETE  raw" in capsys.readouterr().out


def line_run(out, ids=("none", "p1", "p3"), stems=("a", "b")):
    """Markers and .done of a line run that reached the final process, for every variant and file."""
    for vid in ids:
        vd = out / vid
        vd.mkdir(parents=True, exist_ok=True)
        (vd / ".done").write_text("{}")
        for st in stems:
            for stage in ("separation_plan", "conversation_selection"):
                (vd / f".ready_{stage}.{st}").write_text("{}")
            sel = vd / st / "07d_conversation_selection"
            sel.mkdir(parents=True, exist_ok=True)
            (sel / "selection.json").write_text(json.dumps([{"start": 0, "end": 1}]))


def test_line_run_with_new_stages_is_complete(tmp_path, capsys):
    audio, var, out = build(tmp_path)
    line_run(out)
    assert run(audio, var, out) == 0
    assert "3/3 variants complete" in capsys.readouterr().out


def test_missing_new_stage_handover_or_final_done_is_reported(tmp_path, capsys):
    audio, var, out = build(tmp_path)
    line_run(out)
    (out / "p1" / ".ready_separation_plan.b").unlink()
    (out / "p3" / ".done").unlink()
    (out / "none" / ".failed_conversation_selection.a").write_text(json.dumps({"reason": "llm oom"}))
    assert run(audio, var, out) == 1
    text = capsys.readouterr().out
    assert "b: separation_plan hand-over missing" in text
    assert "final process did not finish" in text
    assert "a: conversation_selection failed" in text


def test_without_variants_it_checks_the_production_variants(tmp_path, monkeypatch, capsys):
    """The default was variants_placement.json, which is not the set the production line runs.
    The default is a path from the repo root, not from the cwd, so it works from anywhere."""
    expected = Path(__file__).resolve().parents[1] / "tools" / "variants_production.json"
    assert vl.DEFAULT_VARIANTS == expected and expected.is_file()
    assert [v["id"] for v in load_variants_of(expected)] == ["prod"]
    audio, var, out = build(tmp_path)
    monkeypatch.setattr(vl, "DEFAULT_VARIANTS", var)     # the stand-in set, read through the default
    assert vl.main(["--audio-dir", str(audio), "--out-root", str(out)]) == 0
    assert "3/3 variants complete" in capsys.readouterr().out


def load_variants_of(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["variants"]
