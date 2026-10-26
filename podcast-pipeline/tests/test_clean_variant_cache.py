import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("clean", Path(__file__).resolve().parents[1] / "tools" / "clean_variant_cache.py")
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


def make(tmp_path):
    audio = tmp_path / "audio"; audio.mkdir()
    for n in ("a.wav", "b.wav"):
        (audio / n).write_bytes(b"x")
    var = tmp_path / "variants.json"
    var.write_text(json.dumps({"variants": [{"chain": "none", "id": "v1"}, {"chain": "identity", "id": "v2"}]}))
    cache = tmp_path / "cache"
    for d in ("exp-none_a", "exp-v1_a", "exp-v1_b", "exp-v2_a",          # ours
              "exp-v1_data_a", "exp-other_a", "exp-v1_c", "exp-resident"):  # not ours
        (cache / d).mkdir(parents=True)
        (cache / d / "f").write_text("data")
    out = tmp_path / "out"
    for vid in ("none", "v1"):
        (out / vid / "a").mkdir(parents=True)
        for m in (".done", ".done.separation", ".ready_denoise", ".failed_asr", "keep.txt"):
            (out / vid / m).write_text("{}")
    (out / "variants_manifest.json").write_text("{}")
    (out / "other_variant").mkdir()
    return audio, var, cache, out


def test_dry_run_lists_and_touches_nothing(tmp_path, capsys):
    audio, var, cache, out = make(tmp_path)
    assert clean.main(["--audio-dir", str(audio), "--variants", str(var), "--cache-dir", str(cache)]) == 0
    text = capsys.readouterr().out
    assert "dry run" in text and "exp-v1_a" in text and "exp-v1_data_a" not in text
    assert sorted(p.name for p in cache.iterdir()) == sorted(
        ["exp-none_a", "exp-v1_a", "exp-v1_b", "exp-v2_a", "exp-v1_data_a", "exp-other_a", "exp-v1_c", "exp-resident"])


def test_delete_removes_only_these_files_for_these_variants(tmp_path):
    audio, var, cache, out = make(tmp_path)
    clean.main(["--audio-dir", str(audio), "--variants", str(var), "--cache-dir", str(cache), "--delete"])
    assert sorted(p.name for p in cache.iterdir()) == ["exp-other_a", "exp-resident", "exp-v1_c", "exp-v1_data_a"]


def test_out_root_clears_markers_and_stem_outputs_but_not_other_files(tmp_path):
    audio, var, cache, out = make(tmp_path)
    clean.main(["--audio-dir", str(audio), "--variants", str(var), "--cache-dir", str(cache),
                "--out-root", str(out), "--delete"])
    left = sorted(p.name for p in (out / "v1").iterdir())
    assert left == ["keep.txt"]                              # markers and the <stem> folder went, the rest stayed
    assert not (out / "variants_manifest.json").exists() and (out / "other_variant").is_dir()


def test_per_file_markers_are_cleared_too(tmp_path):
    audio, var, cache, out = make(tmp_path)
    for m in (".ready_separation.a", ".ready_denoise_pre_asr.ep.final", ".failed_diarization.b"):
        (out / "v1" / m).write_text("{}")
    clean.main(["--audio-dir", str(audio), "--variants", str(var), "--cache-dir", str(cache),
                "--out-root", str(out), "--delete"])
    assert sorted(p.name for p in (out / "v1").iterdir()) == ["keep.txt"]
