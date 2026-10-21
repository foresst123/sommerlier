"""Kiểm thử tools/run_variants.py với main giả (không GPU, không pipeline thật).
Chạy trong podcast-pipeline: python -m pytest tests/test_run_variants.py -q"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_variants", ROOT / "tools" / "run_variants.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

STUB = '''
import argparse, json, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
for f in ("--audio_dir", "--save_path", "--output_dir", "--job_id", "--env", "--stop_after",
          "--denoise", "--cache_dir"):
    p.add_argument(f)
a = p.parse_args()
out = a.save_path or a.output_dir
Path(out, "calls.txt").parent.mkdir(parents=True, exist_ok=True)
with open(Path(out, "calls.txt"), "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")
if a.denoise == "bad":
    sys.exit(3)
for audio in sorted(Path(a.audio_dir).iterdir()):
    d = Path(out) / audio.stem
    for sub in ["01_music", "02_diarization", "03_separation", "05_asr"] + (["01b_denoise"] if a.denoise else []):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "01_music" / "after_music.wav").write_bytes(b"x")
    (d / "02_diarization" / "segments.json").write_text("[]")
    (d / "03_separation" / "stats.json").write_text("{}")
    (d / "05_asr" / "asr.json").write_text("[]")
'''


@pytest.fixture
def env(tmp_path):
    main = tmp_path / "stub_main.py"
    main.write_text(STUB)
    audio = tmp_path / "audio"
    audio.mkdir()
    for n in ("a.wav", "b.mp3"):
        (audio / n).write_bytes(b"audio-" + n.encode())
    var = tmp_path / "variants.json"
    var.write_text(json.dumps({"variants": [{"chain": "dfn3"},
                                            {"chain": "dfn3>mf2se48", "label": "L"}]}))
    return tmp_path, main, audio, var


def argv(env, *extra):
    tmp, main, audio, var = env
    return ["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(tmp / "out"),
            "--main", str(main), "--env", "a100", "--stop-after", "asr", *extra]


def calls(tmp, vid):
    p = tmp / "out" / vid / "calls.txt"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def test_ids_and_baseline_always_present(env):
    tmp, _, _, var = env
    vs = rv.load_variants(var)
    assert [v["id"] for v in vs] == ["none", "dfn3", "dfn3+mf2se48"]
    assert vs[2]["label"] == "L" and vs[2]["chain"] == ["dfn3", "mf2se48"]
    assert rv.load_variants(None)[0]["id"] == "none"
    with pytest.raises(ValueError):
        rv._fallback_parse_chain("a>>b")


def test_dry_run_prints_commands_writes_nothing(env, capsys):
    tmp = env[0]
    assert rv.main(argv(env, "--dry-run", "--gpus", "0,1")) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 3
    assert "--denoise" not in out[0] and "--job_id exp-none" in out[0]
    assert "--denoise 'dfn3>mf2se48'" in out[2] and "--job_id exp-dfn3+mf2se48" in out[2]
    assert not (tmp / "out").exists()


def test_run_dirs_inputs_manifest(env):
    tmp, _, audio, _ = env
    assert rv.main(argv(env, "--gpus", "4,5", "--reference", "a=" + str(tmp / "ref.json"))) == 0
    base = calls(tmp, "none")[0]
    assert "--denoise" not in base["argv"] and base["gpu"] == "4"
    assert "--denoise" in calls(tmp, "dfn3")[0]["argv"] and calls(tmp, "dfn3")[0]["gpu"] == "5"
    inp = tmp / "out" / "dfn3" / "input"
    assert (inp / "a.wav").is_symlink() and os.path.realpath(inp / "a.wav") == str((audio / "a.wav").resolve())
    assert not list(audio.glob("_sommelier*"))
    assert (tmp / "out" / "logs" / "dfn3.log").is_file()
    m = json.loads((tmp / "out" / "variants_manifest.json").read_text())
    assert set(m) == {"version", "baseline", "created", "variants", "reference"}
    assert m["version"] == 1 and m["baseline"] == "none"
    assert m["reference"] == {"a": str((tmp / "ref.json").resolve()), "b": None}
    v = {x["id"]: x for x in m["variants"]}
    assert set(v) == {"none", "dfn3", "dfn3+mf2se48"}
    for x in v.values():
        assert {"id", "chain", "label", "out_dir", "job_id", "files", "status"} <= set(x)
        assert x["status"] == "ok" and set(x["files"]) == {"a", "b"}
    assert v["dfn3"]["job_id"] == "exp-dfn3" and v["dfn3+mf2se48"]["chain"] == ["dfn3", "mf2se48"]
    assert not (Path(v["none"]["files"]["a"]) / "01b_denoise").exists()
    assert (Path(v["dfn3"]["files"]["a"]) / "01b_denoise").is_dir()


def test_resume_skips_and_reruns_after_change(env):
    tmp, _, audio, _ = env
    rv.main(argv(env))
    rv.main(argv(env, "--resume"))
    assert len(calls(tmp, "dfn3")) == 1
    (audio / "a.wav").write_bytes(b"changed content!")
    rv.main(argv(env, "--resume"))
    assert len(calls(tmp, "dfn3")) == 2
    rv.main(argv(env))  # without --resume always reruns
    assert len(calls(tmp, "dfn3")) == 3


def test_failed_variant_isolated(env):
    tmp, _, _, var = env
    var.write_text(json.dumps({"variants": [{"chain": "bad"}, {"chain": "dfn3"}]}))
    assert rv.main(argv(env, "--parallel", "2")) == 1
    assert not (tmp / "out" / "bad" / ".done").exists()
    assert (tmp / "out" / "dfn3" / ".done").exists()
    m = json.loads((tmp / "out" / "variants_manifest.json").read_text())
    st = {x["id"]: x["status"] for x in m["variants"]}
    assert st == {"none": "ok", "bad": "failed", "dfn3": "ok"}
    assert "exit status: 3" in (tmp / "out" / "logs" / "bad.log").read_text()


def test_stem_clash_rejected(env, tmp_path):
    other = tmp_path / "x"
    other.mkdir()
    (other / "a.wav").write_bytes(b"zz")
    tmp, main, audio, var = env
    with pytest.raises(SystemExit):
        rv.collect_files(str(audio), [str(other / "a.wav")])
