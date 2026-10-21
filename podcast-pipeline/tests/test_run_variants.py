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
          "--denoise", "--cache_dir", "--music_separator"):
    p.add_argument(f)
p.add_argument("--set", action="append")
p.add_argument("--bss", action="store_true")
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


def _fake_availability(monkeypatch, unavailable):
    import models.denoise as md
    real = md.available_backends
    monkeypatch.setattr(md, "available_backends", lambda cfg=None: {
        **{k: (True, "") for k in ("dfn3", "mf2se48")},
        **{k: (False, why) for k, why in unavailable.items()}})
    return real


def test_groups_and_notes_flow_to_manifest_and_filter(env, capsys):
    tmp, _, _, var = env
    var.write_text(json.dumps({"variants": [
        {"chain": "dfn3", "group": "masking", "note": "n1"},
        {"chain": "dfn3>mf2se48", "group": "chains"}]}))
    assert rv.main(argv(env, "--groups", "chains")) == 0
    m = json.loads((tmp / "out" / "variants_manifest.json").read_text())
    assert [x["id"] for x in m["variants"]] == ["none", "dfn3+mf2se48"]
    assert m["variants"][1]["group"] == "chains" and m["variants"][1]["note"] is None
    assert "group=chains" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        rv.main(argv(env, "--groups", "nope"))
    assert rv.main(argv(env)) == 0
    m = json.loads((tmp / "out" / "variants_manifest.json").read_text())
    assert {x["id"]: x["note"] for x in m["variants"]}["dfn3"] == "n1"


def test_only_available_skips_with_reason_never_silently(env, monkeypatch, capsys):
    tmp, _, _, var = env
    var.write_text(json.dumps({"variants": [
        {"chain": "dfn3"}, {"chain": "dbs_tiger", "group": "analysis_only"},
        {"chain": "dbs_tiger>dfn3"}, {"chain": "ghost"}]}))
    _fake_availability(monkeypatch, {"dbs_tiger": "DBS_TIGER_DIR is not set"})
    assert rv.main(argv(env, "--only-available")) == 0  # skipped is not a failure
    out = capsys.readouterr().out
    assert "[dbs_tiger] SKIPPED" in out and "DBS_TIGER_DIR is not set" in out
    assert "[ghost] SKIPPED" in out and "unknown backend" in out
    assert "skipped 3 variant(s)" in out
    assert calls(tmp, "dfn3") and not calls(tmp, "dbs_tiger") and not calls(tmp, "ghost")
    m = json.loads((tmp / "out" / "variants_manifest.json").read_text())
    v = {x["id"]: x for x in m["variants"]}
    assert v["dfn3"]["status"] == "ok" and v["none"]["status"] == "ok"
    assert v["dbs_tiger"]["status"] == "skipped" and "DBS_TIGER_DIR" in v["dbs_tiger"]["reason"]
    assert v["dbs_tiger+dfn3"]["status"] == "skipped" and v["dfn3"]["reason"] is None
    assert not (tmp / "out" / "dbs_tiger").exists() or not (tmp / "out" / "dbs_tiger" / "input").exists()


def test_without_only_available_nothing_is_skipped(env):
    tmp = env[0]
    assert rv.main(argv(env)) == 0
    m = json.loads((tmp / "out" / "variants_manifest.json").read_text())
    assert all(x["status"] == "ok" for x in m["variants"])


# ------------------------------------------------------- overrides and planning
def write_var(var, variants):
    var.write_text(json.dumps({"variants": variants}))


def test_override_variant_id_tag_and_baseline_rules(env):
    _, _, _, var = env
    write_var(var, [{"chain": "none"},  # exact baseline duplicate merges silently
                    {"chain": "none", "tag": "music stage OFF",
                     "set": {"steps.music_removal": False}},
                    {"chain": "dfn3", "set": {"pipeline.music_scope": "full"}, "id": "dfn3-full"}])
    vs = rv.load_variants(var)
    assert [v["id"] for v in vs] == ["none", "none.music-removal-off", "dfn3-full"]
    assert vs[0]["set"] == {} and vs[1]["tag"] == "music stage OFF" and vs[1]["chain"] == []
    assert vs[1]["id"] == rv.load_variants(var)[1]["id"]  # deterministic
    long = rv.override_tag({"models.bs_roformer.model": "a" * 60})
    assert len(long) == 8 and rv.ID_RE.match(long)


def test_duplicate_and_unsafe_ids_fail_early(env):
    _, _, _, var = env
    write_var(var, [{"chain": "dfn3", "id": "x"}, {"chain": "mf2se48", "id": "x"},
                    {"chain": "none", "id": "none", "set": {"steps.a": False}}])
    with pytest.raises(SystemExit) as e:
        rv.load_variants(var)
    assert "'none'" in str(e.value) and "'x'" in str(e.value)
    write_var(var, [{"chain": "dfn3", "id": "a/b"}])
    with pytest.raises(SystemExit, match="filename-safe"):
        rv.load_variants(var)


def test_command_has_set_music_separator_extra_args(env):
    _, _, _, var = env
    write_var(var, [{"chain": "dfn3", "music_separator": "m.ckpt", "extra_args": "--bss",
                     "set": {"steps.music_removal": False, "models.bs_roformer.model": "x y.ckpt"}}])
    v = rv.load_variants(var)[1]
    args = rv.parse_args(argv(env))
    cmd = rv.build_command(v, [], args)
    assert cmd[cmd.index("--music_separator") + 1] == "m.ckpt"
    sets = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--set"]
    assert sets == ['steps.music_removal=false', 'models.bs_roformer.model="x y.ckpt"']
    assert cmd[-1] == "--bss"
    base = rv.build_command(rv.load_variants(var)[0], [], args)
    assert "--set" not in base and "--music_separator" not in base


def test_dry_run_quotes_set_and_done_hash_depends_on_overrides(env, capsys, tmp_path):
    _, _, _, var = env
    write_var(var, [{"chain": "none", "set": {"models.bs_roformer.model": "a b.ckpt"}}])
    assert rv.main(argv(env, "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "--set 'models.bs_roformer.model=\"a b.ckpt\"'" in out
    f = tmp_path / "f.wav"
    f.write_bytes(b"x")
    a = rv.build_command(rv.load_variants(var)[1], [f], rv.parse_args(argv(env)))
    write_var(var, [{"chain": "none", "id": "none.music-removal-off", "set": {"models.bs_roformer.model": "b.ckpt"}}])
    b = rv.build_command(rv.load_variants(var)[1], [f], rv.parse_args(argv(env)))
    assert rv.run_hash(a, [f]) != rv.run_hash(b, [f])


def test_run_manifest_and_compare_data_passthrough(env):
    tmp, _, _, var = env
    write_var(var, [{"chain": "none", "tag": "music stage OFF", "music_separator": "m.ckpt",
                     "set": {"steps.music_removal": False}}])
    assert rv.main(argv(env)) == 0
    m = json.loads((tmp / "out" / "variants_manifest.json").read_text())
    x = {v["id"]: v for v in m["variants"]}["none.music-removal-off_sep-m"]
    assert x["set"] == {"steps.music_removal": False} and x["tag"] == "music stage OFF"
    assert x["music_separator"] == "m.ckpt"
    argv_ = calls(tmp, "none.music-removal-off_sep-m")[0]["argv"]
    assert "steps.music_removal=false" in argv_ and "m.ckpt" in argv_
    from tools.compare_lib.build import build_compare_data
    data = build_compare_data(str(tmp / "out" / "variants_manifest.json"), embed=False)
    row = {v["id"]: v for v in data["variants"]}["none.music-removal-off_sep-m"]
    assert row["tag"] == "music stage OFF" and row["set"] == {"steps.music_removal": False}
    assert "tag" not in data["variants"][0] and "set" not in data["variants"][0]
    import tools.make_compare_page as mp
    chips = mp._chips(data["variants"], "none")
    assert "music stage OFF" in chips and "steps.music_removal" in chips


def test_list_table_and_arithmetic(env, capsys):
    tmp, _, _, var = env
    write_var(var, [{"chain": "dfn3", "group": "g"},
                    {"chain": "none", "tag": "music OFF", "set": {"steps.music_removal": False}}])
    assert rv.main(argv(env, "--list", "--minutes-per-file-run", "10")) == 0
    out = capsys.readouterr().out
    assert "none.music-removal-off" in out and "steps.music_removal=false" in out and "music OFF" in out
    assert "variants: 3" in out and "files: 2" in out and "variant-file runs: 6" in out
    assert "60 min" in out and "ASSUMING 10 min" in out
    assert not (tmp / "out").exists()  # nothing ran
    assert rv.main(argv(env, "--list")) == 0
    assert "120 min" in capsys.readouterr().out  # default 20 min


def many(var, n=8):
    write_var(var, [{"chain": "none", "id": f"v{i}", "set": {"steps.x": i}} for i in range(n)])


def test_max_variants_shuffle_keeps_baseline_and_is_deterministic(env):
    _, _, _, var = env
    many(var)
    vs = rv.load_variants(var)
    first = rv.pick_variants(vs, max_variants=3)
    assert [v["id"] for v in first] == ["none", "v0", "v1"]
    a = rv.pick_variants(vs, max_variants=4, shuffle_seed=7)
    b = rv.pick_variants(vs, max_variants=4, shuffle_seed=7)
    assert [v["id"] for v in a] == [v["id"] for v in b] and len(a) == 4 and a[0]["id"] == "none"
    ids = [v["id"] for v in a]
    assert ids == sorted(ids, key=[v["id"] for v in vs].index)  # file order kept
    assert any([v["id"] for v in rv.pick_variants(vs, max_variants=4, shuffle_seed=s)] != ids
               for s in range(1, 20))
    assert [v["id"] for v in rv.pick_variants(vs, max_variants=1, shuffle_seed=1)] == ["none"]


def test_only_ids(env, capsys):
    _, _, _, var = env
    many(var)
    got = rv.pick_variants(rv.load_variants(var), only_ids=["v3", "v5"])
    assert [v["id"] for v in got] == ["none", "v3", "v5"]
    with pytest.raises(SystemExit, match="unknown id"):
        rv.pick_variants(rv.load_variants(var), only_ids=["nope"])
    assert rv.main(argv(env, "--list", "--only-ids", "v3", "--max-variants", "5")) == 0
    assert "variants: 2" in capsys.readouterr().out


def test_only_available_ignores_pseudo_steps(env, monkeypatch, capsys):
    _, _, _, var = env
    write_var(var, [{"chain": "dfn3>blend50"}])
    _fake_availability(monkeypatch, {})
    assert rv._backend_tokens(["dfn3", "blend50"]) == ["dfn3"]
    assert rv.check_availability(rv.load_variants(var)) == {}
