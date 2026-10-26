"""Kiểm thử run_variants_resident.py với run_batch_by_stage giả (không GPU, không model, không import main).
Chạy trong podcast-pipeline: python -m pytest tests/test_run_variants_resident.py -q"""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_variants_resident", ROOT / "run_variants_resident.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)
rv = rr.rv()

VARIANTS = {"variants": [
    {"chain": "dfn3"},
    {"chain": "none", "chain_pre_asr": "dfn3", "id": "p3"},
    {"chain": "none", "set": {"steps.music_removal": False}, "id": "nomusic"},
    {"chain": "none", "set": {"models.qwen3.batch_size": 4}, "id": "own"},
]}


class FakeRun:
    """Stands in for utils.batch.run_batch_by_stage: writes what DenoiseService would leave behind."""

    def __init__(self, fail_ids=(), raise_ids=(), errors_ids=()):
        self.calls, self.fail_ids, self.raise_ids, self.errors_ids = [], set(fail_ids), set(raise_ids), set(errors_ids)

    def __call__(self, pipeline, args, config, batch, logger=None, stages=None):
        vid = args.job_id[len("exp-"):]
        self.calls.append((vid, copy.copy(args), [Path(p).name for p in batch]))
        if vid in self.raise_ids:
            raise RuntimeError("boom")
        for p in batch:
            d = Path(args.save_path) / Path(p).stem
            (d / "02_diarization").mkdir(parents=True, exist_ok=True)
            for spec, sub in ((args.denoise, "01b_denoise"), (args.denoise_pre_separation, "02b_denoise_pre_separation"),
                              (args.denoise_pre_asr, "04b_denoise_pre_asr")):
                if spec != "none":
                    (d / sub).mkdir(exist_ok=True)
                    info = {"chain_id": spec, "errors": ["x: boom"] if vid in self.errors_ids else [],
                            "corr_with_input": 0.9, "lag_samples": 0}
                    (d / sub / "denoise.json").write_text(json.dumps(info))
        return [(p, "asr: bad") for p in batch[:1]] if vid in self.fail_ids else []


@pytest.fixture
def env(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    for n in ("a.wav", "b.wav"):
        (audio / n).write_bytes(b"audio-" + n.encode())
    var = tmp_path / "variants.json"
    var.write_text(json.dumps(VARIANTS))
    return tmp_path, audio, var


def make(env, *extra):
    tmp, audio, var = env
    args = rr.parse_args(["--audio-dir", str(audio), "--variants", str(var),
                          "--out-root", str(tmp / "out"), *extra])
    variants, files, skipped = rr.resolve_plan(args)
    return args, variants, files, skipped


def runner_for(env, fake, *extra):
    args, variants, files, skipped = make(env, *extra)
    r = rr.ResidentRunner(args, variants, files, run_batch=fake, skipped=skipped, say=lambda m: None)
    r.prepare()
    return r, args, files


def schedule(r, files, **kw):
    pipeline = SimpleNamespace(separation_svc=None)
    base = SimpleNamespace(env="a100", stop_after="asr", keep_models=True, step_asr=True,
                           denoise="none", job_id="exp-resident", save_path="x")
    config = {"environments": {"a100": {"steps": {"asr": True, "diarization": False}}}}
    return r.scheduler(pipeline, base, config, [str(f) for f in files], **kw)


def manifest(tmp):
    return json.loads((tmp / "out" / "variants_manifest.json").read_text())


def test_plan_order_and_per_variant_args(env):
    tmp = env[0]
    fake = FakeRun()
    r, args, files = runner_for(env, fake)
    schedule(r, files)
    assert [c[0] for c in fake.calls] == ["none", "dfn3", "p3", "nomusic"]   # "own" is skipped
    seen = {c[0]: c[1] for c in fake.calls}
    assert (seen["none"].denoise, seen["none"].denoise_pre_separation, seen["none"].denoise_pre_asr) \
        == ("none", "none", "none")
    assert seen["dfn3"].denoise == "dfn3" and seen["dfn3"].denoise_pre_asr == "none"
    assert seen["p3"].denoise == "none" and seen["p3"].denoise_pre_asr == "dfn3"
    assert seen["dfn3"].job_id == "exp-dfn3" and seen["dfn3"].save_path == str(tmp / "out" / "dfn3")
    assert seen["nomusic"].step_music_removal is False and not hasattr(seen["none"], "step_music_removal")
    assert all(a.stop_after == "asr" and a.keep_models for a in seen.values())
    assert [c[2] for c in fake.calls][0] == ["a.wav", "b.wav"]


def test_done_marker_resume_and_manifest(env):
    tmp = env[0]
    r, args, files = runner_for(env, FakeRun())
    schedule(r, files)
    assert all((tmp / "out" / v / ".done").is_file() for v in ("none", "dfn3", "p3", "nomusic"))
    assert not (tmp / "out" / "own" / ".done").exists()
    m = {x["id"]: x for x in manifest(tmp)["variants"]}
    assert {k: x["status"] for k, x in m.items()} == {"none": "ok", "dfn3": "ok", "p3": "ok",
                                                      "nomusic": "ok", "own": "skipped"}
    # second invocation with --resume: nothing runnable, so main would not even be started
    fake2 = FakeRun()
    r2, _, _ = runner_for(env, fake2, "--resume")
    assert r2.runnable() == [] and fake2.calls == []
    assert {x["id"]: x["status"] for x in manifest(tmp)["variants"]}["dfn3"] == "ok"
    # a changed file or setting invalidates the hash
    (env[1] / "a.wav").write_bytes(b"audio-changed!!")
    r3, _, _ = runner_for(env, FakeRun(), "--resume")
    assert len(r3.runnable()) == 4
    r4, _, _ = runner_for(env, FakeRun(), "--resume", "--stop-after", "separation")
    assert len(r4.runnable()) == 4


def test_unsupported_set_key_is_skipped_with_reason(env):
    tmp = env[0]
    fake = FakeRun()
    r, _, files = runner_for(env, fake)
    schedule(r, files)
    own = {x["id"]: x for x in manifest(tmp)["variants"]}["own"]
    assert own["status"] == "skipped"
    assert own["reason"] == "set key models.qwen3.batch_size needs its own process (use tools/run_variants.py)"
    assert "own" not in [c[0] for c in fake.calls]
    for extra in ({"music_separator": "m.ckpt"}, {"extra_args": ["--bss"]}, {"set": {"outputs.x": 1}},
                  {"set": {"pipeline.bss": False}}, {"set": {"pipeline.denoise": "dfn3"}}):
        assert rr.unsupported_reason({"set": {}, **extra}) is not None
    assert rr.unsupported_reason({"set": {"steps.music_analysis": True, "pipeline.music_scope": "spans"}}) is None


def test_build_gated_step_turned_on_is_skipped(env):
    fake = FakeRun()
    config = {"environments": {"a100": {"steps": {"diarization": False}}}}
    assert rr.gate_problem({"set": {"steps.diarization": True}}, config, "a100")
    assert rr.gate_problem({"set": {"steps.diarization": False}}, config, "a100") is None
    assert rr.gate_problem({"set": {"steps.music_analysis": True}}, config, "a100") is None


def test_failure_isolation(env):
    tmp = env[0]
    fake = FakeRun(fail_ids={"dfn3"}, raise_ids={"p3"})
    r, _, files = runner_for(env, fake)
    failures = schedule(r, files)
    assert [c[0] for c in fake.calls] == ["none", "dfn3", "p3", "nomusic"]   # the others still ran
    m = {x["id"]: x for x in manifest(tmp)["variants"]}
    assert m["none"]["status"] == "ok" and m["nomusic"]["status"] == "ok"
    assert m["dfn3"]["status"] == "failed" and "asr: bad" in m["dfn3"]["reason"]
    assert m["p3"]["status"] == "failed" and "boom" in m["p3"]["reason"]
    assert not (tmp / "out" / "dfn3" / ".done").exists() and not (tmp / "out" / "p3" / ".done").exists()
    assert {e.split("]")[0] for _, e in failures} == {"[dfn3", "[p3"}      # aggregated for main


def test_degraded_has_no_done_unless_accepted(env):
    tmp = env[0]
    r, _, files = runner_for(env, FakeRun(errors_ids={"dfn3"}))
    schedule(r, files)
    m = {x["id"]: x for x in manifest(tmp)["variants"]}
    assert m["dfn3"]["status"] == "degraded" and m["dfn3"]["degraded_files"] == 2
    assert not (tmp / "out" / "dfn3" / ".done").exists()
    assert (tmp / "out" / "none" / ".done").is_file()
    r2, _, files = runner_for(env, FakeRun(errors_ids={"dfn3"}), "--accept-degraded")
    schedule(r2, files)
    assert (tmp / "out" / "dfn3" / ".done").is_file()


def test_manifest_written_after_every_variant(env):
    tmp = env[0]
    seen = []

    class Spy(FakeRun):
        def __call__(self, pipeline, args, config, batch, logger=None, stages=None):
            if self.calls:   # the previous variant is already in the manifest when the next starts
                seen.append({x["id"]: x["status"] for x in manifest(tmp)["variants"]})
            return super().__call__(pipeline, args, config, batch, logger, stages)

    r, _, files = runner_for(env, Spy())
    schedule(r, files)
    assert seen[0]["none"] == "ok" and seen[0]["dfn3"] == "missing"
    assert seen[1]["dfn3"] == "ok" and seen[1]["p3"] == "missing"


def test_several_groups_finish_a_variant_once(env):
    tmp = env[0]
    fake = FakeRun()
    r, _, files = runner_for(env, fake)
    schedule(r, files[:1])
    assert not (tmp / "out" / "none" / ".done").exists()      # file b not seen yet
    schedule(r, files[1:])
    assert (tmp / "out" / "none" / ".done").is_file()
    assert [c[0] for c in fake.calls] == ["none", "dfn3", "p3", "nomusic"] * 2


def test_manifest_schema_equals_old_runner(env, tmp_path):
    r, args, files = runner_for(env, FakeRun())
    schedule(r, files)
    mine = manifest(env[0])
    old_root = tmp_path / "old"
    rv.write_manifest(old_root, r.variants, set(r.stems), {}, {s: None for s in r.stems})
    old = json.loads((old_root / "variants_manifest.json").read_text())
    assert set(mine) == set(old)
    assert all(set(x) == set(old["variants"][0]) for x in mine["variants"])
    assert mine["version"] == 1 and mine["baseline"] == "none"
    assert {x["job_id"] for x in mine["variants"]} == {f"exp-{v['id']}" for v in r.variants}


def test_reset_between_variants_clears_shared_state():
    import threading
    dropped = []
    sep = SimpleNamespace(_prefetch_cache={"/in/a.wav": object()}, _prefetch_lock=threading.RLock(),
                          drop_prefetched_plan=lambda p: dropped.append(p))
    pipe = SimpleNamespace(separation_svc=sep, noise_track=object(), timeline=[1],
                           _pending_diar_jobs={"x": 1}, _pending_asr_jobs={"y": 2})
    rr.reset_between_variants(pipe, ["/in/a.wav"])
    assert dropped == ["/in/a.wav"] and sep._prefetch_cache == {}
    assert pipe.noise_track is None and pipe.timeline == [] and not pipe._pending_diar_jobs \
        and not pipe._pending_asr_jobs


def test_dry_run_prints_plan_and_argv_without_running(env, capsys):
    tmp, audio, var = env
    code = rr.main(["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(tmp / "out"),
                    "--dry-run", "--set", "steps.captioning=false"])
    out = capsys.readouterr().out
    assert code == 0
    assert "--by_stage" in out and "--keep_models" in out and "input_<timestamp>" in out
    assert "[dfn3] " in out and '"job_id": "exp-dfn3"' in out
    assert "[own] SKIPPED: set key models.qwen3.batch_size needs its own process" in out
    assert not (tmp / "out").exists()


def test_argv_has_stage_flags_and_layout_sets(env):
    args, *_ = make(env, "--set", "steps.captioning=false", "--gpu-1", "2", "--gpu-2", "3",
                    "--cache-dir", "/c")
    argv = rr.build_argv(args, Path("/in"))
    assert {"--by_stage", "--keep_models"} <= set(argv)
    assert argv[argv.index("--audio_dir") + 1] == "/in" and argv[argv.index("--job_id") + 1] == "exp-resident"
    assert argv[argv.index("--gpu_1") + 1] == "2" and argv[argv.index("--gpu_2") + 1] == "3"
    assert argv[argv.index("--cache_dir") + 1] == "/c"
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]
    assert sets[0] == 'outputs.clean_two_channel_dir=""'
    assert 'performance.stages.asr.qwen3_gpu="gpu_2"' in sets and sets[-1] == "steps.captioning=false"
    flat, *_ = make(env, "--layout", "default")
    assert [a for a in rr.build_argv(flat, Path("/in")) if a.startswith("performance.")] == []


def test_split_layout_is_valid_against_the_real_schema():
    from utils.config_override import apply_overrides
    from utils import performance_config
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    profile = copy.deepcopy(config["environments"]["a100"])
    for name, preset in rr.LAYOUT_PRESETS.items():
        prof = copy.deepcopy(profile)
        apply_overrides(prof, rr.layout_sets(name))
        resolved = performance_config.resolve(prof)
        assert resolved["_problems"] == [], (name, resolved["_problems"])      # no unknown key, no bad value
    prof = copy.deepcopy(profile)
    apply_overrides(prof, rr.layout_sets("split"))
    st = performance_config.resolve(prof)["stages"]
    assert st["asr"]["qwen3_gpu"] == "gpu_2" and st["asr"]["dynamic_replicas"] is False
    assert st["diarization"]["placement"] == "single" and st["diarization"]["workers"] == 1
    assert st["separation"]["max_workers"] == 1 and st["music"]["max_separator_workers"] == 1
    assert performance_config.resolve_asr_placement(st["asr"], 0, 1)["qwen3"] == 1
    assert all(k.startswith("performance.") for k in rr.LAYOUT_PRESETS["split"])


def test_importing_module_does_not_import_main():
    code = ("import sys, importlib.util\n"
            f"s = importlib.util.spec_from_file_location('m', {str(ROOT / 'run_variants_resident.py')!r})\n"
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
            "print('main' in sys.modules, 'torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/")
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"]


def test_mode_overlap_exits_with_message(env):
    tmp, audio, var = env
    with pytest.raises(SystemExit) as exc:
        rr.parse_args(["--audio-dir", str(audio), "--out-root", str(tmp / "out"), "--mode", "overlap"])
    assert "not implemented" in str(exc.value) and "shares per-stage state" in str(exc.value)
    assert set(rr.MODES) == {"sequential"}


def test_failed_file_is_cleared_with_the_variants_own_args(env):
    fake = FakeRun(fail_ids={"dfn3"}, raise_ids={"p3"})
    r, _, files = runner_for(env, fake)
    cleared = []
    r.discard_partial = lambda ledger, vargs, path, logger, pipeline: cleared.append(
        (vargs.job_id, vargs.save_path, Path(path).name))
    pipeline = SimpleNamespace(separation_svc=None, progress_ledger=object())
    base = SimpleNamespace(env="a100", stop_after="asr", keep_models=True, step_asr=True,
                           denoise="none", job_id="exp-resident", save_path="x")
    config = {"environments": {"a100": {"steps": {"asr": True, "diarization": False}}}}
    r.scheduler(pipeline, base, config, [str(f) for f in files])
    assert [c[0] for c in cleared] == ["exp-dfn3"]            # its own job id, not exp-resident
    assert cleared[0][1].endswith("/dfn3")                    # its own output dir
    assert cleared[0][2] == "a.wav"                           # only the file that failed
    # the variant whose whole pass crashed is not cleared file by file, and the others are untouched
    assert all(c[0] != "exp-p3" and c[0] != "exp-none" for c in cleared)


def test_discard_error_does_not_stop_the_run(env):
    fake = FakeRun(fail_ids={"dfn3"})
    r, _, files = runner_for(env, fake)
    def boom(*a):
        raise OSError("disk")
    r.discard_partial = boom
    pipeline = SimpleNamespace(separation_svc=None, progress_ledger=object())
    base = SimpleNamespace(env="a100", stop_after="asr", keep_models=True, step_asr=True,
                           denoise="none", job_id="exp-resident", save_path="x")
    config = {"environments": {"a100": {"steps": {"asr": True, "diarization": False}}}}
    r.scheduler(pipeline, base, config, [str(f) for f in files])
    assert "nomusic" in [c[0] for c in fake.calls]


def test_split_strict_keeps_assignment_workers_off_the_second_card():
    assert rr.LAYOUT_PRESETS["split_strict"]["performance.stages.separation.assignment_workers_per_gpu"] == 0
    assert "assignment_workers_per_gpu" not in " ".join(rr.LAYOUT_PRESETS["split"])
    from utils.config_override import apply_overrides
    from utils import performance_config
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    prof = copy.deepcopy(config["environments"]["a100"])
    apply_overrides(prof, rr.layout_sets("split_strict"))
    assert performance_config.resolve(prof)["stages"]["separation"]["assignment_workers_per_gpu"] == 0


def test_layout_warnings():
    mk = lambda **k: SimpleNamespace(**{"layout": "split", "gpu_1": 0, "stop_after": "asr", **k})
    assert rr.layout_warnings(mk()) == []
    assert rr.layout_warnings(mk(layout="default", gpu_1=1, stop_after="none")) == []
    assert any("cuda:0" in w for w in rr.layout_warnings(mk(gpu_1=1)))
    assert any("refinement" in w for w in rr.layout_warnings(mk(stop_after="none")))
    assert any("refinement" in w for w in rr.layout_warnings(mk(layout="split_strict", stop_after="refinement")))
