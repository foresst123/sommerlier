"""Kiểm thử run_variants_resident.py với run_batch_by_stage giả (không GPU, không model, không import main).
Chạy trong podcast-pipeline: python -m pytest tests/test_run_variants_resident.py -q"""
import copy
import importlib.util
import json
import os
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
                point_stage = {"01b_denoise": "denoise", "02b_denoise_pre_separation": "denoise_pre_separation",
                               "04b_denoise_pre_asr": "denoise_pre_asr"}[sub]
                if spec != "none" and rr.stage_rank(args.stop_after) >= rr.stage_rank(point_stage):
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
    music = [] if "--music" in extra else ["--music", "profile"]     # these tests are about the plan, not the music switch
    args = rr.parse_args(["--audio-dir", str(audio), "--variants", str(var),
                          "--out-root", str(tmp / "out"), *music, *extra])
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
    # a file changed after the list was frozen keeps the hashes (a warning says so); --refresh-files applies it
    (env[1] / "a.wav").write_bytes(b"audio-changed!!")
    r3, _, _ = runner_for(env, FakeRun(), "--resume")
    assert r3.runnable() == []
    r5, _, _ = runner_for(env, FakeRun(), "--resume", "--refresh-files")
    assert len(r5.runnable()) == 4
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
    assert all(set(x) == set(old["variants"][0]) | {"stop_after"} for x in mine["variants"])
    assert {x["stop_after"] for x in mine["variants"]} == {"asr"}
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


# ------------------------------------------------------------------ assembly line (several processes)
class Clock:
    """Injected clock + sleep: sleeping only moves the clock, and may run a hook (e.g. the producer finishing)."""

    def __init__(self):
        self.now, self.sleeps, self.hook = 0.0, [], None

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        if self.hook:
            self.hook(len(self.sleeps))


def line_runner(env, fake, clock=None, *extra):
    args, variants, files, skipped = make(env, *extra)
    clock = clock or Clock()
    r = rr.ResidentRunner(args, variants, files, run_batch=fake, skipped=skipped, say=lambda m: None,
                          sleep=clock.sleep, clock=clock.time)
    r.prepare()
    return r, args, files


def producer_then_files(env, *extra):
    r, args, files = line_runner(env, FakeRun(), None, "--layout", "default", "--stop-after", "separation",
                                 "--signal", "separation", *extra)
    schedule_stop(r, files, "separation")
    return r, args, files


def schedule_stop(r, files, stop):
    pipeline = SimpleNamespace(separation_svc=None)
    base = SimpleNamespace(env="a100", stop_after=stop, keep_models=True, denoise="none",
                           job_id="exp-resident", save_path="x")
    config = {"environments": {"a100": {"steps": {}}}}
    return r.scheduler(pipeline, base, config, [str(f) for f in files])


def test_producer_writes_ready_with_the_consumers_hash_and_its_own_done(env):
    tmp = env[0]
    r, args, files = producer_then_files(env)
    out = tmp / "out"
    cargs, variants, cfiles, _ = make(env, "--stop-after", "asr", "--wait-signal", "separation")
    for v in variants:
        if v["id"] == "own":
            continue
        vd = out / v["id"]
        ready = json.loads((vd / ".ready_separation").read_text())
        assert ready["hash"] == rr.variant_hash(v, cargs, cfiles, stop_after="separation")
        assert ready["stop_after"] == "separation" and "time" in ready
        assert (vd / ".done.separation").is_file() and not (vd / ".done").exists()
        assert not (vd / ".failed_separation").exists()


def test_stale_ready_is_removed_when_the_producer_starts_fresh(env):
    tmp = env[0]
    for vid in ("dfn3", "none"):
        d = tmp / "out" / vid
        d.mkdir(parents=True)
        (d / ".ready_separation").write_text("{}")
        (d / ".failed_separation").write_text("{}")
    seen = []

    class Spy(FakeRun):
        def __call__(self, pipeline, args, config, batch, logger=None, stages=None):
            vd = Path(args.save_path)
            seen.append((args.job_id, (vd / ".ready_separation").exists(), (vd / ".failed_separation").exists()))
            return super().__call__(pipeline, args, config, batch, logger, stages)

    r, args, files = line_runner(env, Spy(), None, "--layout", "default", "--stop-after", "separation",
                                 "--signal", "separation")
    schedule_stop(r, files, "separation")
    assert ("exp-dfn3", False, False) in seen and ("exp-none", False, False) in seen


def test_producer_failure_writes_failed_marker(env):
    tmp = env[0]
    r, args, files = line_runner(env, FakeRun(fail_ids={"dfn3"}), None, "--layout", "default",
                                 "--stop-after", "separation", "--signal", "separation")
    schedule_stop(r, files, "separation")
    failed = json.loads((tmp / "out" / "dfn3" / ".failed_separation").read_text())
    assert "asr: bad" in failed["reason"] and not (tmp / "out" / "dfn3" / ".ready_separation").exists()


def test_stop_aware_assess_does_not_flag_pre_asr_chain_for_a_producer(env):
    tmp = env[0]
    r, args, files = producer_then_files(env)
    m = {x["id"]: x for x in manifest(tmp)["variants"]}
    assert m["p3"]["status"] == "ok" and m["p3"]["degraded_files"] == 0
    assert m["p3"]["stop_after"] == "separation"
    full = rr.reached_variant({"chain": ["a"], "chain_pre_separation": ["b"], "chain_pre_asr": ["c"]}, "denoise")
    assert full == {"chain": ["a"], "chain_pre_separation": [], "chain_pre_asr": []}
    assert rr.reached_variant({"chain": ["a"], "chain_pre_asr": ["c"]}, "asr")["chain_pre_asr"] == ["c"]
    assert rr.reached_variant({"chain": ["a"], "chain_pre_asr": ["c"]}, None)["chain_pre_asr"] == ["c"]
    assert rr.reached_variant({"chain": ["a"]}, "music")["chain"] == []


def consumer(env, clock, *extra, fake=None):
    fake = fake or FakeRun()
    r, args, files = line_runner(env, fake, clock, "--layout", "default", "--stop-after", "asr",
                                 "--wait-signal", "separation", "--poll-seconds", "5", *extra)
    return r, args, files, fake


def signal_ready(env, args, files, vid):
    variants = {v["id"]: v for v in rr.resolve_plan(args)[0]}
    d = env[0] / "out" / vid
    d.mkdir(parents=True, exist_ok=True)
    digest = rr.variant_hash(variants[vid], args, files, stop_after="separation")
    (d / ".ready_separation").write_text(json.dumps({"hash": digest, "stop_after": "separation"}))


def test_consumer_waits_then_runs_ready_variants_out_of_plan_order(env):
    clock = Clock()
    r, args, files, fake = consumer(env, clock)
    # nothing ready at first; after the 2nd sleep the producer finishes p3, after the 4th dfn3, then the rest
    order = {2: ["p3"], 4: ["dfn3"], 5: ["none", "nomusic"]}
    clock.hook = lambda n: [signal_ready(env, args, files, vid) for vid in order.get(n, [])]
    schedule_stop(r, files, "asr")
    assert [c[0] for c in fake.calls] == ["p3", "dfn3", "none", "nomusic"]     # plan order is none, dfn3, p3, nomusic
    assert clock.sleeps[:2] == [5, 5]
    assert (env[0] / "out" / "p3" / ".done").is_file() and not (env[0] / "out" / "p3" / ".done.asr").exists()


def test_consumer_honours_failed_marker_without_running(env):
    clock = Clock()
    fake = FakeRun()
    r, args, files, fake = consumer(env, clock, fake=fake)
    for vid in ("none", "p3", "nomusic"):
        signal_ready(env, args, files, vid)
    (env[0] / "out" / "dfn3").mkdir(parents=True)
    (env[0] / "out" / "dfn3" / ".failed_separation").write_text(json.dumps({"reason": "OOM in sidon"}))
    failures = schedule_stop(r, files, "asr")
    assert "dfn3" not in [c[0] for c in fake.calls] and clock.sleeps == []
    m = {x["id"]: x for x in manifest(env[0])["variants"]}
    assert m["dfn3"]["status"] == "failed" and m["dfn3"]["reason"] == "producer failed: OOM in sidon"
    assert any("producer failed: OOM in sidon" in e for _, e in failures)


def test_consumer_timeout_marks_pending_failed(env):
    clock = Clock()
    r, args, files, fake = consumer(env, clock, "--wait-timeout", "1")
    signal_ready(env, args, files, "none")
    schedule_stop(r, files, "asr")
    assert [c[0] for c in fake.calls] == ["none"]
    m = {x["id"]: x for x in manifest(env[0])["variants"]}
    assert m["none"]["status"] == "ok"
    for vid in ("dfn3", "p3", "nomusic"):
        assert m[vid]["status"] == "failed" and m[vid]["reason"] == "producer never signalled"
    assert clock.now >= 60


def test_consumer_ignores_a_ready_with_a_stale_hash(env):
    clock = Clock()
    r, args, files, fake = consumer(env, clock, "--wait-timeout", "1")
    d = env[0] / "out" / "none"
    d.mkdir(parents=True)
    (d / ".ready_separation").write_text(json.dumps({"hash": "deadbeef", "stop_after": "separation"}))
    schedule_stop(r, files, "asr")
    assert fake.calls == []
    assert {x["id"]: x for x in manifest(env[0])["variants"]}["none"]["reason"] == "producer never signalled"


def test_consumer_resume_uses_plain_done_and_skips_finished(env):
    clock = Clock()
    r, args, files, fake = consumer(env, clock, "--wait-timeout", "1")
    for vid in ("none", "dfn3", "p3", "nomusic"):
        signal_ready(env, args, files, vid)
    schedule_stop(r, files, "asr")
    assert len(fake.calls) == 4
    r2, _, _, fake2 = consumer(env, Clock(), "--resume")
    assert r2.runnable() == [] and fake2.calls == []


def test_shards_are_disjoint_and_cover_the_plan(env):
    _, variants, files, skipped = make(env)
    plan = [v["id"] for v in variants if v["id"] != "own"]       # "own" is skipped, so not in the index
    for n in (1, 2, 3):
        parts = [rr.plan_ownership(variants, skipped, (k, n)) for k in range(1, n + 1)]
        assert set().union(*parts) == set(plan) and sum(len(x) for x in parts) == len(plan)
    assert rr.plan_ownership(variants, skipped, (1, 2)) == {"none", "p3"}
    assert rr.plan_ownership(variants, skipped, (2, 2)) == {"dfn3", "nomusic"}
    fake = FakeRun()
    r, _, files2 = line_runner(env, fake, None, "--shard", "2/2")
    schedule(r, files2)
    assert [c[0] for c in fake.calls] == ["dfn3", "nomusic"]


def test_manifest_merge_keeps_other_process_entries_and_later_stage_wins(env):
    tmp = env[0]
    prod, _, files = line_runner(env, FakeRun(), None, "--layout", "default", "--stop-after", "separation",
                                 "--signal", "separation", "--shard", "1/2")
    schedule_stop(prod, files, "separation")
    m = manifest(tmp)
    assert [x["id"] for x in m["variants"]] == ["none", "p3", "own"]        # only what shard 1 owns (+ skipped)
    cons, cargs, files = line_runner(env, FakeRun(), None, "--layout", "default", "--stop-after", "asr",
                                     "--wait-signal", "separation", "--shard", "1/2")
    m = {x["id"]: x for x in manifest(tmp)["variants"]}
    assert m["none"]["stop_after"] == "separation" and m["none"]["status"] == "ok"   # consumer's "missing" did not hide it
    schedule_stop(cons, files, "asr")
    m = manifest(tmp)
    ids = [x["id"] for x in m["variants"]]
    assert ids == ["none", "p3", "own"]
    assert {x["id"]: x["stop_after"] for x in m["variants"]}["none"] == "asr"
    # an earlier stage never overwrites a later one
    prod2, _, files = line_runner(env, FakeRun(), None, "--layout", "default", "--stop-after", "separation",
                                  "--signal", "separation", "--shard", "1/2", "--resume")
    assert {x["id"]: x["stop_after"] for x in manifest(tmp)["variants"]}["none"] == "asr"
    # the other shard's entries appear in plan order
    other, _, files = line_runner(env, FakeRun(), None, "--shard", "2/2")
    schedule(other, files)
    assert [x["id"] for x in manifest(tmp)["variants"]] == ["none", "dfn3", "p3", "nomusic", "own"]


def test_manifest_written_by_two_writers_stays_valid(env):
    import threading
    r1, _, _ = line_runner(env, FakeRun(), None, "--shard", "1/2")
    r2, _, _ = line_runner(env, FakeRun(), None, "--shard", "2/2")
    errors = []

    def hammer(r):
        try:
            for _ in range(25):
                r._manifest()
        except Exception as exc:        # a torn read or a clash on the scratch / tmp files
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(r,)) for r in (r1, r2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    m = manifest(env[0])
    assert [x["id"] for x in m["variants"]] == ["none", "dfn3", "p3", "nomusic", "own"]
    assert not list((env[0] / "out").glob(".manifest_*"))


def test_other_tools_ignore_the_extra_manifest_key(env):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from compare_lib.build import build_compare_data
    finally:
        sys.path.pop(0)
    r, args, files = runner_for(env, FakeRun())
    schedule(r, files)
    data = build_compare_data(str(env[0] / "out" / "variants_manifest.json"))
    assert {v["id"] for v in data["variants"]} >= {"none", "dfn3"}


def test_validation_errors(env, capsys):
    tmp, audio, var = env
    base = ["--audio-dir", str(audio), "--out-root", str(tmp / "out")]

    def bad(*extra):
        with pytest.raises(SystemExit) as exc:
            rr.parse_args(base + list(extra))
        assert exc.value.code == 2
        return capsys.readouterr().err

    assert "--shard" in bad("--shard", "0/2") and "--shard" in bad("--shard", "3/2") and "--shard" in bad("--shard", "x")
    assert "must equal --stop-after" in bad("--signal", "separation")               # stop-after defaults to asr
    assert "must equal --stop-after" in bad("--signal", "asr", "--stop-after", "separation")
    assert "earlier than --stop-after" in bad("--wait-signal", "asr")
    assert "earlier than --stop-after" in bad("--wait-signal", "asr", "--stop-after", "separation")
    assert "earlier than --stop-after" in bad("--wait-signal", "nonsense")
    assert "--poll-seconds" in bad("--poll-seconds", "0")
    ok = rr.parse_args(base + ["--shard", "2/3", "--signal", "separation", "--stop-after", "separation"])
    assert ok.shard == (2, 3)


def test_stage_order_matches_pipeline_service_source():
    import ast
    tree = ast.parse((ROOT / "services" / "pipeline_service.py").read_text(encoding="utf-8"))
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_STOP_ORDER" for t in node.targets):
            found = tuple(ast.literal_eval(node.value))
    # separation_plan is added to _STOP_ORDER by the H1 branch: until it is there, the rest must match exactly
    assert tuple(s for s in rr.STAGE_ORDER if s != "separation_plan") == tuple(
        s for s in found if s != "separation_plan")
    if "separation_plan" in found:
        assert found == rr.STAGE_ORDER


def test_input_dir_name_has_the_pid():
    src = (ROOT / "run_variants_resident.py").read_text(encoding="utf-8")
    assert "{os.getpid()}" in src and "input_{datetime.now()" in src


def test_dry_run_shows_role_shard_and_owned_variants(env, capsys):
    tmp, audio, var = env
    base = ["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(tmp / "out"), "--dry-run"]
    assert rr.main(base + ["--stop-after", "separation", "--signal", "separation", "--shard", "1/2"]) == 0
    out = capsys.readouterr().out
    assert "role: producer" in out and "shard: 1/2" in out and "signal: separation" in out
    assert "this process owns: none, p3" in out
    assert "[dfn3] {" not in out and "[p3] {" in out
    assert rr.main(base + ["--wait-signal", "separation", "--poll-seconds", "7", "--wait-timeout", "30"]) == 0
    out = capsys.readouterr().out
    assert "role: consumer" in out and "wait-signal: separation (poll 7s, timeout 30 min)" in out
    assert rr.main(base) == 0
    assert "role: all" in capsys.readouterr().out


def test_music_off_is_the_default_and_forces_both_music_steps_off(env):
    tmp, audio, var = env
    args = rr.parse_args(["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(tmp / "out")])
    assert args.music == "off"
    assert args.global_set[:2] == ["steps.music_analysis=false", "steps.music_removal=false"]
    assert args.cache_dir == "cache_nomusic"                      # never read a cache written with music removal on
    argv = rr.build_argv(args, Path("/in"))
    assert "steps.music_removal=false" in argv and argv[argv.index("--cache_dir") + 1] == "cache_nomusic"


def test_music_profile_changes_nothing(env):
    args, *_ = make(env)
    assert args.music == "profile" and args.cache_dir is None
    assert not [g for g in args.global_set if g.startswith("steps.music")]


def test_music_off_keeps_an_explicit_cache_dir_and_user_sets_last(env):
    tmp, audio, var = env
    args = rr.parse_args(["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(tmp / "out"),
                          "--cache-dir", "mycache", "--set", "steps.music_removal=true"])
    assert args.cache_dir == "mycache"
    assert args.global_set[-1] == "steps.music_removal=true"       # the user's own --set wins (applied last)


def test_variant_that_only_turns_music_off_is_the_baseline_again(env):
    tmp, audio, var = env
    raw = {"variants": [{"chain": "none", "id": "raw", "set": {"steps.music_analysis": False, "steps.music_removal": False}},
                        {"chain": "dfn3", "id": "f"}]}
    var.write_text(json.dumps(raw))
    args = rr.parse_args(["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(tmp / "out")])
    variants, files, skipped = rr.resolve_plan(args)
    assert "raw" in skipped and "f" not in skipped and "none" not in skipped
    args = rr.parse_args(["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(tmp / "out"),
                          "--music", "profile"])
    assert rr.resolve_plan(args)[2] == {}


def test_hash_ignores_speed_knobs_but_not_result_settings(env):
    tmp, audio, var = env
    base = ["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(tmp / "out"), "--music", "profile"]
    a = rr.parse_args(base)
    b = rr.parse_args(base + ["--set", "performance.stages.separation.workers_per_gpu=1"])
    c = rr.parse_args(base + ["--set", "steps.music_removal=false"])
    variants, files, _ = rr.resolve_plan(a)
    v = variants[0]
    assert rr.variant_hash(v, a, files) == rr.variant_hash(v, b, files)       # workers_per_gpu: same variant
    assert rr.variant_hash(v, a, files) != rr.variant_hash(v, c, files)       # a step changes the result
    d = rr.parse_args(base + ["--set", 'outputs.clean_two_channel_dir="/x/final"'])
    assert rr.variant_hash(v, a, files) == rr.variant_hash(v, d, files)       # the final root is set on "all" only


# ------------------------------------------------- no-op stages and per-file hand-over
def mark_ready(env, args, files, vid, stage, stem=None):
    """What a producer of `stage` leaves for a consumer (stem=None: variant-level marker)."""
    variants = {v["id"]: v for v in rr.resolve_plan(args)[0]}
    d = env[0] / "out" / vid
    d.mkdir(parents=True, exist_ok=True)
    digest = rr.variant_hash(variants[vid], args, files, stop_after=stage)
    (d / (f".ready_{stage}" + (f".{stem}" if stem else ""))).write_text(
        json.dumps({"hash": digest, "stop_after": stage}))


def test_noop_stage_detection_and_music_mode():
    none = {"chain": [], "chain_pre_separation": [], "chain_pre_asr": [], "set": {}}
    p1 = {**none, "chain": ["dfn3"]}
    p2 = {**none, "chain_pre_separation": ["dfn3"]}
    p3 = {**none, "chain_pre_asr": ["dfn3"]}
    assert rr.noop_stages(none, "off") == ["denoise", "denoise_pre_separation", "denoise_pre_asr"]
    assert rr.noop_stages(p1, "off") == ["denoise_pre_separation", "denoise_pre_asr"]
    assert rr.noop_stages(p2, "off") == ["denoise", "denoise_pre_asr"]
    assert rr.noop_stages(p3, "off") == ["denoise", "denoise_pre_separation"]
    # with the profile's music removal the denoise stage is real work even without a chain
    assert rr.noop_stages(none, "profile") == ["denoise_pre_separation", "denoise_pre_asr"]
    off_set = {**none, "set": {"steps.music_analysis": False, "steps.music_removal": False}}
    assert "denoise" in rr.noop_stages(off_set, "profile")
    on_set = {**none, "set": {"steps.music_removal": True}}
    assert "denoise" not in rr.noop_stages(on_set, "off")
    for stage in ("diarization", "separation", "asr"):
        assert not rr.is_noop(none, stage, "off")


def test_effective_stage_walks_back_over_noops():
    none = {"chain": [], "chain_pre_separation": [], "chain_pre_asr": [], "set": {}}
    p2 = {**none, "chain_pre_separation": ["x"]}
    assert rr.effective_stage(none, "denoise") is None                       # whole line so far is a no-op
    assert rr.effective_stage(none, "diarization") == "diarization"
    assert rr.effective_stage(none, "denoise_pre_separation") == "diarization"
    assert rr.effective_stage(p2, "denoise_pre_separation") == "denoise_pre_separation"
    assert rr.effective_stage(none, "denoise_pre_asr") == "separation"
    assert rr.effective_stage(none, "denoise", "profile") == "denoise"
    assert rr.effective_stage(none, "asr") == "asr"


def test_consumer_skips_noop_stages_and_starts_at_once_when_nothing_is_upstream(env):
    clock = Clock()
    fake = FakeRun()
    r, args, files = line_runner(env, fake, clock, "--music", "off", "--layout", "default", "--stop-after", "diarization",
                                 "--wait-signal", "denoise", "--signal", "diarization", "--wait-timeout", "1")
    schedule_stop(r, files, "diarization")
    # none and p3 have no point-1 chain and music is off: no marker needed. dfn3 needs .ready_denoise (never comes)
    assert [c[0] for c in fake.calls] == ["none", "p3"]
    m = {x["id"]: x for x in manifest(env[0])["variants"]}
    assert m["dfn3"]["reason"] == "producer never signalled" and m["none"]["status"] == "ok"
    # the next process looks past the no-op denoise_pre_separation stage straight at the diarization marker
    fake2 = FakeRun()
    r2, args2, files2 = line_runner(env, fake2, Clock(), "--music", "off", "--layout", "default", "--stop-after", "separation",
                                    "--wait-signal", "denoise_pre_separation", "--signal", "separation",
                                    "--poll-seconds", "5", "--wait-timeout", "1")
    nonev = next(v for v in r2.variants if v["id"] == "none")
    (env[0] / "out" / "none" / ".ready_diarization").unlink()
    assert r2._signal_state(nonev) == (None, None)         # it waits for the diarization marker, not denoise_pre_separation
    mark_ready(env, args2, files2, "none", "diarization")
    assert r2._signal_state(nonev) == ("ready", None)


def test_noop_producer_hands_over_without_running_and_only_after_its_upstream(env):
    tmp = env[0]
    clock = Clock()
    fake = FakeRun()
    r, args, files = line_runner(env, fake, clock, "--layout", "default", "--stop-after", "denoise_pre_separation",
                                 "--wait-signal", "diarization", "--signal", "denoise_pre_separation",
                                 "--poll-seconds", "5", "--wait-timeout", "5")
    ready = lambda vid: tmp / "out" / vid / ".ready_denoise_pre_separation"
    seen = {}

    def hook(n):
        seen[n] = (ready("none").exists(), ready("p3").exists())
        if n == 2:
            mark_ready(env, args, files, "none", "diarization")
            mark_ready(env, args, files, "p3", "diarization")
    clock.hook = hook
    schedule_stop(r, files, "denoise_pre_separation")
    assert fake.calls == []                                            # never decodes the audio again
    assert seen[1] == (False, False) and seen[2] == (False, False)     # nothing handed over before its upstream
    assert ready("none").is_file() and ready("p3").is_file()
    assert json.loads(ready("none").read_text())["stop_after"] == "denoise_pre_separation"
    assert (tmp / "out" / "none" / ".done.denoise_pre_separation").is_file()
    m = {x["id"]: x for x in manifest(tmp)["variants"]}
    assert m["none"]["status"] == "ok" and m["none"]["reason"] == rr.NOOP_NOTE
    # dfn3 has no pre_separation chain either, but its upstream never came: failed after the timeout
    assert m["dfn3"]["status"] == "failed"


def test_noop_producer_with_nothing_upstream_writes_its_marker_at_once(env):
    fake = FakeRun()
    r, args, files = line_runner(env, fake, Clock(), "--music", "off", "--layout", "default", "--stop-after", "denoise",
                                 "--signal", "denoise")
    schedule_stop(r, files, "denoise")
    assert [c[0] for c in fake.calls] == ["dfn3"]                       # only dfn3 has real work at point 1
    for vid in ("none", "dfn3", "p3"):
        assert (env[0] / "out" / vid / ".ready_denoise").is_file()


def pf(env, fake, clock, *extra, msgs=None):
    args, variants, files, skipped = make(env, "--per-file", "--only-ids", "dfn3,p3", *extra)
    r = rr.ResidentRunner(args, variants, files, run_batch=fake, skipped=skipped,
                          say=(msgs.append if msgs is not None else (lambda m: None)),
                          sleep=clock.sleep, clock=clock.time)
    r.prepare()
    return r, args, files


class FileFake(FakeRun):
    """Fails the files whose stem is listed for a variant, like run_batch_by_stage reports a failed file."""

    def __init__(self, bad=()):
        super().__init__()
        self.bad = set(bad)

    def __call__(self, pipeline, args, config, batch, logger=None, stages=None):
        vid = args.job_id[len("exp-"):]
        out = super().__call__(pipeline, args, config, batch, logger, stages)
        return [(p, "sidon: boom") for p in batch if (vid, Path(p).stem) in self.bad]


@pytest.fixture
def env_sizes(env):
    (env[1] / "a.wav").write_bytes(b"x" * 100)         # a is the long file
    return env


def test_per_file_producer_small_file_first_with_markers_per_stem(env_sizes):
    env = env_sizes
    tmp = env[0]
    fake = FileFake()
    r, args, files = pf(env, fake, Clock(), "--layout", "default", "--stop-after", "separation",
                        "--signal", "separation")
    schedule_stop(r, files, "separation")
    assert [(c[0], c[2]) for c in fake.calls] == [(v, [n]) for v in ("none", "dfn3", "p3") for n in ("b.wav", "a.wav")]
    for vid in ("none", "dfn3", "p3"):
        d = tmp / "out" / vid
        for stem in ("a", "b"):
            data = json.loads((d / f".ready_separation.{stem}").read_text())
            assert data["stop_after"] == "separation" and "hash" in data and "time" in data
        assert not (d / ".ready_separation").exists()                  # no variant-level marker in this mode
        assert (d / ".done.separation").is_file() and not (d / ".done").exists()


def test_per_file_failed_stem_does_not_stop_the_others_and_variant_is_finalized_once(env_sizes):
    env = env_sizes
    tmp = env[0]
    msgs = []
    fake = FileFake(bad={("dfn3", "a")})
    r, args, files = pf(env, fake, Clock(), "--layout", "default", "--stop-after", "separation",
                        "--signal", "separation", msgs=msgs)
    failures = schedule_stop(r, files, "separation")
    d = tmp / "out" / "dfn3"
    assert (d / ".failed_separation.a").is_file() and (d / ".ready_separation.b").is_file()
    assert "sidon: boom" in json.loads((d / ".failed_separation.a").read_text())["reason"]
    assert not (d / ".done.separation").exists()
    assert len([m for m in msgs if m.startswith("[dfn3] FAILED")]) == 1
    assert len([m for m in msgs if m.startswith("[none] ok")]) == 1
    assert [p for p, _ in failures] and all("[dfn3]" in e for _, e in failures)
    m = {x["id"]: x for x in manifest(tmp)["variants"]}
    assert m["dfn3"]["status"] == "failed" and m["none"]["status"] == "ok"


def test_per_file_consumer_runs_units_as_markers_appear_out_of_order(env_sizes):
    env = env_sizes
    tmp = env[0]
    clock = Clock()
    fake = FileFake()
    msgs = []
    r, args, files = pf(env, fake, clock, "--layout", "default", "--stop-after", "asr",
                        "--wait-signal", "separation", "--poll-seconds", "5", msgs=msgs)
    steps = {1: [("dfn3", "a")], 2: [("none", "b")],
             3: [(v, s) for v in ("none", "dfn3", "p3") for s in ("a", "b")]}
    clock.hook = lambda n: [mark_ready(env, args, files, v, "separation", s) for v, s in steps.get(n, [])]
    schedule_stop(r, files, "asr")
    got = [(c[0], c[2][0][0]) for c in fake.calls]
    assert got[:2] == [("dfn3", "a"), ("none", "b")]
    assert got[2:] == [("none", "a"), ("dfn3", "b"), ("p3", "b"), ("p3", "a")]     # plan order, small file first
    for vid in ("none", "dfn3", "p3"):
        assert (tmp / "out" / vid / ".done").is_file() and not (tmp / "out" / vid / ".done.asr").exists()
    assert len([m for m in msgs if m.startswith("[none] ok")]) == 1


def test_per_file_consumer_honours_failed_stem_and_finishes_the_rest(env_sizes):
    env = env_sizes
    tmp = env[0]
    fake = FileFake()
    r, args, files = pf(env, fake, Clock(), "--layout", "default", "--stop-after", "asr",
                        "--wait-signal", "separation", "--wait-timeout", "1")
    for v in ("none", "dfn3", "p3"):
        for s in ("a", "b"):
            if (v, s) != ("none", "a"):
                mark_ready(env, args, files, v, "separation", s)
    (tmp / "out" / "none").mkdir(parents=True, exist_ok=True)
    (tmp / "out" / "none" / ".failed_separation.a").write_text(json.dumps({"reason": "OOM in sidon"}))
    schedule_stop(r, files, "asr")
    assert ("none", ["a.wav"]) not in [(c[0], c[2]) for c in fake.calls]
    assert ("none", ["b.wav"]) in [(c[0], c[2]) for c in fake.calls]
    m = {x["id"]: x for x in manifest(tmp)["variants"]}
    assert m["none"]["status"] == "failed" and "producer failed: OOM in sidon" in m["none"]["reason"]
    assert m["dfn3"]["status"] == "ok"


def test_per_file_timeout_fails_only_unsignalled_stems(env_sizes):
    env = env_sizes
    fake = FileFake()
    r, args, files = pf(env, fake, Clock(), "--layout", "default", "--stop-after", "asr",
                        "--wait-signal", "separation", "--wait-timeout", "1")
    mark_ready(env, args, files, "none", "separation", "b")
    schedule_stop(r, files, "asr")
    assert [(c[0], c[2]) for c in fake.calls] == [("none", ["b.wav"])]
    m = {x["id"]: x for x in manifest(env[0])["variants"]}
    assert m["none"]["status"] == "failed" and "producer never signalled" in m["none"]["reason"]


def test_per_file_noop_producer_passes_stems_through(env_sizes):
    env = env_sizes
    clock = Clock()
    fake = FileFake()
    r, args, files = pf(env, fake, clock, "--layout", "default", "--stop-after", "denoise_pre_separation",
                        "--wait-signal", "diarization", "--signal", "denoise_pre_separation",
                        "--poll-seconds", "5", "--wait-timeout", "2")
    d = env[0] / "out" / "none"
    seen = {}

    def hook(n):
        seen[n] = sorted(p.name for p in d.glob(".ready_denoise_pre_separation.*")) if d.exists() else []
        if n == 1:
            mark_ready(env, args, files, "none", "diarization", "b")       # the short file arrives first
        if n == 3:
            mark_ready(env, args, files, "none", "diarization", "a")
    clock.hook = hook
    schedule_stop(r, files, "denoise_pre_separation")
    assert fake.calls == []
    assert seen[1] == [] and seen[2] == [".ready_denoise_pre_separation.b"]
    assert sorted(p.name for p in d.glob(".ready_denoise_pre_separation.*")) == [
        ".ready_denoise_pre_separation.a", ".ready_denoise_pre_separation.b"]
    assert (d / ".done.denoise_pre_separation").is_file()


def test_per_file_all_noop_upstream_runs_files_at_once_small_first(env_sizes):
    env = env_sizes
    fake = FileFake()
    r, args, files = pf(env, fake, Clock(), "--music", "off", "--layout", "default", "--stop-after", "diarization",
                        "--wait-signal", "denoise", "--signal", "diarization", "--wait-timeout", "1")
    schedule_stop(r, files, "diarization")
    first = [(c[0], c[2]) for c in fake.calls][:2]
    assert first == [("none", ["b.wav"]), ("none", ["a.wav"])]


def test_per_file_resume_skips_finished_variants_and_reruns_unfinished_files(env_sizes):
    env = env_sizes
    tmp = env[0]
    r, args, files = pf(env, FileFake(), Clock(), "--layout", "default", "--stop-after", "separation",
                        "--signal", "separation")
    schedule_stop(r, files, "separation")
    fake2 = FileFake()
    r2, _, files2 = pf(env, fake2, Clock(), "--layout", "default", "--stop-after", "separation",
                       "--signal", "separation", "--resume")
    assert r2.runnable() == [] and fake2.calls == []
    # one variant lost its .done: it reruns all its files (checkpoints make that cheap), the others stay
    (tmp / "out" / "dfn3" / ".done.separation").unlink()
    fake3 = FileFake()
    r3, _, files3 = pf(env, fake3, Clock(), "--layout", "default", "--stop-after", "separation",
                       "--signal", "separation", "--resume")
    schedule_stop(r3, files3, "separation")
    assert [(c[0], c[2]) for c in fake3.calls] == [("dfn3", ["b.wav"]), ("dfn3", ["a.wav"])]
    assert (tmp / "out" / "dfn3" / ".ready_separation.a").is_file()


def test_per_file_manifest_merges_producer_and_consumer(env_sizes):
    env = env_sizes
    r, args, files = pf(env, FileFake(), Clock(), "--layout", "default", "--stop-after", "separation",
                        "--signal", "separation")
    schedule_stop(r, files, "separation")
    c, cargs, cfiles = pf(env, FileFake(), Clock(), "--layout", "default", "--stop-after", "asr",
                          "--wait-signal", "separation", "--wait-timeout", "1")
    schedule_stop(c, cfiles, "asr")
    m = {x["id"]: x for x in manifest(env[0])["variants"]}
    assert [x["id"] for x in manifest(env[0])["variants"]][:3] == ["none", "dfn3", "p3"]
    assert {m[v]["stop_after"] for v in ("none", "dfn3", "p3")} == {"asr"}


def test_dry_run_lists_noop_stages_and_per_file(env, capsys):
    tmp, audio, var = env
    assert rr.main(["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(tmp / "out"),
                    "--dry-run", "--music", "off", "--per-file", "--stop-after", "denoise_pre_separation", "--signal",
                    "denoise_pre_separation", "--wait-signal", "diarization"]) == 0
    out = capsys.readouterr().out
    assert "per-file: yes" in out
    assert "no-op stages: denoise, denoise_pre_separation, denoise_pre_asr   (this process only hands it over)" in out
    assert "waits for: diarization" in out


def _child_cuda(env_value, visible):
    """CUDA_VISIBLE_DEVICES a spawned child really gets when the parent owns `visible`."""
    import subprocess
    undo = rr.install_child_gpu_mapping(visible)
    try:
        env = {**__import__("os").environ}
        if env_value is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = env_value
        out = subprocess.run([sys.executable, "-c", "import os;print(os.environ.get('CUDA_VISIBLE_DEVICES'))"],
                             env=env, capture_output=True, text=True).stdout.strip()
    finally:
        undo()
    return out


def test_a_worker_asking_for_cuda_0_gets_the_card_its_parent_owns():
    assert _child_cuda("0", "1") == "1"            # parent sees card 1 only; its worker asks for index 0
    assert _child_cuda("0", "5") == "5"
    assert _child_cuda("0", "0,1") == "0" and _child_cuda("1", "0,1") == "1"   # a parent that sees both: unchanged
    assert _child_cuda("1", "1") == "1"            # inherited value of the parent itself stays
    assert _child_cuda("0,1", "1") == "0,1"        # lists are not touched
    assert _child_cuda(None, "1") == "None"        # no variable asked: nothing added


def test_without_a_visible_list_nothing_is_patched():
    import subprocess
    real = subprocess.Popen
    undo = rr.install_child_gpu_mapping("")
    assert subprocess.Popen is real
    undo()
    undo = rr.install_child_gpu_mapping("1")
    assert subprocess.Popen is not real
    undo()
    assert subprocess.Popen is real


class _FakeSeparation:
    def __init__(self):
        self.calls = []

    def prefetch_overlap_plan(self, *a, **k):
        self.calls.append(a)
        return "built"


def _run_scheduler(env, stop_after, sep):
    r, args, files = runner_for(env, FakeRun(), "--stop-after", stop_after)
    pipeline = SimpleNamespace(separation_svc=sep)
    base = SimpleNamespace(env="a100", stop_after=stop_after, keep_models=True, step_asr=True,
                           denoise="none", job_id="exp-resident", save_path="x")
    config = {"environments": {"a100": {"steps": {"asr": True, "diarization": False}}}}
    msgs = []
    r.say = msgs.append
    r.scheduler(pipeline, base, config, [str(f) for f in files])
    return msgs


def test_window_prefetch_is_off_where_separation_does_not_run(env):
    for stop in ("denoise", "diarization", "denoise_pre_separation"):
        sep = _FakeSeparation()
        msgs = _run_scheduler(env, stop, sep)
        assert sep.prefetch_overlap_plan("segments") is None and sep.calls == [], stop
        assert any("prefetch is off" in m for m in msgs)


def test_window_prefetch_stays_on_when_this_process_runs_separation(env):
    for stop in ("separation", "asr"):
        sep = _FakeSeparation()
        _run_scheduler(env, stop, sep)
        assert sep.prefetch_overlap_plan("segments") == "built" and sep.calls == [("segments",)], stop


# ------------------------------------------- separation_plan, conversation_selection, --stop-after all, claims
def test_line_and_stage_order_have_the_new_stages_in_place():
    order = rr.STAGE_ORDER
    assert order.index("denoise_pre_separation") < order.index("separation_plan") < order.index("separation")
    assert order.index("asr") < order.index("conversation_selection") < order.index("word_alignment")
    assert rr.LINE == ("denoise", "diarization", "denoise_pre_separation", "separation_plan", "separation",
                       "denoise_pre_asr", "asr", "conversation_selection")
    assert [s for s in rr.LINE] == sorted(rr.LINE, key=order.index)
    none = {"chain": [], "chain_pre_separation": [], "chain_pre_asr": [], "set": {}}
    assert not rr.is_noop(none, "separation_plan") and not rr.is_noop(none, "conversation_selection")
    assert rr.effective_stage(none, "separation") == "separation"
    assert "separation_plan" in rr.STOPS_BEFORE_REFINEMENT


def test_stop_after_all_gives_main_no_stop_after(env):
    args, variants, files, skipped = make(env, "--layout", "default", "--stop-after", "all")
    argv = rr.build_argv(args, env[0] / "in")
    assert "--stop_after" not in argv and "--job_id" in argv
    assert all(rr.variant_overrides(v, args.stop_after)["stop_after"] is None for v in variants)
    args2, _, _, _ = make(env, "--layout", "default", "--stop-after", "conversation_selection")
    argv2 = rr.build_argv(args2, env[0] / "in")
    assert argv2[argv2.index("--stop_after") + 1] == "conversation_selection"


def test_stop_after_new_values_and_wait_rules(env):
    base = ["--audio-dir", str(env[1]), "--variants", str(env[2]), "--out-root", str(env[0] / "out")]
    assert rr.parse_args(base + ["--stop-after", "separation_plan", "--signal", "separation_plan"]).stop_after \
        == "separation_plan"
    ok = rr.parse_args(base + ["--stop-after", "all", "--wait-signal", "conversation_selection"])
    assert ok.stop_after == "all" and ok.wait_signal == "conversation_selection"
    with pytest.raises(SystemExit):
        rr.parse_args(base + ["--stop-after", "asr", "--wait-signal", "conversation_selection"])
    with pytest.raises(SystemExit):
        rr.parse_args(base + ["--stop-after", "not_a_stage"])


def test_claim_needs_per_file(env):
    base = ["--audio-dir", str(env[1]), "--variants", str(env[2]), "--out-root", str(env[0] / "out")]
    with pytest.raises(SystemExit):
        rr.parse_args(base + ["--claim"])
    args = rr.parse_args(base + ["--per-file", "--claim", "--reclaim-after", "30"])
    assert args.claim and args.reclaim_after == 30


def claim_copy(env, fake, clock, *extra, msgs=None, sleep=None, now=None):
    """One copy of the stage `separation` (per-file, --claim): its own runner, same out-root."""
    args, variants, files, skipped = make(env, "--layout", "default", "--stop-after", "separation",
                                          "--signal", "separation", "--per-file", "--claim",
                                          "--only-ids", "dfn3,p3", *extra)
    say = msgs.append if msgs is not None else (lambda m: None)
    r = rr.ResidentRunner(args, variants, files, run_batch=fake, skipped=skipped, say=say,
                          sleep=sleep or clock.sleep, clock=now or clock.time)
    r.prepare()
    return r, args, files


def other_copy_finishes(env, args, files, stem):
    """What the other copy leaves for dfn3 / one file: its denoise.json and its ready marker."""
    d = env[0] / "out" / "dfn3" / stem / "01b_denoise"
    d.mkdir(parents=True, exist_ok=True)
    (d / "denoise.json").write_text(json.dumps({"chain_id": "dfn3", "errors": [], "corr_with_input": 0.9,
                                                "lag_samples": 0}))
    mark_ready(env, args, files, "dfn3", "separation", stem)


def write_claim(env, vid, stem, age_s=0.0, now=None):
    p = env[0] / "out" / vid / f".claim_separation.{stem}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"host": "job-a", "pid": 7, "time": "x"}))
    if now is not None:
        import os as _os
        _os.utime(p, (now - age_s, now - age_s))
    return p


def test_copy_waits_for_a_file_another_copy_holds_and_adopts_its_result(env):
    import time as _time
    clock = Clock()
    clock.now = _time.time()
    write_claim(env, "dfn3", "a", now=clock.now, age_s=0)     # the other copy holds it: no result yet
    fake, msgs = FakeRun(), []
    r, args, files = claim_copy(env, fake, clock, msgs=msgs)
    clock.hook = lambda n: other_copy_finishes(env, args, files, "a") if n == 3 else None
    schedule_stop(r, files, "separation")
    calls = [(vid, names) for vid, _, names in fake.calls]
    assert ("dfn3", ["a.wav"]) not in calls and ("dfn3", ["b.wav"]) in calls
    assert "dfn3" in r.finalized
    assert any("by another copy" in m for m in msgs)
    assert not (env[0] / "out" / "dfn3" / ".failed_separation.a").exists()


def test_stale_claim_is_taken_over_only_with_reclaim_after(env):
    import socket
    import time as _time
    clock = Clock()
    clock.now = _time.time()
    write_claim(env, "dfn3", "a", now=clock.now, age_s=3600)     # an hour old, no result: its copy died
    fake, msgs = FakeRun(), []
    r, args, files = claim_copy(env, fake, clock, "--reclaim-after", "30", msgs=msgs)
    schedule_stop(r, files, "separation")
    assert ("dfn3", ["a.wav"]) in [(vid, names) for vid, _, names in fake.calls]
    assert any("taking it over" in m for m in msgs)
    assert json.loads((env[0] / "out" / "dfn3" / ".claim_separation.a").read_text())["host"] == socket.gethostname()


def test_without_reclaim_after_a_dead_claim_is_never_taken(env):
    import time as _time
    clock = Clock()
    clock.now = _time.time()
    write_claim(env, "dfn3", "a", now=clock.now, age_s=10 ** 6)
    fake, msgs = FakeRun(), []
    r, args, files = claim_copy(env, fake, clock, "--wait-timeout", "2", msgs=msgs)
    schedule_stop(r, files, "separation")
    assert ("dfn3", ["a.wav"]) not in [(vid, names) for vid, _, names in fake.calls]
    assert any("gave up waiting" in m for m in msgs)
    out = env[0] / "out" / "dfn3"
    assert not (out / ".failed_separation.a").exists() and not (out / ".ready_separation.a").exists()


def test_total_marker_is_written_only_when_every_file_is_ready(env):
    import time as _time
    clock = Clock()
    clock.now = _time.time()
    write_claim(env, "dfn3", "b", now=clock.now, age_s=0)
    seen_at_first_wait = []
    fake = FakeRun()
    r, args, files = claim_copy(env, fake, clock)

    def hook(n):
        if n == 1:
            seen_at_first_wait.append((env[0] / "out" / "dfn3" / ".ready_separation").exists())
        if n == 3:
            other_copy_finishes(env, args, files, "b")
    clock.hook = hook
    schedule_stop(r, files, "separation")
    assert seen_at_first_wait == [False]
    total = json.loads((env[0] / "out" / "dfn3" / ".ready_separation").read_text())
    assert total["hash"] == rr.variant_hash(
        {v["id"]: v for v in rr.resolve_plan(args)[0]}["dfn3"], args, files, stop_after="separation")


def test_a_file_that_failed_in_one_copy_is_not_rerun_by_the_other(env):
    # both copies started before the failure: it belongs to this run, so the other copy adopts it
    clock = Clock()
    fa = FileFake(bad={("dfn3", "a")})
    ra, _, files = claim_copy(env, fa, clock)
    fb = FakeRun()
    rb, _, files = claim_copy(env, fb, Clock())
    schedule_stop(ra, files, "separation")
    assert (env[0] / "out" / "dfn3" / ".failed_separation.a").is_file()
    schedule_stop(rb, files, "separation")
    assert ("dfn3", ["a.wav"]) not in [(vid, names) for vid, _, names in fb.calls]
    assert rb.status["dfn3"] == "failed"


def test_two_copies_of_one_stage_share_the_files_and_each_file_runs_once(env):
    import threading
    import time as _time

    class Slow(FakeRun):
        def __call__(self, pipeline, args, config, batch, logger=None, stages=None):
            _time.sleep(0.02)                 # the copies overlap
            return super().__call__(pipeline, args, config, batch, logger, stages)

    fake = Slow()
    copies = [claim_copy(env, fake, None, sleep=lambda s: _time.sleep(0.003), now=_time.time)
              for _ in range(2)]
    expected = {(v["id"], n) for v in copies[0][0].runnable() for n in ("a.wav", "b.wav")}
    threads = [threading.Thread(target=schedule_stop, args=(r, f, "separation")) for r, _, f in copies]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    runs = [(vid, n) for vid, _, names in fake.calls for n in names]
    assert sorted(runs) == sorted(expected)            # every file of every variant exactly once
    out = env[0] / "out"
    for vid in {v for v, _ in expected}:
        assert (out / vid / ".ready_separation").is_file()
        assert all((out / vid / f".ready_separation.{s}").is_file() for s in ("a", "b"))
        assert (out / vid / ".claim_separation.a").is_file()



# ------------------------------------------------------------------ claims that recover, degraded files, frozen list
def dead_pid():
    """The pid of a process that has exited: no process has it now."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def write_claim_as(env, vid, stem, host, pid):
    p = env[0] / "out" / vid / f".claim_separation.{stem}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"host": host, "pid": pid, "time": "x"}))
    return p


def test_dead_copy_claim_on_this_host_is_removed_at_start(env):
    import socket
    fake, msgs = FakeRun(), []
    write_claim_as(env, "dfn3", "a", socket.gethostname(), dead_pid())   # reclaim-after 0: only this rule removes it
    r, args, files = claim_copy(env, fake, Clock(), msgs=msgs)
    schedule_stop(r, files, "separation")
    assert ("dfn3", ["a.wav"]) in [(vid, names) for vid, _, names in fake.calls]
    assert any("dead copy on this host removed" in m for m in msgs)
    assert json.loads((env[0] / "out" / "dfn3" / ".claim_separation.a").read_text())["host"] == socket.gethostname()


def test_other_hosts_and_live_claims_are_kept_at_start(env):
    import socket
    write_claim_as(env, "dfn3", "a", "job-a", dead_pid())            # another host: only --reclaim-after takes it
    write_claim_as(env, "dfn3", "b", socket.gethostname(), os.getpid())   # this host, process alive
    fake = FakeRun()
    r, args, files = claim_copy(env, fake, Clock(), "--wait-timeout", "1")
    schedule_stop(r, files, "separation")
    calls = [(vid, names) for vid, _, names in fake.calls]
    assert ("dfn3", ["a.wav"]) not in calls and ("dfn3", ["b.wav"]) not in calls
    assert (env[0] / "out" / "dfn3" / ".claim_separation.a").is_file()
    assert (env[0] / "out" / "dfn3" / ".claim_separation.b").is_file()


def test_dead_local_claim_is_taken_during_the_run_without_reclaim_after(env):
    import socket
    r, args, files = claim_copy(env, FakeRun(), Clock())               # prepare has nothing to clean yet
    path = write_claim_as(env, "dfn3", "a", socket.gethostname(), dead_pid())
    assert r._take_claim("dfn3", "a", lambda m: None) is True
    assert json.loads(path.read_text())["pid"] == os.getpid()


def test_heartbeat_stamps_the_claim_while_the_file_runs(env):
    import threading
    import time as _time
    seen = []

    class Stamp(FakeRun):
        def __call__(self, pipeline, args, config, batch, logger=None, stages=None):
            if args.job_id == "exp-dfn3" and Path(batch[0]).stem == "a":
                claim = env[0] / "out" / "dfn3" / ".claim_separation.a"
                old = _time.time() - 3600              # as if nobody stamped it for an hour
                os.utime(claim, (old, old))
                _time.sleep(0.3)                        # the heartbeat runs meanwhile
                seen.append(claim.stat().st_mtime - old)
            return super().__call__(pipeline, args, config, batch, logger, stages)

    r, args, files = claim_copy(env, Stamp(), Clock())
    r.heartbeat_seconds = 0.02
    schedule_stop(r, files, "separation")
    assert seen and seen[0] > 1800
    assert not [t for t in threading.enumerate() if t.name == "claim-heartbeat"]


def test_claim_steal_by_rename_has_one_winner(env, monkeypatch):
    import time as _time
    clock = Clock()
    clock.now = _time.time()
    real_rename = os.rename
    r, args, files = claim_copy(env, FakeRun(), clock, "--reclaim-after", "30")
    path = write_claim(env, "dfn3", "a", now=clock.now, age_s=3600)
    # another copy renamed the stale claim first: our rename finds nothing
    def gone(src, dst):
        raise FileNotFoundError(src)
    monkeypatch.setattr(os, "rename", gone)
    assert r._take_claim("dfn3", "a", lambda m: None) is False
    monkeypatch.setattr(os, "rename", real_rename)
    assert path.is_file()

    def fresh_first(src, dst):
        # between our look at the claim and our rename, another copy stole it and created a fresh one
        Path(src).write_text(json.dumps({"host": "job-a", "pid": 9, "time": "x"}))
        os.utime(src, (clock.now, clock.now))
        return real_rename(src, dst)
    write_claim(env, "dfn3", "a", now=clock.now, age_s=3600)
    monkeypatch.setattr(os, "rename", fresh_first)
    assert r._take_claim("dfn3", "a", lambda m: None) is False
    monkeypatch.setattr(os, "rename", real_rename)
    assert json.loads(path.read_text())["host"] == "job-a"            # the fresh claim is put back
    assert not [p for p in path.parent.iterdir() if ".stale." in p.name]


def test_failed_marker_of_an_earlier_run_does_not_stop_the_file(env):
    import time as _time
    fake, msgs = FakeRun(), []
    r, args, files = claim_copy(env, fake, Clock(), msgs=msgs)
    variants = {v["id"]: v for v in rr.resolve_plan(args)[0]}
    d = env[0] / "out" / "dfn3"
    d.mkdir(parents=True, exist_ok=True)
    failed = d / ".failed_separation.a"
    failed.write_text(json.dumps({"reason": "sidon: boom", "hash": rr.variant_hash(
        variants["dfn3"], args, files, stop_after="separation")}))
    old = r.started_at - 3600                                        # written by an earlier run
    os.utime(failed, (old, old))
    write_claim(env, "dfn3", "a", now=r.started_at, age_s=3600)
    schedule_stop(r, files, "separation")
    assert ("dfn3", ["a.wav"]) in [(vid, names) for vid, _, names in fake.calls]
    assert any("failed marker of an earlier run" in m for m in msgs)
    assert not failed.exists() and (d / ".ready_separation.a").is_file()


class StemErrors(FakeRun):
    """Reports an error in denoise.json for the (variant, file stem) pairs given: the filter fell back."""

    def __init__(self, bad):
        super().__init__()
        self.bad = set(bad)

    def __call__(self, pipeline, args, config, batch, logger=None, stages=None):
        out = super().__call__(pipeline, args, config, batch, logger, stages)
        vid = args.job_id[len("exp-"):]
        for p in batch:
            if (vid, Path(p).stem) in self.bad:
                f = Path(args.save_path) / Path(p).stem / "01b_denoise" / "denoise.json"
                info = json.loads(f.read_text())
                info["errors"] = ["rf_crowd: boom"]
                f.write_text(json.dumps(info))
        return out


@pytest.mark.parametrize("claim", [False, True])
def test_degraded_file_is_failed_and_its_ok_neighbour_stays_ready(env, claim):
    fake, clock = StemErrors({("dfn3", "a")}), Clock()
    if claim:
        r, args, files = claim_copy(env, fake, clock)
    else:
        r, args, files = line_runner(env, fake, clock, "--layout", "default", "--stop-after", "separation",
                                     "--signal", "separation", "--per-file", "--only-ids", "dfn3")
    schedule_stop(r, files, "separation")
    d = env[0] / "out" / "dfn3"
    failed = json.loads((d / ".failed_separation.a").read_text())
    assert "degraded" in failed["reason"] and "rf_crowd: boom" in failed["reason"]
    assert not (d / ".ready_separation.a").exists()
    assert (d / ".ready_separation.b").is_file()
    assert r.status["dfn3"] == "degraded"
    assert not (d / ".done.separation").exists()


def test_files_added_later_keep_the_hashes_and_markers_of_the_line(env, capsys):
    tmp, audio, _ = env
    r1, args1, files1 = producer_then_files(env)
    frozen = json.loads((tmp / "out" / "_resident" / "files.json").read_text(encoding="utf-8"))
    assert [e["name"] for e in frozen["files"]] == ["a.wav", "b.wav"]
    assert {"size", "mtime", "mtime_ns", "path"} <= set(frozen["files"][0])
    (audio / "c.wav").write_bytes(b"added after the first copy started")
    capsys.readouterr()
    r2, args2, files2 = line_runner(env, FakeRun(), None, "--layout", "default", "--stop-after", "asr",
                                    "--wait-signal", "separation")
    assert "audio-dir đã đổi" in capsys.readouterr().out
    assert [f.name for f in files2] == ["a.wav", "b.wav"]
    v = {x["id"]: x for x in r2.variants}["dfn3"]
    assert rr.variant_hash(v, args2, files2, stop_after="separation") == \
        rr.variant_hash({x["id"]: x for x in r1.variants}["dfn3"], args1, files1, stop_after="separation")
    assert r2._marker_state(v, "separation") == ("ready", None)


def test_refresh_files_rewrites_the_list_and_changes_the_hashes(env):
    tmp, audio, _ = env
    r1, args1, files1 = producer_then_files(env)
    v = {x["id"]: x for x in r1.variants}["dfn3"]
    before = rr.variant_hash(v, args1, files1, stop_after="separation")
    (audio / "c.wav").write_bytes(b"added")
    args2, _, files2, _ = make(env, "--refresh-files")
    assert [f.name for f in files2] == ["a.wav", "b.wav", "c.wav"]
    frozen = json.loads((tmp / "out" / "_resident" / "files.json").read_text(encoding="utf-8"))
    assert len(frozen["files"]) == 3
    assert rr.variant_hash(v, args2, files2, stop_after="separation") != before


def test_frozen_file_removed_from_the_audio_dir_is_an_error(env):
    _, audio, _ = env
    producer_then_files(env)
    (audio / "b.wav").unlink()
    with pytest.raises(SystemExit, match="no longer in the audio dir"):
        make(env)


def test_dry_run_does_not_freeze_the_list(env):
    tmp = env[0]
    make(env, "--dry-run")
    assert not (tmp / "out" / "_resident" / "files.json").exists()
