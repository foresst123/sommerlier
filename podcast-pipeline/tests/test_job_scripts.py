"""tools/job_front.sh, job_asr.sh, job_llm.sh write one script per process; DRY=1 starts nothing.

The GPU line of each process is run for real (bash, with a fake CUDA_VISIBLE_DEVICES) to check
which card it sees, since that value comes from the job at run time.
"""
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
COMMON = TOOLS / "job_common.sh"
KNOBS = ("CUDA_VISIBLE_DEVICES", "JOBID", "SLURM_JOB_ID", "GPUS", "REPLICAS_PER_GPU", "PLAN_REPLICAS",
         "SIDON_WORKERS", "DIAR_WORKERS", "AUDIO", "OUT", "CACHE", "FORCE", "SESSION",
         "FINAL", "RECLAIM_AFTER", "WAIT_TIMEOUT")


def run_job(job, tmp_path, *, env=None, out="out", audio=True):
    e = {k: v for k, v in os.environ.items() if k not in KNOBS}
    e.update({"DRY": "1", "NO_SRUN": "1"})
    if audio:
        (tmp_path / "audio").mkdir(exist_ok=True)
        e["AUDIO"] = str(tmp_path / "audio")
    e["OUT"] = str(tmp_path / out)
    e.update(env or {})
    r = subprocess.run(["bash", str(TOOLS / f"job_{job}.sh")], capture_output=True, text=True, env=e)
    return r, tmp_path / out / "_resident" / f"launch_{job}"


def scripts_of(launch):
    """{process name: script path}, names as the job script gave them (no index, no .sh)."""
    return {re.sub(r"^\d+_|\.sh$", "", p.name): p for p in sorted(launch.glob("*.sh"))}


def command_of(script):
    return next(line for line in script.read_text().splitlines() if line.startswith("python run_variants_resident.py"))


def opt(cmd, flag):
    m = re.search(rf"{re.escape(flag)} (\S+)", cmd)
    return m.group(1) if m else None


def gpu_line_of(script):
    return next(line for line in script.read_text().splitlines()
                if line.startswith(("job_set_gpus", "export CUDA_VISIBLE_DEVICES=")))


def cvd_of(script, cvd=None):
    """CUDA_VISIBLE_DEVICES the process would get: its GPU line run with the job's value `cvd` (None = unset)."""
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
    line = gpu_line_of(script)
    r = subprocess.run(["bash", "-c", f'source {COMMON}; {line}; printf %s "$CUDA_VISIBLE_DEVICES"'],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout.rsplit("\n", 1)[-1]          # job_set_gpus echoes a [gpu] line before the value


def clean_sets(cmd):
    """The `outputs.clean_two_channel_dir=...` words of a command, in order (shell-unquoted)."""
    return [w for w in shlex.split(cmd) if w.startswith("outputs.clean_two_channel_dir=")]


def stage_of(name):
    return re.sub(r"_(gpu\d+|cpu\d+)(_r\d+)?$", "", name)


# -- JOB 1: front ---------------------------------------------------------------------

def test_front_gpu_stages_have_one_copy_per_card(tmp_path):
    r, d = run_job("front", tmp_path)
    assert r.returncode == 0, r.stderr
    scripts = scripts_of(d)
    for stage in ("denoise", "diarization", "separation", "all"):
        copies = [n for n in scripts if n.startswith(f"{stage}_gpu")]
        assert len(copies) == 2, copies
        cmds = [command_of(scripts[n]) for n in copies]
        assert all("--claim" in c for c in cmds)
        assert all(f"--stop-after {stage} " in c for c in cmds)
        values = sorted(cvd_of(scripts[n], cvd="4,5") for n in copies)
        assert values == ["4", "5"]                       # each copy sees exactly one card, the other one
        assert {opt(c, "--gpu-1") for c in cmds} == {"0"}  # a single visible card: both gpu slots are 0


def test_front_separation_plan_is_cpu_only(tmp_path):
    r, d = run_job("front", tmp_path)
    scripts = scripts_of(d)
    plan = [n for n in scripts if n.startswith("separation_plan_cpu")]
    assert plan == ["separation_plan_cpu1"]
    script = scripts[plan[0]]
    assert gpu_line_of(script) == 'export CUDA_VISIBLE_DEVICES=""'
    assert cvd_of(script, cvd="4,5") == ""
    cmd = command_of(script)
    assert "--gpu-1" not in cmd and "--claim" not in cmd
    assert "--stop-after separation_plan" in cmd and "--wait-signal diarization" in cmd


def test_front_signal_chain(tmp_path):
    r, d = run_job("front", tmp_path)
    scripts = scripts_of(d)
    expected = {
        "denoise": (None, "denoise"),
        "diarization": ("denoise", "diarization"),
        "separation_plan": ("diarization", "separation_plan"),
        "separation": ("separation_plan", "separation"),
        "all": ("conversation_selection", None),
    }
    for name, path in scripts.items():
        wait, signal = expected[stage_of(name)]
        cmd = command_of(path)
        assert opt(cmd, "--wait-signal") == wait, name
        assert opt(cmd, "--signal") == signal, name
        assert f"--stop-after {stage_of(name)} " in cmd
    assert len(scripts) == 9                              # 2 + 2 + 1 + 2 + 2


def test_front_gpus_change_follows_the_card_positions(tmp_path):
    r, d = run_job("front", tmp_path, env={"GPUS": "1 0"}, out="o2")
    assert r.returncode == 0, r.stderr
    scripts = scripts_of(d)
    assert cvd_of(scripts["denoise_gpu1"], cvd="4,5") == "5"
    assert cvd_of(scripts["denoise_gpu0"], cvd="4,5") == "4"
    assert cvd_of(scripts["denoise_gpu1"]) == "1"         # outside a job: the position is the card id
    assert "job_set_gpus 1 || exit 7" in scripts["denoise_gpu1"].read_text()


def test_front_replicas_and_plan_copies(tmp_path):
    r, d = run_job("front", tmp_path, env={"REPLICAS_PER_GPU": "2", "PLAN_REPLICAS": "2"})
    assert r.returncode == 0, r.stderr
    scripts = scripts_of(d)
    denoise = [n for n in scripts if n.startswith("denoise_gpu")]
    assert len(denoise) == 4 and len(set(denoise)) == 4
    assert sorted(cvd_of(scripts[n], cvd="4,5") for n in denoise) == ["4", "4", "5", "5"]
    assert all("--claim" in command_of(scripts[n]) for n in denoise)
    plans = [n for n in scripts if n.startswith("separation_plan_cpu")]
    assert len(plans) == 2 and all("--claim" in command_of(scripts[n]) for n in plans)


def test_front_knobs_reach_the_commands(tmp_path):
    r, d = run_job("front", tmp_path, env={"SIDON_WORKERS": "1"})
    scripts = scripts_of(d)
    assert "performance.stages.separation.workers_per_gpu=1" in command_of(scripts["separation_gpu0"])
    assert "performance.stages.diarization.workers=1" in command_of(scripts["diarization_gpu1"])


def test_front_denoise_env_only_for_filter_stages(tmp_path):
    r, d = run_job("front", tmp_path)
    scripts = scripts_of(d)
    for name, path in scripts.items():
        text = path.read_text()
        wants = stage_of(name) in ("denoise", "all")
        assert ("source tools/denoise_env.sh" in text) == wants, name


def test_front_table_names_every_process(tmp_path):
    r, d = run_job("front", tmp_path)
    assert "denoise_gpu0" in r.stdout and "GPU 1" in r.stdout
    assert "separation_plan_cpu1" in r.stdout and "none (CPU)" in r.stdout
    assert "DRY=1" in r.stdout


# -- JOB 2 and 3: one process on both cards ----------------------------------------

def test_asr_job_is_one_process_on_both_cards(tmp_path):
    r, d = run_job("asr", tmp_path, env={"CUDA_VISIBLE_DEVICES": "4,5"}, out="a")
    assert r.returncode == 0, r.stderr
    scripts = scripts_of(d)
    assert list(scripts) == ["asr"]
    cmd = command_of(scripts["asr"])
    assert "--stop-after asr" in cmd and "--wait-signal separation" in cmd and "--signal asr" in cmd
    assert "--claim" not in cmd
    assert (opt(cmd, "--gpu-1"), opt(cmd, "--gpu-2")) == ("0", "1")
    assert cvd_of(scripts["asr"], cvd="4,5") == "4,5"


def test_asr_job_gpu_order_follows_gpus(tmp_path):
    r, d = run_job("asr", tmp_path, env={"GPUS": "1 0"}, out="a2")
    scripts = scripts_of(d)
    assert cvd_of(scripts["asr"], cvd="4,5") == "5,4"


def test_llm_job_waits_for_asr_and_signals_selection(tmp_path):
    r, d = run_job("llm", tmp_path, out="l")
    assert r.returncode == 0, r.stderr
    scripts = scripts_of(d)
    assert list(scripts) == ["conversation_selection"]
    cmd = command_of(scripts["conversation_selection"])
    assert "--stop-after conversation_selection" in cmd
    assert opt(cmd, "--wait-signal") == "asr" and opt(cmd, "--signal") == "conversation_selection"
    assert (opt(cmd, "--gpu-1"), opt(cmd, "--gpu-2")) == ("0", "1")
    assert cvd_of(scripts["conversation_selection"], cvd="6,7") == "6,7"


# -- checks ----------------------------------------------------------------------------

def test_every_generated_script_is_valid_shell(tmp_path):
    for job in ("front", "asr", "llm"):
        r, d = run_job(job, tmp_path, out=job)
        assert r.returncode == 0, r.stderr
        for s in d.glob("*.sh"):
            assert subprocess.run(["bash", "-n", str(s)]).returncode == 0


def test_no_claim_flag_for_single_copy_stage(tmp_path):
    r, d = run_job("front", tmp_path, env={"GPUS": "0 1", "REPLICAS_PER_GPU": "1", "PLAN_REPLICAS": "1"}, out="x")
    assert "--reclaim-after" not in r.stdout
    assert all("--reclaim-after" not in command_of(p) for p in d.glob("*.sh"))


def test_bad_gpus_and_missing_dirs_are_refused(tmp_path):
    r, _ = run_job("front", tmp_path, env={"GPUS": "0"}, out="b1")
    assert r.returncode == 2 and "two GPU ids" in r.stderr
    r, _ = run_job("asr", tmp_path, env={"GPUS": "0 1 2"}, out="b2")
    assert r.returncode == 2
    r, _ = run_job("llm", tmp_path, env={"GPUS": "a b"}, out="b3")
    assert r.returncode == 2 and "numbers" in r.stderr
    r, _ = run_job("front", tmp_path, out="b4", audio=False, env={"AUDIO": ""})
    assert r.returncode == 2 and "usage" in r.stderr


def test_jobs_drop_the_stage_wavs_unless_asked(tmp_path):
    import os, subprocess
    env = {**os.environ, "DRY": "1", "NO_SRUN": "1", "AUDIO": str(tmp_path), "OUT": str(tmp_path / "o")}
    for script in ("job_front", "job_asr", "job_llm"):
        subprocess.run(["bash", f"tools/{script}.sh"], env=env, check=True, capture_output=True, text=True)
        text = "".join(p.read_text() for p in (tmp_path / "o" / "_resident").glob("launch_*/*.sh"))
        assert "--no-stage-audio" in text
    keep = {**env, "STAGE_AUDIO": "1", "OUT": str(tmp_path / "o2")}
    subprocess.run(["bash", "tools/job_asr.sh"], env=keep, check=True, capture_output=True, text=True)
    assert "--no-stage-audio" not in "".join(p.read_text() for p in (tmp_path / "o2" / "_resident").glob("launch_*/*.sh"))


def test_gpus_are_physical_ids_and_no_jobid_is_needed(tmp_path):
    import os, re, subprocess
    env = {k: v for k, v in os.environ.items() if k not in ("JOBID", "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", *KNOBS)}
    env.update({"AUDIO": str(tmp_path), "OUT": str(tmp_path / "o"), "GPUS": "4 5", "DRY": "1"})   # no NO_SRUN, no JOBID
    r = subprocess.run(["bash", "tools/job_front.sh"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "physical GPU ids" in r.stdout
    seen = set()
    for script in (tmp_path / "o" / "_resident").glob("launch_front/*.sh"):
        text = script.read_text()
        if "separation_plan" in script.name:
            continue
        assert "export JOB_GPU_MODE=physical" in text and "srun" not in text
        line = next(l for l in text.splitlines() if l.startswith("job_set_gpus"))
        out = subprocess.run(["bash", "-c", f'source tools/job_common.sh; export JOB_GPU_MODE=physical; '
                              f'export CUDA_VISIBLE_DEVICES=0,1; {line.split("||")[0]} >/dev/null; printf %s "$CUDA_VISIBLE_DEVICES"'],
                             capture_output=True, text=True).stdout
        seen.add(out)
    assert seen == {"4", "5"}          # one copy per card, the physical id, whatever CUDA_VISIBLE_DEVICES said


def test_start_runs_nohup_processes_and_never_touches_tmux(tmp_path):
    import os, subprocess, time
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "tmux").write_text(f"#!/bin/sh\necho called > {tmp_path}/tmux_called\nexit 1\n")
    (fakebin / "tmux").chmod(0o755)
    for n in ("a", "b"):
        (tmp_path / f"{n}.sh").write_text(f"#!/bin/bash\necho {n} > {tmp_path}/{n}.ran\nsleep 30\n")
    logs = tmp_path / "logs"
    logs.mkdir()
    script = (f'source tools/job_common.sh; JOB_NAME=t LOG_DIR={logs} OUT={tmp_path} DRY=0 FORCE=0 JOBID= GPUS="0 1" '
              f'REPLICAS_PER_GPU=1; PROC_NAMES=(a b); PROC_GPU=(none none); PROC_STAGE=(s s); '
              f'PROC_FILES=({tmp_path}/a.sh {tmp_path}/b.sh); job_start')
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}"}
    try:
        r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        for _ in range(50):
            if (tmp_path / "a.ran").exists() and (tmp_path / "b.ran").exists():
                break
            time.sleep(0.1)
        assert (tmp_path / "a.ran").exists() and (tmp_path / "b.ran").exists()
        assert not (tmp_path / "tmux_called").exists() and "no tmux" in r.stdout
        pids = (logs / "t.pids").read_text().split("\n")
        assert [l.split()[1] for l in pids if l] == ["a", "b"]
        again = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)
        assert again.returncode == 4 and "still running" in again.stderr      # no second copy without FORCE
    finally:
        subprocess.run(["bash", "-c", f"source tools/job_common.sh; job_stop_pids {logs}/t.pids"], capture_output=True)
    time.sleep(0.3)
    for line in (logs / "t.pids").read_text().splitlines():
        state = subprocess.run(["ps", "-o", "stat=", "-p", line.split()[0]], capture_output=True, text=True).stdout.strip()
        assert state == "" or state.startswith("Z")                       # ended (a zombie nobody reaped counts)


# -- final root (FINAL) and the flags of the claim / wait --------------------------------

def test_final_root_is_set_on_the_all_processes_only(tmp_path):
    r, d = run_job("front", tmp_path)
    assert r.returncode == 0, r.stderr
    final = str((tmp_path / "out" / "final").resolve())
    for name, path in scripts_of(d).items():
        sets = clean_sets(command_of(path))
        if stage_of(name) == "all":
            assert sets == [f'outputs.clean_two_channel_dir="{final}"'], name
        else:
            assert sets == [], name


def test_final_knob_sets_the_root_and_keeps_spaces_in_one_word(tmp_path):
    final = tmp_path / "data disk" / "final set"
    r, d = run_job("front", tmp_path, env={"FINAL": str(final)})
    assert r.returncode == 0, r.stderr
    for name, path in scripts_of(d).items():
        if stage_of(name) == "all":
            assert clean_sets(command_of(path)) == [f'outputs.clean_two_channel_dir="{final}"'], name


def test_final_refuses_quotes_and_backslashes(tmp_path):
    r, _ = run_job("front", tmp_path, env={"FINAL": str(tmp_path / 'a"b')}, out="q1")
    assert r.returncode == 2 and "FINAL" in r.stderr
    r, _ = run_job("front", tmp_path, env={"FINAL": str(tmp_path / "a\\b")}, out="q2")
    assert r.returncode == 2 and "FINAL" in r.stderr


def test_default_commands_have_no_reclaim_or_wait_flags(tmp_path):
    r, d = run_job("front", tmp_path, out="dflt")
    assert r.returncode == 0, r.stderr
    for script in d.glob("*.sh"):
        cmd = command_of(script)
        assert "--reclaim-after" not in cmd and "--wait-timeout" not in cmd, script.name


def test_reclaim_and_wait_flags_go_to_the_right_processes(tmp_path):
    env = {"RECLAIM_AFTER": "30", "WAIT_TIMEOUT": "60", "PLAN_REPLICAS": "2"}
    r, d = run_job("front", tmp_path, env=env, out="rw")
    assert r.returncode == 0, r.stderr
    for name, path in scripts_of(d).items():
        cmd = command_of(path)
        assert opt(cmd, "--reclaim-after") == ("30" if "--claim" in cmd else None), name
        assert opt(cmd, "--wait-timeout") == ("60" if "--wait-signal" in cmd else None), name
    assert opt(command_of(scripts_of(d)["denoise_gpu0"]), "--wait-timeout") is None


def test_reclaim_and_wait_flags_on_the_other_jobs(tmp_path):
    env = {"RECLAIM_AFTER": "5", "WAIT_TIMEOUT": "0.5"}
    r, d = run_job("asr", tmp_path, env=env, out="rwa")
    assert r.returncode == 0, r.stderr
    cmd = command_of(scripts_of(d)["asr"])
    assert opt(cmd, "--wait-timeout") == "0.5" and "--reclaim-after" not in cmd
    r, d = run_job("llm", tmp_path, env=env, out="rwl")
    cmd = command_of(scripts_of(d)["conversation_selection"])
    assert opt(cmd, "--wait-timeout") == "0.5" and "--reclaim-after" not in cmd


def test_zero_and_bad_reclaim_or_wait_values(tmp_path):
    r, d = run_job("front", tmp_path, env={"RECLAIM_AFTER": "0", "WAIT_TIMEOUT": "0.0"}, out="z")
    assert r.returncode == 0, r.stderr
    assert all("--reclaim-after" not in command_of(p) and "--wait-timeout" not in command_of(p) for p in d.glob("*.sh"))
    for bad in ("-1", "abc", "1.2.3"):
        r, _ = run_job("llm", tmp_path, env={"WAIT_TIMEOUT": bad}, out=f"bad{len(bad)}")
        assert r.returncode == 2 and "RECLAIM_AFTER and WAIT_TIMEOUT" in r.stderr, bad


def test_run_variants_dry_run_takes_the_final_set_after_the_empty_one(tmp_path):
    """run_variants_resident.py puts "" first and the job's FINAL after it, so FINAL wins in main.py's argv."""
    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    (audio / "rec01.wav").write_bytes(b"")
    final = tmp_path / "final root"
    r, d = run_job("front", tmp_path, env={"FINAL": str(final)}, out="dr")
    assert r.returncode == 0, r.stderr
    args = shlex.split(command_of(scripts_of(d)["all_gpu0"]))[2:]   # drop "python run_variants_resident.py"
    dry = subprocess.run([sys.executable, "run_variants_resident.py", *args, "--dry-run"], cwd=TOOLS.parent,
                         capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert dry.stdout.index('outputs.clean_two_channel_dir=""') < dry.stdout.index(f'outputs.clean_two_channel_dir="{final}"')


def test_only_restarts_the_matching_processes(tmp_path):
    out = tmp_path / "out"
    logs = out / "_resident" / "logs"
    logs.mkdir(parents=True)
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "a.wav").write_bytes(b"x")
    keep = subprocess.Popen(["sleep", "30"], start_new_session=True)
    gone = subprocess.Popen(["sleep", "30"], start_new_session=True)
    (logs / "front.pids").write_text(f"{keep.pid} denoise_gpu0\n{gone.pid} all_gpu0\n")
    env = dict(os.environ, AUDIO=str(audio), OUT=str(out), GPUS="0 1", ONLY="^all_", CONDA_ENV="",
               AUDIO_SEPARATOR_MODEL_DIR=str(tmp_path))
    script = Path(__file__).resolve().parents[1] / "tools" / "job_front.sh"
    r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=60)
    try:
        assert "restarted 2 process(es)" in r.stdout, r.stdout + r.stderr
        lines = (logs / "front.pids").read_text().splitlines()
        assert any(l.endswith("denoise_gpu0") and l.startswith(str(keep.pid)) for l in lines)
        assert sum(" all_gpu" in l for l in lines) == 2
        assert keep.poll() is None            # not touched
        gone.wait(timeout=5)                   # stopped
    finally:
        for pid_line in (logs / "front.pids").read_text().splitlines():
            try:
                os.killpg(int(pid_line.split()[0]), 15)
            except (OSError, ValueError):
                pass
        keep.kill()
        gone.kill()
