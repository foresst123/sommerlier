"""tools/job_front.sh, job_asr.sh, job_llm.sh write one script per process; DRY=1 starts nothing.

The GPU line of each process is run for real (bash, with a fake CUDA_VISIBLE_DEVICES) to check
which card it sees, since that value comes from the job at run time.
"""
import os
import re
import subprocess
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
COMMON = TOOLS / "job_common.sh"
KNOBS = ("CUDA_VISIBLE_DEVICES", "JOBID", "SLURM_JOB_ID", "GPUS", "REPLICAS_PER_GPU", "PLAN_REPLICAS",
         "SIDON_WORKERS", "DIAR_WORKERS", "AUDIO", "OUT", "CACHE", "FORCE", "SESSION")


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
    assert r.returncode == 2 and "two positions" in r.stderr
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
