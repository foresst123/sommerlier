"""tools/launch_line.sh writes one script per process; DRY=1 starts nothing (no tmux, no conda, no GPU)."""
import os
import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "launch_line.sh"


def launch(tmp_path, *extra, env=None, out="out"):
    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    e = {**os.environ, "DRY": "1", **(env or {})}
    r = subprocess.run(["bash", str(SCRIPT), str(audio), str(tmp_path / out), *extra],
                       capture_output=True, text=True, env=e)
    return r, tmp_path / out / "_resident" / "launch"


def command_of(script):
    return next(line for line in script.read_text().splitlines() if line.startswith("python run_variants_resident.py"))


def opt(cmd, flag):
    m = re.search(rf"{re.escape(flag)} (\S+)", cmd)
    return m.group(1) if m else None


def test_six_scripts_form_a_chain(tmp_path):
    r, d = launch(tmp_path)
    assert r.returncode == 0, r.stderr
    scripts = sorted(d.glob("*.sh"))
    assert [s.name for s in scripts] == ["1_p1_denoise1.sh", "2_p2_diarization.sh", "3_p3_denoise2.sh",
                                          "4_p4_separation.sh", "5_p5_denoise3.sh", "6_p6_asr.sh"]
    cmds = [command_of(s) for s in scripts]
    assert opt(cmds[0], "--wait-signal") is None and opt(cmds[5], "--signal") is None
    for prev, cur in zip(cmds, cmds[1:]):
        assert opt(cur, "--wait-signal") == opt(prev, "--signal")          # each waits for the one before it
    for c in cmds:
        assert "--skip-bootstrap" in c and "--resume" in c and "--layout default" in c
    assert [opt(c, "--stop-after") for c in cmds] == ["denoise", "diarization", "denoise_pre_separation",
                                                       "separation", "denoise_pre_asr", "asr"]


def test_gpus_and_sidon_workers_are_knobs(tmp_path):
    r, d = launch(tmp_path, env={"GPUS": "0 1 0 1 0 1", "SIDON_WORKERS": "1"})
    gpus = [re.search(r"^_slot=(\d)", s.read_text(), re.M).group(1) for s in sorted(d.glob("*.sh"))]
    assert gpus == ["0", "1", "0", "1", "0", "1"]
    assert "workers_per_gpu=1" in command_of(d / "4_p4_separation.sh")
    r, d = launch(tmp_path, env={"GPUS": "0 1"}, out="out2")
    assert r.returncode == 2 and "6 numbers" in r.stderr


def test_extra_args_reach_every_process_and_paths_are_quoted(tmp_path):
    r, d = launch(tmp_path, "--accept-degraded", "--wait-timeout", "240", out="out with space")
    assert r.returncode == 0, r.stderr
    for s in d.glob("*.sh"):
        assert "--accept-degraded" in command_of(s) and "--wait-timeout 240" in command_of(s)
        assert "out\\ with\\ space" in command_of(s)
        assert subprocess.run(["bash", "-n", str(s)]).returncode == 0       # valid shell


def test_an_out_dir_made_with_music_on_is_refused(tmp_path):
    out = tmp_path / "old"
    out.mkdir()
    (out / "variants_manifest.json").write_text("{}")
    r, _ = launch(tmp_path, out="old")
    assert r.returncode == 3 and "new OUT" in r.stderr                      # default is music off, none made it
    r, _ = launch(tmp_path, env={"FORCE": "1"}, out="old")
    assert r.returncode == 0
    r, _ = launch(tmp_path, out="old")                                       # now the off marker exists
    assert r.returncode == 0
    r, _ = launch(tmp_path, "--music", "profile", out="old")                 # but a profile run may not reuse it
    assert r.returncode == 3


def test_usage_without_arguments():
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 2 and "launch_line.sh AUDIO_DIR OUT_DIR" in r.stdout


def test_jobid_wraps_every_window_in_srun(tmp_path):
    r, d = launch(tmp_path, env={"JOBID": "18790"})
    assert r.returncode == 0, r.stderr
    windows = [l for l in r.stdout.splitlines() if l.startswith("== window")]
    assert len(windows) == 6
    for w in windows:
        assert "srun --jobid=18790 --overlap --pty bash " in w and w.endswith("; exec bash")
    r, d = launch(tmp_path, out="plain")
    assert all("srun" not in l for l in r.stdout.splitlines() if l.startswith("== window"))


def test_jobid_must_be_a_number(tmp_path):
    r, _ = launch(tmp_path, env={"JOBID": "abc; rm"}, out="bad")
    assert r.returncode == 2 and "number" in r.stderr


def test_a_shell_without_gpu_is_refused_unless_told_otherwise(tmp_path):
    audio = tmp_path / "audio"; audio.mkdir()
    fake = tmp_path / "bin"; fake.mkdir()
    (fake / "nvidia-smi").write_text("#!/bin/sh\necho 'No devices were found'; exit 9\n")
    (fake / "nvidia-smi").chmod(0o755)
    (fake / "tmux").write_text("#!/bin/sh\nexit 1\n"); (fake / "tmux").chmod(0o755)   # no session, never started
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}
    r = subprocess.run(["bash", str(SCRIPT), str(audio), str(tmp_path / "o")], capture_output=True, text=True, env=env)
    assert r.returncode == 6 and "JOBID=" in r.stderr


def _gpu_snippet(script_text):
    lines = script_text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("_slot="))
    return "\n".join(lines[start:start + 3])          # _slot, _job, the if


def _chosen(script, cvd):
    code = _gpu_snippet(script.read_text()) + '\necho "$CUDA_VISIBLE_DEVICES"'
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
    return subprocess.run(["bash", "-c", code], capture_output=True, text=True, env=env).stdout.strip()


def test_gpu_slot_is_a_position_in_the_jobs_own_gpu_list(tmp_path):
    r, d = launch(tmp_path, env={"GPUS": "0 1 0 1 1 0"})
    scripts = sorted(d.glob("*.sh"))
    # the job was given the physical cards 4 and 5: slot 0 is card 4, slot 1 is card 5, never card 0 or 1
    assert [_chosen(s, "4,5") for s in scripts] == ["4", "5", "4", "5", "5", "4"]
    # a job with one card: every slot lands on it
    assert [_chosen(s, "6") for s in scripts] == ["6"] * 6
    # no list from Slurm (run by hand): the slot is the device number
    assert [_chosen(s, None) for s in scripts] == ["0", "1", "0", "1", "1", "0"]


def _fake_bin(tmp_path, **tools):
    d = tmp_path / "bin"; d.mkdir(exist_ok=True)
    for name, body in tools.items():
        (d / name).write_text(f"#!/bin/sh\n{body}\n"); (d / name).chmod(0o755)
    return {**os.environ, "PATH": f"{d}:{os.environ['PATH']}"}


def test_on_a_slurm_machine_a_job_id_is_required_even_if_gpus_are_visible(tmp_path):
    audio = tmp_path / "audio"; audio.mkdir()
    env = _fake_bin(tmp_path, srun="exit 0", **{"nvidia-smi": "echo GPU 0: A100; exit 0"}, tmux="exit 1")
    env.pop("SLURM_JOB_ID", None)
    r = subprocess.run(["bash", str(SCRIPT), str(audio), str(tmp_path / "o")], capture_output=True, text=True, env=env)
    assert r.returncode == 6 and "JOBID=<slurm job id>" in r.stderr
    r = subprocess.run(["bash", str(SCRIPT), str(audio), str(tmp_path / "o2")], capture_output=True, text=True,
                       env={**env, "NO_SRUN": "1"})
    assert r.returncode != 6                      # explicit opt-out goes on (then fails later: no real tmux here)


def test_job_id_defaults_to_the_slurm_job_of_the_shell(tmp_path):
    r, _ = launch(tmp_path, env={"SLURM_JOB_ID": "777"})
    windows = [l for l in r.stdout.splitlines() if l.startswith("== window")]
    assert windows and all("srun --jobid=777 --overlap --pty bash " in w for w in windows)
    r, _ = launch(tmp_path, env={"SLURM_JOB_ID": "777", "JOBID": "18790"}, out="o2")
    assert all("--jobid=18790 " in l for l in r.stdout.splitlines() if l.startswith("== window"))


def test_every_handover_of_the_line_is_recognised_by_the_next_process(tmp_path):
    """The hash a process writes in .ready_<stage> must equal the one the next process recomputes."""
    import importlib.util
    import shlex
    spec = importlib.util.spec_from_file_location("rr", SCRIPT.parents[1] / "run_variants_resident.py")
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    (tmp_path / "audio").mkdir(exist_ok=True)
    (tmp_path / "audio" / "a.wav").write_bytes(b"x")
    r, d = launch(tmp_path, env={"JOBID": "1"})
    procs = []
    for script in sorted(d.glob("*.sh")):
        args = rr.parse_args(shlex.split(command_of(script))[2:])
        procs.append((script.name, args) + tuple(rr.resolve_plan(args)[:2]))
    stages = ["denoise", "diarization", "denoise_pre_separation", "separation", "denoise_pre_asr"]
    for (pn, pa, pv, pf), (cn, ca, cv, cf), stage in zip(procs, procs[1:], stages):
        for vp, vc in zip(pv, cv):
            assert rr.variant_hash(vp, pa, pf, stop_after=stage) == rr.variant_hash(vc, ca, cf, stop_after=stage), \
                f"{pn} -> {cn} would not see {vp['id']} as ready"
