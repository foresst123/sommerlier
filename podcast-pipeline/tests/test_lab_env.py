"""tools/lab_env.py: where each denoise backend's env / weights live. Run: python -m pytest tests/test_lab_env.py -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lab_env as le  # noqa: E402


def make(tmp_path, *rel):
    for r in rel:
        p = tmp_path / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.mkdir() if not Path(r).suffix and not r.endswith(("python", "ckpt")) else p.write_text("x")
    return tmp_path


def test_first_existing_candidate_wins_and_missing_ones_are_skipped(tmp_path):
    home = make(tmp_path, "NemotronLabs_REUSE", "miniforge3/envs/reuse/bin/python")
    out, report = le.resolve(environ={}, home=home)
    assert out["REUSE_DIR"] == str(home / "NemotronLabs_REUSE")
    assert out["REUSE_PYTHON"] == str(home / "miniforge3/envs/reuse/bin/python")
    assert any(line.startswith("REUSE_PYTHON=") and line.endswith("(file)") for line in report)


def test_unfound_variables_are_reported_not_exported(tmp_path):
    out, report = le.resolve(environ={}, home=tmp_path)
    assert "AUDIO_SEPARATOR_MODEL_DIR" not in out and "REUSE_DIR" not in out
    assert any(line.startswith("AUDIO_SEPARATOR_MODEL_DIR: KHÔNG TÌM THẤY") for line in report)
    assert not any(line.startswith("DFN_MODEL_DIR") for line in report)  # no candidates configured: nothing to complain about


def test_cli_beats_shell_beats_file(tmp_path):
    home = make(tmp_path, "NemotronLabs_REUSE")
    shell = {"REUSE_DIR": "/from/shell", "AUDIO_SEPARATOR_MODEL_DIR": "/shell/sep"}
    out, report = le.resolve({"REUSE_DIR": "/from/cli"}, environ=shell, home=home)
    assert out["REUSE_DIR"] == "/from/cli" and out["AUDIO_SEPARATOR_MODEL_DIR"] == "/shell/sep"
    assert "  (--set-env)" in [x for x in report if x.startswith("REUSE_DIR=")][0]
    out2, _ = le.resolve(environ={}, home=home)
    assert out2["REUSE_DIR"] == str(home / "NemotronLabs_REUSE")


def test_dollar_candidate_reuses_another_variable_only_when_it_points_somewhere(tmp_path):
    models = tmp_path / "bsr_models"
    models.mkdir()
    out, report = le.resolve(environ={"BS_ROFORMER_MODEL_DIR": str(models)}, home=tmp_path)
    assert out["AUDIO_SEPARATOR_MODEL_DIR"] == str(models) and "($BS_ROFORMER_MODEL_DIR)" in [x for x in report if x.startswith("AUDIO_SEPARATOR")][0]
    gone, _ = le.resolve(environ={"BS_ROFORMER_MODEL_DIR": str(tmp_path / "nope")}, home=tmp_path)
    assert "AUDIO_SEPARATOR_MODEL_DIR" not in gone


def test_names_this_file_does_not_know_still_pass_through_from_the_command_line(tmp_path):
    out, _ = le.resolve({"MY_OWN_VAR": "1"}, environ={}, home=tmp_path)
    assert out == {"MY_OWN_VAR": "1"}


def test_active_conda_is_searched_before_the_usual_install_places(tmp_path):
    home = make(tmp_path, "somewhere/envs/reuse/bin/python", "miniforge3/envs/reuse/bin/python")
    exe = home / "somewhere" / "bin" / "conda"
    exe.parent.mkdir(parents=True)
    exe.write_text("x")
    out, _ = le.resolve(environ={"CONDA_EXE": str(exe)}, home=home)
    assert out["REUSE_PYTHON"] == str(home / "somewhere/envs/reuse/bin/python")


def test_lab_server_exports_what_lab_env_resolved(tmp_path):
    import lab_server as ls
    home = make(tmp_path, "NemotronLabs_REUSE")
    resolved, _ = le.resolve(environ={}, home=home)
    cfg = ls.Config([tmp_path], tmp_path / "out", "1", "srun --jobid={jobid} --overlap", "som", "", "python", True, resolved)
    script = ls.launch_argv(cfg, [], Path("/x/v.json"), Path("/x/out"))[0][-1]
    assert f"export REUSE_DIR={home / 'NemotronLabs_REUSE'}" in script and script.index("export REUSE_DIR") < script.index("exec python")


def test_every_variable_the_denoise_backends_read_is_covered():
    """lab_env must know each env var the backends consult, so none of them silently skips a variant."""
    import re
    names = set()
    for f in (Path(__file__).resolve().parents[1] / "models" / "denoise").glob("*.py"):
        names |= set(re.findall(r"""(?:environ\.get\(|getenv\(|dir_env\s*=\s*|ckpt_env\s*=\s*)\s*["']([A-Z][A-Z0-9_]{3,})["']""", f.read_text(encoding="utf-8")))
    known = set(le.default_candidates(Path("/nonexistent"), {}))
    assert names - known - {"BS_ROFORMER_MODEL_DIR"} == set(), sorted(names - known)


def test_repo_clones_are_found_in_the_usual_parent_folders(tmp_path):
    home = make(tmp_path, "sommerlier/MP-SENet", "data/models/Dynamic-Band-Splitting", "models/NemotronLabs_REUSE")
    out, _ = le.resolve(environ={}, home=home)
    assert out["MPSENET_DIR"] == str(home / "sommerlier/MP-SENet")
    assert out["DBS_TIGER_DIR"] == str(home / "data/models/Dynamic-Band-Splitting")
    assert out["REUSE_DIR"] == str(home / "models/NemotronLabs_REUSE")


def test_the_documented_bsrnn_checkpoint_location_is_found(tmp_path):
    home = make(tmp_path, "urgent26_ckpt/bsrnn.ckpt")
    out, _ = le.resolve(environ={}, home=home)
    assert out["URGENT26_BSRNN_CKPT"] == str(home / "urgent26_ckpt/bsrnn.ckpt")   # scripts/denoise_bsrnn_urgent.md
