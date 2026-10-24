import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lab_server as ls  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "a.wav").write_bytes(b"RIFF")
    (audio / "notes.txt").write_text("x")
    (audio / ".hidden.wav").write_bytes(b"RIFF")
    (audio / "sub").mkdir()
    return ls.Config([audio], tmp_path / "out", "17405", "srun --jobid={jobid} --overlap", "som", "", "python", True)


def payload(cfg, chain=("dfn3",), steps=(), **settings):
    base = {"env": "a100", "stop_after": "asr", "chunk_s": 30, "only_available": True, "resume": True,
            "input_files": [str(cfg.audio_roots[0] / "a.wav")]}
    base.update(settings)
    return {"runId": "r1", "trialId": "t1", "experiment": {
        "variants": [{"id": "t1", "label": "T", "chain": list(chain), "threshold": 0.3, "stepParams": list(steps)}],
        "settings": base}}


def test_list_dir_hides_dotfiles_and_non_audio(cfg):
    d = ls.list_dir(cfg, str(cfg.audio_roots[0]))
    assert d["dirs"] == ["sub"] and [f["name"] for f in d["files"]] == ["a.wav"]


def test_list_dir_without_path_returns_roots(cfg):
    assert ls.list_dir(cfg, "")["roots"] is True


@pytest.mark.parametrize("raw", ["/etc", "/etc/passwd", "", "a\0b"])
def test_paths_outside_roots_are_rejected(cfg, raw):
    with pytest.raises(ls.BadRequest):
        ls.safe_path(raw, cfg.audio_roots)


def test_traversal_and_symlink_escape_are_rejected(cfg, tmp_path):
    with pytest.raises(ls.BadRequest):
        ls.safe_path(str(cfg.audio_roots[0] / ".." / ".." / "etc"), cfg.audio_roots)
    (cfg.audio_roots[0] / "link").symlink_to("/etc")
    with pytest.raises(ls.BadRequest):
        ls.safe_path(str(cfg.audio_roots[0] / "link" / "hosts"), cfg.audio_roots)


def test_build_run_maps_chain_and_step_params(cfg):
    steps = [{"model": "dfn3", "chunk_s": "30", "overlap_s": 2}, {"model": "blend", "percent": 50},
             {"model": "mf2se48", "chunk_s": "12"}]
    variants, argv, info = ls.build_run(cfg, payload(cfg, ("dfn3", "blend50", "mf2se48", "dry25"), steps))
    v = variants["variants"][0]
    assert v["chain"] == ["dfn3", "blend50", "mf2se48", "dry25"]
    assert v["set"] == {"denoise.chunk_s": 30, "audio_quality.min_normalized_quality": 0.3,
                        "denoise.dfn3.chunk_s": 30, "denoise.dfn3.overlap_s": 2, "denoise.mf2se48.chunk_s": 12}
    assert "--files" in argv and "--set" not in argv  # run_variants has no global --set
    assert argv[argv.index("--stop-after") + 1] == "asr" and "--resume" in argv and "--only-available" in argv


@pytest.mark.parametrize("chain", [["blend150"], ["dfn3;rm"], [], ["a b"]])
def test_bad_chains_are_rejected(cfg, chain):
    with pytest.raises(ls.BadRequest):
        ls.build_run(cfg, payload(cfg, chain))


def test_same_model_with_different_params_is_rejected(cfg):
    steps = [{"model": "dfn3", "chunk_s": 15}, {"model": "dfn3", "chunk_s": 60}]
    with pytest.raises(ls.BadRequest):
        ls.build_run(cfg, payload(cfg, ("dfn3", "dfn3"), steps))


def test_input_outside_audio_root_is_rejected(cfg):
    with pytest.raises(ls.BadRequest):
        ls.build_run(cfg, payload(cfg, input_files=["/etc/passwd"]))


def test_audio_dir_is_used_when_no_files_are_chosen(cfg):
    _, argv, info = ls.build_run(cfg, payload(cfg, input_files=None, audio_dir=str(cfg.audio_roots[0])))
    assert argv[argv.index("--audio-dir") + 1] == str(cfg.audio_roots[0])


def test_launch_script_activates_conda_before_running(cfg):
    full, plain = ls.launch_argv(cfg, ["--env", "a100"], Path("/x/variants.json"), Path("/x/out"))
    assert full[:4] == ["srun", "--jobid=17405", "--overlap", "bash"] and full[4] == "-lc"
    script = full[-1]
    assert script.index("conda activate som") < script.index("exec python")
    assert "run_variants.py" in plain[1]


def test_dry_run_writes_files_but_starts_nothing(cfg):
    out = ls.start_run(cfg, payload(cfg))
    run_dir = Path(out["runDir"])
    assert out["status"] == "dry-run" and (run_dir / "variants.json").exists()
    assert json.loads((run_dir / "status.json").read_text())["state"] == "dry-run"
    assert ls.run_status(cfg, "r1")["state"] == "dry-run"


def test_run_id_cannot_escape_the_runs_directory(cfg):
    p = payload(cfg)
    p["runId"] = "../../evil"
    out = ls.start_run(cfg, p)
    assert Path(out["runDir"]).parent == cfg.out_root / "lab_runs"


def test_file_audio_root_falls_back_to_its_folder(tmp_path):
    f = tmp_path / "data" / "a1.mp3"
    f.parent.mkdir()
    f.write_bytes(b"x")
    c = ls.Config([f], tmp_path / "out", "1", "", "som", "", "python", True)
    assert c.audio_roots == [f.parent.resolve()]
    assert [x["name"] for x in ls.list_dir(c, "")["files"]] == [] and ls.list_dir(c, str(f.parent))["files"][0]["name"] == "a1.mp3"


def test_listing_a_file_path_lists_its_folder(cfg):
    d = ls.list_dir(cfg, str(cfg.audio_roots[0] / "a.wav"))
    assert d["path"] == str(cfg.audio_roots[0]) and [f["name"] for f in d["files"]] == ["a.wav"]


def test_status_reports_elapsed_and_the_pipeline_log(cfg):
    out = ls.start_run(cfg, payload(cfg))
    run_dir = Path(out["runDir"])
    st = ls.run_status(cfg, "r1")
    assert st["elapsedSeconds"] == 0 and st["pipelineLog"] == ""  # dry-run: nothing ran
    (run_dir / "out" / "logs").mkdir(parents=True)
    (run_dir / "out" / "logs" / "t1.log").write_text("step 1/6\nstep 2/6\n")
    assert "step 2/6" in ls.run_status(cfg, "r1")["pipelineLogTail"]


def test_elapsed_stops_at_the_finish_time_and_runs_while_running():
    assert ls.elapsed({"state": "finished", "startedEpoch": 100.0, "finishedEpoch": 112.5}) == 12.5
    assert ls.elapsed({"state": "running", "startedEpoch": ls.time.time() - 30}) >= 30
    assert ls.elapsed({"state": "running"}) == 0


def test_list_runs_is_newest_first_and_capped(cfg):
    for rid in ("a1", "b1", "c1"):
        p = payload(cfg)
        p["runId"] = rid
        ls.start_run(cfg, p)
    runs = ls.list_runs(cfg, 2)["runs"]
    assert [r["runId"] for r in runs] == ["c1", "b1"]
    assert ls.list_runs(ls.Config([cfg.audio_roots[0]], cfg.out_root / "none", "1", "", "som", "", "python", True))["runs"] == []
