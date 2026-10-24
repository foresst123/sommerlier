import json
import sys
import wave
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
    assert "export PYTHONUNBUFFERED=1" in script and script.index("PYTHONUNBUFFERED") < script.index("exec python")
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


def test_wave_metadata_falls_back_to_wav_header_without_ffprobe(tmp_path, monkeypatch):
    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(8000)
        out.writeframes(b"\0\0" * 8000)
    monkeypatch.setattr(ls.subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(OSError("no ffprobe")))
    duration, rate = ls._ffprobe(path)
    assert duration == 1 and rate == 8000 and ls.WAVEFORM_BINS == 4096


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


def test_workspace_runs_is_read_only_metadata_from_lab_runs_only(cfg):
    out = ls.start_run(cfg, payload(cfg))
    run_dir = Path(out["runDir"])
    # Status is the only manifest read by the workspace listing; no artifact
    # walk or client-supplied path is involved.
    meta = ls.list_workspace_runs(cfg)
    assert meta["runs"][0]["runId"] == "r1"
    assert meta["runs"][0]["state"] == "dry-run"
    assert "runDir" not in meta["runs"][0]
    outside = cfg.out_root / "outside"
    outside.mkdir()
    (cfg.out_root / "lab_runs" / "link-out").symlink_to(outside, target_is_directory=True)
    assert all(item["runId"] != "link-out" for item in ls.list_workspace_runs(cfg)["runs"])


def _asr_workspace_fixture(cfg, payload_text='[{"start": 0, "end": 1, "whisper": "xin chao"}]'):
    run_dir = Path(ls.start_run(cfg, payload(cfg))["runDir"])
    out = run_dir / "out"
    variant_dir = out / "variant-output"
    (variant_dir / "05_asr").mkdir(parents=True)
    (variant_dir / "05_asr" / "transcripts.json").write_text(payload_text)
    (out / "variants_manifest.json").write_text(json.dumps({"variants": [
        {"id": "t1", "files": {"clip": str(variant_dir)}}]}))
    return run_dir


def test_workspace_asr_is_selected_from_manifest_and_reports_timeline(cfg):
    _asr_workspace_fixture(cfg)
    assert ls.workspace_asr(cfg, "r1", "t1")["recordings"] == ["clip"]
    result = ls.workspace_asr(cfg, "r1", "t1", "clip")
    assert result["status"] == "ready" and result["segments"][0]["whisper"] == "xin chao"


def test_workspace_asr_missing_artifact_is_pending_and_empty_is_empty(cfg):
    run_dir = _asr_workspace_fixture(cfg, "[]")
    assert ls.workspace_asr(cfg, "r1", "t1", "clip")["status"] == "empty"
    (run_dir / "out" / "variant-output" / "05_asr" / "transcripts.json").unlink()
    assert ls.workspace_asr(cfg, "r1", "t1", "clip")["status"] == "pending"


def test_workspace_asr_paging_search_and_strict_controls(cfg):
    payload_text = json.dumps([{"start": i, "end": i + 1, "text": "hello" if i % 2 else "target"} for i in range(3)])
    _asr_workspace_fixture(cfg, payload_text)
    first = ls.workspace_asr(cfg, "r1", "t1", "clip", limit="1")
    assert len(first["segments"]) == 1 and first["resultCount"] == 3
    assert first["nextCursor"] == "1"
    second = ls.workspace_asr(cfg, "r1", "t1", "clip", limit="1", cursor="1")
    assert second["cursor"] == "1" and second["prevCursor"] == "0"
    found = ls.workspace_asr(cfg, "r1", "t1", "clip", search="target")
    assert found["resultCount"] == 2
    with pytest.raises(ls.BadRequest):
        ls.workspace_asr(cfg, "r1", "t1", "clip", limit="101")
    with pytest.raises(ls.BadRequest):
        ls.workspace_asr(cfg, "r1", "t1", "clip", search="x" * 121)


def test_workspace_observe_is_bounded_and_reports_only_readiness(cfg):
    run_dir = _asr_workspace_fixture(cfg)
    observed = ls.workspace_observe(cfg, "r1")
    assert observed["status"] == "dry-run"
    assert observed["asr"] == {"available": True, "status": "ready"}
    assert observed["conversations"] == {"available": False, "status": "pending"}
    # The observer must not decode a malformed artifact or enumerate its contents.
    (run_dir / "out" / "variant-output" / "05_asr" / "transcripts.json").write_text("not json")
    observed = ls.workspace_observe(cfg, "r1")
    assert observed["asr"] == {"available": True, "status": "ready"}


def test_workspace_asr_rejects_manifest_path_escape_and_symlink(cfg, tmp_path):
    run_dir = _asr_workspace_fixture(cfg)
    manifest = run_dir / "out" / "variants_manifest.json"
    manifest.write_text(json.dumps({"variants": [{"id": "t1", "files": {"clip": "/etc"}}]}))
    with pytest.raises(ls.ASRArtifactError):
        ls.workspace_asr(cfg, "r1", "t1", "clip")
    safe = run_dir / "out" / "safe"
    safe.mkdir()
    (safe / "05_asr").symlink_to("/etc", target_is_directory=True)
    manifest.write_text(json.dumps({"variants": [{"id": "t1", "files": {"clip": str(safe)}}]}))
    with pytest.raises(ls.ASRArtifactError):
        ls.workspace_asr(cfg, "r1", "t1", "clip")


def _conversation_workspace_fixture(cfg, *, folder="turn_by_turn/c1", verified=True):
    run_dir = Path(ls.start_run(cfg, payload(cfg))["runDir"])
    out = run_dir / "out"
    base = out / "variant-output"
    export = base / "09_conversation_exports" / folder
    export.mkdir(parents=True)
    (export / "mixture.wav").write_bytes(b"RIFF")
    (export / "conversation.json").write_text(json.dumps({
        "id": "c1", "folder": folder, "category": "_quarantine" if not verified else "turn_by_turn",
        "files": {"mixture": "mixture.wav"}, "verification": {"ok": verified},
        "conversation": [{"role": "user", "text": "xin chao", "start": 0, "end": 1}],
    }))
    (base / "09_conversation_exports" / "exports.json").write_text(json.dumps([{
        "id": "c1", "folder": folder, "category": "_quarantine" if not verified else "turn_by_turn",
        "quarantined": not verified, "verified": verified, "duration": 1.0,
    }]))
    (out / "variants_manifest.json").write_text(json.dumps({"variants": [
        {"id": "t1", "files": {"clip": str(base)}}]}))
    return run_dir


def test_workspace_conversations_uses_manifest_and_returns_bounded_detail(cfg):
    _conversation_workspace_fixture(cfg, verified=False)
    listing = ls.workspace_conversations(cfg, "r1", "t1", "clip")
    assert listing["status"] == "ready" and listing["conversations"][0]["quarantined"]
    detail = ls.workspace_conversations(cfg, "r1", "t1", "clip", "c1")
    assert detail["turns"][0]["text"] == "xin chao"
    assert detail["warnings"] and detail["audio"] == {"mixture": "ready"}


def test_workspace_conversations_review_search_and_paging_use_canonical_fields(cfg):
    _conversation_workspace_fixture(cfg, verified=False)
    run_dir = Path(ls.find_run_dir(cfg, "r1"))
    export = run_dir / "out" / "variant-output" / "09_conversation_exports"
    rows = [{"id": "c1", "folder": "turn_by_turn/c1", "quarantined": True, "verified": False, "topic": "target"},
            {"id": "c2", "folder": "turn_by_turn/c1", "quarantined": False, "verified": True, "topic": "target"}]
    (export / "exports.json").write_text(json.dumps(rows))
    flagged = ls.workspace_conversations(cfg, "r1", "t1", "clip", limit=1, review="flagged")
    assert flagged["resultCount"] == 1 and flagged["conversations"][0]["id"] == "c1"
    found = ls.workspace_conversations(cfg, "r1", "t1", "clip", search="target")
    assert found["resultCount"] == 2
    with pytest.raises(ls.BadRequest):
        ls.workspace_conversations(cfg, "r1", "t1", "clip", review="other")


def test_workspace_conversations_rejects_external_folder_and_audio_track(cfg):
    _conversation_workspace_fixture(cfg)
    out = Path(ls.find_run_dir(cfg, "r1")) / "out"
    export = out / "variant-output" / "09_conversation_exports"
    (export / "exports.json").write_text(json.dumps([{"id": "c1", "folder": "/etc"}]))
    with pytest.raises(ls.ASRArtifactError):
        ls.workspace_conversations(cfg, "r1", "t1", "clip", "c1")
    with pytest.raises(ls.BadRequest):
        ls.conversation_audio(cfg, "r1", "t1", "clip", "c1", "raw_path")


def test_conversation_audio_preserves_server_owned_mixture_track(cfg):
    _conversation_workspace_fixture(cfg)
    path = ls.conversation_audio(cfg, "r1", "t1", "clip", "c1", "mixture")
    assert path.name == "mixture.wav"
    assert path.is_file()
class _Done:
    def __init__(self, rc):
        self.rc = rc

    def wait(self):
        return self.rc


def _finished_run(cfg, variant_status, rc=0, reason=None):
    run_dir = Path(ls.start_run(cfg, payload(cfg))["runDir"])
    (run_dir / "out").mkdir()
    entries = [{"id": "none", "status": "ok"}, {"id": "dfn3", "status": variant_status, "reason": reason}]
    (run_dir / "out" / "variants_manifest.json").write_text(json.dumps({"variants": entries}))
    ls.reap(_Done(rc), run_dir)
    return ls.run_status(cfg, "r1")


def test_a_skipped_variant_is_a_failure_even_when_the_process_exits_zero(cfg):
    st = _finished_run(cfg, "skipped", reason="dfn3: missing python package(s): df")
    assert st["state"] == "failed" and st["returncode"] == 0
    assert "skipped" in st["problem"] and "missing python package" in st["problem"]
    assert [v["status"] for v in st["variants"]] == ["ok", "skipped"]


def test_an_ok_variant_finishes_cleanly(cfg):
    st = _finished_run(cfg, "ok")
    assert st["state"] == "finished" and st["problem"] is None


def test_a_nonzero_exit_is_a_failure(cfg):
    assert _finished_run(cfg, "ok", rc=3)["state"] == "failed"


def test_job_log_starts_with_gpu_facts_and_status_parses_them(cfg):
    script = ls.launch_argv(cfg, [], Path("/x/v.json"), Path("/x/out"))[0][-1]
    assert "nvidia-smi -L" in script and "cuda_available=" in script
    run_dir = Path(ls.start_run(cfg, payload(cfg))["runDir"])
    (run_dir / "run.log").write_text("[lab] host=n1 job=17405 CUDA_VISIBLE_DEVICES=0\nGPU 0: A100\n[lab] torch 2.8 cuda_available= False devices= 0\n")
    st = ls.run_status(cfg, "r1")
    assert st["cuda"] is False and st["gpuInfo"][1] == "GPU 0: A100"
    (run_dir / "run.log").write_text("no diagnostics yet\n")
    assert ls.run_status(cfg, "r1")["cuda"] is None


def test_a_skipped_variant_is_flagged_while_the_job_is_still_running(cfg):
    run_dir = Path(ls.start_run(cfg, payload(cfg))["runDir"])
    (run_dir / "out").mkdir()
    (run_dir / "out" / "variants_manifest.json").write_text(json.dumps({"variants": [
        {"id": "none", "status": "missing"}, {"id": "dfn3", "status": "skipped", "reason": "re_use: REUSE_PYTHON is not set"}]}))
    ls.write_status(run_dir, state="running", pid=None)
    st = ls.run_status(cfg, "r1")
    assert st["state"] == "running" and "REUSE_PYTHON" in st["warning"] and st["problem"] is None if "problem" in st else True
    ls.write_status(run_dir, state="running")
    (run_dir / "out" / "variants_manifest.json").write_text(json.dumps({"variants": [{"id": "none", "status": "missing"}, {"id": "dfn3", "status": "missing"}]}))
    assert ls.run_status(cfg, "r1")["warning"] is None  # nothing wrong yet: still waiting for results


def test_set_env_is_exported_inside_the_job_and_only_names_are_logged(tmp_path):
    c = ls.Config([tmp_path], tmp_path / "out", "1", "srun --jobid={jobid} --overlap", "som", "", "python", True,
                  {"REUSE_PYTHON": "/envs/reuse/bin/python", "HF_TOKEN": "s3cr3t value"})
    script = ls.launch_argv(c, [], Path("/x/v.json"), Path("/x/out"))[0][-1]
    assert "export REUSE_PYTHON=/envs/reuse/bin/python" in script
    assert "export HF_TOKEN='s3cr3t value'" in script  # quoted, never interpreted by the shell
    assert script.index("export REUSE_PYTHON") < script.index("exec python")
    assert "[lab] env set: REUSE_PYTHON HF_TOKEN" in script and 'echo "[lab] env set: REUSE_PYTHON HF_TOKEN"' in script
    assert script.count("s3cr3t") == 1  # the value appears once (the export), not in the log line


def test_a_bad_set_env_key_stops_the_server(tmp_path):
    with pytest.raises(SystemExit):
        ls.main(["--audio-root", str(tmp_path), "--out-root", str(tmp_path / "o"), "--launcher", "", "--set-env", "no-equals-sign"])
    with pytest.raises(SystemExit):
        ls.main(["--audio-root", str(tmp_path), "--out-root", str(tmp_path / "o"), "--launcher", "", "--set-env", "BAD KEY=1"])


def test_variant_is_named_after_its_chain_not_the_ui_trial_id(cfg):
    p = payload(cfg, ("re_use", "blend30"))
    variants, argv, info = ls.build_run(cfg, p)
    assert variants["variants"][0]["id"] == info["id"] == "re_use+blend30"
    assert ls.sanitize("a b/c") == "a-b-c"


def test_music_removal_off_becomes_a_global_set_and_unset_keeps_the_default(cfg):
    _, argv, _ = ls.build_run(cfg, payload(cfg, music_removal=False))
    assert argv[argv.index("--set") + 1] == "steps.music_removal=false"
    assert "--set" not in ls.build_run(cfg, payload(cfg, music_removal=True))[1]
    assert "--set" not in ls.build_run(cfg, payload(cfg))[1]


def test_true_false_params_become_real_booleans(cfg):
    """auto_download="false" as a string would be truthy in the backend (bool("false") is True)."""
    steps = [{"model": "rf_denoise", "auto_download": "true"}, {"model": "dfn3", "chunk_s": "30"}]
    v, argv, _ = ls.build_run(cfg, payload(cfg, ("rf_denoise", "dfn3"), steps))
    assert v["variants"][0]["set"]["denoise.rf_denoise.auto_download"] is True
    assert ls.number("False") is False and ls.number(True) is True and ls.number("30") == 30
