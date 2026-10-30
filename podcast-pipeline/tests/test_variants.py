"""Variant registry (tools/variants.py), its dry-run commands, and the PLDA embedder guard of windowed_vbx_stage.

Fakes only: nothing here runs a model or a subprocess. Run: python -m pytest tests/test_variants.py -q
"""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import diar_arms_experiment as DA  # noqa: E402
from tools import embedders as EMB  # noqa: E402
from tools import variants as VR  # noqa: E402
from tools import windowed_vbx_stage as V  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "tools" / "variant_params.example.json"
SAMPLES = ROOT / "tools" / "diar_arms_samples.json"

def example_params():
    with open(EXAMPLE, encoding="utf-8") as fh:
        return json.load(fh)


def filled_params(**extra):
    """Every example key filled with a value: a dir, an lda_dim, a holdout result and a cos cut for each kind."""
    p = {}
    for kind in EMB.KINDS:
        p[f"plda.{kind}.dir"] = f"/plda/{kind}"
        p[f"plda.{kind}.lda_dim"] = 96
        p[f"plda.{kind}.holdout"] = 0.1
        p[f"cos.{kind}"] = 0.5
    p.update(extra)
    return p


def arms_main_cfg():
    """The cfg tools/diar_arms_experiment.main builds, from its argparse defaults. Written out here on purpose: the
    tests compare the variant commands with the arms script's own helpers fed with this dict, not with VR.arms_cfg."""
    return {"python": "SOM", "vbx_python": "DZ", "windowed_python": "NEMO", "demucs_python": "SOM",
            "embedder_python": {}, "reuse": False, "gpu": "0", "config": "CFG", "env": "a100",
            "windowed_piece_s": DA.WINDOWED_PIECE_S, "windowed_hop_s": DA.WINDOWED_HOP_S,
            "dense_window_s": DA.DENSE_WINDOW_S, "dense_hop_s": DA.DENSE_HOP_S,
            "dz_thr": 0.5, "dz_median_s": 0.22, "vbx_ahc_threshold": DA.VBX_AHC_THRESHOLD, "vbx_fa": DA.VBX_FA,
            "vbx_fb": DA.VBX_FB, "plda_dir": None, "windowed_min_speakers": 1, "r293_onnx": None,
            "simam_onnx": None, "simam_dir": None, "espnet_tag": EMB.DEFAULT_ESPNET_TAG}


def vr_cfg():
    return VR.arms_cfg("SOM", "DZ", "NEMO", "SOM", {}, "a100", "CFG")


def paths(tmp_path):
    sample = tmp_path / "01_x"
    probs = sample / "runs" / "nemotron_dz" / "probs"
    return sample, probs


def test_registry_is_consistent_with_the_example_file():
    assert VR.check_registry(list(example_params().keys())) == []


def test_ids_are_unique_and_the_count_is_known():
    ids = [v.id for v in VR.REGISTRY]
    assert len(ids) == len(set(ids)) == 17
    assert set(VR.BY_ID) == set(ids)


def test_every_referenced_param_exists_in_the_example_file():
    keys = set(example_params())
    for v in VR.REGISTRY:
        for key in VR.param_refs(v):
            assert key in keys, (v.id, key)


def test_example_file_has_the_four_kinds_and_their_keys():
    keys = set(example_params())
    for kind in EMB.KINDS:
        for name in (f"plda.{kind}.dir", f"plda.{kind}.lda_dim", f"plda.{kind}.holdout", f"cos.{kind}"):
            assert name in keys
    assert all(v is None for k, v in example_params().items() if not k.startswith("_"))


def test_nemo_sc_stays_todo_and_refuses_to_run():
    v = VR.BY_ID["sc_pna"]
    assert v.branch == "nemo_sc" and v.status == "todo"
    with pytest.raises(VR.VariantRefused, match="todo"):
        VR.resolve(v, filled_params())


def test_a_ready_nemo_sc_variant_is_reported_by_the_registry_check(monkeypatch):
    broken = replace(VR.BY_ID["sc_pna"], status="ready", todo_reason="")
    monkeypatch.setattr(VR, "REGISTRY", [broken])
    problems = VR.check_registry(list(example_params()))
    assert any("nemo_sc must stay todo" in p for p in problems)


def test_a_duplicate_id_is_reported(monkeypatch):
    monkeypatch.setattr(VR, "REGISTRY", [VR.BY_ID["J_base"], VR.BY_ID["J_base"]])
    assert any("duplicate variant id" in p for p in VR.check_registry(list(example_params())))


def test_null_param_refuses_the_variant_and_names_the_parameter():
    with pytest.raises(VR.VariantRefused) as exc:
        VR.resolve(VR.BY_ID["emb_r34lm_dzahc"], example_params())
    assert "cos.r34lm" in str(exc.value)


def test_null_plda_dir_refuses_the_vbx_variant_and_names_it():
    with pytest.raises(VR.VariantRefused) as exc:
        VR.resolve(VR.BY_ID["emb_r34lm_vbx"], example_params())
    assert "plda.r34lm.dir" in str(exc.value)


def test_missing_key_counts_as_null():
    params = filled_params()
    del params["plda.r34lm.holdout"]
    with pytest.raises(VR.VariantRefused, match="plda.r34lm.holdout"):
        VR.resolve(VR.BY_ID["emb_r34lm_vbx"], params)


def test_baseline_variants_need_no_parameter():
    for vid in ("J_base", "K_base", "J_ov_thr0.35_on0.2", "K_ov_thr0.35_on0.2", "J_fb0.2", "J_fb0.4",
                "K_fb0.2", "K_fb0.4"):
        assert VR.resolve(VR.BY_ID[vid], example_params())["plda_dir"] is None


def test_filled_params_resolve_to_values():
    got = VR.resolve(VR.BY_ID["emb_r34lm_vbx"], filled_params())
    assert got == {"plda_dir": "/plda/r34lm", "lda_dim": 96, "cos": None}
    got = VR.resolve(VR.BY_ID["emb_wespeaker_r293_dzahc"], filled_params())
    assert got["cos"] == 0.5 and got["plda_dir"] is None


def test_J_dry_run_command_equals_the_arms_script_command(tmp_path):
    cfg = arms_main_cfg()
    sample, probs = paths(tmp_path)
    result = sample / "runs" / "nemotron_dz" / "out" / "segments.json"
    expected = DA.stage2_cmd("J", sample, probs, result, cfg, fb=str(cfg["vbx_fb"]), embed_audio=None)
    got = VR.stage2_command(VR.BY_ID["J_base"], sample, vr_cfg(),
                            {"plda_dir": None, "lda_dim": None, "cos": None})
    assert got == expected
    assert got[got.index("--clustering") + 1] == "dz"
    assert got[got.index("--thr") + 1] == "0.5"
    assert got[got.index("--median-s") + 1] == "0.22"
    assert got[got.index("--fb") + 1] == "0.8"
    assert got[got.index("--fa") + 1] == "0.07"
    assert got[got.index("--ahc-threshold") + 1] == "0.6"
    assert "--plda-dir" not in got and "--min-on-s" not in got and "--embed-audio" not in got


def test_K_dry_run_command_equals_the_arms_script_command(tmp_path):
    cfg = arms_main_cfg()
    sample, probs = paths(tmp_path)
    result = sample / "runs" / "nemotron_dz_demucs" / "out" / "segments.json"
    expected = DA.stage2_cmd("K", sample, probs, result, cfg, fb=str(cfg["vbx_fb"]),
                             embed_audio=sample / DA.VOCALS_NAME)
    got = VR.stage2_command(VR.BY_ID["K_base"], sample, vr_cfg(), {"plda_dir": None, "lda_dim": None, "cos": None})
    assert got == expected
    assert got[got.index("--embed-audio") + 1] == str(sample / DA.VOCALS_NAME)


def test_shared_steps_equal_the_arms_script_steps(tmp_path):
    cfg = arms_main_cfg()
    sample, _ = paths(tmp_path)
    calls = []

    def fake_runner(cmd, env):          # records the command; writes what the arms script waits for
        calls.append(cmd)
        if "--dump-probs" in cmd:
            probs = Path(cmd[cmd.index("--dump-probs") + 1])
            probs.mkdir(parents=True, exist_ok=True)
            (probs / "plan.json").write_text("{}")
        if "--out" in cmd and cmd[1].endswith("demucs_vocals.py"):
            Path(cmd[cmd.index("--out") + 1]).write_bytes(b"x")
        return 0

    DA.dump_probs("J", sample, cfg, fake_runner, skip_existing=False)
    DA.ensure_vocals(sample, cfg, fake_runner, skip_existing=False)
    assert calls[0] == VR.stage1_command(sample, vr_cfg())
    assert calls[1] == VR.demucs_command(sample, vr_cfg())


def test_fb_variant_is_the_arms_fb_command(tmp_path):
    cfg = arms_main_cfg()
    sample, probs = paths(tmp_path)
    result = sample / "runs" / "nemotron_dz_fb0.4" / "out" / "segments.json"
    expected = DA.stage2_cmd("J", sample, probs, result, cfg, fb="0.4", embed_audio=None)
    v = VR.BY_ID["J_fb0.4"]
    got = VR.stage2_command(v, sample, vr_cfg(), {"plda_dir": None, "lda_dim": None, "cos": None})
    assert got == expected
    assert VR.run_name(v, {"cos": None}) == DA.arm_name("J@0.4") == "nemotron_dz_fb0.4"


def test_overlap_variant_is_the_arms_overlap_command(tmp_path):
    cfg = arms_main_cfg()
    sample, probs = paths(tmp_path)
    result = sample / "runs" / "nemotron_dz_ov_thr0.35_on0.2" / "out" / "segments.json"
    expected = DA.stage2_cmd("J", sample, probs, result, cfg, fb=str(cfg["vbx_fb"]), embed_audio=None,
                             thr=0.35, min_on_s=0.2)
    v = VR.BY_ID["J_ov_thr0.35_on0.2"]
    got = VR.stage2_command(v, sample, vr_cfg(), {"plda_dir": None, "lda_dim": None, "cos": None})
    assert got == expected
    assert got[got.index("--thr") + 1] == "0.35" and got[got.index("--min-on-s") + 1] == "0.2"
    assert VR.run_name(v, {"cos": None}) == DA.arm_name("J+ov|0.35|0.2")


def test_embedder_dzahc_variant_is_the_arms_trial_command(tmp_path):
    cfg = arms_main_cfg()
    sample, probs = paths(tmp_path)
    result = sample / "runs" / "nemotron_dz_emb_wespeaker_r293_c0.5" / "out" / "segments.json"
    expected = DA.trial_cmd("wespeaker_r293", 0.5, sample, probs, result, cfg)
    v = VR.BY_ID["emb_wespeaker_r293_dzahc"]
    resolved = VR.resolve(v, filled_params())
    got = VR.stage2_command(v, sample, vr_cfg(), resolved)
    assert got == expected
    assert got[got.index("--clustering") + 1] == "dz-ahc"
    assert got[got.index("--embedder-kind") + 1] == "wespeaker_r293"
    assert got[got.index("--cos-threshold") + 1] == "0.5"
    assert VR.run_name(v, resolved) == "nemotron_dz_emb_wespeaker_r293_c0.5"


def test_vbx_variant_takes_the_plda_folder_and_lda_dim_from_params(tmp_path):
    cfg = arms_main_cfg()
    cfg["plda_dir"] = "/plda/r34lm"
    sample, probs = paths(tmp_path)
    result = sample / "runs" / "nemotron_dz_vbx_r34lm" / "out" / "segments.json"
    expected = DA.stage2_cmd("J", sample, probs, result, cfg, fb=str(cfg["vbx_fb"]), embed_audio=None) \
        + ["--lda-dim", "96"]
    v = VR.BY_ID["emb_r34lm_vbx"]
    got = VR.stage2_command(v, sample, vr_cfg(), VR.resolve(v, filled_params()))
    assert got == expected
    assert got[got.index("--plda-dir") + 1] == "/plda/r34lm"


def test_cli_refuses_a_null_variant_and_prints_the_others(tmp_path, capsys):
    rc = VR.main(["commands", "--params", str(EXAMPLE), "--samples", str(SAMPLES), "--out", str(tmp_path),
                  "--only", "J_base,emb_r34lm_dzahc"])
    out, err = capsys.readouterr()
    assert rc == 2
    assert "cos.r34lm" in err
    assert "# J_base" in out and "emb_r34lm_dzahc" not in out
    with open(SAMPLES, encoding="utf-8") as fh:
        n_samples = len(json.load(fh))
    assert out.count("windowed_vbx_stage.py") == n_samples


def test_cli_refuses_sc_pna(tmp_path, capsys):
    rc = VR.main(["commands", "--params", str(EXAMPLE), "--samples", str(SAMPLES), "--out", str(tmp_path),
                  "--only", "sc_pna"])
    out, err = capsys.readouterr()
    assert rc == 2 and "todo" in err and "windowed_vbx_stage" not in out


def test_cli_unknown_id_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        VR.main(["commands", "--samples", str(SAMPLES), "--out", str(tmp_path), "--only", "nope"])


def test_cli_list_runs(capsys):
    assert VR.main(["list"]) == 0
    assert "emb_r34lm_vbx" in capsys.readouterr().out


# ---- the embedder guard in windowed_vbx_stage (pure function, no files besides the meta dict) ----

def test_guard_r34lm_without_meta_is_the_default_and_allowed():
    assert V.plda_embedder_problem("r34lm", None) is None


def test_guard_r34lm_with_matching_meta_is_allowed():
    assert V.plda_embedder_problem("r34lm", {"embedder_id": "r34lm"}) is None


def test_guard_r34lm_with_another_embedders_meta_is_refused():
    msg = V.plda_embedder_problem("r34lm", {"embedder_id": "wespeaker_r293"})
    assert msg and "wespeaker_r293" in msg and "r34lm" in msg


def test_guard_other_embedder_without_meta_is_refused():
    msg = V.plda_embedder_problem("wespeaker_r293", None)
    assert msg and "absent" in msg and "wespeaker_r293" in msg


def test_guard_other_embedder_with_matching_meta_is_allowed():
    assert V.plda_embedder_problem("wespeaker_r293", {"embedder_id": "wespeaker_r293", "lda_dim": 128}) is None


def test_guard_other_embedder_with_mismatched_meta_is_refused():
    msg = V.plda_embedder_problem("espnet_spk", {"embedder_id": "wespeaker_simam100"})
    assert msg and "wespeaker_simam100" in msg and "espnet_spk" in msg


def test_read_plda_meta_absent_and_present(tmp_path):
    assert V.read_plda_meta(str(tmp_path)) is None
    (tmp_path / "plda_meta.json").write_text(json.dumps({"embedder_id": "r34lm"}))
    assert V.read_plda_meta(str(tmp_path)) == {"embedder_id": "r34lm"}
