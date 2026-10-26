"""run_variants with a chain at each of the three noise-filter points, and the shipped 14-variant plan."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_variants", ROOT / "tools" / "run_variants.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

L = "rf_crowd>rf_vocals_resurrection>rf_dereverb_anvuew"
M = "mpsenet_dns"
DIRS = {"": "01b_denoise", "_pre_separation": "02b_denoise_pre_separation", "_pre_asr": "04b_denoise_pre_asr"}


def load(tmp_path, entries):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"variants": entries}))
    return {v["id"]: v for v in rv.load_variants(p)}


def cmd(v, tmp_path):
    a = SimpleNamespace(out_root=str(tmp_path / "out"), python="py", main="main.py", env="a100",
                        stop_after="none", cache_dir=None, global_set=[])
    return rv.build_command(v, [], a)


def test_command_flags_only_for_non_empty_points(tmp_path):
    v = load(tmp_path, [{"chain": "dfn3"}, {"chain_pre_separation": M}, {"chain_pre_asr": L},
                        {"chain": "dfn3", "chain_pre_separation": "mf2se48", "chain_pre_asr": M}])
    c = cmd(v["dfn3"], tmp_path)
    assert c[c.index("--denoise") + 1] == "dfn3"
    assert "--denoise_pre_separation" not in c and "--denoise_pre_asr" not in c
    c = cmd(v[f"none.sep-{M}"], tmp_path)
    assert "--denoise" not in c and c[c.index("--denoise_pre_separation") + 1] == M
    c = cmd(v["none.asr-" + L.replace(">", "+")], tmp_path)
    assert c[c.index("--denoise_pre_asr") + 1] == L and "--denoise_pre_separation" not in c
    c = cmd(v["dfn3.sep-mf2se48.asr-mpsenet_dns"], tmp_path)
    assert [c[c.index(f) + 1] for f in ("--denoise", "--denoise_pre_separation", "--denoise_pre_asr")] == [
        "dfn3", "mf2se48", M]
    c = cmd(v["none"], tmp_path)
    assert not any(f.startswith("--denoise") for f in c)


def test_ids_stay_the_same_for_p1_only_and_are_unique_and_safe_otherwise(tmp_path):
    v = load(tmp_path, [{"chain": "dfn3"}, {"chain": "dfn3>mf2se48"}, {"chain_pre_asr": "dfn3"},
                        {"chain_pre_separation": "dfn3"}, {"chain": "dfn3", "chain_pre_asr": "dfn3"},
                        {"chain_pre_asr": "dfn3", "id": "mine"}])
    assert list(v) == ["none", "dfn3", "dfn3+mf2se48", "none.asr-dfn3", "none.sep-dfn3",
                       "dfn3.asr-dfn3", "mine"]
    assert all(rv.ID_RE.match(i) for i in v)


def test_a_variant_with_a_chain_is_never_the_baseline(tmp_path):
    v = load(tmp_path, [{"chain_pre_asr": "dfn3"}])
    assert [i for i in v if i == "none"] == ["none"] and not rv.has_chain(v["none"])
    with pytest.raises(SystemExit):
        load(tmp_path, [{"chain_pre_asr": "dfn3", "id": "none"}])
    kept = rv.pick_variants(list(v.values()), only_ids=["none.asr-dfn3"])
    assert [x["id"] for x in kept] == ["none", "none.asr-dfn3"]


def test_availability_covers_all_three_chains(tmp_path, monkeypatch):
    import models.denoise as md
    monkeypatch.setattr(md, "available_backends", lambda cfg=None: {
        "dfn3": (True, ""), "mf2se48": (False, "no weights"), M: (False, "no ckpt")})
    v = load(tmp_path, [{"chain": "dfn3"}, {"chain_pre_separation": "mf2se48"},
                        {"chain": "dfn3", "chain_pre_asr": f"dfn3>{M}"}, {"chain_pre_asr": "dfn3"}])
    skipped = rv.check_availability(list(v.values()))
    assert set(skipped) == {"none.sep-mf2se48", f"dfn3.asr-dfn3+{M}"}
    assert "mf2se48: no weights" in skipped["none.sep-mf2se48"] and M in skipped[f"dfn3.asr-dfn3+{M}"]


def _write(root, stem, sfx, info):
    d = root / stem / DIRS[sfx]
    d.mkdir(parents=True)
    if info is not None:
        (d / "denoise.json").write_text(info if isinstance(info, str) else json.dumps(info))
    (root / stem / "02_diarization").mkdir(exist_ok=True)


def test_verdict_is_read_per_point_and_names_the_failed_one(tmp_path):
    v = load(tmp_path, [{"chain": "dfn3", "chain_pre_separation": "mf2se48", "chain_pre_asr": M}])
    v = v["dfn3.sep-mf2se48.asr-" + M]
    good = {"errors": [], "corr_with_input": 0.9, "lag_samples": 0}
    root = tmp_path / "var"
    _write(root, "a", "", good)
    _write(root, "a", "_pre_separation", good)
    _write(root, "a", "_pre_asr", {"errors": ["mpsenet_dns: out of memory"], "corr_with_input": 1.0})
    dirs = rv.discover_dirs(root, {"a"})
    rep = rv.variant_report(v, dirs, ["a"])
    assert rep["a"]["verdict"] == "fallback"
    assert rep["a"]["errors"] == ["pre_asr: mpsenet_dns: out of memory"]
    assert {k: x["verdict"] for k, x in rep["a"]["points"].items()} == {
        "pre_diarization": "ok", "pre_separation": "ok", "pre_asr": "fallback"}
    counts = rv.report_counts(rep, 1)
    assert counts["degraded_files"] == 1
    assert "pre_asr" in rv.degraded_reason(rep, counts)


def test_missing_point_json_is_degraded_and_noop_is_a_warning(tmp_path):
    v = list(load(tmp_path, [{"chain_pre_separation": "dfn3"}]).values())[1]
    root = tmp_path / "var"
    _write(root, "a", "_pre_separation", None)       # dir exists, no denoise.json
    rep = rv.variant_report(v, rv.discover_dirs(root, {"a"}), ["a"])
    assert rep["a"]["verdict"] == "missing"
    assert rep["a"]["errors"] == ["pre_separation: 02b_denoise_pre_separation/denoise.json is absent"]
    root = tmp_path / "var2"
    _write(root, "a", "_pre_separation", {"errors": [], "corr_with_input": 1.0})
    rep = rv.variant_report(v, rv.discover_dirs(root, {"a"}), ["a"])
    assert rep["a"]["verdict"] == "suspect_noop" and rv.report_counts(rep, 1)["degraded_files"] == 0


def test_p1_only_report_is_exactly_the_old_one(tmp_path):
    v = list(load(tmp_path, [{"chain": "dfn3"}]).values())[1]
    root = tmp_path / "var"
    _write(root, "a", "", {"errors": ["dfn3: x"], "corr_with_input": 1.0})
    dirs = rv.discover_dirs(root, {"a"})
    assert rv.variant_report(v, dirs, ["a"]) == rv.denoise_report(["dfn3"], dirs, ["a"])
    assert "points" not in rv.variant_report(v, dirs, ["a"])["a"]


def test_manifest_rows_and_plan_show_all_chains(tmp_path, capsys):
    v = list(load(tmp_path, [{"chain": "dfn3", "chain_pre_separation": "mf2se48", "chain_pre_asr": M}]).values())
    rv.write_manifest(tmp_path / "out", v, {"a"}, {}, {})
    row = json.loads((tmp_path / "out" / rv.MANIFEST_NAME).read_text())["variants"][1]
    assert (row["chain"], row["chain_pre_separation"], row["chain_pre_asr"]) == (["dfn3"], ["mf2se48"], [M])
    args = SimpleNamespace(only_available=False, minutes_per_file_run=1.0)
    rv.print_plan(v, [Path("a.wav")], {}, args)
    out = capsys.readouterr().out
    assert "pre_separation" in out and "pre_asr" in out and "mf2se48" in out and M in out


def test_compare_tooling_labels_a_variant_without_a_p1_chain(tmp_path):
    from tools.compare_lib.io_stage import denoise_entry
    ent = denoise_entry(["mf2se48"], {"errors": [], "corr_with_input": 0.9, "steps": [{"token": "mf2se48"}]})
    assert ent["verdict"] == "ok" and ent["steps"]
    sys_path = str(ROOT)
    import sys
    sys.path.insert(0, sys_path)
    try:
        import tools.make_compare_page as mcp
    finally:
        sys.path.remove(sys_path)
    chips = mcp._chips([{"id": "x", "label": "X", "chain": [], "chain_pre_asr": ["mf2se48"]}], "none")
    assert "pre_asr: mf2se48" in chips


# --- the shipped plan ---------------------------------------------------------------
def test_placement_plan_is_the_14_variant_table():
    vs = rv.load_variants(ROOT / "tools" / "variants_placement.json")
    by = {v["id"]: v for v in vs}
    assert len(vs) == 14 == len(by)
    assert [v["id"] for v in vs if not rv.has_chain(v) and not v["set"]] == ["none"]
    assert vs[0]["id"] == "none" and vs[0]["group"] == "control"

    def chains(v):
        return tuple(">".join(v[k]) for k in ("chain", "chain_pre_separation", "chain_pre_asr"))

    table = {
        "raw": ("", "", ""), "none": ("", "", ""),
        "l_p1": (L, "", ""), "l_p2": ("", L, ""), "l_p3": ("", "", L),
        "m_p1": (M, "", ""), "m_p2": ("", M, ""), "m_p3": ("", "", M),
        "lm_p1": (f"{L}>{M}", "", ""), "lm_p2": ("", f"{L}>{M}", ""), "lm_p3": ("", "", f"{L}>{M}"),
        "l_p1_m_p2": (L, M, ""), "l_p1_m_p3": (L, "", M), "l_p2_m_p3": ("", L, M),
    }
    assert {i: chains(v) for i, v in by.items()} == table
    groups = {"raw": "control", "none": "control", "l_p1": "light", "l_p2": "light", "l_p3": "light",
              "m_p1": "mpsenet", "m_p2": "mpsenet", "m_p3": "mpsenet", "lm_p1": "same_point",
              "lm_p2": "same_point", "lm_p3": "same_point", "l_p1_m_p2": "light_earlier",
              "l_p1_m_p3": "light_earlier", "l_p2_m_p3": "light_earlier"}
    assert {i: v["group"] for i, v in by.items()} == groups
    assert by["raw"]["set"] == {"steps.music_analysis": False, "steps.music_removal": False}
    assert all(not v["set"] for i, v in by.items() if i != "raw")
    for v in vs:
        assert v["label"] != v["id"] and "→" in v["label"] and v["label"].startswith("audio gốc")
    assert by["l_p1_m_p3"]["label"] == ("audio gốc → tiền xử lý → rf_crowd → rf_vocals_resurrection → "
                                        "rf_dereverb_anvuew → diarization → separation → mpsenet_dns → asr")


def test_placement_plan_steps_keys_exist_in_the_profiles():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    for name, profile in cfg["environments"].items():
        if "steps" in profile:
            assert {"music_analysis", "music_removal"} <= set(profile["steps"]), name
