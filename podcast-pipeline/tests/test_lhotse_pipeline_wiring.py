"""Wiring of the final output root: quarantine, layouts, shared manifests, auto Lhotse."""
import gzip
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("utils.lhotse_duplex")

from services.conversation_export_service import ConversationExportService as Svc
from services.lhotse_manifest_service import (LhotseManifestParams, build_manifest,
                                              build_recording_manifest, convert_item, find_items)
from tests.test_lhotse_manifest import _make_item
from utils.final_output import (conversation_export_dir, final_output_root,
                                run_lhotse_after_export)


class Args:
    env = "e"


def _cfg(value):
    return {"environments": {"e": {"outputs": {"clean_two_channel_dir": value}}}}


def _item(root, folder, name, stem, **kw):
    return _make_item(root / folder / name, source_stem=stem, **kw)


# -- quarantine ---------------------------------------------------------------------

def test_find_items_skips_quarantine_and_lhotse_dirs(tmp_path):
    _item(tmp_path, "interrupt", "a_1s", "a")
    _item(tmp_path, "_quarantine", "a_2s", "a")
    _item(tmp_path, "lhotse", "a_3s", "a")
    assert [os.path.basename(i[0]) for i in find_items(str(tmp_path))] == ["a_1s"]


def test_convert_item_rejects_quarantine_by_metadata(tmp_path):
    p = _item(tmp_path, "interrupt", "a_1s", "a", category="_quarantine")
    assert convert_item(p, "b", "t", LhotseManifestParams()).reasons[0] == "quarantine"
    p = _item(tmp_path, "interrupt", "a_2s", "a", quality={"quarantine_reason": ["x"]})
    res = convert_item(p, "b", "t", LhotseManifestParams())
    assert res.cut is None and res.reasons[0] == "quarantine"
    p = _item(tmp_path, "interrupt", "a_3s", "a", quality={"quarantine_reason": None})
    assert convert_item(p, "b", "t", LhotseManifestParams()).cut is not None


# -- layouts ------------------------------------------------------------------------

def test_find_items_two_layouts(tmp_path):
    flat = tmp_path / "final"
    _item(flat, "interrupt", "a_1s", "a")
    assert [(b, t) for _, b, t in find_items(str(flat))] == [("final", "interrupt")]
    deep = tmp_path / "deep"
    _make_item(deep / "b1" / "t1" / "interrupt" / "x")
    assert [(b, t) for _, b, t in find_items(str(deep))] == [("b1", "t1")]


# -- choice of out_dir / legacy exporter --------------------------------------------

def test_export_dir_follows_the_config_key(tmp_path):
    out, shared = conversation_export_dir(_cfg(str(tmp_path / "root")), Args, "/run/x")
    assert out == str(tmp_path / "root") and shared
    out, shared = conversation_export_dir(_cfg(""), Args, "/run/x")
    assert out == "/run/x/conversation_exports" and not shared
    assert final_output_root({}, Args) == ""


def test_legacy_exporter_not_called_when_key_set():
    pytest.importorskip("torch")
    from services.pipeline_service import PipelineService

    class Boom:
        def resolve_root(self, v):
            raise AssertionError("legacy exporter used")

        def export(self, **kw):
            raise AssertionError("legacy exporter used")

    svc = PipelineService.__new__(PipelineService)
    svc.clean_dataset_svc, svc.logger = Boom(), None
    assert svc._export_clean_two_channel_dataset(
        Args, _cfg("/x"), None, None, None, None, "a.wav") is None


# -- shared manifests ---------------------------------------------------------------

def _rows(stem, n):
    return [{"path": f"interrupt/{stem}_{i}s/s.wav", "source_stem": stem} for i in range(n)]


def test_write_manifests_keeps_other_recordings(tmp_path):
    d = str(tmp_path)
    Svc._write_manifests(d, _rows("a", 2), _rows("a", 1), "a")
    Svc._write_manifests(d, _rows("b", 1), [], "b")
    Svc._write_manifests(d, _rows("a", 1), [], "a")          # rerun of a replaces only a
    lines = [json.loads(x) for x in open(tmp_path / "manifest.jsonl", encoding="utf-8")]
    assert sorted((r["source_stem"], r["path"]) for r in lines) == [
        ("a", "interrupt/a_0s/s.wav"), ("b", "interrupt/b_0s/s.wav")]
    assert open(tmp_path / "manifest_quarantine.jsonl").read() == ""
    assert not [f for f in os.listdir(d) if ".tmp-" in f]


def test_write_manifests_keeps_rows_without_source_stem_and_default_overwrites(tmp_path):
    (tmp_path / "manifest.jsonl").write_text('{"path": "old"}\n', encoding="utf-8")
    Svc._write_manifests(str(tmp_path), _rows("a", 1), [], "a")
    assert len(open(tmp_path / "manifest.jsonl").readlines()) == 2
    Svc._write_manifests(str(tmp_path), _rows("a", 1), [])
    assert len(open(tmp_path / "manifest.jsonl").readlines()) == 1


def test_clear_previous_scoped_to_one_recording(tmp_path):
    _item(tmp_path, "interrupt", "a_1s", "a")
    _item(tmp_path, "interrupt", "b_1s", "b")
    Svc._clear_previous(str(tmp_path), "a")
    assert os.listdir(tmp_path / "interrupt") == ["b_1s"]
    Svc._clear_previous(str(tmp_path))
    assert not (tmp_path / "interrupt").exists()


# -- automatic Lhotse ---------------------------------------------------------------

def _cuts(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh]


def test_recording_manifest_only_current_recording_and_idempotent(tmp_path):
    _item(tmp_path, "interrupt", "a_1s", "a")
    _item(tmp_path, "backchannel", "a_9s", "a", left="assistant")
    _item(tmp_path, "interrupt", "b_1s", "b")
    _item(tmp_path, "_quarantine", "a_5s", "a")
    r1 = build_recording_manifest(str(tmp_path), "a")
    assert r1.output_path == str(tmp_path / "lhotse" / "manifest.jsonl.gz") and r1.written == 2
    assert r1.part_path == str(tmp_path / "lhotse" / ".parts" / "a.jsonl.gz")
    first = _cuts(r1.output_path)
    swapped = [c["recording"]["sources"][0]["source"] for c in first]
    assert any(os.path.dirname(s) == str(tmp_path / "lhotse" / "audio") for s in swapped)
    r2 = build_recording_manifest(str(tmp_path), "a")
    assert r2.written == 2 and len(_cuts(r2.output_path)) == 2
    assert not (tmp_path / "lhotse" / "a.jsonl.gz").exists()          # no loose per-recording file
    assert not (tmp_path / "lhotse" / "b.jsonl.gz").exists()
    # the CLI-style scan ignores the lhotse folder and merges every recording
    full = build_manifest(str(tmp_path), str(tmp_path / "all.jsonl.gz"))
    assert full.written == 3


def test_converter_failure_does_not_raise(tmp_path, monkeypatch):
    import services.lhotse_manifest_service as m

    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(m, "build_recording_manifest", boom)
    report = {}
    info = run_lhotse_after_export(str(tmp_path), "a", report)
    assert info == {"error": "disk full"} and report["lhotse"] == info


def test_run_lhotse_after_export_reports_success(tmp_path):
    _item(tmp_path, "interrupt", "a_1s", "a")
    report = {}
    run_lhotse_after_export(str(tmp_path), "a", report)
    assert report["lhotse"]["written"] == 1


def test_lhotse_params_come_from_the_profile_and_default_to_none():
    from utils.final_output import lhotse_params
    assert lhotse_params(_cfg("/x"), Args()) is None
    cfg = {"environments": {"e": {"outputs": {"lhotse": {"coverage_max_uncovered": 1.5,
                                                          "coverage_rms_threshold": 0.02}}}}}
    params = lhotse_params(cfg, Args())
    assert params.coverage_max_uncovered == 1.5 and params.coverage_rms_threshold == 0.02
    assert params.coverage_min_speech_frames == 3             # untouched keys keep the converter's default


def test_a_typo_in_outputs_lhotse_is_an_error():
    from utils.final_output import lhotse_params
    with pytest.raises(ValueError):
        lhotse_params({"environments": {"e": {"outputs": {"lhotse": {"coverage_max": 1}}}}}, Args())


def test_run_lhotse_after_export_hands_the_params_to_the_converter(tmp_path, monkeypatch):
    import services.lhotse_manifest_service as m
    seen = {}

    def fake(root, stem, params=None):
        seen["params"] = params
        return type("R", (), {"written": 0, "skipped": [], "output_path": "x"})()
    monkeypatch.setattr(m, "build_recording_manifest", fake)
    marker = LhotseManifestParams(coverage_max_uncovered=1.5)
    run_lhotse_after_export(str(tmp_path), "a", {}, params=marker)
    assert seen["params"] is marker


def test_a_loose_tolerance_keeps_an_item_the_default_drops(tmp_path):
    import numpy as np
    import soundfile as sf
    from tests.test_lhotse_manifest import SR, _base_rows
    item = _make_item(tmp_path / "item", _base_rows())
    data, _ = sf.read(os.path.join(item, "stereo_2ch.wav"))
    data[int(7.2 * SR):int(8.0 * SR), 1] = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(0.8 * SR)) / SR)
    sf.write(os.path.join(item, "stereo_2ch.wav"), data, SR, subtype="PCM_16")   # 0.8 s unannotated voice
    assert convert_item(item, "b", "t", LhotseManifestParams(), index=0).cut is None
    loose = convert_item(item, "b", "t", LhotseManifestParams(coverage_max_uncovered=1.5), index=0)
    assert loose.cut is not None
