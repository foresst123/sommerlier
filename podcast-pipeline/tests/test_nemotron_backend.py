"""Nemotron diarization backend: conversion, selection and folder naming.

Run:  python -m pytest tests/test_nemotron_backend.py -q     (from podcast-pipeline/)
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.pipeline_service import PipelineService
from utils.diarization_backend import resolve_diarization_backend
from utils.nemotron_segments import to_pipeline_segments


def _args(**kw):
    base = dict(save_path="./output", bss=True, vad=True, dia3=False, merge_gap=2.0,
                seg_th=0.11, min_cluster_size=11, clust_th=0.5, LLM="case_0")
    base.update(kw)
    return types.SimpleNamespace(**base)


# --- conversion ------------------------------------------------------------

def test_string_segments_become_pipeline_dicts():
    out = to_pipeline_segments(["0.00 3.20 speaker_0", "3.50 5.00 speaker_1"])
    assert out == [{"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00"},
                   {"start": 3.5, "end": 5.0, "speaker": "SPEAKER_01"}]


def test_speakers_are_renamed_by_first_appearance():
    out = to_pipeline_segments(["0 1 speaker_5", "1 2 speaker_2", "2 3 speaker_5"])
    assert [s["speaker"] for s in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_tuple_and_processor_dict_shapes():
    out = to_pipeline_segments([(0.0, 1.0, 0), {"Start": 1.0, "End": 2.0, "Speaker": 1}])
    assert [s["speaker"] for s in out] == ["SPEAKER_00", "SPEAKER_01"]


def test_same_speaker_gap_is_merged_but_not_across_speakers():
    raw = ["0 1 speaker_0", "1.2 2 speaker_0", "2.1 3 speaker_1"]
    out = to_pipeline_segments(raw, merge_gap=0.5)
    assert out == [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                   {"start": 2.1, "end": 3.0, "speaker": "SPEAKER_01"}]
    assert len(to_pipeline_segments(raw)) == 3


def test_overlap_between_speakers_is_preserved():
    out = to_pipeline_segments(["0 4 speaker_0", "2 3 speaker_1"], merge_gap=1.0)
    assert len(out) == 2


def test_min_duration_and_empty_and_degenerate_input():
    assert to_pipeline_segments(["0 0.1 speaker_0", "1 2 speaker_0"], min_duration=0.2) == [
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    assert to_pipeline_segments([]) == []
    assert to_pipeline_segments(["2 2 speaker_0"]) == []


def test_unparseable_segment_raises():
    with pytest.raises(ValueError):
        to_pipeline_segments(["garbage"])


# --- backend selection -----------------------------------------------------

def test_default_backend_is_diarizen():
    assert resolve_diarization_backend(_args()) == "diarizen"
    assert resolve_diarization_backend(_args(diarization_backend=None), {}) == "diarizen"


def test_profile_selects_backend_and_cli_wins():
    profile = {"models": {"diarization_backend": "nemotron"}}
    assert resolve_diarization_backend(_args(), profile) == "nemotron"
    assert resolve_diarization_backend(
        _args(diarization_backend="diarizen"), profile) == "diarizen"


def test_unknown_backend_raises_clearly():
    with pytest.raises(ValueError, match="Unknown diarization backend 'sortformer'"):
        resolve_diarization_backend(_args(diarization_backend="sortformer"))


def test_nemotron_with_dia3_is_rejected():
    with pytest.raises(ValueError, match="dia3"):
        resolve_diarization_backend(_args(diarization_backend="nemotron", dia3=True))


# --- output folder ---------------------------------------------------------

def _dir(**kw):
    return PipelineService(*(None,) * 8)._resolve_output_dir(_args(**kw), "/audio/talk.mp3")


def test_default_folder_name_is_unchanged():
    base = _dir()
    assert "-diaModel-diarizen-" in base
    assert _dir(diarization_backend="diarizen") == base


def test_nemotron_gets_its_own_folder():
    out = _dir(diarization_backend="nemotron")
    assert "-diaModel-nemotron-" in out
    assert out != _dir()


# --- worker hardening ------------------------------------------------------

def test_the_ready_line_must_be_json_because_nemo_logs_to_stdout():
    from services.nemotron_worker_service import NemotronWorkerService
    assert NemotronWorkerService.ready_requires_json is True


def test_a_broken_conversion_module_is_reported_not_raised(tmp_path, monkeypatch):
    import nemotron_worker
    monkeypatch.setattr(nemotron_worker.importlib.util if hasattr(nemotron_worker, "importlib") else
                        __import__("importlib.util").util, "spec_from_file_location",
                        lambda *a, **k: (_ for _ in ()).throw(ImportError("boom")))
    out = nemotron_worker.diarize(types.SimpleNamespace(diarize=lambda **k: []), "x.wav", {})
    assert isinstance(out, str) and out.startswith("[ERROR]")


# --- thresholds and streaming settings --------------------------------------

def test_postprocessing_is_off_unless_the_profile_sets_it():
    from utils.nemotron_segments import postprocessing_params
    assert postprocessing_params({}) is None and postprocessing_params({"chunk_len": 9}) is None


def test_postprocessing_overrides_land_on_the_base_values():
    from utils.nemotron_segments import POSTPROCESSING_BASE, postprocessing_params
    out = postprocessing_params({"postprocessing": {"onset": 0.7, "min_duration_on": 0.1}})
    assert out["onset"] == 0.7 and out["min_duration_on"] == 0.1
    assert out["offset"] == POSTPROCESSING_BASE["offset"] and set(out) == set(POSTPROCESSING_BASE)


def test_an_unknown_postprocessing_key_is_an_error():
    from utils.nemotron_segments import postprocessing_params
    with pytest.raises(ValueError, match="onsett"):
        postprocessing_params({"postprocessing": {"onsett": 0.5}})


def test_the_yaml_has_the_parameters_block_nemo_reads(tmp_path):
    from utils.nemotron_segments import postprocessing_params, write_postprocessing_yaml
    path = write_postprocessing_yaml(postprocessing_params({"postprocessing": {"onset": 0.6}}), str(tmp_path))
    text = open(path, encoding="utf-8").read().splitlines()
    assert text[0] == "parameters:" and "  onset: 0.6" in text and len(text) == 7


def test_the_worker_passes_the_yaml_to_diarize():
    import nemotron_worker
    seen = {}

    class Model:
        def diarize(self, **kwargs):
            seen.update(kwargs)
            return [["0.00 2.00 speaker_0"]]

    out = nemotron_worker.diarize(Model(), "a.wav", {"_postprocessing_path": "/p.yaml"})
    assert seen["postprocessing_yaml"] == "/p.yaml" and out[0]["speaker"] == "SPEAKER_00"
    seen.clear()
    nemotron_worker.diarize(Model(), "a.wav", {})
    assert "postprocessing_yaml" not in seen


def test_every_documented_key_is_allowed():
    import nemotron_worker
    for key in ("chunk_len", "chunk_right_context", "fifo_len", "spkcache_len",
                "spkcache_update_period", "postprocessing", "postprocessing_yaml",
                "merge_gap", "min_duration"):
        assert key in nemotron_worker.ALLOWED_KEYS


def test_an_empty_postprocessing_block_means_the_base_values():
    from utils.nemotron_segments import POSTPROCESSING_BASE, postprocessing_params
    assert postprocessing_params({"postprocessing": {}}) == POSTPROCESSING_BASE
    assert postprocessing_params({"postprocessing": None}) is None
