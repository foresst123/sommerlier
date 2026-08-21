"""Tests for output layout across a multi-file run.

These guard against silent data loss: the pipeline reported success for every
file while a shared output directory was quietly overwriting the previous one's
audio. A wrong path is not visible in any log line, so it needs a test.
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.segment import EnhancedSegment
from services.export_service import ExportService
from services.pipeline_service import PipelineService
from utils.batch import find_name_collisions


def _args(**kw):
    base = dict(save_path="./output", tse=True, panns=True, vad=True, dia3=False,
                merge_gap=2.0, seg_th=0.11, min_cluster_size=11, clust_th=0.5,
                LLM="case_0")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _pipeline():
    return PipelineService(*(None,) * 8)


# --- per-file isolation --------------------------------------------------

def test_each_file_gets_its_own_directory_by_default():
    p = _pipeline()
    a = p._resolve_output_dir(_args(), "/audio/talk_a.mp3")
    b = p._resolve_output_dir(_args(), "/audio/talk_b.mp3")
    assert a != b
    assert a.endswith(os.path.join("talk_a")) and b.endswith(os.path.join("talk_b"))


def test_explicit_save_path_still_isolates_each_file():
    """The regression: --save_path used to bypass the per-file directory
    entirely, so a five-file batch shared one folder."""
    p = _pipeline()
    a = p._resolve_output_dir(_args(save_path="/out"), "/audio/talk_a.mp3")
    b = p._resolve_output_dir(_args(save_path="/out"), "/audio/talk_b.mp3")
    assert a == os.path.join("/out", "talk_a")
    assert b == os.path.join("/out", "talk_b")
    assert a != b


def test_default_layout_keeps_the_parameter_suffix():
    p = _pipeline()
    out = p._resolve_output_dir(_args(), "/audio/talk.mp3")
    assert "_final" in out
    assert "-tse-True" in out and "-merge_gap-2.0" in out
    assert out.startswith(os.path.join("/audio", "_final"))


def test_changing_a_parameter_changes_the_directory():
    """Two configurations must not write into one another's results."""
    p = _pipeline()
    a = p._resolve_output_dir(_args(merge_gap=2.0), "/audio/talk.mp3")
    b = p._resolve_output_dir(_args(merge_gap=0.5), "/audio/talk.mp3")
    assert a != b


def test_save_path_none_does_not_crash():
    p = _pipeline()
    out = p._resolve_output_dir(_args(save_path=None), "/audio/talk.mp3")
    assert "_final" in out, "a missing save_path must fall back to the default layout"


# --- the collision this prevents -----------------------------------------

def test_separated_audio_of_two_files_cannot_overwrite_each_other(tmp_path):
    """export_separated_audio names files "{index}_{speaker}_separated.wav" and
    the index restarts at 00000 for every audio, so two files sharing a
    directory lose one set entirely."""
    p = _pipeline()
    svc = ExportService(logger=None)

    written = []
    for name, value in (("talk_a", 0.25), ("talk_b", 0.75)):
        out = p._resolve_output_dir(_args(save_path=str(tmp_path)), f"/audio/{name}.mp3")
        os.makedirs(out, exist_ok=True)
        seg = EnhancedSegment(index="00000", start=0.0, end=1.0, speaker="SPEAKER_00")
        seg.enhanced_audio = np.full(16000, value, dtype=np.float32)
        svc.export_separated_audio([seg], 16000, out)
        written.append(os.path.join(out, "separation", "00000_SPEAKER_00_separated.wav"))

    assert written[0] != written[1], "both files wrote to the same path"
    for path in written:
        assert os.path.exists(path), f"missing {path}"

    import soundfile as sf
    a, _ = sf.read(written[0])
    b, _ = sf.read(written[1])
    assert abs(float(a[0]) - 0.25) < 1e-3, "talk_a's audio was overwritten"
    assert abs(float(b[0]) - 0.75) < 1e-3


def test_same_stem_different_extension_is_reported_before_running():
    """talk.mp3 and talk.wav resolve to one directory; no path scheme fixes
    that, so it has to be refused up front rather than lost mid-run."""
    hits = find_name_collisions(["/audio/talk.mp3", "/audio/talk.wav", "/audio/other.mp3"])
    assert set(hits) == {"talk"}
    assert len(hits["talk"]) == 2


def test_distinct_names_report_no_collision():
    assert find_name_collisions(["/a/x.mp3", "/a/y.mp3", "/b/z.wav"]) == {}
