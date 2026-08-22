"""Per-stage artifacts and the warnings that make them worth reading.

Run:  python -m pytest tests/test_stage_output.py -q     (from podcast-pipeline/)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.stage_output_service import StageOutputService


class _Seg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _turns(n=6, dur=10.0):
    return [_Seg(index=str(i).zfill(5), start=i * dur, end=(i + 1) * dur,
                 speaker="1" if i % 2 else "2") for i in range(n)]


def test_each_stage_writes_its_own_directory(tmp_path):
    so = StageOutputService(str(tmp_path))
    so.write_diarization(_turns(), 60.0)
    so.write_asr([_Seg(index="00000", text="xin chào")])
    so.write_manifest({"audio_file": "a.mp3"})

    assert (tmp_path / "01_diarization" / "segments.json").exists()
    assert (tmp_path / "01_diarization" / "stats.json").exists()
    assert (tmp_path / "04_asr" / "transcripts.json").exists()
    assert (tmp_path / "manifest.json").exists()


def test_waveforms_never_reach_the_json(tmp_path):
    """enhanced_audio is megabytes of float32; serialising it would be fatal."""
    seg = _Seg(index="00000", start=0.0, end=1.0, speaker="1",
               enhanced_audio=np.zeros(24000, dtype=np.float32))
    so = StageOutputService(str(tmp_path))
    so.write_diarization([seg], 1.0)

    raw = (tmp_path / "01_diarization" / "segments.json").read_text()
    assert "enhanced_audio" not in raw
    json.loads(raw)                       # and it is still valid JSON


def test_too_little_overlap_is_flagged():
    """The real regression: 0.98% overlap went unnoticed for several runs."""
    segs = _turns(10, 10.0)
    segs.append(_Seg(index="99999", start=15.0, end=15.3, speaker="2"))
    stats = StageOutputService.segment_stats(segs, 100.0)

    assert stats["overlap"]["pct_of_audio"] < 3.0
    assert any("overlap" in w for w in stats["warnings"])


def test_a_healthy_run_raises_no_warnings():
    segs, t = [], 0.0
    for i in range(20):
        spk = "1" if i % 2 else "2"
        segs.append(_Seg(index=str(i).zfill(5), start=t, end=t + 5.0, speaker=spk))
        # A backchannel from the other speaker inside each turn.
        segs.append(_Seg(index=f"b{i}", start=t + 2.0, end=t + 2.8,
                         speaker="2" if spk == "1" else "1"))
        t += 5.0
    stats = StageOutputService.segment_stats(segs, t)
    assert 3.0 <= stats["overlap"]["pct_of_audio"] <= 25.0
    assert "warnings" not in stats


def test_separation_stats_count_spans_not_segments():
    segs = [
        _Seg(index="0", start=0, end=5, speaker="1", tse=True,
             tse_spans=[(1.0, 1.4, 0.7)], tse_failed_spans=[]),
        _Seg(index="1", start=5, end=9, speaker="2", tse=True,
             tse_spans=[(6.0, 6.3, 0.6)], tse_failed_spans=[(7.0, 7.2, "qc_sim", "")]),
        _Seg(index="2", start=9, end=14, speaker="1", tse=False,
             tse_spans=[], tse_failed_spans=[]),
    ]
    st = StageOutputService.separation_stats(segs)
    assert st["segments_total"] == 3
    assert st["segments_separated"] == 2
    assert st["spans_spliced"] == 2 and st["spans_failed"] == 1
    assert st["similarity"]["p50"] > 0


def test_an_empty_track_failure_is_called_out():
    segs = [_Seg(index="0", start=0, end=1, speaker="1", tse=False, tse_spans=[],
                 tse_failed_spans=[(0.1, 0.4, "empty_track", "silent")])]
    st = StageOutputService.separation_stats(segs)
    assert any("silent" in w for w in st["warnings"])


def test_manifest_flags_segments_disappearing_between_stages(tmp_path):
    so = StageOutputService(str(tmp_path))
    so.write_diarization(_turns(6), 60.0)
    so.write_asr([_Seg(index=str(i), text="x") for i in range(3)])   # 6 -> 3
    so.write_manifest({"audio_file": "a.mp3"})

    m = json.loads((tmp_path / "manifest.json").read_text())
    assert [f["stage"] for f in m["flow"]] == ["diarization", "asr"]
    assert any("fell from 6 to 3" in w for w in m["warnings"])


def test_refinement_records_what_the_llm_rewrote(tmp_path):
    before = [_Seg(index="0", text="ban co khoe"), _Seg(index="1", text="vang")]
    after = [_Seg(index="0", text="Bạn có khỏe?"), _Seg(index="1", text="vang")]
    so = StageOutputService(str(tmp_path))
    so.write_refinement(after, before=before)

    changes = json.loads((tmp_path / "05_refinement" / "changes.json").read_text())
    assert len(changes) == 1 and changes[0]["index"] == "0"
    assert json.loads(
        (tmp_path / "05_refinement" / "stats.json").read_text())["segments_changed"] == 1


def test_disabling_stage_output_writes_nothing(tmp_path):
    so = StageOutputService(str(tmp_path), enabled=False)
    so.write_diarization(_turns(), 60.0)
    so.write_manifest({"audio_file": "a.mp3"})
    assert not list(tmp_path.iterdir())
