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


def test_rewriting_most_segments_is_not_itself_a_warning(tmp_path):
    """Vietnamese ASR arrives unpunctuated and stuttering, so a good pass edits
    nearly everything. A real run rewrote 171/200 while dropping 3.9% of the
    words -- almost all of it stutter removal."""
    # Punctuation and diacritics on most rows, a stutter removed on one of
    # them: 1 word out of 100, close to the 3.9% the real run measured.
    before = [_Seg(index=str(i), text="cai nay la mot cau day du") for i in range(20)]
    after = [_Seg(index=str(i), text="Cái này là một câu đầy đủ.") for i in range(20)]
    before[0] = _Seg(index="0", text="cai nay la la mot cau day du")
    so = StageOutputService(str(tmp_path))
    so.write_refinement(after, before=before)

    stats = json.loads((tmp_path / "05_refinement" / "stats.json").read_text())
    assert stats["segments_changed"] == 20
    assert "warnings" not in stats


def test_bulk_word_deletion_is_flagged(tmp_path):
    before = [_Seg(index=str(i), text="mot hai ba bon nam sau bay tam") for i in range(10)]
    after = [_Seg(index=str(i), text="mot hai") for i in range(10)]
    so = StageOutputService(str(tmp_path))
    so.write_refinement(after, before=before)

    stats = json.loads((tmp_path / "05_refinement" / "stats.json").read_text())
    assert stats["word_drop_pct"] > 15.0
    assert any("removed" in w for w in stats["warnings"])


def test_short_pauses_do_not_count_against_coverage():
    """merge_gap=0.3 leaves every inter-phrase pause unlabelled; a real run read
    79.1% coverage with 184 of 196 gaps under a second. That is breathing room,
    not missing speech."""
    segs, t = [], 0.0
    for i in range(60):
        segs.append(_Seg(index=str(i).zfill(5), start=t, end=t + 4.0,
                         speaker="1" if i % 2 else "2"))
        segs.append(_Seg(index=f"b{i}", start=t + 1.5, end=t + 2.1,
                         speaker="2" if i % 2 else "1"))
        t += 4.6                      # 0.6s unlabelled between every turn
    stats = StageOutputService.segment_stats(segs, t)

    assert stats["coverage_pct"] < 90.0            # would have warned before
    assert stats["unlabelled_gaps"]["short_count"] > 50
    assert stats["unlabelled_gaps"]["long_count"] == 0
    assert not any("gap" in w for w in stats.get("warnings", []))


def test_a_long_unlabelled_stretch_is_still_flagged():
    segs = [_Seg(index="0", start=0.0, end=20.0, speaker="1"),
            _Seg(index="1", start=8.0, end=9.0, speaker="2"),
            _Seg(index="2", start=60.0, end=80.0, speaker="2"),
            _Seg(index="3", start=68.0, end=69.0, speaker="1")]
    stats = StageOutputService.segment_stats(segs, 80.0)

    assert stats["unlabelled_gaps"]["long_count"] == 1
    assert stats["unlabelled_gaps"]["longest"] == 40.0
    assert any("unlabelled gaps" in w for w in stats["warnings"])


def test_disabling_stage_output_writes_nothing(tmp_path):
    so = StageOutputService(str(tmp_path), enabled=False)
    so.write_diarization(_turns(), 60.0)
    so.write_manifest({"audio_file": "a.mp3"})
    assert not list(tmp_path.iterdir())


def test_stage_artifacts_are_written_once_not_once_per_stage():
    """Under stage-major execution run() is re-entered per stage, so a later
    stage re-reads every earlier checkpoint. Rewriting their output each time
    re-emitted the same JSON, the same WAV clips and the same warnings four
    times for a single file."""
    import os
    import re

    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "services/pipeline_service.py"), encoding="utf-8").read()

    for stage, writer in (("diarization", "write_diarization"),
                          ("separation", "write_separation"),
                          ("music_removal", "write_music_removal"),
                          ("asr", "write_asr")):
        call = re.search(rf".*stage_out\.{writer}\(.*", src).group(0)
        before = src[:src.index(call)]
        assert f'if "{stage}" in computed:' in before.rsplit("\n\n", 1)[-1], (
            f"{writer} runs even when {stage} came from a checkpoint")


def test_the_manifest_accumulates_across_separate_service_instances(tmp_path):
    """Stage-major execution builds a new StageOutputService inside every run(),
    so self.manifest only ever holds the one stage that call computed. Writing
    it plain left the file describing the last stage alone -- a real two-file
    run produced a manifest listing refinement and nothing else, while all five
    stage directories held data."""
    def _segments(n):
        return [_Seg(index=str(i).zfill(5), start=i * 2.0, end=i * 2.0 + 1.8,
                     speaker="1" if i % 2 else "2") for i in range(n)]

    for writer in ("write_diarization", "write_separation", "write_music_removal"):
        service = StageOutputService(str(tmp_path))          # fresh, as run() does
        getattr(service, writer)(_segments(40), 90.0)
        service.write_manifest({"audio_file": "a.mp3"})

    service = StageOutputService(str(tmp_path))
    service.write_asr([_Seg(index=str(i), text="x") for i in range(40)])
    service.write_manifest({"audio_file": "a.mp3"})

    flow = json.loads((tmp_path / "manifest.json").read_text())["flow"]
    assert [r["stage"] for r in flow] == [
        "diarization", "separation", "music_removal", "asr"]


def test_a_recomputed_stage_replaces_its_earlier_entry(tmp_path):
    def _segments(n):
        return [_Seg(index=str(i).zfill(5), start=float(i), end=i + 0.9,
                     speaker="1") for i in range(n)]

    first = StageOutputService(str(tmp_path))
    first.write_diarization(_segments(10), 20.0)
    first.write_manifest({"audio_file": "a.mp3"})

    second = StageOutputService(str(tmp_path))
    second.write_diarization(_segments(25), 40.0)
    second.write_manifest({"audio_file": "a.mp3"})

    flow = json.loads((tmp_path / "manifest.json").read_text())["flow"]
    assert len(flow) == 1
    assert flow[0]["n"] == 25, "the rerun's count must win"


def test_a_corrupt_manifest_does_not_stop_the_run(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    service = StageOutputService(str(tmp_path))
    service.write_asr([_Seg(index="0", text="x")])
    service.write_manifest({"audio_file": "a.mp3"})

    assert json.loads((tmp_path / "manifest.json").read_text())["flow"]
