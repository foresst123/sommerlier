"""Kiểm tra cửa sổ mới bằng âm thanh tổng hợp; không tải model/GPU."""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas.segment import Segment
from utils.separation_window import WindowPlanner, AcousticCuts, clean_segments, Piece
from algorithms.diarization.overlap import detect_overlapping_segments
from services.separation_service import SeparationService

SR = 1000


def segments(rows):
    return [Segment(str(i), a, b, speaker) for i, (a, b, speaker) in enumerate(rows)]


def waveform(duration=80):
    # Cụm lời cách nhau bằng khoảng nghỉ 80 ms để có điểm cắt tự nhiên.
    t = np.arange(duration*SR)/SR
    wave = (0.1*np.sin(2*np.pi*83*t)).astype(np.float32)
    wave[(t % 0.5) < 0.08] = 0
    return wave


def setup(rows, **kwargs):
    segs = segments(rows)
    pairs = detect_overlapping_segments([s.__dict__ for s in segs], overlap_threshold=0)
    planner = WindowPlanner(segs, pairs, waveform(), SR, **kwargs)
    svc = SeparationService()
    return planner, svc._group_jobs(pairs)


def basic():
    return [(10,30,"A"), (20,21,"B"), (0,8,"A"), (40,48,"B")]


def test_support_excludes_entire_segment_for_any_positive_intersection():
    segs = segments([(0,8,"A"), (7.999999,9,"B"), (10,12,"A"),
                     (12,14,"B"), (20,22,"A"), (21.999999,23,"A")])
    assert [s.index for s in clean_segments(segs)] == ["2","3"]


def test_grouping_never_bridges_a_positive_gap_and_preserves_sources():
    planner, jobs = setup([(0,30,"A"),(10,11,"B"),(11,12,"B"),(12.00001,13,"B")])
    assert [len(j[2]) for j in jobs] == [2,1]
    assert jobs[0][2][0]["seg2"]["index"] != jobs[0][2][1]["seg2"]["index"]


def test_window_anchor_support_crossfade_and_exact_core_mapping():
    planner, jobs = setup(basic())
    before = planner.waveform.copy()
    result = planner.build(jobs[0][2])
    assert result is not None, planner.detail
    # The module's own band is final_core_min..final_core_max = 5..8s. The
    # assertion used to read 6, which matched an anchor that aimed at 6.5 and
    # so starved the right context -- every second of core position past 5 is
    # a second the right side cannot have.
    assert 5 <= result.core[0]/SR <= 8
    left, right = result.layout["context_seconds"]
    assert right >= 2.0, f"right context starved: {left:.2f}s left vs {right:.2f}s right"
    assert len(result.audio) <= 15*SR
    assert np.array_equal(result.audio[slice(*result.core)],before[20*SR:21*SR])
    assert np.array_equal(planner.waveform,before)
    pieces = result.layout["pieces"]
    for p in pieces:
        if p["kind"] == "support":
            assert p["source_segment"] in {"2","3"}
            assert p["source_samples"][1]-p["source_samples"][0] >= 1.5*SR
    expected = sum(p["source_samples"][1]-p["source_samples"][0] for p in pieces)
    assert len(result.audio) == expected-(len(pieces)-1)*20
    assert result.probes["A"] and result.probes["B"]
    for p in pieces[1:]:
        a = p["window_samples"][0]
        for probes in result.probes.values():
            assert all(b <= a or c >= a+20 for c,b in probes)


def test_short_support_is_not_added_or_replaced_with_silence():
    planner, _ = setup(basic())
    base = Piece(14*SR,23*SR,"A","0","base")
    assert planner._sequences("B",20*SR,0,round(1.4*SR),base) == [()]


def test_long_support_is_cut_at_a_pause():
    planner, _ = setup(basic())
    choices = planner._support("B",20*SR)
    assert any(p.end-p.start < 8*SR and "energy" in (p.start_cut,p.end_cut)
               for p in choices)
    assert all(p.end-p.start >= 1.5*SR for p in choices)


def test_disconnected_overlap_is_a_wall_even_for_the_same_pair():
    planner, jobs = setup(basic()+[(23,24,"B")])
    result = planner.build(jobs[0][2])
    assert result is not None, planner.detail
    base = next(p for p in result.layout["pieces"] if p["kind"] == "base")
    assert base["source_samples"][1] <= 23*SR


def test_connected_secondary_overlap_must_clear_original_and_retained_edges():
    planner, jobs = setup(basic()+[(21,22,"B")])
    result = planner.build(jobs[0][2])
    assert result is not None, planner.detail
    base = next(p for p in result.layout["pieces"] if p["kind"] == "base")
    assert 21*SR-base["source_samples"][0] >= SR
    assert base["source_samples"][1]-22*SR >= SR
    planner, jobs = setup([(10,22.5,"A"),(20,21,"B"),(21,22,"B"),(0,8,"A"),(40,48,"B")])
    assert planner.build(jobs[0][2]) is None
    assert planner.detail == "secondary_overlap_near_original_edge"


def test_third_speaker_inside_target_is_rejected():
    planner, jobs = setup(basic()+[(20.2,20.4,"C")])
    ab = next(j for j in jobs if j[:2] == ("A","B"))
    assert planner.build(ab[2]) is None
    assert planner.reason == "multi_speaker"


def test_expansion_stops_at_another_speaker():
    planner, jobs = setup([(19,24,"A"),(20,21,"B"),(17,18.5,"C"),(0,8,"A"),(40,48,"B")])
    result = planner.build(jobs[0][2])
    assert result is not None, planner.detail
    base = next(p for p in result.layout["pieces"] if p["kind"] == "base")
    assert base["source_samples"][0] >= 18.5*SR
    assert result.layout["context_seconds"][0] <= 1.5


def test_no_pause_does_not_authorize_arbitrary_interior_cuts():
    wave = np.ones(5000,dtype=np.float32)*0.1
    cuts,_,method = AcousticCuts(wave,SR).analyse(0,5000)
    assert set(cuts) == {0,5000}
    assert method == "energy"


def test_silero_result_is_cached_and_failure_uses_energy():
    class VAD:
        calls = 0
        def get_speech_timestamps(self,wave,sampling_rate):
            self.calls += 1
            return [{"start":100,"end":900}]
    vad = VAD()
    finder = AcousticCuts(waveform(2),SR,vad)
    a = finder.analyse(0,1000)
    assert finder.analyse(0,1000) is a
    assert vad.calls == 1 and a[2] == "silero"
    class Broken:
        def get_speech_timestamps(self,*args,**kwargs):
            raise RuntimeError("offline")
    assert AcousticCuts(waveform(2),SR,Broken()).analyse(0,1000)[2] == "energy"


def test_connected_splice_targets_are_merged_per_original_segment():
    _,jobs = setup([(10,30,"A"),(20,21.5,"B"),(21,22,"B")])
    spans = SeparationService._splice_pairs(jobs[0][2])
    a = [(lo,hi) for s,lo,hi in spans if s["speaker"] == "A"]
    assert a == [(20,22)]
    assert len(spans) == 3


def test_long_target_and_recording_seam_fail_explicitly():
    planner,jobs = setup([(10,40,"A"),(20,30,"B")])
    assert planner.build(jobs[0][2]) is None
    assert planner.detail == "overlap_does_not_fit_15s"
    planner,jobs = setup(basic(),seams=[20.5])
    assert planner.build(jobs[0][2]) is None
    assert planner.detail == "target_crosses_recording_seam"


def test_checkpoint_namespace_preserves_old_results(tmp_path):
    from utils.checkpoint import CheckpointManager
    old = CheckpointManager(str(tmp_path),"job")
    old.save("separation",["old"])
    new = CheckpointManager(str(tmp_path),"job")
    new.namespaces = {"separation":"new-policy"}
    assert not new.exists("separation")
    new.save("separation",["new"])
    assert new.load("separation") == ["new"]
    assert old.load("separation") == ["old"]


# --- hồi quy cho ba lỗi cửa sổ (core nở, kẹp floor, trần 15s) --------------

def _sparse_waveform(duration=80, quiet=((15.4, 15.6), (42.4, 42.6))):
    """Âm liên tục, chỉ có vài điểm lặng rời rạc.

    Giữ cho `AcousticCuts` chỉ tìm được đúng mấy điểm cắt đó, để kiểm tra
    nhánh fallback khi vật liệu cắt ở xa core.
    """
    t = np.arange(duration * SR) / SR
    wave = (0.1 * np.sin(2 * np.pi * 83 * t)).astype(np.float32)
    for a, b in quiet:
        wave[int(a * SR):int(b * SR)] = 0
    return wave


def test_expanded_core_still_reports_the_real_overlap_position():
    """core_source_samples phải là mốc duy nhất để ánh xạ ngược về nguồn.

    Với overlap dưới 0.2 s, core nở ±2 s. Ai ánh xạ bằng overlap gốc thay vì
    bằng core_source_samples sẽ lệch đúng phần đã nở -- đó là lỗi ghép trả
    sớm 2 giây ở separation_service.
    """
    planner, jobs = setup([(10, 30, "A"), (20, 20.15, "B"), (0, 8, "A"), (40, 48, "B")])
    result = planner.build(jobs[0][2])
    assert result is not None, planner.detail
    layout = result.layout
    assert layout["core_expanded"] is True
    base = next(p for p in layout["pieces"] if p["kind"] == "base")
    core_src_lo = layout["core_source_samples"][0]
    # core[0] là vị trí của core_source_samples[0] trong cửa sổ, không phải
    # của overlap gốc.
    assert result.core[0] == base["window_samples"][0] + core_src_lo - base["source_samples"][0]
    assert core_src_lo < int(20.0 * SR), "core phải nở về bên trái"


def test_short_overlap_does_not_fail_where_a_longer_one_passes():
    """Nở core không được ăn hết chỗ mà base cần.

    Cùng hình học, chỉ khác độ dài overlap: ca ngắn (bị nở) từng fail với
    no_safe_left_cut_for_3_10s_base trong khi ca dài lại dựng được.
    """
    rows = [(19, 24, "A"), (0, 8, "A"), (40, 48, "B"), (17, 18.5, "C")]
    planner_long, jobs_long = setup(rows + [(20, 20.5, "B")])
    long_job = next(j for j in jobs_long if j[:2] == ("A", "B"))
    assert planner_long.build(long_job[2]) is not None, planner_long.detail

    planner_short, jobs_short = setup(rows + [(20, 20.15, "B")])
    short_job = next(j for j in jobs_short if j[:2] == ("A", "B"))
    assert planner_short.build(short_job[2]) is not None, planner_short.detail


def test_pad_less_fallback_never_exceeds_the_15s_target():
    """Nhánh fallback cũng phải tôn trọng trần 15 s như đường chính.

    Khi điểm cắt sạch nằm xa core, fallback từng ghép ra cửa sổ 27 s và đưa
    thẳng vào model vốn thiết kế cho 15 s.
    """
    segs = segments([(5, 55, "A"), (25, 33, "B")])
    pairs = detect_overlapping_segments([s.__dict__ for s in segs], overlap_threshold=0)
    planner = WindowPlanner(segs, pairs, _sparse_waveform(), SR)
    jobs = SeparationService()._group_jobs(pairs)
    result = planner.build(jobs[0][2])
    if result is None:
        # Từ chối hẳn còn hơn đưa cửa sổ quá khổ vào model; lý do phải nói rõ
        # là vượt ngân sách, không lẫn với "không có support sạch".
        assert planner.detail == "fallback_base_exceeds_15s"
    else:
        assert result.layout["duration_seconds"] <= 15.0, result.layout
