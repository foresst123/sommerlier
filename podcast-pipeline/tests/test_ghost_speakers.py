"""Kiểm thử xử lý nhãn speaker giả và ghi nhận overlap không cần tách.
Trường hợp cũ trên podcast 41 phút có nhãn thứ ba chỉ chiếm 3.2 s;
nhãn này làm thiếu mẫu và khiến cửa sổ bị từ chối. Các kiểm thử bảo vệ
việc phân biệt nhãn giả với người ít nói thật.
Chạy: python -m pytest tests/test_ghost_speakers.py -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.segment_utils import merge_ghost_speakers


def _seg(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def _shares(segments):
    held = {}
    for s in segments:
        held[s["speaker"]] = held.get(s["speaker"], 0.0) + (s["end"] - s["start"])
    return held


# --- Hợp nhãn speaker giả -------------------------------------------------

def test_a_three_second_speaker_in_an_hour_is_dissolved():
    """Tái hiện tỷ lệ thời lượng của trường hợp đã đo."""
    # Nhãn giả có 2.3 s trên khoảng 41 phút hội thoại.
    segments = ([_seg(i * 10.0, i * 10.0 + 8.0, "1") for i in range(250)]
                + [_seg(i * 10.0 + 8.0, i * 10.0 + 9.5, "2") for i in range(250)]
                + [_seg(100.1, 101.3, "0"), _seg(102.2, 103.3, "0")])
    merged = merge_ghost_speakers(segments)
    assert "0" not in _shares(merged)
    assert len(merged) == len(segments), "segments are relabelled, never dropped"


def test_a_ghost_joins_whoever_was_speaking_around_it():
    segments = ([_seg(0.0, 60.0, "1"), _seg(60.0, 120.0, "2")]
                + [_seg(300.0, 360.0, "1"), _seg(360.0, 420.0, "2")]
                + [_seg(119.0, 119.6, "0")])
    merged = merge_ghost_speakers(segments)
    ghost = [s for s in merged if 119.0 <= s["start"] < 119.7][0]
    assert ghost["speaker"] in ("1", "2")


def test_total_speech_time_is_unchanged():
    segments = ([_seg(i * 10.0, i * 10.0 + 9.0, "1") for i in range(40)]
                + [_seg(i * 10.0 + 9.0, i * 10.0 + 9.8, "2") for i in range(40)]
                + [_seg(55.0, 56.0, "0")])
    before = sum(s["end"] - s["start"] for s in segments)
    after = sum(s["end"] - s["start"] for s in merge_ghost_speakers(segments))
    assert abs(before - after) < 1e-9


# --- Không hợp nhầm người thật --------------------------------------------

def test_a_quiet_but_real_participant_is_left_alone():
    """Người ít nói vẫn có lượt trọn vẹn; nhãn giả thường chỉ có mẩu rất ngắn."""
    segments = ([_seg(i * 10.0, i * 10.0 + 8.0, "1") for i in range(30)]
                + [_seg(i * 10.0 + 8.0, i * 10.0 + 9.5, "2") for i in range(30)]
                + [_seg(500.0, 512.0, "3")])          # Một lượt dài 12 s
    assert "3" in _shares(merge_ghost_speakers(segments))


def test_two_speakers_are_never_reduced_to_one():
    """Chỉ có hai người thì người nói ít vẫn là thành viên hội thoại."""
    segments = [_seg(0.0, 600.0, "1"), _seg(600.0, 600.9, "2")]
    assert set(_shares(merge_ghost_speakers(segments))) == {"1", "2"}


def test_a_balanced_conversation_is_untouched():
    segments = [_seg(i * 10.0, i * 10.0 + 9.0, str(i % 3)) for i in range(60)]
    assert merge_ghost_speakers(segments) == sorted(segments, key=lambda s: s["start"])


def test_an_empty_list_is_returned_as_is():
    assert merge_ghost_speakers([]) == []


# --- Kiểm tra người thứ ba trong cửa sổ -----------------------------------

def _service():
    import services.separation_service as sep
    svc = sep.SeparationService.__new__(sep.SeparationService)
    svc.logger = None
    svc.music_map = None
    return svc








# --- Overlap không cần tách vẫn phải được ghi nhận -------------------------

def test_a_speaker_overlapping_themselves_is_recorded_not_dropped():
    """Hợp nhãn giả với người bên cạnh có thể tạo overlap cùng speaker.
    Không cần tách hai nguồn nhưng phải ghi bss_spans hoặc bss_failed_spans;
    không được continue rồi mất dấu vùng đó."""
    import numpy as np

    from schemas.audio import AudioData
    from schemas.segment import Segment
    import services.separation_service as sep

    class Stub:
        def separate_two_speakers(self, mixture, *a, **k):
            n = len(mixture)
            return (np.zeros(n), np.zeros(n), 0.6, 0.6,
                    {"anchor_self": 0.6, "anchor_other": 0.1, "other_rms": 0.1})

        def reset_speakers(self):
            pass

    segments = [Segment(index="00001", start=0.0, end=5.0, speaker="A"),
                Segment(index="00002", start=4.0, end=9.0, speaker="A")]
    service = sep.SeparationService(Stub(), logger=None)
    audio = AudioData(name="t", waveform=np.zeros(24000 * 10, dtype=np.float32),
                      sample_rate=24000, duration=10.0, audio_segment=None)

    out = service.process_overlaps(segments, audio, overlap_threshold=0.1)
    spliced = sum(len(s.bss_spans) for s in out)
    failed = [f for s in out for f in s.bss_failed_spans]
    # same_speaker giờ ghi vào bss_spans (sim=-2 passthrough), không vào
    # bss_failed_spans. Quan trọng là vùng không bị mất hoàn toàn.
    assert spliced > 0 or len(failed) > 0, "the overlap vanished from both lists"


def test_fusing_sorts_before_merging():
    """Nhóm phải sắp xếp trước để danh sách đầu vào lộn thứ tự không làm mất
    các overlap xuất hiện sớm hơn."""
    import services.separation_service as sep

    def pair(start, end):
        return {"overlap_start": start, "overlap_end": end,
                "overlap_duration": end - start,
                "seg1": {"speaker": "1"}, "seg2": {"speaker": "2"}}

    fused = [job[2][0] for job in sep.SeparationService()._group_jobs(
        [pair(5.0, 5.2), pair(1.0, 1.2)])]
    assert len(fused) == 2
    assert fused[0]["overlap_start"] == 1.0
