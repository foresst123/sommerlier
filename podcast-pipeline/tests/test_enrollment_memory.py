"""Kiểm thử bộ nhớ mẫu giọng và cách nhóm overlap nối liền.
Bộ nhớ chỉ thêm các kết quả có điểm cao. Chính sách mới không gộp
các overlap rời dù cùng hai speaker.
Chạy: python -m pytest tests/test_enrollment_memory.py -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.enrollment_memory import EnrollmentMemory

SR = 24000


def _clip(seconds=1.0, seed=0):
    return np.random.default_rng(seed).standard_normal(int(SR * seconds)) * 0.1


def _memory(**kw):
    kw.setdefault("enabled", True)
    return EnrollmentMemory(**kw)


# --- Công tắc bật bộ nhớ ---------------------------------------------------

def test_it_is_off_unless_asked_for():
    """Bộ nhớ thay đổi mẫu đối chiếu nên phải chủ động bật."""
    off = EnrollmentMemory(enabled=False)
    assert not off.offer("1", _clip(), 0.9, SR)
    assert off.extend("1", ["mined"], SR) == ["mined"]


def test_both_profiles_declare_the_setting():
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for name, profile in config["environments"].items():
        assert "enrollment_memory" in profile["models"]["bss"], name


# --- Điều kiện nhận mẫu ---------------------------------------------------

def test_a_well_separated_track_is_kept():
    memory = _memory()
    assert memory.offer("1", _clip(), 0.80, SR)


def test_a_poorly_separated_track_is_refused():
    """Mẫu dùng đối chiếu phải tốt hơn mức chỉ vừa đủ vượt QC."""
    memory = _memory()
    assert not memory.offer("1", _clip(), 0.30, SR)


def test_a_track_just_above_the_qc_gate_is_still_refused():
    """Điểm 0.25 vượt QC 0.20 nhưng chưa đủ làm mẫu đối chiếu."""
    memory = _memory()
    assert not memory.offer("1", _clip(), 0.25, SR)


def test_a_silent_track_is_refused_however_it_scored():
    memory = _memory()
    assert not memory.offer("1", np.zeros(SR), 0.95, SR)


def test_a_clip_too_short_to_embed_is_refused():
    memory = _memory()
    assert not memory.offer("1", _clip(0.2), 0.95, SR)


def test_a_missing_similarity_is_not_treated_as_zero_or_as_pass():
    """Điểm None là chưa đánh giá được, không đồng nghĩa chất lượng kém."""
    memory = _memory()
    assert not memory.offer("1", _clip(), None, SR)


# --- Cách dùng mẫu --------------------------------------------------------

def test_the_mined_enrollment_is_extended_never_replaced():
    """Mẫu ban đầu phải được giữ để không mất đường đối chiếu gốc."""
    memory = _memory()
    memory.offer("1", _clip(seed=1), 0.9, SR)
    mined = ["mined-a", "mined-b"]
    grown = memory.extend("1", mined, SR)
    assert grown[:2] == mined
    assert len(grown) == 3


def test_a_speaker_with_nothing_earned_gets_its_enrollment_back():
    memory = _memory()
    mined = ["mined"]
    assert memory.extend("2", mined, SR) is mined


def test_memory_is_per_speaker():
    memory = _memory()
    memory.offer("1", _clip(seed=2), 0.9, SR)
    assert len(memory.extend("1", [], SR)) == 1
    assert memory.extend("2", [], SR) == []


def test_the_budget_keeps_the_strongest_clips():
    """Vượt ngân sách thì giữ các mẫu có similarity cao hơn."""
    memory = _memory(budget=2.0)
    memory.offer("1", _clip(1.0, seed=3), 0.70, SR)
    memory.offer("1", _clip(1.0, seed=4), 0.95, SR)
    memory.offer("1", _clip(1.0, seed=5), 0.80, SR)
    held = memory.extend("1", [], SR)
    assert len(held) == 2, "budget should cap the stored audio"


def test_one_clip_is_kept_even_when_it_exceeds_the_budget():
    memory = _memory(budget=0.5)
    memory.offer("1", _clip(3.0, seed=6), 0.9, SR)
    assert len(memory.extend("1", [], SR)) == 1


# --- Ranh giới giữa các file ----------------------------------------------

def test_reset_clears_everything():
    """Speaker mang nhãn 1 ở file sau có thể là người khác."""
    memory = _memory()
    memory.offer("1", _clip(seed=7), 0.9, SR)
    memory.reset()
    assert memory.extend("1", [], SR) == []
    assert memory.summary()["clips"] == 0


def test_the_separation_service_clears_it_between_files():
    """reset_stats() của dịch vụ phải gọi đến bộ nhớ, không chỉ xóa bộ đếm."""
    import services.separation_service as sep

    service = sep.SeparationService.__new__(sep.SeparationService)
    service.logger = None
    service._bss_model = None
    service.model_loader = None
    service.memory = _memory()
    service.memory.offer("1", _clip(seed=8), 0.9, SR)
    service.reset_stats()
    assert service.memory.extend("1", [], SR) == []


# --- Chỉ nhóm overlap giao hoặc chạm nhau ---------------------------------

def _pair(start, end, a="1", b="2"):
    return {"overlap_start": start, "overlap_end": end,
            "overlap_duration": end - start,
            "seg1": {"speaker": a}, "seg2": {"speaker": b}}


def _groups(pairs):
    import services.separation_service as sep
    return [job[2] for job in sep.SeparationService()._group_jobs(pairs)]


def test_a_half_second_gap_keeps_targets_separate():
    """Không lấp khoảng trống giữa hai overlap dù cùng cặp speaker."""
    grouped = _groups([_pair(1427.07, 1427.57), _pair(1428.07, 1428.11)])
    assert len(grouped) == 2


def test_overlaps_far_apart_stay_separate():
    assert len(_groups([_pair(10.0, 10.5), _pair(30.0, 30.5)])) == 2


def test_touching_overlaps_keep_their_original_ranges():
    pairs = [_pair(5.0,5.4),_pair(5.4,6.2)]
    grouped = _groups(pairs)
    assert len(grouped) == 1
    assert grouped[0] == pairs


def test_fusing_does_not_mutate_the_input():
    pairs = [_pair(5.0, 5.4), _pair(5.8, 6.2)]
    _groups(pairs)
    assert pairs[0]["overlap_end"] == 5.4


def test_any_positive_gap_keeps_targets_separate():
    pairs = [_pair(5.0, 5.4), _pair(5.5, 6.2)]
    assert len(_groups(pairs)) == 2


def test_a_single_overlap_is_returned_unchanged():
    pairs = [_pair(5.0, 5.4)]
    assert _groups(pairs) == [pairs]
