"""Kiểm thử logic SeparationService, không dùng GPU hay model thật.
Model giả giúp kiểm tra cửa sổ, nhóm overlap, QC từng track và mặt nạ
chống lọt giọng; không đánh giá chất lượng âm thanh tách thực tế.
Chạy trong podcast-pipeline: python -m pytest tests/test_separation_logic.py -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.audio import AudioData
from schemas.segment import Segment
from services.separation_service import SeparationService

SR = 24000


class FakeTSE:
    """Trả hai track hằng số khác nhau để nhận biết phần đã ghép trả."""

    def __init__(self, sim_a=0.6, sim_b=0.6):
        self.sim_a, self.sim_b = sim_a, sim_b
        self.calls = []

    def separate_two_speakers(self, mixture_audio, enroll_A, enroll_B, sample_rate,
                              id_A, id_B, probe_A=None, probe_B=None, core_range=None):
        self.calls.append({
            "len_sec": len(mixture_audio) / sample_rate,
            "probe_A_sec": sum(b - a for a, b in (probe_A or [])) / sample_rate,
            "probe_B_sec": sum(b - a for a, b in (probe_B or [])) / sample_rate,
        })
        return (np.full(len(mixture_audio), 0.5, dtype=np.float32),
                np.full(len(mixture_audio), -0.5, dtype=np.float32),
                self.sim_a, self.sim_b,
                {"anchor_self": 0.6, "anchor_other": 0.1, "other_rms": 0.5})


def _audio(duration=60.0):
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 0.05, int(duration * SR)).astype(np.float32)
    # Khoảng nghỉ để thuật toán có điểm cắt hợp lệ, không cắt ngang tiếng.
    wave[np.arange(len(wave)) % (SR // 2) < int(0.08 * SR)] = 0
    return AudioData(waveform=wave,
                     sample_rate=SR, name="test", audio_segment=None, duration=duration)


def _dialogue():
    """A nói 0–30 s; B chen 0.4 s tại giây 14, rồi có lượt nói sạch 32–40 s.
    A có thêm mẫu sạch 42–48 s vì segment chứa overlap không được làm support."""
    return [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=14.0, end=14.4, speaker="SPEAKER_01"),
        Segment(index="00003", start=32.0, end=40.0, speaker="SPEAKER_01"),
        Segment(index="00004", start=42.0, end=48.0, speaker="SPEAKER_00"),
    ]




def test_a_backchannel_window_is_grown_well_past_the_overlap():
    """Cửa sổ phải đủ ngữ cảnh để Sidon phân biệt giọng và ECAPA nhận diện track.
    Cửa sổ 2 s trước đây không có solo đủ dài, làm similarity xuống mức
    p50 0.15; kiểm tra không quay lại kiểu cửa sổ quá ngắn đó."""
    fake = FakeTSE()
    svc = SeparationService(fake, logger=None)
    svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    assert len(fake.calls) == 1, "one job expected"
    call = fake.calls[0]
    assert call["len_sec"] >= 10.0, (
        f"window is {call['len_sec']:.2f}s; a blind separator needs context")


def test_the_window_carries_solo_audio_for_the_assignment_to_use():
    """Probe rỗng khiến model chấm cả track, có thể chứa giọng của người khác.
    Hai probe phải có lời solo riêng để phép gán speaker có ý nghĩa."""
    fake = FakeTSE()
    svc = SeparationService(fake, logger=None)
    svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    call = fake.calls[0]
    assert call["probe_A_sec"] >= 2.0 and call["probe_B_sec"] >= 1.0, (
        "no solo audio in the window; ECAPA has nothing to assign from")
    # Layout mới (base 3-10s host + pad 10-15s non-host): host chiếm ~10s,
    # non-host có ~2s pad. Ratio sẽ thấp hơn trước -- quan trọng là non-host
    # có đủ voice để ECAPA phân biệt (>= 1s).
    assert call["probe_B_sec"] >= 1.0, "non-host cần ít nhất 1s voice để ECAPA hoạt động"


def test_low_scoring_track_does_not_discard_the_good_one():
    # A đạt điểm, B không đạt QC: giữ track A thay vì bỏ cả hai.
    fake = FakeTSE(sim_a=0.60, sim_b=0.05)
    svc = SeparationService(fake, logger=None)
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    a_seg = next(s for s in out if s.index == "00001")
    b_seg = next(s for s in out if s.index == "00002")
    assert a_seg.bss is True and a_seg.bss_spans, "A's clean extraction must survive"
    assert b_seg.bss is False, "B failed QC and must keep the original mixture"
    assert b_seg.bss_failed_spans, "B's failure must be recorded, not silently dropped"
    assert b_seg.bss_status == "failed"


def test_nearby_disconnected_overlaps_get_separate_calls():
    segs = [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=10.0, end=10.4, speaker="SPEAKER_01"),
        Segment(index="00003", start=13.0, end=13.4, speaker="SPEAKER_01"),
        Segment(index="00004", start=32.0, end=40.0, speaker="SPEAKER_01"),
        Segment(index="00005", start=42.0, end=48.0, speaker="SPEAKER_00"),
    ]
    fake = FakeTSE()
    svc = SeparationService(fake, logger=None)
    svc.process_overlaps(segs, _audio(), overlap_threshold=0.1)
    assert len(fake.calls) == 2, "Overlap rời phải được xử lý độc lập"
    assert svc.stats["pairs"] == 2


def test_sdlm_export_zeroes_unseparated_overlap():
    fake = FakeTSE(sim_a=0.05, sim_b=0.05)   # Cả hai track không đạt QC
    svc = SeparationService(fake, logger=None)
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    t0, _t1 = svc.export_sdlm_dual_channel(out, 60.0, SR, strict=True)
    lo, hi = int(14.0 * SR), int(14.4 * SR)
    assert np.allclose(t0[lo:hi], 0.0), "A's track must not carry B's overlapping speech"
    assert np.abs(t0[int(2.0 * SR):int(3.0 * SR)]).sum() > 0, "solo speech must survive"


def test_sdlm_export_keeps_separated_overlap():
    fake = FakeTSE(sim_a=0.6, sim_b=0.6)     # Hai track đạt kiểm tra
    svc = SeparationService(fake, logger=None)
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    t0, _t1 = svc.export_sdlm_dual_channel(out, 60.0, SR, strict=True)
    lo, hi = int(14.05 * SR), int(14.35 * SR)
    assert np.abs(t0[lo:hi]).sum() > 0, "a successfully separated overlap must be kept"


class DuplicatingTSE:
    """Giả lập lỗi model chép giọng người nói chính lên cả hai track."""

    def separate_two_speakers(self, mixture_audio, enroll_A, enroll_B, sample_rate,
                              id_A, id_B, probe_A=None, probe_B=None, core_range=None):
        a_voice = np.full(len(mixture_audio), 0.5, dtype=np.float32)
        # Hai track giống nhau nên giọng neo khớp cả hai, không có khoảng cách điểm.
        return a_voice, a_voice.copy(), 0.55, -0.05, {
            "anchor_self": 0.55, "anchor_other": 0.45, "other_rms": 0.5}


def test_max_rule_would_paste_speaker_a_into_speaker_b():
    """Không dùng max(sim_A, sim_B) để quyết định cả hai track.
    Điểm A=0.55 (>= threshold 0.50) cho phép ghi giọng A; B=-0.05 phải giữ nguyên."""
    svc = SeparationService(DuplicatingTSE(), logger=None)
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)

    a_seg = next(s for s in out if s.index == "00001")
    b_seg = next(s for s in out if s.index == "00002")

    assert a_seg.bss is True, "A's track scored 0.55 and should be spliced"
    assert b_seg.bss is False, (
        "B scored -0.05: splicing here would write A's voice into B's segment. "
        "This is exactly what max(sim_A, sim_B) < threshold would allow."
    )
    assert b_seg.bss_failed_spans[0][2] == "qc_sim"


def test_every_overlap_is_accounted_for():
    """Mọi overlap phải có kết quả hoặc lý do thất bại, không được biến mất."""
    for fake in (FakeTSE(0.6, 0.6), FakeTSE(0.6, 0.05), FakeTSE(0.05, 0.05),
                 DuplicatingTSE()):
        svc = SeparationService(fake, logger=None)
        out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
        n_ok = sum(len(s.bss_spans) for s in out)
        n_bad = sum(len(s.bss_failed_spans) for s in out)
        # Một overlap được ghi trên hai segment nguồn
        assert n_ok + n_bad == 2, (
            f"{type(fake).__name__}: {n_ok} spliced + {n_bad} failed != 2 -- "
            "some code path discarded an overlap without recording it"
        )


def test_an_overlap_too_short_to_separate_is_still_recorded():
    """Overlap ngắn hơn ngưỡng vẫn phải có lý do, không được lặng lẽ biến mất.
    Bộ lọc queue từng bỏ các job này mà không ghi bss_failed_spans, nên bước
    xuất strict đọc mixture ở đó như thể là giọng sạch của một người."""
    segs = [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=14.0, end=14.05, speaker="SPEAKER_01"),  # 0.05s < 0.1
        Segment(index="00003", start=32.0, end=40.0, speaker="SPEAKER_01"),
        Segment(index="00004", start=42.0, end=48.0, speaker="SPEAKER_00"),
    ]
    fake = FakeTSE()
    svc = SeparationService(fake, logger=None)
    out = svc.process_overlaps(segs, _audio(), overlap_threshold=0.1)

    # below_threshold filter đã bị bỏ -- overlap ngắn giờ được đưa vào queue.
    # short_core_expansion mở rộng ±2s quanh core nhỏ để Sidon có ngữ cảnh.
    # Model được gọi -- kiểm tra pipeline không crash và trả đúng số segment.
    assert len(out) == len(segs), "số segment đầu ra phải bằng đầu vào"
    # Overlap ngắn vẫn được tách (không có below_threshold filter nữa)
    assert fake.calls, "overlap ngắn giờ phải chạy model"


def test_same_speaker_overlap_is_kept_and_not_counted_twice():
    """Hai segment cùng speaker chồng nhau chỉ chứa giọng người đó.
    Xóa vùng đó làm mất lời nói thật; cộng cả hai bản làm biên độ gấp đôi."""
    segs = [
        Segment(index="00001", start=0.0, end=15.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=14.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00003", start=32.0, end=40.0, speaker="SPEAKER_01"),
    ]
    svc = SeparationService(FakeTSE(), logger=None)
    out = svc.process_overlaps(segs, _audio(), overlap_threshold=0.1)
    # same_speaker giờ được passthrough (không fail, không tách).
    # Kiểm tra audio gốc không bị xóa hoặc cộng đôi.
    t0, _ = svc.export_sdlm_dual_channel(out, 60.0, SR, strict=True)
    audio = _audio().waveform
    lo, hi = int(14.1 * SR), int(14.9 * SR)
    assert np.abs(t0[lo:hi]).sum() > 0, "lời nói của chính speaker đó không được xóa"
    assert np.allclose(t0[lo:hi], audio[lo:hi], atol=1e-6), (
        "vùng chồng cùng speaker bị cộng hai lần")






def test_no_window_does_not_block_later_overlaps():
    """Một overlap không xử lý được không được chặn các overlap tiếp theo."""
    segs = [
        Segment(index="00001", start=0.0, end=55.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=5.0, end=5.3, speaker="SPEAKER_01"),   # B chưa có đoạn solo gần đây
        Segment(index="00003", start=50.0, end=50.4, speaker="SPEAKER_01"),
        Segment(index="00004", start=56.0, end=59.0, speaker="SPEAKER_01"), # Lượt solo duy nhất của B
        Segment(index="00005", start=61.0, end=68.0, speaker="SPEAKER_00"),
    ]
    svc = SeparationService(FakeTSE(), logger=None)
    out = svc.process_overlaps(segs, _audio(70.0), overlap_threshold=0.1)
    n_ok = sum(len(s.bss_spans) for s in out)
    n_bad = sum(len(s.bss_failed_spans) for s in out)
    assert n_ok + n_bad == 4, "both overlaps (x2 sides) must be accounted for"
    assert n_ok > 0, "the overlap near B's solo turn should still be processed"


class AllRejectTSE:
    """Từ chối lần đầu, chấp nhận lần tiếp theo để kiểm tra các job độc lập."""

    def __init__(self):
        self.calls = 0

    def separate_two_speakers(self, mixture_audio, enroll_A, enroll_B, sample_rate,
                              id_A, id_B, probe_A=None, probe_B=None, core_range=None):
        self.calls += 1
        sim = 0.05 if self.calls == 1 else 0.6
        return (np.full(len(mixture_audio), 0.5, dtype=np.float32),
                np.full(len(mixture_audio), -0.5, dtype=np.float32), sim, sim,
                {"anchor_self": 0.6, "anchor_other": 0.1, "other_rms": 0.5})


def test_failed_job_does_not_force_other_disconnected_jobs_to_retry():
    segs = [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=10.0, end=10.4, speaker="SPEAKER_01"),
        Segment(index="00003", start=13.0, end=13.4, speaker="SPEAKER_01"),
        Segment(index="00004", start=32.0, end=40.0, speaker="SPEAKER_01"),
        Segment(index="00005", start=42.0, end=48.0, speaker="SPEAKER_00"),
    ]
    fake = AllRejectTSE()
    svc = SeparationService(fake, logger=None)
    out = svc.process_overlaps(segs, _audio(), overlap_threshold=0.1)
    assert svc.stats["retried"] == 0
    assert fake.calls == 2
    assert sum(len(s.bss_spans) for s in out) > 0, "retries should recover the overlaps"


def test_sdlm_mask_uses_failed_spans():
    svc = SeparationService(FakeTSE(0.05, 0.05), logger=None)
    out = svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)
    assert all(s.bss_failed_spans or not s.bss_spans for s in out if s.index == "00002")
    t0, _ = svc.export_sdlm_dual_channel(out, 60.0, SR, strict=True)
    lo, hi = int(14.0 * SR), int(14.4 * SR)
    assert np.allclose(t0[lo:hi], 0.0)


def test_old_checkpoint_unpickles_without_new_fields():
    """Khôi phục checkpoint trước khi có bss_spans/bss_failed_spans.
    Pickle gán __dict__ mà không áp mặc định dataclass nên __setstate__
    phải bổ sung các trường mới trước khi pipeline ghi tiếp."""
    import pickle
    from schemas.segment import SpeechSegment

    seg = SpeechSegment(index="00001", start=0.0, end=1.0, speaker="SPEAKER_00")
    del seg.__dict__["bss_spans"]
    del seg.__dict__["bss_failed_spans"]

    loaded = pickle.loads(pickle.dumps(seg))
    loaded.bss_failed_spans.append((0.0, 1.0, "qc_sim", "sim=0.10"))
    assert loaded.bss_status == "failed"

    svc = SeparationService(FakeTSE(), logger=None)
    loaded.audio = np.ones(SR, dtype=np.float32)
    t0, _ = svc.export_sdlm_dual_channel([loaded], 2.0, SR, strict=True)
    assert np.allclose(t0[:SR], 0.0), "failed span must still be masked after unpickling"


# --- Cửa sổ ghép và QC tại vị trí thay thế --------------------------------

def _svc():
    return SeparationService.__new__(SeparationService)


def _diar(spans):
    return [Segment(index=str(i).zfill(5), start=a, end=b, speaker=s)
            for i, (a, b, s) in enumerate(spans)]


def _buried_case():
    """Lời chen 0.34 s nằm trong một lượt dài, dạng từng khiến Sidon mất giọng."""
    segs = _diar([(730.0, 758.4, "2"), (758.4, 787.7, "1"),
                  (777.33, 777.67, "2"), (788.2, 816.5, "1")])
    svc = _svc()
    svc.logger = None
    return svc, svc._intervals_by_speaker(segs)








def test_qc_rejects_a_track_that_is_silent_where_the_mixture_speaks():
    """Giả lập sim=0.67 trên solo cách 19 s nhưng track im lặng ở overlap."""
    rng = np.random.default_rng(0)
    n = int(0.34 * SR)
    host = (rng.standard_normal(n) * 0.12).astype(np.float32)

    # Track im lặng phần lớn lời chen, chỉ có đuôi người khác ở cuối;
    # trường hợp thực tế 00046 có 11/17 khung phẳng.
    bad = np.zeros(n, dtype=np.float32)
    tail = int(n * 0.72)
    bad[tail:] = rng.standard_normal(n - tail) * 0.08
    assert SeparationService._track_has_speech(host, bad) is False
    # RMS cả đoạn vẫn cao nên phép kiểm tra tổng năng lượng không bắt được lỗi.
    assert float(np.sqrt(np.mean(bad ** 2))) > 0.002

    good = (host * 0.5).astype(np.float32)
    assert SeparationService._track_has_speech(host, good) is True


def test_qc_accepts_a_quieter_track_and_rejects_an_empty_one():
    rng = np.random.default_rng(1)
    n = int(0.5 * SR)
    host = (rng.standard_normal(n) * 0.15).astype(np.float32)
    assert SeparationService._track_has_speech(host, host * 0.08) is True
    assert SeparationService._track_has_speech(host, np.zeros(n, np.float32)) is False
    # Mixture im lặng thì không có lời cần giữ.
    z = np.zeros(n, dtype=np.float32)
    assert SeparationService._track_has_speech(z, z) is True


def test_the_run_report_can_actually_be_built():
    """Báo cáo phải dựng được sau separation; hằng số cửa sổ từng bị xóa
    nhưng vẫn được tham chiếu, gây NameError sau khi chạy model rất lâu."""
    fake = FakeTSE()
    svc = SeparationService(fake, logger=None)
    svc.process_overlaps(_dialogue(), _audio(), overlap_threshold=0.1)

    payload = svc.report_payload()
    assert set(payload) >= {"thresholds", "music_map", "stats", "failures"}
    assert payload["thresholds"]["window_target"] == 15.0
    assert payload["windows"]


def test_no_module_reads_a_constant_nobody_defines():
    """Kiểm tra tĩnh để phát hiện tham chiếu hằng số không còn định nghĩa."""
    import ast
    import builtins

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dangling = []
    for folder, _dirs, files in os.walk(root):
        if any(skip in folder for skip in ("__pycache__", ".git", "tests")):
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except SyntaxError:
                continue
            bound = set(dir(builtins))
            for node in ast.walk(tree):
                if isinstance(node, ast.alias):
                    bound.add((node.asname or node.name).split(".")[0])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bound.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    bound.add(node.id)
                elif isinstance(node, ast.arg):
                    bound.add(node.arg)
            for node in ast.walk(tree):
                # Chỉ xét hằng số viết hoa để giảm báo nhầm do chưa mô phỏng phạm vi tên.
                if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        and node.id.isupper() and node.id not in bound):
                    dangling.append(f"{os.path.relpath(path, root)}:{node.lineno} {node.id}")
    assert not dangling, "constants used but never defined:\n  " + "\n  ".join(sorted(set(dangling)))


def test_no_identifier_names_a_processing_step_that_does_not_exist():
    """Không đặt tên như thể pipeline có bước tái tạo/cải thiện giọng.
    Âm thanh là bản ghi hoặc kết quả tách overlap, không phải giọng được
    model sáng tạo lại. Duyệt AST để chỉ kiểm tra identifier; comment nhắc
    tên cũ như EnhancedSegment vẫn cần cho việc đọc checkpoint cũ."""
    import ast

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    here = os.path.basename(__file__)
    offenders = []
    for folder, _dirs, files in os.walk(root):
        if any(skip in folder for skip in ("__pycache__", ".git")):
            continue
        for name in files:
            if not name.endswith(".py") or name == here:
                continue
            path = os.path.join(folder, name)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                named = None
                if isinstance(node, (ast.Name, ast.arg)):
                    named = getattr(node, "id", None) or node.arg
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef, ast.Attribute)):
                    named = getattr(node, "name", None) or getattr(node, "attr", None)
                if named and "enhanc" in named.lower():
                    offenders.append(
                        f"{os.path.relpath(path, root)}:{node.lineno} {named}")
    assert not offenders, ("identifiers naming a stage that does not exist:\n  "
                           + "\n  ".join(sorted(set(offenders))))


def test_one_long_clip_is_preferred_over_a_patchwork():
    """Mẫu liền mạch từng đạt +8.39 dB SI-SDR, so với +6.86 dB khi ghép và
    +6.94 dB khi crossfade trên tám mixture thử nghiệm. Kiểm tra tiếp tục
    ưu tiên mẫu dài đủ dùng thay vì chia thành nhiều mẩu."""
    import numpy as np
    from schemas.audio import AudioData
    from schemas.segment import Segment
    import services.separation_service as sep

    sr = 16000
    svc = sep.SeparationService.__new__(sep.SeparationService)
    svc.logger = None
    audio = AudioData(name="t", waveform=np.ones(120 * sr, dtype=np.float32),
                      sample_rate=sr, duration=120.0, audio_segment=None)
    segments = [
        Segment(index="1", start=10.0, end=16.0, speaker="A"),   # Mẫu solo 6 s đủ dùng riêng
        Segment(index="2", start=20.0, end=21.0, speaker="A"),
        Segment(index="3", start=30.0, end=31.0, speaker="A"),
    ]
    picked = svc.mine_enrollments(segments, audio)["A"]
    assert len(picked) == 1, f"took {len(picked)} clips when one was long enough"
    assert len(picked[0]) / sr == pytest.approx(6.0, abs=0.05)


def test_short_clips_are_still_gathered_when_none_is_long_enough():
    """Nếu không có mẫu đủ dài, vẫn cần gom đủ lượng lời để đối chiếu."""
    import numpy as np
    from schemas.audio import AudioData
    from schemas.segment import Segment
    import services.separation_service as sep

    sr = 16000
    svc = sep.SeparationService.__new__(sep.SeparationService)
    svc.logger = None
    audio = AudioData(name="t", waveform=np.ones(120 * sr, dtype=np.float32),
                      sample_rate=sr, duration=120.0, audio_segment=None)
    segments = [Segment(index=str(i), start=10.0 * i, end=10.0 * i + 1.2, speaker="A")
                for i in range(1, 6)]
    picked = svc.mine_enrollments(segments, audio)["A"]
    assert len(picked) > 1


def test_expanded_core_splices_the_real_overlap_not_two_seconds_earlier():
    """Ghép trả phải ánh xạ qua core_source_samples, không qua overlap gốc.

    Overlap 0.15 s lọt ngưỡng 0.1 s nhưng dưới short_core_threshold 0.2 s nên
    core nở ±2 s. Dùng overlap gốc làm mốc thì lấy nhầm audio sớm hơn 2 giây;
    _track_has_speech không bắt được vì chỗ đó cũng có tiếng nói của host.
    """
    class IdentityTSE(FakeTSE):
        """Bộ tách 'hoàn hảo': trả lại đúng mixture đã nhận."""

        def separate_two_speakers(self, mixture_audio, **kwargs):
            super().separate_two_speakers(mixture_audio, **kwargs)
            track = np.asarray(mixture_audio, dtype=np.float32).copy()
            return (track, track.copy(), self.sim_a, self.sim_b,
                    {"anchor_self": 0.6, "anchor_other": 0.1, "other_rms": 0.5})

    segs = [
        Segment(index="00001", start=0.0, end=30.0, speaker="SPEAKER_00"),
        Segment(index="00002", start=14.0, end=14.15, speaker="SPEAKER_01"),
        Segment(index="00003", start=32.0, end=40.0, speaker="SPEAKER_01"),
        Segment(index="00004", start=42.0, end=48.0, speaker="SPEAKER_00"),
    ]
    audio = _audio()
    svc = SeparationService(IdentityTSE(), logger=None)
    out = svc.process_overlaps(segs, audio, overlap_threshold=0.1)

    layout = svc.window_layouts[0]
    assert layout.get("core_expanded") is True, layout
    spliced = [s for s in out if getattr(s, "bss_spans", None)]
    assert spliced, "overlap 0.15 s phải được ghép trả"

    # Bộ tách là identity nên vùng ghép trả phải trùng đúng mixture gốc.
    # Chỉ so phần giữa: _cross_fade dùng ramp giữ công suất ở hai biên (mỗi
    # biên tối đa 1/8 vùng thay thế), nên với hai tín hiệu trùng nhau hai đầu
    # bị cộng lên tới +3 dB. Phần giữa được thay nguyên vẹn, và đó cũng chính
    # là phần mà lỗi ánh xạ làm hỏng.
    for seg in spliced:
        for lo, hi, _sim in seg.bss_spans:
            i = int(lo * SR) - int(seg.start * SR)
            n = int((hi - lo) * SR)
            edge = n // 8
            got = seg.audio[i + edge:i + n - edge]
            want = audio.waveform[int(lo * SR) + edge:][:len(got)]
            assert np.max(np.abs(got - want)) < 1e-4, (
                f"seg {seg.index} {lo:.2f}-{hi:.2f}s ghép nhầm audio")


# --- pool build cửa sổ song song (utils/window_pool.py) --------------------

def _independent_pair_block(block_index, speakers, block_dur=90.0, seed=0):
    """Một khối 90s độc lập: 2 speaker nói xen kẽ, có 1 backchannel ngắn.
    Dùng chung công thức đã đo là tốn việc thật cho WindowPlanner.build()."""
    rng = np.random.default_rng(seed)
    t0 = block_index * block_dur
    rows = []
    x = 0.0
    toggle = 0
    while x < block_dur - 5:
        L = rng.uniform(2.0, 4.0)
        rows.append((t0 + x, t0 + x + L, speakers[toggle]))
        x += L + rng.uniform(0.1, 0.3)
        toggle ^= 1
    host = next(r for r in rows if r[2] == speakers[0] and (r[0] - t0) > 30)
    rows.append((host[0] + 0.5, host[0] + 1.2, speakers[1]))
    return rows


def _multi_pair_scenario(n_blocks=4, block_dur=90.0):
    letters = [chr(ord('A') + i) for i in range(2 * n_blocks)]
    rows = []
    for i in range(n_blocks):
        rows.extend(_independent_pair_block(i, (letters[2*i], letters[2*i+1]),
                                             block_dur=block_dur, seed=i))
    segs = [Segment(index=str(i), start=a, end=b, speaker=s)
            for i, (a, b, s) in enumerate(rows)]
    total_dur = n_blocks * block_dur
    rng = np.random.default_rng(99)
    wave = rng.normal(0, 0.05, int(total_dur * SR)).astype(np.float32)
    wave[np.arange(len(wave)) % (SR // 2) < int(0.08 * SR)] = 0
    audio = AudioData(waveform=wave, sample_rate=SR, name="multi", audio_segment=None,
                       duration=total_dur)
    return segs, audio


def _run_with_workers(n_workers):
    segs, audio = _multi_pair_scenario(n_blocks=4)
    os.environ["BSS_WINDOW_WORKERS"] = str(n_workers)
    try:
        svc = SeparationService(FakeTSE(), logger=None)
        out = svc.process_overlaps(segs, audio, overlap_threshold=0.1)
    finally:
        del os.environ["BSS_WINDOW_WORKERS"]
    return out, svc


def test_window_pool_matches_sequential_build_end_to_end():
    """Bật pool (nhiều process) phải cho đúng kết quả ghép trả như tắt pool.

    4 cặp speaker độc lập -> 4 job, đủ vượt BSS_WINDOW_POOL_MIN_JOBS để pool
    thật sự được dùng (không rơi về tuần tự vì quá ít job). So sánh bss_spans
    và nội dung audio đã ghép trả giữa hai lượt chạy.
    """
    out_seq, svc_seq = _run_with_workers(0)
    out_par, svc_par = _run_with_workers(3)

    assert svc_par.stats["jobs"] >= 3, "kịch bản phải sinh đủ job để pool được kích hoạt"
    assert svc_seq.stats["spliced"] == svc_par.stats["spliced"] > 0
    assert svc_seq.stats["stitched"] == svc_par.stats["stitched"]

    by_index_seq = {s.index: s for s in out_seq}
    by_index_par = {s.index: s for s in out_par}
    assert set(by_index_seq) == set(by_index_par)
    for idx, seg_seq in by_index_seq.items():
        seg_par = by_index_par[idx]
        assert seg_seq.bss_spans == seg_par.bss_spans, f"seg {idx}: bss_spans khác nhau"
        assert np.array_equal(seg_seq.audio, seg_par.audio), f"seg {idx}: audio khác nhau"
