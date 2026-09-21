import collections
import copy
import json
import os
import threading
import time as _time
import traceback
from concurrent.futures import ThreadPoolExecutor

# Set BSS_TIMING=1 để bật log thời gian chi tiết từng bước trong separation loop.
# Tắt mặc định vì mỗi job log thêm ~4 dòng, với 55 job sẽ rất dài.
_BSS_TIMING = os.environ.get("BSS_TIMING", "0") == "1"
import numpy as np
from typing import Dict, List, Optional, Tuple
from schemas.audio import AudioData
from schemas.segment import Segment, SpeechSegment
from algorithms.diarization.overlap import detect_overlapping_segments
from utils.audio_normalize import safe_limit
from utils.separation_window import POLICY_VERSION, WindowPlanner, clean_segments, merge_ranges, subtract
from utils.window_pool import WindowBuildPool
from utils.separation_quality import track_quality, base_view, continuity_evidence
from utils.cpu_plan import usable_cores
from utils.music_map import MusicMap
from utils.enrollment_memory import EnrollmentMemory, ENABLED as BSS_MEMORY

# Đối tượng thay thế khi dịch vụ không dùng bộ nhớ mẫu giọng.
_NO_MEMORY = EnrollmentMemory(enabled=False)


# Ngữ cảnh thật quanh overlap; điểm cắt phải nằm trong vùng sạch.
# Context chỉ cần đủ để giữ continuity quanh overlap. Phần còn lại của cửa
# sổ ưu tiên clean padding vì đó là bằng chứng trực tiếp để gán hai track.
BSS_STITCH_EDGE_PAD = float(os.environ.get("BSS_STITCH_EDGE_PAD", "2.0"))
BSS_STITCH_EDGE_MAX = float(os.environ.get("BSS_STITCH_EDGE_MAX", "2.2"))
BSS_PADDING_MIN_PER_SPEAKER = float(
    os.environ.get("BSS_PADDING_MIN_PER_SPEAKER", "1.0")
)
# 1800s (30 phút mỗi hướng): trên corpus thật, nhiều speaker chỉ xuất hiện vài
# lần cách nhau hàng trăm-nghìn giây (một speaker phụ/hiếm nói vài câu rải
# suốt file). 400s từng bỏ sót các đoạn sạch xa hơn -- speaker gần như không
# có nguồn nào trong bán kính đó phải nhận ZERO_PROBE dù nguồn thật sự tồn
# tại, chỉ là ở xa. Xem doc/window-policy-v13-audit.md, phần "Nguyên nhân A".
BSS_STITCH_SEARCH = float(os.environ.get("BSS_STITCH_SEARCH", "1800.0"))
# Context luôn trần ở BSS_STITCH_EDGE_MAX mỗi phía, kể cả khi phía đối diện
# bị SP3/mép file chặn hẳn -- từng thử nới trần bù (BSS_STITCH_EDGE_OVERFLOW)
# nhưng dựng lại bằng chứng thật (job 34, doc/window-policy-v13-audit.md)
# cho thấy nới context chỉ kéo dài thêm phía vốn đã dài, đẩy overlap LỆCH
# TÂM hơn chứ không giúp gì. Bù cho phía bị chặn giờ là việc của padding
# (anchor_penalty trong choose_padding, utils/separation_window.py), không
# phải context.

# --- Song song hoá việc dựng cửa sổ ----------------------------------------
# WindowPlanner.build() là CPU/numpy thuần và có thể tốn vài giây một job khi
# overlap có nhiều support candidate (đo được ~3.5s/job trên kịch bản lượt
# nói ngắn xen kẽ) -- trong khi bss_model.separate_two_speakers() sau đó lại
# chạy GPU tuần tự. Build của job N+1 không phụ thuộc gì vào việc tách của
# job N nên chạy song song trong lúc GPU bận: pipeline, không phải theo lô.
# Xem utils/window_pool.py để biết cơ chế đầy đủ.
#
# Đọc từ config.json (models.bss.window_workers, main.py bắc cầu sang biến
# này) hoặc đặt tay để sweep nhanh không cần sửa config. Mặc định trong
# config.json là 0 (TẮT): kết quả (build() không có GPU, tách biệt) rất khác
# khi đo lồng vào bss_model.separate_two_speakers() thật -- Sidon mất vài
# giây/lần gọi, và kết quả window PHẢI trả đúng thứ tự nộp (giữ nguyên hành vi
# enrollment_memory tích luỹ tuần tự), nên nếu job đầu hàng đợi lại là job
# build nặng nhất, main thread vẫn chờ đúng nó dù các job sau đã build xong.
# Đo trên máy dev (8 lõi, GPU giả lập 3s/job): 8 job -> pool chậm hơn tuần tự
# (~28.6s so với ~28.0s), phần build "lộ" ra chỉ ~4s trên tổng ~28s GPU. Lợi
# ích thật phụ thuộc số job overlap/file và độ trễ Sidon thật -- cần tự đo
# trên dữ liệu sản xuất trước khi bật > 0.
#
# Để trống (None, không đặt trong config lẫn env) = tự tính theo
# usable_cores() khi có đủ job (xem BSS_WINDOW_POOL_MIN_JOBS); 0 = tắt hẳn.
_BSS_WINDOW_WORKERS_ENV = os.environ.get("BSS_WINDOW_WORKERS")
BSS_WINDOW_POOL_MIN_JOBS = int(os.environ.get("BSS_WINDOW_POOL_MIN_JOBS", "3"))
BSS_WINDOW_MAX_PENDING = int(os.environ.get("BSS_WINDOW_MAX_PENDING", "32"))
MIN_OVERLAP_SECONDS = 0.05

def _resolve_pool_size() -> int:
    """Bao nhiêu process cho pool build cửa sổ; 0 nghĩa là tắt hẳn cho cả batch.

    Quyết định MỘT LẦN khi pool được tạo (lười, ở lần process_overlaps() đầu
    tiên cần đến nó) -- không phụ thuộc số job của riêng một file, vì pool
    (utils/window_pool.py) giờ sống suốt cả batch chứ không tạo/đóng mỗi file.

    Tôn trọng usable_cores() (utils/cpu_plan.py) thay vì hard-code: module đó
    đo được rằng trên máy ít core, mở thêm process làm CHẬM đi vì tranh CPU
    với main process và Sidon worker (SIDON_CPU_THREADS đã giữ phần của nó).
    Số worker = usable_cores() - 2, không có trần cứng; trên máy nhiều core
    (H100 server 8+ core) dùng toàn bộ core còn lại là đúng.
    """
    if _BSS_WINDOW_WORKERS_ENV is not None:
        try:
            return max(0, int(_BSS_WINDOW_WORKERS_ENV))
        except ValueError:
            pass
    # Main process + Sidon worker subprocess đã chiếm ít nhất 2 "chỗ"; phần
    # còn lại mới dành cho pool build cửa sổ.
    return max(0, usable_cores() - 2)


def _worth_pooling(n_jobs: int) -> bool:
    """File này có đủ job để đáng nộp vào pool không.

    Tách khỏi việc quyết định KÍCH THƯỚC pool (giờ cố định cho cả batch, xem
    _resolve_pool_size): pool tồn tại không có nghĩa MỌI file đều nên dùng
    nó -- dựng shared_memory + pickle segments cho 1-2 job vẫn tốn hơn chạy
    tuần tự ngay trong main process. Luôn áp dụng kể cả khi kích thước pool bị
    ép tay qua config/env: đo trên Kaggle thật, window_workers=8 ép tay từng
    khiến một file chỉ 2 job vẫn đi qua pool và mất 22.87s cho window đầu
    tiên -- việc ép số worker và việc "file này có đáng dùng pool" là hai
    quyết định độc lập, không nên để cái này ăn theo cái kia.
    """
    return n_jobs >= BSS_WINDOW_POOL_MIN_JOBS

# --- Mẫu giọng đối chiếu --------------------------------------------------
BSS_ENROLL_BUDGET = float(os.environ.get("BSS_ENROLL_BUDGET", "8.0"))
BSS_ENROLL_MIN_CLIP = float(os.environ.get("BSS_ENROLL_MIN_CLIP", "0.35"))
BSS_ENROLL_MIN_TOTAL = float(os.environ.get("BSS_ENROLL_MIN_TOTAL", "1.5"))
# Ưu tiên mẫu giọng liền mạch. Thử nghiệm cũ trên 8 mixture hai người, cùng mức 0 dB:
# Ghép thường: +6.86 dB SI-SDR; ghép crossfade: +6.94 dB;
# đoạn liền mạch: +8.39 dB; đoạn liền mạch bù zero: +8.56 dB.
# Kết quả gợi ý tính liên tục của lời nói quan trọng hơn riêng việc làm mượt
# mối nối. Đây là số đo cũ của mẫu đối chiếu, không phải chính sách chèn pad
# cho cửa sổ mới. Đủ độ dài dưới đây thì dùng một mẫu; nếu thiếu mới gom thêm.
BSS_ENROLL_PREFER_SINGLE = float(os.environ.get("BSS_ENROLL_PREFER_SINGLE", "4.0"))

# --- Kiểm tra chất lượng --------------------------------------------------
# Ngưỡng CHƯA HIỆU CHỈNH. Đối chiếu phân vị similarity trong log [TSE] và
# nghe các đoạn thất bại trước khi kết luận. Điểm ECAPA trên giọng đã tách
# có thể thấp hơn giọng tự nhiên nên giá trị mặc định tương đối thấp.
BSS_QC_SIM_THRESHOLD = float(os.environ.get("BSS_QC_SIM_THRESHOLD", "0.40"))
BSS_NOT_A_MARGIN = float(os.environ.get("BSS_NOT_A_MARGIN", "0.15"))
BSS_SILENCE_RMS = float(os.environ.get("BSS_SILENCE_RMS", "0.001"))

# --- Hiệu chỉnh mức âm lượng track trước khi splice -----------------------
# Không lấy RMS của mixture overlap làm chuẩn: mixture chứa cả hai speaker nên
# thường lớn hơn từng speaker riêng và có thể boost track tách lên quá mức.
# Thay vào đó, so track Sidon với audio gốc ở CHÍNH các probe sạch của speaker
# gần overlap nhất để đo gain bias do separator tạo ra, rồi áp gain đó cho patch.
BSS_LEVEL_REF_SEC = float(os.environ.get("BSS_LEVEL_REF_SEC", "1.5"))
BSS_LEVEL_MAX_ADJUST_DB = float(os.environ.get("BSS_LEVEL_MAX_ADJUST_DB", "6.0"))
BSS_LEVEL_MIN_VALID_SEC = float(os.environ.get("BSS_LEVEL_MIN_VALID_SEC", "0.10"))

BSS_DUMP_FAILED = os.environ.get("BSS_DUMP_FAILED", "1") not in ("0", "false", "False")

# Danh sách lý do cố định để mọi overlap bị bỏ đều có thống kê và dấu vết.
REASONS = (
    "no_enroll",        # Không đủ âm thanh sạch làm mẫu đối chiếu
    "no_window",        # Không dựng được cửa sổ hợp lệ theo chính sách 15 giây
    "multi_speaker",    # Có hơn hai người trong vùng cần xử lý
    "qc_sim",           # Điểm track thấp hơn BSS_QC_SIM_THRESHOLD
    "unscorable",       # Không đủ lời để đánh giá, chưa khẳng định tách thất bại
    "not_a_fail",       # Không vượt phép kiểm tra tương đối not-A
    "already_spliced",  # Vùng này đã được lượt khác ghi kết quả
    "short_track",      # Model trả về thiếu mẫu âm thanh
    "empty_track",      # Track im lặng ngay tại nơi mixture có lời
    "low_energy_coverage",
    "insufficient_evidence",
    "nonfinite_audio",
    "window_error",     # Planner ném exception trước khi có window
    "model_error",      # Separator/ECAPA ném exception trên window đã dựng
    "same_speaker",     # Hai segment cùng speaker, không cần tách hai giọng
    "below_threshold",  # Overlap ngắn hơn ngưỡng chạy model
)

# Lý do mà vùng giữ nguyên mixture vẫn chỉ chứa giọng của chính speaker đó.
# Bước xuất dual-channel xóa vùng thất bại để chặn giọng người khác lọt sang
# track sai; với các lý do dưới đây không có giọng nào để chặn, nên xóa chỉ
# làm mất lời nói thật.
SAFE_FAIL_REASONS = frozenset({"same_speaker"})


class SeparationService:
    """Tách lời nói chồng bằng BSS và gán người nói bằng ECAPA.
    Mỗi vùng được xử lý phải có trong bss_spans hoặc bss_failed_spans với lý do.
    Kiểm thử test_every_overlap_is_accounted_for bảo vệ điều kiện này."""

    checkpoint_version = POLICY_VERSION

    def __init__(self, bss_model=None, logger=None, dump_dir: Optional[str] = None,
                 model_loader=None, performance_config=None):
        self._bss_model = bss_model
        self.model_loader = model_loader
        self.logger = logger
        self.dump_dir = dump_dir
        self.stats = collections.Counter()
        self.sims = []
        self.overlap_durations = []
        self.window_layouts = []
        self.failures = []          # (bắt đầu, kết thúc, speaker, lý do, chi tiết)
        self.failure_artifacts = []
        self._dump_warned = False
        self._failure_artifact_counter = 0
        # Pipeline gán bản đồ nhạc; bản đồ rỗng nghĩa là chưa có vùng cần tránh.
        self.music_map = MusicMap()
        # Nhãn speaker chỉ có ý nghĩa trong từng file; reset_stats() xóa bộ nhớ
        # để không gán giọng của file trước cho người cùng nhãn ở file sau.
        self.memory = EnrollmentMemory(logger=logger)
        # Pool build cửa sổ song song (utils/window_pool.py) -- tạo lười ở lần
        # process_overlaps() đầu tiên cần đến nó, sống suốt các lần gọi tiếp
        # theo (nhiều file trong một batch dùng chung một SeparationService).
        # close_window_pool() phải được PipelineService gọi ở cuối stage
        # 'separation' của cả batch, không phải sau mỗi file -- xem
        # utils/window_pool.py để biết lý do (chi phí spawn không chia sẻ
        # được giữa các file nếu tạo/đóng theo từng file).
        self._window_pool = None
        self._window_pool_state = {
            "pool": None,
            "lock": threading.Lock(),
        }
        self.performance_config = dict(performance_config or {})
        self._async_state = {
            "gpu_executor": None,
            "post_executor": None,
            "lock": threading.Lock(),
            "next_file_id": 0,
        }
        self._file_run_id = None
        self._fork_bss_model = False
        self._owns_bss_model = False

    # Lấy model từ loader khi dùng vì model được tải theo từng giai đoạn.
    # Giữ tham chiếu lúc khởi tạo có thể giữ mãi giá trị None của model chưa tải.
    @property
    def bss_model(self):
        if self._bss_model is not None:
            return self._bss_model
        base = (getattr(self, "model_loader", None)
                and self.model_loader.get("separator"))
        if base is not None and self._fork_bss_model:
            fork = getattr(base, "fork", None)
            self._bss_model = fork() if callable(fork) else base
            self._owns_bss_model = self._bss_model is not base
            self._fork_bss_model = False
            return self._bss_model
        return base

    @bss_model.setter
    def bss_model(self, model):
        """Cho phép truyền model trực tiếp, dùng khi tạo dịch vụ trong kiểm thử."""
        self._bss_model = model

    def fork_for_file(self):
        """Isolate mutable separation state for one concurrent audio file."""
        clone = copy.copy(self)
        clone._bss_model = None
        clone._fork_bss_model = True
        clone._owns_bss_model = False
        clone.stats = collections.Counter()
        clone.sims = []
        clone.overlap_durations = []
        clone.window_layouts = []
        clone.failures = []
        clone.failure_artifacts = []
        clone._dump_warned = False
        clone._failure_artifact_counter = 0
        clone._file_run_id = None
        clone.memory = copy.copy(self.memory)
        clone.memory._clips = {}
        clone.memory.added = 0
        clone.memory.rejected = 0
        # _window_pool_state is intentionally shared: it owns stateless CPU
        # workers, while each open_file call owns separate shared memory.
        return clone

    def _async_runtime(self):
        """Return shared GPU/post executors and a file-local scheduling id."""
        cfg = self.performance_config
        if not cfg.get("enabled", False) or not cfg.get("ordered_postprocess", True):
            return None

        model = self.bss_model
        if not (callable(getattr(model, "separate_raw", None))
                and callable(getattr(model, "postprocess_separated", None))):
            return None

        gpu_workers = max(1, int(cfg.get("max_workers", 1)))
        post_workers = max(1, int(cfg.get("postprocess_workers", 1)))
        state = self._async_state
        with state["lock"]:
            if state["gpu_executor"] is None:
                state["gpu_executor"] = ThreadPoolExecutor(
                    max_workers=gpu_workers, thread_name_prefix="sidon-gpu")
                state["post_executor"] = ThreadPoolExecutor(
                    max_workers=post_workers, thread_name_prefix="sidon-post")
            if self._file_run_id is None:
                self._file_run_id = state["next_file_id"]
                state["next_file_id"] += 1
        return state["gpu_executor"], state["post_executor"], model

    def close_async_pools(self):
        """Drain and close the shared Sidon pipeline executors once per stage."""
        state = getattr(self, "_async_state", None)
        if state is None:
            return
        with state["lock"]:
            gpu_executor = state["gpu_executor"]
            post_executor = state["post_executor"]
            state["gpu_executor"] = None
            state["post_executor"] = None
        if gpu_executor is not None:
            gpu_executor.shutdown(wait=True, cancel_futures=False)
        if post_executor is not None:
            post_executor.shutdown(wait=True, cancel_futures=False)

    def close_file_model(self):
        """Release scratch state owned by a concurrent per-file model clone."""
        if not self._owns_bss_model or self._bss_model is None:
            return
        close = getattr(self._bss_model, "close", None)
        if callable(close):
            close()
        self._bss_model = None
        self._owns_bss_model = False

    def reset_stats(self):
        """Xóa thống kê từng file vì cùng một dịch vụ được dùng lại cho cả batch.
        Nếu không xóa, báo cáo file sau sẽ chứa số cộng dồn từ file trước."""
        self.stats = collections.Counter()
        self.sims = []
        self.overlap_durations = []
        self.window_layouts = []
        self.failures = []
        self.failure_artifacts = []
        self._failure_artifact_counter = 0
        # Mẫu giọng của file trước không được dùng cho file sau dù nhãn trùng nhau.
        # getattr cho phép dịch vụ trong kiểm thử không có bộ nhớ này.
        memory = getattr(self, "memory", None)
        if memory is not None:
            memory.reset()
        # Embedding được cache theo nhãn speaker, mà nhãn bắt đầu lại ở mỗi file.
        reset = getattr(self.bss_model, "reset_speakers", None)
        if reset:
            reset()

    def close_window_pool(self):
        """Đóng pool build cửa sổ song song (utils/window_pool.py), nếu có.

        Phải gọi ĐÚNG MỘT LẦN ở cuối stage 'separation' của CẢ BATCH -- không
        phải sau mỗi file. PipelineService gọi qua begin_stage_scope()/
        end_stage_scope(), cùng cơ chế và cùng thời điểm Sidon worker được
        giữ sống qua nhiều file rồi mới dừng. reset_stats() KHÔNG gọi hàm
        này: reset_stats() chạy giữa các file trong cùng một stage, lúc đó
        pool vẫn còn cần dùng tiếp cho file kế."""
        state = getattr(self, "_window_pool_state", None)
        if state is None:
            pool = self._window_pool
            self._window_pool = None
        else:
            with state["lock"]:
                pool = state["pool"]
                state["pool"] = None
                self._window_pool = None
        if pool is not None:
            pool.close()

    # ------------------------------------------------------------------
    def _fail(self, enh_seg, start, end, reason, detail=""):
        """Ghi vùng giữ nguyên mixture cùng lý do không thay bằng kết quả tách."""
        assert reason in REASONS, f"unknown reason {reason!r}"
        self.stats[f"fail_{reason}"] += 1
        if enh_seg is not None:
            enh_seg.bss_failed_spans.append((start, end, reason, detail))
        self.failures.append((start, end, getattr(enh_seg, "speaker", "?"), reason, detail))
        if self.logger:
            self.logger.debug(f"[TSE] {reason} @ {start:.2f}s {detail}")

    def _report_stats(self):
        if not self.logger:
            return
        s = self.stats
        self.logger.info(
            f"[TSE] jobs={s['jobs']} pairs={s['pairs']} spliced={s['spliced']} "
            f"retried={s['retried']}"
        )
        fails = {r: s[f"fail_{r}"] for r in REASONS if s[f"fail_{r}"]}
        self.logger.info(f"[TSE] failures: {fails or 'none'}")
        if self.overlap_durations:
            d = np.array(self.overlap_durations)
            self.logger.info(
                f"[TSE] overlap dur: min={d.min():.2f}s p50={np.median(d):.2f}s "
                f"max={d.max():.2f}s | <0.5s={int((d < 0.5).sum())}"
            )
        if self.sims:
            a = np.array(self.sims)
            self.logger.info(
                f"[TSE] ECAPA sim: p10={np.percentile(a, 10):.2f} "
                f"p50={np.percentile(a, 50):.2f} p90={np.percentile(a, 90):.2f} "
                f"max={a.max():.2f} (threshold {BSS_QC_SIM_THRESHOLD}, NOT calibrated)"
            )
        else:
            self.logger.info("[TSE] no similarity was computed at all")

    def report_payload(self) -> dict:
        """Trả báo cáo để bên gọi dùng trực tiếp thay vì tự tính lại."""
        return self._report_payload()

    def write_report(self, save_dir: str, audio_name: str, payload: dict = None):
        """Ghi báo cáo từng vùng để kiểm tra nguyên nhân thất bại.
        Khi chạy theo giai đoạn, bước xuất diễn ra ở lần run() khác và bộ đếm đã
        được reset. payload nhận báo cáo lưu trong checkpoint của lúc separation."""
        path = os.path.join(save_dir, f"{audio_name}_bss_report.json")
        if payload is None:
            payload = self._report_payload()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to write TSE report: {e}")
        return path

    def _report_payload(self) -> dict:
        payload = {
            "window_policy": POLICY_VERSION,
            "windows": getattr(self, "window_layouts", []),
            # Các ngưỡng dựng cửa sổ và kiểm tra dùng cho lần chạy này.
            "thresholds": {
                "qc_sim": BSS_QC_SIM_THRESHOLD, "not_a_margin": BSS_NOT_A_MARGIN,
                "window_target": 15.0,
                "window_target_soft": None,
                "quality_split_seconds": 12.0,
                "support_min_seconds": 1.5,
                "secondary_edge_margin_seconds": 0.0,
                "crossfade_seconds": 0.02,
                "window_max": 15.0,
                "context_per_side_seconds": BSS_STITCH_EDGE_PAD,
                "context_max_per_side_seconds": BSS_STITCH_EDGE_MAX,
                "padding_min_per_speaker_seconds": BSS_PADDING_MIN_PER_SPEAKER,
                "min_solo": 0.0,
                "only_hard_boundary": "third_speaker",
                "enroll_budget": BSS_ENROLL_BUDGET,
                "enroll_min_clip": BSS_ENROLL_MIN_CLIP,
                "enroll_min_total": BSS_ENROLL_MIN_TOTAL,
            },
            # Lưu cấu hình bộ nhớ và bản đồ nhạc để có thể đối chiếu các lần chạy.
            "enrollment_memory": (self.memory.summary()
                                  if BSS_MEMORY and getattr(self, "memory", None)
                                  else None),
            "music_map": self.music_map.summary(),
            "stats": dict(self.stats),
            "sim_percentiles": (
                {p: float(np.percentile(self.sims, p)) for p in (10, 50, 90)}
                if self.sims else None
            ),
            "failures": [
                {"start": a, "end": b, "speaker": spk, "reason": r, "detail": d}
                for a, b, spk, r, d in self.failures
            ],
            "failure_artifacts": list(getattr(self, "failure_artifacts", [])),
        }
        return payload

    def mine_enrollments(self, segments: List[Segment], audio: AudioData, min_dur: float = 2.0, top_k: int = 5) -> Dict[str, List[np.ndarray]]:
        """Lấy mẫu đối chiếu cho từng speaker từ segment hoàn toàn không giao nhau."""
        if self.logger: self.logger.info("Mining enrollments for speaker assignment...")
        
        enrollments = {}
        sr = audio.sample_rate
        waveform = audio.waveform
        clean_by_speaker = {}
        # Có giao nhau là loại nguyên segment; không khai thác phần còn lại.
        for s in clean_segments(segments):
            clean_by_speaker.setdefault(s.speaker, []).append((s.start, s.end))

        # Tránh lấy mẫu trên vùng có nhạc vì embedding khi đó chứa cả nhạc nền.
        # Dùng getattr để hỗ trợ dịch vụ dựng riêng trong kiểm thử, chưa có music_map.
        music_map = getattr(self, "music_map", None)
        if music_map:
            for spk, spans in list(clean_by_speaker.items()):
                kept = []
                for a, b in spans:
                    kept.extend(music_map.clean_parts(a, b))
                clean_by_speaker[spk] = kept

        for spk in set(s.speaker for s in segments):
            candidates = clean_by_speaker.get(spk, [])

            # Mẫu ECAPA quá ngắn có thể tạo embedding nhiễu. Gom theo ngân sách thời
            # gian và từ chối nếu tổng lượng mẫu chưa đủ, thay vì hạ xuống mẫu 0.1 s.
            candidates = [c for c in candidates if (c[1] - c[0]) >= BSS_ENROLL_MIN_CLIP]
            candidates.sort(key=lambda c: c[1] - c[0], reverse=True)

            # Ưu tiên đoạn dài nhất; đủ BSS_ENROLL_PREFER_SINGLE thì dừng gom mẫu.
            picked, total = [], 0.0
            for start, end in candidates:
                if total >= BSS_ENROLL_BUDGET:
                    break
                picked.append(waveform[int(start * sr):int(end * sr)].copy())
                total += end - start
                if len(picked) == 1 and total >= BSS_ENROLL_PREFER_SINGLE:
                    break

            if total < BSS_ENROLL_MIN_TOTAL:
                if self.logger:
                    self.logger.warning(
                        f"Speaker {spk}: only {total:.2f}s of clean audio "
                        f"(need >={BSS_ENROLL_MIN_TOTAL}s); enrollment would be unreliable, "
                        "window-local probes and complement assignment will be tried."
                    )
                enrollments[spk] = []
            else:
                enrollments[spk] = picked

        return enrollments

    @staticmethod
    def _window_probe_enrollment(window_audio, probes, sr):
        """Ghép clean probes trong window thành enrollment best-effort."""
        pieces = []
        for start, end in probes or ():
            start = max(0, int(start))
            end = min(len(window_audio), int(end))
            if end > start:
                pieces.append(np.asarray(window_audio[start:end], dtype=np.float32))
        if not pieces or sum(len(piece) for piece in pieces) < BSS_ENROLL_MIN_CLIP * sr:
            return []

        overlap = max(1, int(0.02 * sr))
        joined = pieces[0].copy()
        for piece in pieces[1:]:
            if len(joined) >= overlap and len(piece) >= overlap:
                ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                joined[-overlap:] = (
                    joined[-overlap:] * (1.0 - ramp) + piece[:overlap] * ramp
                )
                joined = np.concatenate((joined, piece[overlap:]))
            else:
                joined = np.concatenate((joined, piece))
        return [joined]

    @staticmethod
    def _track_has_speech(host: np.ndarray, track: np.ndarray,
                          frame_sec: float = 0.02, sr_hint: int = 24000) -> bool:
        """Kiểm tra track có lời tại những nơi mixture có lời.
        So năng lượng từng khung ngắn: RMS toàn đoạn có thể bỏ sót trường hợp
        track im lặng gần hết overlap nhưng chỉ có đuôi giọng khác ở cuối."""
        return track_quality(host, track, sr_hint, BSS_SILENCE_RMS, frame_sec)["accepted"]

    @staticmethod
    def _probe_source_mid(probe, layout):
        """Map tâm một probe trong window về sample trên source timeline thật.

        WindowPlanner có thể ghép support ở xa overlap vào sát core trong window.
        Vì vậy khoảng cách trong window không phản ánh khoảng cách thời gian thật.
        Probe nằm ngoài vùng crossfade an toàn, nên một điểm giữa là đủ để xác định
        piece chứa nó và ánh xạ tuyến tính về source_samples.
        """
        a, b = int(probe[0]), int(probe[1])
        if b <= a:
            return None
        mid = 0.5 * (a + b)

        for piece in layout.get("pieces", ()):
            w0, w1 = piece.get("window_samples", (0, 0))
            s0, s1 = piece.get("source_samples", (0, 0))
            if w1 <= w0:
                continue
            if w0 <= mid < w1:
                ratio = (mid - w0) / float(w1 - w0)
                return float(s0 + ratio * (s1 - s0))
        return None

    @classmethod
    def _speaker_level_gain(
        cls,
        original_window: np.ndarray,
        separated_track: np.ndarray,
        probes,
        layout,
        overlap_center_source: int,
        sr: int,
        max_ref_sec: float = BSS_LEVEL_REF_SEC,
        max_adjust_db: float = BSS_LEVEL_MAX_ADJUST_DB,
    ) -> Tuple[float, dict]:
        """Ước lượng gain hiệu chỉnh separator từ clean probe của đúng speaker.

        Ta KHÔNG ép RMS của overlap bằng RMS đoạn sạch. Thay vào đó, tại cùng
        timestamp sạch ta so:

            original clean speaker / separated clean speaker

        để đo riêng gain bias do separator gây ra. Median của chênh lệch dB trên
        frame 20 ms chống silence/peak bất thường tốt hơn RMS toàn đoạn.

        Probe được ưu tiên theo source timestamp thật gần overlap nhất. Chỉ dùng
        tối đa ``max_ref_sec`` audio và clamp gain để một probe lỗi không phá mức.
        """
        info = {
            "gain": 1.0,
            "gain_db": 0.0,
            "used_seconds": 0.0,
            "valid_seconds": 0.0,
            "probe_count": 0,
            "reason": "no_probe",
        }
        if not probes or sr <= 0:
            return 1.0, info

        # Xếp probe theo khoảng cách thật trên source timeline tới overlap.
        ranked = []
        for probe in probes:
            source_mid = cls._probe_source_mid(probe, layout)
            distance = (abs(source_mid - overlap_center_source)
                        if source_mid is not None else float("inf"))
            ranked.append((distance, probe))
        ranked.sort(key=lambda item: item[0])

        remaining = max(1, int(max_ref_sec * sr))
        refs, seps = [], []
        used_probes = 0

        for _distance, probe in ranked:
            if remaining <= 0:
                break

            a, b = int(probe[0]), int(probe[1])
            a = max(0, a)
            b = min(b, len(original_window), len(separated_track))
            if b <= a:
                continue

            take = min(b - a, remaining)
            if take <= 0:
                continue

            refs.append(
                np.asarray(original_window[a:a + take], dtype=np.float64)
            )
            seps.append(
                np.asarray(separated_track[a:a + take], dtype=np.float64)
            )
            remaining -= take
            used_probes += 1

        if not refs:
            info["reason"] = "no_usable_probe"
            return 1.0, info

        ref = np.concatenate(refs)
        sep = np.concatenate(seps)
        info["used_seconds"] = float(min(len(ref), len(sep)) / sr)
        info["probe_count"] = int(used_probes)

        # So theo frame 20 ms để bỏ các frame im lặng và dùng median dB.
        frame = max(1, int(0.020 * sr))
        n_frames = min(len(ref), len(sep)) // frame
        if n_frames <= 0:
            info["reason"] = "probe_too_short"
            return 1.0, info

        ref_frames = ref[:n_frames * frame].reshape(n_frames, frame)
        sep_frames = sep[:n_frames * frame].reshape(n_frames, frame)

        ref_rms = np.sqrt(np.mean(ref_frames * ref_frames, axis=1) + 1e-12)
        sep_rms = np.sqrt(np.mean(sep_frames * sep_frames, axis=1) + 1e-12)

        # Cả original lẫn separated phải có mức đủ lớn. Điều này loại silence ở
        # original và residual/noise-floor ở track tách khỏi phép hiệu chỉnh.
        valid = (
            (ref_rms >= BSS_SILENCE_RMS)
            & (sep_rms >= BSS_SILENCE_RMS)
        )
        valid_samples = int(valid.sum()) * frame
        info["valid_seconds"] = float(valid_samples / sr)

        min_valid_samples = max(frame, int(BSS_LEVEL_MIN_VALID_SEC * sr))
        if valid_samples < min_valid_samples:
            info["reason"] = "not_enough_voiced_reference"
            return 1.0, info

        delta_db = (
            20.0 * np.log10(ref_rms[valid] + 1e-12)
            - 20.0 * np.log10(sep_rms[valid] + 1e-12)
        )

        gain_db = float(np.median(delta_db))
        if not np.isfinite(gain_db):
            info["reason"] = "non_finite_gain"
            return 1.0, info

        gain_db = float(np.clip(gain_db, -max_adjust_db, max_adjust_db))
        gain = float(10.0 ** (gain_db / 20.0))

        info.update({
            "gain": gain,
            "gain_db": gain_db,
            "reason": "ok",
        })
        return gain, info

    def _cross_fade(self, orig_audio: np.ndarray, new_audio: np.ndarray,
                    fade_samples: int, fade_left=True, fade_right=True) -> np.ndarray:
        """Thay vùng âm thanh gốc bằng track đã tách, làm mượt hai biên.

        Dùng linear crossfade thay cho equal-power sin/cos. Original overlap và
        separated track có thành phần tương quan mạnh; equal-power có tổng hệ số
        ~1.414 ở giữa fade và có thể làm transition phồng gần +3 dB. Linear fade
        luôn giữ tổng trọng số bằng 1, phù hợp hơn cho phép thay thế cùng nguồn.
        """
        result = orig_audio.copy()
        limit = min(len(orig_audio), len(new_audio))
        if limit == 0:
            return result

        # Mỗi ramp không chiếm quá một phần tám độ dài vùng thay thế.
        fade_samples = min(fade_samples, limit // 8)
        if fade_samples <= 0:
            result[:limit] = new_audio[:limit]
            return result

        t = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)
        ramp_in = t
        ramp_out = 1.0 - t

        # Chuyển dần từ âm thanh gốc sang track đã tách.
        result[:fade_samples] = (orig_audio[:fade_samples] * ramp_out
                                 + new_audio[:fade_samples] * ramp_in)

        # Phần giữa dùng nguyên mức âm thanh đã tách.
        mid_limit = limit - fade_samples
        result[fade_samples:mid_limit] = new_audio[fade_samples:mid_limit]

        # Chuyển dần về âm thanh gốc ở biên cuối.
        result[mid_limit:limit] = (orig_audio[mid_limit:limit] * ramp_in
                                   + new_audio[mid_limit:limit] * ramp_out)
        if not fade_left:
            result[:fade_samples] = new_audio[:fade_samples]
        if not fade_right:
            result[mid_limit:limit] = new_audio[mid_limit:limit]
        return result

            # --- Công cụ khoảng thời gian và nhóm overlap ----------------------------
    @staticmethod
    def _intervals_by_speaker(segments) -> Dict[str, List[Tuple[float, float]]]:
        by_spk: Dict[str, List[Tuple[float, float]]] = {}
        for s in segments:
            by_spk.setdefault(s.speaker, []).append((s.start, s.end))
        for spk in by_spk:
            by_spk[spk].sort()
        return by_spk







    def seams(self):
        """Trả vị trí các mối nối trong timeline mà separation đang dùng.
        Seam là metadata; cửa sổ được đi qua seam nhưng không qua speaker thứ ba.
        Dịch vụ không có timeline được hiểu là bản ghi chưa cắt."""
        timeline = getattr(self, "timeline", None)
        return timeline.seams() if timeline else []


    @staticmethod
    def _solo_spans(by_spk, spk, lo, hi):
        """Các phần trong [lo, hi] chỉ có spk nói, không có người khác."""
        own = [(max(a, lo), min(b, hi)) for a, b in by_spk.get(spk, [])
               if b > lo and a < hi]
        own = [(a, b) for a, b in own if b > a]
        if not own:
            return []
        others = []
        for other, ivals in by_spk.items():
            if other == spk:
                continue
            for a, b in ivals:
                if b > lo and a < hi:
                    others.append((max(a, lo), min(b, hi)))
        others.sort()

        solo = []
        for a, b in own:
            cur = a
            for oa, ob in others:
                if ob <= cur:
                    continue
                if oa >= b:
                    break
                if cur < oa:
                    solo.append((cur, min(oa, b)))
                cur = max(cur, ob)
                if cur >= b:
                    break
            if cur < b:
                solo.append((cur, b))
        return [(a, b) for a, b in solo if b - a > 1e-6]

    @staticmethod
    def _speakers_in(by_spk, lo, hi):
        return {spk for spk, ivals in by_spk.items()
                if any(b > lo and a < hi for a, b in ivals)}

    def _group_jobs(self, pairs):
        """Chỉ nhóm overlap giao/chạm nhau của cùng hai người; giữ mọi nguồn."""
        buckets = {}
        self._same_speaker_pairs = []
        for p in pairs:
            key = tuple(sorted({p["seg1"]["speaker"], p["seg2"]["speaker"]}))
            if len(key) != 2:
                self._same_speaker_pairs.append(p)
                continue
            buckets.setdefault(key, []).append(p)
        jobs = []
        for (a, b), plist in sorted(buckets.items()):
            current, end = [], None
            for p in sorted(plist, key=lambda p: (p["overlap_start"], p["overlap_end"])):
                if current and p["overlap_start"] > end:
                    jobs.append((a, b, current))
                    current = []
                current.append(p)
                end = max(end, p["overlap_end"]) if end is not None else p["overlap_end"]
            if current:
                jobs.append((a, b, current))
        return sorted(jobs, key=lambda job: job[2][0]["overlap_start"])

    @staticmethod
    def _splice_pairs(plist):
        """Hợp phần giao theo từng segment để không ghi đè lặp cùng mẫu."""
        by_segment = {}
        for p in plist:
            for sd in (p["seg1"], p["seg2"]):
                entry = by_segment.setdefault(sd["index"], (sd, []))
                entry[1].append((max(sd["start"], p["overlap_start"]),
                                 min(sd["end"], p["overlap_end"])))
        return [(sd, lo, hi) for sd, spans in by_segment.values()
                for lo, hi in merge_ranges(spans)]

    # ------------------------------------------------------------------
    def _dump_tracks(self, subdir, tag, mixture, track_1, track_2, sr):
        """Ghi mixture và hai track để nghe đối chiếu, cả khi lượt tách thành công.
        Cần nghe để phân biệt lỗi model với việc đặt ngưỡng QC chưa phù hợp."""
        if not (BSS_DUMP_FAILED and self.dump_dir):
            return
        try:
            import soundfile as sf
            d = os.path.join(self.dump_dir, subdir)
            os.makedirs(d, exist_ok=True)
            _t0 = _time.perf_counter()
            if _BSS_TIMING and self.logger:
                n_bytes = (mixture.nbytes
                           + (track_1.nbytes if track_1 is not None else 0)
                           + (track_2.nbytes if track_2 is not None else 0))
                self.logger.debug(
                    f"[TIMING] dump_tracks: writing {n_bytes/1e6:.1f} MB "
                    f"to {subdir}/{tag}_*.wav")
            sf.write(os.path.join(d, f"{tag}_mix.wav"), mixture, sr)
            if track_1 is not None:
                sf.write(os.path.join(d, f"{tag}_trackA.wav"), track_1, sr)
                sf.write(os.path.join(d, f"{tag}_trackB.wav"), track_2, sr)
            if _BSS_TIMING and self.logger:
                self.logger.debug(
                    f"[TIMING] dump_tracks: done in {_time.perf_counter()-_t0:.2f}s")
        except Exception as e:
            # Chỉ cảnh báo một lần nếu thiếu soundfile hoặc không ghi được thư mục,
            # tránh lặp thông báo trên từng clip nhưng vẫn báo mất dữ liệu đối chiếu.
            if self.logger and not self._dump_warned:
                self._dump_warned = True
                self.logger.warning(
                    f"[TSE] track dumps disabled: {type(e).__name__}: {e}"
                )

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        raise TypeError(f"not JSON serializable: {type(value).__name__}")

    @staticmethod
    def _signal_summary(signal, sr):
        if signal is None:
            return None
        array = np.asarray(signal, dtype=np.float32).reshape(-1)
        if not len(array):
            return {"samples": 0, "duration_seconds": 0.0,
                    "rms": 0.0, "peak": 0.0}
        return {
            "samples": len(array),
            "duration_seconds": len(array) / sr,
            "rms": float(np.sqrt(np.mean(array.astype(np.float64) ** 2))),
            "peak": float(np.max(np.abs(array))),
        }

    def _dump_failure_artifact(
        self, *, reason, detail, sr, waveform, overlap_start, overlap_end,
        speaker="?", segment_index=None, job=None, planner_actions=None,
        layout=None, window_audio=None, track_a=None, track_b=None,
        model_diag=None, service_actions=None, window_overlap=None,
    ):
        """Ghi một bundle tự đủ để điều tra lại một separation failure."""
        # Failure metadata là bắt buộc khi pipeline đã cấp dump_dir. Biến
        # BSS_DUMP_FAILED chỉ còn quyền tắt các track dump thông thường.
        if not self.dump_dir:
            return None
        try:
            self._failure_artifact_counter += 1
            safe_speaker = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in str(speaker)
            )
            name = (
                f"{self._failure_artifact_counter:04d}_"
                f"{overlap_start:.3f}-{overlap_end:.3f}_"
                f"{safe_speaker}_{reason}"
            )
            directory = os.path.join(self.dump_dir, "failed", name)
            while os.path.exists(directory):
                self._failure_artifact_counter += 1
                name = (
                    f"{self._failure_artifact_counter:04d}_"
                    f"{overlap_start:.3f}-{overlap_end:.3f}_"
                    f"{safe_speaker}_{reason}"
                )
                directory = os.path.join(self.dump_dir, "failed", name)
            os.makedirs(directory)

            source_start = max(0, round(overlap_start * sr))
            source_end = min(len(waveform), round(overlap_end * sr))
            overlap_mix = np.asarray(
                waveform[source_start:source_end], dtype=np.float32
            )
            planned_context = None
            planned_context_samples = None
            if window_audio is None:
                for action in reversed(planner_actions or []):
                    bounds = (
                        action.get("base_after")
                        or action.get("base_samples")
                        or (
                            action.get("source_samples")
                            if action.get("action") == "take_full_segment_envelope"
                            else None
                        )
                    )
                    if bounds and len(bounds) == 2:
                        context_start = max(0, int(bounds[0]))
                        context_end = min(len(waveform), int(bounds[1]))
                        if context_end > context_start:
                            planned_context_samples = [context_start, context_end]
                            planned_context = np.asarray(
                                waveform[context_start:context_end], dtype=np.float32
                            )
                            break
            overlap_track_a = overlap_track_b = None
            if window_overlap is not None:
                offset, length = map(int, window_overlap)
                offset = max(0, offset)
                length = max(0, length)
                if track_a is not None and length:
                    overlap_track_a = np.asarray(
                        track_a[offset:offset + length], dtype=np.float32
                    )
                if track_b is not None and length:
                    overlap_track_b = np.asarray(
                        track_b[offset:offset + length], dtype=np.float32
                    )

            audio_errors = []
            try:
                import soundfile as sf

                def write_audio(filename, signal):
                    if signal is None or not len(signal):
                        return
                    sf.write(
                        os.path.join(directory, filename),
                        np.asarray(signal, dtype=np.float32), sr,
                    )

                write_audio("overlap_mix.wav", overlap_mix)
                write_audio("planned_context_mix.wav", planned_context)
                write_audio("window_mix.wav", window_audio)
                write_audio("window_track_A.wav", track_a)
                write_audio("window_track_B.wav", track_b)
                write_audio("overlap_track_A.wav", overlap_track_a)
                write_audio("overlap_track_B.wav", overlap_track_b)
            except Exception as exc:
                audio_errors.append(f"{type(exc).__name__}: {exc}")

            metadata = {
                "artifact_version": 1,
                "policy_version": POLICY_VERSION,
                "failure": {
                    "reason": reason,
                    "detail": detail,
                    "speaker": str(speaker),
                    "segment_index": segment_index,
                    "overlap_seconds": [overlap_start, overlap_end],
                    "overlap_source_samples": [source_start, source_end],
                    "planned_context_source_samples": planned_context_samples,
                },
                "job": job or {},
                "planner": {
                    "actions": planner_actions or [],
                    "layout": layout,
                },
                "model": model_diag,
                "service_actions": service_actions or [],
                "audio_write_errors": audio_errors,
                "signals": {
                    "overlap_mix": self._signal_summary(overlap_mix, sr),
                    "planned_context_mix": self._signal_summary(planned_context, sr),
                    "window_mix": self._signal_summary(window_audio, sr),
                    "window_track_A": self._signal_summary(track_a, sr),
                    "window_track_B": self._signal_summary(track_b, sr),
                    "overlap_track_A": self._signal_summary(overlap_track_a, sr),
                    "overlap_track_B": self._signal_summary(overlap_track_b, sr),
                },
                "files": sorted(os.listdir(directory)) + ["metadata.json"],
            }
            with open(os.path.join(directory, "metadata.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(
                    metadata, handle, ensure_ascii=False, indent=2,
                    default=self._json_default,
                )
            self.failure_artifacts.append({
                "reason": reason,
                "speaker": str(speaker),
                "start": overlap_start,
                "end": overlap_end,
                "path": os.path.relpath(directory, self.dump_dir),
                "attempt": (job or {}).get("attempt", 0),
            })
            return directory
        except Exception as exc:
            if self.logger and not self._dump_warned:
                self._dump_warned = True
                self.logger.warning(
                    f"[TSE] failure artifact disabled: {type(exc).__name__}: {exc}"
                )
            return None

    def _finalize_failure_artifacts(self, speech, sr):
        """Keep failed attempts, annotating their final sample coverage after retries."""
        by_index = {seg.index: seg for seg in speech}
        for artifact in self.failure_artifacts:
            path = os.path.join(self.dump_dir, artifact["path"], "metadata.json")
            try:
                with open(path, encoding="utf-8") as handle:
                    metadata = json.load(handle)
                failure = metadata["failure"]
                targets = metadata.get("job", {}).get("targets", [])
                if failure.get("segment_index") is not None:
                    targets = [{"segment_index": failure["segment_index"],
                                "speaker": failure["speaker"],
                                "start": failure["overlap_seconds"][0],
                                "end": failure["overlap_seconds"][1]}]
                coverage = []
                for target in targets:
                    seg = by_index.get(target["segment_index"])
                    lo, hi = round(target["start"] * sr), round(target["end"] * sr)
                    completed = [(round(a * sr), round(b * sr))
                                 for a, b, sim in (seg.bss_spans if seg else []) if sim != -2.0]
                    missing = subtract([(lo, hi)], completed)
                    coverage.append({**target, "recovered": not missing,
                                     "remaining_source_samples": missing,
                                     "separated_samples": max(0, hi - lo) - sum(b - a for a, b in missing)})
                outcome = "recovered" if coverage and all(t["recovered"] for t in coverage) else (
                    "partially_recovered" if any(t["separated_samples"] for t in coverage) else "failed")
                metadata["recovery"] = {"outcome": outcome, "targets": coverage}
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, ensure_ascii=False, indent=2, default=self._json_default)
                artifact["final_outcome"] = outcome
            except Exception as exc:
                artifact["finalization_error"] = f"{type(exc).__name__}: {exc}"
                if self.logger:
                    self.logger.warning(f"[TSE] could not finalize failure artifact {path}: {exc}")

    # ------------------------------------------------------------------
    def passthrough(self, segments, audio):
        """Trả SpeechSegment chứa mixture khi cấu hình tắt separation.
        Giữ cùng cấu trúc với process_overlaps để ASR và bước xuất vẫn đọc audio."""
        sr = audio.sample_rate
        speech = [SpeechSegment(**s.__dict__) for s in segments]
        for e in speech:
            e.audio = audio.waveform[round(e.start * sr):round(e.end * sr)].copy()
        return speech

    def process_overlaps(self, segments: List[Segment], audio: AudioData, overlap_threshold: float = 0.1) -> List[SpeechSegment]:
        if not self.bss_model:
            # passthrough gắn mixture vào từng segment. Trả SpeechSegment rỗng
            # audio ở đây làm checkpoint, bước xuất clip và dual-channel đều
            # nhận về giá trị None mà không có dấu hiệu nào là separation đã
            # không chạy.
            if self.logger:
                self.logger.warning(
                    "[TSE] no separator is loaded; every overlap stays raw mixture")
            return self.passthrough(segments, audio)

        if self.logger:
            self.logger.info("Processing overlaps with blind source separation")

        seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "index": s.index} for s in segments]
        # Phát hiện mọi giao dương; ngưỡng chạy model chỉ áp dụng sau khi nhóm.
        pairs = detect_overlapping_segments(seg_dicts, overlap_threshold=0.0, logger=self.logger)

        # Bỏ qua micro-overlap (< 60ms): thường là khoảng lặng ở ranh giới
        # segment, không đáng tách và dễ gây insufficient_evidence khi retry.
       
        micro = [p for p in pairs if p["overlap_end"] - p["overlap_start"] < MIN_OVERLAP_SECONDS]
        if micro and self.logger:
            self.logger.info(
                f"[TSE] skipping {len(micro)} micro-overlap(s) < {MIN_OVERLAP_SECONDS*1000:.0f}ms")
        pairs = [p for p in pairs if p["overlap_end"] - p["overlap_start"] >= MIN_OVERLAP_SECONDS]

        # Tạo danh sách các SpeechSegment từ danh sách segments
        speech = [SpeechSegment(**s.__dict__) for s in segments]
        sr = audio.sample_rate
        waveform = audio.waveform
        total_dur = len(waveform) / sr
        for e in speech:
            e.audio = waveform[round(e.start * sr):round(e.end * sr)].copy()

        if not pairs:
            if self.logger:
                self.logger.info(
                    f"[TSE] no overlap >= {overlap_threshold}s among {len(segments)} segments"
                )
            return speech

        enrollments = self.mine_enrollments(segments, audio)
        seg_by_index = {s.index: s for s in speech}

        self._same_speaker_pairs = []
        # Tách hết, không lọc theo overlap_threshold. WindowPlanner giữ nguyên
        # core dù rất ngắn rồi lấy context có giới hạn và padding sạch để bù.
        queue = list(self._group_jobs(pairs))
        below = []  # giữ để không vỡ bss_spans tracking
        buildable = []
        for spk_a, spk_b, plist in queue:
            self.stats["jobs"] += 1
            self.stats["pairs"] += len(plist)
            for p in plist:
                self.overlap_durations.append(p["overlap_end"] - p["overlap_start"])
            targets = self._splice_pairs(plist)
            buildable.append((spk_a, spk_b, plist, targets))

        # same_speaker: hai segment cùng nhãn chồng nhau — không có gì để tách.
        # Mở rộng vùng base ra hai bên (±2s) rồi giữ nguyên mixture.
        # Không gọi model, không cần speaker thứ hai, không thay đổi audio.
        # Chỉ ghi vào bss_spans với sim=-2 (passthrough marker) để downstream
        # biết vùng này đã được xem xét, không phải bỏ sót.
        by_index = {e.index: e for e in speech}
        for p in self._same_speaker_pairs:
            lo, hi = p["overlap_start"], p["overlap_end"]
            # Same-speaker là passthrough; vùng đánh dấu vẫn có context nền.
            pad = 2.0
            lo_ext = max(0.0, lo - pad)
            hi_ext = min(total_dur, hi + pad)
            for side in ("seg1", "seg2"):
                enh = by_index.get(p[side].get("index"))
                if enh is None:
                    continue
                # Kiểm tra vùng chưa bị ghi bởi job khác
                if any(not (hi_ext <= a or lo_ext >= b) for a, b, _ in enh.bss_spans):
                    continue
                dst = round(lo_ext * sr) - round(enh.start * sr)
                limit = round(hi_ext * sr) - round(lo_ext * sr)
                if dst < 0 or dst + limit > len(enh.audio):
                    continue
                enh.bss_spans.append((lo_ext, lo_ext + limit / sr, -2.0))
            if self.logger:
                self.logger.info(
                    f"[TSE] same_speaker {lo:.2f}-{hi:.2f}s: "
                    f"base extended ±{pad}s, mixture kept")
        if self.logger:
            self.logger.info(
                f"[TSE] {len(pairs)} overlap pairs -> {len(queue)} separation jobs "
                "(missing enrollment uses best-effort/complement assignment)"
            )

        # Dựng cửa sổ song song khi file này đủ job bù chi phí dùng pool; job
        # build (CPU) không phụ thuộc lượt tách (GPU) trước nó nên chồng lấp
        # được -- xem utils/window_pool.py. Pool bản thân SỐNG SUỐT CẢ BATCH
        # (self._window_pool, tạo lười ở đây lần đầu, đóng bởi
        # close_window_pool() ở cuối stage 'separation' của cả batch) -- chỉ
        # phần shared memory của RIÊNG FILE NÀY (FileWindows) mở/đóng ở đây.
        # Lý do tách hai tầng: pool tạo lại mỗi file từng trả chi phí spawn
        # (~22.87s đo được trên Kaggle thật, 8 worker/4 core) N lần cho N file
        # trong một batch, thay vì trả một lần.
        #
        # Lỗi lúc mở file trên pool (môi trường không cho fork/spawn, hết RAM
        # cho shared_memory, ...) thì rơi về tuần tự cho file này thay vì làm
        # hỏng cả lượt chạy.
        #
        # use_vad=False: nhiều worker mới spawn cùng lúc tự tải Silero VAD gây
        # crash tầng native thật (libc++abi recursive_mutex, không phải
        # exception Python bắt được) khi cache torch.hub còn rỗng -- xem
        # window_pool.py. Cửa sổ dựng song song dùng energy-based cut thay vì
        # Silero, kém chính xác hơn một chút so với nhánh tuần tự (vốn dùng
        # Silero GPU của chính bss_model); đổi lấy không crash cả lượt chạy.
        file_windows = None
        if buildable and _worth_pooling(len(buildable)):
            pool_size = _resolve_pool_size()
            if pool_size > 0:
                try:
                    state = self._window_pool_state
                    with state["lock"]:
                        if state["pool"] is None:
                            state["pool"] = WindowBuildPool(
                                n_workers=pool_size,
                                max_pending=min(BSS_WINDOW_MAX_PENDING,
                                                max(1, pool_size * 2)))
                            if self.logger:
                                self.logger.info(
                                    f"[TSE] window pool started with {pool_size} worker "
                                    "process(es) (persists for the rest of this batch)")
                        self._window_pool = state["pool"]
                    file_windows = self._window_pool.open_file(
                        segments, pairs, waveform, sr, music_map=self.music_map,
                        seams=self.seams(), context_seconds=BSS_STITCH_EDGE_PAD,
                        max_context_seconds=BSS_STITCH_EDGE_MAX,
                        padding_min_seconds=BSS_PADDING_MIN_PER_SPEAKER,
                        search_seconds=BSS_STITCH_SEARCH, use_vad=False)
                    if self.logger:
                        self.logger.info("[TSE] building windows for this file in parallel")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"[TSE] window pool failed for this file ({type(e).__name__}: {e}); "
                            "falling back to sequential build")
                    file_windows = None

        recovery_planner = WindowPlanner(
            segments, pairs, waveform, sr, music_map=self.music_map,
            seams=self.seams(), vad=getattr(self.bss_model, "_vad", None),
            context_seconds=BSS_STITCH_EDGE_PAD,
            max_context_seconds=BSS_STITCH_EDGE_MAX,
            padding_min_seconds=BSS_PADDING_MIN_PER_SPEAKER,
            search_seconds=BSS_STITCH_SEARCH)
        if file_windows is not None:
            window_iter = file_windows.build_all([plist for _a, _b, plist, _t in buildable])
        else:
            planner = WindowPlanner(
                segments, pairs, waveform, sr, music_map=self.music_map,
                seams=self.seams(), vad=getattr(self.bss_model, "_vad", None),
                context_seconds=BSS_STITCH_EDGE_PAD,
                max_context_seconds=BSS_STITCH_EDGE_MAX,
                padding_min_seconds=BSS_PADDING_MIN_PER_SPEAKER,
                search_seconds=BSS_STITCH_SEARCH)

            def window_iter():
                for _a, _b, plist, _t in buildable:
                    try:
                        r = planner.build_many(plist)
                        yield r, planner.reason, planner.detail, list(planner.actions)
                    except Exception as exc:
                        actions = list(getattr(planner, "actions", []))
                        actions.append({
                            "step": len(actions) + 1,
                            "action": "window_builder_exception",
                            "detail": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        })
                        yield None, "window_error", actions[-1]["detail"], actions
            window_iter = window_iter()

        def expanded_window_iter():
            """Flatten one group into non-overlapping per-core model jobs."""
            for job, outcome in zip(buildable, window_iter):
                spk_a, spk_b, plist, targets = job
                plans, reason, detail, actions = outcome
                if plans is None:
                    yield job, (None, reason, detail, actions)
                    continue

                emitted = False
                for plan in plans:
                    core_lo, core_hi = plan.core_source_samples
                    if core_hi <= core_lo:
                        clipped_targets = list(targets)
                    else:
                        lo_seconds, hi_seconds = core_lo / sr, core_hi / sr
                        clipped_targets = [
                            (sd, max(lo, lo_seconds), min(hi, hi_seconds))
                            for sd, lo, hi in targets
                            if min(hi, hi_seconds) > max(lo, lo_seconds)
                        ]
                    if not clipped_targets:
                        continue
                    emitted = True
                    subjob = (spk_a, spk_b, plist, clipped_targets)
                    yield subjob, (
                        plan.window, plan.reason, plan.detail, plan.actions
                    )

                if not emitted:
                    yield job, (None, reason, detail or "no_core_targets", actions)

        pending_retries = collections.deque()
        previous_outputs = {}

        def retry_failed(failures, attempt, plist, speakers, actions):
            if not failures:
                return
            if attempt >= 2:
                for sd, lo, hi, reason, detail in failures:
                    self._fail(seg_by_index.get(sd["index"]), lo, hi, reason, detail)
                return
            next_attempt = attempt + 1
            lo = min(round(item[1] * sr) for item in failures)
            hi = max(round(item[2] * sr) for item in failures)
            bounds = [(lo, hi)]
            split_actions = []
            try:
                if hi - lo > recovery_planner.target:
                    recovery_planner.actions = []
                    bounds = recovery_planner._split_core_bounds(lo, hi)
                    split_actions.extend(recovery_planner.actions)
                elif next_attempt == 2 and hi - lo > 5 * sr:
                    midpoint = (lo + hi) // 2
                    decision = recovery_planner.cuts.find_cut(
                        midpoint, search_min=max(lo + sr, midpoint - sr),
                        search_max=min(hi - sr, midpoint + sr), hard_bounds=(lo + 1, hi - 1))
                    bounds = [(lo, decision.sample), (decision.sample, hi)]
                    split_actions.append({"action": "retry_acoustic_split",
                                          "sample": decision.sample, "method": decision.method})
            except Exception as exc:
                width = 5 * sr
                bounds = [(a, min(hi, a + width)) for a in range(lo, hi, width)]
                split_actions.append({"action": "retry_split_timestamp_fallback", "detail": str(exc)})
            for a, b in bounds:
                targets = [(sd, max(start, a / sr), min(end, b / sr))
                           for sd, start, end, _reason, _detail in failures
                           if min(end, b / sr) > max(start, a / sr)]
                if not targets:
                    continue
                trace = list(actions) + split_actions + [{
                    "action": "retry_window", "attempt": next_attempt,
                    "core_source_samples": [a, b],
                    "context_seconds": 1.0 if next_attempt == 1 else BSS_STITCH_EDGE_MAX,
                    "padding": True,    
                    "fill_context": next_attempt != 1,
                    "previous_failures": [{"speaker": sd["speaker"], "start": start,
                                           "end": end, "reason": reason, "detail": detail}
                                          for sd, start, end, reason, detail in failures],
                }]
                recovery_planner.context = round(
                    (1.0 if next_attempt == 1 else BSS_STITCH_EDGE_MAX) * sr
                )
                try:
                    retry = recovery_planner.build(
                        plist, core_bounds=(a, b), initial_actions=trace,
                        padding=True, fill_context=next_attempt != 1)
                    outcome = (retry, recovery_planner.reason, recovery_planner.detail,
                               list(recovery_planner.actions))
                except Exception as exc:
                    trace.append({"action": "retry_builder_error",
                                  "detail": str(exc), "traceback": traceback.format_exc()})
                    outcome = (None, "window_error", str(exc), trace)
                pending_retries.append(((speakers[0], speakers[1], plist, targets),
                                        outcome, next_attempt))
                self.stats["retried"] += 1

        def processing_iter():
            initial = iter(expanded_window_iter())
            while True:
                if pending_retries:
                    yield pending_retries.popleft()
                else:
                    try:
                        job, outcome = next(initial)
                    except StopIteration:
                        return
                    yield job, outcome, 0

        async_runtime = self._async_runtime()

        def scheduled_processing_iter():
            """Prefetch raw Sidon work while preserving file-local commit order.

            Raw separation is blind and has no dependency on enrollment memory.
            Speaker assignment, continuity, retries and writes remain in the
            consumer below, one ordered stream per SeparationService clone.
            """
            if async_runtime is None:
                for item in processing_iter():
                    yield item, None, None
                return

            gpu_executor, _post_executor, model = async_runtime
            prefetch_per_worker = max(
                1, int(self.performance_config.get("gpu_prefetch_per_worker", 1)))
            prefetch_limit = max(
                1, int(self.performance_config.get("max_workers", 1))
                * prefetch_per_worker)
            initial = iter(expanded_window_iter())
            scheduled = collections.deque()
            initial_done = False
            next_sequence = 0

            def schedule(item):
                nonlocal next_sequence
                job, outcome, attempt = item
                built = outcome[0]
                sequence_id = next_sequence
                next_sequence += 1
                future = None
                if built is not None:
                    # The future owns only immutable window audio. All mutable
                    # state stays in this file's ordered consumer.
                    future = gpu_executor.submit(
                        model.separate_raw, built.audio, sr)
                return item, future, sequence_id

            while True:
                if pending_retries:
                    retries = []
                    while pending_retries:
                        retries.append(pending_retries.popleft())
                    for retry in reversed(retries):
                        scheduled.appendleft(schedule(retry))

                while len(scheduled) < prefetch_limit and not initial_done:
                    try:
                        job, outcome = next(initial)
                    except StopIteration:
                        initial_done = True
                        break
                    scheduled.append(schedule((job, outcome, 0)))

                if not scheduled:
                    return
                yield scheduled.popleft()

        from tqdm import tqdm
        pbar = tqdm(desc="[TSE Extractor]", unit="window", leave=True)
        try:
            for ((spk_a, spk_b, plist, targets), (
                built, reason, detail, planner_actions
            ), attempt), raw_future, sequence_id in scheduled_processing_iter():
                pbar.update(1)
                _t_job = _time.perf_counter()
                uncovered = []
                for sd, lo, hi in targets:
                    enh = seg_by_index.get(sd["index"])
                    completed = [(round(a * sr), round(b * sr))
                                 for a, b, sim in (enh.bss_spans if enh else []) if sim != -2.0]
                    uncovered.extend((sd, a / sr, b / sr) for a, b in subtract(
                        [(round(lo * sr), round(hi * sr))], completed))
                targets = uncovered
                if not targets:
                    continue

                job_lo = min((lo for _sd, lo, _hi in targets), default=min(
                    p["overlap_start"] for p in plist
                ))
                job_hi = max((hi for _sd, _lo, hi in targets), default=max(
                    p["overlap_end"] for p in plist
                ))
                job_info = {
                    "attempt": attempt,
                    "speakers": [str(spk_a), str(spk_b)],
                    "overlap_seconds": [job_lo, job_hi],
                    "pairs": plist,
                    "targets": [
                        {
                            "segment_index": sd["index"],
                            "speaker": str(sd["speaker"]),
                            "start": lo,
                            "end": hi,
                        }
                        for sd, lo, hi in targets
                    ],
                }

                _t_window = _time.perf_counter()
                if _BSS_TIMING and self.logger:
                    _wait = _t_window - _t_job
                    self.logger.debug(
                        f"[TIMING] {job_lo:.2f}s: window_wait={_wait:.2f}s"
                        + (" (GPU idle!)" if _wait > 1.0 else ""))

                if built is None:
                    failures = [(sd, lo, hi, reason, detail) for sd, lo, hi in targets]
                    retry_failed(failures, 2 if reason == "multi_speaker" else attempt,
                                 plist, (spk_a, spk_b), planner_actions)
                    self.window_layouts.append({
                        "source_overlap": [job_lo, job_hi],
                        "status": "failed_attempt", "attempt": attempt, "reason": reason,
                        "detail": detail, "actions": planner_actions})
                    self._dump_failure_artifact(
                        reason=reason, detail=detail, sr=sr, waveform=waveform,
                        overlap_start=job_lo, overlap_end=job_hi,
                        speaker=f"{spk_a}+{spk_b}", job=job_info,
                        planner_actions=planner_actions,
                        service_actions=[{
                            "action": "build_window",
                            "status": "failed",
                            "reason": reason,
                            "detail": detail,
                        }],
                    )
                    continue
                window_audio, core = built.audio, built.core
                probe_a_s, probe_b_s = built.probes[spk_a], built.probes[spk_b]
                layout = built.layout
                layout["attempt"] = attempt
                planner_actions = layout.get("actions", planner_actions)
                self.window_layouts.append(layout)
                service_actions = [{
                    "action": "build_window",
                    "status": "ok",
                    "duration_seconds": len(window_audio) / sr,
                    "core_window_samples": list(core),
                }]
                # Map through core_source_samples because clean support may be
                # stitched before the continuous base and shift core in the window.
                core_src_lo = layout["core_source_samples"][0]
                win_lo = (core_src_lo - core[0]) / sr
                win_hi = win_lo + len(window_audio) / sr
                solo_a = [(a/sr, b/sr) for a, b in probe_a_s]
                solo_b = [(a/sr, b/sr) for a, b in probe_b_s]
                anchor = spk_a if sum(b-a for a,b in solo_a) >= sum(b-a for a,b in solo_b) else spk_b
                self.stats["stitched"] += 1
                if self.logger:
                    self.logger.info(
                        f"[BSS:window] {job_lo:.2f}-{job_hi:.2f}s -> "
                        f"{len(window_audio)/sr:.2f}s; overlap @ {core[0]/sr:.3f}s; "
                        f"balance={layout['ratio']:.2f}:1")

                # Nếu bật bộ nhớ, bổ sung mẫu từ các kết quả tốt trước đó trong cùng file.
                # Mẫu sạch khai thác ban đầu luôn ở đầu danh sách và không bị thay thế.
                memory = getattr(self, "memory", None) or _NO_MEMORY
                source_enroll_a = enrollments.get(spk_a, [])
                source_enroll_b = enrollments.get(spk_b, [])
                if not source_enroll_a:
                    source_enroll_a = self._window_probe_enrollment(
                        window_audio, probe_a_s, sr
                    )
                if not source_enroll_b:
                    source_enroll_b = self._window_probe_enrollment(
                        window_audio, probe_b_s, sr
                    )
                enroll_a = memory.extend(spk_a, source_enroll_a, sr)
                enroll_b = memory.extend(spk_b, source_enroll_b, sr)
                layout["enrollment_source"] = {
                    str(spk_a): (
                        "global" if enrollments.get(spk_a) else
                        "window_probe" if source_enroll_a else "missing"
                    ),
                    str(spk_b): (
                        "global" if enrollments.get(spk_b) else
                        "window_probe" if source_enroll_b else "missing"
                    ),
                }
                service_actions.append({
                    "action": "prepare_enrollment",
                    "sources": dict(layout["enrollment_source"]),
                })

                _t_sidon = _time.perf_counter()
                if _BSS_TIMING and self.logger:
                    self.logger.debug(
                        f"[TIMING] {job_lo:.2f}s: → Sidon "
                        f"file={self._file_run_id} seq={sequence_id}")
                try:
                    if raw_future is None:
                        track_A, track_B, sim_A, sim_B, diag = (
                            self.bss_model.separate_two_speakers(
                                window_audio,
                                enroll_A=enroll_a, enroll_B=enroll_b,
                                sample_rate=sr, id_A=spk_a, id_B=spk_b,
                                probe_A=probe_a_s,
                                probe_B=probe_b_s,
                                core_range=core,
                            )
                        )
                    else:
                        raw_tracks = raw_future.result()
                        post_future = async_runtime[1].submit(
                            async_runtime[2].postprocess_separated,
                            window_audio, raw_tracks,
                            enroll_A=enroll_a, enroll_B=enroll_b,
                            sample_rate=sr, id_A=spk_a, id_B=spk_b,
                            probe_A=probe_a_s, probe_B=probe_b_s,
                            core_range=core,
                        )
                        track_A, track_B, sim_A, sim_B, diag = post_future.result()
                except Exception as exc:
                    error_detail = f"{type(exc).__name__}: {exc}"
                    retry_failed([(sd, lo, hi, "model_error", error_detail)
                                  for sd, lo, hi in targets], attempt, plist,
                                 (spk_a, spk_b), planner_actions)
                    service_actions.append({
                        "action": "run_separator",
                        "status": "failed",
                        "detail": error_detail,
                        "traceback": traceback.format_exc(),
                    })
                    layout["status"] = "failed_attempt"
                    self._dump_failure_artifact(
                        reason="model_error", detail=error_detail,
                        sr=sr, waveform=waveform,
                        overlap_start=job_lo, overlap_end=job_hi,
                        speaker=f"{spk_a}+{spk_b}", job=job_info,
                        planner_actions=planner_actions, layout=layout,
                        window_audio=window_audio,
                        service_actions=service_actions,
                    )
                    continue
                service_actions.append({
                    "action": "run_separator",
                    "status": "ok",
                })
                previous = previous_outputs.get((spk_a, spk_b))
                continuity = continuity_evidence(previous, built, (track_A, track_B),
                                                 sr, BSS_SILENCE_RMS)
                strong_identity = (
                    diag.get("assignment_mode") == "backend_ordered"
                    or (max((v for v in (sim_A, sim_B) if v is not None), default=-1)
                        >= BSS_QC_SIM_THRESHOLD
                        and diag.get("assignment_margin", 0.0) >= BSS_NOT_A_MARGIN))
                if continuity["swap"] and not strong_identity:
                    track_A, track_B = track_B, track_A
                    scores = diag.get("output_scores", [[None, None], [None, None]])
                    sim_A, sim_B = scores[1][0], scores[0][1]
                    diag["output_scores"] = scores[::-1]
                    diag["assignment_mode"] = "shared_context_continuity"
                    continuity["applied"] = True
                else:
                    continuity["applied"] = False
                continuity["identity_locked"] = strong_identity
                layout["continuity"] = continuity
                diag["continuity"] = continuity
                service_actions.append({"action": "align_window_speakers", **continuity})
                _t_sidon_done = _time.perf_counter()
                if _BSS_TIMING and self.logger:
                    self.logger.debug(
                        f"[TIMING] {job_lo:.2f}s: ← Sidon {_t_sidon_done - _t_sidon:.2f}s")

                # Hiệu chỉnh level bằng clean probe của CHÍNH speaker, ưu tiên probe
                # có source timestamp thật gần overlap. Chỉ tính gain ở đây; KHÔNG
                # sửa track_A/B vì chúng còn được dùng cho QC và enrollment memory.
                overlap_center_source = int(0.5 * (job_lo + job_hi) * sr)
                gain_A, level_A = self._speaker_level_gain(
                    window_audio, track_A, probe_a_s, layout,
                    overlap_center_source, sr,
                )
                gain_B, level_B = self._speaker_level_gain(
                    window_audio, track_B, probe_b_s, layout,
                    overlap_center_source, sr,
                )
                level_gains = {spk_a: gain_A, spk_b: gain_B}
                level_info = {spk_a: level_A, spk_b: level_B}

                # layout đã được append vào self.window_layouts ở trên; mutate dict
                # này để report lưu luôn gain đã dùng cho chính window đó.
                layout["level_calibration"] = {
                    str(spk): {
                        "gain": float(info["gain"]),
                        "gain_db": float(info["gain_db"]),
                        "used_seconds": float(info["used_seconds"]),
                        "valid_seconds": float(info["valid_seconds"]),
                        "probe_count": int(info["probe_count"]),
                        "reason": info["reason"],
                    }
                    for spk, info in level_info.items()
                }
                service_actions.append({
                    "action": "calibrate_output_level",
                    "result": layout["level_calibration"],
                })

                if self.logger:
                    self.logger.info(
                        f"[TSE:level] {job_lo:.2f}-{job_hi:.2f}s "
                        f"{spk_a}={level_A['gain_db']:+.2f}dB({level_A['reason']}) "
                        f"{spk_b}={level_B['gain_db']:+.2f}dB({level_B['reason']})"
                    )

                _t_qc = _time.perf_counter()
                if _BSS_TIMING and self.logger:
                    self.logger.debug(
                        f"[TIMING] {job_lo:.2f}s: → QC ECAPA "
                        f"(sim_A={'?' if sim_A is None else f'{sim_A:.2f}'} "
                        f"sim_B={'?' if sim_B is None else f'{sim_B:.2f}'})")

                for sim in (sim_A, sim_B):
                    if sim is not None:
                        self.sims.append(sim)

                # BSS model đã ánh xạ hai output theo ECAPA nếu đo được một hoặc
                # cả hai phía. Track không đo được nhận nhãn còn lại bằng loại
                # trừ; similarity là confidence metadata, không còn là hard gate.
                # Hard gate lúc splice chỉ là output rỗng tại chính overlap.
                if _BSS_TIMING and self.logger:
                    self.logger.debug(f"[TIMING] {job_lo:.2f}s: → sim threshold check (th={BSS_QC_SIM_THRESHOLD})")
                accepted = {
                    spk_a: (track_A, sim_A),
                    spk_b: (track_B, sim_B),
                }
                assignment_warnings = {}
                for spk, track, sim in ((spk_a, track_A, sim_A), (spk_b, track_B, sim_B)):
                    if sim is None:
                        assignment_warnings[str(spk)] = "complement_or_unscorable"
                    elif sim < BSS_QC_SIM_THRESHOLD:
                        assignment_warnings[str(spk)] = (
                            f"low_sim:{sim:.3f}<{BSS_QC_SIM_THRESHOLD:.3f}"
                        )
                layout["assignment"] = {
                    "mode": diag.get("assignment_mode", "best_effort"),
                    "score_sources": diag.get("score_sources", {}),
                    "warnings": assignment_warnings,
                    "sim": {
                        str(spk_a): None if sim_A is None else float(sim_A),
                        str(spk_b): None if sim_B is None else float(sim_B),
                    },
                }
                service_actions.append({
                    "action": "assign_output_tracks",
                    "result": layout["assignment"],
                })

                # Ghi thông tin đầu vào, kết quả và điểm của các track đạt kiểm tra.
                if self.logger:
                    who = ", ".join(
                        f"{spk}=" + (f"{sim:.2f}" if sim is not None else "not-A")
                        for spk, (_, sim) in accepted.items()
                    )
                    self.logger.info(
                        f"[TSE:sep] {win_lo:.2f}-{win_hi:.2f}s ({win_hi - win_lo:.1f}s window) "
                        f"anchor={anchor} solo_a={sum(b - a for a, b in solo_a):.1f}s "
                        f"solo_b={sum(b - a for a, b in solo_b):.1f}s | accepted: {who}"
                        + (f" | warnings: {assignment_warnings}" if assignment_warnings else "")
                    )
                if _BSS_TIMING and self.logger:
                    self.logger.debug(f"[TIMING] {job_lo:.2f}s: → dump_tracks (accepted={sorted(accepted)})")
                _t_dump = _time.perf_counter()
                self._dump_tracks("separated", f"{job_lo:.2f}_{spk_a}_{spk_b}",
                                  window_audio, track_A, track_B, sr)
                if _BSS_TIMING and self.logger:
                    self.logger.debug(
                        f"[TIMING] {job_lo:.2f}s: ← dump_tracks {_time.perf_counter()-_t_dump:.2f}s")

                if _BSS_TIMING and self.logger:
                    self.logger.debug(
                        f"[TIMING] {job_lo:.2f}s: → splice loop ({len(targets)} target(s))")
                _t_splice = _time.perf_counter()
                fade_samples = int(0.02 * sr)
                failed_targets = []
                original_targets = self._splice_pairs(plist)
                spliced_count = 0
                spliced_speakers = set()
                for sd, ov_lo, ov_hi in targets:
                    spk = sd["speaker"]
                    enh = seg_by_index.get(sd["index"])
                    if enh is None:
                        continue
                    track, sim = accepted[spk]
                    sample_lo, sample_hi = round(ov_lo * sr), round(ov_hi * sr)
                    src = core[0] + sample_lo - core_src_lo
                    dst = sample_lo - round(enh.start * sr)
                    if src < 0 or dst < 0:
                        failure_detail = f"negative offset src={src} dst={dst}"
                        failed_targets.append((sd, ov_lo, ov_hi, "short_track", failure_detail))
                        self._dump_failure_artifact(
                            reason="short_track", detail=failure_detail,
                            sr=sr, waveform=waveform,
                            overlap_start=ov_lo, overlap_end=ov_hi,
                            speaker=spk, segment_index=sd["index"], job=job_info,
                            planner_actions=planner_actions, layout=layout,
                            window_audio=window_audio, track_a=track_A,
                            track_b=track_B, model_diag=diag,
                            service_actions=service_actions + [{
                                "action": "map_overlap_to_window",
                                "status": "failed", "src": src, "dst": dst,
                            }],
                        )
                        continue
                    limit = min(sample_hi - sample_lo, len(track) - src,
                                len(enh.audio) - dst)
                    if limit <= 0 or limit < sample_hi - sample_lo:
                        failure_detail = f"limit={limit} src={src} dst={dst}"
                        failed_targets.append((sd, ov_lo, ov_hi, "short_track", failure_detail))
                        self._dump_failure_artifact(
                            reason="short_track", detail=failure_detail,
                            sr=sr, waveform=waveform,
                            overlap_start=ov_lo, overlap_end=ov_hi,
                            speaker=spk, segment_index=sd["index"], job=job_info,
                            planner_actions=planner_actions, layout=layout,
                            window_audio=window_audio, track_a=track_A,
                            track_b=track_B, model_diag=diag,
                            service_actions=service_actions + [{
                                "action": "map_overlap_to_window",
                                "status": "failed", "src": src, "dst": dst,
                                "limit": limit,
                            }],
                        )
                        continue
                    if any(not (sample_hi <= round(a * sr) or sample_lo >= round(b * sr))
                           for a, b, score in enh.bss_spans if score != -2.0):
                        failure_detail = "target overlaps an existing bss_spans entry"
                        self._fail(enh, ov_lo, ov_hi, "already_spliced", failure_detail)
                        self._dump_failure_artifact(
                            reason="already_spliced", detail=failure_detail,
                            sr=sr, waveform=waveform,
                            overlap_start=ov_lo, overlap_end=ov_hi,
                            speaker=spk, segment_index=sd["index"], job=job_info,
                            planner_actions=planner_actions, layout=layout,
                            window_audio=window_audio, track_a=track_A,
                            track_b=track_B, model_diag=diag,
                            service_actions=service_actions + [{
                                "action": "check_existing_splice",
                                "status": "failed", "src": src, "dst": dst,
                                "limit": limit,
                            }],
                            window_overlap=(src, limit),
                        )
                        continue

                    # Kiểm tra ngay vùng sắp ghép trả có lời. Điểm tốt trên solo ở xa không
                    # bảo đảm overlap có lời: từng có trường hợp sim=0.67 trên mẫu trước đó
                    # 19 giây nhưng track im lặng ở chính overlap.
                    # So track với mixture GỐC (waveform), không phải enh.audio
                    # vì enh.audio có thể đã bị splice bởi job trước -> RMS bị lệch
                    # -> _track_has_speech trả False dù track có âm thanh thật.
                    mix_dst = sample_lo
                    host = waveform[mix_dst:mix_dst + limit]
                    if _BSS_TIMING and self.logger:
                        self.logger.debug(
                            f"[TIMING] {job_lo:.2f}s: → _track_has_speech "
                            f"seg={sd['index']} spk={spk} limit={limit/sr:.3f}s")
                    quality = track_quality(host, track[src:src + limit], sr, BSS_SILENCE_RMS)
                    layout.setdefault("target_quality", []).append({
                        "speaker": spk, "source_samples": [sample_lo, sample_hi], **quality})
                    if not quality["accepted"]:
                        failure_reason = (quality["status"] if quality["status"] in REASONS
                                          else "empty_track")
                        failure_detail = quality["status"]
                        failed_targets.append((sd, ov_lo, ov_hi, failure_reason, failure_detail))
                        self._dump_failure_artifact(
                            reason=failure_reason, detail=failure_detail,
                            sr=sr, waveform=waveform,
                            overlap_start=ov_lo, overlap_end=ov_hi,
                            speaker=spk, segment_index=sd["index"], job=job_info,
                            planner_actions=planner_actions, layout=layout,
                            window_audio=window_audio, track_a=track_A,
                            track_b=track_B, model_diag=diag,
                            service_actions=service_actions + [{
                                "action": "validate_overlap_track",
                                "status": "failed", "src": src, "dst": dst,
                                "limit": limit,
                                "quality": quality, "attempt": attempt,
                            }],
                            window_overlap=(src, limit),
                        )
                        continue

                    # Không match level với mixture overlap: mixture chứa cả hai
                    # speaker nên thường lớn hơn từng track riêng và dễ làm patch bị
                    # boost. Dùng gain bias đã đo từ clean probe cùng speaker.
                    level_gain = float(level_gains.get(spk, 1.0))
                    if _BSS_TIMING and self.logger:
                        self.logger.debug(
                            f"[TIMING] {job_lo:.2f}s: → calibrated splice + crossfade "
                            f"seg={sd['index']} spk={spk} "
                            f"gain={20.0*np.log10(level_gain + 1e-12):+.2f}dB")

                    patch = (
                        np.asarray(track[src:src + limit], dtype=np.float32).copy()
                        * level_gain
                    )
                    spans = [(round(a * sr), round(b * sr)) for original, a, b in original_targets
                             if original["index"] == sd["index"]]
                    internal_left = any(a < sample_lo < b for a, b in spans)
                    internal_right = any(a < sample_hi < b for a, b in spans)
                    # Crossfade only separated sources at internal boundaries.
                    # Never reintroduce the original two-speaker mixture there.
                    if (internal_left and previous is not None
                        and spk in previous["speakers"] and not (
                            continuity["swap"] and strong_identity)):
                        old_lo, old_hi = previous["bounds"]
                        take = min(fade_samples, limit, old_hi - sample_lo)
                        if old_lo <= sample_lo and take > 0:
                            index = 0 if spk == spk_a else 1
                            old = previous["tracks"][index][sample_lo - old_lo:sample_lo - old_lo + take]
                            if len(old) == take:
                                ramp = np.linspace(0, 1, take, dtype=np.float32)
                                patch[:take] = old * (1 - ramp) + patch[:take] * ramp

                    # Limiter chỉ là hàng rào chống clipping sau khi áp gain; nó
                    # không dùng mixture làm reference và không normalize patch.
                    patch, patch_limit_gain = safe_limit(patch)
                    if self.logger and patch_limit_gain < 0.999:
                        self.logger.debug(
                            f"[TSE:level] limiter seg={sd['index']} spk={spk} "
                            f"x{patch_limit_gain:.3f}"
                        )

                    enh.audio[dst:dst + limit] = self._cross_fade(
                        enh.audio[dst:dst + limit], patch, fade_samples,
                        fade_left=not internal_left, fade_right=not internal_right)
                    enh.bss = True
                    enh.bss_spans.append((sample_lo / sr, (sample_lo + limit) / sr,
                                          float(sim) if sim is not None else -1.0))
                    self.stats["spliced"] += 1
                    spliced_count += 1
                    spliced_speakers.add(spk)
                    if self.logger:
                        self.logger.info(
                            f"[TSE:splice] seg {sd['index']} spk={spk} "
                            f"{ov_lo:.2f}-{ov_lo + limit / sr:.2f}s "
                            f"({limit / sr:.2f}s) sim="
                            + (f"{sim:.2f}" if sim is not None else "not-A")
                        )

                if spliced_count:
                    bounds, tracks = base_view(built, (track_A * gain_A, track_B * gain_B))
                    if all(np.isfinite(t).all() for t in tracks):
                        previous_outputs[(spk_a, spk_b)] = {
                            "bounds": bounds, "speakers": spliced_speakers,
                            "tracks": tuple(t if spk in spliced_speakers else np.zeros_like(t)
                                            for spk, t in zip((spk_a, spk_b), tracks))}
                    for spk in spliced_speakers:
                        track, sim = accepted[spk]
                        memory.offer(spk, track, sim, sr)
                layout["status"] = "partial" if failed_targets and spliced_count else (
                    "failed_attempt" if failed_targets else "spliced")
                layout["spliced_targets"] = spliced_count
                layout["failed_targets"] = len(failed_targets)
                retry_failed(failed_targets, attempt, plist, (spk_a, spk_b),
                             planner_actions)

                if _BSS_TIMING and self.logger:
                    _t_end = _time.perf_counter()
                    self.logger.debug(
                        f"[TIMING] {job_lo:.2f}s: ← splice loop {_t_end - _t_splice:.3f}s")
                    self.logger.debug(
                        f"[TIMING] {job_lo:.2f}s: TOTAL={_t_end-_t_job:.2f}s | "
                        f"window_wait={_t_window-_t_job:.2f}s "
                        f"enroll={_t_sidon-_t_window:.3f}s "
                        f"sidon={_t_sidon_done-_t_sidon:.2f}s "
                        f"qc={_t_qc-_t_sidon_done:.3f}s "
                        f"dump={_t_end-_t_dump:.3f}s "
                        f"splice={_t_end-_t_splice:.3f}s")
        finally:
            pbar.close()
            # Chỉ đóng shared memory của FILE NÀY -- pool process (nếu có)
            # sống tiếp cho file kế trong batch, đóng ở close_window_pool().
            if file_windows is not None:
                file_windows.close()
        self._finalize_failure_artifacts(speech, sr)
        self._report_stats()
        return speech

    # ------------------------------------------------------------------
    def export_sdlm_dual_channel(self, speech_segments: List[SpeechSegment], audio_duration: float, sr: int,
                                 strict: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Dựng hai track liên tục để huấn luyện SDLM/full-duplex.
        strict=True đặt vùng tách thất bại về zero để không đưa giọng người khác
        vào track có nhãn của speaker mục tiêu."""
        total_samples = round(audio_duration * sr)
        track_0 = np.zeros(total_samples, dtype=np.float32)
        track_1 = np.zeros(total_samples, dtype=np.float32)

        speakers = sorted({s.speaker for s in speech_segments})
        if not speakers:
            return track_0, track_1
        if len(speakers) > 2 and self.logger:
            self.logger.warning(
                f"export_sdlm_dual_channel: expected 2 speakers, found {len(speakers)}; "
                "the rest are ignored."
            )
        spk_0 = speakers[0]
        spk_1 = speakers[1] if len(speakers) > 1 else None

        # Hai segment cùng speaker chồng nhau chứa cùng một đoạn ghi âm. Cộng cả
        # hai vào track làm biên độ vùng đó gấp đôi rồi safe_limit kéo cả track
        # xuống để bù, nên chỉ lấy bản đầu tiên chạm tới mỗi mẫu.
        filled = {spk: np.zeros(total_samples, dtype=bool)
                  for spk in (spk_0, spk_1) if spk is not None}

        written = dropped = 0
        for seg in speech_segments:
            if seg.speaker not in filled or seg.audio is None:
                continue
            start_idx = round(seg.start * sr)
            end_idx = min(total_samples, start_idx + len(seg.audio))
            if end_idx <= start_idx:
                continue
            chunk = seg.audio[: end_idx - start_idx].copy()

            fresh = ~filled[seg.speaker][start_idx:end_idx]
            filled[seg.speaker][start_idx:end_idx] = True

            if strict:
                keep = np.array(fresh)
                # Checkpoint cũ có thể chưa có bss_failed_spans; pickle không tự điền
                # giá trị mặc định mới của dataclass nên đọc bằng getattr.
                for a, b, reason, _detail in getattr(seg, "bss_failed_spans", ()):
                    # Vùng chỉ có một người nói không mang giọng ai khác sang
                    # track sai; xóa nó chỉ làm mất lời nói thật.
                    if reason in SAFE_FAIL_REASONS:
                        continue
                    i = max(start_idx, round(a * sr)) - start_idx
                    j = min(end_idx, round(b * sr)) - start_idx
                    if j > i:
                        keep[i:j] = False
                dropped += int((fresh & ~keep).sum())
                written += int(keep.sum())
                chunk = chunk * keep
            else:
                written += int(fresh.sum())
                chunk = chunk * fresh

            if seg.speaker == spk_0:
                track_0[start_idx:end_idx] += chunk
            else:
                track_1[start_idx:end_idx] += chunk

        if strict and self.logger:
            total = written + dropped
            pct = (100.0 * dropped / total) if total else 0.0
            self.logger.info(
                f"[SDLM] wrote {written / sr:.1f}s, zeroed {dropped / sr:.1f}s ({pct:.1f}%) "
                "of un-separated overlap to avoid cross-speaker leakage."
            )
            if pct > 5.0:
                self.logger.warning(
                    f"[SDLM] {pct:.1f}% of speech dropped as contaminated -- that is the "
                    "TSE failure rate landing in your dataset. Check the [TSE] counters."
                )

        # Cộng các segment cùng speaker có thể vượt mức biên độ. Giảm đồng đều
        # bằng hệ số để giữ tương quan mức, tránh méo do cắt đỉnh cứng.
        track_0, g0 = safe_limit(track_0)
        track_1, g1 = safe_limit(track_1)
        if self.logger and (g0 < 1.0 or g1 < 1.0):
            self.logger.info(
                f"[SDLM] limiter applied: track_0 x{g0:.3f}, track_1 x{g1:.3f} "
                "(summed segments exceeded full scale)")
        return track_0, track_1
