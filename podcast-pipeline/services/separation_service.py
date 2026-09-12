import collections
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from schemas.audio import AudioData
from schemas.segment import Segment, SpeechSegment
from algorithms.diarization.overlap import detect_overlapping_segments
from utils.audio_normalize import match_splice_level, safe_limit
from utils.separation_window import POLICY_VERSION, WindowPlanner, clean_segments, merge_ranges
from utils.window_pool import WindowBuildPool
from utils.cpu_plan import usable_cores
from utils.music_map import MusicMap
from utils.enrollment_memory import EnrollmentMemory, ENABLED as BSS_MEMORY

# Đối tượng thay thế khi dịch vụ không dùng bộ nhớ mẫu giọng.
_NO_MEMORY = EnrollmentMemory(enabled=False)


# Ngữ cảnh thật quanh overlap; điểm cắt phải nằm trong vùng sạch.
BSS_STITCH_EDGE_PAD = float(os.environ.get("BSS_STITCH_EDGE_PAD", "2.0"))
BSS_STITCH_SEARCH = float(os.environ.get("BSS_STITCH_SEARCH", "400.0"))
BSS_MIN_SOLO = 1.0

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
BSS_QC_SIM_THRESHOLD = float(os.environ.get("BSS_QC_SIM_THRESHOLD", "0.50"))
BSS_NOT_A_MARGIN = float(os.environ.get("BSS_NOT_A_MARGIN", "0.15"))
BSS_SILENCE_RMS = float(os.environ.get("BSS_SILENCE_RMS", "0.002"))

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
                 model_loader=None):
        self._bss_model = bss_model
        self.model_loader = model_loader
        self.logger = logger
        self.dump_dir = dump_dir
        self.stats = collections.Counter()
        self.sims = []
        self.overlap_durations = []
        self.window_layouts = []
        self.failures = []          # (bắt đầu, kết thúc, speaker, lý do, chi tiết)
        self._dump_warned = False
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

    # Lấy model từ loader khi dùng vì model được tải theo từng giai đoạn.
    # Giữ tham chiếu lúc khởi tạo có thể giữ mãi giá trị None của model chưa tải.
    @property
    def bss_model(self):
        if self._bss_model is not None:
            return self._bss_model
        return getattr(self, "model_loader", None) and self.model_loader.get("separator")

    @bss_model.setter
    def bss_model(self, model):
        """Cho phép truyền model trực tiếp, dùng khi tạo dịch vụ trong kiểm thử."""
        self._bss_model = model

    def reset_stats(self):
        """Xóa thống kê từng file vì cùng một dịch vụ được dùng lại cho cả batch.
        Nếu không xóa, báo cáo file sau sẽ chứa số cộng dồn từ file trước."""
        self.stats = collections.Counter()
        self.sims = []
        self.overlap_durations = []
        self.window_layouts = []
        self.failures = []
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
        pool = self._window_pool
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
                "support_min_seconds": 1.5,
                "secondary_edge_margin_seconds": 1.0,
                "crossfade_seconds": 0.02,
                "window_max": 15.0,
                "min_solo": BSS_MIN_SOLO,
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
                        "so every overlap involving this speaker is skipped."
                    )
                enrollments[spk] = []
            else:
                enrollments[spk] = picked

        return enrollments

    @staticmethod
    def _track_has_speech(host: np.ndarray, track: np.ndarray,
                          frame_sec: float = 0.02, sr_hint: int = 24000) -> bool:
        """Kiểm tra track có lời tại những nơi mixture có lời.
        So năng lượng từng khung ngắn: RMS toàn đoạn có thể bỏ sót trường hợp
        track im lặng gần hết overlap nhưng chỉ có đuôi giọng khác ở cuối."""
        n = min(len(host), len(track))
        if n == 0:
            return False
        frame = max(1, int(frame_sec * sr_hint))
        if n < frame * 2:
            # Đoạn quá ngắn để chia khung: kiểm tra còn năng lượng hay không.
            return float(np.sqrt(np.mean(track[:n] ** 2) + 1e-12)) >= BSS_SILENCE_RMS

        m = n // frame
        h = np.sqrt((host[:m * frame].reshape(m, frame) ** 2).mean(axis=1) + 1e-12)
        t = np.sqrt((track[:m * frame].reshape(m, frame) ** 2).mean(axis=1) + 1e-12)

        # Xác định khung có lời theo đỉnh năng lượng của chính mixture.
        voiced = h >= max(h.max() * 0.25, BSS_SILENCE_RMS)
        if not voiced.any():
            return True          # Mixture im lặng nên không có lời cần bảo toàn

        # Track đã bỏ giọng nhiễu có thể nhỏ hơn; so theo thang của chính track.
        # Sàn tuyệt đối bằng đúng ngưỡng im lặng dùng ở cổng unscorable: chỉ so
        # theo t.max() thì một track nhiễu đều ở mức rất thấp cũng có mọi khung
        # vượt ngưỡng, đúng kiểu lọt mà _gather_probe đã phải thêm sàn để chặn.
        alive = t >= max(t.max() * 0.15, BSS_SILENCE_RMS)
        return float(np.mean(alive[voiced])) >= 0.35

    def _cross_fade(self, orig_audio: np.ndarray, new_audio: np.ndarray, fade_samples: int) -> np.ndarray:
        """Thay vùng âm thanh gốc bằng track đã tách, làm mượt hai biên.
        Dùng ramp sin/cos giữ công suất thay vì ramp tuyến tính làm hụt mức giữa
        mối nối của hai tín hiệu ít tương quan. Giới hạn ramp theo độ dài vùng
        thay thế để đoạn overlap ngắn vẫn giữ ít nhất 3/4 phần giữa ở mức đầy đủ.
        Đây là phép ghép trả, khác crossfade chồng 20 ms khi dựng đầu vào."""
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
        ramp_in = np.sin(t * (np.pi / 2.0))
        ramp_out = np.cos(t * (np.pi / 2.0))

        # Chuyển dần từ âm thanh gốc sang track đã tách.
        result[:fade_samples] = (orig_audio[:fade_samples] * ramp_out
                                 + new_audio[:fade_samples] * ramp_in)

        # Phần giữa dùng nguyên mức âm thanh đã tách.
        mid_limit = limit - fade_samples
        result[fade_samples:mid_limit] = new_audio[fade_samples:mid_limit]

        # Chuyển dần về âm thanh gốc ở biên cuối.
        result[mid_limit:limit] = (orig_audio[mid_limit:limit] * ramp_in
                                   + new_audio[mid_limit:limit] * ramp_out)
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
        Sau khi cắt nhạc, hai vùng vốn không liền nhau có thể nằm sát nhau. Không
        mở rộng cửa sổ xuyên mối nối đó vì sẽ kéo thêm ngữ cảnh khác vào model.
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
            sf.write(os.path.join(d, f"{tag}_mix.wav"), mixture, sr)
            if track_1 is not None:
                sf.write(os.path.join(d, f"{tag}_trackA.wav"), track_1, sr)
                sf.write(os.path.join(d, f"{tag}_trackB.wav"), track_2, sr)
        except Exception as e:
            # Chỉ cảnh báo một lần nếu thiếu soundfile hoặc không ghi được thư mục,
            # tránh lặp thông báo trên từng clip nhưng vẫn báo mất dữ liệu đối chiếu.
            if self.logger and not self._dump_warned:
                self._dump_warned = True
                self.logger.warning(
                    f"[TSE] track dumps disabled: {type(e).__name__}: {e}"
                )

    def _dump_failed(self, tag, mixture, track_1, track_2, sr):
        self._dump_tracks("failed", tag, mixture, track_1, track_2, sr)

    # ------------------------------------------------------------------
    def passthrough(self, segments, audio):
        """Trả SpeechSegment chứa mixture khi cấu hình tắt separation.
        Giữ cùng cấu trúc với process_overlaps để ASR và bước xuất vẫn đọc audio."""
        sr = audio.sample_rate
        speech = [SpeechSegment(**s.__dict__) for s in segments]
        for e in speech:
            e.audio = audio.waveform[int(e.start * sr):int(e.end * sr)].copy()
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

        # Tạo danh sách các SpeechSegment từ danh sách segments
        speech = [SpeechSegment(**s.__dict__) for s in segments]
        sr = audio.sample_rate
        waveform = audio.waveform
        total_dur = len(waveform) / sr
        for e in speech:
            e.audio = waveform[int(e.start * sr):int(e.end * sr)].copy()

        if not pairs:
            if self.logger:
                self.logger.info(
                    f"[TSE] no overlap >= {overlap_threshold}s among {len(segments)} segments"
                )
            return speech

        enrollments = self.mine_enrollments(segments, audio)
        seg_by_index = {s.index: s for s in speech}

        self._same_speaker_pairs = []
        queue, below = [], []
        for job in self._group_jobs(pairs):
            span = (max(p["overlap_end"] for p in job[2])
                    - min(p["overlap_start"] for p in job[2]))
            (queue if span >= overlap_threshold else below).append(job)
        # Quá ngắn để chạy model vẫn là overlap. Bỏ qua mà không ghi lý do thì
        # vùng đó vắng mặt trong cả bss_spans lẫn bss_failed_spans, và bước xuất
        # dual-channel đọc mixture ở đó như thể là giọng sạch của một người.
        for _a, _b, plist in below:
            for sd, lo, hi in self._splice_pairs(plist):
                self._fail(seg_by_index.get(sd["index"]), lo, hi, "below_threshold",
                           f"span={hi - lo:.3f}s th={overlap_threshold}")
        # Bỏ trước những job thiếu enrollment: build cửa sổ cho chúng chỉ để
        # vứt đi ngay sau đó là lãng phí CPU (và với pool, cả RAM/IPC).
        buildable = []
        for spk_a, spk_b, plist in queue:
            self.stats["jobs"] += 1
            self.stats["pairs"] += len(plist)
            for p in plist:
                self.overlap_durations.append(p["overlap_end"] - p["overlap_start"])
            targets = self._splice_pairs(plist)
            if not enrollments.get(spk_a) or not enrollments.get(spk_b):
                missing = spk_a if not enrollments.get(spk_a) else spk_b
                for sd, lo, hi in targets:
                    self._fail(seg_by_index.get(sd["index"]), lo, hi, "no_enroll",
                               f"speaker={missing}")
                continue
            buildable.append((spk_a, spk_b, plist, targets))

        # Ghi nhận overlap cùng speaker trên cả hai segment để không mất dấu.
        by_index = {e.index: e for e in speech}
        for p in self._same_speaker_pairs:
            for side in ("seg1", "seg2"):
                self._fail(by_index.get(p[side].get("index")),
                           p["overlap_start"], p["overlap_end"],
                           "same_speaker", f"speaker={p[side].get('speaker')}")
        if self.logger:
            self.logger.info(f"[TSE] {len(pairs)} overlap pairs -> {len(queue)} separation jobs "
                              f"({len(buildable)} with enrollment)")

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
                    if self._window_pool is None:
                        self._window_pool = WindowBuildPool(n_workers=pool_size)
                        if self.logger:
                            self.logger.info(
                                f"[TSE] window pool started with {pool_size} worker "
                                "process(es) (persists for the rest of this batch)")
                    file_windows = self._window_pool.open_file(
                        segments, pairs, waveform, sr, music_map=self.music_map,
                        seams=self.seams(), context_seconds=BSS_STITCH_EDGE_PAD,
                        search_seconds=BSS_STITCH_SEARCH, use_vad=False)
                    if self.logger:
                        self.logger.info("[TSE] building windows for this file in parallel")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"[TSE] window pool failed for this file ({type(e).__name__}: {e}); "
                            "falling back to sequential build")
                    file_windows = None

        if file_windows is not None:
            window_iter = file_windows.build_all([plist for _a, _b, plist, _t in buildable])
        else:
            planner = WindowPlanner(
                segments, pairs, waveform, sr, music_map=self.music_map,
                seams=self.seams(), vad=getattr(self.bss_model, "_vad", None),
                context_seconds=BSS_STITCH_EDGE_PAD, search_seconds=BSS_STITCH_SEARCH)

            def window_iter():
                for _a, _b, plist, _t in buildable:
                    r = planner.build(plist)
                    yield r, planner.reason, planner.detail
            window_iter = window_iter()

        from tqdm import tqdm
        pbar = tqdm(total=len(buildable), desc="[TSE Extractor]", leave=True)
        try:
            for (spk_a, spk_b, plist, targets), (built, reason, detail) in zip(buildable, window_iter):
                pbar.update(1)

                def fail_all(reason, detail="", targets=targets):
                    for sd, lo, hi in targets:
                        self._fail(seg_by_index.get(sd["index"]), lo, hi, reason, detail)

                job_lo = min(p["overlap_start"] for p in plist)
                job_hi = max(p["overlap_end"] for p in plist)

                if built is None:
                    fail_all(reason, detail)
                    self.window_layouts.append({
                        "source_overlap": [job_lo, job_hi],
                        "status": "skipped", "reason": detail})
                    continue
                window_audio, core = built.audio, built.core
                probe_a_s, probe_b_s = built.probes[spk_a], built.probes[spk_b]
                layout = built.layout
                self.window_layouts.append(layout)
                # core[0] is where core_source_samples[0] lands in the window, and
                # that is NOT job_lo: for an overlap under short_core_threshold the
                # planner pads the core outward, so the two differ by up to the pad.
                # Mapping window <-> source through job_lo read the window 2s early.
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
                enroll_a = memory.extend(spk_a, enrollments[spk_a], sr)
                enroll_b = memory.extend(spk_b, enrollments[spk_b], sr)

                track_A, track_B, sim_A, sim_B, diag = self.bss_model.separate_two_speakers(
                    window_audio,
                    enroll_A=enroll_a, enroll_B=enroll_b,
                    sample_rate=sr, id_A=spk_a, id_B=spk_b,
                    probe_A=probe_a_s,
                    probe_B=probe_b_s,
                    core_range=core,
                )

                for sim in (sim_A, sim_B):
                    if sim is not None:
                        self.sims.append(sim)

                # Chỉ nhận vào bộ nhớ các track đạt ngưỡng cao để hạn chế học nhầm giọng.
                for spk, track, sim in ((spk_a, track_A, sim_A), (spk_b, track_B, sim_B)):
                    memory.offer(spk, track, sim, sr)

                # Đánh giá từng track độc lập; một track kém không làm mất track còn tốt.
                accepted, rejected = {}, {}
                for spk, track, sim in ((spk_a, track_A, sim_A), (spk_b, track_B, sim_B)):
                    if sim is not None:
                        if sim >= BSS_QC_SIM_THRESHOLD:
                            accepted[spk] = (track, sim)
                        else:
                            rejected[spk] = ("qc_sim", f"sim={sim:.2f} th={BSS_QC_SIM_THRESHOLD}")
                        continue

                    # Nếu thiếu solo để chấm điểm, dùng phép so tương đối với giọng neo
                    # để kiểm tra track có chỉ là bản sao của giọng đó hay không.
                    own, other, rms = diag["anchor_self"], diag["anchor_other"], diag["other_rms"]
                    if rms is not None and rms < BSS_SILENCE_RMS:
                        rejected[spk] = ("unscorable", f"rms={rms:.5f}")
                    elif own is None or other is None:
                        rejected[spk] = ("unscorable", "no core embedding")
                    elif (own - other) > BSS_NOT_A_MARGIN:
                        accepted[spk] = (track, None)
                        self.stats["accept_not_a"] += 1
                    else:
                        rejected[spk] = ("not_a_fail",
                                         f"margin={own - other:.2f} th={BSS_NOT_A_MARGIN}")

                if not accepted:
                    for sd, lo, hi in targets:
                        r, d = rejected.get(sd["speaker"], ("unscorable", "no verdict"))
                        self._fail(seg_by_index.get(sd["index"]), lo, hi, r, d)
                    self._dump_failed(f"{job_lo:.2f}_{spk_a}_{spk_b}", window_audio, track_A, track_B, sr)
                    continue

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
                        + (f" | rejected: {sorted(rejected)}" if rejected else "")
                    )
                self._dump_tracks("separated", f"{job_lo:.2f}_{spk_a}_{spk_b}",
                                  window_audio, track_A, track_B, sr)

                fade_samples = int(0.02 * sr)
                for sd, ov_lo, ov_hi in targets:
                    spk = sd["speaker"]
                    enh = seg_by_index.get(sd["index"])
                    if enh is None:
                        continue
                    if spk not in accepted:
                        r, d = rejected.get(spk, ("unscorable", "no verdict"))
                        self._fail(enh, ov_lo, ov_hi, r, d)
                        continue

                    track, sim = accepted[spk]
                    src = core[0] + int(ov_lo * sr) - core_src_lo
                    dst = int(ov_lo * sr) - int(enh.start * sr)
                    if src < 0 or dst < 0:
                        self._fail(enh, ov_lo, ov_hi, "short_track", "negative offset")
                        continue
                    limit = min(int(ov_hi * sr) - int(ov_lo * sr), len(track) - src,
                                len(enh.audio) - dst)
                    if limit <= 0:
                        self._fail(enh, ov_lo, ov_hi, "short_track", f"limit={limit}")
                        continue
                    if any(not (ov_hi <= a or ov_lo >= b) for a, b, _ in enh.bss_spans):
                        self._fail(enh, ov_lo, ov_hi, "already_spliced", "")
                        continue

                    # Kiểm tra ngay vùng sắp ghép trả có lời. Điểm tốt trên solo ở xa không
                    # bảo đảm overlap có lời: từng có trường hợp sim=0.67 trên mẫu trước đó
                    # 19 giây nhưng track im lặng ở chính overlap.
                    host = enh.audio[dst:dst + limit]
                    if not self._track_has_speech(host, track[src:src + limit], sr_hint=sr):
                        self._fail(enh, ov_lo, ov_hi, "empty_track",
                                   "silent where mixture has speech")
                        continue

                    # Track đã bỏ giọng nhiễu thường nhỏ hơn mixture. Khớp mức RMS trước
                    # khi ghép trả để tránh bước nhảy âm lượng; crossfade tiếp tục làm mượt biên.
                    patch = match_splice_level(
                        enh.audio[dst:dst + limit], track[src:src + limit])
                    enh.audio[dst:dst + limit] = self._cross_fade(
                        enh.audio[dst:dst + limit], patch, fade_samples)
                    enh.bss = True
                    enh.bss_spans.append((ov_lo, ov_lo + limit / sr,
                                          float(sim) if sim is not None else -1.0))
                    self.stats["spliced"] += 1
                    if self.logger:
                        self.logger.info(
                            f"[TSE:splice] seg {sd['index']} spk={spk} "
                            f"{ov_lo:.2f}-{ov_lo + limit / sr:.2f}s "
                            f"({limit / sr:.2f}s) sim="
                            + (f"{sim:.2f}" if sim is not None else "not-A")
                        )

            pbar.close()
        finally:
            # Chỉ đóng shared memory của FILE NÀY -- pool process (nếu có)
            # sống tiếp cho file kế trong batch, đóng ở close_window_pool().
            if file_windows is not None:
                file_windows.close()
        self._report_stats()
        return speech

    # ------------------------------------------------------------------
    def export_sdlm_dual_channel(self, speech_segments: List[SpeechSegment], audio_duration: float, sr: int,
                                 strict: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Dựng hai track liên tục để huấn luyện SDLM/full-duplex.
        strict=True đặt vùng tách thất bại về zero để không đưa giọng người khác
        vào track có nhãn của speaker mục tiêu."""
        total_samples = int(audio_duration * sr)
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
            start_idx = int(seg.start * sr)
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
                    i = max(start_idx, int(a * sr)) - start_idx
                    j = min(end_idx, int(b * sr)) - start_idx
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
