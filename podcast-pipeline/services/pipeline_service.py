import os
from typing import Any
from utils.checkpoint import CheckpointManager
from utils.music_map import MusicMap, build_maps
from utils.noise_map import NoiseTrack
from utils.excise import TimelineMap, excise
from utils.steps import LEGACY_FLAG, step_enabled
from services.stage_output_service import StageOutputService
from schemas.audio import AudioData

# Nếu tỷ lệ bản ghi bị đánh dấu xóa quá lớn, không tin bản đồ nhạc.
# Có thể ngưỡng sai hoặc chọn nhầm file; không để toàn bộ âm thanh biến mất.
CUT_SHARE_LIMIT = float(os.environ.get("MUSIC_CUT_SHARE_LIMIT", "0.60"))


class PipelineService:
    """Điều phối các dịch vụ xử lý âm thanh và lưu/khôi phục bằng checkpoint."""
    
    def __init__(self, 
                 audio_svc, 
                 diarization_svc, 
                 separation_svc, 
                 music_svc, 
                 asr_svc, 
                 caption_svc, 
                 refinement_svc, 
                 export_svc,
                 logger=None,
                 model_loader=None,
                 worker_services=None):
        self.audio_svc = audio_svc
        self.diarization_svc = diarization_svc
        self.separation_svc = separation_svc
        self.music_svc = music_svc
        self.asr_svc = asr_svc
        self.caption_svc = caption_svc
        self.refinement_svc = refinement_svc
        self.export_svc = export_svc
        self.logger = logger
        self.model_loader = model_loader
        # Ánh xạ tên sang dịch vụ worker để giải phóng VRAM ngay khi xong bước.
        self.worker_services = worker_services or {}
        # Timeline sau cắt nhạc. Rỗng nghĩa là chưa cắt phần nào của bản ghi.
        self.timeline = TimelineMap()
        # Bản đồ nhiễu ngoài lời nói theo khung. None nghĩa là chưa kiểm tra,
        # không đồng nghĩa với bản ghi sạch.
        self.noise_track = None
        # Khi chạy theo giai đoạn cho cả batch, gom các yêu cầu giải phóng tại đây.
        self.defer_free = None
        self.defer_workers = None
        # Dọn dẹp không nằm trong model_loader/worker_services (ví dụ pool
        # build cửa sổ song song của SeparationService) đăng ký callback vào
        # đây, chạy ở end_stage_scope() cùng lúc với defer_free/defer_workers.
        self.defer_callbacks = None

    def _load(self, group: str):
        """Chỉ tải model khi bắt đầu bước cần dùng.
        Tải tất cả ngay đầu khiến VRAM phải chứa đồng thời DiariZen, Whisper,
        PhoWhisper, Demucs và model caption. Loader có thể gọi lại an toàn nên
        chạy theo giai đoạn chỉ trả chi phí tải ở lần đầu. Bước đọc checkpoint
        không gọi hàm này và không tốn VRAM cho model không sử dụng."""
        if self.model_loader is None:
            return
        w = self.worker_services

        # Bảo đảm worker đang chạy khi bước thực sự bắt đầu. main() có thể khởi
        # động sớm để làm nóng, nhưng tại đây lỗi phải được báo; worker đang chạy
        # thì không cần khởi động thêm.
        worker = self.WORKER_FOR_STAGE.get(group)
        if worker:
            self._ensure_worker(worker)

        {
            "base":        lambda: self.model_loader.load_base_models(),
            "diarization": lambda: self.model_loader.load_diarization_models(w.get("diarizen")),
            "separation":  lambda: self.model_loader.load_separation_models(w.get("sidon")),
            "music":       lambda: self.model_loader.load_music_models(),
            "tagger":      lambda: self.model_loader.load_tagger(),
            "asr":         lambda: self.model_loader.load_asr_models(w.get("qwen3")),
            "caption":     lambda: self.model_loader.load_caption_model(),
        }[group]()

    # Worker cần cho mỗi bước. Loader đọc service.process nên phải có tiến
    # trình thật trước khi dựng client kết nối.
    WORKER_FOR_STAGE = {
        "diarization": "diarizen",
        "separation": "sidon",
        "asr": "qwen3",
    }

    # Công tắc bước lấy từ profile, dự phòng bằng cờ cũ. Dùng chung utils.steps
    # để model loader và pipeline có cùng quyết định.
    _LEGACY_FLAG = LEGACY_FLAG

    @classmethod
    def step_enabled(cls, args, name: str) -> bool:
        """Kiểm tra bước có chạy không; bước không được liệt kê vẫn chạy như trước."""
        return step_enabled(args, name)

    def _free(self, args, *model_names):
        """Giải phóng model khi xong, trừ khi bật --keep_models.
        Giữ model giảm thời gian tải lại nhưng tăng đỉnh VRAM. Khi chạy theo
        giai đoạn, chờ xử lý hết file của bước hiện tại mới giải phóng, tránh
        gỡ model ngay trước khi file tiếp theo cần nó."""
        if getattr(args, "keep_models", False) or not self.model_loader:
            return
        if self.defer_free is not None:
            self.defer_free.update(model_names)
            return
        for name in model_names:
            self.model_loader.unload(name)

    def begin_stage_scope(self):
        """Hoãn giải phóng model đến end_stage_scope() khi chạy theo giai đoạn."""
        self.defer_free = set()
        self.defer_workers = set()
        self.defer_callbacks = []

    def end_stage_scope(self):
        """Giải phóng các model/worker đã hoãn; an toàn khi không có phạm vi mở."""
        names, workers, callbacks = self.defer_free, self.defer_workers, self.defer_callbacks
        self.defer_free = None
        self.defer_workers = None
        self.defer_callbacks = None
        if names and self.model_loader:
            for name in sorted(names):
                self.model_loader.unload(name)
        for worker in sorted(workers or ()):
            service = self.worker_services.get(worker)
            if not service or getattr(service, "process", None) is None:
                continue
            if self.logger:
                self.logger.info(f"Releasing {worker} worker (stage complete, freeing VRAM)")
            try:
                service.stop()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to stop {worker} worker: {e}")
        for callback in (callbacks or ()):
            try:
                callback()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Deferred cleanup callback failed: {e}")

    def _defer_or_run(self, callback):
        """Chạy callback ngay, hoặc hoãn tới end_stage_scope() nếu đang mở
        phạm vi giai đoạn (chạy theo batch).

        Dùng cho dọn dẹp không nằm trong model_loader/worker_services -- ví dụ
        pool build cửa sổ song song của SeparationService, vốn phải sống suốt
        cả batch giống Sidon worker chứ không đóng lại sau mỗi file."""
        if self.defer_callbacks is not None:
            self.defer_callbacks.append(callback)
        else:
            callback()

    def _release_worker(self, args, name: str):
        """Dừng worker khi xong bước, tuân theo --keep_models.
        File tiếp theo cần khởi động lại nếu worker đã dừng, nếu không client
        sẽ trỏ vào tiến trình chết."""
        if getattr(args, "keep_models", False):
            return
        if self.defer_workers is not None:
            self.defer_workers.add(name)
            return
        service = self.worker_services.get(name)
        if not service or getattr(service, "process", None) is None:
            return
        if self.logger:
            self.logger.info(f"Releasing {name} worker (stage complete, freeing VRAM)")
        try:
            service.stop()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to stop {name} worker: {e}")

    def _rebind_worker(self, args, worker_name: str, service, attr: str):
        """Khởi động lại worker nếu cần và gán tiến trình sống cho client hiện có.
        Worker đang chạy thì không làm thêm."""
        if service is None or self.model_loader is None:
            return
        client = getattr(service, attr, None)
        if client is None:
            return
        process = self._ensure_worker(worker_name)
        if process is not None and getattr(client, "process", None) is not process:
            client.process = process

    def _ensure_worker(self, name: str):
        """Khởi động worker chưa chạy và trả tiến trình của nó.
        _load gọi khi bước bắt đầu; _rebind_worker dùng để cập nhật client đã có.
        Trả None nếu bước không cấu hình worker riêng."""
        service = self.worker_services.get(name)
        if service is None:
            return None
        if getattr(service, "process", None) is not None:
            return service.process
        if self.logger:
            self.logger.info(f"Starting {name} worker for this stage")
        try:
            service.spawn()
            service.wait_ready()
        except Exception as e:
            # Báo lỗi thay vì trả None: bước sắp chạy không thể dùng client thiếu
            # worker. Nuốt lỗi ở đây từng dẫn đến đầu ra rỗng ở bước phía sau.
            if self.logger:
                self.logger.error(f"Could not start the {name} worker: {e}")
            raise
        return getattr(service, "process", None)


    def _resolve_output_dir(self, args, audio_path: str) -> str:
        """Xác định duy nhất thư mục đầu ra riêng cho mỗi file.
        Ngay cả khi có --save_path, vẫn thêm tên audio để các clip có chỉ số
        bắt đầu lại từ 00000 không ghi đè kết quả của file trước trong batch."""
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        save_path = getattr(args, "save_path", None) or "./output"

        if save_path == "./output":
            suffix = "pyannote" if getattr(args, "dia3", False) else "diarizen"
            root = os.path.join(
                os.path.dirname(audio_path), "_final",
                f"-bss-{getattr(args, 'bss', False)}"
                f"-bs_roformer-{getattr(args, 'music', False)}"
                f"-vad-{getattr(args, 'vad', False)}"
                f"-diaModel-{suffix}-initPrompt-True"
                f"-merge_gap-{getattr(args, 'merge_gap', 2.0)}"
                f"-seg_th-{getattr(args, 'seg_th', 0.11)}"
                f"-cl_min-{getattr(args, 'min_cluster_size', 11)}"
                f"-cl-th-{getattr(args, 'clust_th', 0.5)}"
                f"-LLM-{getattr(args, 'LLM', 'case_0')}",
            )
        else:
            root = save_path

        return os.path.join(root, audio_name)

    def run(self, args: Any, config: dict, audio_path: str):
        # Checkpoint phải gắn với từng file; dùng chung job_id có thể khiến file
        # sau đọc diarization của file trước. Tính đường dẫn một lần để dữ liệu
        # đối chiếu separation và đầu ra cuối luôn cùng thư mục.
        output_dir = self._resolve_output_dir(args, audio_path)
        if self.logger:
            self.logger.info(f"Outputs for this file: {output_dir}")

        # Ghi kết quả ngay khi xong từng bước để --stop_after hoặc lỗi giữa chừng
        # vẫn giữ được phần đã hoàn thành.
        stage_out = StageOutputService(
            output_dir, logger=self.logger,
            enabled=not getattr(args, "no_stage_output", False))

        base_job = getattr(args, "job_id", "default_job")
        job_id = f"{base_job}_{os.path.splitext(os.path.basename(audio_path))[0]}"
        cache_dir = getattr(args, "cache_dir", "cache")
        checkpoint = CheckpointManager(cache_dir, job_id)
        # Separation mới làm thay đổi đầu vào ASR và các bước tiếp theo.
        # Tách không gian lưu để cả chuỗi dùng cùng phiên bản, không xóa bản cũ.
        window_version = getattr(self.separation_svc, "checkpoint_version", None)
        if window_version:
            checkpoint.namespaces = {
                stage: window_version for stage in (
                    "separation", "separation_report", "asr", "captioning", "refinement")}

        # Các dịch vụ được tạo một lần rồi dùng lại cho mọi file trong batch.
        for svc in (self.separation_svc, self.refinement_svc):
            reset = getattr(svc, "reset_stats", None)
            if reset:
                reset()

        # Nếu không giữ model, worker của file trước đã dừng và cần khởi động
        # lại. Chỉ khởi động worker cho bước sẽ chạy; hồi sinh tất cả từng khiến
        # LLM thiếu VRAM dù các bước trước đã giải phóng. Bước có checkpoint
        # không cần worker riêng.
        if not checkpoint.exists("diarization"):
            self._rebind_worker(args, "diarizen", self.diarization_svc, "diarizer")

        if not checkpoint.exists("asr"):
            self._rebind_worker(args, "qwen3", self.asr_svc, "qwen3")

        # Chỉ backend chạy ngoài tiến trình mới có worker cần gán lại.
        if not checkpoint.exists("separation"):
            self._rebind_worker(args, "sidon", self.separation_svc, "bss_model")
        
        # 1. Chuẩn bị âm thanh
        audio_data = self.audio_svc.load_audio(audio_path, target_sr=24000)
        
        # Chỉ ghi dữ liệu đầu vào một lần mỗi file, không ghi lại ở từng lần
        # run() khi chạy theo giai đoạn.
        computed_stages = set()

        # 2. Phân loại nội dung trước các bước xử lý khác.
        # SSLAM đánh nhãn 527 lớp AudioSet: nhạc không có lời được cắt, nhạc nền
        # được tách để giữ giọng, lời sạch giữ nguyên. Các bước sau dùng cùng
        # quyết định này. Nhóm nhiễu khác chỉ được đánh dấu để lọc corpus,
        # không tự sửa hay tái tạo giọng; xem doc/audio-cleanliness.md.
        if checkpoint.exists("music_map"):
            music_map = MusicMap.from_json(checkpoint.load("music_map"))
            self.noise_track = NoiseTrack.from_json(checkpoint.load("noise_track", fmt="json"))
        else:
            music_map = MusicMap()
            if self.step_enabled(args, "music_analysis"):
                # Chỉ tải model phân loại. Model tách nhạc sẽ tải nếu thực sự phát hiện nền.
                self._load("tagger")
                detector = self.model_loader.get("tagger") if self.model_loader else None
                music_map, self.noise_track = build_maps(
                    audio_data.waveform, audio_data.sample_rate, detector,
                    logger=self.logger)
                checkpoint.save("music_map", music_map.to_json())
                # Giữ timeline GỐC vì bước truy nguồn cần chấm trên các khoảng thời gian gốc.
                checkpoint.save("noise_track", self.noise_track.to_json(), fmt="json")
        # Tách nhạc nền trước diarization. Cache các lát được thay thế thay vì
        # cả waveform dài. Mỗi lần run() đọc lại audio gốc phải áp các lát cache
        # để mọi bước đều nhìn thấy cùng âm thanh đã bỏ nhạc.
        from utils.music_map import MUSIC
        if music_map.total_of(MUSIC) > 0 and self.step_enabled(args, "music_removal"):
            patches = checkpoint.load("music_patches")
            if patches is None:
                # Chỉ tải model tách khi bản đồ đã tìm thấy vùng cần xử lý.
                self._load("music")
                self.music_svc.bs_roformer = (self.model_loader.get("bs_roformer")
                                         if self.model_loader else None)
                patches = self.music_svc.strip_music_spans(
                    audio_data, music_map, logger=self.logger,
                    source_path=audio_path)
                checkpoint.save("music_patches", patches)
                self._free(args, "bs_roformer")
            else:
                self.music_svc.apply_music_patches(audio_data, patches)
                if self.logger:
                    self.logger.info(f"Re-applied {len(patches)} cached music "
                                     "patch(es) to the waveform")

        # Cắt phần hát/nhạc độc lập trước diarization và ASR để lời bài hát không
        # lọt vào hội thoại. Timeline bị rút ngắn nên lưu ánh xạ để đổi về thời
        # gian gốc lúc xuất.
        timeline = TimelineMap.from_json(checkpoint.load("timeline", fmt="json"))
        cuts = (music_map.excised_spans()
                if self.step_enabled(args, "cut_music") else [])

        # Không cho phép xóa gần hết bản ghi. Bộ phân loại có thể đặt ngưỡng sai
        # hoặc file không phù hợp; báo lỗi thay vì cho pipeline chạy trên âm thanh rỗng.
        if cuts:
            share = sum(b - a for a, b, _ in cuts) / max(audio_data.duration, 1e-9)
            if share > CUT_SHARE_LIMIT:
                if self.logger:
                    self.logger.warning(
                        f"Music analysis wants to cut {share * 100:.0f}% of "
                        f"{os.path.basename(audio_path)}; keeping the audio "
                        "whole. Check MUSIC_MAP_* thresholds, or whether this "
                        "file is a recording of music.")
                cuts = []

        if cuts and not timeline:
            trimmed, timeline = excise(audio_data.waveform,
                                       audio_data.sample_rate,
                                       [(a, b) for a, b, _ in cuts])
            audio_data.waveform = trimmed
            audio_data.duration = len(trimmed) / float(audio_data.sample_rate)
            checkpoint.save("timeline", timeline.to_json(), fmt="json")
            if self.logger:
                self.logger.info(
                    f"Cut {timeline.removed:.1f}s of standalone music from "
                    f"{len(cuts)} stretch(es); {audio_data.duration / 60:.1f} min remain")
        elif timeline:
            # Ở lần run() sau, cắt lại audio gốc theo timeline đã lưu, không theo
            # bản đồ vừa tính bằng ngưỡng/cờ có thể đã đổi. Nếu không, audio và
            # checkpoint diarization sẽ dùng hai hệ thời gian khác nhau.
            replay = timeline.removed_spans(audio_data.duration)
            if self.logger:
                # Hợp các khoảng trước khi so sánh; padding riêng từng loại nhạc có thể
                # làm các khoảng giao nhau và khiến tổng thô bị đếm lặp.
                from utils.excise import _merge
                wanted = sum(b - a for a, b in _merge([(a, b) for a, b, _ in cuts]))
                have = sum(b - a for a, b in replay)
                if abs(wanted - have) > 0.5:
                    self.logger.warning(
                        f"Music settings have changed since this file was "
                        f"checkpointed ({wanted:.1f}s would be cut now, "
                        f"{have:.1f}s was cut then). Replaying the checkpointed "
                        "cut so the stages already computed stay valid; delete "
                        "the music_map and timeline checkpoints to re-cut.")
            trimmed, _ = excise(audio_data.waveform, audio_data.sample_rate, replay)
            audio_data.waveform = trimmed
            audio_data.duration = len(trimmed) / float(audio_data.sample_rate)
        self.timeline = timeline

        # Đã phân loại và tách nhạc xong: giải phóng model tại đây để không giữ
        # VRAM suốt các bước DiariZen, embedding, separation và ASR phía sau.
        self._free(args, "tagger")

        # Sau khi cắt, bản đồ nhạc cũng phải chuyển sang timeline đã rút ngắn.
        self.separation_svc.music_map = (music_map.remap(timeline) if timeline
                                         else music_map)
        # Separation không được mở rộng cửa sổ xuyên các mối nối do cắt.
        self.separation_svc.timeline = timeline
        # Diarization cũng cần mối nối để không tạo segment vượt qua chúng.
        self.diarization_svc.timeline = timeline

        if "music" not in computed_stages:
            stage_out.write_music(music_map, timeline,
                                  audio_data.waveform, audio_data.sample_rate)
            computed_stages.add("music")

        if getattr(args, "stop_after", None) == "music":
            if self.logger:
                self.logger.info("Stopping after music analysis as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "music"})
            return None

        # 3. Phân đoạn người nói, VAD và chia khối.
        # Chỉ ghi artifact khi thực sự tính toán. Chạy theo giai đoạn sẽ đọc lại
        # checkpoint nhiều lần; ghi lại mỗi lần sẽ lặp JSON, clip và cảnh báo.
        computed = set()

        diarization_result = None
        if not self.step_enabled(args, "diarization"):
            if self.logger:
                self.logger.info("Step 'diarization' is off in the profile; skipping")
        elif checkpoint.exists("diarization"):
            if self.logger: self.logger.info("Loading Diarization from checkpoint")
            diarization_result = checkpoint.load("diarization")
        else:
            self._load("base")
            self._load("diarization")
            chunks, _ = self.diarization_svc.prepare_chunks(audio_data)
            diarization_result = self.diarization_svc.run_diarization(chunks, audio_data, args)
            if self.logger: self.logger.info(f"[DEBUG] Diarization returned {len(diarization_result.segments)} segments via {diarization_result.method}")

            if not diarization_result.segments:
                raise RuntimeError(
                    f"Diarization ({diarization_result.method}) produced no segments for "
                    f"{audio_path}. Check the worker log above for the underlying error; "
                    "continuing would write an empty transcript."
                )

            checkpoint.save("diarization", diarization_result)
            computed.add("diarization")

        if "diarization" in computed:
            stage_out.write_diarization(
                diarization_result.segments,
                total_dur=audio_data.duration,
                raw_segments=getattr(diarization_result, "raw_segments", None),
                audio=audio_data.waveform,
                sample_rate=audio_data.sample_rate,
            )

        self._free(args, "diarizer", "vad")
        self._release_worker(args, "diarizen")

        if getattr(args, "stop_after", None) == "diarization":
            if self.logger: self.logger.info("Stopping pipeline after diarization as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "diarization"})
            return None
            
        # 4. Tách giọng tại vùng overlap
        speech_segments = None
        if not self.step_enabled(args, "separation"):
            if self.logger:
                self.logger.info("Step 'separation' is off in the profile; skipping")
            # Dù tắt separation, các bước sau vẫn cần SpeechSegment;
            # passthrough bọc segment diarization với âm thanh gốc.
            if diarization_result is not None:
                if checkpoint.exists("separation"):
                    speech_segments = checkpoint.load("separation")
                else:
                    speech_segments = self.separation_svc.passthrough(
                        diarization_result.segments, audio_data)
                    checkpoint.save("separation", speech_segments)
        elif diarization_result is None:
            if self.logger:
                self.logger.info("Skipping separation (no diarization segments)")
        elif checkpoint.exists("separation"):
            if self.logger: self.logger.info("Loading Separation from checkpoint")
            try:
                speech_segments = checkpoint.load("separation")
            except (AttributeError, ModuleNotFoundError) as exc:
                # Checkpoint trước khi đổi tên EnhancedSegment tham chiếu lớp không còn.
                # Cần tính lại bước thay vì để lỗi pickle khó đọc tới người vận hành.
                if self.logger:
                    self.logger.warning(
                        f"Separation checkpoint predates a rename ({exc}); "
                        "recomputing this stage")
                speech_segments = None
        else:
            self._load("separation")
            self.separation_svc.dump_dir = os.path.join(
                output_dir, "03_separation", "audio", "raw")
            speech_segments = self.separation_svc.process_overlaps(diarization_result.segments, audio_data)
            if self.logger: self.logger.info(f"[DEBUG] After Separation: {len(speech_segments)} segments")
            checkpoint.save("separation", speech_segments)
            if hasattr(self.separation_svc, "report_payload"):
                checkpoint.save("separation_report",
                                self.separation_svc.report_payload())
            computed.add("separation")

        if "separation" in computed:
            stage_out.write_separation(
                speech_segments, audio_data.duration,
                report=self.separation_svc.report_payload()
                if hasattr(self.separation_svc, "report_payload") else None)
            stage_out.write_separated_audio(speech_segments, audio_data.sample_rate)

        self._free(args, "separator", "embedder")
        # Worker giữ trọng số separator trên GPU mà ASR sắp cần; dừng sau bước
        # này. Backend trong cùng tiến trình không có worker nên không cần làm gì.
        self._release_worker(args, "sidon")
        # Pool build cửa sổ song song của SeparationService (nếu có) phải sống
        # suốt cả batch giống Sidon worker ở trên -- tạo/đóng lại mỗi file
        # từng trả chi phí spawn process nhiều lần thay vì một lần cho cả
        # batch. _defer_or_run hoãn việc này tới end_stage_scope() khi đang
        # chạy theo giai đoạn (nhiều file), hoặc đóng ngay nếu chạy đơn file.
        self._defer_or_run(
            lambda: self.separation_svc.close_window_pool()
            if self.separation_svc else None)

        if getattr(args, "stop_after", None) == "separation":
            if self.logger: self.logger.info("Stopping pipeline after separation as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "separation"})
            return None
            
        # Không còn tách nhạc theo từng segment ở đây. Bước quét và bỏ nhạc đã
        # chuyển lên đầu để diarization và các bước sau cùng dùng âm thanh sạch.

        # 5. Nhận dạng lời nói bằng tổ hợp ASR (MoE)
        transcripts = None
        if not self.step_enabled(args, "asr"):
            if self.logger:
                self.logger.info("Step 'asr' is off in the profile; skipping")
        elif speech_segments is None:
            if self.logger:
                self.logger.info("Skipping ASR (no segments available)")
        elif checkpoint.exists("asr"):
            if self.logger: self.logger.info("Loading ASR from checkpoint")
            transcripts = checkpoint.load("asr")
        else:
            self._load("asr")
            if self.logger: self.logger.info(f"[DEBUG] Sending {len(speech_segments)} segments to ASR")
            transcripts = self.asr_svc.process(speech_segments, audio_data)
            if self.logger: self.logger.info(f"[DEBUG] ASR returned {len(transcripts)} transcripts")
            checkpoint.save("asr", transcripts)
            computed.add("asr")

        if "asr" in computed:
            stage_out.write_asr(transcripts)

        self._release_worker(args, "qwen3")

        if getattr(args, "stop_after", None) == "asr":
            if self.logger: self.logger.info("Stopping pipeline after asr as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "asr"})
            return transcripts
            
        # 6. Mô tả âm thanh bằng Qwen3-Omni
        if not self.step_enabled(args, "captioning"):
            if self.logger:
                self.logger.info("Step 'captioning' is off in the profile; skipping")
        elif transcripts is None or speech_segments is None:
            if self.logger:
                self.logger.info("Skipping captioning (no transcripts or segments available)")
        elif checkpoint.exists("captioning"):
            if self.logger: self.logger.info("Loading Captioning from checkpoint")
            transcripts = checkpoint.load("captioning")
        else:
            self._load("caption")
            segment_audio = {s.index: s.audio for s in speech_segments if s.audio is not None}
            transcripts = self.caption_svc.add_captions(transcripts, audio_data, segment_audio)
            checkpoint.save("captioning", transcripts)
        
        if getattr(args, "stop_after", None) == "captioning":
            if self.logger: self.logger.info("Stopping pipeline after captioning as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "captioning"})
            return transcripts
                
        # 7. Hiệu chỉnh bằng LLM
        if not self.step_enabled(args, "refinement"):
            if self.logger:
                self.logger.info("Step 'refinement' is off in the profile; skipping")
        elif transcripts is None:
            if self.logger:
                self.logger.info("Skipping refinement (no transcripts available)")
        elif checkpoint.exists("refinement"):
            if self.logger: self.logger.info("Loading Refinement from checkpoint")
            transcripts = checkpoint.load("refinement")
        else:
            import copy
            before = copy.deepcopy(transcripts)
            transcripts = self.refinement_svc.refine(
                transcripts, getattr(args, "llm_prompt", None) or None
            )
            checkpoint.save("refinement", transcripts)
            computed.add("refinement")
            stage_out.write_refinement(transcripts, before=before)

        if self.step_enabled(args, "refinement"):
            if not getattr(args, "keep_models", False):
                self.refinement_svc.unload()

        
        # 8. Xuất kết quả
        save_path = output_dir
        os.makedirs(save_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        if not self.step_enabled(args, "export"):
            if self.logger:
                self.logger.info("Step 'export' is off in the profile; skipping")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "audio_name": base_name,
                                      "music_enabled": getattr(args, "music", False)})
            if self.logger: self.logger.info(f"Pipeline completed successfully. Results saved to {save_path}")
            return transcripts

        if transcripts is None:
            if self.logger:
                self.logger.info("Skipping full export (no transcripts available)")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "audio_name": base_name,
                                      "music_enabled": getattr(args, "music", False)})
            if self.logger: self.logger.info(f"Pipeline completed successfully. Results saved to {save_path}")
            return transcripts

        # Các bước trên dùng timeline đã cắt. Ghi ánh xạ về bản ghi nguồn trước
        # khi xuất để truy lại timestamp và phân biệt khoảng nghỉ thật với mối nối.
        from utils.provenance import annotate as annotate_provenance, summary as provenance_summary
        annotate_provenance(transcripts, self.timeline,
                            noise=getattr(self, "noise_track", None))
        provenance = provenance_summary(transcripts)
        if self.logger and provenance.get("gaps_broken_by_a_cut"):
            self.logger.info(
                f"{provenance['gaps_broken_by_a_cut']} turn gap(s) span a cut "
                f"and carry no duration; {provenance['segments_crossing_a_cut']} "
                "segment(s) are glued from two stretches")

        if hasattr(self.separation_svc, "write_report"):
            self.separation_svc.write_report(
                save_path, base_name,
                payload=checkpoint.load("separation_report"))
        
        metadata = {
            "audio_file": os.path.basename(audio_path),
            "audio_name": base_name,
            "diarization_model": "pyannote" if getattr(args, "dia3", False) else "diarizen",
            "asr_models": self.asr_svc.active_models(),
            "music_enabled": getattr(args, "music", False),
            "vad_enabled": getattr(args, "vad", False),
            "bss_enabled": getattr(args, "bss", False),
            "llm_refinement": getattr(args, "llm_refinement", False),
            "qwen3omni_caption": getattr(args, "qwen3omni", False),
            # Timestamp của segments thuộc timeline đã cắt; orig_spans ánh xạ về
            # các khoảng tương ứng trong file gốc.
            "timeline": self.timeline.to_json() if self.timeline else None,
            "provenance": provenance,
        }
        
        self.export_svc.export_json(transcripts, os.path.join(save_path, f"{base_name}.json"), metadata=metadata)
        self.export_svc.export_srt(transcripts, os.path.join(save_path, f"{base_name}.srt"))
        self.export_svc.export_mp3_segments(transcripts, audio_data, save_path, base_name)
        
        if speech_segments is not None:
            self.export_svc.export_separated_audio(speech_segments, audio_data.sample_rate, save_path)
        
        import json
        try:
            if diarization_result is not None:
                with open(os.path.join(save_path, f"{base_name}_intermediate_diarization.json"), "w", encoding="utf-8") as f:
                    json.dump([seg.__dict__ for seg in diarization_result.segments], f, ensure_ascii=False, indent=2)
            
            if speech_segments is not None:
                with open(os.path.join(save_path, f"{base_name}_intermediate_separation.json"), "w", encoding="utf-8") as f:
                    sep_data = []
                    for s in speech_segments:
                        s_dict = s.__dict__.copy()
                        s_dict.pop('audio', None)
                        sep_data.append(s_dict)
                    json.dump(sep_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.logger: self.logger.error(f"Failed to export intermediate results: {e}")
        
        review_page = None
        if getattr(args, "review_page", True):
            try:
                from tools.make_review_page import build_review_page
                review_page = build_review_page(
                    save_path,
                    max_mb=getattr(args, "review_max_mb", None),
                    logger=self.logger)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not build the review page: {e}")

        final = {"json": f"{base_name}.json", "srt": f"{base_name}.srt"}
        if review_page:
            final["review"] = os.path.basename(review_page)
        stage_out.write_manifest(metadata, extra={"final": final})

        if self.logger: self.logger.info(f"Pipeline completed successfully. Results saved to {save_path}")
        return transcripts
