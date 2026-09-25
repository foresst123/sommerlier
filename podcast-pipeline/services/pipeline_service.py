import os
import re
import copy
import contextlib
import threading
from typing import Any
from utils.checkpoint import CheckpointManager
from utils.music_map import MusicMap, build_maps
from utils.noise_map import NoiseTrack
from utils.excise import TimelineMap, excise
from utils.steps import LEGACY_FLAG, opt_in_step_enabled, step_enabled
from services.speaker_relabel_service import apply_relabels
from services.word_alignment_service import apply_word_alignments
from services.stage_output_service import StageOutputService
from schemas.audio import AudioData
from utils.sdpa_guard import capture_baseline, restore_if_changed

# Taken at import, before any stage can run a third-party sdpa_kernel().
_SDPA_BASELINE = capture_baseline()

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
                 worker_services=None,
                 performance_monitor=None,
                 relabel_svc=None,
                 conversation_export_svc=None,
                 word_alignment_svc=None,
                 clean_dataset_svc=None):
        self.audio_svc = audio_svc
        self.diarization_svc = diarization_svc
        self.separation_svc = separation_svc
        self.music_svc = music_svc
        self.asr_svc = asr_svc
        self.caption_svc = caption_svc
        self.refinement_svc = refinement_svc
        self.export_svc = export_svc
        # Optional post-refinement passes. Relabel and conversation-export judging reuse the
        # resident LLM; word alignment uses a separate Wav2Vec2 model. All are
        # opt-in: passing a service does not switch its step on.
        self.relabel_svc = relabel_svc
        self.conversation_export_svc = conversation_export_svc
        self.word_alignment_svc = word_alignment_svc
        self.clean_dataset_svc = clean_dataset_svc
        self.logger = logger
        self.model_loader = model_loader
        self.performance_monitor = performance_monitor
        # One file at a time through SSLAM. It loads its model on first call
        # and keeps a set it mutates while running, so two files entering it
        # together can build it twice. Serialising also gives the music stage
        # its shape: file N+1 is classified on device_1 while file N is
        # separated on device_2, instead of both queuing for the same card.
        # Created once here, so every parallel_stage_view copy holds the
        # same lock object.
        self._tagger_lock = threading.Lock()
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
        self._model_load_lock = threading.RLock()
        # Futures from a deferred diarize_postprocess() (see run()'s
        # "diarization" stop-point). Created once here, not per file, so
        # every parallel_stage_view("diarization") copy (a shallow
        # copy.copy()) shares the same dict -- utils.batch.run_batch_by_stage
        # drains it after the diarization stage's file loop, before
        # "separation" can start reading checkpoints that might not be
        # written yet.
        self._pending_diar_jobs = {}
        self._pending_diar_lock = threading.RLock()

    def parallel_stage_view(self, stage: str):
        """Return an isolated per-file view for a parallel pipeline stage.

        ModelLoader and worker pools remain shared. Timeline-bearing services
        are copied so two files cannot overwrite each other's excision seams
        while their DiariZen requests are in flight.
        """
        if stage not in ("music", "diarization", "separation"):
            return self
        view = copy.copy(self)
        view.diarization_svc = copy.copy(self.diarization_svc)
        if stage == "separation" and hasattr(self.separation_svc, "fork_for_file"):
            view.separation_svc = self.separation_svc.fork_for_file()
        else:
            view.separation_svc = copy.copy(self.separation_svc)
        # music_svc holds no per-file state beyond the cached bs_roformer
        # reference -- copied so `self.music_svc.bs_roformer = ...` in run()
        # assigns each file's own slot instead of racing on one. What must NOT
        # be per-file is BS-RoFormer's checkout queue: it lives in a dict that
        # music_svc creates once, and a shallow copy shares that dict, so every
        # file's view checks separators out of the same queue.
        view.music_svc = copy.copy(self.music_svc)
        view.refinement_svc = copy.copy(self.refinement_svc)
        view.timeline = TimelineMap()
        # noise_track is written mid-run() (music-analysis step, before
        # diarization even for a "diarization"-stage pass reading it back from
        # checkpoint), so a shared PipelineService would let two concurrent
        # files stomp on each other's here regardless of which stage asked
        # for a view.
        view.noise_track = None
        return view

    def _load(self, group: str):
        """Chỉ tải model khi bắt đầu bước cần dùng.
        Tải tất cả ngay đầu khiến VRAM phải chứa đồng thời DiariZen, Whisper,
        PhoWhisper, Demucs và model caption. Loader có thể gọi lại an toàn nên
        chạy theo giai đoạn chỉ trả chi phí tải ở lần đầu. Bước đọc checkpoint
        không gọi hàm này và không tốn VRAM cho model không sử dụng."""
        if self.model_loader is None:
            return
        # Some focused tests and integrations construct this service through
        # __new__. Initialise lazily as well as in __init__ for compatibility.
        if not hasattr(self, "_model_load_lock"):
            self._model_load_lock = threading.RLock()
        with self._model_load_lock:
            w = self.worker_services

            # Bảo đảm worker đang chạy khi bước thực sự bắt đầu. main() có thể khởi
            # động sớm để làm nóng, nhưng tại đây lỗi phải được báo; worker đang chạy
            # thì không cần khởi động thêm.
            for worker in self.WORKERS_FOR_STAGE.get(group, ()):
                self._ensure_worker(worker)

            {
                "base":        lambda: self.model_loader.load_base_models(),
                "diarization": lambda: self.model_loader.load_diarization_models(w.get("diarizen")),
                "separation":  lambda: self.model_loader.load_separation_models(w.get("sidon")),
                "music":       lambda: self.model_loader.load_music_models(),
                "tagger":      lambda: self.model_loader.load_tagger(),
                "asr":         lambda: self.model_loader.load_asr_models(
                    w.get("qwen3"), w.get("whisper")),
                "caption":     lambda: self.model_loader.load_caption_model(),
            }[group]()

    # Worker cần cho mỗi bước. Loader đọc service.process nên phải có tiến
    # trình thật trước khi dựng client kết nối.
    WORKERS_FOR_STAGE = {
        "diarization": ("diarizen",),
        "separation": ("sidon",),
        "asr": ("qwen3", "whisper"),
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

    def prepare_stage_scope(self, stage):
        """Prepare lifecycle rules for one corpus-wide stage pass.

        ASRService used to unload Whisper and PhoWhisper inside every file.
        During a stage-major pass that turns one model load per corpus back into
        one load per file. Temporarily retaining them lets the loader's existing
        idempotence guard reuse both models; end_stage_scope releases them once.
        """
        self._stage_scope_name = stage
        self._stage_asr_keep_models = None
        asr = getattr(self, "asr_svc", None)
        if stage == "asr" and asr is not None:
            self._stage_asr_keep_models = bool(getattr(asr, "keep_models", False))
            asr.keep_models = True

    def end_stage_scope(self):
        """Giải phóng các model/worker đã hoãn; an toàn khi không có phạm vi mở."""
        stage = getattr(self, "_stage_scope_name", None)
        asr_keep_models = getattr(self, "_stage_asr_keep_models", None)
        self._stage_scope_name = None
        self._stage_asr_keep_models = None
        names = getattr(self, "defer_free", None)
        workers = getattr(self, "defer_workers", None)
        callbacks = getattr(self, "defer_callbacks", None)
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
        if stage == "asr" and asr_keep_models is not None:
            self.asr_svc.keep_models = asr_keep_models
            if not asr_keep_models and self.model_loader:
                # These two are in-process models. Qwen3 is the worker released
                # through defer_workers above.
                self.model_loader.unload("whisper")
                self.model_loader.unload("phowhisper")
        self._reclaim_vram()

    def _reclaim_vram(self):
        """Trả VRAM về driver ở ranh giới stage, rồi ghi lại còn trống bao nhiêu.

        Dòng "Unloaded X from VRAM" chỉ nói tham chiếu đã bị bỏ. Bộ nhớ vẫn
        nằm trong cache của allocator cho tới khi được thu hồi, và đo được là
        nó có thể nán lại khoảng một phút -- đủ để stage sau xin 82% VRAM mỗi
        card và trượt. Đây là con số thật để đối chiếu, không phải giả định.
        """
        free = {}
        restore_if_changed(_SDPA_BASELINE, self.logger, "the next stage")
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                # empty_cache() dọn allocator của MỌI card, không riêng card
                # hiện tại, nên một lần gọi là đủ cho cả hai.
                torch.cuda.empty_cache()
                for index in range(torch.cuda.device_count()):
                    with torch.cuda.device(index):
                        free_bytes, total = torch.cuda.mem_get_info()
                    free[str(index)] = round(free_bytes / (1024 ** 3), 2)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not reclaim VRAM at the stage boundary: {e}")
            return
        if free and self.logger:
            self.logger.info(
                "Stage boundary: VRAM free "
                + ", ".join(f"GPU{i} {g:.2f} GiB" for i, g in sorted(free.items())))
        monitor = getattr(self, "performance_monitor", None)
        if monitor and free:
            monitor.record("stage_vram_reclaimed", free_gib=free)

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
        monitor = getattr(self, "performance_monitor", None)
        if monitor:
            monitor.record("worker_stopping", worker=name)
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
        worker_service = self.worker_services.get(worker_name)
        endpoint = (worker_service
                    if worker_service is not None
                    and hasattr(worker_service, "processes")
                    else process)
        if endpoint is not None and getattr(client, "process", None) is not endpoint:
            client.process = endpoint

    def _register_worker_pids(self, name: str, service):
        """Báo PID của worker cho monitor để quy VRAM theo tên worker.
        Pool (nhiều tiến trình Sidon) có `processes`; worker đơn có `process`."""
        processes = (getattr(service, "processes", None)
                     or [getattr(service, "process", None)])
        pids = [p.pid for p in processes if getattr(p, "pid", None)]
        monitor = getattr(self, "performance_monitor", None)
        register = getattr(monitor, "register_process", None)
        if register:
            for index, pid in enumerate(pids):
                register(pid, name if len(pids) == 1 else f"{name}[{index}]")
        return pids

    # Thứ tự các điểm dừng của run(); None (chạy hết) tới được mọi bước.
    _STOP_ORDER = ("music", "diarization", "separation", "music_removal",
                   "asr", "captioning", "refinement", "speaker_relabel",
                   "word_alignment", "conversation_exports")

    @classmethod
    def _reaches(cls, args, stage: str) -> bool:
        """Lượt chạy này có đi tới `stage` không, xét theo --stop_after."""
        stop = getattr(args, "stop_after", None)
        if stop is None or stop not in cls._STOP_ORDER:
            return True
        return cls._STOP_ORDER.index(stop) >= cls._STOP_ORDER.index(stage)

    def _ensure_worker(self, name: str):
        """Khởi động worker chưa chạy và trả tiến trình của nó.
        _load gọi khi bước bắt đầu; _rebind_worker dùng để cập nhật client đã có.
        Trả None nếu bước không cấu hình worker riêng."""
        service = self.worker_services.get(name)
        if service is None:
            return None
        if getattr(service, "process", None) is not None:
            self._register_worker_pids(name, service)
            return service.process
        if self.logger:
            self.logger.info(f"Starting {name} worker for this stage")
        monitor = getattr(self, "performance_monitor", None)
        if monitor:
            monitor.record("worker_loading", worker=name)
        try:
            service.spawn()
            service.wait_ready()
            pids = self._register_worker_pids(name, service)
            if monitor:
                monitor.record("worker_ready", worker=name, pids=pids)
        except Exception as e:
            # Báo lỗi thay vì trả None: bước sắp chạy không thể dùng client thiếu
            # worker. Nuốt lỗi ở đây từng dẫn đến đầu ra rỗng ở bước phía sau.
            if self.logger:
                self.logger.error(f"Could not start the {name} worker: {e}")
            raise
        return getattr(service, "process", None)

    def _export_conversation_exports(self, stage_out, transcripts, audio_data, music_map,
                                     output_dir, audio_path, speech_segments=None):
        """Export selected two-person conversation excerpts from processed audio.

        Runs after relabel, so a stray third label does not end a conversation,
        and while the LLM is still resident. The waveform is the processed one --
        music already stripped, cuts already made -- so the segments' timestamps
        line up with it. The noise track and music map are the SSLAM outputs of
        this same run; when there is no noise track the pass says so and cuts
        nothing rather than reading "not measured" as "clean".

        Not checkpointed: judging a couple of dozen candidates is quick next to
        the stages before it, and the files are overwritten in place.
        """
        # The music map is in the original timeline; the excerpts are cut in the
        # shortened one. With nothing cut the two are the same map.
        music_cut = (music_map.remap(self.timeline)
                     if music_map is not None and self.timeline else music_map)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        result = self.conversation_export_svc.run(
            transcripts, timeline=self.timeline,
            noise=getattr(self, "noise_track", None), music_map=music_cut,
            waveform=audio_data.waveform, sample_rate=audio_data.sample_rate,
            out_dir=os.path.join(output_dir, "conversation_exports"), base_name=base_name,
            # The two-channel file is laid out from the separated speaker tracks.
            speech_segments=speech_segments,
            separation_service=getattr(self, "separation_svc", None))
        stage_out.write_conversation_exports(result.report, result.exports)
        return result

    def _export_clean_two_channel_dataset(self, args, config, transcripts,
                                          speech_segments, audio_data,
                                          music_map, audio_path):
        """Write the optional external clean corpus, or perform no filesystem IO.

        The output root is intentionally not an argparse default. An empty or
        missing config value means disabled, so a normal run keeps exactly the
        same directory layout it had before this feature.
        """
        profile = (config.get("environments", {})
                   .get(getattr(args, "env", ""), {}))
        root = ((profile.get("outputs") or {})
                .get("clean_two_channel_dir", ""))
        if self.clean_dataset_svc is None:
            return None
        root = self.clean_dataset_svc.resolve_root(root)
        if not root:
            return None
        if not getattr(args, "bss", False):
            message = "BSS is off; clean two-channel export was skipped"
            if self.logger:
                self.logger.warning(f"[clean-2ch] {message}")
            return {"enabled": True, "skipped": "bss_disabled", "error": message}
        if transcripts is None or speech_segments is None:
            message = "final transcript or separation segments are unavailable"
            if self.logger:
                self.logger.warning(f"[clean-2ch] {message}; export skipped")
            return {"enabled": True, "skipped": "missing_input", "error": message}

        music_cut = (music_map.remap(self.timeline)
                     if music_map is not None and self.timeline else music_map)
        try:
            return self.clean_dataset_svc.export(
                root=root, source_path=audio_path, transcripts=transcripts,
                speech_segments=speech_segments,
                separation_service=self.separation_svc,
                timeline=self.timeline,
                noise=getattr(self, "noise_track", None), music_map=music_cut,
                sample_rate=audio_data.sample_rate,
                audio_duration=audio_data.duration,
                selection_settings=(profile.get("models", {})
                                    .get("conversation_selection", {})),
            )
        except Exception as exc:
            if self.logger:
                self.logger.exception(
                    f"[clean-2ch] optional dataset export failed: {exc}")
            return {"enabled": True, "skipped": "export_error", "error": str(exc)}

    # -- music stage: taking the music out, and measuring what is left --------------
    @staticmethod
    def _music_scope(args) -> str:
        """`spans`: strip only the stretches the tagger called a bed (the old way).
        `full`: run the separator over the whole recording."""
        scope = str(getattr(args, "music_scope", None) or "spans").strip().lower()
        if scope not in ("full", "spans"):
            raise ValueError(
                f"pipeline.music_scope must be 'full' or 'spans', got {scope!r}")
        return scope

    @staticmethod
    def _music_checkpoint_name(args, config) -> str:
        """The separator checkpoint in use, so what it produced is filed under it."""
        name = getattr(args, "music_separator", None)
        if not name:
            profile = ((config or {}).get("environments") or {}).get(
                getattr(args, "env", ""), {})
            name = ((profile.get("models") or {}).get("bs_roformer") or {}).get("model")
        return str(name or "default")

    def _strip_music(self, args, config, checkpoint, audio_data, audio_path,
                     music_map) -> bool:
        """Take the music out of the waveform. True when the waveform was changed.

        What was cached is filed under the scope and the checkpoint that made it:
        a recording separated by one model, or over other stretches, must not be
        reused when the profile has moved to another -- run() reloads the
        original audio on every entry and re-applies whatever is cached.
        """
        from utils.music_map import MUSIC
        if not self.step_enabled(args, "music_removal"):
            return False
        scope = self._music_scope(args)
        if scope == "spans" and music_map.total_of(MUSIC) <= 0:
            return False

        stem = os.path.splitext(self._music_checkpoint_name(args, config))[0]
        namespace = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{scope}-{stem}")
        checkpoint.namespaces["music_patches"] = namespace
        checkpoint.namespaces["noise_track_processed"] = namespace

        patches = checkpoint.load("music_patches")
        if patches is None:
            # Only load the separator once there is something for it to do.
            self._load("music")
            self.music_svc.bs_roformer = (self.model_loader.get("bs_roformer")
                                          if self.model_loader else None)
            if scope == "full":
                patches = self.music_svc.strip_full_recording(
                    audio_data, logger=self.logger, source_path=audio_path)
            else:
                patches = self.music_svc.strip_music_spans(
                    audio_data, music_map, logger=self.logger,
                    source_path=audio_path)
            checkpoint.save("music_patches", patches)
            self._free(args, "bs_roformer")
        else:
            self.music_svc.apply_music_patches(audio_data, patches)
            if self.logger:
                self.logger.info(f"Re-applied {len(patches)} cached music "
                                 f"patch(es) ({scope}) to the waveform")
        return bool(patches)

    def _measure_processed_noise(self, args, checkpoint, audio_data) -> None:
        """Measure the noise of the waveform as it is now.

        The first sweep listens to the recording as it was made, and that is what
        decides what music there is to strip or cut. Once the music is out, the
        audio the dataset is cut from is a different signal, and the noise a
        conversation excerpt is judged by should be that signal's. So the noise
        track is replaced by a second sweep over the stripped waveform; the
        first is kept in the checkpoints as `noise_track` for comparison.
        """
        if not self.step_enabled(args, "music_analysis"):
            return
        if checkpoint.exists("noise_track_processed", fmt="json"):
            self.noise_track = NoiseTrack.from_json(
                checkpoint.load("noise_track_processed", fmt="json"))
            return
        self._load("tagger")
        detector = self.model_loader.get("tagger") if self.model_loader else None
        tagger_lock = getattr(self, "_tagger_lock", None)
        with (tagger_lock if tagger_lock is not None else contextlib.nullcontext()):
            _, noise = build_maps(audio_data.waveform, audio_data.sample_rate,
                                  detector, logger=self.logger)
        if not noise:
            if self.logger:
                self.logger.warning("Noise sweep over the stripped audio gave nothing; "
                                    "keeping the measurement made before stripping")
            return
        before = self.noise_track
        self.noise_track = noise
        checkpoint.save("noise_track_processed", noise.to_json(), fmt="json")
        if self.logger and before:
            import numpy as np
            self.logger.info(
                "Noise p90 of the recording: "
                f"{float(np.percentile(before.combined, 90)):.3f} as recorded, "
                f"{float(np.percentile(noise.combined, 90)):.3f} after music removal")

    def _relabel_speakers(self, checkpoint, stage_out, transcripts, speech_segments):
        """Reassign speaker labels from the whole transcript. Only `speaker` changes.

        What is checkpointed is the decision (segment index -> speaker), not the
        relabelled transcripts: run() is re-entered once per stage and reloads
        the refinement checkpoint, which predates this pass. The decision is
        applied again to whatever was loaded, transcripts and separation
        segments alike, so both stay in step on every entry.
        """
        svc = self.relabel_svc
        checkpoint.namespaces["speaker_relabel"] = svc.checkpoint_namespace
        if checkpoint.exists("speaker_relabel"):
            payload = checkpoint.load("speaker_relabel") or {}
            changed = apply_relabels(
                transcripts, payload.get("mapping", {}), speech_segments)
            if self.logger:
                self.logger.info(
                    f"Loading speaker relabel from checkpoint ({changed} segment(s))")
            return transcripts

        # The model reads the gap before each turn; it is written by the same
        # pass that fills the export, and is safe to write twice.
        from utils.provenance import annotate as annotate_provenance
        annotate_provenance(transcripts, self.timeline,
                            noise=getattr(self, "noise_track", None))
        result = svc.relabel(transcripts)
        apply_relabels(transcripts, result.mapping, speech_segments)
        report = result.to_report()
        # A pass that never got an answer must be tried again next time, not
        # remembered as done.
        if result.skipped is None and result.failed_windows < result.windows:
            checkpoint.save("speaker_relabel",
                            {"mapping": result.mapping, "report": report})
        stage_out.write_relabel(report)
        return transcripts

    def _align_words(self, checkpoint, stage_out, transcripts, audio_data,
                     speech_segments):
        """Attach final-text word times, reapplying a cached result on resume."""
        svc = self.word_alignment_svc
        checkpoint.namespaces["word_alignment"] = svc.checkpoint_namespace_for(transcripts)
        if checkpoint.exists("word_alignment"):
            payload = checkpoint.load("word_alignment") or {}
            applied = apply_word_alignments(
                transcripts, payload.get("words_by_index", {}))
            if self.logger:
                self.logger.info(
                    f"Loading word alignment from checkpoint ({applied} segment(s))")
            return transcripts

        result = svc.align(transcripts, audio_data, speech_segments)
        checkpoint.save("word_alignment", {
            "words_by_index": result.words_by_index,
            "report": result.report,
        })
        stage_out.write_word_alignment(transcripts, result.report)
        return transcripts


    def _resolve_output_dir(self, args, audio_path: str) -> str:
        """Xác định duy nhất thư mục đầu ra riêng cho mỗi file.
        Ngay cả khi có --save_path, vẫn thêm tên audio để các clip có chỉ số
        bắt đầu lại từ 00000 không ghi đè kết quả của file trước trong batch."""
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        save_path = getattr(args, "save_path", None) or "./output"

        if save_path == "./output":
            suffix = "pyannote" if getattr(args, "dia3", False) else "diarizen"
            # The performance path changes what is computed -- ASR batching,
            # refinement placement and token ceiling, diarization component
            # placement -- so its results must not land in, or be resumed from,
            # the directory the baseline owns. "off" reproduces the old path
            # exactly, which is what keeps existing caches readable.
            from utils import performance_config
            perf = getattr(args, "performance_config", None)
            perf_token = (performance_config.fingerprint(perf)
                          if isinstance(perf, dict) else "off")
            root = os.path.join(
                os.path.dirname(audio_path), "_final",
                (f"-perf-{perf_token}" if perf_token != "off" else "")
                + f"-bss-{getattr(args, 'bss', False)}"
                f"-bs_roformer-{getattr(args, 'music', False)}"
                f"-vad-{getattr(args, 'vad', False)}"
                f"-diaModel-{suffix}-initPrompt-True"
                f"-merge_gap-{getattr(args, 'merge_gap', 0.5)}"
                f"-bridge_gap-{getattr(args, 'bridge_gap', 3.0)}"
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
        # Cũng chỉ khi lượt này thực sự chạy tới bước đó: chạy theo giai đoạn,
        # client của lượt trước vẫn còn, nên pass "music" từng hồi sinh worker
        # ASR và để nó giữ ~6 GB VRAM suốt music/diarization/separation.
        if not checkpoint.exists("diarization") and self._reaches(args, "diarization"):
            self._rebind_worker(args, "diarizen", self.diarization_svc, "diarizer")

        if not checkpoint.exists("asr") and self._reaches(args, "asr"):
            self._rebind_worker(args, "qwen3", self.asr_svc, "qwen3")

        # Chỉ backend chạy ngoài tiến trình mới có worker cần gán lại.
        if not checkpoint.exists("separation") and self._reaches(args, "separation"):
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
                # _load above must stay outside this lock: the loader has its own
                # lock, and taking them in opposite orders from two files would
                # deadlock.
                tagger_lock = getattr(self, "_tagger_lock", None)
                with (tagger_lock if tagger_lock is not None
                      else contextlib.nullcontext()):
                    music_map, self.noise_track = build_maps(
                        audio_data.waveform, audio_data.sample_rate, detector,
                        logger=self.logger)
                checkpoint.save("music_map", music_map.to_json())
                # Giữ timeline GỐC vì bước truy nguồn cần chấm trên các khoảng thời gian gốc.
                checkpoint.save("noise_track", self.noise_track.to_json(), fmt="json")
        # Tách nhạc nền trước diarization. Cache các lát được thay thế thay vì
        # cả waveform dài. Mỗi lần run() đọc lại audio gốc phải áp các lát cache
        # để mọi bước đều nhìn thấy cùng âm thanh đã bỏ nhạc.
        # pipeline.music_scope: "spans" chỉ tách các khoảng có nhạc nền dưới lời;
        # "full" chạy model tách trên toàn bộ file. Sau khi audio đã đổi, độ nhiễu
        # được đo lại trên chính audio đó (xem _measure_processed_noise).
        if self._strip_music(args, config, checkpoint, audio_data, audio_path, music_map):
            self._measure_processed_noise(args, checkpoint, audio_data)

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
        # 3. Diarization
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
            raw = self.diarization_svc.diarize_raw(chunks, audio_data, args)
            self._free(args, "diarizer", "vad")
            self._release_worker(args, "diarizen")

            if getattr(args, "stop_after", None) == "diarization":
                # This is the one pass that sets stop_after == "diarization"
                # -- a later pass re-enters this section of run() only to
                # load the checkpoint on its way to another stage -- so this
                # fires exactly once per file. Deferring diarize_postprocess()
                # (pure CPU: filter/VAD/merge/split/write, see
                # services/diarization_service.py) to the background means
                # the DiariZen worker diarize_raw() just released is free for
                # the NEXT file immediately, instead of waiting for this
                # file's CPU-only tail. Window building is pure CPU too and
                # does not need Sidon loaded, so prefetch_overlap_plan() can
                # also run now, in the background, before the separation
                # stage has even started the Sidon worker -- see
                # services/separation_service.py:prefetch_overlap_plan.
                # utils/batch.py's run_batch_by_stage drains
                # self._pending_diar_jobs before "separation" starts, so no
                # later stage can read a checkpoint this hasn't written yet.
                future = self.diarization_svc.submit_postprocess(raw, audio_data, args)
                with self._pending_diar_lock:
                    self._pending_diar_jobs[audio_path] = future

                def _finish(fut, audio_path=audio_path, checkpoint=checkpoint,
                            stage_out=stage_out, audio_data=audio_data, args=args):
                    try:
                        result = fut.result()
                    except Exception:
                        # Surfaced to run_batch_by_stage's `failures` by the
                        # drain step, which calls future.result() again on
                        # this same future -- deliberately not handled here.
                        return
                    checkpoint.save("diarization", result)
                    stage_out.write_diarization(
                        result.segments,
                        total_dur=audio_data.duration,
                        raw_segments=getattr(result, "raw_segments", None),
                        audio=audio_data.waveform,
                        sample_rate=audio_data.sample_rate,
                    )
                    if self.step_enabled(args, "separation"):
                        self.separation_svc.prefetch_overlap_plan(
                            result.segments, audio_data, audio_path)

                future.add_done_callback(_finish)
                if self.logger:
                    self.logger.info(
                        "Stopping pipeline after diarization as requested by "
                        "--stop_after (post-processing continues in the background).")
                stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                          "stopped_after": "diarization"})
                return None

            diarization_result = self.diarization_svc.diarize_postprocess(raw, audio_data, args)
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
            speech_segments = self.separation_svc.process_overlaps(
                diarization_result.segments, audio_data, audio_path=audio_path)
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
        close_file_model = getattr(self.separation_svc, "close_file_model", None)
        if callable(close_file_model):
            close_file_model()
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
        self._defer_or_run(
            lambda: self.separation_svc.close_prefetch_pool()
            if self.separation_svc
            and hasattr(self.separation_svc, "close_prefetch_pool") else None)
        self._defer_or_run(
            lambda: self.separation_svc.close_async_pools()
            if self.separation_svc
            and hasattr(self.separation_svc, "close_async_pools") else None)

        if getattr(args, "stop_after", None) == "separation":
            if self.logger: self.logger.info("Stopping pipeline after separation as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "separation"})
            return None

        # Pass "music_removal" của chạy theo giai đoạn: bước tách nhạc theo
        # segment từng nằm ở đây nay đã chuyển lên đầu (bước 2), nên lượt này
        # không còn việc gì. Thiếu điểm dừng, nó chạy thẳng ASR -> LLM -> export
        # cho từng file, khiến LLM của file trước và ASR của file sau cùng nằm
        # trên GPU -- chính là lỗi OOM -- và pass "asr"/"refinement" chỉ còn
        # đọc checkpoint.
        if getattr(args, "stop_after", None) == "music_removal":
            if self.logger: self.logger.info("Stopping pipeline after music_removal as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "music_removal"})
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
        self._release_worker(args, "whisper")

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

        if getattr(args, "stop_after", None) == "refinement":
            if not getattr(args, "keep_models", False):
                self._defer_or_run(self.refinement_svc.unload)
            if self.logger:
                self.logger.info(
                    "Stopping pipeline after refinement as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "refinement"})
            return transcripts

        # 7b. Speaker relabel: the same resident LLM reads the whole transcript
        # and may reassign a segment's speaker. Nothing else about a segment is
        # touched. Opt-in: a profile that does not list it does not run it.
        relabel_on = (self.relabel_svc is not None
                      and opt_in_step_enabled(args, "speaker_relabel"))
        if relabel_on and transcripts is not None:
            transcripts = self._relabel_speakers(
                checkpoint, stage_out, transcripts, speech_segments)

        if getattr(args, "stop_after", None) == "speaker_relabel":
            if not getattr(args, "keep_models", False):
                self._defer_or_run(self.refinement_svc.unload)
            if self.logger:
                self.logger.info(
                    "Stopping pipeline after speaker_relabel as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "speaker_relabel"})
            return transcripts

        # 7c. Forced alignment: refinement may have changed Whisper's words, so
        # its old word timestamps are no longer valid. Align the final text with
        # Vietnamese Wav2Vec2 before any clean-data selection reads it.
        alignment_on = (self.word_alignment_svc is not None
                        and opt_in_step_enabled(args, "word_alignment"))
        if alignment_on and transcripts is not None:
            # The refinement model can span both GPUs. Free it before loading
            # Wav2Vec2, then the conversation-export pass below reloads it only if it is needed.
            # A checkpointed alignment loads no model and does not need this.
            checkpoint.namespaces["word_alignment"] = (
                self.word_alignment_svc.checkpoint_namespace_for(transcripts))
            alignment_cached = checkpoint.exists("word_alignment")
            if (not alignment_cached and not getattr(args, "keep_models", False)
                    and (self.step_enabled(args, "refinement") or relabel_on)):
                self.refinement_svc.unload()
            try:
                transcripts = self._align_words(
                    checkpoint, stage_out, transcripts, audio_data, speech_segments)
            finally:
                if not getattr(args, "keep_models", False):
                    self._defer_or_run(self.word_alignment_svc.unload)

        if getattr(args, "stop_after", None) == "word_alignment":
            if self.logger:
                self.logger.info(
                    "Stopping pipeline after word_alignment as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "word_alignment"})
            return transcripts

        # 7d. Conversation exports: cut self-contained two-person stretches, judged by
        # the same resident LLM. Opt-in and controlled by the active profile.
        conversation_exports_on = (self.conversation_export_svc is not None
                    and opt_in_step_enabled(args, "conversation_exports")
                    # The final export pass replays checkpointed state. Clip
                    # judging is deliberately not checkpointed, therefore its
                    # dedicated stage is the only invocation allowed to ask
                    # the LLM and write its files.
                    and not getattr(args, "postprocess_only", False))
        if conversation_exports_on and transcripts is not None:
            self._export_conversation_exports(
                stage_out, transcripts, audio_data, music_map, output_dir, audio_path,
                speech_segments=speech_segments)

        if getattr(args, "stop_after", None) == "conversation_exports":
            if not getattr(args, "keep_models", False):
                self._defer_or_run(self.refinement_svc.unload)
            if self.logger:
                self.logger.info(
                    "Stopping pipeline after conversation_exports as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "conversation_exports"})
            return transcripts

        if self.step_enabled(args, "refinement") or relabel_on or conversation_exports_on:
            if not getattr(args, "keep_models", False):
                self._defer_or_run(self.refinement_svc.unload)

        # Optional independent corpus. It runs after relabel/alignment, and its
        # audio is built from strict separation tracks rather than the
        # conversation-exporter's time-gated mixture. A blank config value returns above
        # without creating a root directory.
        clean_dataset_result = self._export_clean_two_channel_dataset(
            args, config, transcripts, speech_segments, audio_data,
            music_map, audio_path)

        
        # 8. Xuất kết quả
        save_path = output_dir
        os.makedirs(save_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        if not self.step_enabled(args, "export"):
            if self.logger:
                self.logger.info("Step 'export' is off in the profile; skipping")
            extra = ({"clean_two_channel_dataset": clean_dataset_result}
                     if clean_dataset_result else None)
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "audio_name": base_name,
                                      "music_enabled": getattr(args, "music", False)},
                                     extra=extra)
            if self.logger: self.logger.info(f"Pipeline completed successfully. Results saved to {save_path}")
            return transcripts

        if transcripts is None:
            if self.logger:
                self.logger.info("Skipping full export (no transcripts available)")
            extra = ({"clean_two_channel_dataset": clean_dataset_result}
                     if clean_dataset_result else None)
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "audio_name": base_name,
                                      "music_enabled": getattr(args, "music", False)},
                                     extra=extra)
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
            "word_alignment": alignment_on,
            "word_alignment_model": (
                getattr(self.word_alignment_svc, "model_name", None)
                if alignment_on else None),
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
        manifest_extra = {"final": final}
        if clean_dataset_result:
            manifest_extra["clean_two_channel_dataset"] = clean_dataset_result
        stage_out.write_manifest(metadata, extra=manifest_extra)

        if self.logger: self.logger.info(f"Pipeline completed successfully. Results saved to {save_path}")
        return transcripts
