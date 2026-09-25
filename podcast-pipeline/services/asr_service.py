import concurrent.futures
import itertools
import shutil
import tempfile
import threading
import librosa
from typing import List
from schemas.audio import AudioData
from schemas.segment import SpeechSegment
from schemas.transcript import TranscriptSegment
from algorithms.asr.rover import RoverEnsembler
import numpy as np
from algorithms.asr.hallucination import filter_short_segment_outputs
from utils.audio_normalize import normalize_for_asr, remove_dc, measure
from services.asr_scheduler import AsrScheduler, ByteBudget, Lane, LaneWorker
from utils import profiling
from utils.asr_progress import AsrProgress

# Segments shorter than this are padded with surrounding audio before ASR.
CONTEXT_PAD_BELOW = 2.0
CONTEXT_PAD_SECONDS = 2.0
EDGE_PAD_SECONDS = 0.02
def estimate_replica_gain(remaining, rate, load_seconds, replica_rate,
                          peer_remaining=0.0):
    """Seconds a second Qwen worker would take off the ASR stage.

    The formula is the plan's (section 6.3): T_old = W / r1 for the worker
    already running, T_new = L + (W - r1*L) / (r1 + r2) once a replica that
    takes L seconds to load joins it. Whatever the running worker finishes
    during the load is gone before the replica can take it, so a small W is
    finished alone.

    The stage ends when its slowest model does, so both times are compared
    against the other models' remaining work. While PhoWhisper is still the
    tail, finishing Qwen sooner saves nothing -- the gain is zero however
    large Qwen's backlog is.

    Returns (gain, t_old, t_new), all in seconds.
    """
    if rate <= 0 or remaining <= 0:
        return 0.0, 0.0, 0.0
    t_old = remaining / rate
    left = remaining - rate * load_seconds
    if left <= 0 or replica_rate <= 0:
        t_new = t_old
    else:
        t_new = load_seconds + left / (rate + replica_rate)
    peer = max(0.0, float(peer_remaining or 0.0))
    gain = max(t_old, peer) - max(t_new, peer)
    return max(0.0, gain), t_old, t_new


class _Prepared:
    """One file's clips, ready for the lanes."""

    def __init__(self, valid_segments, audios_16k, core_audios_16k, dummy_vads,
                 chunk_indices, word_offsets):
        self.valid_segments = valid_segments
        self.audios_16k = audios_16k
        self.core_audios_16k = core_audios_16k
        self.dummy_vads = dummy_vads
        self.chunk_indices = chunk_indices
        self.word_offsets = word_offsets


class _SegMeta:
    """The fields of a SpeechSegment the vote reads, without its audio arrays."""
    __slots__ = ("index", "start", "end", "speaker", "bs_roformer", "bss",
                 "has_music", "bss_failed_spans")

    def __init__(self, seg):
        self.index, self.start, self.end = seg.index, seg.start, seg.end
        self.speaker = seg.speaker
        self.bs_roformer = seg.bs_roformer
        self.bss = seg.bss
        self.has_music = getattr(seg, "has_music", False)
        self.bss_failed_spans = getattr(seg, "bss_failed_spans", []) or []


_EMPTY_RESULT = {"whisper": ("", None, []), "phowhisper": "", "qwen3": ""}


class ASRService:
    """Coordinates MoE ASR models and ROVER ensemble."""
    
    def __init__(self, whisper=None, phowhisper=None, qwen3=None, logger=None,
                 model_loader=None, qwen3_service=None,
                 qwen3_replica_service=None, performance_config=None,
                 performance_monitor=None,
                 language: str = "vi", batch_size: int = 4,
                 keep_models: bool = False, edge_pad: float = EDGE_PAD_SECONDS,
                 replica_factories=None, asr_workers=None, asr_placement=None):
        self._whisper = whisper
        self._phowhisper = phowhisper
        self._qwen3 = qwen3
        self.logger = logger
        self.model_loader = model_loader
        self.qwen3_service = qwen3_service
        self.qwen3_replica_service = qwen3_replica_service
        self.performance_config = performance_config or {}
        self.performance_monitor = performance_monitor
        # What the last replica actually cost and delivered in this process.
        # None until one has run; the profile's estimates stand in until then.
        self._replica_load_seconds = None
        self._replica_speed_ratio = None
        self.language = language
        self.batch_size = batch_size
        self.keep_models = keep_models
        # PipelineService turns keep_models on for the whole ASR stage; what the
        # user asked for is what decides whether a finished model is released.
        self._user_keep_models = bool(keep_models)
        self.edge_pad = max(0.0, float(edge_pad))
        self._warned = False
        # Cross-file scheduling (see services/asr_scheduler.py).
        # replica_factories: {"qwen3"|"whisper"|"phowhisper": f(gpu, batch) -> (model, release)}
        # asr_workers: {"qwen3"|"whisper": worker service whose process is stopped on release}
        # asr_placement: {"qwen3"|"whisper"|"phowhisper": physical GPU id}
        self.replica_factories = dict(replica_factories or {})
        self.asr_workers = dict(asr_workers or {})
        self.asr_placement = dict(asr_placement or {})
        self._cross_active = False
        self._scheduler = None
        self._scheduler_lock = threading.Lock()
        self._expected_files = None
        self._skipped_before_start = 0
        self._tmp_dir = None
        self._local = threading.local()
        self._file_counter = itertools.count(1)
        self._budget = None
        self._vote_executor = None

    # Models are fetched from the loader on use, not captured at construction.
    # PipelineService loads each stage's models when that stage runs, so a
    # reference taken here would be None for every stage that had not loaded
    # yet -- and would stay None after it did.
    def _model(self, held, name):
        if held is not None:
            return held
        return self.model_loader.get(name) if self.model_loader else None

    @property
    def whisper(self):
        return self._model(self._whisper, "whisper")

    @whisper.setter
    def whisper(self, model):
        self._whisper = model

    @property
    def phowhisper(self):
        return self._model(self._phowhisper, "phowhisper")

    @phowhisper.setter
    def phowhisper(self, model):
        self._phowhisper = model

    @property
    def qwen3(self):
        return self._model(self._qwen3, "qwen3")

    @qwen3.setter
    def qwen3(self, model):
        self._qwen3 = model

    def _warn_missing(self):
        """Report absent voters once, after the models have had a chance to load.

        Warning from __init__ instead would fire before this stage's loader has
        run and name every model as missing on every run.
        """
        if self._warned or not self.logger:
            return
        self._warned = True
        if not self.whisper:
            self.logger.warning(
                "Whisper is not loaded (requires --ASRMoE with --lang vi); "
                "ROVER will vote without it."
            )
        if not self.qwen3:
            self.logger.warning("Qwen3-ASR is not loaded; ROVER will vote without it.")

    def active_models(self) -> List[str]:
        """Names of the ASR models actually available for this run."""
        names = []
        if self.whisper: names.append("whisper")
        if self.phowhisper: names.append("phowhisper")
        if self.qwen3: names.append("qwen3")
        return names
    def _edge_pad_bounds(self, segments, total_dur: float) -> dict:
        """Nới mỗi segment ±edge_pad, nhưng clamp để không lấn segment kế bên
        và không pad vào vùng overlap. Trả về {id(seg): (start_mới, end_mới)}."""
        if self.edge_pad <= 0:
            return {id(s): (s.start, s.end) for s in segments}

        ordered = sorted(segments, key=lambda s: (s.start, s.end))
        bounds = {}
        for i, seg in enumerate(ordered):
            # Về sau: nếu là segment đầu -> full pad; nếu không -> tối đa nửa gap
            back = self.edge_pad if i == 0 else min(
                self.edge_pad, max(0.0, (seg.start - ordered[i - 1].end) / 2))
            # Về trước: tương tự với segment sau
            fwd = self.edge_pad if i == len(ordered) - 1 else min(
                self.edge_pad, max(0.0, (ordered[i + 1].start - seg.end) / 2))
            bounds[id(seg)] = (max(0.0, seg.start - back), min(total_dur, seg.end + fwd))
        return bounds


    def _run_whisper(self, audio_16k, dummy_vad, language=None):
        if not self.whisper:
            return "", None, []
        language = language or self.language
        try:
            res = self.whisper.transcribe(audio_16k, dummy_vad, language=language)
            return res.get("text", ""), res.get("language", language), res.get("words", [])
        except Exception as e:
            if self.logger: self.logger.error(f"Whisper error: {e}")
            return "", None, []

    def _run_phowhisper(self, audio_16k):
        if not self.phowhisper: return ""
        try:
            return self.phowhisper.transcribe(audio_16k)
        except Exception as e:
            if self.logger: self.logger.error(f"PhoWhisper error: {e}")
            return ""

    def _run_qwen3(self, audio_16k, chunk_index: str, tmp_dir: str):
        if not self.qwen3: return ""
        path = None
        try:
            import os
            import numpy as np
            # The audio is already float32 @ 16 kHz here; a .npy dump lets the
            # worker memory-map it instead of decoding a WAV per segment.
            path = os.path.join(tmp_dir, f"qwen3_{chunk_index}.npy")
            np.save(path, np.ascontiguousarray(audio_16k, dtype=np.float32))
            text = self.qwen3.transcribe(path)
            if not text and self.logger:
                self.logger.warning(f"Qwen3 returned empty for chunk {chunk_index}")
            return text
        except Exception as e:
            if self.logger: self.logger.error(f"Qwen3 error: {e}")
            return ""
        finally:
            if path:
                import os
                try:
                    if os.path.exists(path): os.remove(path)
                except OSError:
                    pass

    def _run_whisper_batch(self, audios_16k: list, dummy_vads: list, callback=None,
                           released_event=None) -> list:
        """Transcribe every clip, then signal whether this card was freed.

        `released_event` is what lets a replica claim Whisper's GPU, so it is
        set only once the weights are actually gone. Under --keep_models they
        stay resident for the next file, and a replica loading onto the same
        card would be competing with them for VRAM rather than inheriting it.
        """
        if not self.whisper:
            if callback:
                for _ in audios_16k: callback()
            if released_event is not None:
                released_event.set()
            return [("", None, [])] * len(audios_16k)

        try:
            batch = self.whisper.transcribe_batch(
                audios_16k, dummy_vads, language=self.language,
                callback=callback)
            results = [(item.get("text", ""),
                        item.get("language", self.language),
                        item.get("words", [])) for item in batch]
        except Exception as e:
            if self.logger:
                self.logger.error(f"Whisper batch error: {e}; retrying individually")
            results = []
            for a, v in zip(audios_16k, dummy_vads):
                results.append(self._run_whisper(a, v))
                if callback:
                    callback()
        if getattr(self, "model_loader", None) and not self.keep_models:
            self.model_loader.unload("whisper")
            if released_event is not None:
                released_event.set()
        return results

    def _run_phowhisper_batch(self, audios_16k: list, callback=None) -> list:
        if not self.phowhisper:
            if callback:
                for _ in audios_16k: callback()
            return [""] * len(audios_16k)
        try:
            # Batch size is kept modest because Qwen3 shares the GPU.
            # No batch_size here: PhoWhisper uses the one it was configured
            # with. self.batch_size comes from models.qwen3 -- a different
            # model, on a different device, with its own memory budget.
            res = self.phowhisper.transcribe_batch(
                audios_16k, logger=self.logger, callback=callback
            )
            if getattr(self, "model_loader", None) and not self.keep_models:
                self.model_loader.unload("phowhisper")
            return res
        except Exception as e:
            if self.logger: self.logger.error(f"PhoWhisper batch error: {e}")
            return [""] * len(audios_16k)

    def _learn_from_replica(self, load_seconds, rate_alone, jobs_together,
                            seconds_together, record=None):
        """Keep what the replica really cost and added, for the next decision.

        The throughput a replica adds is measured, not assumed: on 2x T4 the
        first one loaded in 31s and lifted combined throughput from 1.36 to
        1.44 jobs/s -- a 6% gain that the textbook r2 = r1 predicted as 100%.
        The ratio is what the pair did together over what one did alone, so
        CPU, PCIe or disk contention that slows the first worker is counted.
        """
        if seconds_together <= 0 or rate_alone <= 0:
            return
        together = jobs_together / seconds_together
        self._replica_load_seconds = float(load_seconds)
        self._replica_speed_ratio = max(0.0, together / rate_alone - 1.0)
        if record:
            record("asr_replica_measured", load_seconds=round(load_seconds, 1),
                   rate_alone=round(rate_alone, 3), rate_together=round(together, 3),
                   speed_ratio=round(self._replica_speed_ratio, 3))
        if self.logger:
            self.logger.info(
                f"[ASR scheduler] replica measured: load {load_seconds:.0f}s, "
                f"{rate_alone:.2f} -> {together:.2f} jobs/s "
                f"(adds {self._replica_speed_ratio:.0%})")

    def _run_qwen3_batch(self, audios_16k: list, chunk_indices: list,
                         tmp_dir: str, callback=None, replica_event=None,
                         peer_remaining=None) -> list:
        if not self.qwen3:
            if callback:
                for _ in audios_16k: callback()
            return [""] * len(audios_16k)

        import os
        import queue
        import threading
        import time
        paths = []
        replica_service = self.qwen3_replica_service
        try:
            jobs = []
            for audio, index in zip(audios_16k, chunk_indices):
                path = os.path.join(tmp_dir, f"qwen3_{index}.npy")
                np.save(path, np.ascontiguousarray(audio, dtype=np.float32))
                paths.append(path)
                jobs.append((str(index), path))
            work = queue.Queue()
            for position, job in enumerate(jobs):
                work.put((position, job))
            results = [""] * len(jobs)
            result_lock = threading.Lock()
            batch_size = max(1, self.batch_size)

            done = {"primary": 0, "replica": 0}

            def consume(client, who="primary"):
                while True:
                    picked = []
                    try:
                        picked.append(work.get_nowait())
                    except queue.Empty:
                        return
                    while len(picked) < batch_size:
                        try:
                            picked.append(work.get_nowait())
                        except queue.Empty:
                            break
                    values = client.transcribe_batch(
                        [job for _position, job in picked], language=self.language)
                    if len(values) < len(picked):
                        values.extend([""] * (len(picked) - len(values)))
                    with result_lock:
                        done[who] += len(picked)
                        for (position, _job), value in zip(picked, values):
                            results[position] = value
                            if callback:
                                callback()
                    for _ in picked:
                        work.task_done()

            cfg = self.performance_config
            min_jobs = int(cfg.get("replica_min_pending_jobs", batch_size * 2))
            min_gain = float(cfg.get("replica_min_gain_seconds", 15.0))
            load_estimate = (self._replica_load_seconds
                             if self._replica_load_seconds is not None
                             else float(cfg.get("replica_load_seconds", 30.0)))
            speed_ratio = (self._replica_speed_ratio
                           if self._replica_speed_ratio is not None
                           else float(cfg.get("replica_speed_ratio", 0.5)))
            # A rate from the first batch alone is mostly warm-up.
            min_samples = max(2, 2 * batch_size)
            total_jobs = len(jobs)

            def record(event, **fields):
                if self.performance_monitor:
                    self.performance_monitor.record(event, model="qwen3", **fields)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                started = time.time()
                primary = pool.submit(consume, self.qwen3, "primary")
                replica = None
                replica_ready_at = None
                done_at_ready = 0
                skipped_reason = None
                while not primary.done():
                    time.sleep(0.05)
                    if (replica is not None or replica_service is None
                            or replica_event is None or not replica_event.is_set()):
                        continue
                    with result_lock:
                        primary_done = done["primary"]
                    remaining = total_jobs - primary_done
                    if remaining < min_jobs:
                        # Remaining only shrinks from here: this call is settled.
                        if skipped_reason is None:
                            record("asr_replica_skipped", reason="backlog_below_threshold",
                                   pending_jobs=remaining, threshold=min_jobs)
                        replica_service = None
                        continue
                    elapsed = time.time() - started
                    if primary_done < min_samples or elapsed <= 0:
                        continue
                    peer = peer_remaining() if peer_remaining else 0.0
                    if peer is None:
                        # The other model has not reported yet, so which one is
                        # the tail is unknown. Wait rather than guess.
                        continue
                    rate = primary_done / elapsed
                    gain, t_old, t_new = estimate_replica_gain(
                        remaining, rate, load_estimate, rate * speed_ratio, peer)
                    estimate = dict(pending_jobs=remaining, rate=round(rate, 3),
                                    load_seconds=round(load_estimate, 1),
                                    speed_ratio=round(speed_ratio, 3),
                                    peer_remaining=round(peer, 1),
                                    t_old=round(t_old, 1), t_new=round(t_new, 1),
                                    gain=round(gain, 1), min_gain=min_gain)
                    if gain <= 0 or gain < min_gain:
                        # Strict: a replica that saves nothing never opens,
                        # even with the threshold at zero. Re-evaluated every
                        # tick, since the gain can grow once the other model
                        # finishes. Recorded once per reason.
                        reason = "not_the_tail" if peer >= t_old else "gain_below_threshold"
                        if reason != skipped_reason:
                            skipped_reason = reason
                            record("asr_replica_skipped", reason=reason, **estimate)
                        continue
                    if self.logger:
                        self.logger.info(
                            f"[ASR scheduler] starting Qwen replica: {remaining} job(s) left, "
                            f"{t_old:.0f}s alone vs {t_new:.0f}s with a replica "
                            f"(expected gain {gain:.0f}s)")
                    record("asr_replica_loading", **estimate)
                    load_started = time.time()
                    try:
                        replica_service.spawn()
                        replica_service.wait_ready()
                        from models.qwen3_asr import Qwen3ASRClient
                        replica_ready_at = time.time()
                        with result_lock:
                            done_at_ready = done["primary"]
                        replica = pool.submit(
                            consume, Qwen3ASRClient(replica_service.process), "replica")
                        pid = getattr(replica_service.process, "pid", None)
                        if self.performance_monitor and pid:
                            register = getattr(self.performance_monitor,
                                               "register_process", None)
                            if register:
                                register(pid, "qwen3_replica")
                        record("asr_replica_started", pid=pid,
                               load_seconds=round(replica_ready_at - load_started, 1),
                               pending_jobs=total_jobs - done_at_ready)
                    except Exception as exc:
                        if self.logger:
                            self.logger.warning(
                                f"Qwen replica could not start: {type(exc).__name__}: {exc}")
                        try:
                            replica_service.stop()
                        except Exception:
                            pass
                        replica_service = None
                        record("asr_replica_failed", error=f"{type(exc).__name__}: {exc}")
                primary.result()
                if replica is not None:
                    replica.result()
                    finished = time.time()
                    self._learn_from_replica(
                        load_seconds=replica_ready_at - load_started,
                        rate_alone=done_at_ready / max(1e-6, replica_ready_at - started),
                        jobs_together=total_jobs - done_at_ready,
                        seconds_together=finished - replica_ready_at,
                        record=record)
        finally:
            if (replica_service is not None
                    and replica_service.process is not None):
                replica_service.stop()
                if self.performance_monitor:
                    self.performance_monitor.record(
                        "asr_replica_stopped", model="qwen3")
            for path in paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
        # The worker is owned by main.py's finally block; stopping it here would
        # leave a dead Popen behind that a second process() call would write to.
        return results

    def _transcribe_per_file(self, audios_16k, core_audios_16k, dummy_vads,
                             chunk_indices, tmp_dir):
        """The original path: three models for one file, then wait for all three."""
        import time

        progress = {"whisper": 0, "pho": 0, "qwen": 0}
        total = len(audios_16k)
        whisper_released = threading.Event()

        # The same reporter the cross-file scheduler uses: plain log lines, one
        # writer, numbers that only move forward. It replaces a `\r` line per file
        # that files in flight overwrote each other on.
        view = AsrProgress(self.logger, interval=10.0,
                           label=profiling.current_file())
        view.expect_files(1)
        present = {"whisper": bool(self.whisper), "pho": bool(self.phowhisper),
                   "qwen": bool(self.qwen3)}
        lane_names = {"whisper": "whisper", "pho": "phowhisper", "qwen": "qwen3"}
        for key, name in lane_names.items():
            if present[key]:
                view.queued("file", name, total)
                view.lane_started(name)
        view.start()

        def bump(key):
            progress[key] += 1
            if present[key]:
                view.done(lane_names[key])

        def cb_whisper(): bump("whisper")
        def cb_pho(): bump("pho")
        def cb_qwen(): bump("qwen")

        pho_started = time.time()

        def pho_remaining():
            """PhoWhisper's projected seconds left; None until it has reported."""
            if not self.phowhisper:
                return 0.0
            finished = progress["pho"]
            if finished >= total:
                return 0.0
            if finished == 0:
                return None
            return (total - finished) * (time.time() - pho_started) / finished

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                fw = executor.submit(self._run_whisper_batch, audios_16k, dummy_vads,
                                     cb_whisper, whisper_released)
                fp = executor.submit(self._run_phowhisper_batch, core_audios_16k, cb_pho)
                fq = executor.submit(
                    self._run_qwen3_batch, core_audios_16k, chunk_indices,
                    tmp_dir, cb_qwen, whisper_released, pho_remaining)

                try:
                    whisper_results = fw.result()
                finally:
                    # A Whisper that died still holds nothing, but the flag is what
                    # the Qwen replica waits on; leaving it clear on the failure
                    # path would just deny the card to the model still working.
                    if not self.keep_models:
                        whisper_released.set()
                pho_results = fp.result()
                qwen_results = fq.result()
            
            view.lanes_done("file")
        finally:
            view.stop()

        return whisper_results, pho_results, qwen_results

    # -- cross-file scheduling ---------------------------------------------------
    @property
    def cross_file_enabled(self) -> bool:
        return self._cross_active

    def begin_cross_file_stage(self, total_files: int):
        """Open a stage in which `total_files` files will flow through shared lanes."""
        if not self.performance_config.get("cross_file"):
            return
        with self._scheduler_lock:
            self._cross_active = True
            self._expected_files = int(total_files)
            self._skipped_before_start = 0
            self._scheduler = None

    def settle_file(self):
        """Call once when a file's ASR turn is over.

        A file that never submitted (checkpointed, failed, or without segments)
        must still count, or the scheduler would wait for it forever before
        releasing the models that have run out of work.
        """
        submitted = getattr(self._local, "submitted", False)
        self._local.submitted = False
        if submitted or not self._cross_active:
            return
        with self._scheduler_lock:
            if self._scheduler is not None:
                self._scheduler.add_intake(1)
            else:
                self._skipped_before_start += 1

    def end_cross_file_stage(self):
        with self._scheduler_lock:
            scheduler, self._scheduler = self._scheduler, None
            tmp_dir, self._tmp_dir = self._tmp_dir, None
            self._cross_active = False
            executor, self._vote_executor = self._vote_executor, None
            self._budget = None
        if scheduler is not None:
            scheduler.shutdown()
        if executor is not None:
            executor.shutdown(wait=True)
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _get_scheduler(self):
        with self._scheduler_lock:
            if self._scheduler is None:
                self._tmp_dir = tempfile.mkdtemp(prefix="asr_lanes_")
                scheduler = self._build_scheduler(self._tmp_dir)
                if self._expected_files is not None:
                    scheduler.expect_files(self._expected_files)
                if self._skipped_before_start:
                    scheduler.add_intake(self._skipped_before_start)
                    self._skipped_before_start = 0
                self._scheduler = scheduler
            return self._scheduler

    def _build_scheduler(self, tmp_dir):
        cfg = self.performance_config
        shared = int(cfg.get("shared_batch_size", 16))
        boost = int(cfg.get("boost_batch_size", 48))
        models = {"whisper": self.whisper, "phowhisper": self.phowhisper,
                  "qwen3": self.qwen3}
        models = {kind: model for kind, model in models.items() if model}
        gpus = {kind: self.asr_placement.get(kind) for kind in models}
        lanes = []
        for kind, model in models.items():
            shares = gpus[kind] is not None and any(
                other != kind and gpus[other] == gpus[kind] for other in models)
            if kind == "qwen3":
                batch = self.batch_size          # the engine's own size is fixed
            elif shares:
                batch = shared
            else:
                batch = int(getattr(model, "batch_size", 0) or shared)
            holder = {"model": model}
            worker_service = self.asr_workers.get(kind)
            # A pool of worker processes is served by one lane worker per process, so
            # the lane's threads lease distinct processes and run side by side.
            copies = max(1, len(getattr(worker_service, "services", ()) or ()))
            ready = (worker_service.wait_ready
                     if worker_service is not None
                     and hasattr(worker_service, "wait_ready") else None)
            runner = self._runner_for(kind, holder, tmp_dir)
            release = self._release_fn(kind, holder)
            made = []
            for _ in range(copies):
                worker = LaneWorker(kind, gpus[kind], runner, batch, release=release)
                worker.ready = ready
                made.append(worker)
            lane = Lane(kind, gpus[kind], made[0], empty=_EMPTY_RESULT[kind],
                        replica_factory=self._replica_factory(kind, tmp_dir),
                        boostable=(kind != "qwen3"))
            lane.workers.extend(made[1:])
            lanes.append(lane)
        return AsrScheduler(lanes, boost_batch=boost, config=cfg, logger=self.logger,
                            monitor=self.performance_monitor)

    def _release_fn(self, kind, holder):
        """Frees a finished model. The holder is cleared first: the runner reads
        the model through it, so nothing here keeps the weights alive."""
        if self._user_keep_models:
            return None

        def release():
            holder["model"] = None
            if self.model_loader:
                self.model_loader.unload(kind)
            worker = self.asr_workers.get(kind)
            if worker is not None and getattr(worker, "process", None) is not None:
                worker.stop()
        return release

    def _replica_factory(self, kind, tmp_dir):
        factory = self.replica_factories.get(kind)
        if factory is None:
            return None

        def make(gpu, batch):
            model, release_model = factory(gpu, batch)
            holder = {"model": model}

            def release():
                holder["model"] = None
                release_model()
            return LaneWorker(kind, gpu, self._runner_for(kind, holder, tmp_dir),
                              batch, release=release)
        return make

    def _runner_for(self, kind, holder, tmp_dir):
        if kind == "whisper":
            return self._whisper_runner(holder)
        if kind == "phowhisper":
            return self._phowhisper_runner(holder)
        return self._qwen3_runner(holder, tmp_dir)

    @staticmethod
    def _held(holder):
        model = holder["model"]
        if model is None:
            raise RuntimeError("this ASR model was released")
        return model

    def _whisper_runner(self, holder):
        def run(payloads, batch_size):
            model = self._held(holder)
            audios = [audio for audio, _vad in payloads]
            vads = [vad for _audio, vad in payloads]
            try:
                batch = model.transcribe_batch(
                    audios, vads, language=self.language, batch_size=batch_size)
                return [(item.get("text", ""), item.get("language", self.language),
                         item.get("words", [])) for item in batch]
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Whisper batch error: {e}; retrying individually")
                rows = []
                for audio, vad in payloads:
                    try:
                        res = model.transcribe(audio, vad, language=self.language)
                        rows.append((res.get("text", ""),
                                     res.get("language", self.language),
                                     res.get("words", [])))
                    except Exception as inner:
                        if self.logger:
                            self.logger.error(f"Whisper error: {inner}")
                        rows.append(("", None, []))
                return rows
        return run

    def _phowhisper_runner(self, holder):
        def run(payloads, batch_size):
            model = self._held(holder)
            try:
                return list(model.transcribe_batch(
                    list(payloads), batch_size=batch_size, logger=self.logger))
            except Exception as e:
                if self.logger:
                    self.logger.error(f"PhoWhisper batch error: {e}")
                return [""] * len(payloads)
        return run

    def _qwen3_runner(self, holder, tmp_dir):
        import os
        import uuid

        def run(payloads, batch_size):
            client = self._held(holder)
            paths, jobs = [], []
            try:
                for position, (_index, audio) in enumerate(payloads):
                    path = os.path.join(tmp_dir, f"qwen3_{uuid.uuid4().hex}.npy")
                    np.save(path, np.ascontiguousarray(audio, dtype=np.float32))
                    paths.append(path)
                    jobs.append((str(position), path))
                values = client.transcribe_batch(jobs, language=self.language)
                if len(values) < len(jobs):
                    values = list(values) + [""] * (len(jobs) - len(values))
                return list(values)
            finally:
                for path in paths:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        return run

    def _transcribe_cross_file(self, audios_16k, core_audios_16k, dummy_vads, chunk_indices):
        scheduler = self._get_scheduler()
        payloads = {}
        if self.whisper:
            payloads["whisper"] = list(zip(audios_16k, dummy_vads))
        if self.phowhisper:
            payloads["phowhisper"] = list(core_audios_16k)
        if self.qwen3:
            payloads["qwen3"] = list(zip(chunk_indices, core_audios_16k))
        file_id = f"file-{next(self._file_counter)}"
        ticket = scheduler.submit(file_id, payloads)
        self._local.submitted = True
        self._local.progress = (scheduler.progress, file_id)
        results = ticket.wait()
        count = len(audios_16k)
        return (results.get("whisper") or [("", None, [])] * count,
                results.get("phowhisper") or [""] * count,
                results.get("qwen3") or [""] * count)

    def _prepare(self, segments, audio):
        """Cut, pad, resample and level-condition every segment's clip (CPU only)."""
        valid_segments = []
        audios_16k = []
        core_audios_16k = []
        dummy_vads = []
        chunk_indices = []
        word_offsets = []
        sr = audio.sample_rate

        total_samples = len(audio.waveform)
        total_dur = total_samples / sr
        edge_bounds = self._edge_pad_bounds(segments, total_dur)
        for seg in segments:
            # Short segments get surrounding audio as context. Whisper pads its
            # input to 30s regardless, so a 0.24s backchannel arrives as 99.2%
            # silence and the decoder fills that void with training-set
            # boilerplate ("Hẹn gặp lại các bạn..."). Real audio either side
            # gives it something to condition on. The VAD range below still
            # marks only the segment itself, so the extra audio informs the
            # encoder without being transcribed.
            core_start, core_end = edge_bounds[id(seg)]
            pad = CONTEXT_PAD_SECONDS if (core_end - core_start) < CONTEXT_PAD_BELOW else 0.0

            start_frame = int(core_start * sr)
            end_frame = int(core_end * sr)
            pad_frames = int(pad * sr)
            lo = max(0, start_frame - pad_frames)
            hi = min(total_samples, end_frame + pad_frames)

            if seg.audio is not None:
                seg_start_frame = int(seg.start * sr)
                seg_end_frame = seg_start_frame + len(seg.audio)
                if start_frame < seg_start_frame or end_frame > seg_end_frame:
                    core = np.concatenate([
                        audio.waveform[start_frame:seg_start_frame],  # pad đầu, từ mixture
                        seg.audio,                            # lõi đã tách, nguyên vẹn
                        audio.waveform[seg_end_frame:end_frame],       # pad cuối, từ mixture
                    ])
                else:
                    core = seg.audio
                if pad > 0:
                    # Pad from the mixture: the separated track only covers the
                    # segment, and its neighbours belong to the other speaker
                    # anyway -- which is exactly the context that tells the
                    # decoder a conversation is happening here.
                    raw_audio = np.concatenate([
                        audio.waveform[lo:start_frame], core, audio.waveform[end_frame:hi]])
                    lead = (start_frame - lo) / sr
                else:
                    raw_audio = core
                    lead = 0.0
            else:
                raw_audio = audio.waveform[lo:hi]
                lead = (start_frame - lo) / sr

            if len(raw_audio) == 0:
                if self.logger: self.logger.warning(f"Segment {seg.index} has empty audio, skipping")
                continue

            if sr != 16000:
                audio_16k = librosa.resample(raw_audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = raw_audio

            # Level-condition after resampling and after the context pad is
            # concatenated, so segment and padding receive the same gain and
            # the join between them stays seamless. One scalar per buffer: it
            # moves the whole clip to the loudness the recognisers were trained
            # at without altering its spectrum. Separated audio arrives quieter
            # than the mixture -- the interferer has been removed -- and a quiet
            # clip is where Whisper starts emitting training-set boilerplate.
            pre = measure(audio_16k)
            audio_16k = normalize_for_asr(remove_dc(audio_16k))
            if self.logger and pre["rms"] > 0:
                post = measure(audio_16k)
                gain = post["rms"] / pre["rms"]
                if gain > 2.0 or gain < 0.5:
                    self.logger.debug(
                        f"[ASR:level] seg {seg.index} rms {pre['rms']:.4f} -> "
                        f"{post['rms']:.4f} (x{gain:.2f}) peak {post['peak']:.3f}")

            core_len = (end_frame - start_frame) / sr
            if (seg.end - seg.start) * 16000 < 160:
                if self.logger: self.logger.warning(f"Segment {seg.index} too short ({seg.end - seg.start:.3f}s), skipping")
                continue

            # Only whisperx honours an explicit VAD range, so only it can be
            # given padded audio and told which part to transcribe. PhoWhisper
            # builds its own full-span range and the Qwen3 worker takes a bare
            # array, so both would transcribe the padding as if it were the
            # segment. They keep the unpadded audio.
            i0 = int(lead * 16000)
            core_16k = audio_16k[i0:i0 + int(core_len * 16000)] if lead > 0 else audio_16k

            valid_segments.append(seg)
            audios_16k.append(audio_16k)
            core_audios_16k.append(core_16k)
            # Transcribe only the segment, not the padding.
            dummy_vads.append([{"start": lead, "end": min(lead + core_len, len(audio_16k) / 16000)}])
            chunk_indices.append(seg.index)
            word_offsets.append(lo / sr)
        return _Prepared(valid_segments, audios_16k, core_audios_16k, dummy_vads,
                         chunk_indices, word_offsets)

    def _vote(self, segs, whisper_results, pho_results, qwen_results, word_offsets,
              enable_word_timestamps=False):
        """Filter hallucinations, run ROVER and build the transcripts.

        `segs` need only carry the fields read below (index, start, end, speaker,
        bs_roformer, bss, has_music, bss_failed_spans), so a vote can run long after
        the file's audio has been dropped. Returns (transcripts, hallucinations_dropped).
        """
        results = []
        rover = RoverEnsembler()
        hallucinations_dropped = 0
        for i, seg in enumerate(segs):
            t_whisper, lang, words = whisper_results[i]
            t_pho = pho_results[i]
            t_qwen = qwen_results[i]
            duration = seg.end - seg.start

            # ROVER anchors its alignment on the first transcript, so a Whisper
            # hallucination on a backchannel does not merely get a vote -- it
            # becomes the skeleton every other model is aligned against. Drop
            # bad candidates before voting rather than after.
            candidates = [t_whisper, t_qwen, t_pho]
            cleaned = filter_short_segment_outputs(
                candidates, duration, logger=self.logger, segment_id=seg.index
            )
            hallucinations_dropped += sum(
                1 for before, after in zip(candidates, cleaned) if before != after
            )
            # The blanked values are also what gets exported per model: a
            # hallucination is not evidence of what Whisper heard, and leaving
            # it in text_whisper would poison anything training off these fields.
            t_whisper, t_qwen, t_pho = cleaned

            # ROVER aligns everything against its first entry, so the anchor has
            # to be a transcript we still trust. Whisper stays the anchor -- it
            # is the strongest model when it behaves -- unless this segment is
            # exactly where it misbehaved and got blanked above. Reordering
            # unconditionally on short clips measurably degraded segments where
            # Whisper was fine.
            if t_whisper:
                order = [t_whisper, t_qwen, t_pho]
            else:
                order = [t_pho, t_qwen, t_whisper]

            final_text = rover.align_and_vote(order)

            if enable_word_timestamps and words:
                for w in words:
                    w["start"] += word_offsets[i]
                    w["end"] += word_offsets[i]

            results.append(TranscriptSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker,
                text=final_text,
                text_whisper=t_whisper,
                text_phowhisper=t_pho,
                text_qwen3=t_qwen,
                # Only Whisper detects language; without it, report the requested
                # language rather than inventing one.
                language=lang or self.language,
                bs_roformer=seg.bs_roformer,
                bss=seg.bss,
                has_music=getattr(seg, "has_music", False),
                # Carried through so the review page can warn that these spans
                # still hold two voices; without it the reviewer has no way to
                # tell a clean segment from one separation gave up on.
                unseparated=[
                    {"start": float(a), "end": float(b), "reason": str(r)}
                    for a, b, r, _detail in getattr(seg, "bss_failed_spans", []) or []
                ] or None,
                words=words if enable_word_timestamps else None
            ))

        return results, hallucinations_dropped

    def _log_voted(self, results, dropped, vote_started):
        if not self.logger:
            return
        import time as _time
        self.logger.info(
            f"[ASR] {profiling.current_file() or 'file'}: {len(results)} transcripts "
            f"voted in {_time.monotonic() - vote_started:.1f}s")
        self.logger.info(f"ASR completed: {len(results)} transcripts produced")
        if dropped:
            self.logger.info(
                f"Discarded {dropped} hallucinated ASR outputs before voting")

    def process(self, segments: List[SpeechSegment], audio: AudioData, enable_word_timestamps: bool = False) -> List[TranscriptSegment]:
        self._warn_missing()
        import time as _time
        tmp_dir = tempfile.mkdtemp(prefix="qwen3_asr_")
        try:
            if self.logger:
                self.logger.info(f"ASR processing {len(segments)} segments using Batched Inference")
                self.logger.info("Note: ASR models require 16kHz audio. Resampling from base 24kHz to 16kHz internally.")

            # 1. Prepare data
            prep = self._prepare(segments, audio)
            if not prep.valid_segments:
                return []

            # 2. Run the three models. Cross-file mode queues this file's clips on
            # lanes shared with the other files in flight; otherwise the three
            # models run for this file alone and meet at a barrier.
            if self.cross_file_enabled:
                whisper_results, pho_results, qwen_results = self._transcribe_cross_file(
                    prep.audios_16k, prep.core_audios_16k, prep.dummy_vads, prep.chunk_indices)
            else:
                whisper_results, pho_results, qwen_results = self._transcribe_per_file(
                    prep.audios_16k, prep.core_audios_16k, prep.dummy_vads,
                    prep.chunk_indices, tmp_dir)

            # 3. Zip and vote
            vote_started = _time.monotonic()
            tracker = getattr(self._local, "progress", None) if self.cross_file_enabled else None
            self._local.progress = None
            if tracker:
                tracker[0].voting(tracker[1])
            results, dropped = self._vote(
                prep.valid_segments, whisper_results, pho_results, qwen_results,
                prep.word_offsets, enable_word_timestamps)
            if tracker:
                tracker[0].voted(tracker[1])
            self._log_voted(results, dropped, vote_started)
            return results
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # -- asynchronous intake ------------------------------------------------------
    @property
    def async_vote_enabled(self) -> bool:
        """Whether `process_async` frees the calling thread once clips are queued."""
        return bool(self._cross_active
                    and self.performance_config.get("async_vote", False))

    def _prepared_budget(self):
        with self._scheduler_lock:
            if self._budget is None:
                cap_mb = int(self.performance_config.get("max_prepared_mb", 2048))
                self._budget = ByteBudget(cap_mb * 1024 * 1024 if cap_mb > 0 else 0)
            return self._budget

    def _vote_pool(self):
        with self._scheduler_lock:
            if not self._cross_active:
                raise RuntimeError("the ASR stage has ended")
            if self._vote_executor is None:
                workers = max(1, int(self.performance_config.get("vote_workers", 2)))
                # Threads, not processes: ROVER is pure Python so the GIL serialises
                # it, but a vote is short next to the model time it follows, and the
                # lane feeders it shares the interpreter with mostly wait on worker
                # processes. A process pool would have to fork or spawn a parent
                # that already holds CUDA state and vLLM threads; measure first.
                self._vote_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="asr-vote")
            return self._vote_executor

    @staticmethod
    def _keyed_results(file_id, chunk_indices, whisper, pho, qwen):
        """Lane results as {(file_id, segment_id, model_id): result}."""
        table = {}
        for i, index in enumerate(chunk_indices):
            table[(file_id, index, "whisper")] = whisper[i]
            table[(file_id, index, "phowhisper")] = pho[i]
            table[(file_id, index, "qwen3")] = qwen[i]
        return table

    def process_async(self, segments, audio, enable_word_timestamps: bool = False):
        """Prepare this file, queue its clips, and return a Future for its transcripts.

        The calling thread is free as soon as the clips are on the lanes, so the
        next file can be prepared while this one waits for its slowest model. When
        every lane has delivered, the vote runs on the vote pool and resolves the
        future; a lane that could not run resolves it with that error. Falls back
        to the synchronous path outside a cross-file stage.
        """
        future = concurrent.futures.Future()
        if not self.cross_file_enabled:
            try:
                future.set_result(self.process(segments, audio, enable_word_timestamps))
            except BaseException as exc:
                future.set_exception(exc)
            return future
        self._warn_missing()
        if self.logger:
            self.logger.info(f"ASR processing {len(segments)} segments using Batched Inference")
        prep = self._prepare(segments, audio)
        if not prep.valid_segments:
            future.set_result([])
            return future

        scheduler = self._get_scheduler()
        # Drop everything the vote does not need (separated tracks, the mixture)
        # so a file waiting on a slow lane holds only its clips in the queues.
        metas = [_SegMeta(seg) for seg in prep.valid_segments]
        chunk_indices = list(prep.chunk_indices)
        word_offsets = list(prep.word_offsets)
        payloads = {}
        if self.whisper:
            payloads["whisper"] = list(zip(prep.audios_16k, prep.dummy_vads))
        if self.phowhisper:
            payloads["phowhisper"] = list(prep.core_audios_16k)
        if self.qwen3:
            payloads["qwen3"] = list(zip(prep.chunk_indices, prep.core_audios_16k))
        nbytes = sum(int(getattr(a, "nbytes", 0)) for a in prep.audios_16k)
        count = len(metas)
        del prep

        budget = self._prepared_budget()
        budget.acquire(nbytes)              # backpressure: blocks this file's thread
        file_id = f"file-{next(self._file_counter)}"
        try:
            ticket = scheduler.submit(file_id, payloads)
        except BaseException:
            budget.release(nbytes)
            raise
        del payloads
        self._local.submitted = True
        progress = scheduler.progress

        def run_vote():
            import time as _time
            started = _time.monotonic()
            try:
                results = ticket.wait()
                whisper = results.get("whisper") or [("", None, [])] * count
                pho = results.get("phowhisper") or [""] * count
                qwen = results.get("qwen3") or [""] * count
                table = self._keyed_results(file_id, chunk_indices, whisper, pho, qwen)
                order = [(file_id, index) for index in chunk_indices]
                transcripts, dropped = self._vote(
                    metas,
                    [table[(f, i, "whisper")] for f, i in order],
                    [table[(f, i, "phowhisper")] for f, i in order],
                    [table[(f, i, "qwen3")] for f, i in order],
                    word_offsets, enable_word_timestamps)
            except BaseException as exc:
                progress.failed(file_id)
                future.set_exception(exc)
                return
            progress.voted(file_id)
            self._log_voted(transcripts, dropped, started)
            future.set_result(transcripts)

        def settled(error):
            budget.release(nbytes)          # the lanes no longer hold the clips
            if error is not None:
                progress.failed(file_id)
                future.set_exception(error)
                return
            progress.voting(file_id)
            try:
                self._vote_pool().submit(run_vote)
            except BaseException as exc:    # pool already shut down
                progress.failed(file_id)
                future.set_exception(exc)

        ticket.when_settled(settled)
        return future

