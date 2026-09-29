import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from schemas.audio import AudioData

class MusicService:
    """Strips a music bed out of the stretches the map points at.

    It used to also walk the segment list after diarization and re-check each
    one; that pass is gone. The sweep runs at the front of the run now, so by
    the time segments exist the beds are already out of the waveform.
    """

    def __init__(self, bs_roformer_model=None, logger=None, model_loader=None,
                 performance_config=None):
        self._bs_roformer = bs_roformer_model
        self.model_loader = model_loader
        self.logger = logger
        # BS-RoFormer owns a fixed work directory and can serve only one job
        # at a time (see _checkout_queue). Built lazily, once, and reused for
        # the life of this service -- not per call -- so two files running the
        # music stage at once share ONE checkout point instead of each
        # believing it holds an instance exclusively.
        #
        # A dict, not two attributes: parallel_stage_view hands each file a
        # copy.copy() of this service, and a shallow copy shares the dict
        # object but not a rebound attribute. With a plain `self._state = X`
        # every copy would assign its own and the queues would diverge again.
        self._checkout = {"lock": threading.Lock(), "key": None, "queue": None}
        # Runs postprocess_separated()/separate_span_postprocess() in the
        # background during strip_music_spans(), so a BS-RoFormer instance a
        # raw call just freed is picked up by the next pending raw job
        # immediately, not gated on that same thread finishing its own CPU
        # work first. Mirrors SeparationService._async_state. Shared (not
        # per-file) the same way self._checkout already is --
        # parallel_stage_view("music") only shallow-copies MusicService.
        self.performance_config = dict(performance_config or {})
        self._async_state = {
            "gpu_executor": None, "post_executor": None,
            "decode_executor": None, "decode_slots": None,
            "lock": threading.Lock(),
        }

    def _checkout_queue(self, models) -> "queue.Queue":
        """The one checkout queue for this exact set of model instances.

        Keyed by identity rather than cached unconditionally: model_loader can
        unload and reload bs_roformer between batch passes, and a queue built
        for instances that no longer exist must not be handed to a caller
        holding the new ones. Within one set of instances, every caller --
        two threads inside one file's own multi-span removal, or two
        different files running the music stage concurrently -- shares this
        same queue, which is what makes "checked out" mean something across
        callers instead of only within one strip_music_spans() call.
        """
        key = tuple(id(model) for model in models)
        state = self._checkout
        with state["lock"]:
            if state["queue"] is None or state["key"] != key:
                built = queue.Queue()
                for model in models:
                    built.put(model)
                state["key"], state["queue"] = key, built
            return state["queue"]

    def _async_runtime(self):
        """Return shared (gpu_executor, post_executor), or None to use the
        plain inline path (performance off, or a model in the pool predates
        the raw/post split)."""
        cfg = self.performance_config
        if not cfg.get("enabled", False) or not cfg.get("ordered_postprocess", True):
            return None
        model = self.bs_roformer
        if model is None:
            return None
        pool_models = getattr(model, "models", None) or [model]
        if not pool_models or not all(
                callable(getattr(m, "separate_raw", None)) for m in pool_models):
            return None
        state = self._async_state
        with state["lock"]:
            if state["gpu_executor"] is None:
                state["gpu_executor"] = ThreadPoolExecutor(
                    max_workers=len(pool_models), thread_name_prefix="music-gpu")
                post_workers = max(1, int(cfg.get("postprocess_workers", 2)))
                state["post_executor"] = ThreadPoolExecutor(
                    max_workers=post_workers, thread_name_prefix="music-post")
            return state["gpu_executor"], state["post_executor"]

    def _decode_runtime(self, pool_models):
        """Shared (decode_executor, slots) for hi-res decode-ahead, or None.

        Off unless `hires_decode_ahead` > 0. `slots` bounds how many decoded
        spans exist at once (decoded, waiting for a separator, or waiting for
        postprocessing), so the RAM held is limited however many spans a file
        has."""
        ahead = int(self.performance_config.get("hires_decode_ahead", 0) or 0)
        if ahead <= 0 or not any(callable(getattr(m, "decode_span", None))
                                 for m in pool_models):
            return None
        state = self._async_state
        with state["lock"]:
            if state["decode_executor"] is None:
                state["decode_executor"] = ThreadPoolExecutor(
                    max_workers=min(ahead, max(1, len(pool_models))),
                    thread_name_prefix="music-decode")
                state["decode_slots"] = threading.BoundedSemaphore(ahead)
            return state["decode_executor"], state["decode_slots"]

    def close_async_pools(self):
        """Shut down the background pools. Safe to call with nothing ever
        submitted, and safe to call twice."""
        state = getattr(self, "_async_state", None)
        if state is None:
            return
        with state["lock"]:
            gpu_executor, state["gpu_executor"] = state["gpu_executor"], None
            post_executor, state["post_executor"] = state["post_executor"], None
            decode_executor, state["decode_executor"] = state["decode_executor"], None
            state["decode_slots"] = None
        # Decode feeds the GPU pool and the GPU pool feeds post, so drain in
        # that order.
        for executor in (decode_executor, gpu_executor, post_executor):
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

    # Models are fetched from the loader on use, not captured at construction.
    # PipelineService loads each stage's models when that stage runs, so a
    # reference taken here would be None for every stage that had not loaded
    # yet -- and would stay None after it did.
    def _model(self, held, name):
        if held is not None:
            return held
        return self.model_loader.get(name) if self.model_loader else None

    @property
    def bs_roformer(self):
        return self._model(self._bs_roformer, "bs_roformer")

    @bs_roformer.setter
    def bs_roformer(self, model):
        self._bs_roformer = model

    @staticmethod
    def _writable_waveform(audio: AudioData) -> np.ndarray:
        """Return a waveform that can be edited in place.

        Cached audio is loaded with ``mmap_mode="r"`` by AudioService. That is
        perfect for read-only stages, but music stripping writes vocal patches
        back into the waveform before diarization. Copy only when the backing
        array is read-only or not already float32.
        """
        waveform = np.asarray(audio.waveform, dtype=np.float32)
        if waveform is not audio.waveform or not waveform.flags.writeable:
            waveform = np.array(waveform, dtype=np.float32, copy=True)
            audio.waveform = waveform
        return waveform
        
    def strip_music_spans(self, audio: AudioData, music_map, logger=None,
                          source_path: str = None):
        """Replace the music-bed stretches of the waveform with their vocals.

        Runs before diarization, so the diarizer segments audio that no longer
        has a bed under it -- and so does everything after it. The per-segment
        pass below stays as a fallback for runs without a map.

        Only stretches the map calls `music` are touched: `singing` is skipped
        rather than cleaned, because there the thing to remove would be the
        voice itself.

        The waveform is modified in place, which is what lets the rest of the
        pipeline stay unchanged. Returns the patches as (start_sample, audio)
        so they can be cached and re-applied without running the separator
        again -- run() is re-entered once per stage and reloads the audio each
        time.

        `source_path` is the original file. When it is given the separator
        decodes each span from it at the checkpoint's own 44.1kHz instead of
        reusing the 16kHz slice held here, which is where most of the SDR the
        model is chosen for actually lives. It is an optimisation, not a
        requirement: without it, or when the decode fails, the 16kHz slice is
        separated exactly as before.
        """
        from utils.music_map import MUSIC

        if not self.bs_roformer or not music_map:
            return []

        sr = audio.sample_rate
        waveform = self._writable_waveform(audio)
        total = len(waveform)
        patches = []
        hi_res = 0
        jobs = [(start, end) for start, end, kind in music_map.spans
                if kind == MUSIC]
        pool_models = getattr(self.bs_roformer, "models", None)

        # A separator owns one work directory and writes a fixed in.wav into it,
        # so an instance may serve only one job at a time. Checking out the
        # instance is what guarantees that; picking it by job index does not,
        # because the pool's threads take whichever job is next rather than
        # every other one. The queue itself is shared across calls -- see
        # _checkout_queue -- so this also holds when a second file's own
        # strip_music_spans() call is running at the same time.
        available = self._checkout_queue(pool_models or [self.bs_roformer])

        def _finish_job(ordinal, i, j, vocals, used_hi_res):
            if vocals is None or len(vocals) != j - i:
                return ordinal, i, None, used_hi_res
            return ordinal, i, np.asarray(vocals, dtype=np.float32), used_hi_res

        def _raw_step(item, decoded_future=None):
            ordinal, (start, end) = item
            i, j = max(0, int(start * sr)), min(total, int(end * sr))
            if j - i < sr // 2:
                # Under half a second there is not enough for the separator to
                # work with, and the seams would cost more than the bed does.
                return ordinal, i, j, None, None, False, None
            reference = np.asarray(waveform[i:j], dtype=np.float32).copy()
            decoded = None
            if decoded_future is not None:
                # Wait for the decode before taking a separator, so an
                # instance is never held while only the CPU is working.
                try:
                    decoded = decoded_future.result()
                except Exception:
                    decoded = None
            model = available.get()

            if not hasattr(model, "separate_raw"):
                # Predates the raw/post split (a test double, or a custom
                # separator plugin) -- run the old, single-call interface
                # with the checkout held for the whole job, exactly as
                # before the split.
                try:
                    vocals = None
                    used_hi_res = False
                    if source_path is not None:
                        separate_span = getattr(model, "separate_span", None)
                        if separate_span is not None:
                            vocals = separate_span(source_path, start, end, sr, reference)
                            used_hi_res = vocals is not None
                    if vocals is None:
                        vocals = model.separate_segment(reference, sr)
                finally:
                    available.put(model)
                return ordinal, i, j, reference, ("legacy", vocals), used_hi_res, model

            # The raw/post split is available: release the instance right
            # after its GPU work finishes, instead of holding it through
            # CPU-only postprocessing too -- see _checkout_queue's own
            # docstring for why this matters across concurrent files
            # sharing this same queue.
            used_hi_res = False
            raw_result = None
            try:
                if source_path is not None:
                    separate_span_raw = getattr(model, "separate_span_raw", None)
                    if separate_span_raw is not None:
                        if decoded_future is None:
                            raw_result = separate_span_raw(source_path, start, end)
                        elif decoded is not None:
                            raw_result = separate_span_raw(
                                source_path, start, end, decoded=decoded)
                        # else: the decode already failed; go straight to the
                        # 16kHz fallback, as a failed in-place decode would.
                        used_hi_res = raw_result is not None
                if raw_result is None:
                    raw_result = model.separate_raw(reference, sr)
            finally:
                available.put(model)
            return ordinal, i, j, reference, ("raw", raw_result), used_hi_res, model

        def _post_step(step):
            ordinal, i, j, reference, payload, used_hi_res, model = step
            if reference is None:
                return _finish_job(ordinal, i, j, None, False)
            kind, data = payload
            if kind == "legacy":
                return _finish_job(ordinal, i, j, data, used_hi_res)
            if used_hi_res:
                vocals = model.separate_span_postprocess(data, reference, sr)
            else:
                vocals = model.postprocess_separated(data, reference, sr)
                if vocals is None:
                    # separate_segment()'s own contract: stay mixture rather
                    # than go silent, since silence would enter the dataset
                    # labelled as speech.
                    vocals = reference
            return _finish_job(ordinal, i, j, vocals, used_hi_res)

        indexed_jobs = list(enumerate(jobs))
        async_runtime = self._async_runtime()
        if async_runtime is not None and pool_models and len(indexed_jobs) > 1:
            # A background gpu/post split: a raw call's model instance is
            # released (see _raw_step's `finally`) before this thread ever
            # reaches postprocessing, and postprocessing itself now runs on
            # a different pool -- so the freed instance is available to
            # whichever pending raw job is next, not gated on this job's own
            # CPU work finishing first.
            gpu_executor, post_executor = async_runtime
            decode_runtime = (self._decode_runtime(pool_models)
                              if source_path is not None else None)
            decoder = None
            if decode_runtime is not None:
                decoder = next(m for m in pool_models
                               if callable(getattr(m, "decode_span", None)))

            def _post_release(step, slots):
                try:
                    return _post_step(step)
                finally:
                    if slots is not None:
                        slots.release()

            def _raw_then_post(item, decoded_future, slots):
                # Postprocess is queued from the GPU thread the moment its raw
                # call returns, so it never waits behind an earlier, slower job.
                try:
                    step = _raw_step(item, decoded_future)
                except BaseException:
                    if slots is not None:
                        slots.release()
                    raise
                return post_executor.submit(_post_release, step, slots)

            raw_futures = []
            for item in indexed_jobs:
                decoded_future = slots = None
                _ordinal, (start, end) = item
                i, j = max(0, int(start * sr)), min(total, int(end * sr))
                if decoder is not None and j - i >= sr // 2:
                    decode_executor, slots = decode_runtime
                    slots.acquire()
                    try:
                        decoded_future = decode_executor.submit(
                            decoder.decode_span, source_path, start, end)
                    except BaseException:
                        slots.release()
                        raise
                raw_futures.append(gpu_executor.submit(
                    _raw_then_post, item, decoded_future, slots))
            # Wait for every job before raising: a failed one must not leave
            # others running against a waveform the caller is about to discard.
            first_error = None
            separated = []
            for raw_future in raw_futures:
                try:
                    separated.append(raw_future.result().result())
                except BaseException as exc:
                    first_error = first_error or exc
            if first_error is not None:
                raise first_error
        elif pool_models and len(indexed_jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor as _TPE
            with _TPE(max_workers=min(len(pool_models), len(indexed_jobs))) as executor:
                separated = list(executor.map(
                    lambda item: _post_step(_raw_step(item)), indexed_jobs))
        else:
            separated = [_post_step(_raw_step(item)) for item in indexed_jobs]

        # Only this thread mutates the shared waveform, in source order.
        for _ordinal, start_sample, vocals, used_hi_res in sorted(separated):
            if vocals is None:
                continue
            waveform[start_sample:start_sample + len(vocals)] = vocals
            patches.append((start_sample, vocals))
            hi_res += int(used_hi_res)

        if logger and patches:
            seconds = sum(len(p) for _, p in patches) / sr
            logger.info(f"Stripped music from {seconds:.1f}s of the recording "
                        f"before diarization ({len(patches)} stretch(es), "
                        f"{hi_res} at 44.1kHz)")
        return patches

    def strip_full_recording(self, audio: AudioData, logger=None,
                             source_path: str = None):
        """Replace the whole waveform with its vocals, wherever music was or was not.

        The other way to use the separator: `strip_music_spans` touches only the
        stretches the map calls a bed and leaves everything else as recorded,
        while this runs the model over the entire recording. What it buys is that
        what remains is no longer a question of whether the tagger noticed a quiet
        bed, and the noise measured afterwards describes the audio the dataset
        is cut from. What it costs is that every second is now the model's
        output, so a failure has to leave the recording as it was -- a NaN
        written over the waveform would poison everything after it.

        Returns [(0, vocals)], the same shape `strip_music_spans` returns, so the
        result is cached and re-applied by the same code. An empty list means the
        recording is unchanged.

        Decoded from `source_path` at the checkpoint's own 44.1kHz when it can be
        (see `BSRoformerRemover.separate_span`); otherwise the 16kHz waveform held
        here is separated, which works but discards what that rate gives up.
        """
        if not self.bs_roformer:
            return []
        sr = audio.sample_rate
        waveform = self._writable_waveform(audio)
        if len(waveform) < sr:
            return []
        reference = waveform.copy()
        pool_models = getattr(self.bs_roformer, "models", None)
        available = self._checkout_queue(pool_models or [self.bs_roformer])

        model = available.get()
        try:
            vocals, hi_res = None, False
            separate_span = getattr(model, "separate_span", None) if source_path else None
            if separate_span is not None:
                vocals = separate_span(source_path, 0.0, len(reference) / float(sr),
                                       sr, reference)
                hi_res = vocals is not None
            if vocals is None:
                separate_full = getattr(model, "separate_full", None)
                vocals = separate_full(reference, sr) if separate_full else None
        finally:
            available.put(model)

        problem = self._unusable_vocals(vocals, reference)
        if problem:
            if logger:
                logger.error(f"Full-recording vocal separation {problem}; keeping the "
                             "recording as it was")
            return []
        vocals = np.asarray(vocals, dtype=np.float32)
        waveform[:] = vocals
        if logger:
            logger.info(f"Stripped music from the whole recording "
                        f"({len(vocals) / float(sr):.1f}s, "
                        f"{'44.1kHz' if hi_res else f'{sr}Hz'} separation)")
        return [(0, vocals)]

    @staticmethod
    def _unusable_vocals(vocals, reference):
        """Why a whole-recording separation cannot replace the recording, or None."""
        if vocals is None:
            return "produced nothing"
        vocals = np.asarray(vocals)
        if len(vocals) != len(reference):
            return f"came back {len(vocals)} samples long, expected {len(reference)}"
        if not np.isfinite(vocals).all():
            return "contains NaN or Inf (try autocast instead of native fp16 for this checkpoint)"
        rms = float(np.sqrt(np.mean(np.square(vocals, dtype=np.float64))))
        ref_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
        if ref_rms > 1e-6 and rms < 0.01 * ref_rms:
            return "is almost silent next to the recording"
        return None

    @staticmethod
    def apply_music_patches(audio: AudioData, patches):
        """Write cached vocal stretches back over the waveform."""
        if not patches:
            return
        waveform = MusicService._writable_waveform(audio)
        total = len(waveform)
        for start, chunk in patches:
            end = min(total, start + len(chunk))
            if end > start:
                waveform[start:end] = chunk[:end - start]
