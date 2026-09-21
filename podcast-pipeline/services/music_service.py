import queue
import threading

import numpy as np
from schemas.audio import AudioData

class MusicService:
    """Strips a music bed out of the stretches the map points at.

    It used to also walk the segment list after diarization and re-check each
    one; that pass is gone. The sweep runs at the front of the run now, so by
    the time segments exist the beds are already out of the waveform.
    """

    def __init__(self, bs_roformer_model=None, logger=None, model_loader=None):
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

        def separate_job(item):
            ordinal, (start, end) = item
            i, j = max(0, int(start * sr)), min(total, int(end * sr))
            if j - i < sr // 2:
                # Under half a second there is not enough for the separator to
                # work with, and the seams would cost more than the bed does.
                return ordinal, i, None, False
            reference = np.asarray(waveform[i:j], dtype=np.float32).copy()
            model = available.get()
            try:
                vocals = None
                if source_path is not None:
                    separate_span = getattr(model, "separate_span", None)
                    if separate_span is not None:
                        vocals = separate_span(source_path, start, end, sr, reference)
                        used_hi_res = vocals is not None
                    else:
                        used_hi_res = False
                else:
                    used_hi_res = False
                if vocals is None:
                    vocals = model.separate_segment(reference, sr)
            finally:
                available.put(model)
            if vocals is None or len(vocals) != j - i:
                return ordinal, i, None, used_hi_res
            return ordinal, i, np.asarray(vocals, dtype=np.float32), used_hi_res

        indexed_jobs = list(enumerate(jobs))
        if pool_models and len(indexed_jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(pool_models), len(indexed_jobs))) as executor:
                separated = list(executor.map(separate_job, indexed_jobs))
        else:
            separated = [separate_job(item) for item in indexed_jobs]

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
