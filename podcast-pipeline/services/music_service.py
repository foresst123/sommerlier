import numpy as np
from typing import List
from schemas.audio import AudioData
from schemas.segment import SpeechSegment

class MusicService:
    """Detects and removes background music from segments."""
    
    def __init__(self, panns_model=None, bs_roformer_model=None, logger=None,
                 model_loader=None):
        self._panns = panns_model
        self._bs_roformer = bs_roformer_model
        self.model_loader = model_loader
        self.logger = logger
        self.full_vocals = None

    # Models are fetched from the loader on use, not captured at construction.
    # PipelineService loads each stage's models when that stage runs, so a
    # reference taken here would be None for every stage that had not loaded
    # yet -- and would stay None after it did.
    def _model(self, held, name):
        if held is not None:
            return held
        return self.model_loader.get(name) if self.model_loader else None

    @property
    def panns(self):
        return self._model(self._panns, "panns")

    @panns.setter
    def panns(self, model):
        self._panns = model

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
        
    def _prepare_full_vocals(self, audio: AudioData):
        if self.bs_roformer and self.full_vocals is None:
            if self.logger: self.logger.info("Running BS-RoFormer on full audio to extract vocals.")
            self.full_vocals = self.bs_roformer.separate_full(audio.waveform, audio.sample_rate)
            
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
        for start, end, kind in music_map.spans:
            if kind != MUSIC:
                continue
            i, j = max(0, int(start * sr)), min(total, int(end * sr))
            if j - i < sr // 2:
                # Under half a second there is not enough for the separator to
                # work with, and the seams would cost more than the bed does.
                continue
            reference = waveform[i:j]
            vocals = None
            if source_path is not None:
                separate_span = getattr(self.bs_roformer, "separate_span", None)
                if separate_span is not None:
                    vocals = separate_span(source_path, start, end, sr, reference)
                    if vocals is not None:
                        hi_res += 1
            if vocals is None:
                vocals = self.bs_roformer.separate_segment(reference, sr)
            if vocals is None or len(vocals) != j - i:
                continue
            waveform[i:j] = vocals
            patches.append((i, np.asarray(vocals, dtype=np.float32)))

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

    def process_segments(self, segments: List[SpeechSegment], audio: AudioData) -> List[SpeechSegment]:
        if not self.panns or not self.bs_roformer:
            return segments
            
        self._prepare_full_vocals(audio)
        sr = audio.sample_rate
        waveform = audio.waveform
        
        from tqdm import tqdm
        for seg in tqdm(segments, desc="[PANNs+BSRoformer]", leave=True):
            # Check if it was already processed by TSE, which also inherently isolates voice
            if seg.tse:
                # Separated audio is a resynthesis, so the detector's verdict on
                # it would say nothing about the original recording. Left unset
                # rather than guessed: `has_music` stays False meaning "not
                # checked", which is what the review page shows.
                seg.bs_roformer = False
                continue
                
            start_frame = int(seg.start * sr)
            end_frame = int(seg.end * sr)
            raw_audio = waveform[start_frame:end_frame]
            
            has_music, prob = self.panns.detect_music(raw_audio, sr)
            seg.has_music = bool(has_music)

            if has_music:
                if self.full_vocals is not None:
                    target_len = len(raw_audio)
                    start = min(start_frame, len(self.full_vocals))
                    end = min(end_frame, len(self.full_vocals))
                    vocal_slice = self.full_vocals[start:end]
                    
                    if len(vocal_slice) < target_len:
                        import numpy as np
                        pad_width = target_len - len(vocal_slice)
                        vocal_slice = np.pad(vocal_slice, (0, pad_width), mode='constant')
                        
                    seg.audio = vocal_slice
                else:
                    seg.audio = self.bs_roformer.separate_segment(raw_audio, sr)
                seg.bs_roformer = True
            else:
                seg.bs_roformer = False
                
        return segments
