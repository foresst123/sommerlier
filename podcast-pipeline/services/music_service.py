from typing import List
from schemas.audio import AudioData
from schemas.segment import EnhancedSegment

class MusicService:
    """Detects and removes background music from segments."""
    
    def __init__(self, panns_model=None, demucs_model=None, logger=None,
                 model_loader=None):
        self._panns = panns_model
        self._demucs = demucs_model
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
    def demucs(self):
        return self._model(self._demucs, "demucs")

    @demucs.setter
    def demucs(self, model):
        self._demucs = model
        
    def _prepare_full_vocals(self, audio: AudioData):
        if self.demucs and self.full_vocals is None:
            if self.logger: self.logger.info("Running Demucs on full audio to extract vocals.")
            self.full_vocals = self.demucs.separate_full(audio.waveform, audio.sample_rate)
            
    def process_segments(self, segments: List[EnhancedSegment], audio: AudioData) -> List[EnhancedSegment]:
        if not self.panns or not self.demucs:
            return segments
            
        self._prepare_full_vocals(audio)
        sr = audio.sample_rate
        waveform = audio.waveform
        
        from tqdm import tqdm
        for seg in tqdm(segments, desc="[PANNs+Demucs]", leave=True):
            # Check if it was already processed by TSE, which also inherently isolates voice
            if seg.tse:
                # Separated audio is a resynthesis, so the detector's verdict on
                # it would say nothing about the original recording. Left unset
                # rather than guessed: `has_music` stays False meaning "not
                # checked", which is what the review page shows.
                seg.demucs = False
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
                        
                    seg.enhanced_audio = vocal_slice
                else:
                    seg.enhanced_audio = self.demucs.separate_segment(raw_audio, sr)
                seg.demucs = True
            else:
                seg.demucs = False
                
        return segments
