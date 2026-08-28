from typing import List
from schemas.audio import AudioData
from schemas.segment import EnhancedSegment

class MusicService:
    """Detects and removes background music from segments."""
    
    def __init__(self, panns_model, demucs_model, logger=None):
        self.panns = panns_model
        self.demucs = demucs_model
        self.logger = logger
        self.full_vocals = None
        
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
