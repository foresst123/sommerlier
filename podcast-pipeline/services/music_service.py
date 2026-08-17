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
        
        for seg in segments:
            # Check if it was already processed by SR-CorrNet, which also inherently isolates voice
            if seg.srcorrnet:
                seg.demucs = False
                continue
                
            start_frame = int(seg.start * sr)
            end_frame = int(seg.end * sr)
            raw_audio = waveform[start_frame:end_frame]
            
            has_music, prob = self.panns.detect_music(raw_audio, sr)
            
            if has_music:
                if self.logger: self.logger.info(f"Music detected (prob {prob:.2f}) for {seg.index}. Removing...")
                if self.full_vocals is not None:
                    seg.enhanced_audio = self.full_vocals[start_frame:end_frame]
                else:
                    seg.enhanced_audio = self.demucs.separate_segment(raw_audio, sr)
                seg.demucs = True
            else:
                seg.demucs = False
                
        return segments
