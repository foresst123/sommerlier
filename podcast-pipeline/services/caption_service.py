from typing import List
from schemas.audio import AudioData
from schemas.transcript import TranscriptSegment

class CaptionService:
    """Adds audio captions using Qwen3-Omni."""
    
    def __init__(self, captioner, logger=None):
        self.captioner = captioner
        self.logger = logger
        
    def add_captions(self, segments: List[TranscriptSegment], audio: AudioData, segments_audio_data: dict) -> List[TranscriptSegment]:
        if not self.captioner:
            return segments
            
        if self.logger: self.logger.info(f"Adding Qwen3-Omni captions to {len(segments)} segments...")
        
        for seg in segments:
            # We need the exact audio slice.
            if seg.index in segments_audio_data:
                seg_audio = segments_audio_data[seg.index]
            else:
                start_frame = int(seg.start * audio.sample_rate)
                end_frame = int(seg.end * audio.sample_rate)
                seg_audio = audio.waveform[start_frame:end_frame]
                
            caption = self.captioner.caption(seg_audio, audio.sample_rate)
            seg.qwen3omni_caption = caption
            
        return segments
