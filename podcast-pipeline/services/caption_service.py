from typing import List
from schemas.audio import AudioData
from schemas.transcript import TranscriptSegment

class CaptionService:
    """Adds audio captions using Qwen3-Omni."""
    
    def __init__(self, captioner=None, logger=None, model_loader=None):
        self._captioner = captioner
        self.model_loader = model_loader
        self.logger = logger

    # Models are fetched from the loader on use, not captured at construction.
    # PipelineService loads each stage's models when that stage runs, so a
    # reference taken here would be None for every stage that had not loaded
    # yet -- and would stay None after it did.
    @property
    def captioner(self):
        if self._captioner is not None:
            return self._captioner

    @captioner.setter
    def captioner(self, model):
        self._captioner = model
        return self.model_loader.get("captioner") if self.model_loader else None
        
    def add_captions(self, segments: List[TranscriptSegment], audio: AudioData, segments_audio_data: dict) -> List[TranscriptSegment]:
        if not self.captioner:
            return segments
            
        if self.logger: self.logger.info(f"Adding Qwen3-Omni captions to {len(segments)} segments...")
        
        from tqdm import tqdm
        for seg in tqdm(segments, desc="[Qwen3-Omni] Captioning", leave=True):
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
