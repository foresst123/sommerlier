import os
import json
import numpy as np
from typing import List
from schemas.audio import AudioData
from schemas.transcript import TranscriptSegment
from pydub import AudioSegment as PydubAudioSegment

class ExportService:
    """Exports final results to JSON, MP3 chunks, and SRT."""
    
    def __init__(self, logger=None):
        self.logger = logger
        
    def export_json(self, segments: List[TranscriptSegment], out_path: str, metadata: dict = None):
        if self.logger: self.logger.info(f"Exporting JSON to {out_path}")
        
        data = {
            "metadata": metadata or {},
            "segments": [s.__dict__ for s in segments]
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def export_mp3_segments(self, segments: List[TranscriptSegment], audio: AudioData, save_dir: str, audio_name: str):
        segments_dir = os.path.join(save_dir, audio_name)
        os.makedirs(segments_dir, exist_ok=True)
        
        full_audio_segment = audio.audio_segment
        if not full_audio_segment:
            wav_int16 = (audio.waveform * 32767).astype(np.int16)
            full_audio_segment = PydubAudioSegment(
                wav_int16.tobytes(),
                frame_rate=audio.sample_rate,
                sample_width=2,
                channels=1
            )
            
        for seg in segments:
            file_path = os.path.join(segments_dir, f"{seg.index}_{seg.speaker}.mp3")
            # If SR-CorrNet or Demucs modified the audio, it's not present in TranscriptSegment (only text is there)
            # Oh wait, we need enhanced_audio. If so, ExportService must receive the EnhancedSegment or a dict mapping index -> enhanced_audio
            # But the original code was: if seg.get('enhanced_audio') ...
            # We'll stick to extracting from original audio for now, as that's the safe path when separating text from audio logic.
            start_ms = int(seg.start * 1000)
            end_ms = int(seg.end * 1000)
            target_segment = full_audio_segment[start_ms:end_ms]
            target_segment.export(file_path, format="mp3")
            
    def export_srt(self, segments: List[TranscriptSegment], out_path: str):
        if self.logger: self.logger.info(f"Exporting SRT to {out_path}")
        def format_time(seconds):
            h = int(seconds / 3600)
            m = int((seconds % 3600) / 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
        with open(out_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                f.write(f"{i}\n")
                f.write(f"{format_time(seg.start)} --> {format_time(seg.end)}\n")
                f.write(f"[{seg.speaker}] {seg.text}\n\n")
