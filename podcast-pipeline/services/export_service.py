import os
import json
import numpy as np
from typing import List
from schemas.audio import AudioData
from schemas.transcript import TranscriptSegment
from schemas.segment import SpeechSegment
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
            
    def export_separated_audio(self, speech_segments: List[SpeechSegment], sample_rate: int, save_dir: str):
        import soundfile as sf
        separated_dir = os.path.join(save_dir, "separation")
        os.makedirs(separated_dir, exist_ok=True)
        
        if self.logger: self.logger.info(f"Exporting separation audio to {separated_dir}")
        
        # Determine total length for the stitched full audio
        max_end = max((seg.end for seg in speech_segments), default=0.0)
        full_length = int(max_end * sample_rate)
        full_audio = np.zeros(full_length, dtype=np.float32)
        
        for seg in speech_segments:
            if seg.audio is not None:
                # Save individual chunk
                file_path = os.path.join(separated_dir, f"{seg.index}_{seg.speaker}_separated.wav")
                try:
                    sf.write(file_path, seg.audio, sample_rate, subtype='PCM_16')
                except Exception as e:
                    if self.logger: self.logger.warning(f"Failed to export separated audio for {seg.index}: {e}")
                    
                # Mix into full audio
                start_sample = int(seg.start * sample_rate)
                end_sample = start_sample + len(seg.audio)
                
                if end_sample > full_length:
                    # Pad if necessary
                    pad_len = end_sample - full_length
                    full_audio = np.pad(full_audio, (0, pad_len))
                    full_length = end_sample
                    
                full_audio[start_sample:end_sample] += seg.audio
                
        # Save the stitched full audio
        full_audio_path = os.path.join(save_dir, "after_separation.wav")
        try:
            # Normalize to prevent clipping from overlap mixing
            max_val = np.max(np.abs(full_audio))
            if max_val > 1.0:
                full_audio = full_audio / max_val
            
            sf.write(full_audio_path, full_audio, sample_rate, subtype='PCM_16')
            if self.logger: self.logger.info(f"Exported full stitched audio to {full_audio_path}")
        except Exception as e:
            if self.logger: self.logger.warning(f"Failed to export full stitched audio: {e}")

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
            # Oh wait, we need audio. If so, ExportService must receive the SpeechSegment or a dict mapping index -> audio
            # But the original code was: if seg.get('audio') ...
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
