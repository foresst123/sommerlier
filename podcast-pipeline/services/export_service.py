import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List
from schemas.audio import AudioData
from schemas.transcript import TranscriptSegment
from schemas.segment import SpeechSegment
from pydub import AudioSegment as PydubAudioSegment
from utils.cpu_plan import usable_cores

# Each MP3 is its own ffmpeg subprocess and each wav write releases the GIL, so
# threads (not processes) are enough; more than this stops paying off.
MAX_EXPORT_WORKERS = 8


class ExportService:
    """Exports final results to JSON, MP3 chunks, and SRT."""
    
    def __init__(self, logger=None, workers: int = None):
        self.logger = logger
        if workers is None:
            workers = min(MAX_EXPORT_WORKERS, max(1, usable_cores() - 1))
        self.workers = max(1, int(workers))
        
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
        
        def write_chunk(seg):
            file_path = os.path.join(separated_dir, f"{seg.index}_{seg.speaker}_separated.wav")
            try:
                sf.write(file_path, seg.audio, sample_rate, subtype='PCM_16')
            except Exception as e:
                if self.logger: self.logger.warning(f"Failed to export separated audio for {seg.index}: {e}")

        # Chunk writes run on threads while this thread mixes the stitched track.
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="export-wav") as pool:
            writes = []
            for seg in speech_segments:
                if seg.audio is not None:
                    writes.append(pool.submit(write_chunk, seg))

                    # Mix into full audio
                    start_sample = int(seg.start * sample_rate)
                    end_sample = start_sample + len(seg.audio)

                    if end_sample > full_length:
                        # Pad if necessary
                        pad_len = end_sample - full_length
                        full_audio = np.pad(full_audio, (0, pad_len))
                        full_length = end_sample

                    full_audio[start_sample:end_sample] += seg.audio
            for write in writes:
                write.result()

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
            
        def export_one(seg):
            file_path = os.path.join(segments_dir, f"{seg.index}_{seg.speaker}.mp3")
            # Cut from the original audio: the text-only TranscriptSegment carries
            # no separated audio.
            start_ms = int(seg.start * 1000)
            end_ms = int(seg.end * 1000)
            full_audio_segment[start_ms:end_ms].export(file_path, format="mp3")

        # One ffmpeg process per segment: encode several at a time. map() raises
        # the first failure, as the sequential loop did.
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="export-mp3") as pool:
            list(pool.map(export_one, segments))

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
