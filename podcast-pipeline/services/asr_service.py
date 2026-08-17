import concurrent.futures
import librosa
from typing import List
from schemas.audio import AudioData
from schemas.segment import EnhancedSegment
from schemas.transcript import TranscriptSegment
from algorithms.asr.rover import RoverEnsembler

class ASRService:
    """Coordinates MoE ASR models and ROVER ensemble."""
    
    def __init__(self, whisper, phowhisper, qwen3, logger=None):
        self.whisper = whisper
        self.phowhisper = phowhisper
        self.qwen3 = qwen3
        self.logger = logger
        
    def _run_whisper(self, audio_16k, dummy_vad):
        try:
            res = self.whisper.transcribe(audio_16k, dummy_vad)
            return res.get("text", ""), res.get("language", "en"), res.get("words", [])
        except Exception as e:
            if self.logger: self.logger.error(f"Whisper error: {e}")
            return "", "en", []
            
    def _run_phowhisper(self, audio_16k):
        if not self.phowhisper: return ""
        try:
            return self.phowhisper.transcribe(audio_16k)
        except Exception as e:
            if self.logger: self.logger.error(f"PhoWhisper error: {e}")
            return ""
            
    def _run_qwen3(self, audio_16k, chunk_index: str, tmp_dir: str):
        if not self.qwen3: return ""
        try:
            import os, soundfile as sf
            path = os.path.join(tmp_dir, f"qwen3_{chunk_index}.wav")
            sf.write(path, audio_16k, 16000)
            text = self.qwen3.transcribe(path)
            if os.path.exists(path): os.remove(path)
            return text
        except Exception as e:
            if self.logger: self.logger.error(f"Qwen3 error: {e}")
            return ""

    def process(self, segments: List[EnhancedSegment], audio: AudioData, enable_word_timestamps: bool = False) -> List[TranscriptSegment]:
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="qwen3_asr_")
        results = []
        rover = RoverEnsembler()

        if self.logger: self.logger.info(f"ASR processing {len(segments)} segments")

        for seg in segments:
            sr = audio.sample_rate

            if seg.enhanced_audio is not None:
                raw_audio = seg.enhanced_audio
            else:
                start_frame = int(seg.start * sr)
                end_frame = int(seg.end * sr)
                raw_audio = audio.waveform[start_frame:end_frame]

            if len(raw_audio) == 0:
                if self.logger: self.logger.warning(f"Segment {seg.index} has empty audio, skipping")
                continue

            if sr != 16000:
                audio_16k = librosa.resample(raw_audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = raw_audio

            if len(audio_16k) < 160:
                if self.logger: self.logger.warning(f"Segment {seg.index} too short ({len(audio_16k)} samples), skipping")
                continue

            dummy_vad = [{"start": 0.0, "end": len(audio_16k) / 16000}]

            # Run 3 ASR models in parallel using a *separate* short-lived executor per segment
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as seg_executor:
                fw = seg_executor.submit(self._run_whisper, audio_16k, dummy_vad)
                fp = seg_executor.submit(self._run_phowhisper, audio_16k)
                fq = seg_executor.submit(self._run_qwen3, audio_16k, seg.index, tmp_dir)

                t_whisper, lang, words = fw.result()
                t_pho = fp.result()
                t_qwen = fq.result()

            final_text = rover.align_and_vote([t_whisper, t_qwen, t_pho])

            if enable_word_timestamps and words:
                for w in words:
                    w["start"] += seg.start
                    w["end"] += seg.start

            results.append(TranscriptSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker,
                text=final_text,
                text_whisper=t_whisper,
                text_phowhisper=t_pho,
                text_qwen3=t_qwen,
                language=lang,
                demucs=seg.demucs,
                srcorrnet=seg.srcorrnet,
                words=words if enable_word_timestamps else None
            ))

        if self.logger: self.logger.info(f"ASR completed: {len(results)} transcripts produced")
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return results

