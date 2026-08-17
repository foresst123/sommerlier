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
        
    def _run_whisper(self, audio_16k, dummy_vad, language="vi"):
        try:
            res = self.whisper.transcribe(audio_16k, dummy_vad, language=language)
            return res.get("text", ""), res.get("language", language), res.get("words", [])
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
            if not text and self.logger:
                self.logger.warning(f"Qwen3 returned empty for chunk {chunk_index}")
            if os.path.exists(path): os.remove(path)
            return text
        except Exception as e:
            if self.logger: self.logger.error(f"Qwen3 error: {e}")
            return ""

    def _run_whisper_batch(self, audios_16k: list, dummy_vads: list) -> list:
        results = []
        for a, v in zip(audios_16k, dummy_vads):
            results.append(self._run_whisper(a, v))
        return results

    def _run_phowhisper_batch(self, audios_16k: list) -> list:
        if not self.phowhisper:
            return [""] * len(audios_16k)
        try:
            # Reduced batch_size from 16 to 4 to prevent CUDA OOM on 15GB GPU 
            # since Qwen3 is also occupying ~10GB on the same GPU.
            return self.phowhisper.transcribe_batch(audios_16k, batch_size=4)
        except Exception as e:
            if self.logger: self.logger.error(f"PhoWhisper batch error: {e}")
            return [""] * len(audios_16k)

    def _run_qwen3_batch(self, audios_16k: list, chunk_indices: list, tmp_dir: str) -> list:
        results = []
        for a, idx in zip(audios_16k, chunk_indices):
            results.append(self._run_qwen3(a, idx, tmp_dir))
        return results

    def process(self, segments: List[EnhancedSegment], audio: AudioData, enable_word_timestamps: bool = False) -> List[TranscriptSegment]:
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="qwen3_asr_")
        results = []
        rover = RoverEnsembler()

        if self.logger:
            self.logger.info(f"ASR processing {len(segments)} segments using Batched Inference")
            self.logger.info("Note: ASR models require 16kHz audio. Resampling from base 24kHz to 16kHz internally.")

        valid_segments = []
        audios_16k = []
        dummy_vads = []
        chunk_indices = []

        sr = audio.sample_rate

        # 1. Prepare data
        for seg in segments:
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

            valid_segments.append(seg)
            audios_16k.append(audio_16k)
            dummy_vads.append([{"start": 0.0, "end": len(audio_16k) / 16000}])
            chunk_indices.append(seg.index)

        if not valid_segments:
            return []

        # 2. Run batched inference in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            fw = executor.submit(self._run_whisper_batch, audios_16k, dummy_vads)
            fp = executor.submit(self._run_phowhisper_batch, audios_16k)
            fq = executor.submit(self._run_qwen3_batch, audios_16k, chunk_indices, tmp_dir)

            whisper_results = fw.result()
            pho_results = fp.result()
            qwen_results = fq.result()

        # 3. Zip and vote
        for i, seg in enumerate(valid_segments):
            t_whisper, lang, words = whisper_results[i]
            t_pho = pho_results[i]
            t_qwen = qwen_results[i]

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

