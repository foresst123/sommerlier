import concurrent.futures
import librosa
from typing import List
from schemas.audio import AudioData
from schemas.segment import EnhancedSegment
from schemas.transcript import TranscriptSegment
from algorithms.asr.rover import RoverEnsembler
from algorithms.asr.hallucination import filter_short_segment_outputs

class ASRService:
    """Coordinates MoE ASR models and ROVER ensemble."""
    
    def __init__(self, whisper, phowhisper, qwen3, logger=None, model_loader=None, qwen3_service=None,
                 language: str = "vi", batch_size: int = 4):
        self.whisper = whisper
        self.phowhisper = phowhisper
        self.qwen3 = qwen3
        self.logger = logger
        self.model_loader = model_loader
        self.qwen3_service = qwen3_service
        self.language = language
        self.batch_size = batch_size

        if not self.whisper and self.logger:
            self.logger.warning(
                "Whisper is not loaded (requires --ASRMoE with --lang vi); "
                "ROVER will vote without it."
            )
        if not self.qwen3 and self.logger:
            self.logger.warning("Qwen3-ASR is not loaded; ROVER will vote without it.")

    def active_models(self) -> List[str]:
        """Names of the ASR models actually available for this run."""
        names = []
        if self.whisper: names.append("whisper")
        if self.phowhisper: names.append("phowhisper")
        if self.qwen3: names.append("qwen3")
        return names

    def _run_whisper(self, audio_16k, dummy_vad, language=None):
        if not self.whisper:
            return "", None, []
        language = language or self.language
        try:
            res = self.whisper.transcribe(audio_16k, dummy_vad, language=language)
            return res.get("text", ""), res.get("language", language), res.get("words", [])
        except Exception as e:
            if self.logger: self.logger.error(f"Whisper error: {e}")
            return "", None, []

    def _run_phowhisper(self, audio_16k):
        if not self.phowhisper: return ""
        try:
            return self.phowhisper.transcribe(audio_16k)
        except Exception as e:
            if self.logger: self.logger.error(f"PhoWhisper error: {e}")
            return ""

    def _run_qwen3(self, audio_16k, chunk_index: str, tmp_dir: str):
        if not self.qwen3: return ""
        path = None
        try:
            import os
            import numpy as np
            # The audio is already float32 @ 16 kHz here; a .npy dump lets the
            # worker memory-map it instead of decoding a WAV per segment.
            path = os.path.join(tmp_dir, f"qwen3_{chunk_index}.npy")
            np.save(path, np.ascontiguousarray(audio_16k, dtype=np.float32))
            text = self.qwen3.transcribe(path)
            if not text and self.logger:
                self.logger.warning(f"Qwen3 returned empty for chunk {chunk_index}")
            return text
        except Exception as e:
            if self.logger: self.logger.error(f"Qwen3 error: {e}")
            return ""
        finally:
            if path:
                import os
                try:
                    if os.path.exists(path): os.remove(path)
                except OSError:
                    pass

    def _run_whisper_batch(self, audios_16k: list, dummy_vads: list, callback=None) -> list:
        if not self.whisper:
            if callback:
                for _ in audios_16k: callback()
            return [("", None, [])] * len(audios_16k)

        results = []
        for a, v in zip(audios_16k, dummy_vads):
            results.append(self._run_whisper(a, v))
            if callback: callback()
        if getattr(self, "model_loader", None):
            self.model_loader.unload("whisper")
        return results

    def _run_phowhisper_batch(self, audios_16k: list, callback=None) -> list:
        if not self.phowhisper:
            if callback:
                for _ in audios_16k: callback()
            return [""] * len(audios_16k)
        try:
            # Batch size is kept modest because Qwen3 shares the GPU.
            res = self.phowhisper.transcribe_batch(
                audios_16k, batch_size=self.batch_size, logger=self.logger, callback=callback
            )
            if getattr(self, "model_loader", None):
                self.model_loader.unload("phowhisper")
            return res
        except Exception as e:
            if self.logger: self.logger.error(f"PhoWhisper batch error: {e}")
            return [""] * len(audios_16k)

    def _run_qwen3_batch(self, audios_16k: list, chunk_indices: list, tmp_dir: str, callback=None) -> list:
        if not self.qwen3:
            if callback:
                for _ in audios_16k: callback()
            return [""] * len(audios_16k)

        results = []
        for a, idx in zip(audios_16k, chunk_indices):
            results.append(self._run_qwen3(a, idx, tmp_dir))
            if callback: callback()
        # The worker is owned by main.py's finally block; stopping it here would
        # leave a dead Popen behind that a second process() call would write to.
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
        import threading
        import sys
        import time
        
        progress = {"whisper": 0, "pho": 0, "qwen": 0}
        total = len(audios_16k)
        stop_event = threading.Event()
        
        def monitor_progress():
            while not stop_event.is_set():
                w, p, q = progress["whisper"], progress["pho"], progress["qwen"]
                sys.stdout.write(f"\r[ASR] Whisper: {w}/{total} | PhoWhisper: {p}/{total} | Qwen3: {q}/{total}")
                sys.stdout.flush()
                time.sleep(1.0)
            w, p, q = progress["whisper"], progress["pho"], progress["qwen"]
            sys.stdout.write(f"\r[ASR] Whisper: {w}/{total} | PhoWhisper: {p}/{total} | Qwen3: {q}/{total}\n")
            sys.stdout.flush()
            
        monitor_thread = threading.Thread(target=monitor_progress)
        monitor_thread.start()
        
        def cb_whisper(): progress["whisper"] += 1
        def cb_pho(): progress["pho"] += 1
        def cb_qwen(): progress["qwen"] += 1
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            fw = executor.submit(self._run_whisper_batch, audios_16k, dummy_vads, cb_whisper)
            fp = executor.submit(self._run_phowhisper_batch, audios_16k, cb_pho)
            fq = executor.submit(self._run_qwen3_batch, audios_16k, chunk_indices, tmp_dir, cb_qwen)

            whisper_results = fw.result()
            pho_results = fp.result()
            qwen_results = fq.result()
            
        stop_event.set()
        monitor_thread.join()

        # 3. Zip and vote
        from tqdm import tqdm
        hallucinations_dropped = 0
        for i, seg in enumerate(tqdm(valid_segments, desc="[ROVER] Bầu chọn", leave=True)):
            t_whisper, lang, words = whisper_results[i]
            t_pho = pho_results[i]
            t_qwen = qwen_results[i]
            duration = seg.end - seg.start

            # ROVER anchors its alignment on the first transcript, so a Whisper
            # hallucination on a backchannel does not merely get a vote -- it
            # becomes the skeleton every other model is aligned against. Drop
            # bad candidates before voting rather than after.
            candidates = [t_whisper, t_qwen, t_pho]
            cleaned = filter_short_segment_outputs(
                candidates, duration, logger=self.logger, segment_id=seg.index
            )
            hallucinations_dropped += sum(
                1 for before, after in zip(candidates, cleaned) if before != after
            )
            # The blanked values are also what gets exported per model: a
            # hallucination is not evidence of what Whisper heard, and leaving
            # it in text_whisper would poison anything training off these fields.
            t_whisper, t_qwen, t_pho = cleaned

            # ROVER aligns everything against its first entry, so the anchor has
            # to be a transcript we still trust. Whisper stays the anchor -- it
            # is the strongest model when it behaves -- unless this segment is
            # exactly where it misbehaved and got blanked above. Reordering
            # unconditionally on short clips measurably degraded segments where
            # Whisper was fine.
            if t_whisper:
                order = [t_whisper, t_qwen, t_pho]
            else:
                order = [t_pho, t_qwen, t_whisper]

            final_text = rover.align_and_vote(order)

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
                # Only Whisper detects language; without it, report the requested
                # language rather than inventing one.
                language=lang or self.language,
                demucs=seg.demucs,
                tse=seg.tse,
                words=words if enable_word_timestamps else None
            ))

        if self.logger:
            self.logger.info(f"ASR completed: {len(results)} transcripts produced")
            if hallucinations_dropped:
                self.logger.info(
                    f"Discarded {hallucinations_dropped} hallucinated ASR outputs before voting"
                )
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return results

