# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT


"""
ASR and ensemble utilities for podcast pipeline.
Includes ROVER ensemble, repetition filtering, and multi-model ASR processing.
"""

import collections
import time
import tempfile
import concurrent.futures
from typing import List, Tuple, Dict, Any
from itertools import zip_longest
from difflib import SequenceMatcher
import numpy as np
import librosa
import soundfile as sf
from utils.logger import time_logger

# Logger will be initialized from main module
logger = None

def set_logger(log_instance):
    """Set logger instance from main module."""
    global logger
    logger = log_instance


class RoverEnsembler:
    """
    ROVER (Recognizer Output Voting Error Reduction) ensemble implementation.
    Combines outputs from multiple ASR models to produce more accurate transcriptions.

    [Updated] Uses SequenceMatcher-based alignment to perform correct majority voting even with token length differences.
    """

    @staticmethod
    def align_tokens_with_sequencematcher(base_tokens: List[str], candidate_tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Align two token sequences using SequenceMatcher.

        Args:
            base_tokens: Base token list (e.g., Whisper)
            candidate_tokens: Token list to align (e.g., Canary/Parakeet)

        Returns:
            List of aligned (base_token, candidate_token) tuples.
            Unmatched positions are represented as None.
        """
        matcher = SequenceMatcher(None, base_tokens, candidate_tokens)
        aligned_pairs = []

        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode == 'equal':
                # Fully matched section
                for base_tok, cand_tok in zip(base_tokens[i1:i2], candidate_tokens[j1:j2]):
                    aligned_pairs.append((base_tok, cand_tok))

            elif opcode == 'replace':
                # Replacement section - align to the longer side
                base_chunk = base_tokens[i1:i2]
                cand_chunk = candidate_tokens[j1:j2]

                max_len = max(len(base_chunk), len(cand_chunk))
                for idx in range(max_len):
                    base_tok = base_chunk[idx] if idx < len(base_chunk) else None
                    cand_tok = cand_chunk[idx] if idx < len(cand_chunk) else None
                    aligned_pairs.append((base_tok, cand_tok))

            elif opcode == 'delete':
                # Exists only in base (deleted from candidate)
                for base_tok in base_tokens[i1:i2]:
                    aligned_pairs.append((base_tok, None))

            elif opcode == 'insert':
                # Exists only in candidate (inserted into base)
                for cand_tok in candidate_tokens[j1:j2]:
                    aligned_pairs.append((None, cand_tok))

        return aligned_pairs

    @staticmethod
    def build_confusion_network(aligned_sequences: List[List[Tuple]]) -> List[List[str]]:
        """
        Build a confusion network from aligned sequences.

        Args:
            aligned_sequences: List of aligned (token, token, ...) tuples

        Returns:
            List of candidate tokens for each position
        """
        if not aligned_sequences:
            return []

        # Find the length of the longest sequence
        max_len = max(len(seq) for seq in aligned_sequences)

        # Collect candidates for each position
        confusion_network = []
        for pos in range(max_len):
            candidates = []
            for seq in aligned_sequences:
                if pos < len(seq):
                    token = seq[pos]
                    if token is not None and token != "":
                        candidates.append(token)
            confusion_network.append(candidates)

        return confusion_network

    @staticmethod
    def align_and_vote(transcripts: List[str]) -> str:
        """
        Align multiple transcription results and perform majority voting.

        [Updated] Resolves token length difference issues with SequenceMatcher-based alignment.

        Args:
            transcripts: List of transcription results from ASR models (e.g., [whisper, canary, parakeet])

        Returns:
            Final ensembled transcription result
        """
        if not transcripts:
            return ""

        # Remove empty strings
        transcripts = [t.strip() for t in transcripts if t and t.strip()]
        if not transcripts:
            return ""

        if len(transcripts) == 1:
            return transcripts[0]

        # Tokenize by word
        tokenized = [t.split() for t in transcripts]

        # Set the first transcription (Whisper) as the base
        base_tokens = tokenized[0]

        # -----------------------------------------------------------------------
        # Step 1: Align each candidate to base
        # -----------------------------------------------------------------------
        aligned_sequences = [base_tokens]  # Base는 그대로

        for i in range(1, len(tokenized)):
            candidate_tokens = tokenized[i]
            aligned_pairs = RoverEnsembler.align_tokens_with_sequencematcher(base_tokens, candidate_tokens)

            # Extract only candidate tokens from aligned_pairs (aligned to base positions)
            aligned_candidate = [pair[1] for pair in aligned_pairs]
            aligned_sequences.append(aligned_candidate)

        # -----------------------------------------------------------------------
        # Step 2: Build Confusion Network
        # -----------------------------------------------------------------------
        # Pad all sequences to the same length
        max_len = max(len(seq) for seq in aligned_sequences)
        padded_sequences = []
        for seq in aligned_sequences:
            padded = seq + [None] * (max_len - len(seq))
            padded_sequences.append(padded)

        # -----------------------------------------------------------------------
        # Step 3: Majority voting at each position
        # -----------------------------------------------------------------------
        final_output = []

        for pos in range(max_len):
            # Collect all candidates at the current position
            candidates = []
            for seq in padded_sequences:
                token = seq[pos]
                if token is not None and token != "":
                    candidates.append(token)

            if not candidates:
                continue

            # Vote
            votes = collections.Counter(candidates)
            best_word, count = votes.most_common(1)[0]

            # Accept if 2 or more agree, otherwise use base (Whisper)
            if count >= 2:
                final_output.append(best_word)
            else:
                # Prefer base
                base_word = padded_sequences[0][pos]
                if base_word is not None and base_word != "":
                    final_output.append(base_word)
                else:
                    final_output.append(best_word)

        result = " ".join(final_output)

        # Debug log (optional)
        if logger and len(transcripts) > 1:
            logger.debug(f"[ROVER] Input transcripts: {len(transcripts)}")
            logger.debug(f"[ROVER] Base length: {len(base_tokens)}, Aligned length: {max_len}")
            logger.debug(f"[ROVER] Final output length: {len(final_output)}")

        return result


class RepetitionFilter:
    """
    Filters low-quality transcriptions by detecting repeated n-grams.
    Paper: Removes samples if a 15-gram appears more than 5 times.
    """

    def __init__(self, use_mock_tokenizer=True):
        self.use_mock_tokenizer = use_mock_tokenizer

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace-based tokenization (SentencePiece would be used in practice)"""
        if self.use_mock_tokenizer:
            return text.split()
        else:
            # Use SentencePiece for actual implementation
            pass

    def filter(self, text: str) -> bool:
        """
        Filtering conditions:
        1. Remove empty text
        2. Remove if a 15-gram appears more than 5 times

        Returns:
            bool: True to keep, False to remove
        """
        # Check for empty text
        if not text or not text.strip():
            logger.debug(f"[RepetitionFilter] Empty text detected.")
            return False

        tokens = self.tokenize(text)

        # 15-gram repetition check
        N = 15
        THRESHOLD = 5

        if len(tokens) < N:
            return True  # Short text passes through

        # Generate n-grams
        ngrams = [tuple(tokens[i:i+N]) for i in range(len(tokens) - N + 1)]

        # Count frequencies
        counts = collections.Counter(ngrams)

        # Check for more than 5 occurrences
        for ngram, count in counts.items():
            if count > THRESHOLD:
                logger.debug(f"[RepetitionFilter] Repetition detected! Span '{' '.join(ngram[:3])}...' occurs {count} times.")
                return False

        return True


@time_logger
def asr(vad_segments, audio, asr_model):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments of the given audio.
    [Updated] Now processes segments iteratively exactly like asr_MoE to ensure 'enhanced_audio'
    is correctly utilized without relying on global buffer sandwiching.
    """
    if len(vad_segments) == 0:
        return []

    # Full audio (for fallback)
    full_waveform = audio["waveform"]
    global_sample_rate = audio["sample_rate"]

    final_results = []

    # Following the asr_MoE approach (individual processing), batch size is set to 1.
    # It can be adjusted depending on library support, but individual processing is prioritized for accuracy.
    batch_size = 1

    logger.info(f"ASR Processing: {len(vad_segments)} segments (Iterative Mode)")

    for idx, segment in enumerate(vad_segments):
        start_time = segment["start"]
        end_time = segment["end"]
        speaker = segment.get("speaker", "Unknown")

        # ---------------------------------------------------------------------
        # 1. Audio Selection Logic (Identical to asr_MoE)
        # ---------------------------------------------------------------------
        segment_audio = None
        is_enhanced = False

        if "enhanced_audio" in segment:
            # Use SepReformer-separated audio if available
            raw_audio = segment["enhanced_audio"]
            is_enhanced = True
        else:
            # Otherwise, slice the corresponding segment from the full audio
            start_frame = int(start_time * global_sample_rate)
            end_frame = int(end_time * global_sample_rate)
            raw_audio = full_waveform[start_frame:end_frame]
            is_enhanced = False

        # Resample to 16kHz (for Whisper input)
        if global_sample_rate != 16000:
            segment_audio_16k = librosa.resample(raw_audio, orig_sr=global_sample_rate, target_sr=16000)
        else:
            segment_audio_16k = raw_audio

        # Skip audio that is too short
        if len(segment_audio_16k) < 160:
            continue

        # ---------------------------------------------------------------------
        # 2. Prepare Dummy VAD & Transcribe
        # ---------------------------------------------------------------------
        # Since we're inputting an already-sliced audio segment, the relative time is 0 ~ duration.
        duration_sec = len(segment_audio_16k) / 16000
        dummy_vad = [{"start": 0.0, "end": duration_sec}]

        try:
            # Language detection (can be performed per segment if needed, or fixed to 'en')
            # Here we default to 'en' following the existing flow; use detect_language if detection is needed
            # language, prob = asr_model.detect_language(segment_audio_16k)
            language = "en"

            transcribe_result = asr_model.transcribe(
                segment_audio_16k,
                dummy_vad,
                batch_size=batch_size,
                language=language,
                print_progress=False,
            )

            # Process results
            if transcribe_result and "segments" in transcribe_result:
                for res_seg in transcribe_result["segments"]:
                    # 1. Only process if text is not empty
                    if res_seg["text"].strip():
                        # 2. Convert relative time (0~duration) to absolute time (start_time~)
                        res_seg["start"] += start_time
                        res_seg["end"] += start_time

                        # 3. Restore metadata
                        res_seg["speaker"] = speaker
                        res_seg["language"] = transcribe_result.get("language", language)
                        res_seg["sepreformer"] = segment.get("sepreformer", False)
                        res_seg["is_separated"] = is_enhanced

                        if is_enhanced:
                            res_seg["enhanced_audio"] = raw_audio

                        # 4. Adjust word timestamps if present
                        if "words" in res_seg:
                            for w in res_seg["words"]:
                                w["start"] += start_time
                                w["end"] += start_time

                        final_results.append(res_seg)

        except Exception as e:
            logger.error(f"ASR failed for segment {idx} ({start_time:.2f}-{end_time:.2f}): {e}")
            continue

    return final_results


@time_logger
def asr_MoE(vad_segments, audio, asr_model, asr_model_2, canary_model, segment_demucs_flags=None, enable_word_timestamps=False, device="cuda"):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments using MoE with Parallel Execution.
    [Updated] Runs Whisper, Parakeet, and Canary in parallel using ThreadPoolExecutor.
    """
    if len(vad_segments) == 0:
        return [], 0.0, 0.0

    if segment_demucs_flags is None:
        segment_demucs_flags = [False] * len(vad_segments)

    # Full audio (for fallback)
    full_waveform = audio["waveform"]
    global_sample_rate = audio["sample_rate"]

    final_results = []
    total_whisper_time = 0.0
    total_alignment_time = 0.0

    rover = RoverEnsembler()

    # --- Helper Functions for Parallel Execution ---
    def run_whisper_task(segment_audio_16k, dummy_vad):
        w_start = time.time()
        try:
            transcribe_result = asr_model.transcribe(
                segment_audio_16k,
                dummy_vad,
                batch_size=1,
                print_progress=False
            )

            text_whisper = ""
            detected_language = "en"
            words = []

            if transcribe_result and "segments" in transcribe_result and len(transcribe_result["segments"]) > 0:
                text_whisper = " ".join([s["text"] for s in transcribe_result["segments"]]).strip()
                detected_language = transcribe_result.get("language", "en")
                if enable_word_timestamps:
                    for s in transcribe_result["segments"]:
                        if "words" in s: words.extend(s["words"])

            w_end = time.time()
            return {
                "text": text_whisper,
                "language": detected_language,
                "words": words,
                "time": w_end - w_start
            }
        except Exception as e:
            logger.error(f"Whisper failed: {e}")
            return {"text": "", "language": "en", "words": [], "time": 0.0}

    def run_parakeet_task(segment_audio_16k):
        try:
            # Parakeet input requires list
            p_res = asr_model_2.transcribe([segment_audio_16k])

            text_parakeet = ""
            if p_res:
                first_result = p_res[0]
                if isinstance(first_result, str):
                    text_parakeet = first_result
                elif hasattr(first_result, 'text'):
                    text_parakeet = first_result.text
                else:
                    text_parakeet = str(first_result)
            return text_parakeet
        except Exception as e:
            logger.error(f"Parakeet failed: {e}")
            return ""

    def run_canary_task(segment_audio_16k):
        try:
            # Canary requires a file path usually, creating temp file safely inside thread
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                sf.write(temp_wav.name, segment_audio_16k, 16000)
                # Ensure write is flushed
                temp_wav.flush()

                answer_ids = canary_model.generate(
                    prompts=[[{"role": "user", "content": f"Transcribe the following: {canary_model.audio_locator_tag}", "audio": [temp_wav.name]}]],
                    max_new_tokens=128,
                )
                text_canary = canary_model.tokenizer.ids_to_text(answer_ids[0].cpu())
                return text_canary
        except Exception as e:
            logger.error(f"Canary failed: {e}")
            return ""
    # ---------------------------------------------

    # Create a ThreadPoolExecutor
    # max_workers=3 allows all three models to be attempted roughly at the same time.
    # Note: Python GIL exists, but since these calls release GIL for C++/CUDA ops, it works for parallelization.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

        for idx, segment in enumerate(vad_segments):
            start_time = segment["start"]
            end_time = segment["end"]
            speaker = segment.get("speaker", "Unknown")

            # 1. Audio Selection Logic
            segment_audio = None
            is_enhanced = False

            if "enhanced_audio" in segment:
                raw_audio = segment["enhanced_audio"]
                is_enhanced = True
            else:
                start_frame = int(start_time * global_sample_rate)
                end_frame = int(end_time * global_sample_rate)
                raw_audio = full_waveform[start_frame:end_frame]

            # Resample to 16kHz
            if global_sample_rate != 16000:
                segment_audio_16k = librosa.resample(raw_audio, orig_sr=global_sample_rate, target_sr=16000)
            else:
                segment_audio_16k = raw_audio

            if len(segment_audio_16k) < 160:
                continue

            # Dummy VAD for Whisper
            duration_sec = len(segment_audio_16k) / 16000
            dummy_vad = [{"start": 0.0, "end": duration_sec}]

            # ---------------------------------------------------------------------
            # Submit Tasks in Parallel
            # ---------------------------------------------------------------------
            future_whisper = executor.submit(run_whisper_task, segment_audio_16k, dummy_vad)
            future_parakeet = executor.submit(run_parakeet_task, segment_audio_16k)
            future_canary = executor.submit(run_canary_task, segment_audio_16k)

            # ---------------------------------------------------------------------
            # Wait for results (Barrier)
            # ---------------------------------------------------------------------
            # .result() blocks until the future is done
            whisper_res = future_whisper.result()
            text_parakeet = future_parakeet.result()
            text_canary = future_canary.result()

            # Unpack Whisper results
            text_whisper = whisper_res["text"]
            detected_language = whisper_res["language"]
            words = whisper_res["words"]
            total_whisper_time += whisper_res["time"]

            # ---------------------------------------------------------------------
            # 5. Ensemble & Result Construction
            # ---------------------------------------------------------------------
            text_ensemble = rover.align_and_vote([text_whisper, text_canary, text_parakeet])

            seg_result = {
                "start": start_time,
                "end": end_time,
                "text": text_ensemble,
                "text_whisper": text_whisper,
                "text_parakeet": text_parakeet,
                "text_canary": text_canary,
                "speaker": speaker,
                "language": detected_language,
                "demucs": segment_demucs_flags[idx] if idx < len(segment_demucs_flags) else False,
                "is_separated": is_enhanced,
                "sepreformer": segment.get("sepreformer", False)
            }

            if is_enhanced:
                seg_result["enhanced_audio"] = raw_audio

            if enable_word_timestamps and words:
                for w in words:
                    w["start"] += start_time
                    w["end"] += start_time
                seg_result["words"] = words

            final_results.append(seg_result)

    return final_results, total_whisper_time, total_alignment_time
