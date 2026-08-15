# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT
import torch

# Fix for PyTorch 2.6+ weights_only=True default breaking pyannote model loading
# Patch lightning_fabric's _load function to use weights_only=False
import lightning_fabric.utilities.cloud_io as cloud_io
from pathlib import Path
from typing import Union, IO, Any

_original_load = cloud_io._load

def _patched_load(path_or_url: Union[IO, str, Path], map_location=None) -> Any:
    """Patched version of lightning_fabric's _load that uses weights_only=False for pyannote compatibility"""
    if not isinstance(path_or_url, (str, Path)):
        return torch.load(path_or_url, map_location=map_location, weights_only=False)

    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(str(path_or_url), map_location=map_location)

    from lightning_fabric.utilities.cloud_io import get_filesystem
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location, weights_only=False)

cloud_io._load = _patched_load

# Continue with other imports
import argparse
import json
import librosa
import numpy as np
import ast
import sys
import os
import shutil
import tqdm
import re
import warnings
import tempfile
from openai import OpenAI

import requests
from pydub import AudioSegment
import os
from tritony import InferenceClient
import numpy as np
import librosa
from pyannote.audio import Pipeline, Inference
import pandas as pd
#from prompt import DIAR_PROMPT, WEAK_DIAR_PROMPT, NEW_DIAR_PROMPT, SPK_SUMMERIZE_PROMPT, NEW_DIAR_PROMPT_with_spk_inform, DIAR_PROMPT_KO
from utils.tool import (
    export_to_mp3,
    export_to_mp3_new,
    load_cfg,
    get_audio_files,
    detect_gpu,
    check_env,
    calculate_audio_stats,
)
from utils.logger import Logger, time_logger
from models import separate_fast, dnsmos, whisper_asr, silero_vad
import time
import datetime
from panns_inference import AudioTagging
import soundfile as sf

from nemo.collections.asr.models import SortformerEncLabelModel
try:
    from nemo.collections.speechlm2.models import SALM
except ImportError:
    SALM = None

import json
import re
import argparse
from g2pk import G2p
import collections
import difflib
from typing import List, Tuple, Dict
from itertools import zip_longest

warnings.filterwarnings("ignore")


def _apply_sortformer_segment_padding_from_args(
    df: pd.DataFrame, args, logger, audio_duration: float | None = None
) -> pd.DataFrame:
    """
    Shift diarization segment boundaries (frame-level tweak) outside NeMo's internal post-processing.
    When --sortformer-param is set, this ensures observable timing changes even if model cfg overrides are ignored.
    """
    if df is None or df.empty:
        return df
    if not getattr(args, "sortformer_param", False):
        return df

    pad_onset = float(getattr(args, "sortformer_pad_onset", 0.0))
    pad_offset = float(getattr(args, "sortformer_pad_offset", 0.0))

    if pad_onset == 0.0 and pad_offset == 0.0:
        return df

    df = df.copy()
    df["start"] = (df["start"].astype(float) + pad_onset).clip(lower=0.0)
    df["end"] = df["end"].astype(float) + pad_offset
    if audio_duration is not None and audio_duration > 0:
        df["end"] = df["end"].clip(lower=0.0, upper=float(audio_duration))
    else:
        df["end"] = df["end"].clip(lower=0.0)
    df["end"] = df[["start", "end"]].max(axis=1)

    return df
audio_count = 0
# Limit diarization chunks to under 3 minutes, preferring VAD-detected silences as cut points.
MAX_DIA_CHUNK_DURATION = 2 * 60  # 3 minutes
MIN_SPLIT_SILENCE = 0.3  # seconds of silence required for splitting (more sensitive)
MIN_EMBED_DURATION = 0.5  # seconds; skip embedding if audio is shorter
QWEN_3_OMNI_PORT = "11500"
class RoverEnsembler:
    """
    ROVER (Recognizer Output Voting Error Reduction) ensemble implementation.
    Combines outputs from multiple ASR models to produce more accurate transcriptions.
    """

    @staticmethod
    def build_confusion_network(all_tokens: List[List[str]]) -> List[List[str]]:
        """
        Build a Confusion Network from multiple token sequences.
        Considers all sequences simultaneously to produce a unified alignment.

        Args:
            all_tokens: Token lists from all transcripts [[tok1, tok2, ...], ...]

        Returns:
            List of candidate tokens per position [[cand1, cand2, ...], [cand1, cand2, ...], ...]
        """
        if not all_tokens:
            return []

        if len(all_tokens) == 1:
            return [[tok] for tok in all_tokens[0]]

        # Select the longest sequence as pivot (usually the most accurate)
        pivot_idx = max(range(len(all_tokens)), key=lambda i: len(all_tokens[i]))
        pivot = all_tokens[pivot_idx]
        
        import string
        def _norm(t_list):
            return [t.lower().strip(string.punctuation) for t in t_list]
            
        pivot_norm = _norm(pivot)

        # Initialize confusion network: start each position with the pivot token
        confusion_net = [[pivot[i]] for i in range(len(pivot))]

        # Align all other sequences to the pivot and add to the confusion network
        for idx, tokens in enumerate(all_tokens):
            if idx == pivot_idx:
                continue
                
            tokens_norm = _norm(tokens)
            matcher = difflib.SequenceMatcher(None, pivot_norm, tokens_norm)

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    # Match: add token at corresponding position
                    for i, j in zip(range(i1, i2), range(j1, j2)):
                        if i < len(confusion_net):
                            confusion_net[i].append(tokens[j])

                elif tag == 'replace':
                    # Substitution: add corresponding candidate tokens at each pivot position
                    pivot_len = i2 - i1
                    cand_len = j2 - j1

                    if pivot_len == cand_len:
                        # 1:1 mapping
                        for i, j in zip(range(i1, i2), range(j1, j2)):
                            if i < len(confusion_net):
                                confusion_net[i].append(tokens[j])
                    elif pivot_len > cand_len:
                        # Pivot is longer: distribute candidates across positions
                        for i in range(i1, i2):
                            offset = (i - i1) * cand_len // pivot_len
                            if j1 + offset < j2 and i < len(confusion_net):
                                confusion_net[i].append(tokens[j1 + offset])
                    else:
                        # Candidate is longer: merge multiple candidates at pivot position
                        # Add all candidates at the first pivot position
                        if i1 < len(confusion_net):
                            for j in range(j1, j2):
                                confusion_net[i1].append(tokens[j])

                elif tag == 'delete':
                    # Exists only in pivot: already in confusion_net (other sequences have empty values)
                    pass

                elif tag == 'insert':
                    # Exists only in candidate: insert at the nearest pivot position
                    # Add after the previous matching position
                    insert_pos = min(i1, len(confusion_net) - 1) if confusion_net else 0
                    if insert_pos >= 0 and insert_pos < len(confusion_net):
                        for j in range(j1, j2):
                            confusion_net[insert_pos].append(tokens[j])

        return confusion_net

    @staticmethod
    def has_local_repetition(output: List[str], word: str, window: int = 3) -> bool:
        """
        Check if the same word is repeated within the recent window.

        Args:
            output: List of words output so far
            word: Word to check
            window: Window size to check

        Returns:
            True if repetition detected, False otherwise
        """
        if len(output) < 1:
            return False

        # Check the most recent 'window' words
        recent = output[-window:] if len(output) >= window else output

        # Consider it a repetition if the same word has appeared 2 or more times
        return recent.count(word) >= 2

    @staticmethod
    def calculate_transcript_similarity(t1_tokens: List[str], t2_tokens: List[str]) -> float:
        """
        Calculate similarity between two transcripts (based on Jaccard similarity).

        Args:
            t1_tokens: First transcript tokens
            t2_tokens: Second transcript tokens

        Returns:
            Similarity score between 0 and 1
        """
        if not t1_tokens or not t2_tokens:
            return 0.0

        set1 = set(t1_tokens)
        set2 = set(t2_tokens)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def align_and_vote(transcripts: List[str]) -> str:
        """
        Align multiple transcription results using Confusion Network and perform improved voting.
        - Unified alignment via Confusion Network
        - Repetition pattern detection and filtering
        - Similarity-based outlier downweighting

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
        all_tokens = [t.split() for t in transcripts]
        
        # Normalize tokens (lowercase, remove punctuation) for fair similarity calculation
        import string
        norm_tokens = [[tok.lower().strip(string.punctuation) for tok in tokens] for tokens in all_tokens]

        # Calculate similarity between transcripts to detect outliers
        similarities = []
        for i in range(len(norm_tokens)):
            sim_scores = []
            for j in range(len(norm_tokens)):
                if i != j:
                    sim = RoverEnsembler.calculate_transcript_similarity(norm_tokens[i], norm_tokens[j])
                    sim_scores.append(sim)
            avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
            similarities.append(avg_sim)

        # Reduce trust for transcripts with too low average similarity
        # Threshold: consider as outlier if average similarity is below 0.3
        outlier_threshold = 0.3
        trusted_indices = [i for i, sim in enumerate(similarities) if sim >= outlier_threshold]

        # If all transcripts are outliers, use only the ones with highest similarity
        if len(trusted_indices) == 0:
            trusted_indices = [i for i, _ in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:2]]

        # Build Confusion Network
        confusion_net = RoverEnsembler.build_confusion_network(all_tokens)

        # Improved voting for each position
        final_output = []
        for pos_idx, candidates in enumerate(confusion_net):
            if not candidates:
                continue

            # Remove empty strings then vote
            valid_candidates = [c for c in candidates if c]
            if not valid_candidates:
                continue

            # Filter to only candidates from trusted transcripts
            # (first in confusion_net is pivot, rest are in order)
            trusted_candidates = []
            for i, cand in enumerate(valid_candidates):
                # Estimate which transcript the i-th candidate came from
                # (simply: pivot is always trusted, rest are in order)
                if i == 0 or (i - 1) in trusted_indices:
                    trusted_candidates.append(cand)

            # If no trusted candidates, use original candidates
            if not trusted_candidates:
                trusted_candidates = valid_candidates

            # Select the word with the most votes using normalized words
            norm_map = collections.defaultdict(list)
            for c in trusted_candidates:
                n_w = c.lower().strip(string.punctuation)
                if n_w:
                    norm_map[n_w].append(c)

            if not norm_map:
                continue

            votes = {n_w: len(orig_ws) for n_w, orig_ws in norm_map.items()}
            # Find the normalized word with the max votes
            best_norm_word = max(votes, key=votes.get)
            count = votes[best_norm_word]

            # Reconstruct the original word: prefer pivot's format if the pivot matches the winning word
            pivot_word = candidates[0] if candidates[0] else ""
            pivot_norm = pivot_word.lower().strip(string.punctuation)
            
            if pivot_norm == best_norm_word:
                best_word = pivot_word
            else:
                best_word = norm_map[best_norm_word][0]

            # Repetition pattern check: skip if the same word was recently repeated
            if RoverEnsembler.has_local_repetition([w.lower().strip(string.punctuation) for w in final_output], best_norm_word):
                # If repetition, select the next most frequent candidate if available
                sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_votes) > 1:
                    best_norm_word = sorted_votes[1][0]
                    best_word = norm_map[best_norm_word][0]
                    count = sorted_votes[1][1]
                else:
                    # No other candidates available, skip
                    continue

            # Accept if majority vote, otherwise prioritize pivot (already handled by preferring pivot format above)
            final_output.append(best_word)

        return " ".join(final_output)


class RepetitionFilter:
    """
    Filters low-quality transcriptions by detecting repeated n-grams.
    Paper criterion: remove a sample if a 15-gram appears more than 5 times.
    """

    def __init__(self, use_mock_tokenizer=True):
        self.use_mock_tokenizer = use_mock_tokenizer

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace-based tokenization (use SentencePiece in production)"""
        if self.use_mock_tokenizer:
            return text.split()
        else:
            # Use SentencePiece for actual implementation
            pass

    def filter(self, text: str) -> bool:
        """
        Filtering criteria:
        1. Remove empty text
        2. Remove if a 15-gram appears more than 5 times

        Returns:
            bool: True to keep, False to remove
        """
        # Empty text check
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

        # Calculate frequency counts
        counts = collections.Counter(ngrams)

        # Check for more than 5 occurrences
        for ngram, count in counts.items():
            if count > THRESHOLD:
                logger.debug(f"[RepetitionFilter] Repetition detected! Span '{' '.join(ngram[:3])}...' occurs {count} times.")
                return False

        return True

import subprocess
import tempfile
import hashlib
from pathlib import Path
import subprocess
import os

# Replace the corresponding function in main_original_ASR_MoE.py with the code below.

def convert_opus_to_wav_cached(audio_path: str, target_sr: int, cache_dir: str, logger, ffmpeg_threads: int = 1):
    """
    Convert .opus/.ogg to wav and cache in cache_dir.
    Reuse cached wav if it exists and is newer than the input.
    [Fixed] Sanitize filenames to prevent errors from special characters.
    """
    lower = audio_path.lower()
    if not (lower.endswith(".opus") or lower.endswith(".ogg")):
        return audio_path

    os.makedirs(cache_dir, exist_ok=True)

    p = Path(audio_path)
    
    # Collision prevention: hash based on (path + mtime + size)
    key = f"{str(p.resolve())}|{p.stat().st_mtime}|{p.stat().st_size}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    
    # [Important fix] Replace all characters except alphanumeric, -, _, . with _ in the original filename (p.stem)
    # e.g.: "![CDATA[Title]]" -> "__CDATA_Title__"
    safe_stem = re.sub(r'[^\w\-\.]', '_', p.stem)
    
    # Generate safe filename
    out_wav = os.path.join(cache_dir, f"{safe_stem}.{h}.wav")

    # Return on cache hit
    if os.path.exists(out_wav):
        return out_wav

    cmd = [
        "ffmpeg", "-y",
        "-threads", str(int(ffmpeg_threads)),   
        "-i", audio_path,
        "-ac", "1",
        "-ar", str(int(target_sr)),
        "-sample_fmt", "s16",
        out_wav,
    ]

    # Log source -> destination conversion paths
    logger.info(f"[OPUS][CACHE] Converting...\n  Src: {audio_path}\n  Dst: {out_wav}")
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or (not os.path.exists(out_wav)):
        logger.error(f"[OPUS][CACHE] ffmpeg failed.\nSTDERR:\n{proc.stderr}")
        raise RuntimeError(f"ffmpeg opus->wav conversion failed for {audio_path}")

    return out_wav

def convert_opus_to_wav_if_needed(audio_path: str, target_sr: int, logger):
    """
    If input is .opus (or .ogg), convert to a temporary wav file and return the new path.
    Returns: (processing_path, temp_dir_or_None)
    """
    lower = audio_path.lower()
    if not (lower.endswith(".opus") or lower.endswith(".ogg")):
        return audio_path, None

    temp_dir = tempfile.mkdtemp(prefix="opus2wav_")
    out_wav = os.path.join(temp_dir, "converted.wav")

    # Prefer ffmpeg for robust opus decode
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ac", "1",
        "-ar", str(int(target_sr)),
        "-sample_fmt", "s16",
        out_wav,
    ]

    try:
        logger.info(f"[OPUS] Converting to wav via ffmpeg: {audio_path} -> {out_wav}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or (not os.path.exists(out_wav)):
            logger.error(f"[OPUS] ffmpeg conversion failed.\nSTDERR:\n{proc.stderr}")
            raise RuntimeError("ffmpeg opus->wav conversion failed")
        return out_wav, temp_dir
    except Exception as e:
        # As a fallback, try pydub (still requires ffmpeg underneath in most envs)
        logger.warning(f"[OPUS] ffmpeg path failed, trying pydub fallback: {e}")
        try:
            seg = AudioSegment.from_file(audio_path)
            seg = seg.set_channels(1).set_frame_rate(int(target_sr)).set_sample_width(2)
            seg.export(out_wav, format="wav")
            if not os.path.exists(out_wav):
                raise RuntimeError("pydub export failed")
            return out_wav, temp_dir
        except Exception as e2:
            logger.error(f"[OPUS] pydub fallback also failed: {e2}")
            # cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

@time_logger
def standardization(audio):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.

    Returns:
        dict: A dictionary containing the preprocessed audio waveform, audio file name, and sample rate, formatted as:
              {
                  "waveform": np.ndarray, the preprocessed audio waveform, dtype is np.float32, shape is (num_samples,)
                  "name": str, the audio file name
                  "sample_rate": int, the audio sample rate
              }

    Raises:
        ValueError: If the audio parameter is neither a str nor an AudioSegment.
    """
    global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    logger.debug("Entering the preprocessing of audio")

    # Convert the audio file to WAV format
    audio = audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
    audio = audio.set_sample_width(2)  # Set bit depth to 16bit
    audio = audio.set_channels(1)  # Set to mono

    logger.debug("Audio file converted to WAV format")

    # Calculate the gain to be applied
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)

    # Ensure waveform is 1D (mono)
    if waveform.ndim > 1:
        logger.warning(f"Waveform has {waveform.ndim} dimensions with shape {waveform.shape}, converting to mono")
        waveform = waveform.flatten()

    max_amplitude = np.max(np.abs(waveform))
    if max_amplitude > 0:
        waveform /= max_amplitude  # Normalize
    else:
        logger.warning("Audio has zero amplitude, skipping normalization")

    logger.debug(f"waveform shape: {waveform.shape}")
    logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
        "audio_segment": normalized_audio,
    }


# Step 2: Speaker Diarization
@time_logger
def detect_background_music(audio, panns_model, threshold=0.3):
    """
    Detect background music using PANNs.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        panns_model (AudioTagging): Loaded PANNs model instance.
        threshold (float): Music probability threshold. Background music is detected if above this value.

    Returns:
        tuple: (has_music: bool, music_prob: float)
    """
    if panns_model is None:
        logger.warning("PANNs model is not loaded, skipping music detection")
        return False, 0.0

    logger.debug("Detecting background music using PANNs")

    # PANNs expects 32kHz audio, so resample
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if sample_rate != 32000:
        waveform_32k = librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000)
    else:
        waveform_32k = waveform

    # PANNs inference (model is already loaded)
    (clipwise_output, embedding) = panns_model.inference(waveform_32k[None, :])

    # Get labels
    labels = panns_model.labels

    # Find Music probability
    music_idx = labels.index('Music') if 'Music' in labels else None
    if music_idx is not None:
        music_prob = float(clipwise_output[0, music_idx])
        logger.info(f"Music probability: {music_prob:.3f}")
        has_music = music_prob > threshold
        return has_music, music_prob
    else:
        logger.warning("Music label not found in PANNs output")
        return False, 0.0


def detect_segment_background_music(segment_audio, sample_rate, panns_model, threshold=0.3):
    """
    Detect background music in a segment audio.

    Args:
        segment_audio (np.ndarray): Segment audio waveform.
        sample_rate (int): Sample rate.
        panns_model (AudioTagging): Loaded PANNs model instance.
        threshold (float): Music probability threshold.

    Returns:
        tuple: (has_music: bool, music_prob: float)
    """
    if panns_model is None:
        logger.warning("PANNs model is not loaded, skipping music detection")
        return False, 0.0

    # PANNs expects 32kHz audio, so resample
    if sample_rate != 32000:
        waveform_32k = librosa.resample(segment_audio, orig_sr=sample_rate, target_sr=32000)
    else:
        waveform_32k = segment_audio

    # Check minimum length required by PANNs model (about 1 second = 32000 samples)
    # Minimum length is needed to pass through Cnn14 model's pooling layers
    min_length = 32000  # 1 second at 32kHz
    if len(waveform_32k) < min_length:
        logger.warning(f"Segment too short for music detection ({len(waveform_32k)/32000:.2f}s < 1.0s), skipping music detection")
        return False, 0.0

    # PANNs inference (model is already loaded)
    (clipwise_output, embedding) = panns_model.inference(waveform_32k[None, :])

    # Get labels
    labels = panns_model.labels

    # Find Music probability
    music_idx = labels.index('Music') if 'Music' in labels else None
    if music_idx is not None:
        music_prob = float(clipwise_output[0, music_idx])
        has_music = music_prob > threshold
        return has_music, music_prob
    else:
        return False, 0.0


def separate_full_vocals_demucs(full_audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """
    Run Demucs once on the entire audio to obtain a reusable vocal stem.
    Returns the vocal waveform resampled to the original sample_rate.
    """
    temp_dir = tempfile.mkdtemp(prefix="demucs_full_")

    try:
        temp_input = os.path.join(temp_dir, "full.wav")
        sf.write(temp_input, full_audio, sample_rate)

        import subprocess
        demucs_output_dir = os.path.join(temp_dir, "separated")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running single Demucs pass on device: {device}")

        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-d", device,
            "-o", demucs_output_dir,
            temp_input,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Demucs full-audio run failed: {result.stderr}")
            return None

        input_stem = Path(temp_input).stem
        vocal_path = os.path.join(demucs_output_dir, "htdemucs", input_stem, "vocals.wav")
        if not os.path.exists(vocal_path):
            logger.error(f"Vocal track not found at {vocal_path}")
            return None

        vocal_waveform, _ = librosa.load(vocal_path, sr=sample_rate, mono=True)
        return vocal_waveform.astype(np.float32)

    except Exception as e:
        logger.error(f"Error during full Demucs processing: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def remove_segment_background_music_demucs(segment_audio, sample_rate, full_vocals=None, start_frame=None, end_frame=None):
    """
    Remove background music from segment audio using Demucs and extract vocals only.
    If full_vocals is provided, slice from the precomputed stem.

    Args:
        segment_audio (np.ndarray): Segment audio waveform.
        sample_rate (int): Sample rate.
        full_vocals (np.ndarray | None): Vocal waveform pre-extracted from full audio using Demucs.
        start_frame (int | None): Segment start frame (relative to full waveform).
        end_frame (int | None): Segment end frame (relative to full waveform).

    Returns:
        np.ndarray: Vocal-only waveform, or original on failure.
    """
    if full_vocals is not None and start_frame is not None and end_frame is not None:
        # Fast path: slice from pre-separated vocal track
        start = max(0, int(start_frame))
        end = min(int(end_frame), len(full_vocals))
        if end <= start:
            return segment_audio

        vocal_slice = full_vocals[start:end]
        if len(vocal_slice) == 0:
            return segment_audio

        target_length = len(segment_audio)
        if len(vocal_slice) >= target_length:
            return vocal_slice[:target_length].astype(np.float32)

        padded = np.zeros(target_length, dtype=np.float32)
        padded[: len(vocal_slice)] = vocal_slice.astype(np.float32)
        return padded

    # Create temporary directory for demucs output
    temp_dir = tempfile.mkdtemp(prefix="demucs_seg_")

    try:
        # Save segment audio to temporary file
        temp_input = os.path.join(temp_dir, "segment.wav")
        sf.write(temp_input, segment_audio, sample_rate)

        # Run demucs to separate vocals
        import subprocess
        demucs_output_dir = os.path.join(temp_dir, "separated")

        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Running Demucs on device: {device}")

        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-d", device,  # Explicitly specify device (cuda or cpu)
            "-o", demucs_output_dir,
            temp_input
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Demucs failed for segment: {result.stderr}")
            return segment_audio

        # Load separated vocal track
        vocal_path = os.path.join(demucs_output_dir, "htdemucs", "segment", "vocals.wav")

        if not os.path.exists(vocal_path):
            logger.error(f"Vocal track not found at {vocal_path}")
            return segment_audio

        # Load vocal-only audio
        vocal_waveform, _ = librosa.load(vocal_path, sr=sample_rate, mono=True)
        return vocal_waveform.astype(np.float32)

    except Exception as e:
        logger.error(f"Error during segment Demucs processing: {e}")
        return segment_audio
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@time_logger
def preprocess_segments_with_demucs(segment_list, audio, panns_model=None, use_demucs=False, padding=0.5):
    """
    Detect background music and apply Demucs per segment before ASR.
    (Adds padding to account for ASR timestamp shifts)
    """
    if not use_demucs:
        logger.info("Demucs preprocessing skipped (flag disabled)")
        return audio, [False] * len(segment_list)

    logger.info(f"Preprocessing {len(segment_list)} segments with background music detection and removal (padding={padding}s)")

    waveform = audio["waveform"].copy()
    sample_rate = audio["sample_rate"]
    total_samples = len(waveform)
    segment_demucs_flags = []
    vocal_full = None
    full_demucs_attempted = False

    for idx, segment in enumerate(segment_list):
        # Apply padding to cover ASR boundary shifts
        start_time = max(0, segment["start"] - padding)
        end_time = segment["end"] + padding # Boundary overflow is handled automatically by slicing
        
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        
        # Prevent exceeding total length
        end_frame = min(end_frame, total_samples)

        segment_audio = waveform[start_frame:end_frame]
        
        # Skip if segment is too short
        if len(segment_audio) < 16000: # Less than 0.5 seconds
            segment_demucs_flags.append(False)
            continue

        # Detect background music (check the padded region)
        has_music, music_prob = detect_segment_background_music(segment_audio, sample_rate, panns_model, threshold=0.3)
        
        if has_music:
            logger.info(f"Segment {idx} (with padding): Background music detected (prob={music_prob:.3f}), applying Demucs...")
            # Apply demucs (prefer single full-audio stem to avoid per-segment overhead)
            if vocal_full is None and not full_demucs_attempted:
                vocal_full = separate_full_vocals_demucs(waveform, sample_rate)
                full_demucs_attempted = True
                if vocal_full is None:
                    logger.warning("Full Demucs pass failed; falling back to per-segment separation.")

            if vocal_full is not None:
                vocal_audio = remove_segment_background_music_demucs(
                    segment_audio,
                    sample_rate,
                    full_vocals=vocal_full,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            else:
                vocal_audio = remove_segment_background_music_demucs(segment_audio, sample_rate)
            
            # Replace the segment in the waveform
            # Adjust length since vocal_audio may differ from segment_audio
            target_length = len(segment_audio)
            if len(vocal_audio) >= target_length:
                waveform[start_frame:end_frame] = vocal_audio[:target_length]
            else:
                # Rare case: pad if vocal_audio is shorter
                waveform[start_frame : start_frame + len(vocal_audio)] = vocal_audio

            segment_demucs_flags.append(True)
            logger.info(f"Segment {idx}: Demucs applied successfully")
        else:
            segment_demucs_flags.append(False)

    # Update audio dictionary with processed waveform
    updated_audio = audio.copy()
    updated_audio["waveform"] = waveform

    # Also update audio_segment for export
    from pydub import AudioSegment as PydubAudioSegment
    waveform_clipped = np.clip(waveform, -1.0, 1.0)
    waveform_int16 = (waveform_clipped * 32767).astype(np.int16)
    updated_audio_segment = PydubAudioSegment(
        waveform_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    updated_audio["audio_segment"] = updated_audio_segment

    logger.info(f"Demucs preprocessing completed: {sum(segment_demucs_flags)}/{len(segment_list)} segments processed")
    return updated_audio, segment_demucs_flags


@time_logger
def speaker_diarization(audio):
    """
    Perform speaker diarization on the given audio.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        pd.DataFrame: A dataframe containing segments with speaker labels.
    """
    logger.debug(f"Start speaker diarization")
    logger.debug(f"audio waveform shape: {audio['waveform'].shape}")

    waveform = torch.tensor(audio["waveform"]).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    segments = dia_pipeline(
        {
            "waveform": waveform,
            "sample_rate": audio["sample_rate"],
            "channel": 0,
        },
        max_speakers=4
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    logger.debug(f"diarize_df: {diarize_df}")

    return diarize_df


@time_logger
def cut_by_speaker_label(vad_list):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = args.merge_gap  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 3  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 30  # max segment length in seconds

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            logger.warning(
                f"cut_by_speaker_label > segment longer than 30s, force trimming to 30s smaller segments"
            )
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                vad["end"] = current_start + MAX_SEGMENT_LENGTH  # update end time
                updated_list.append(vad)
                vad = vad.copy()
                current_start += MAX_SEGMENT_LENGTH
                vad["start"] = current_start  # update start time
                vad["end"] = segment_end
            updated_list.append(vad)
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time

    logger.debug(
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
    )

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    logger.debug(
        f"cut_by_speaker_label > removed: {len(updated_list) - len(filter_list)} segments by length"
    )

    return filter_list

@time_logger
def detect_overlapping_segments(segment_list, overlap_threshold=0.2):
    """
    Detect segments that overlap for more than overlap_threshold seconds.

    Args:
        segment_list (list): List of segments with 'start', 'end', and 'speaker' keys
        overlap_threshold (float): Minimum overlap duration in seconds to be considered

    Returns:
        list: List of overlapping segment pairs with overlap info
            [{'seg1': segment1, 'seg2': segment2, 'overlap_start': float, 'overlap_end': float, 'overlap_duration': float}]
    """
    overlapping_pairs = []

    # Sort segments by start time
    sorted_segments = sorted(segment_list, key=lambda x: x['start'])

    for i in range(len(sorted_segments)):
        for j in range(i + 1, len(sorted_segments)):
            seg1 = sorted_segments[i]
            seg2 = sorted_segments[j]

            # If seg2 starts after seg1 ends, no more overlaps possible for seg1
            if seg2['start'] >= seg1['end']:
                break

            # Calculate overlap
            overlap_start = max(seg1['start'], seg2['start'])
            overlap_end = min(seg1['end'], seg2['end'])
            overlap_duration = overlap_end - overlap_start

            # Check if overlap exceeds threshold
            if overlap_duration >= overlap_threshold:
                overlapping_pairs.append({
                    'seg1': seg1,
                    'seg2': seg2,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_duration': overlap_duration
                })
                logger.info(f"Overlap detected: {overlap_duration:.2f}s between "
                           f"[{seg1['start']:.2f}-{seg1['end']:.2f}] and "
                           f"[{seg2['start']:.2f}-{seg2['end']:.2f}]")

    return overlapping_pairs


class SepReformerSeparator:
    """
    Class to load the SepReformer model once and perform multiple inferences.
    """
    def __init__(self, sepreformer_path, device):
        """
        Initialize and load the SepReformer model.

        Args:
            sepreformer_path: Path to SepReformer model directory
            device: torch device (cuda/cpu)
        """
        import sys
        import yaml

        self.sepreformer_path = sepreformer_path
        self.device = device

        print(f"[SepReformer] Initializing on device: {self.device}")

        # Store original sys.path to restore later
        original_sys_path = sys.path.copy()

        try:
            # Save the current 'models' and 'utils' modules if they exist
            original_models = sys.modules.get('models', None)
            original_utils = sys.modules.get('utils', None)

            # Remove podcast-pipeline from sys.path temporarily
            podcast_pipeline_path = os.path.dirname(os.path.abspath(__file__))
            paths_to_remove = [p for p in sys.path if podcast_pipeline_path in p]
            for path in paths_to_remove:
                sys.path.remove(path)

            # Add SepReformer to path
            if sepreformer_path not in sys.path:
                sys.path.insert(0, sepreformer_path)

            # Clear conflicting modules
            modules_to_clear = [key for key in sys.modules.keys()
                              if key.startswith('models.') or key.startswith('utils.') or key in ['models', 'utils']]
            cleared_modules = {}
            for module_name in modules_to_clear:
                cleared_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]

            # Import SepReformer's model
            from models.SepReformer_Base_WSJ0.model import Model

            # Restore the original modules
            for module_name, module_obj in cleared_modules.items():
                sys.modules[module_name] = module_obj

            # Load SepReformer config
            config_path = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/configs.yaml")
            with open(config_path, 'r') as f:
                yaml_dict = yaml.safe_load(f)
            self.config = yaml_dict["config"]

            # Load model
            print("[SepReformer] Loading model...")
            self.model = Model(**self.config["model"])

            # Load checkpoint
            checkpoint_dir = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/log/pretrain_weights")
            if not os.path.exists(checkpoint_dir) or not os.listdir(checkpoint_dir):
                checkpoint_dir = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/log/scratch_weights")

            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pt', '.pth'))]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(device)
            self.model.eval()

            print("[SepReformer] Model initialization complete!")

        finally:
            # Restore original sys.path
            sys.path = original_sys_path

    def separate(self, audio_segment, sample_rate):
        """
        Perform audio separation.

        Args:
            audio_segment (np.ndarray): Audio segment to separate
            sample_rate (int): Audio sample rate

        Returns:
            tuple: (separated_audio_1, separated_audio_2) as numpy arrays
        """
        try:
            # Resample to 8kHz if needed
            if sample_rate != 8000:
                audio_8k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=8000)
            else:
                audio_8k = audio_segment

            # Prepare tensor
            mixture_tensor = torch.tensor(audio_8k, dtype=torch.float32).unsqueeze(0)

            # Padding
            stride = self.config["model"]["module_audio_enc"]["stride"]
            remains = mixture_tensor.shape[-1] % stride
            if remains != 0:
                padding = stride - remains
                mixture_padded = torch.nn.functional.pad(mixture_tensor, (0, padding), "constant", 0)
            else:
                mixture_padded = mixture_tensor

            # Inference
            with torch.inference_mode():
                nnet_input = mixture_padded.to(self.device)
                estim_src, _ = self.model(nnet_input)

                # Extract separated sources
                src1 = estim_src[0][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()
                src2 = estim_src[1][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()

            # Resample back to original sample rate if needed
            if sample_rate != 8000:
                src1 = librosa.resample(src1, orig_sr=8000, target_sr=sample_rate)
                src2 = librosa.resample(src2, orig_sr=8000, target_sr=sample_rate)

            return src1, src2

        except Exception as e:
            logger.error(f"SepReformer separation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return audio_segment, audio_segment


@time_logger
def identify_speaker_with_embedding(audio_segment, sample_rate, reference_embeddings, speaker_labels, embedding_model):
    """
    Identify which speaker an audio segment belongs to using speaker embeddings.

    Args:
        audio_segment (np.ndarray): Audio segment to identify
        sample_rate (int): Sample rate of the audio
        reference_embeddings (dict): Dictionary of {speaker_label: embedding_tensor}
        speaker_labels (list): List of possible speaker labels
        embedding_model: Pre-loaded pyannote embedding model

    Returns:
        str: Identified speaker label
    """

    # Extract embedding from audio segment
    # Resample to 16kHz if needed (pyannote expects 16kHz)
    if sample_rate != 16000:
        audio_16k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=16000)
    else:
        audio_16k = audio_segment

    # Skip if too short for TDNN receptive field
    if len(audio_16k) < int(MIN_EMBED_DURATION * 16000):
        logger.warning(
            f"Embedding skip: segment too short ({len(audio_16k)/16000:.2f}s < {MIN_EMBED_DURATION}s); "
            f"falling back to first candidate {speaker_labels[0] if speaker_labels else 'Unknown'}"
        )
        return speaker_labels[0] if speaker_labels else None

    # Convert to tensor
    audio_tensor = torch.tensor(audio_16k, dtype=torch.float32).unsqueeze(0).to(device)

    # Extract embedding (guard against short/invalid audio)
    try:
        with torch.inference_mode():
            embedding = embedding_model(audio_tensor)
    except Exception as e:
        logger.warning(
            f"Embedding model failed on segment ({len(audio_16k)/16000:.2f}s): {e}. "
            f"Falling back to first candidate {speaker_labels[0] if speaker_labels else 'Unknown'}"
        )
        return speaker_labels[0] if speaker_labels else None

    # Compare with reference embeddings using cosine similarity
    best_speaker = None
    best_similarity = -1.0

    for speaker_label in speaker_labels:
        if speaker_label in reference_embeddings:
            ref_embedding = reference_embeddings[speaker_label]
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                embedding.mean(dim=1),
                ref_embedding.mean(dim=1),
                dim=0
            ).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_label

    logger.debug(f"Speaker identification: {best_speaker} (similarity: {best_similarity:.3f})")
    return best_speaker


@time_logger
def process_overlapping_segments_with_separation(segment_list, audio, overlap_threshold=1.0,
                                                 separator=None, embedding_model=None):
    """
    Process overlapping segments by separating them with SepReformer.
    [Updated] Matches the volume of separated audio to the original overlap audio to prevent volume jumps.

    Args:
        segment_list: List of segments
        audio: Audio dictionary
        overlap_threshold: Overlap threshold
        separator: Pre-loaded SepReformerSeparator object
        embedding_model: Pre-loaded pyannote embedding model
    """
    if separator is None:
        logger.warning("SepReformer separator not provided, skipping separation")
        return audio, segment_list

    if embedding_model is None:
        logger.warning("Embedding model not provided, skipping separation")
        return audio, segment_list

    logger.info(f"Processing overlapping segments with SepReformer (threshold: {overlap_threshold}s)")

    # -------------------------------------------------------------------------
    # [Added] Volume matching helper function
    # -------------------------------------------------------------------------
    def match_target_amplitude(source_wav, target_wav):
        """
        Match the volume (RMS) of source_wav to that of target_wav.
        """
        # Epsilon to prevent division by zero
        epsilon = 1e-10

        # Calculate RMS (Root Mean Square) energy
        src_rms = np.sqrt(np.mean(source_wav**2))
        tgt_rms = np.sqrt(np.mean(target_wav**2))
        
        if src_rms < epsilon:
            return source_wav
        
        # Calculate ratio (how much larger/smaller target is compared to source)
        gain = tgt_rms / (src_rms + epsilon)
        
        # Apply gain
        adjusted_wav = source_wav * gain
        
        # Prevent clipping (-1.0 ~ 1.0)
        return np.clip(adjusted_wav, -1.0, 1.0)
    # -------------------------------------------------------------------------

    # 1. Initialize 'enhanced_audio' for all segments with the original audio
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    
    for seg in segment_list:
        if 'enhanced_audio' not in seg:
            start_frame = int(seg['start'] * sample_rate)
            end_frame = int(seg['end'] * sample_rate)
            seg['enhanced_audio'] = waveform[start_frame:end_frame].copy()
        
        if 'sepreformer' not in seg:
            seg['sepreformer'] = False

    # Detect overlapping segments
    overlapping_pairs = detect_overlapping_segments(segment_list, overlap_threshold)

    if not overlapping_pairs:
        logger.info("No overlapping segments found")
        return audio, segment_list

    logger.info(f"Found {len(overlapping_pairs)} overlapping segment pairs")

    # (Reference Embeddings extraction logic - kept as-is)
    reference_embeddings = {}
    all_speakers = list(set([seg['speaker'] for seg in segment_list]))

    # ... (Reference Embedding extraction - kept as-is) ...
    for speaker in all_speakers:
        speaker_segments = [seg for seg in segment_list if seg['speaker'] == speaker]
        for seg in speaker_segments:
            is_overlapping = any(pair['seg1'] == seg or pair['seg2'] == seg for pair in overlapping_pairs)
            if not is_overlapping and (seg['end'] - seg['start']) >= 2.0:
                start_frame = int(seg['start'] * sample_rate)
                end_frame = int(seg['end'] * sample_rate)
                seg_audio = waveform[start_frame:end_frame]
                if sample_rate != 16000:
                    seg_audio_16k = librosa.resample(seg_audio, orig_sr=sample_rate, target_sr=16000)
                else:
                    seg_audio_16k = seg_audio

                if len(seg_audio_16k) < int(MIN_EMBED_DURATION * 16000):
                    logger.debug(
                        f"Skipping reference embedding for {speaker}: segment too short "
                        f"({len(seg_audio_16k)/16000:.2f}s < {MIN_EMBED_DURATION}s)"
                    )
                    continue

                seg_tensor = torch.tensor(seg_audio_16k, dtype=torch.float32).unsqueeze(0).to(device)
                try:
                    with torch.inference_mode():
                        embedding = embedding_model(seg_tensor)
                    reference_embeddings[speaker] = embedding
                    break
                except Exception as e:
                    logger.warning(f"Failed to compute reference embedding for {speaker}: {e}")
                    continue

    # 2. Process overlap pairs
    for pair_idx, pair in enumerate(overlapping_pairs):
        overlap_start = pair['overlap_start']
        overlap_end = pair['overlap_end']
        seg1 = pair['seg1']
        seg2 = pair['seg2']
        
        seg1_speaker = seg1['speaker']
        seg2_speaker = seg2['speaker']

        # Extract overlapping audio (Original Mixture)
        start_frame = int(overlap_start * sample_rate)
        end_frame = int(overlap_end * sample_rate)
        overlap_audio = waveform[start_frame:end_frame]

        # Separate with SepReformer
        separated_src1, separated_src2 = separator.separate(
            overlap_audio, sample_rate
        )

        # Identify speakers
        speaker1_identity = identify_speaker_with_embedding(
            separated_src1, sample_rate, reference_embeddings, [seg1_speaker, seg2_speaker], embedding_model
        )
        
        if speaker1_identity == seg1_speaker:
            seg1_part = separated_src1
            seg2_part = separated_src2
        else:
            seg1_part = separated_src2
            seg2_part = separated_src1

        # ---------------------------------------------------------------------
        # [Fixed] Apply volume correction (match to original overlap region volume)
        # ---------------------------------------------------------------------
        # Adjust separated audio to match the RMS energy of the original overlap region (mixed sound)
        # (Note: the original has 2 speakers mixed so its energy is naturally higher than a single separated source,
        #  but this reference is much more natural than letting SepReformer output spike to 0dB.)
        
        logger.debug(f"   Adjusting volume for overlap {pair_idx+1}...")
        seg1_part = match_target_amplitude(seg1_part, overlap_audio)
        seg2_part = match_target_amplitude(seg2_part, overlap_audio)
        # ---------------------------------------------------------------------

        # 1) Update Seg1
        seg1_start_global = int(seg1['start'] * sample_rate)
        rel_start_1 = start_frame - seg1_start_global
        
        limit_len_1 = min(len(seg1_part), len(seg1['enhanced_audio'][rel_start_1:]))
        if limit_len_1 > 0:
            seg1['enhanced_audio'][rel_start_1 : rel_start_1 + limit_len_1] = seg1_part[:limit_len_1]
            seg1['sepreformer'] = True
            logger.info(f"  ✓ Updated Seg1 enhanced_audio with volume-adjusted separated audio") 

        # 2) Update Seg2
        seg2_start_global = int(seg2['start'] * sample_rate)
        rel_start_2 = start_frame - seg2_start_global

        limit_len_2 = min(len(seg2_part), len(seg2['enhanced_audio'][rel_start_2:]))
        if limit_len_2 > 0:
            seg2['enhanced_audio'][rel_start_2 : rel_start_2 + limit_len_2] = seg2_part[:limit_len_2]
            seg2['sepreformer'] = True
            logger.info(f"  ✓ Updated Seg2 enhanced_audio with volume-adjusted separated audio")

    return audio, segment_list


@time_logger
def asr(vad_segments, audio):
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

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    # Following the asr_MoE approach (individual processing), batch size is set to 1.
    # Can be adjusted based on library support, but individual processing is prioritized for accuracy.
    batch_size = 1

    if multilingual_flag:
        # ... (Existing multilingual logic kept or needs to be converted to same loop structure) ...
        # Multilingual implementation is extensive, left as placeholder for now
        pass
        return []

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
            # Prefer using audio separated by SepReformer if available
            raw_audio = segment["enhanced_audio"]
            is_enhanced = True
        else:
            # Otherwise, slice the corresponding section from full audio
            start_frame = int(start_time * global_sample_rate)
            end_frame = int(end_time * global_sample_rate)
            raw_audio = full_waveform[start_frame:end_frame]
            is_enhanced = False

        # 16kHz resampling (for Whisper input)
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
        # Since we're feeding pre-sliced audio, relative time is 0 ~ duration.
        duration_sec = len(segment_audio_16k) / 16000
        dummy_vad = [{"start": 0.0, "end": duration_sec}]

        try:
            # Language detection (can be done per segment or fixed to 'en')
            # Default to 'en' following existing flow; use detect_language if needed
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
                    # 1. Only process non-empty text
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

                        # 4. Correct timestamps if word-level timestamps exist
                        if "words" in res_seg:
                            for w in res_seg["words"]:
                                w["start"] += start_time
                                w["end"] += start_time

                        final_results.append(res_seg)

        except Exception as e:
            logger.error(f"ASR failed for segment {idx} ({start_time:.2f}-{end_time:.2f}): {e}")
            continue

    return final_results

import concurrent.futures

@time_logger
def asr_MoE(vad_segments, audio, segment_demucs_flags=None, enable_word_timestamps=False, device="cuda", lang="en"):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments using MoE with Parallel Execution.
    [Updated] Runs Whisper, Parakeet, and Canary in parallel using ThreadPoolExecutor.

    When `lang == "vi"`, the second and third ASR slots are swapped to VN-capable
    models: PhoWhisper-large (asr_model_2) and ChunkFormer-CTC (canary_model).
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

    def run_phowhisper_task(segment_audio_16k):
        # PhoWhisper-large via a transformers ASR pipeline stored in asr_model_2 (VN path).
        try:
            result = asr_model_2(
                {"array": segment_audio_16k, "sampling_rate": 16000},
                generate_kwargs={"language": "vi", "task": "transcribe"},
            )
            if isinstance(result, dict):
                return result.get("text", "") or ""
            return str(result) if result else ""
        except Exception as e:
            logger.error(f"PhoWhisper failed: {e}")
            return ""

    def run_chunkformer_task(segment_audio_16k):
        # Run Qwen3-ASR for inference via subprocess worker
        try:
            if not _qwen3_ready or qwen3_process is None:
                return ""
            
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                tmp_audio_path = tmp_audio.name
            sf.write(tmp_audio_path, segment_audio_16k, 16000)

            # Send request
            req = {"cmd": "transcribe", "audio_path": tmp_audio_path, "language": "vi"}
            qwen3_process.stdin.write(json.dumps(req) + "\n")
            
            # Read response
            resp_line = qwen3_process.stdout.readline()
            
            # Cleanup temp file
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
                
            if not resp_line:
                logger.error("Qwen3-ASR worker returned empty response.")
                return ""
                
            resp = json.loads(resp_line)
            if "error" in resp:
                logger.error(f"Qwen3-ASR worker error: {resp['error']}")
                return ""
            return resp.get("text", "")
        except Exception as e:
            logger.error(f"Qwen3-ASR worker communication failed: {e}")
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
            
            # 16kHz resampling
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
            if lang == "vi":
                future_2 = executor.submit(run_phowhisper_task, segment_audio_16k)
                future_3 = executor.submit(run_chunkformer_task, segment_audio_16k)
            else:
                future_2 = executor.submit(run_parakeet_task, segment_audio_16k)
                future_3 = executor.submit(run_canary_task, segment_audio_16k)

            # ---------------------------------------------------------------------
            # Wait for results (Barrier)
            # ---------------------------------------------------------------------
            # .result() blocks until the future is done
            whisper_res = future_whisper.result()
            text_2 = future_2.result()
            text_3 = future_3.result()

            # Unpack Whisper results
            text_whisper = whisper_res["text"]
            detected_language = whisper_res["language"]
            words = whisper_res["words"]
            total_whisper_time += whisper_res["time"]

            # ---------------------------------------------------------------------
            # 5. Ensemble & Result Construction
            # ---------------------------------------------------------------------
            text_ensemble = rover.align_and_vote([text_whisper, text_3, text_2])

            seg_result = {
                "start": start_time,
                "end": end_time,
                "text": text_ensemble,
                "text_whisper": text_whisper,
                "speaker": speaker,
                "language": detected_language,
                "demucs": segment_demucs_flags[idx] if idx < len(segment_demucs_flags) else False,
                "is_separated": is_enhanced,
                "sepreformer": segment.get("sepreformer", False)
            }
            if lang == "vi":
                seg_result["text_phowhisper"] = text_2
                seg_result["text_qwen3"] = text_3
            else:
                seg_result["text_parakeet"] = text_2
                seg_result["text_canary"] = text_3
            
            if is_enhanced:
                seg_result["enhanced_audio"] = raw_audio

            if enable_word_timestamps and words:
                for w in words:
                    w["start"] += start_time
                    w["end"] += start_time
                seg_result["words"] = words

            final_results.append(seg_result)

    return final_results, total_whisper_time, total_alignment_time

def add_qwen3omni_caption(filtered_list, audio, save_path):
    """
    Call Qwen3-Omni API for each ASR result segment to add audio captions.

    Args:
        filtered_list (list): ASR result segment list
        audio (dict): Audio dictionary (containing waveform and sample_rate)
        save_path (str): Path to save temporary audio files

    Returns:
        tuple: (segment list with qwen3omni_caption added, processing time in seconds)
    """
    import soundfile as sf

    logger.info(f"Adding Qwen3-Omni captions to {len(filtered_list)} segments...")
    caption_start_time = time.time()

    for idx, segment in enumerate(filtered_list):
        try:
            # Extract segment audio
            # [CRITICAL] Prefer audio processed by SepReformer if available (to match the saved file)
            if "enhanced_audio" in segment:
                segment_audio = segment["enhanced_audio"]
                sample_rate = audio["sample_rate"]
            else:
                start_time = segment["start"]
                end_time = segment["end"]
                sample_rate = audio["sample_rate"]
                start_frame = int(start_time * sample_rate)
                end_frame = int(end_time * sample_rate)
                segment_audio = audio["waveform"][start_frame:end_frame]

            # Save as temporary audio file
            temp_audio_path = os.path.join(save_path, f"temp_segment_{idx:05d}.wav")
            sf.write(temp_audio_path, segment_audio, sample_rate)

            # Call Qwen3-Omni API
            url = f"http://localhost:{QWEN_3_OMNI_PORT}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}

            # Must encode local file as base64 or provide as URL
            # Here we use the temporary file path (actual deployment may need a server-accessible URL)
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": f"file://{temp_audio_path}"}
                            }
                        ]
                    }
                ]
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                # Extract content from response
                caption = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                segment["qwen3omni_caption"] = caption
                logger.debug(f"Segment {idx}: Successfully added Qwen3-Omni caption")
            else:
                logger.warning(f"Segment {idx}: API call failed with status {response.status_code}")
                segment["qwen3omni_caption"] = ""

            # Delete temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        except Exception as e:
            logger.error(f"Segment {idx}: Error calling Qwen3-Omni API: {e}")
            segment["qwen3omni_caption"] = ""

    caption_end_time = time.time()
    caption_processing_time = caption_end_time - caption_start_time

    logger.info("Qwen3-Omni caption addition completed")
    return filtered_list, caption_processing_time


# Cost calculation function
def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = {
        "gpt-4.1": {
            "input": 2.00 / 1_000_000,
            "cached_input": 0.50 / 1_000_000,
            "output": 8.00 / 1_000_000,
        },
        "gpt-4.1-mini": {
            "input": 0.40 / 1_000_000,
            "cached_input": 0.10 / 1_000_000,
            "output": 1.60 / 1_000_000,
        },
        "gpt-4.1-nano": {
            "input": 0.10 / 1_000_000,
            "cached_input": 0.025 / 1_000_000,
            "output": 0.40 / 1_000_000,
        },
        "openai-o3": {
            "input": 2.00 / 1_000_000,
            "cached_input": 0.50 / 1_000_000,
            "output": 8.00 / 1_000_000,
        },
        "openai-o4-mini": {
            "input": 1.10 / 1_000_000,
            "cached_input": 0.275 / 1_000_000,
            "output": 4.40 / 1_000_000,
        },
    }

    if model_name not in pricing:
        raise ValueError(f"Model '{model_name}' not found in pricing table.")

    rates = pricing[model_name]
    input_cost = input_tokens * rates["input"]
    output_cost = output_tokens * rates["output"]
    total_cost = input_cost + output_cost

    return total_cost
import json
from collections import defaultdict



def speaker_tagged_text(data):
    """
    Add speaker tags to the given data and reassign speaker numbers
    sequentially (s0, s1, s2...) based on order of appearance in the text.
    """
    # 1. Generate initial tags and record the order of unique speaker appearances.
    initially_tagged_data = []
    unique_speakers_in_order = []
    seen_speakers = set()

    for item in data:
        # Generate original tag in 'SPEAKER_01' -> '[s1]' format
        speaker_num = item['speaker'].replace('SPEAKER_', '')
        original_tag = f"[s{int(speaker_num)}]"

        # Add new unique speaker tags to the list in order of appearance
        if original_tag not in seen_speakers:
            unique_speakers_in_order.append(original_tag)
            seen_speakers.add(original_tag)
        
        # Temporarily store original text and tags for later mapping application
        initially_tagged_data.append({
            'text': item['text'],
            'start': item['start'],
            'end': item['end'],
            'original_tag': original_tag
        })

    # 2. Create a mapping to convert original tags to new sequential tags.
    # e.g.: {'[s2]': '[s0]', '[s0]': '[s1]', '[s1]': '[s2]'}
    speaker_map = {
        original_tag: f"[s{i}]" 
        for i, original_tag in enumerate(unique_speakers_in_order)
    }

    # 3. Apply the generated mapping to produce the final result.
    result = []
    for item in initially_tagged_data:
        original_tag = item['original_tag']
        new_tag = speaker_map[original_tag]  # Get new tag from mapping
        
        final_item = {
            'text': f"{new_tag}{item['text']}",  # Prepend new tag to text
            'start': item['start'],
            'end': item['end']
        }
        result.append(final_item)
        
    return result

import re
import json
import ast

def parse_speaker_summary(llm_output: str) -> list | None:
    """
    Extract and parse a JSON array from the LLM output string.
    Handles 'json' prefix, code blocks (```), and surrounding whitespace.
    """
    if not llm_output:
        return None

    try:
        # Remove code blocks like ```json ... ``` or ``` ... ```
        # Use regex to find content between '[' and ']'
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Convert JSON string to Python object (list of dicts)
            return json.loads(json_str)
        else:
            print("Parsing Error: Could not find a valid JSON array format ([]).")
            return None
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"Unknown parsing error: {e}")
        return None

def process_llm_diarization_output(llm_output: str) -> list[dict]:

    # 1. Find ```json ... ``` code blocks in LLM output
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_output)
    if not json_match:
        # If no ```json block found, try parsing the entire string
        json_string = llm_output
    else:
        json_string = json_match.group(1)

    # 2. Parse JSON string into Python object
    try:
        llm_data = json.loads(json_string)
    except json.JSONDecodeError:
        # Fallback for when LLM outputs Python list format ('[{"text":...}]')
        try:
            # ast.literal_eval is a more secure version of eval.
            import ast
            llm_data = ast.literal_eval(json_string)
        except (ValueError, SyntaxError) as e:
            print(f"Error: Both JSON and Python literal parsing failed. {e!r}")
            return []


    return llm_data

def sortformer_dia(predicted_segments):
    lists = [x for x in predicted_segments if isinstance(x, (list, tuple))]
    if not lists:
        lists = predicted_segments
    segs = [s for sub in lists for s in sub]

    rows = []
    for idx, seg in enumerate(segs):
        start_s, end_s, sp = seg.split()
        start, end = float(start_s), float(end_s)
        # Convert SPEAKER format
        num = int(sp.split('_')[1])
        speaker = f"SPEAKER_{num:02d}"
        # Labels (A, B, C, ...)
        label = chr(ord('A') + idx)
        # Time formatting function
        def fmt(sec):
            td = datetime.timedelta(seconds=sec)
            hrs = td.seconds // 3600 + td.days * 24
            mins = (td.seconds // 60) % 60
            secs = td.seconds % 60
            ms = int(td.microseconds / 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
        segment_str = f"[ {fmt(start)} --> {fmt(end)}]"
        rows.append({
            'segment': segment_str,
            'label': label,
            'speaker': speaker,
            'start': start,
            'end': end
        })

    df = pd.DataFrame(rows, columns=['segment','label','speaker','start','end'])
    df = df.sort_values(by='start').reset_index(drop=True)
    return df

def df_to_list(df: pd.DataFrame) -> list[dict]:
    """
    Convert each row of the DataFrame to a dict with the following format:
      - index: 5-digit zero-padded string
      - start: float
      - end: float
      - speaker: str
    """
    records = []
    for i, row in df.iterrows():
        records.append({
            'index': f"{i:05d}",
            'start': float(row['start']),
            'end': float(row['end']),
            'speaker': row['speaker']
        })
    return records


def deduplicate_segments_by_index(segments: list[dict], logger=None) -> list[dict]:
    """
    Ensure each segment index is unique, keeping the first occurrence.
    """
    seen = set()
    deduped = []
    for seg in segments:
        idx = seg.get("index")
        if idx is None or idx not in seen:
            if idx is not None:
                seen.add(idx)
            deduped.append(seg)
        else:
            if logger:
                logger.warning(f"Duplicate segment index detected and skipped: {idx}")
    return deduped

def split_long_segments(segment_list, max_duration=30.0):
    """
    Split segments longer than max_duration by time (without using VAD).

    Args:
        segment_list (list): List of segment dictionaries to split.
        max_duration (float): Maximum allowed segment length in seconds.

    Returns:
        list: New segment list after splitting.
    """
    new_segments = []
    new_index = 0

    for segment in segment_list:
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment['speaker']
        duration = end_time - start_time

        # If segment length is less than or equal to max, add as-is
        if duration <= max_duration:
            segment['index'] = str(new_index).zfill(5)
            new_segments.append(segment)
            new_index += 1
        # If segment length exceeds max, split it
        else:
            current_start = start_time
            # Repeat while current start time is less than the original segment's end time
            while current_start < end_time:
                # Calculate next split point (add max length, but don't exceed original end time)
                chunk_end = min(current_start + max_duration, end_time)
                
                new_segments.append({
                    'index': str(new_index).zfill(5),
                    'start': round(current_start, 3),
                    'end': round(chunk_end, 3),
                    'speaker': speaker
                })
                new_index += 1
                # Update next chunk's start time to current chunk's end time
                current_start = chunk_end
                
    return new_segments


def _build_silence_intervals(waveform, sample_rate, min_silence):
    """
    Use VAD to find silence intervals that can be used as cut points.
    """
    vad_model = globals().get("vad")
    if vad_model is None:
        return len(waveform) / sample_rate, []

    if len(waveform) == 0:
        return 0.0, []

    resampled = librosa.resample(
        waveform, orig_sr=sample_rate, target_sr=silero_vad.SAMPLING_RATE
    )
    if resampled.size == 0:
        return len(waveform) / sample_rate, []

    speech_ts = vad_model.get_speech_timestamps(
        resampled,
        vad_model.vad_model,
        sampling_rate=silero_vad.SAMPLING_RATE,
    )
    total_duration = len(waveform) / sample_rate
    if not speech_ts:
        return total_duration, [(0.0, total_duration)]

    silence = []
    first_start = speech_ts[0]["start"] / silero_vad.SAMPLING_RATE
    if first_start >= min_silence:
        silence.append((0.0, first_start))

    for prev_seg, next_seg in zip(speech_ts[:-1], speech_ts[1:]):
        sil_start = prev_seg["end"] / silero_vad.SAMPLING_RATE
        sil_end = next_seg["start"] / silero_vad.SAMPLING_RATE
        if sil_end - sil_start >= min_silence:
            silence.append((sil_start, sil_end))

    last_end = speech_ts[-1]["end"] / silero_vad.SAMPLING_RATE
    trailing = total_duration - last_end
    if trailing >= min_silence:
        silence.append((last_end, last_end + trailing))
    return total_duration, silence


def _build_chunk_ranges(total_duration, silence_intervals, max_duration):
    """
    Determine chunk ranges by splitting at silence points, ensuring all chunks are under max_duration.
    Prioritizes splitting at silence (non-speech) intervals to avoid cutting in the middle of speech.
    """
    epsilon = 1e-3
    if total_duration <= max_duration + epsilon:
        return [(0.0, total_duration)]

    # Extract midpoints of silence intervals as potential cut points
    silence_points = sorted([(start + end) / 2.0 for start, end in silence_intervals])

    if not silence_points:
        # No silence found, fall back to hard cuts at max_duration
        chunk_ranges = []
        chunk_start = 0.0
        while chunk_start < total_duration - epsilon:
            chunk_end = min(chunk_start + max_duration, total_duration)
            chunk_ranges.append((chunk_start, chunk_end))
            chunk_start = chunk_end
        return chunk_ranges if chunk_ranges else [(0.0, total_duration)]

    chunk_ranges = []
    chunk_start = 0.0

    while chunk_start < total_duration - epsilon:
        # Find the best silence point within max_duration from chunk_start
        limit = chunk_start + max_duration

        # Get all silence points that are after chunk_start and before/at limit
        candidates = [p for p in silence_points if chunk_start + epsilon < p <= limit]

        if candidates:
            # Use the last (furthest) silence point within the limit
            chunk_end = candidates[-1]
        else:
            # No silence point found within max_duration
            # Check if there's any silence point after limit
            future_candidates = [p for p in silence_points if p > limit]
            if future_candidates:
                # There's silence ahead but beyond max_duration, use limit as hard cut
                chunk_end = min(limit, total_duration)
            else:
                # No more silence points, go to the end
                chunk_end = total_duration

        # Ensure we're making progress
        if chunk_end - chunk_start < epsilon:
            chunk_end = min(chunk_start + max_duration, total_duration)
            if chunk_end - chunk_start < epsilon:
                break

        chunk_ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end

    return chunk_ranges if chunk_ranges else [(0.0, total_duration)]


def _extract_speaker_embedding(
    audio_info,
    start: float,
    end: float,
    embedder: Inference | None,
    sample_window: float = 2.0,
    min_duration: float = 0.5,
):
    """
    Compute a single speaker embedding from a segment of the full audio.
    """
    if embedder is None:
        return None

    waveform = audio_info.get("waveform")
    sample_rate = audio_info.get("sample_rate")
    if waveform is None or sample_rate is None:
        return None

    total_duration = len(waveform) / sample_rate
    start = max(0.0, min(start, total_duration))
    end = max(start, min(end, total_duration))
    duration = end - start
    if duration < min_duration:
        return None

    # Center-crop long segments to a fixed analysis window.
    if duration > sample_window:
        center = (start + end) / 2.0
        start = center - sample_window / 2.0
        end = center + sample_window / 2.0

    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    segment = waveform[start_idx:end_idx]
    if segment.size == 0:
        return None

    target_sr = getattr(embedder, "sample_rate", 16000)
    try:
        if sample_rate != target_sr:
            segment = librosa.resample(segment, orig_sr=sample_rate, target_sr=target_sr)
    except Exception:
        # If resampling fails, fall back to the original segment.
        target_sr = sample_rate

    # pyannote Inference expects a mapping with waveform + sample_rate
    try:
        torch_seg = torch.as_tensor(segment, dtype=torch.float32).unsqueeze(0)
        emb = embedder({"waveform": torch_seg, "sample_rate": target_sr})
    except Exception:
        return None
    if emb is None:
        return None
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    if isinstance(emb, np.ndarray) and emb.ndim > 1:
        emb = emb.mean(axis=0)
    return emb


def _cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return -1.0
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return -1.0
    return float(np.dot(vec_a, vec_b) / denom)


def _compute_chunk_speaker_centroids(chunk_df: pd.DataFrame, audio_info, embedder: Inference | None):
    """
    Build per-speaker centroids for a diarization chunk using short audio snippets.
    """
    if embedder is None or chunk_df is None or chunk_df.empty:
        return {}

    centroids = {}
    for speaker, rows in chunk_df.groupby("speaker"):
        embeddings = []
        for _, row in rows.sort_values("start").iterrows():
            emb = _extract_speaker_embedding(
                audio_info, row["start"], row["end"], embedder=embedder
            )
            if emb is not None:
                embeddings.append(emb)
            if len(embeddings) >= 3:
                break
        if embeddings:
            centroids[speaker] = np.mean(embeddings, axis=0)
    return centroids


def align_speakers_across_chunks(
    chunk_frames: list[pd.DataFrame],
    audio_info,
    embedder: Inference | None,
    similarity_threshold: float = 0.75,
):
    """
    Link local speaker labels from sequential diarization chunks into a single
    recording-level speaker inventory using embedding similarity.
    """
    if embedder is None or not chunk_frames:
        logger.warning("Speaker embedder unavailable; skipping cross-chunk speaker linking.")
        return chunk_frames

    global_centroids: dict[str, np.ndarray | None] = {}
    global_counts: dict[str, int] = {}
    next_global_idx = 0
    aligned_frames: list[pd.DataFrame] = []

    for chunk_idx, df in enumerate(chunk_frames):
        if df is None or df.empty:
            aligned_frames.append(df)
            continue

        local_centroids = _compute_chunk_speaker_centroids(df, audio_info, embedder)
        mapping: dict[str, str] = {}
        used_global_ids_in_chunk: set[str] = set()  # Track global IDs already used in this chunk

        for local_speaker in df["speaker"].unique():
            emb = local_centroids.get(local_speaker)

            best_id = None
            best_sim = -1.0
            if emb is not None:
                for gid, centroid in global_centroids.items():
                    if centroid is None:
                        continue
                    # Skip global IDs already mapped to other local speakers in this chunk
                    if gid in used_global_ids_in_chunk:
                        continue
                    sim = _cosine_similarity(emb, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = gid

            if best_sim >= similarity_threshold and best_id is not None:
                mapping[local_speaker] = best_id
                used_global_ids_in_chunk.add(best_id)  # Mark this global ID as used in this chunk
                count = global_counts.get(best_id, 0)
                global_centroids[best_id] = (global_centroids[best_id] * count + emb) / (
                    count + 1
                )
                global_counts[best_id] = count + 1
            else:
                global_id = f"SPEAKER_{next_global_idx:02d}"
                next_global_idx += 1
                mapping[local_speaker] = global_id
                used_global_ids_in_chunk.add(global_id)  # Mark this new global ID as used
                global_centroids[global_id] = emb
                global_counts[global_id] = 1 if emb is not None else 0

        remapped_df = df.copy()
        remapped_df["speaker"] = remapped_df["speaker"].map(mapping)
        aligned_frames.append(remapped_df)

    return aligned_frames


def prepare_diarization_chunks(
    audio_path,
    audio_info,
    max_duration=MAX_DIA_CHUNK_DURATION,
    min_silence=MIN_SPLIT_SILENCE,
):
    """
    Split long audio files prior to diarization using silence from VAD.
    Returns chunk metadata and optional temp directory for cleanup.
    """
    waveform = audio_info["waveform"]
    sample_rate = audio_info["sample_rate"]
    total_duration, silence_intervals = _build_silence_intervals(
        waveform, sample_rate, min_silence
    )
    chunk_ranges = _build_chunk_ranges(total_duration, silence_intervals, max_duration)

    epsilon = 1e-3
    normalized_audio = audio_info.get("audio_segment")

    if (
        len(chunk_ranges) == 1
        and chunk_ranges[0][0] <= epsilon
        and abs(chunk_ranges[0][1] - total_duration) <= epsilon
    ):
        # Single chunk case: use normalized_audio if available, otherwise create temp mono file
        if normalized_audio is not None:
            # Create a temporary file with the normalized mono audio
            temp_dir = tempfile.mkdtemp(prefix="pre_diar_")
            temp_path = os.path.join(temp_dir, "full_audio.wav")
            normalized_audio.export(temp_path, format="wav", parameters=["-ac", "1"])
            return [{"path": temp_path, "offset": 0.0, "duration": total_duration}], temp_dir
        else:
            # Fallback: load and ensure mono
            temp_audio = AudioSegment.from_file(audio_path).set_channels(1)
            temp_dir = tempfile.mkdtemp(prefix="pre_diar_")
            temp_path = os.path.join(temp_dir, "full_audio.wav")
            temp_audio.export(temp_path, format="wav", parameters=["-ac", "1"])
            return [{"path": temp_path, "offset": 0.0, "duration": total_duration}], temp_dir


    if normalized_audio is None:
        normalized_audio = AudioSegment.from_file(audio_path)
        # Ensure mono audio for diarization
        normalized_audio = normalized_audio.set_channels(1)
    temp_dir = tempfile.mkdtemp(prefix="pre_diar_")
    chunk_entries = []

    for idx, (start_sec, end_sec) in enumerate(chunk_ranges):
        start_ms = max(0, int(round(start_sec * 1000)))
        end_ms = max(start_ms, int(round(end_sec * 1000)))
        chunk_audio = normalized_audio[start_ms:end_ms]
        chunk_path = os.path.join(temp_dir, f"chunk_{idx:03d}.wav")
        # Explicitly export as mono WAV with 24kHz sample rate
        chunk_audio.export(chunk_path, format="wav", parameters=["-ac", "1"])
        chunk_entries.append(
            {
                "path": chunk_path,
                "offset": start_sec,
                "duration": end_sec - start_sec,
            }
        )

    logger.info(
        f"Pre-diarization chunking created {len(chunk_entries)} chunks "
        f"(max {max_duration}s) from {os.path.basename(audio_path)}"
    )
    return chunk_entries, temp_dir

def ko_transliterate_english(text: str) -> str:
    """
    Find English segments in the input string and convert them to Korean pronunciation.
    """
    def _repl(m: re.Match) -> str:
        segment = m.group(0)
        return G2P(segment)
    return ENG_PATTERN.sub(_repl, text)


def ko_process_json(input_list: str) -> None:
    for entry in input_list:
        text = entry.get("text", "")
        # Convert if English is included
        if re.search(r"[A-Za-z]", text):
            entry["text"] = ko_transliterate_english(text)

def export_segments_with_enhanced_audio(audio_info, segment_list, save_dir, audio_name):
    """
    Export segments to MP3 files.
    If 'enhanced_audio' exists in the segment (processed by SepReformer), use it.
    Otherwise, slice from the original audio.
    """
    import os
    from pydub import AudioSegment as PydubAudioSegment
    
    # Create folder for saving segments
    segments_dir = os.path.join(save_dir, audio_name)
    os.makedirs(segments_dir, exist_ok=True)
    
    # Original full audio (Pydub object)
    full_audio_segment = audio_info.get("audio_segment")
    sample_rate = audio_info["sample_rate"]
    
    if full_audio_segment is None:
        # If audio_segment is unavailable, create from waveform (fallback)
        waveform_int16 = (audio_info["waveform"] * 32767).astype(np.int16)
        full_audio_segment = PydubAudioSegment(
            waveform_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

    logger.info(f"Exporting {len(segment_list)} segments with enhanced audio check...")

    for i, seg in enumerate(segment_list):
        # Generate filename (e.g.: 00001_SPEAKER_01.mp3)
        idx_str = seg.get("index", f"{i:05d}")
        spk = seg.get("speaker", "Unknown")
        filename = f"{idx_str}_{spk}.mp3"
        file_path = os.path.join(segments_dir, filename)

        # 1. Check for SepReformer-processed 'enhanced_audio'
        if seg.get("is_separated", False) and "enhanced_audio" in seg:
            # Convert Numpy array -> Pydub AudioSegment
            enhanced_waveform = seg["enhanced_audio"]

            # Convert float32 (-1.0 ~ 1.0) -> int16 range
            # Apply clipping to prevent overflow
            enhanced_waveform = np.clip(enhanced_waveform, -1.0, 1.0)
            wav_int16 = (enhanced_waveform * 32767).astype(np.int16)

            target_segment = PydubAudioSegment(
                wav_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            # logger.debug(f"Segment {idx_str}: Saved using SepReformer output.")

        else:
            # 3. If neither applied, extract from original
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            target_segment = full_audio_segment[start_ms:end_ms]

        # Save as MP3
        target_segment.export(file_path, format="mp3")
        
def main_process(audio_path, save_path=None, audio_name=None,
                 do_vad = False,
                 LLM = "",
                 use_demucs = False,
                 use_sepreformer = False,
                 overlap_threshold = 1.0,
                 sepreformer_separator = None,
                 embedding_model = None,
                 panns_model = None,
                 speaker_embedder = None,
                 speaker_link_threshold: float = 0.75):

    
    proc_audio_path = audio_path
    opus_temp_dir = None
    try:
        target_sr = int(cfg["entrypoint"]["SAMPLE_RATE"])
        proc_audio_path, opus_temp_dir = convert_opus_to_wav_if_needed(
            audio_path, target_sr=target_sr, logger=logger
        )

        if not proc_audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac", ".opus")):
            logger.warning(f"Unsupported file type: {proc_audio_path}")

        # for a single audio from path Ïaaa/bbb/ccc.wav ---> save to aaa/bbb_processed/ccc/ccc_0.wav
        audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
        suffix = "dia3" if args.dia3 else "ori"
        save_path = save_path or os.path.join(
            os.path.dirname(audio_path), "_final", f"-sepreformer-{args.sepreformer}" +f"-demucs-{args.demucs}"  + f"-vad-{do_vad}"+ f"-diaModel-{suffix}"
            # initial prompt off or on
            + f"-initPrompt-{args.initprompt}"
            + f"-merge_gap-{args.merge_gap}" +f"-seg_th-{args.seg_th}"+ f"-cl_min-{args.min_cluster_size}" +f"-cl-th-{args.clust_th}"+ f"-LLM-{LLM}", audio_name
        )
        os.makedirs(save_path, exist_ok=True)
        logger.debug(
            f"Processing audio: {audio_name}, from {audio_path}, save to: {save_path}"
        )

        logger.info(
            "Step 0: Preprocess all audio files --> 24k sample rate + wave format + loudnorm + bit depth 16"
        )
        audio = standardization(audio_path)
        diar_chunks, temp_chunk_dir = prepare_diarization_chunks(audio_path, audio)

        # Calculate total audio duration
        audio_duration = len(audio["waveform"]) / audio["sample_rate"]
        logger.info(f"Total audio duration: {audio_duration:.2f} seconds")

        logger.info("Step 2: Speaker Diarization")
        dia_start = time.time()


        diarization_frames = []
        try:
            for chunk in diar_chunks:
                if args.dia3:
                    import torchaudio
                    waveform, sr = torchaudio.load(chunk["path"])
                    waveform = waveform.to(device)
                    segments = dia_pipeline({"waveform": waveform, "sample_rate": sr})
                    data = []
                    for turn, _, speaker in (segments.speaker_diarization if hasattr(segments, "speaker_diarization") else segments).itertracks(yield_label=True):
                        data.append({"speaker": speaker, "start": turn.start, "end": turn.end})
                    import pandas as pd
                    chunk_df = pd.DataFrame(data) if data else pd.DataFrame(columns=["speaker", "start", "end"])
                else:
                    predicted_segments, _ = diar_model.diarize(
                        audio=chunk["path"], batch_size=1, include_tensor_outputs=True
                    )
                    chunk_df = sortformer_dia(predicted_segments)
                if not chunk_df.empty:
                    chunk_df["start"] += chunk["offset"]
                    chunk_df["end"] += chunk["offset"]
                    chunk_df = _apply_sortformer_segment_padding_from_args(
                        chunk_df, args=args, logger=logger, audio_duration=audio_duration
                    )
                diarization_frames.append(chunk_df)
        finally:
            if temp_chunk_dir:
                shutil.rmtree(temp_chunk_dir, ignore_errors=True)

        if diarization_frames:
            diarization_frames = align_speakers_across_chunks(
                diarization_frames,
                audio_info=audio,
                embedder=speaker_embedder,
                similarity_threshold=speaker_link_threshold,
            )

        if diarization_frames:
            speakerdia = pd.concat(diarization_frames, ignore_index=True)
        else:
            speakerdia = pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])
        ori_list = df_to_list(speakerdia)
        dia_end = time.time()

        # Calculate VAD + Sortformer RT factor
        vad_sortformer_processing_time = dia_end - dia_start
        vad_sortformer_rt = vad_sortformer_processing_time / audio_duration if audio_duration > 0 else 0
        logger.info(f"VAD + Sortformer - Processing time: {vad_sortformer_processing_time:.2f}s, RT factor: {vad_sortformer_rt:.4f}")

        # TEST
        ######################
        segment_list = ori_list
        segment_list = split_long_segments(segment_list)
        ######################

        # [Fixed] Execute Step 3 before Step 2.5!
        # Step 3: Background Music Detection and Removal
        # Clean the full audio first before running SepReformer.
        logger.info("Step 3: Background Music Detection and Removal")
        # Add padding to cover ASR timestamp error margin
        audio, segment_demucs_flags = preprocess_segments_with_demucs(segment_list, audio, panns_model=panns_model, use_demucs=use_demucs, padding=0.5)

        # [Fixed] Now run SepReformer with the cleaned audio
        # Step 2.5: Overlap control using SepReformer
        logger.info("Step 2.5: Overlap Control with SepReformer")
        separation_time = 0.0
        if use_sepreformer and sepreformer_separator is not None and embedding_model is not None:
            separation_start = time.time()
            # At this point, audio has already been processed by Demucs.
            audio, segment_list = process_overlapping_segments_with_separation(
                segment_list,
                audio,
                overlap_threshold=overlap_threshold,
                separator=sepreformer_separator,
                embedding_model=embedding_model
            )
            separation_end = time.time()
            separation_time = separation_end - separation_start

            # Calculate SepReformer RT factor
            separation_rt = separation_time / audio_duration if audio_duration > 0 else 0
            logger.info(f"SepReformer separation - Processing time: {separation_time:.2f}s, RT factor: {separation_rt:.4f}")
        else:
            logger.info("SepReformer overlap separation skipped (flag disabled)")
            
        logger.info("Step 4: ASR (Automatic Speech Recognition)")
        if args.ASRMoE:
            asr_start = time.time()

            asr_result, whisper_time, alignment_time = asr_MoE(
                segment_list,
                audio,
                segment_demucs_flags=segment_demucs_flags,
                enable_word_timestamps=args.whisperx_word_timestamps,
                device=device_name,
                lang=args.lang,
            )

            asr_end = time.time()

            dia_time = dia_end-dia_start
            asr_time = whisper_time
        else:
            asr_start = time.time()

            asr_result = asr(segment_list, audio)

            asr_end = time.time()

            dia_time = dia_end-dia_start
            asr_time = asr_end-asr_start
            alignment_time = 0.0

        # Calculate Whisper large v3 RT factor
        whisper_processing_time = asr_time
        whisper_rt = whisper_processing_time / audio_duration if audio_duration > 0 else 0

        # Calculate WhisperX alignment RT factor
        alignment_rt = alignment_time / audio_duration if audio_duration > 0 else 0

        if LLM == "case_0":
            print("LLM case_0")
            filtered_list = asr_result
            print(f"ASR result contains {len(filtered_list)} segments")




        # LLM post diarization start
        ####################################################################################################
        # "LLM post-processing"
        elif LLM == "case_2":
            print(f"asr_result len: {len(asr_result)}")
            print("Warning: llm_inference functions are commented out. Using ASR results directly.")
            filtered_list = asr_result
            

        else:
            raise ValueError("LLM variable must be one of case_0, case_1, case_2.")

        ############################################################################################################
            # LLM post diarization end

        # Step 4.5: Add Qwen3-Omni captions (if enabled)
        caption_time = 0.0
        if args.qwen3omni:
            logger.info("Step 4.5: Adding Qwen3-Omni captions")
            filtered_list, caption_time = add_qwen3omni_caption(filtered_list, audio, save_path)
        else:
            logger.info("Step 4.5: Qwen3-Omni caption generation skipped (flag disabled)")

        # Calculate Qwen3-Omni RT factor
        caption_rt = caption_time / audio_duration if audio_duration > 0 else 0

        # Print all timing information
        print(f"\n{'='*60}")
        print(f"Audio duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)")
        print(f"{'='*60}")
        print(f"VAD + Sortformer:")
        print(f"  - Processing time: {dia_time:.2f} seconds")
        print(f"  - RT factor: {vad_sortformer_rt:.4f}")
        print(f"{'='*60}")
        if use_sepreformer:
            print(f"SepReformer Overlap Separation:")
            print(f"  - Processing time: {separation_time:.2f} seconds")
            print(f"  - RT factor: {separation_rt:.4f}")
            print(f"{'='*60}")
        print(f"Whisper large v3:")
        print(f"  - Processing time: {asr_time:.2f} seconds")
        print(f"  - RT factor: {whisper_rt:.4f}")
        print(f"{'='*60}")
        if args.whisperx_word_timestamps:
            print(f"WhisperX Alignment:")
            print(f"  - Processing time: {alignment_time:.2f} seconds")
            print(f"  - RT factor: {alignment_rt:.4f}")
            print(f"{'='*60}")
        if args.qwen3omni:
            print(f"Qwen3-Omni Caption:")
            print(f"  - Processing time: {caption_time:.2f} seconds")
            print(f"  - RT factor: {caption_rt:.4f}")
            print(f"{'='*60}")
        print()

        logger.info("Step 5: Write result into MP3 and JSON file")
        print(f"Exporting {len(filtered_list)} segments to MP3 and JSON...")
        export_segments_with_enhanced_audio(audio, filtered_list, save_path, audio_name)

        # Korean G2P post-processing
        if args.korean:
            ko_process_json(filtered_list)

        cleaned_list = []
        for item in filtered_list:
            # Use shallow copy to avoid modifying the original filtered_list
            clean_item = item.copy()
            
            # 1. Remove 'enhanced_audio' (audio matrix) key if present
            if "enhanced_audio" in clean_item:
                del clean_item["enhanced_audio"]
            # 2. (Error prevention) Convert Numpy float/int types to Python native types
            for k, v in clean_item.items():
                if hasattr(v, 'item'):  # If numpy type
                    clean_item[k] = v.item()
                    
            cleaned_list.append(clean_item)

        # Prepare output with RT factor metrics
        output_data = {
            "metadata": {
                "audio_duration_seconds": audio_duration,
                "audio_duration_minutes": audio_duration / 60,
                "vad_sortformer": {
                    "processing_time_seconds": vad_sortformer_processing_time,
                    "rt_factor": vad_sortformer_rt
                },
                "whisper_large_v3": {
                    "processing_time_seconds": whisper_processing_time,
                    "rt_factor": whisper_rt
                },
                # [Fixed] Use cleaned_list length instead of filtered_list
                "total_segments": len(cleaned_list)
            },
            # [Fixed] Must use cleaned_list instead of filtered_list here.
            "segments": cleaned_list  
        }

        # Add WhisperX alignment metadata if enabled
        if args.whisperx_word_timestamps:
            output_data["metadata"]["whisperx_alignment"] = {
                "processing_time_seconds": alignment_time,
                "rt_factor": alignment_rt,
                "enabled": True
            }

        # Add Qwen3-Omni caption metadata if enabled
        if args.qwen3omni:
            output_data["metadata"]["qwen3omni_caption"] = {
                "processing_time_seconds": caption_time,
                "rt_factor": caption_rt,
                "enabled": True
            }

        # Add SepReformer separation metadata if enabled
        if use_sepreformer:
            output_data["metadata"]["sepreformer_separation"] = {
                "processing_time_seconds": separation_time,
                "rt_factor": separation_rt,
                "overlap_threshold_seconds": overlap_threshold,
                "enabled": True
            }

        final_path = os.path.join(save_path, audio_name + ".json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"All done, Saved to: {final_path}")
        print(f"Processing complete! Results saved to: {final_path}")
        print(f"Total segments processed: {len(filtered_list)}")
        return final_path, filtered_list
    finally:
        if opus_temp_dir:
            shutil.rmtree(opus_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="",
        help="input folder path, this will override config if set",
    )
    parser.add_argument(
        "--config_path", type=str, default="config.json", help="config path"
    )
    
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    # 32 seems reasonable

    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="The compute type to use for the model",
    )
    parser.add_argument(
        "--whisper_arch",
        type=str,
        default="large-v3-turbo",
        help="The name of the Whisper model to load.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="The number of CPU threads to use per worker, e.g. will be multiplied by num workers.",
    )
    parser.add_argument(
        "--exit_pipeline",
        type=bool,
        default=False,
        help="Exit pipeline when task done.",
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turning on vad.",
    )
    parser.add_argument(
        "--dia3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turning on diaization 3.0 model",
    )
    parser.add_argument(
        "--LLM",
        type=str,
        default="case_2",
        help="LLM diarization cases",
    )

    # hyperparameter
    parser.add_argument(
        "--seg_th",
        type=float,
        default=0.15,
        help="diarization model segmentation threshold",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=10,
        help="diarization model clustering min_cluster_size",
    )
    parser.add_argument(
        "--clust_th",
        type=float,
        default=0.5,
        help="diarization model clustering threshold",
    )

    parser.add_argument(
        "--merge_gap",
        type=float,
        default=2,
        help="merge gap in seconds, if smaller than this, merge",
    )
    parser.add_argument(
        "--speaker-link-threshold",
        type=float,
        default=0.49,
        help="Cosine similarity threshold for linking speakers across diarization chunks",
    )

    parser.add_argument(
        "--initprompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Turning on initial prompt on whisper model",
    )
    parser.add_argument(
        "--korean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="korean_g2p",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "vi"],
        help="Pipeline language. 'en' uses Whisper+Parakeet+Canary MoE. 'vi' swaps in Whisper+PhoWhisper+ChunkFormer.",
    )

    parser.add_argument(
        "--ASRMoE",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="parakeet",
    )

    parser.add_argument(
        "--demucs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable background music detection and removal using PANNs and Demucs",
    )

    parser.add_argument(
        "--whisperx_word_timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable WhisperX word-level timestamps with alignment",
    )

    parser.add_argument(
        "--qwen3omni",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3-Omni audio captioning for each segment",
    )

    parser.add_argument(
        "--sepreformer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable SepReformer for overlapping speech separation",
    )

    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=1.0,
        help="Minimum overlap duration in seconds to trigger SepReformer separation",
    )

    # Sortformer diarization segment boundary adjustment (optional)
    parser.add_argument(
        "--sortformer-param",
        "--sortformerParam",
        "--sortformerParma",
        dest="sortformer_param",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Sortformer segment boundary adjustment (pad_offset/pad_onset applied after model output).",
    )
    parser.add_argument( "--sortformer-pad-offset", 
                        type=float, 
                        default=-0.24, 
                        help="Seconds to add to segment end time (negative pulls ends earlier). Used with --sortformer-param.", )
    
    parser.add_argument( "--sortformer-pad-onset", 
                        type=float, 
                        default=0.0, 
                        help="Seconds to add to segment start time (negative pulls starts earlier). Used with --sortformer-param.", )
    
    parser.add_argument(
    "--opus_decode_workers",
    type=int,
    default=8,
    help="Number of parallel workers for opus/ogg -> wav decoding pre-stage.",
    )
    parser.add_argument(
        "--ffmpeg_threads_per_decode",
        type=int,
        default=1,
        help="ffmpeg -threads per decode process (keep small when using many workers).",
    )

    args = parser.parse_args()

    batch_size = args.batch_size
    cfg = load_cfg(args.config_path)

    logger = Logger.get_logger()

    if args.input_folder_path:
        logger.info(f"Using input folder path: {args.input_folder_path}")
        cfg["entrypoint"]["input_folder_path"] = args.input_folder_path

    logger.debug("Loading models...")

    # Load models
    if detect_gpu():
        logger.info("Using GPU")
        n_gpus = torch.cuda.device_count()
        # GPU 0: Whisper (slot 1) + Diarization + VAD + PANNs
        device_name = "cuda"  # CTranslate2/faster-whisper requires "cuda", not "cuda:0"
        device = torch.device("cuda:0")
        # GPU 1: PhoWhisper (slot 2) + Qwen3-ASR (slot 3) + Sortformer
        if n_gpus >= 2:
            device_2 = torch.device("cuda:1")
            logger.info(f"Dual-GPU mode: GPU 0 (Whisper/Diar/VAD) + GPU 1 (PhoWhisper/Qwen3/Sortformer)")
        else:
            device_2 = device
            logger.info("Single-GPU mode: all models on cuda:0")
    else:
        logger.info("Using CPU")
        device_name = "cpu"
        device = torch.device(device_name)
        device_2 = device
        # whisperX expects compute type: int8
        logger.info("Overriding the compute type to int8")
        args.compute_type = "int8"

    check_env(logger)

    # Speaker Diarization
    logger.debug(" * Loading Speaker Diarization Model")
    if not cfg["huggingface_token"].startswith("hf"):
        raise ValueError(
            "huggingface_token must start with 'hf', check the config file. "
            "You can get the token at https://huggingface.co/settings/tokens. "
            "Remeber grant access following https://github.com/pyannote/pyannote-audio?tab=readme-ov-file#tldr"
        )
    if args.dia3 == True:
        print("Using diarization-3.1 model")
        dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        #"pyannote/speaker-diarization",
        token=cfg["huggingface_token"],


    )
        dia_pipeline.to(device)
        
    else:
        dia_pipeline = Pipeline.from_pretrained(
            #"pyannote/speaker-diarization-community-1",
            "pyannote/speaker-diarization",
            token=cfg["huggingface_token"]
        )
        dia_pipeline.to(device)

        # hyperparameters
        dia_pipeline.instantiate({
        "segmentation": {
            "min_duration_off": 0.0, 
            "threshold": args.seg_th
        },
        "clustering": {
            "method": "centroid",
            "min_cluster_size": args.min_cluster_size,
            "threshold": args.clust_th   
        }
    })
    # ASR
    logger.debug(" * Loading ASR Model")

    if args.initprompt == True:
        # initial_prompt seeds Whisper's decoder context to anchor transcription style
        # (keeps disfluencies/backchannels rather than stripping them like training subs do).
        # Keep prompt monolingual: cross-script tokens can pull the language head off VN.
        if args.lang == "vi":
            initial_prompt_text = (
                "Ừ. Dạ. Vâng. À. Ờ. Ừm. Đúng rồi. Vậy hả. Thế à. Ok. Ừ ha. Ờ ha. À ha. Ừ nhỉ."
            )
        else:
            initial_prompt_text = (
                "Um. Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? "
                "okay. Alright. Emm. Mm. So. Oh. Hoo hoo."
                "生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!"
                "微斯人,吾谁与归?ええと、あの、ま、そう、ええ。"
                "äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, "
                "eh bien, du coup. genre, comme, style. 응,어,그,음."
            )
        asr_options_dict = {
            "initial_prompt": initial_prompt_text,
        }
        # Add word_timestamps if flag is enabled
        if args.whisperx_word_timestamps:
            asr_options_dict["word_timestamps"] = True

        asr_model = whisper_asr.load_asr_model(
            "large-v3-turbo",
            device_name,
            compute_type=args.compute_type,
            threads=args.threads,
            language=args.lang,

        # ASR model options can be modified via default_asr_options in whisper_asr.py.

            asr_options=asr_options_dict,
        )
    else:
        asr_options_dict = {}
        # Add word_timestamps if flag is enabled
        if args.whisperx_word_timestamps:
            asr_options_dict["word_timestamps"] = True

        asr_model = whisper_asr.load_asr_model(
            "large-v3-turbo",
            device_name,
            compute_type=args.compute_type,
            threads=args.threads,

            language=args.lang,
            asr_options=asr_options_dict if asr_options_dict else None,

            )
    if args.ASRMoE:
        if args.lang == "vi":
            # VN MoE: replace Parakeet with PhoWhisper-large and Canary with ChunkFormer-CTC.
            # Variable names asr_model_2 / canary_model are reused as the second/third ASR slots.
            from transformers import pipeline as hf_asr_pipeline
            # from chunkformer import ChunkFormerModel

            logger.debug(f" * Loading PhoWhisper-large (VN, slot 2) on {device_2}")
            asr_model_2 = hf_asr_pipeline(
                "automatic-speech-recognition",
                model="vinai/PhoWhisper-large",
                device=device_2,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

            # Qwen3-ASR runs in a separate environment (qwen3_env) via subprocess worker
            # to avoid huggingface-hub dependency conflict with WhisperX
            import subprocess as _sp
            qwen3_env_bin = os.environ.get(
                "QWEN3_PYTHON",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "qwen3_env", "bin", "python"),
            )
            qwen3_worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_worker.py")
            qwen3_env_for_worker = os.environ.copy()
            qwen3_env_for_worker["CUDA_VISIBLE_DEVICES"] = "1"  # Physical GPU 1 → cuda:0 inside worker

            logger.debug(f" * Starting Qwen3-ASR worker subprocess: {qwen3_env_bin}")
            qwen3_process = _sp.Popen(
                [qwen3_env_bin, qwen3_worker_script],
                stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE,
                text=True, bufsize=1,
                env=qwen3_env_for_worker,
            )

            # Wait for worker to become ready
            canary_model = None  # No in-process model; inference via qwen3_process
            _qwen3_ready = False
            for _startup_line in iter(qwen3_process.stdout.readline, ""):
                _startup_line = _startup_line.strip()
                if not _startup_line:
                    continue
                try:
                    _msg = json.loads(_startup_line)
                    logger.debug(f" * Qwen3 worker: {_msg}")
                    if _msg.get("status") == "ready":
                        _qwen3_ready = True
                        break
                except json.JSONDecodeError:
                    logger.debug(f" * Qwen3 worker (raw): {_startup_line}")
            if not _qwen3_ready:
                logger.error(" * Qwen3-ASR worker failed to start!")
                _qwen3_stderr = qwen3_process.stderr.read()
                if _qwen3_stderr:
                    logger.error(f" * Qwen3 worker stderr: {_qwen3_stderr[:2000]}")
            else:
                logger.debug(f" * PhoWhisper + Qwen3-ASR worker loaded successfully")
        else:
            import nemo.collections.asr as nemo_asr
            asr_model_2 = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

            # Load Canary model
            logger.debug(" * Loading Canary Model")
            canary_model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
            canary_model = canary_model.to(device)
            canary_model.eval()
            logger.debug(f" * Canary model loaded on {device}")
        # Client initialization
    #client = OpenAI(api_key="YOUR_API_KEY")
    model_name = "gpt-4.1"
    # VAD
    logger.debug(" * Loading VAD Model")
    vad = silero_vad.SileroVAD(device=device)
    
    # Initialize G2P instance
    G2P = G2p()

    # English segment detection pattern: word groups connected by consecutive alphabets, apostrophes, and spaces
    ENG_PATTERN = re.compile(r"[A-Za-z][A-Za-z']*(?: [A-Za-z][A-Za-z']*)*")

    speaker_embedder = None
    try:
        from pyannote.audio import Model
        emb_model = Model.from_pretrained("pyannote/embedding", token=cfg["huggingface_token"])
        speaker_embedder = Inference(
            emb_model,
            device=device,
            window="whole",
        )
        logger.debug(" * Speaker embedding model loaded for cross-chunk linking")
    except Exception as e:
        logger.error(f" * Failed to load speaker embedding model: {e}")
        speaker_embedder = None

    # load model from Hugging Face model card directly (You need a Hugging Face token)
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
    diar_model = diar_model.to(device_2)
    diar_model.eval()
    logger.debug(f" * Sortformer loaded on {device_2}")

    # Initialize Pyannote embedding model (only when sepreformer is enabled)
    embedding_model = None
    if args.sepreformer:
        logger.debug(" * Loading Pyannote Embedding Model")
        try:
            from pyannote.audio import Model as PyannoteModel
            embedding_model = PyannoteModel.from_pretrained("pyannote/embedding", token=cfg["huggingface_token"])
            embedding_model = embedding_model.to(device)
            logger.debug(" * Pyannote Embedding Model loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load Pyannote Embedding Model: {e}")
            embedding_model = None

    # Initialize SepReformer separator (only when sepreformer is enabled)
    sepreformer_separator = None
    if args.sepreformer:
        logger.debug(" * Loading SepReformer Separator Model")
        try:
            sepreformer_separator = SepReformerSeparator(
                sepreformer_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SepReformer"),
                device=device
            )
            logger.debug(" * SepReformer Separator loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load SepReformer Separator: {e}")
            sepreformer_separator = None

    # Initialize PANNs model (for background music detection)
    panns_model = None
    if args.demucs:
        logger.debug(" * Loading PANNs Model for background music detection")
        try:
            panns_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "panns_data")
            os.makedirs(panns_data_dir, exist_ok=True)
            os.environ['PANNS_DATA'] = panns_data_dir
            checkpoint_path = os.path.join(panns_data_dir, 'Cnn14_mAP=0.431.pth')
            panns_model = AudioTagging(checkpoint_path=checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            logger.debug(" * PANNs Model loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load PANNs Model: {e}")
            panns_model = None

    logger.debug("All models loaded")

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    logger.debug(f"supported languages multilingual {supported_languages}")
    logger.debug(f"using multilingual asr {multilingual_flag}")

    input_folder_path = cfg["entrypoint"]["input_folder_path"]

    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"input_folder_path: {input_folder_path} not found")

    # Get only audio files in the specified directory (not recursive)
    audio_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.opus', '.ogg')
    audio_paths = []
    
    for file in os.listdir(input_folder_path):
        # 1. Check for supported extensions
        if not file.lower().endswith(audio_extensions):
            continue
            
        # 2. Exclude .temp files
        if ".temp" in file:
            continue
            
        # 3. [Important] Prevent listing files from cache folders (_opus_cache...)
        # (os.listdir only shows direct children, so folders won't be listed as files,
        # but this handles the case where input_folder_path itself is a cache folder)
        if "_opus_cache" in input_folder_path:
            logger.warning(f"Skipping execution because input path looks like a cache dir: {input_folder_path}")
            sys.exit(0)

        audio_paths.append(os.path.join(input_folder_path, file))

    logger.debug(f"Scanning {len(audio_paths)} audio files in {input_folder_path} (non-recursive)")

    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # (right after creating audio_paths)
    target_sr = int(cfg["entrypoint"]["SAMPLE_RATE"])
    opus_cache_dir = os.path.join(input_folder_path, f"_opus_cache_wav_{target_sr}")

    # Filter opus/ogg files for parallel decoding
    to_decode = [p for p in audio_paths if p.lower().endswith((".opus", ".ogg"))]
    if to_decode:
        logger.info(f"[OPUS] Pre-decoding {len(to_decode)} files with {args.opus_decode_workers} workers...")
        decoded_map = {}

        def _job(p):
            return p, convert_opus_to_wav_cached(
                p,
                target_sr=target_sr,
                cache_dir=opus_cache_dir,
                logger=logger,
                ffmpeg_threads=args.ffmpeg_threads_per_decode,
            )

        with ThreadPoolExecutor(max_workers=int(args.opus_decode_workers)) as ex:
            futures = [ex.submit(_job, p) for p in to_decode]
            
            for fut in as_completed(futures):
                try:
                    src, dst = fut.result()
                    decoded_map[src] = dst
                except Exception as e:
                    # [Important fix] Handle exceptions so a single file failure doesn't kill the entire process
                    logger.error(f"❌ [OPUS] Failed to decode file, skipping: {e}")
                    # Failed files are not added to the map, so they will be removed from audio_paths

        # Replace audio_paths with "converted wav paths"
        # Swap if in decoded_map (success), exclude from list if .opus but not in map (failed)
        new_audio_paths = []
        for p in audio_paths:
            if p in decoded_map:
                new_audio_paths.append(decoded_map[p])
            elif p.lower().endswith((".opus", ".ogg")):
                # Opus file not in decoding map means it failed, so exclude it
                logger.warning(f"Skipping failed file: {p}")
            else:
                # Keep other files (wav, mp3, etc.) as-is
                new_audio_paths.append(p)
        
        audio_paths = new_audio_paths
        logger.info(f"[OPUS] Pre-decoding done. Valid files: {len(audio_paths)}")
    else:
        logger.info("[OPUS] No opus/ogg files found; skipping pre-decoding.")

    start_time = time.time()
    start_time = time.time()
    
    # [Fixed] Count successful file processing
    success_count = 0
    fail_count = 0

    for path in audio_paths:
        try:
            # 1. File size check: skip if less than 1KB as it's likely a corrupted file
            if os.path.getsize(path) < 1024:
                logger.warning(f"⚠️ [Skip] File too small ({os.path.getsize(path)} bytes): {path}")
                fail_count += 1
                continue

            # 2. Execute main process (with per-file try-except handling)
            main_process(path, do_vad=args.vad, LLM=args.LLM, use_demucs=args.demucs,
                         use_sepreformer=args.sepreformer, overlap_threshold=args.overlap_threshold,
                         sepreformer_separator=sepreformer_separator,
                         embedding_model=embedding_model, panns_model=panns_model,
                         speaker_embedder=speaker_embedder,
                         speaker_link_threshold=args.speaker_link_threshold)
            
            success_count += 1

        except Exception as e:
            # [Important] Don't stop on single file failure; log error and continue to next file
            logger.error(f"❌ [Error] Failed to process file: {path}")
            logger.error(f"   Reason: {e}")
            fail_count += 1
            continue

    end_time = time.time()
    print(f"Directory processing finished.")
    print(f" - Success: {success_count}")
    print(f" - Failed: {fail_count}")
    print(f"Total time: {end_time - start_time:.2f}s")
