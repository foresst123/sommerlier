import pandas as pd
import numpy as np
import torch
import librosa
from pyannote.audio import Inference
from utils.audio_math import cosine_similarity

def _extract_speaker_embedding(audio_info, start: float, end: float, embedder: Inference, sample_window: float = 2.0, min_duration: float = 0.5):
    if embedder is None: return None
    waveform = audio_info.get("waveform")
    sample_rate = audio_info.get("sample_rate")
    if waveform is None or sample_rate is None: return None

    total_duration = len(waveform) / sample_rate
    start = max(0.0, min(start, total_duration))
    end = max(start, min(end, total_duration))
    duration = end - start
    if duration < min_duration: return None

    if duration > sample_window:
        center = (start + end) / 2.0
        start = center - sample_window / 2.0
        end = center + sample_window / 2.0

    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    segment = waveform[start_idx:end_idx]
    if segment.size == 0: return None

    target_sr = getattr(embedder, "sample_rate", 16000)
    try:
        if sample_rate != target_sr:
            segment = librosa.resample(segment, orig_sr=sample_rate, target_sr=target_sr)
    except Exception:
        target_sr = sample_rate

    try:
        torch_seg = torch.as_tensor(segment, dtype=torch.float32).unsqueeze(0)
        emb = embedder({"waveform": torch_seg, "sample_rate": target_sr})
    except Exception:
        return None
    
    if emb is None: return None
    if isinstance(emb, torch.Tensor): emb = emb.detach().cpu().numpy()
    if isinstance(emb, np.ndarray) and emb.ndim > 1: emb = emb.mean(axis=0)
    return emb

def _compute_chunk_speaker_centroids(chunk_df: pd.DataFrame, audio_info, embedder: Inference):
    if embedder is None or chunk_df is None or chunk_df.empty: return {}
    centroids = {}
    for speaker, rows in chunk_df.groupby("speaker"):
        embeddings = []
        for _, row in rows.sort_values("start").iterrows():
            emb = _extract_speaker_embedding(audio_info, row["start"], row["end"], embedder=embedder)
            if emb is not None:
                embeddings.append(emb)
            if len(embeddings) >= 3:
                break
        if embeddings:
            centroids[speaker] = np.mean(embeddings, axis=0)
    return centroids

def align_speakers_across_chunks(chunk_frames: list, audio_info, embedder: Inference, similarity_threshold: float = 0.75, logger=None):
    if embedder is None or not chunk_frames:
        if logger: logger.warning("Speaker embedder unavailable; skipping cross-chunk speaker linking.")
        return chunk_frames

    global_centroids = {}
    global_counts = {}
    next_global_idx = 0
    aligned_frames = []

    for chunk_idx, df in enumerate(chunk_frames):
        if df is None or df.empty:
            aligned_frames.append(df)
            continue

        local_centroids = _compute_chunk_speaker_centroids(df, audio_info, embedder)
        mapping = {}
        used_global_ids_in_chunk = set()

        for local_speaker in df["speaker"].unique():
            emb = local_centroids.get(local_speaker)
            best_id = None
            best_sim = -1.0
            
            if emb is not None:
                for gid, centroid in global_centroids.items():
                    if centroid is None or gid in used_global_ids_in_chunk:
                        continue
                    sim = cosine_similarity(emb, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = gid

            if best_sim >= similarity_threshold and best_id is not None:
                mapping[local_speaker] = best_id
                used_global_ids_in_chunk.add(best_id)
                count = global_counts.get(best_id, 0)
                global_centroids[best_id] = (global_centroids[best_id] * count + emb) / (count + 1)
                global_counts[best_id] = count + 1
            else:
                global_id = f"SPEAKER_{next_global_idx:02d}"
                next_global_idx += 1
                mapping[local_speaker] = global_id
                used_global_ids_in_chunk.add(global_id)
                global_centroids[global_id] = emb
                global_counts[global_id] = 1 if emb is not None else 0

        remapped_df = df.copy()
        remapped_df["speaker"] = remapped_df["speaker"].map(mapping)
        aligned_frames.append(remapped_df)

    return aligned_frames
