import pandas as pd

def apply_sortformer_segment_padding(df: pd.DataFrame, pad_onset: float = 0.0, pad_offset: float = 0.0, audio_duration: float = None) -> pd.DataFrame:
    """Shift diarization segment boundaries (frame-level tweak)."""
    if df is None or df.empty:
        return df
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

def cut_by_speaker_label(vad_list: list, merge_gap: float = 0.5, min_segment_length: float = 0.2, max_segment_length: float = 30.0, logger=None) -> list:
    """Merge and trim VAD segments by speaker labels robustly."""
    if not vad_list:
        return []

    # Phase 1: Group by speaker and merge gaps
    speaker_tracks = {}
    for vad in vad_list:
        spk = vad["speaker"]
        if spk not in speaker_tracks:
            speaker_tracks[spk] = []
        speaker_tracks[spk].append(vad.copy())

    merged_list = []
    for spk, tracks in speaker_tracks.items():
        tracks.sort(key=lambda x: x["start"])
        spk_merged = []
        for vad in tracks:
            if not spk_merged:
                spk_merged.append(vad)
                continue
            
            last_vad = spk_merged[-1]
            gap = vad["start"] - last_vad["end"]
            merged_dur = max(last_vad["end"], vad["end"]) - min(last_vad["start"], vad["start"])
            
            if gap <= merge_gap and merged_dur <= max_segment_length:
                last_vad["end"] = max(last_vad["end"], vad["end"])
                last_vad["start"] = min(last_vad["start"], vad["start"])
            else:
                spk_merged.append(vad)
        merged_list.extend(spk_merged)

    # Re-sort all merged tracks by start time
    merged_list.sort(key=lambda x: x["start"])

    # Phase 2: Split any segment strictly larger than max_segment_length
    final_split = []
    for vad in merged_list:
        if vad["end"] - vad["start"] > max_segment_length:
            if logger:
                logger.warning(f"cut_by_speaker_label > segment longer than {max_segment_length}s, force trimming")
            curr_start = vad["start"]
            while curr_start < vad["end"]:
                chunk_end = min(curr_start + max_segment_length, vad["end"])
                new_vad = vad.copy()
                new_vad["start"] = curr_start
                new_vad["end"] = chunk_end
                final_split.append(new_vad)
                curr_start = chunk_end
        else:
            final_split.append(vad)

    # Phase 3: Filter out ultra-short segments
    filter_list = [v for v in final_split if (v["end"] - v["start"]) >= min_segment_length]
    return filter_list

def deduplicate_segments_by_index(segments: list, logger=None) -> list:
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

def split_long_segments(segment_list: list, max_duration: float = 30.0) -> list:
    new_segments = []
    new_index = 0
    for segment in segment_list:
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment['speaker']
        duration = end_time - start_time
        if duration <= max_duration:
            segment['index'] = str(new_index).zfill(5)
            new_segments.append(segment)
            new_index += 1
        else:
            current_start = start_time
            while current_start < end_time:
                chunk_end = min(current_start + max_duration, end_time)
                new_segments.append({
                    'index': str(new_index).zfill(5),
                    'start': round(current_start, 3),
                    'end': round(chunk_end, 3),
                    'speaker': speaker
                })
                new_index += 1
                current_start = chunk_end
    return new_segments

def df_to_list(df: pd.DataFrame) -> list:
    records = []
    for i, row in df.iterrows():
        records.append({
            'index': f"{i:05d}",
            'start': float(row['start']),
            'end': float(row['end']),
            'speaker': row['speaker']
        })
    return records

def build_silence_intervals(waveform, sample_rate, vad_model_func, min_silence=0.3):
    total_duration = len(waveform) / sample_rate
    if len(waveform) == 0:
        return 0.0, []
    
    # vad_model_func takes waveform and sample_rate and returns list of speech timestamps [{"start": ts, "end": ts}]
    speech_ts = vad_model_func(waveform, sample_rate)
    
    if not speech_ts:
        return total_duration, [(0.0, total_duration)]

    silence = []
    first_start = speech_ts[0]["start"]
    if first_start >= min_silence:
        silence.append((0.0, first_start))

    for prev_seg, next_seg in zip(speech_ts[:-1], speech_ts[1:]):
        sil_start = prev_seg["end"]
        sil_end = next_seg["start"]
        if sil_end - sil_start >= min_silence:
            silence.append((sil_start, sil_end))

    last_end = speech_ts[-1]["end"]
    trailing = total_duration - last_end
    if trailing >= min_silence:
        silence.append((last_end, last_end + trailing))
    return total_duration, silence

def build_chunk_ranges(total_duration, silence_intervals, max_duration):
    epsilon = 1e-3
    if total_duration <= max_duration + epsilon:
        return [(0.0, total_duration)]
    silence_points = sorted([(start + end) / 2.0 for start, end in silence_intervals])
    if not silence_points:
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
        limit = chunk_start + max_duration
        candidates = [p for p in silence_points if chunk_start + epsilon < p <= limit]
        if candidates:
            chunk_end = candidates[-1]
        else:
            future_candidates = [p for p in silence_points if p > limit]
            if future_candidates:
                chunk_end = min(limit, total_duration)
            else:
                chunk_end = total_duration

        if chunk_end - chunk_start < epsilon:
            chunk_end = min(chunk_start + max_duration, total_duration)
            if chunk_end - chunk_start < epsilon:
                break
        chunk_ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return chunk_ranges if chunk_ranges else [(0.0, total_duration)]
