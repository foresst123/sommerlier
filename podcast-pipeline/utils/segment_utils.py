import numpy as np
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

def _quietest_cut(waveform, sample_rate, lo: float, hi: float,
                  frame_sec: float = 0.02) -> float:
    """Time in [lo, hi] where the audio is quietest, or None if unmeasurable.

    Used to place a split inside a search band rather than on a stopwatch. The
    quietest 20ms frame is where a pause is, and cutting there keeps a word
    whole; cutting at a fixed offset lands mid-syllable roughly as often as not.
    """
    if waveform is None or sample_rate is None or hi <= lo:
        return None
    i, j = int(lo * sample_rate), int(hi * sample_rate)
    i, j = max(0, i), min(len(waveform), j)
    if j - i < 2:
        return None

    band = waveform[i:j]
    frame = max(1, int(frame_sec * sample_rate))
    if len(band) < frame * 2:
        return None

    n = len(band) // frame
    rms = np.sqrt((band[:n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    return lo + (int(np.argmin(rms)) + 0.5) * frame / sample_rate


def split_long_segments(segment_list: list, max_duration: float = 30.0,
                        waveform=None, sample_rate: int = None,
                        search_sec: float = 2.0) -> list:
    """Break over-long segments, preferring a pause to the stopwatch.

    Without audio this splits exactly on max_duration, which cuts mid-word as
    often as not -- and a clipped word is a transcription error the recogniser
    has no way to recover from. Given the waveform, the cut moves to the
    quietest point within `search_sec` before the deadline, so the pieces end
    on a pause. The deadline is never exceeded: the search band sits entirely
    before it.
    """
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

                # Only look for a pause on a cut that is actually forced; the
                # final piece ends where the segment ends.
                if chunk_end < end_time and waveform is not None:
                    band_lo = max(current_start + max_duration - search_sec,
                                  current_start + 0.2)
                    quiet = _quietest_cut(waveform, sample_rate, band_lo, chunk_end)
                    # Guard against a degenerate band handing back a cut that
                    # would make no progress.
                    if quiet is not None and quiet > current_start + 0.2:
                        chunk_end = quiet

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


# A speaker holding less than this share of the spoken time is treated as a
# clustering artefact rather than a participant. Set from what these look like
# in practice: on a 41-minute two-person podcast the diarizer produced a third
# speaker with 3.2s across three segments -- 0.13% -- and every one of its
# segments sat inside another speaker's turn. A real third participant in a
# recording of that length holds whole turns, not fragments.
GHOST_SPEAKER_SHARE = 0.005

# Never dissolve a speaker on share alone: a genuine guest who says one thing
# in an hour is rare but real, and merging them corrupts the transcript rather
# than tidying it. Their segments must also be short enough to be fragments.
GHOST_SPEAKER_MAX_SEGMENT = 2.0


def merge_ghost_speakers(segment_list: list, share=GHOST_SPEAKER_SHARE,
                         max_segment=GHOST_SPEAKER_MAX_SEGMENT, logger=None) -> list:
    """Fold vanishingly small speakers into whoever is speaking around them.

    Diarization sometimes splits a handful of frames into a speaker of their
    own -- a cough, a laugh, one word landing between two turns. Downstream
    this is expensive out of proportion to its size: target extraction refuses
    any overlap whose window contains three speakers, and it cannot build an
    enrollment for someone with under 1.5s of clean audio, so a 3-second ghost
    blocked eleven of thirteen separable overlaps on one file here.

    Each ghost segment is relabelled to its nearest neighbour in time, which is
    the speaker whose turn it interrupted. Returns a new list; the input is not
    modified.
    """
    if not segment_list:
        return segment_list

    spoken = {}
    for seg in segment_list:
        spoken[seg["speaker"]] = spoken.get(seg["speaker"], 0.0) + (seg["end"] - seg["start"])
    total = sum(spoken.values())
    if total <= 0 or len(spoken) < 3:
        # With two speakers there is no third to dissolve, and the smaller of
        # the two is a participant however quiet.
        return segment_list

    ghosts = set()
    for speaker, held in spoken.items():
        if held / total >= share:
            continue
        longest = max((s["end"] - s["start"]) for s in segment_list
                      if s["speaker"] == speaker)
        if longest <= max_segment:
            ghosts.add(speaker)

    if not ghosts:
        return segment_list
    if len(set(spoken) - ghosts) < 1:
        # Everything looks like a ghost: the threshold is wrong for this file,
        # so change nothing rather than empty it.
        return segment_list

    ordered = sorted(segment_list, key=lambda s: s["start"])
    real = [s for s in ordered if s["speaker"] not in ghosts]
    if not real:
        return segment_list

    out, moved = [], 0
    for seg in ordered:
        if seg["speaker"] not in ghosts:
            out.append(seg)
            continue
        centre = (seg["start"] + seg["end"]) / 2.0
        nearest = min(real, key=lambda r: abs((r["start"] + r["end"]) / 2.0 - centre))
        merged = dict(seg)
        merged["speaker"] = nearest["speaker"]
        out.append(merged)
        moved += 1

    if logger:
        logger.info(
            f"Merged {len(ghosts)} ghost speaker(s) ({', '.join(sorted(ghosts))}) "
            f"into their neighbours: {moved} segment(s) relabelled")
    return sorted(out, key=lambda s: s["start"])
