import math
from typing import Optional

import numpy as np
import pandas as pd


GHOST_SPEAKER_SHARE = 0.005
GHOST_SPEAKER_MAX_SEGMENT = 2.0


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _overlap_duration(a_start: float, a_end: float,
                      b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _merge_time_ranges(ranges):
    """Merge positive overlapping/touching ranges."""
    merged = []
    for start, end in sorted(
        ((float(a), float(b)) for a, b in ranges if float(b) > float(a)),
        key=lambda x: x[0],
    ):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _union_duration(ranges) -> float:
    return sum(end - start for start, end in _merge_time_ranges(ranges))


def apply_sortformer_segment_padding(
    df: pd.DataFrame,
    pad_onset: float = 0.0,
    pad_offset: float = 0.0,
    audio_duration: float = None,
) -> pd.DataFrame:
    """Shift diarization boundaries safely.

    `pad_onset` and `pad_offset` are signed shifts, preserving the behaviour of
    the old function:
      start = start + pad_onset
      end   = end   + pad_offset

    Therefore a negative pad_onset expands a segment to the left, while a
    positive pad_offset expands it to the right.
    """
    if df is None or df.empty:
        return df

    if "start" not in df.columns or "end" not in df.columns:
        raise ValueError("DataFrame must contain 'start' and 'end' columns")

    if pad_onset == 0.0 and pad_offset == 0.0:
        return df

    df = df.copy()
    starts = pd.to_numeric(df["start"], errors="coerce")
    ends = pd.to_numeric(df["end"], errors="coerce")

    df["start"] = starts + float(pad_onset)
    df["end"] = ends + float(pad_offset)

    # Clamp BOTH boundaries. Clamping only end and then forcing end >= start can
    # re-create an end beyond audio_duration when start itself is out of range.
    if audio_duration is not None and float(audio_duration) > 0:
        duration = float(audio_duration)
        df["start"] = df["start"].clip(lower=0.0, upper=duration)
        df["end"] = df["end"].clip(lower=0.0, upper=duration)
    else:
        df["start"] = df["start"].clip(lower=0.0)
        df["end"] = df["end"].clip(lower=0.0)

    # Keep invalid/NaN rows as NaN rather than silently inventing timestamps.
    valid = df["start"].notna() & df["end"].notna()
    df.loc[valid, "end"] = np.maximum(
        df.loc[valid, "end"].to_numpy(),
        df.loc[valid, "start"].to_numpy(),
    )
    return df


def cut_by_speaker_label(
    vad_list: list,
    merge_gap: float = 0.5,
    min_segment_length: float = 0.2,
    max_segment_length: float = 30.0,
    logger=None,
    seams=None,
) -> list:
    """Merge nearby turns of the same speaker without swallowing another speaker.

    This function intentionally does NOT split over-long segments anymore.
    `split_long_segments()` owns that job so it can use the waveform to place a
    quiet cut instead of cutting twice with conflicting policies.

    `max_segment_length` is still used as a merge guard: merging two pieces is
    refused if their union would already exceed it.
    """
    if not vad_list:
        return []
    if merge_gap < 0:
        raise ValueError("merge_gap must be >= 0")
    if min_segment_length < 0:
        raise ValueError("min_segment_length must be >= 0")
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be > 0")

    joins = sorted(
        float(s) for s in (seams or ())
        if _is_finite_number(s)
    )

    clean_input = []
    for vad in vad_list:
        if not isinstance(vad, dict):
            continue
        if "speaker" not in vad or "start" not in vad or "end" not in vad:
            continue
        if not _is_finite_number(vad["start"]) or not _is_finite_number(vad["end"]):
            continue

        item = dict(vad)
        item["start"] = float(item["start"])
        item["end"] = float(item["end"])
        if item["end"] <= item["start"]:
            continue
        clean_input.append(item)

    if not clean_input:
        return []

    # Build merged tracks for all speakers so a same-speaker merge can check
    # whether someone else actually speaks inside the gap being bridged.
    by_speaker_ranges = {}
    for vad in clean_input:
        by_speaker_ranges.setdefault(vad["speaker"], []).append(
            (vad["start"], vad["end"])
        )
    by_speaker_ranges = {
        speaker: _merge_time_ranges(ranges)
        for speaker, ranges in by_speaker_ranges.items()
    }

    speaker_tracks = {}
    for vad in clean_input:
        speaker_tracks.setdefault(vad["speaker"], []).append(dict(vad))

    merged_list = []
    eps = 1e-9

    for speaker, tracks in speaker_tracks.items():
        tracks.sort(key=lambda x: (x["start"], x["end"]))
        spk_merged = []

        foreign_ranges = [
            interval
            for other_speaker, ranges in by_speaker_ranges.items()
            if other_speaker != speaker
            for interval in ranges
        ]

        for vad in tracks:
            if not spk_merged:
                spk_merged.append(dict(vad))
                continue

            last = spk_merged[-1]
            gap = vad["start"] - last["end"]
            merged_start = min(last["start"], vad["start"])
            merged_end = max(last["end"], vad["end"])
            merged_duration = merged_end - merged_start

            # A seam anywhere in the prospective union is enough to forbid a
            # merge. This also handles a seam exactly on either boundary.
            bridges_join = any(
                merged_start - eps <= join <= merged_end + eps
                for join in joins
            )

            # Only a positive gap can be "swallowed". If another speaker talks
            # in that gap, A ... B ... A must remain three turns, not one A turn.
            foreign_speech_in_gap = False
            if gap > eps:
                foreign_speech_in_gap = any(
                    _overlap_duration(last["end"], vad["start"], a, b) > eps
                    for a, b in foreign_ranges
                )

            can_merge = (
                gap <= merge_gap
                and merged_duration <= max_segment_length
                and not bridges_join
                and not foreign_speech_in_gap
            )

            if can_merge:
                last["start"] = merged_start
                last["end"] = merged_end
            else:
                spk_merged.append(dict(vad))

        merged_list.extend(spk_merged)

    merged_list.sort(key=lambda x: (x["start"], x["end"], str(x["speaker"])))

    # Filter only after merge. Long-segment splitting is intentionally deferred
    # to split_long_segments(), where an acoustic cut can be used.
    return [
        vad for vad in merged_list
        if vad["end"] - vad["start"] >= min_segment_length
    ]


def deduplicate_segments_by_index(segments: list, logger=None) -> list:
    seen = set()
    deduped = []
    for seg in segments or []:
        idx = seg.get("index") if isinstance(seg, dict) else None
        if idx is None or idx not in seen:
            if idx is not None:
                seen.add(idx)
            deduped.append(seg)
        elif logger:
            logger.warning(f"Duplicate segment index detected and skipped: {idx}")
    return deduped


def _quietest_cut(
    waveform,
    sample_rate,
    lo: float,
    hi: float,
    frame_sec: float = 0.02,
) -> Optional[float]:
    """Return the quietest frame centre inside [lo, hi]."""
    if waveform is None or sample_rate is None or sample_rate <= 0 or hi <= lo:
        return None

    i = max(0, int(float(lo) * sample_rate))
    j = min(len(waveform), int(float(hi) * sample_rate))
    if j - i < 2:
        return None

    # Cast before squaring so int16/int32 waveforms cannot overflow.
    band = np.asarray(waveform[i:j], dtype=np.float64)
    frame = max(1, int(float(frame_sec) * sample_rate))
    if len(band) < frame * 2:
        return None

    n = len(band) // frame
    frames = band[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)

    frame_index = int(np.argmin(rms))
    # Use the CLAMPED sample coordinate i, not the original lo. This matters
    # when a negative search boundary was clipped to the start of the waveform.
    return (i + (frame_index + 0.5) * frame) / float(sample_rate)


def split_long_segments(
    segment_list: list,
    max_duration: float = 30.0,
    waveform=None,
    sample_rate: int = None,
    search_sec: float = 2.0,
    min_piece: float = 0.2,
) -> list:
    """Split over-long segments once, preferring an acoustic pause.

    Metadata is preserved on every produced piece. If a strict 30-second split
    would leave a tiny final tail, the remaining span is rebalanced so the tail
    is not silently lost later by a minimum-duration filter.
    """
    if max_duration <= 0:
        raise ValueError("max_duration must be > 0")
    if search_sec < 0:
        raise ValueError("search_sec must be >= 0")
    if min_piece < 0:
        raise ValueError("min_piece must be >= 0")
    if waveform is not None and (sample_rate is None or sample_rate <= 0):
        raise ValueError("sample_rate must be > 0 when waveform is provided")

    new_segments = []
    new_index = 0

    for original in segment_list or []:
        if not isinstance(original, dict):
            continue
        if not _is_finite_number(original.get("start")) or not _is_finite_number(original.get("end")):
            continue

        start_time = float(original["start"])
        end_time = float(original["end"])
        if end_time <= start_time:
            continue

        current_start = start_time

        while end_time - current_start > max_duration:
            remaining = end_time - current_start
            deadline = current_start + max_duration

            # If cutting at the deadline would leave a tiny tail, split the
            # remaining span into two reasonable pieces instead of producing
            # e.g. 30.0s + 0.1s and then losing the 0.1s downstream.
            tail_after_deadline = end_time - deadline
            if 0 < tail_after_deadline < min_piece:
                desired = current_start + remaining / 2.0
                max_cut = deadline
                min_cut = current_start + min_piece
                target = min(max(desired, min_cut), max_cut)
            else:
                target = deadline

            chunk_end = target

            if waveform is not None and search_sec > 0:
                # Search before the chosen target so max_duration is never
                # exceeded. Also do not leave a sub-min_piece tail.
                latest_allowed = min(target, end_time - min_piece) if min_piece > 0 else target
                earliest_allowed = max(
                    current_start + min_piece,
                    latest_allowed - search_sec,
                )
                if latest_allowed > earliest_allowed:
                    quiet = _quietest_cut(
                        waveform,
                        sample_rate,
                        earliest_allowed,
                        latest_allowed,
                    )
                    if (
                        quiet is not None
                        and quiet > current_start + 1e-6
                        and quiet - current_start <= max_duration + 1e-9
                        and end_time - quiet >= min_piece - 1e-9
                    ):
                        chunk_end = quiet

            # Absolute progress guard.
            if chunk_end <= current_start + 1e-9:
                chunk_end = min(current_start + max_duration, end_time)
                if chunk_end <= current_start + 1e-9:
                    break

            piece = dict(original)
            piece["index"] = str(new_index).zfill(5)
            piece["start"] = round(current_start, 3)
            piece["end"] = round(chunk_end, 3)
            new_segments.append(piece)

            new_index += 1
            current_start = chunk_end

        if end_time > current_start + 1e-9:
            piece = dict(original)
            piece["index"] = str(new_index).zfill(5)
            piece["start"] = round(current_start, 3)
            piece["end"] = round(end_time, 3)
            new_segments.append(piece)
            new_index += 1

    return new_segments


def df_to_list(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        if not _is_finite_number(row.get("start")) or not _is_finite_number(row.get("end")):
            continue
        records.append({
            "index": f"{i:05d}",
            "start": float(row["start"]),
            "end": float(row["end"]),
            "speaker": row["speaker"],
        })
    return records


def build_silence_intervals(
    waveform,
    sample_rate,
    vad_model_func,
    min_silence=0.3,
    timestamps_are_samples: bool = False,
):
    """Return total duration and silence intervals.

    By default VAD timestamps are assumed to be SECONDS. Set
    `timestamps_are_samples=True` if the VAD returns sample indices.
    Speech timestamps are sorted, clamped and merged before silence is derived.
    """
    if waveform is None:
        return 0.0, []
    if sample_rate is None or sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if min_silence < 0:
        raise ValueError("min_silence must be >= 0")

    total_duration = len(waveform) / float(sample_rate)
    if len(waveform) == 0:
        return 0.0, []

    speech_ts = vad_model_func(waveform, sample_rate) or []

    normalized = []
    for item in speech_ts:
        if not isinstance(item, dict):
            continue
        if not _is_finite_number(item.get("start")) or not _is_finite_number(item.get("end")):
            continue

        start = float(item["start"])
        end = float(item["end"])
        if timestamps_are_samples:
            start /= float(sample_rate)
            end /= float(sample_rate)

        start = min(max(0.0, start), total_duration)
        end = min(max(0.0, end), total_duration)
        if end > start:
            normalized.append((start, end))

    speech = _merge_time_ranges(normalized)
    if not speech:
        return total_duration, [(0.0, total_duration)]

    silence = []

    first_start = speech[0][0]
    if first_start >= min_silence:
        silence.append((0.0, first_start))

    for (_, prev_end), (next_start, _) in zip(speech[:-1], speech[1:]):
        if next_start - prev_end >= min_silence:
            silence.append((prev_end, next_start))

    last_end = speech[-1][1]
    if total_duration - last_end >= min_silence:
        silence.append((last_end, total_duration))

    return total_duration, silence


def build_chunk_ranges(
    total_duration,
    silence_intervals,
    max_duration,
    min_chunk_ratio: float = 0.5,
):
    """Build chunks <= max_duration, preferring a late silence in each budget.

    The old implementation could make the final chunk exceed max_duration when
    no future silence existed. It could also choose a silence almost immediately
    after chunk_start and create tiny chunks. This version only prefers silence
    points in the latter part of the current budget.
    """
    if not _is_finite_number(total_duration) or float(total_duration) < 0:
        raise ValueError("total_duration must be >= 0")
    if not _is_finite_number(max_duration) or float(max_duration) <= 0:
        raise ValueError("max_duration must be > 0")
    if not (0.0 <= min_chunk_ratio <= 1.0):
        raise ValueError("min_chunk_ratio must be between 0 and 1")

    total_duration = float(total_duration)
    max_duration = float(max_duration)
    epsilon = 1e-3

    if total_duration <= max_duration + epsilon:
        return [(0.0, total_duration)]

    silence_points = sorted(
        (float(start) + float(end)) / 2.0
        for start, end in (silence_intervals or [])
        if _is_finite_number(start)
        and _is_finite_number(end)
        and float(end) > float(start)
    )

    chunk_ranges = []
    chunk_start = 0.0

    while chunk_start < total_duration - epsilon:
        limit = min(chunk_start + max_duration, total_duration)

        if limit >= total_duration - epsilon:
            chunk_end = total_duration
        else:
            preferred_lo = chunk_start + max_duration * min_chunk_ratio
            candidates = [
                point for point in silence_points
                if preferred_lo <= point <= limit
            ]
            chunk_end = candidates[-1] if candidates else limit

        if chunk_end - chunk_start < epsilon:
            chunk_end = min(chunk_start + max_duration, total_duration)
            if chunk_end - chunk_start < epsilon:
                break

        # Hard invariant.
        chunk_end = min(chunk_end, chunk_start + max_duration, total_duration)
        chunk_ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end

    return chunk_ranges if chunk_ranges else [(0.0, total_duration)]


def _ghost_target_for_segment(
    ghost_seg,
    real_segments,
    neighbour_gap: float,
    one_sided_gap: float,
):
    """Return (target_speaker, evidence) or (None, reason)."""
    gs = float(ghost_seg["start"])
    ge = float(ghost_seg["end"])

    # 1) Strongest evidence: direct temporal overlap with a real speaker.
    overlap_by_speaker = {}
    for real in real_segments:
        overlap = _overlap_duration(gs, ge, real["start"], real["end"])
        if overlap > 0:
            speaker = real["speaker"]
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap

    if overlap_by_speaker:
        target, overlap = max(
            overlap_by_speaker.items(),
            key=lambda item: item[1],
        )
        return target, f"overlap:{overlap:.3f}s"

    # 2) Find immediate real turn before and after by BOUNDARY distance, not
    # centre distance.
    before = [
        real for real in real_segments
        if float(real["end"]) <= gs
    ]
    after = [
        real for real in real_segments
        if float(real["start"]) >= ge
    ]

    prev_seg = max(before, key=lambda r: float(r["end"])) if before else None
    next_seg = min(after, key=lambda r: float(r["start"])) if after else None

    prev_gap = gs - float(prev_seg["end"]) if prev_seg else math.inf
    next_gap = float(next_seg["start"]) - ge if next_seg else math.inf

    # A ... ghost ... A is strong evidence when both gaps are small.
    if (
        prev_seg is not None
        and next_seg is not None
        and prev_seg["speaker"] == next_seg["speaker"]
        and prev_gap >= 0
        and next_gap >= 0
        and prev_gap + next_gap <= neighbour_gap
    ):
        return prev_seg["speaker"], f"bracketed:{prev_gap + next_gap:.3f}s"

    # 3) One-sided fallback only when VERY close. This is deliberately
    # conservative: a distant rare guest must not be dissolved merely because
    # somebody is the "nearest" turn.
    candidates = []
    if prev_seg is not None and 0 <= prev_gap <= one_sided_gap:
        candidates.append((prev_gap, prev_seg["speaker"], "left"))
    if next_seg is not None and 0 <= next_gap <= one_sided_gap:
        candidates.append((next_gap, next_seg["speaker"], "right"))

    if candidates:
        gap, target, side = min(candidates, key=lambda item: item[0])
        return target, f"{side}_adjacent:{gap:.3f}s"

    return None, "no_strong_timeline_evidence"


def merge_ghost_speakers(
    segment_list: list,
    share=GHOST_SPEAKER_SHARE,
    max_segment=GHOST_SPEAKER_MAX_SEGMENT,
    logger=None,
    neighbour_gap: float = 0.5,
    one_sided_gap: float = 0.15,
    require_all_fragments_supported: bool = True,
) -> list:
    """Conservatively dissolve clustering-artifact speakers.

    A low-share speaker is only a *candidate* ghost. It is relabelled only when
    its own fragments have strong timeline evidence:
      1. direct overlap with a real speaker;
      2. bracketed by the same real speaker on both sides;
      3. or extremely close to one real neighbour.

    Crucially, there is NO "nearest centre always wins" fallback. If evidence is
    weak, the speaker is kept unchanged.
    """
    if not segment_list:
        return segment_list
    if share < 0:
        raise ValueError("share must be >= 0")
    if max_segment <= 0:
        raise ValueError("max_segment must be > 0")
    if neighbour_gap < 0 or one_sided_gap < 0:
        raise ValueError("ghost neighbour gaps must be >= 0")

    ordered = []
    for seg in segment_list:
        if not isinstance(seg, dict):
            continue
        if (
            "speaker" not in seg
            or not _is_finite_number(seg.get("start"))
            or not _is_finite_number(seg.get("end"))
        ):
            continue
        item = dict(seg)
        item["start"] = float(item["start"])
        item["end"] = float(item["end"])
        if item["end"] > item["start"]:
            ordered.append(item)

    ordered.sort(key=lambda s: (s["start"], s["end"], str(s["speaker"])))
    speakers = sorted({seg["speaker"] for seg in ordered}, key=str)
    if len(speakers) < 3:
        return [dict(seg) for seg in ordered]

    speaker_ranges = {
        speaker: [
            (seg["start"], seg["end"])
            for seg in ordered
            if seg["speaker"] == speaker
        ]
        for speaker in speakers
    }

    # Use the union of all speech as denominator so overlap is not counted twice.
    total_spoken = _union_duration(
        (seg["start"], seg["end"]) for seg in ordered
    )
    if total_spoken <= 0:
        return [dict(seg) for seg in ordered]

    held_by_speaker = {
        speaker: _union_duration(ranges)
        for speaker, ranges in speaker_ranges.items()
    }

    candidates = set()
    for speaker in speakers:
        held = held_by_speaker[speaker]
        if held / total_spoken >= share:
            continue

        durations = [
            seg["end"] - seg["start"]
            for seg in ordered
            if seg["speaker"] == speaker
        ]
        if durations and max(durations) <= max_segment:
            candidates.add(speaker)

    if not candidates:
        return [dict(seg) for seg in ordered]

    real_speakers = set(speakers) - candidates
    if not real_speakers:
        return [dict(seg) for seg in ordered]

    real_segments = [
        seg for seg in ordered
        if seg["speaker"] in real_speakers
    ]

    # Build an evidence plan per candidate speaker FIRST. If conservative mode
    # is on and even one fragment lacks evidence, keep that entire speaker.
    plans = {}
    accepted_ghosts = set()

    for ghost in candidates:
        ghost_segments = [
            seg for seg in ordered
            if seg["speaker"] == ghost
        ]

        ghost_plan = []
        unsupported = 0

        for seg in ghost_segments:
            target, evidence = _ghost_target_for_segment(
                seg,
                real_segments,
                neighbour_gap=neighbour_gap,
                one_sided_gap=one_sided_gap,
            )
            ghost_plan.append((seg, target, evidence))
            if target is None:
                unsupported += 1

        if require_all_fragments_supported and unsupported:
            if logger:
                logger.info(
                    f"Keeping low-share speaker {ghost}: "
                    f"{unsupported}/{len(ghost_segments)} fragment(s) lack strong "
                    f"timeline evidence"
                )
            continue

        if all(target is None for _, target, _ in ghost_plan):
            continue

        accepted_ghosts.add(ghost)
        plans[ghost] = ghost_plan

    if not accepted_ghosts:
        return [dict(seg) for seg in ordered]

    # Relabel only fragments with actual evidence. In conservative mode every
    # fragment is supported by construction; in permissive mode unsupported
    # fragments remain as the original speaker.
    out = []
    moved = 0
    evidence_counts = {}

    for seg in ordered:
        speaker = seg["speaker"]
        if speaker not in accepted_ghosts:
            out.append(dict(seg))
            continue

        target = None
        evidence = None
        for planned_seg, planned_target, planned_evidence in plans[speaker]:
            if (
                planned_seg["start"] == seg["start"]
                and planned_seg["end"] == seg["end"]
            ):
                target = planned_target
                evidence = planned_evidence
                break

        changed = dict(seg)
        if target is not None:
            changed["speaker"] = target
            moved += 1
            evidence_counts[evidence.split(":", 1)[0]] = (
                evidence_counts.get(evidence.split(":", 1)[0], 0) + 1
            )
        out.append(changed)

    # Merge only overlapping/touching pieces created by relabelling. A positive
    # gap is left alone here; normal merge policy can handle it later.
    out = cut_by_speaker_label(
        out,
        merge_gap=0.0,
        min_segment_length=0.0,
        max_segment_length=float("inf"),
        logger=logger,
        seams=None,
    )

    # Dense ordered indices are safer downstream after relabelling/merging.
    for i, seg in enumerate(out):
        seg["index"] = str(i).zfill(5)

    if logger:
        merged_names = ", ".join(sorted(map(str, accepted_ghosts)))
        evidence_text = ", ".join(
            f"{key}={value}" for key, value in sorted(evidence_counts.items())
        ) or "none"
        logger.info(
            f"Merged {len(accepted_ghosts)} ghost speaker(s) "
            f"({merged_names}): {moved} segment(s) relabelled "
            f"[{evidence_text}]"
        )

    return out


def split_at_seams(
    segment_list: list,
    seams,
    min_piece: float = 0.05,
) -> list:
    """Break any segment that straddles an excision seam."""
    if min_piece < 0:
        raise ValueError("min_piece must be >= 0")

    if not segment_list:
        return []

    if not seams:
        out = [dict(seg) for seg in segment_list]
        out.sort(key=lambda s: (float(s["start"]), float(s["end"])))
        for i, seg in enumerate(out):
            seg["index"] = str(i).zfill(5)
        return out

    ordered_seams = sorted(
        float(s) for s in seams
        if _is_finite_number(s)
    )

    out = []
    for seg in segment_list:
        if not isinstance(seg, dict):
            continue
        if not _is_finite_number(seg.get("start")) or not _is_finite_number(seg.get("end")):
            continue

        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue

        inside = [seam for seam in ordered_seams if start < seam < end]
        if not inside:
            piece = dict(seg)
            piece["start"], piece["end"] = start, end
            out.append(piece)
            continue

        for lo, hi in zip([start] + inside, inside + [end]):
            if hi - lo < min_piece:
                continue
            piece = dict(seg)
            piece["start"], piece["end"] = lo, hi
            out.append(piece)

    out.sort(key=lambda s: (s["start"], s["end"], str(s.get("speaker", ""))))
    for i, seg in enumerate(out):
        seg["index"] = str(i).zfill(5)
    return out
