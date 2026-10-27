"""Pure planning and speaker-fusion helpers for Nemotron's two-pass mode.

The worker owns audio/model I/O.  This module deliberately only deals in
seconds and dictionaries so it can be exercised without NeMo, CUDA or an
installed VAD model.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional


UNKNOWN_SPEAKER = "SPEAKER_UNKNOWN"


@dataclass(frozen=True)
class AudioWindow:
    """A source interval plus the context sent to the model."""

    core_start: float
    core_end: float
    context_start: float
    context_end: float

    @property
    def duration(self):
        return self.core_end - self.core_start


@dataclass
class ChunkResult:
    window: AudioWindow
    segments: list[dict]
    chunk_id: str = ""


def _merge_intervals(intervals: Iterable[tuple[float, float]], duration: float):
    clean = sorted(
        (max(0.0, float(start)), min(duration, float(end)))
        for start, end in intervals
        if float(end) > float(start)
    )
    merged = []
    for start, end in clean:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def build_vad_windows(
    duration: float,
    speech_intervals: Iterable[tuple[float, float]],
    *,
    target_seconds: float = 25.0,
    min_seconds: float = 20.0,
    max_seconds: float = 30.0,
    context_seconds: float = 2.0,
) -> list[AudioWindow]:
    """Make contiguous core windows, using VAD only to choose cut points.

    Silence is never removed: the cores form an exact ``[0, duration]``
    partition.  A nearby VAD silence is preferred over cutting through speech,
    but a cut is still made at the target/max bound when no silence is
    available.  Context is metadata and may overlap adjacent cores.
    """
    duration = max(0.0, float(duration))
    if duration == 0.0:
        return []
    if not (0.0 < min_seconds <= target_seconds <= max_seconds):
        raise ValueError("expected 0 < min_seconds <= target_seconds <= max_seconds")
    context_seconds = max(0.0, float(context_seconds))
    speech = _merge_intervals(speech_intervals, duration)
    gaps = []
    cursor = 0.0
    for start, end in speech:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append((cursor, duration))

    windows = []
    start = 0.0
    while start < duration - 1e-9:
        remaining = duration - start
        if remaining <= max_seconds:
            end = duration
        else:
            desired = min(duration, start + target_seconds)
            candidates = [
                (gap_end, abs(gap_end - desired))
                for gap_start, gap_end in gaps
                if gap_end >= start + min_seconds
                and gap_end <= min(duration, start + max_seconds)
                and gap_end > gap_start
            ]
            end = min(candidates, key=lambda item: item[1])[0] if candidates else desired
        if end <= start:
            raise RuntimeError("Nemotron window planner made no progress")
        windows.append(AudioWindow(
            core_start=start,
            core_end=end,
            context_start=max(0.0, start - context_seconds),
            context_end=min(duration, end + context_seconds),
        ))
        start = end
    return windows


def guide_windows_with_pass1(
    windows: Iterable[AudioWindow],
    pass1_segments: Iterable[dict],
    *,
    context_seconds: float = 2.0,
    duration: Optional[float] = None,
) -> list[AudioWindow]:
    """Expand pass-2 context around the pass-1 speech evidence.

    Cores stay fixed, so the VAD partition still covers every source sample.
    The pass-1 evidence makes a speaker crossing a core boundary visible in
    both pass-2 calls, even when the VAD cut happened in a short pause.
    """
    windows = list(windows)
    segments = [s for s in pass1_segments if float(s["end"]) > float(s["start"])]
    if duration is None:
        duration = max((float(w.context_end) for w in windows), default=0.0)
    out = []
    for window in windows:
        relevant = [
            s for s in segments
            if float(s["end"]) >= window.core_start - context_seconds
            and float(s["start"]) <= window.core_end + context_seconds
        ]
        left = max(0.0, window.core_start - context_seconds)
        right = min(duration, window.core_end + context_seconds)
        if relevant:
            left = min(left, min(float(s["start"]) for s in relevant) - context_seconds)
            right = max(right, max(float(s["end"]) for s in relevant) + context_seconds)
            left = max(0.0, left)
            right = min(duration, right)
        out.append(AudioWindow(window.core_start, window.core_end, left, right))
    return out


def crop_to_core(segments: Iterable[dict], window: AudioWindow) -> list[dict]:
    """Convert context-relative timestamps to source time and crop duplicates."""
    out = []
    for segment in segments:
        start = max(window.core_start, window.context_start + float(segment["start"]))
        end = min(window.core_end, window.context_start + float(segment["end"]))
        if end <= start:
            continue
        item = dict(segment)
        item["start"], item["end"] = start, end
        out.append(item)
    return sorted(out, key=lambda s: (s["start"], s["end"], str(s.get("speaker", ""))))


def _clip_absolute_to_core(segments: Iterable[dict], window: AudioWindow) -> list[dict]:
    out = []
    for segment in segments:
        start = max(window.core_start, float(segment["start"]))
        end = min(window.core_end, float(segment["end"]))
        if end <= start:
            continue
        item = dict(segment)
        item["start"], item["end"] = start, end
        out.append(item)
    return sorted(out, key=lambda s: (s["start"], s["end"], str(s.get("speaker", ""))))


def _speaker_key(chunk: ChunkResult, speaker: str) -> str:
    return f"{chunk.chunk_id}:{speaker}" if chunk.chunk_id else str(speaker)


def _union_duration(segments):
    if not segments:
        return 0.0
    ordered = sorted((float(s["start"]), float(s["end"])) for s in segments)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def _clean_duration(segments, label):
    target = [s for s in segments if str(s.get("speaker")) == str(label)]
    foreign = [s for s in segments if str(s.get("speaker")) != str(label)]
    clean = []
    for source in target:
        pending = [(float(source["start"]), float(source["end"]))]
        for other in foreign:
            other_start, other_end = float(other["start"]), float(other["end"])
            next_pending = []
            for start, end in pending:
                if other_end <= start or other_start >= end:
                    next_pending.append((start, end))
                    continue
                if start < other_start:
                    next_pending.append((start, other_start))
                if other_end < end:
                    next_pending.append((other_end, end))
            pending = next_pending
        clean.extend({"start": start, "end": end} for start, end in pending if end > start)
    return _union_duration(clean)


def clean_speech_duration(segments, label):
    """Measure one label after removing foreign-speaker overlap."""
    return _clean_duration(segments, label)


def chunk_clean_speech_seconds(chunks: Iterable[ChunkResult], prefix="pass1"):
    """Return clean, core-cropped seconds keyed by chunk-local speaker."""
    result = {}
    for chunk in chunks:
        absolute = []
        for segment in chunk.segments:
            item = dict(segment)
            item["start"] = max(chunk.window.core_start,
                                chunk.window.context_start + float(segment["start"]))
            item["end"] = min(chunk.window.core_end,
                              chunk.window.context_start + float(segment["end"]))
            if item["end"] > item["start"]:
                absolute.append(item)
        for label in {str(s["speaker"]) for s in absolute}:
            result[(prefix, _speaker_key(chunk, label))] = _clean_duration(absolute, label)
    return result


def _overlap(a, b):
    return max(0.0, min(float(a["end"]), float(b["end"])) -
               max(float(a["start"]), float(b["start"])))


def _copy_with_speaker(segment, speaker, source, confidence):
    item = dict(segment)
    item["speaker"] = speaker
    item["speaker_source"] = source
    item["speaker_confidence"] = bool(confidence)
    return item


def globalize_pass1(
    chunks: Iterable[ChunkResult],
    *,
    min_overlap: float = 0.1,
    speaker_similarity: Optional[Callable[[str, str], float]] = None,
    clean_speech_seconds: Optional[dict[tuple[str, str], float]] = None,
    min_clean_speech_seconds: float = 1.5,
    similarity_threshold: float = 0.6,
) -> list[list[dict]]:
    """Map pass-1 labels globally, then reconcile chunks without overlap."""
    chunks = sorted(list(chunks), key=lambda c: c.window.core_start)
    clean = clean_speech_seconds or {}
    global_segments = []
    next_id = 0
    result = []
    for chunk in sorted(chunks, key=lambda c: c.window.core_start):
        local_to_global = {}
        previous_segments = list(global_segments)
        local_segments = sorted(chunk.segments, key=lambda s: (s["start"], s["end"]))
        local_labels = []
        for segment in local_segments:
            local = str(segment["speaker"])
            if local not in local_labels:
                local_labels.append(local)
        candidates = []
        for local in local_labels:
            local_key = _speaker_key(chunk, local)
            absolute_segments = []
            for segment in local_segments:
                if str(segment["speaker"]) != local:
                    continue
                item = dict(segment)
                item["start"] = chunk.window.context_start + float(segment["start"])
                item["end"] = chunk.window.context_start + float(segment["end"])
                absolute_segments.append(item)
            clean_duration = clean.get(("pass1", local_key),
                                       _clean_duration(absolute_segments, local))
            if clean_speech_seconds is not None and clean_duration < min_clean_speech_seconds:
                local_to_global[local] = (UNKNOWN_SPEAKER, "insufficient_clean_speech", False)
                continue
            votes = {}
            for segment in absolute_segments:
                for previous in previous_segments:
                    if previous["speaker"] == UNKNOWN_SPEAKER:
                        continue
                    amount = _overlap(segment, previous)
                    if amount:
                        votes[previous["speaker"]] = votes.get(previous["speaker"], 0.0) + amount
            for target, score in votes.items():
                if score >= min_overlap:
                    candidates.append((score, local, target, "pass1_overlap"))
        used_locals = set()
        used_targets = set()
        for score, local, target, source in sorted(
                candidates, key=lambda item: (-item[0], item[1], item[2])):
            if local in used_locals or target in used_targets:
                continue
            local_to_global[local] = (target, source, True)
            used_locals.add(local)
            used_targets.add(target)
        for local in local_labels:
            if local in local_to_global:
                continue
            local_key = _speaker_key(chunk, local)
            if speaker_similarity is not None:
                absolute_segments = []
                for segment in local_segments:
                    if str(segment["speaker"]) == local:
                        item = dict(segment)
                        item["start"] = chunk.window.context_start + float(segment["start"])
                        item["end"] = chunk.window.context_start + float(segment["end"])
                        absolute_segments.append(item)
                _register_similarity(speaker_similarity, "candidate", local_key,
                                     absolute_segments)
                targets = sorted({previous["speaker"] for previous in previous_segments
                                  if previous["speaker"] != UNKNOWN_SPEAKER
                                  and previous["speaker"] not in used_targets})
                scored = []
                for target in targets:
                    target_segments = [previous for previous in previous_segments
                                       if previous["speaker"] == target]
                    _register_similarity(speaker_similarity, "reference", target,
                                         target_segments)
                    target_clean = clean.get(("pass1", target),
                                             _clean_duration(previous_segments, target))
                    local_clean = clean.get(("pass1", local_key),
                                            _clean_duration(absolute_segments, local))
                    if (local_clean >= min_clean_speech_seconds
                            and target_clean >= min_clean_speech_seconds):
                        score = float(speaker_similarity(local_key, target))
                        if score >= similarity_threshold:
                            scored.append((score, target))
                if scored:
                    _, target = max(scored)
                    local_to_global[local] = (target, "wespeaker", True)
                    used_targets.add(target)
                    used_locals.add(local)
                    continue
            local_clean = clean.get(("pass1", local_key),
                                    _clean_duration(absolute_segments, local))
            if clean_speech_seconds is not None and local_clean < min_clean_speech_seconds:
                local_to_global[local] = (UNKNOWN_SPEAKER, "insufficient_clean_speech", False)
            else:
                local_to_global[local] = (f"SPEAKER_{next_id:02d}",
                                           "pass1_first_appearance", True)
                next_id += 1

        for segment in local_segments:
            local = str(segment["speaker"])
            mapped, source, confident = local_to_global[local]
            local_key = _speaker_key(chunk, local)
            item = _copy_with_speaker(segment, mapped, source, confident)
            item["_local_speaker"] = local_key
            item["start"] = chunk.window.context_start + float(segment["start"])
            item["end"] = chunk.window.context_start + float(segment["end"])
            global_segments.append(item)
            result.append((chunk, item))
        # Crop after mapping, so context remains available to the next chunk's
        # overlap vote but cannot duplicate the exported pass-1 timeline.
    by_chunk = {}
    for chunk, item in result:
        by_chunk.setdefault(id(chunk), []).append(item)
    output = [
        _clip_absolute_to_core(items, chunk.window)
        for chunk in sorted(chunks, key=lambda c: c.window.core_start)
        for items in [by_chunk.get(id(chunk), [])]
    ]
    if speaker_similarity is not None:
        output = reconcile_pass1_speakers(
            output,
            speaker_similarity=speaker_similarity,
            clean_speech_seconds=clean,
            min_clean_speech_seconds=min_clean_speech_seconds,
            similarity_threshold=similarity_threshold,
        )
    return output


def _register_similarity(similarity, role, label, segments):
    register = getattr(similarity, f"register_{role}", None)
    if callable(register):
        register(label, segments)


def reconcile_pass1_speakers(
    chunks: Iterable[list[dict]],
    *,
    speaker_similarity: Callable[[str, str], float],
    clean_speech_seconds: Optional[dict[tuple[str, str], float]] = None,
    min_clean_speech_seconds: float = 1.5,
    similarity_threshold: float = 0.6,
) -> list[list[dict]]:
    """Merge provisional pass-1 IDs across silent chunk boundaries one-to-one."""
    clean = clean_speech_seconds or {}
    output = []
    previous = []
    for chunk in chunks:
        current = [dict(segment) for segment in chunk]
        labels = sorted({str(s["speaker"]) for s in current
                         if s["speaker"] != UNKNOWN_SPEAKER})
        previous_labels = sorted({str(s["speaker"]) for s in previous
                                  if s["speaker"] != UNKNOWN_SPEAKER})
        used_targets = set(previous_labels) & set(labels)
        pairs = []
        for label in labels:
            if label in used_targets:
                continue
            candidate_segments = [s for s in current if str(s["speaker"]) == label]
            local_key = str(candidate_segments[0].get("_local_speaker", label))
            _register_similarity(speaker_similarity, "candidate", local_key,
                                 candidate_segments)
            local_clean = clean.get(("pass1", local_key),
                                    _clean_duration(current, label))
            if local_clean < min_clean_speech_seconds:
                continue
            for target in previous_labels:
                if target in used_targets:
                    continue
                target_segments = [s for s in previous if str(s["speaker"]) == target]
                _register_similarity(speaker_similarity, "reference", target,
                                     target_segments)
                target_clean = clean.get(("pass1", target),
                                         _clean_duration(previous, target))
                if target_clean < min_clean_speech_seconds:
                    continue
                score = float(speaker_similarity(local_key, target))
                if score >= similarity_threshold:
                    pairs.append((score, label, target))
        assigned = set()
        for _, label, target in sorted(pairs, key=lambda item: (-item[0], item[1], item[2])):
            if label in assigned or target in used_targets:
                continue
            for segment in current:
                if str(segment["speaker"]) == label:
                    segment["speaker"] = target
                    segment["speaker_source"] = "wespeaker"
                    segment["speaker_confidence"] = True
            assigned.add(label)
            used_targets.add(target)
        output.append(current)
        previous.extend(current)
    return output


def map_pass2_to_global(
    chunks: Iterable[ChunkResult],
    pass1_segments: Iterable[dict],
    *,
    min_overlap: float = 0.1,
    min_overlap_ratio: float = 0.5,
    speaker_similarity: Optional[Callable[[str, str], float]] = None,
    clean_speech_seconds: Optional[dict[tuple[str, str], float]] = None,
    min_clean_speech_seconds: float = 1.5,
    similarity_threshold: float = 0.6,
) -> list[list[dict]]:
    """Map pass-2 labels by pass-1 overlap, with guarded WeSpeaker fallback."""
    chunks = sorted(list(chunks), key=lambda c: c.window.core_start)
    pass1 = list(pass1_segments)
    clean = clean_speech_seconds or {}
    output = []
    for chunk in sorted(chunks, key=lambda c: c.window.core_start):
        local_to_mapping = {}
        for segment in sorted(chunk.segments, key=lambda s: (s["start"], s["end"])):
            local = str(segment["speaker"])
            absolute = dict(segment)
            absolute["start"] = chunk.window.context_start + float(segment["start"])
            absolute["end"] = chunk.window.context_start + float(segment["end"])
            if local not in local_to_mapping:
                votes = {}
                for reference in pass1:
                    if str(reference["speaker"]) == UNKNOWN_SPEAKER:
                        continue
                    amount = _overlap(absolute, reference)
                    if amount:
                        label = str(reference["speaker"])
                        votes[label] = votes.get(label, 0.0) + amount
                best = max(votes, key=votes.get) if votes else None
                if best is not None and votes[best] >= min_overlap and votes[best] / max(
                        float(segment["end"]) - float(segment["start"]), 1e-9) >= min_overlap_ratio:
                    local_to_mapping[local] = (best, "pass1_overlap", True)
                else:
                    local_to_mapping[local] = (UNKNOWN_SPEAKER, "insufficient_overlap", False)
                    if speaker_similarity is not None and clean.get(("pass2", local), 0.0) >= min_clean_speech_seconds:
                        candidates = [label for label in {str(s["speaker"]) for s in pass1}
                                      if clean.get(("pass1", label), 0.0) >= min_clean_speech_seconds]
                        scored = [(float(speaker_similarity(local, label)), label)
                                  for label in candidates]
                        if scored:
                            score, label = max(scored)
                            if score >= similarity_threshold:
                                local_to_mapping[local] = (label, "wespeaker", True)
            speaker, source, confident = local_to_mapping[local]
            item = _copy_with_speaker(segment, speaker, source, confident)
            item["start"] = absolute["start"]
            item["end"] = absolute["end"]
            output.append((chunk, item))
    by_chunk = {}
    for chunk, item in output:
        by_chunk.setdefault(id(chunk), []).append(item)
    return [
        _clip_absolute_to_core(items, chunk.window)
        for chunk in sorted(chunks, key=lambda c: c.window.core_start)
        for items in [by_chunk.get(id(chunk), [])]
    ]
