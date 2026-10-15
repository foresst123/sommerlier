"""Backchannel anchoring and speech-coverage checks for Lhotse duplex manifests.

Pure functions on plain dicts (no lhotse import). See spec sections 4.4 and 7.4.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BackchannelParams:
    eps: float = 0.001


@dataclass
class BackchannelResult:
    attached: List[dict] = field(default_factory=list)
    demoted: List[dict] = field(default_factory=list)
    dropped: List[dict] = field(default_factory=list)


def _contains(turn: dict, cand: dict, eps: float) -> bool:
    return (
        turn["start"] - eps <= cand["start"]
        and cand["end"] <= turn["start"] + turn["duration"] + eps
    )


def _speech_contains(turn: dict, t: float, eps: float) -> bool:
    return turn["speech_start"] - eps <= t <= turn["speech_end"] + eps


def _pick_anchor(turns: Sequence[dict], cand: dict, eps: float):
    opposite = [
        t for t in turns
        if t["channel"] == 1 - cand["channel"] and _contains(t, cand, eps)
    ]
    if not opposite:
        return None
    # Prefer the turn whose speech is active at cand.start, then the latest start.
    return max(opposite, key=lambda t: (_speech_contains(t, cand["start"], eps), t["start"]))


def _overlaps_same_channel_speech(turns: Sequence[dict], cand: dict, eps: float) -> bool:
    for t in turns:
        if t["channel"] != cand["channel"]:
            continue
        overlap = min(cand["end"], t["speech_end"]) - max(cand["start"], t["speech_start"])
        if overlap > eps:
            return True
    return False


def _word_index(anchor: dict, cand: dict, eps: float) -> int:
    words = anchor["words"]
    n = len(words)
    count = sum(1 for w in words if w[1] <= cand["start"] + eps)
    return max(1, min(count, n))


def resolve_backchannels(
    turns: Sequence[dict],
    candidates: Sequence[dict],
    params: BackchannelParams = BackchannelParams(),
) -> BackchannelResult:
    """Attach candidates to an opposite-channel anchor turn, or demote/drop them."""
    eps = params.eps
    result = BackchannelResult()
    by_anchor: Dict[str, List[Tuple[dict, dict]]] = {}
    for cand in candidates:
        anchor = _pick_anchor(turns, cand, eps)
        if anchor is None:
            if _overlaps_same_channel_speech(turns, cand, eps):
                result.dropped.append(
                    {"candidate": cand, "reason": "no_anchor_same_channel_overlap"}
                )
            else:
                result.demoted.append(cand)
            continue
        if not anchor.get("words"):
            result.dropped.append({"candidate": cand, "reason": "anchor_without_words"})
            continue
        by_anchor.setdefault(anchor["id"], []).append((anchor, cand))

    for anchor_id, pairs in by_anchor.items():
        pairs.sort(key=lambda p: (p[1]["start"], p[1]["end"]))
        for k, (anchor, cand) in enumerate(pairs):
            result.attached.append(
                dict(
                    cand,
                    anchor_id=anchor_id,
                    anchor_word_index=_word_index(anchor, cand, eps),
                    k=k,
                    id=f"{anchor_id}-bc{k}",
                )
            )
    return result


def anchor_backchannel_ids(attached: Sequence[dict]) -> Dict[str, List[str]]:
    """anchor_id -> backchannel ids ordered by k."""
    grouped: Dict[str, List[dict]] = {}
    for bc in attached:
        grouped.setdefault(bc["anchor_id"], []).append(bc)
    return {a: [b["id"] for b in sorted(v, key=lambda b: b["k"])] for a, v in grouped.items()}


@dataclass(frozen=True)
class CoverageParams:
    frame_seconds: float = 0.02
    max_uncovered_seconds: float = 0.30
    # track_quality uses floor=0.002 as "near silence"; 2.5x that skips
    # separation bleed/noise but keeps quiet speech.
    rms_threshold: float = 0.005
    min_speech_frames: int = 3  # ignore clicks shorter than ~60 ms
    merge_gap_frames: int = 3  # bridge pauses up to ~60 ms inside one region
    span_eps: float = 0.02  # slack around spans (one frame)


def _merge_spans(spans: Sequence[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for s, e in sorted((s - eps, e + eps) for s, e in spans if e > s):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def find_unannotated_speech(
    samples: np.ndarray,
    sample_rate: int,
    spans: Sequence[Tuple[float, float]],
    params: CoverageParams = CoverageParams(),
) -> List[Tuple[float, float]]:
    """Speech regions (frame RMS >= threshold) outside `spans`, longer than max_uncovered."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    frame = max(1, round(params.frame_seconds * sample_rate))
    n = len(x)
    if n == 0:
        return []
    starts = np.arange(0, n, frame)
    counts = np.minimum(frame, n - starts)
    rms = np.sqrt(np.add.reduceat(x ** 2, starts) / counts)
    active = rms >= params.rms_threshold

    # Runs of active frames, bridging short gaps.
    runs: List[List[int]] = []
    for i in np.flatnonzero(active):
        i = int(i)
        if runs and i - runs[-1][1] - 1 <= params.merge_gap_frames:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    regions = [
        (a * frame / sample_rate, min((b + 1) * frame, n) / sample_rate)
        for a, b in runs
        if b - a + 1 >= params.min_speech_frames
    ]

    covered = _merge_spans(spans, params.span_eps)
    found: List[Tuple[float, float]] = []
    for s, e in regions:
        cur = s
        for cs, ce in covered:
            if ce <= cur:
                continue
            if cs >= e:
                break
            if cs > cur:
                found.append((cur, cs))
            cur = max(cur, ce)
        if cur < e:
            found.append((cur, e))
    return [(s, e) for s, e in found if e - s > params.max_uncovered_seconds]


def check_item_coverage(
    wav_path: str,
    channel_spans: Dict[int, Sequence[Tuple[float, float]]],
    params: CoverageParams = CoverageParams(),
) -> List[dict]:
    """Violations [{channel,start,end}] of the no-unannotated-speech rule; empty = ok."""
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    violations: List[dict] = []
    for ch in sorted(channel_spans):
        if ch >= data.shape[1]:
            continue
        for s, e in find_unannotated_speech(data[:, ch], sr, channel_spans[ch], params):
            violations.append({"channel": ch, "start": round(s, 6), "end": round(e, 6)})
    return violations
