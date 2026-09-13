"""Shared acoustic boundary and context-expansion decisions.

All callers express where they may search; this module owns how candidates are
ranked.  Silero pauses are preferred, then stable local energy valleys, with a
timestamp fallback so a missing acoustic cue never drops work.
"""
from dataclasses import dataclass

import numpy as np


def _merge_ranges(ranges):
    merged = []
    for start, end in sorted((int(a), int(b)) for a, b in ranges if b > a):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract(ranges, blockers):
    result = []
    for lo, hi in _merge_ranges(ranges):
        for start, end in _merge_ranges(blockers):
            if end <= lo:
                continue
            if start >= hi:
                break
            if start > lo:
                result.append((lo, start))
            lo = max(lo, end)
        if lo < hi:
            result.append((lo, hi))
    return result


@dataclass(frozen=True)
class BoundaryCandidate:
    sample: int
    method: str
    confidence: float
    distance_samples: int


@dataclass(frozen=True)
class CutDecision:
    requested_sample: int
    sample: int
    method: str
    confidence: float
    search_samples: tuple
    fallback: bool = False


@dataclass(frozen=True)
class ExpansionDecision:
    anchor_sample: int
    boundary_sample: int
    direction: str
    requested_samples: int
    expanded_samples: int
    method: str
    confidence: float
    hard_bound_sample: int
    fallback: bool = False


class AcousticBoundaryFinder:
    """Find reusable speech-safe boundaries in sample coordinates."""

    _METHOD_RANK = {
        "silero_pause": 0,
        "silero": 1,
        "energy_pause": 2,
        "energy_valley": 3,
        "word_gap": 3,
        "timestamp_bound": 4,
        "segment": 5,
    }

    def __init__(self, waveform, sample_rate, vad=None):
        self.waveform = np.asarray(waveform)
        self.sr = int(sample_rate)
        self.vad = vad
        self.cache = {}

    def analyse(self, lo, hi):
        """Return the legacy ``(cuts, voiced, method)`` representation."""
        key = (max(0, int(lo)), min(len(self.waveform), int(hi)))
        if key in self.cache:
            return self.cache[key]

        lo, hi = key
        if hi <= lo:
            result = ({lo: "segment", hi: "segment"}, [], "energy")
            self.cache[key] = result
            return result

        wave = np.asarray(self.waveform[lo:hi], dtype=np.float64)
        voiced, method = None, "energy"
        if self.vad is not None:
            try:
                timestamps = self.vad.get_speech_timestamps(
                    self.waveform[lo:hi], sampling_rate=self.sr
                )
                voiced = _merge_ranges(
                    (lo + max(0, int(t["start"])),
                     min(hi, lo + int(t["end"])))
                    for t in timestamps
                )
                method = "silero"
            except Exception:
                voiced, method = None, "energy"

        energy_frame = max(1, round(self.sr * 0.010))
        starts = np.arange(0, len(wave), energy_frame, dtype=np.int64)
        frame_rms = None
        if len(starts):
            squared = np.square(wave)
            counts = np.minimum(energy_frame, len(wave) - starts)
            sums = np.add.reduceat(squared, starts)
            frame_rms = np.sqrt(sums / np.maximum(counts, 1))

        if voiced is None:
            if frame_rms is not None and len(frame_rms):
                p90 = float(np.percentile(frame_rms, 90))
                threshold = max(1e-5, p90 * 0.10)
                voiced = _merge_ranges(
                    (lo + int(start), min(hi, lo + int(start) + energy_frame))
                    for start, value in zip(starts, frame_rms)
                    if value > threshold
                )
            else:
                voiced = []

        pauses = _subtract([(lo, hi)], voiced)
        cuts = {lo: "segment", hi: "segment"}
        min_pause = max(1, round(self.sr * 0.020))
        pause_kind = "silero_pause" if method == "silero" else "energy_pause"
        for start, end in pauses:
            if end - start >= min_pause:
                cuts[(start + end) // 2] = pause_kind

        frame = max(1, round(self.sr * 0.010))
        hop = max(1, round(self.sr * 0.005))
        if len(wave) >= frame * 3:
            positions = np.arange(0, len(wave) - frame + 1, hop, dtype=np.int64)
            squared = np.square(wave)
            cumsum = np.concatenate(
                (np.array([0.0], dtype=np.float64), np.cumsum(squared))
            )
            valley_rms = np.sqrt(
                (cumsum[positions + frame] - cumsum[positions]) / frame + 1e-12
            )
            if len(valley_rms) >= 3:
                global_ref = float(np.percentile(valley_rms, 75))
                candidates = []
                for index in range(1, len(valley_rms) - 1):
                    value = float(valley_rms[index])
                    if value > valley_rms[index - 1] or value > valley_rms[index + 1]:
                        continue
                    neighbourhood = valley_rms[
                        max(0, index - 6):min(len(valley_rms), index + 7)
                    ]
                    local_ref = float(np.percentile(neighbourhood, 75))
                    if local_ref <= 1e-5:
                        continue
                    ratio = value / max(local_ref, 1e-5)
                    if ratio > 0.60 or (global_ref > 1e-5 and value > global_ref * 0.55):
                        continue
                    point = lo + int(positions[index] + frame // 2)
                    if point - lo < frame or hi - point < frame:
                        continue
                    if any(a <= point <= b and b - a >= min_pause for a, b in pauses):
                        continue
                    candidates.append((ratio, value, point))

                clusters, current = [], []
                for item in sorted(candidates, key=lambda value: value[2]):
                    if current and item[2] - current[-1][2] > round(self.sr * 0.040):
                        clusters.append(current)
                        current = []
                    current.append(item)
                if current:
                    clusters.append(current)
                for cluster in clusters:
                    _, _, point = min(cluster, key=lambda value: (value[0], value[1]))
                    cuts.setdefault(point, "energy_valley")

        result = (cuts, voiced, method)
        self.cache[key] = result
        return result

    def _confidence(self, point, method, lo, hi):
        if method == "silero_pause":
            return 1.0
        if method == "silero":
            return 0.9
        radius = max(1, round(0.02 * self.sr))
        local_lo, local_hi = max(lo, point - radius), min(hi, point + radius)
        ref_lo, ref_hi = max(lo, point - self.sr // 2), min(hi, point + self.sr // 2)
        if local_hi <= local_lo or ref_hi <= ref_lo:
            return 0.0
        local = np.asarray(self.waveform[local_lo:local_hi], dtype=np.float64)
        reference = np.asarray(self.waveform[ref_lo:ref_hi], dtype=np.float64)
        local_rms = float(np.sqrt(np.mean(np.square(local)) + 1e-12))
        ref_rms = float(np.sqrt(np.mean(np.square(reference)) + 1e-12))
        quietness = 1.0 - min(1.0, local_rms / max(ref_rms, 1e-5))
        base = 0.72 if method == "energy_pause" else 0.52
        return min(0.89, base + 0.25 * quietness)

    def find_candidates(self, anchor, direction="both", search_min=None,
                        search_max=None, hard_bounds=None, include_fallback=True):
        if direction not in {"left", "right", "both"}:
            raise ValueError("direction must be left, right, or both")
        anchor = int(anchor)
        floor, ceiling = hard_bounds or (0, len(self.waveform))
        lo = max(0, int(floor), int(search_min if search_min is not None else floor))
        hi = min(len(self.waveform), int(ceiling),
                 int(search_max if search_max is not None else ceiling))
        if direction == "left":
            hi = min(hi, anchor)
        elif direction == "right":
            lo = max(lo, anchor)
        if hi < lo:
            raise ValueError("no cut satisfies search range, direction, and hard bounds")

        cuts, _, _ = self.analyse(lo, hi)
        candidates = []
        for point, method in cuts.items():
            if method == "segment":
                continue
            if direction == "left" and point > anchor:
                continue
            if direction == "right" and point < anchor:
                continue
            candidates.append(BoundaryCandidate(
                sample=int(point), method=method,
                confidence=self._confidence(point, method, lo, hi),
                distance_samples=abs(int(point) - anchor),
            ))

        candidates.sort(key=lambda item: (
            self._METHOD_RANK.get(item.method, 9),
            -item.confidence,
            item.distance_samples,
        ))
        if include_fallback:
            fallback = min(max(anchor, lo), hi)
            candidates.append(BoundaryCandidate(
                sample=fallback, method="timestamp_bound", confidence=0.0,
                distance_samples=abs(fallback - anchor),
            ))
        return candidates

    def find_cut(self, anchor, direction="both", search_min=None,
                 search_max=None, hard_bounds=None):
        candidates = self.find_candidates(
            anchor, direction=direction, search_min=search_min,
            search_max=search_max, hard_bounds=hard_bounds,
            include_fallback=True,
        )
        chosen = candidates[0]
        floor, ceiling = hard_bounds or (0, len(self.waveform))
        lo = max(0, int(floor), int(search_min if search_min is not None else floor))
        hi = min(len(self.waveform), int(ceiling),
                 int(search_max if search_max is not None else ceiling))
        if direction == "left":
            hi = min(hi, int(anchor))
        elif direction == "right":
            lo = max(lo, int(anchor))
        return CutDecision(
            requested_sample=int(anchor), sample=chosen.sample,
            method=chosen.method, confidence=chosen.confidence,
            search_samples=(lo, hi), fallback=chosen.method == "timestamp_bound",
        )

    def quiet_edge(self, point):
        decision = self.find_cut(
            point, search_min=point - round(0.02 * self.sr),
            search_max=point + round(0.02 * self.sr),
        )
        return not decision.fallback and abs(decision.sample - point) <= round(0.02 * self.sr)


class ContextExpander:
    """Expand one side by a requested range without crossing hard bounds."""

    def __init__(self, finder):
        self.finder = finder
        self.sr = finder.sr

    def expand(self, anchor, direction, preferred_seconds,
               minimum_seconds=0.0, maximum_seconds=None, hard_bound=None):
        if direction not in {"left", "right"}:
            raise ValueError("direction must be left or right")
        preferred = max(0, round(float(preferred_seconds) * self.sr))
        minimum = max(0, round(float(minimum_seconds) * self.sr))
        maximum = preferred if maximum_seconds is None else max(
            0, round(float(maximum_seconds) * self.sr))
        minimum = min(minimum, maximum)
        preferred = min(max(preferred, minimum), maximum)
        anchor = int(anchor)
        if not 0 <= anchor <= len(self.finder.waveform):
            raise ValueError("context anchor is outside the waveform")
        if direction == "left":
            hard_bound = 0 if hard_bound is None else max(0, int(hard_bound))
            if hard_bound > anchor:
                raise ValueError("left hard bound is after the context anchor")
            lo, hi = max(hard_bound, anchor - maximum), max(hard_bound, anchor - minimum)
            desired = min(max(anchor - preferred, lo), hi)
        else:
            hard_bound = (len(self.finder.waveform) if hard_bound is None
                          else min(len(self.finder.waveform), int(hard_bound)))
            if hard_bound < anchor:
                raise ValueError("right hard bound is before the context anchor")
            lo, hi = min(hard_bound, anchor + minimum), min(hard_bound, anchor + maximum)
            desired = min(max(anchor + preferred, lo), hi)

        decision = self.finder.find_cut(
            desired, direction="both", search_min=lo, search_max=hi,
            hard_bounds=(min(lo, hi), max(lo, hi)),
        )
        expanded = abs(anchor - decision.sample)
        return ExpansionDecision(
            anchor_sample=anchor, boundary_sample=decision.sample,
            direction=direction, requested_samples=preferred,
            expanded_samples=expanded, method=decision.method,
            confidence=decision.confidence, hard_bound_sample=hard_bound,
            fallback=decision.fallback,
        )
