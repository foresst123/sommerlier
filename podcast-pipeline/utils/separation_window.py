"""Dựng cửa sổ hai người nói, giữ nguồn và ánh xạ theo mẫu âm thanh.

Mô-đun chỉ cần NumPy. Silero được truyền vào từ model đã tải; khi không có
Silero hoặc suy luận lỗi, dùng khoảng năng lượng thấp và khe giữa từ/âm tiết
để tìm điểm cắt an toàn.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from utils.acoustic_boundary import AcousticBoundaryFinder, ContextExpander

POLICY_VERSION = "ranked-boundary-context-padding-v9"


def clean_segments(segments):
    """Loại nguyên segment có bất kỳ giao dương nào, kể cả cùng speaker."""
    ordered = sorted(enumerate(segments), key=lambda item: item[1].start)
    active, dirty = [], set()
    for index, seg in ordered:
        active = [(i, s) for i, s in active if s.end > seg.start]
        if seg.end <= seg.start:
            dirty.add(index)
            continue
        for other_index, other in active:
            if min(seg.end, other.end) > max(seg.start, other.start):
                dirty.update((index, other_index))
        active.append((index, seg))
    return [seg for i, seg in enumerate(segments) if i not in dirty]


def merge_ranges(ranges):
    """Hợp các khoảng giao/chạm nhau, không lấp khoảng trống."""
    result = []
    for a, b in sorted(ranges):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def intersect(ranges, lo, hi):
    return merge_ranges((max(a, lo), min(b, hi)) for a, b in ranges)


def subtract(ranges, blockers):
    result = []
    for lo, hi in merge_ranges(ranges):
        for a, b in merge_ranges(blockers):
            if b <= lo:
                continue
            if a >= hi:
                break
            if a > lo:
                result.append((lo, a))
            lo = max(lo, b)
        if lo < hi:
            result.append((lo, hi))
    return result


class _LegacyAcousticCuts:
    """Tìm điểm cắt an toàn từ VAD, pause và khe năng lượng giữa từ/âm tiết."""

    def __init__(self, waveform, sr, vad=None):
        self.waveform, self.sr, self.vad = waveform, sr, vad
        self.cache = {}

    def analyse(self, lo, hi):
        """Trả cuts, voiced và phương pháp; ưu tiên pause thật rồi mới dùng local energy valley."""
        key = (int(lo), int(hi))
        if key in self.cache:
            return self.cache[key]

        lo, hi = max(0, key[0]), min(len(self.waveform), key[1])
        if hi <= lo:
            result = ({lo: "segment", hi: "segment"}, [], "energy")
            self.cache[key] = result
            return result

        wave = self.waveform[lo:hi]
        voiced, method = None, "energy"

        if self.vad is not None:
            try:
                timestamps = self.vad.get_speech_timestamps(wave, sampling_rate=self.sr)
                voiced = merge_ranges((lo + max(0, int(t["start"])), min(hi, lo + int(t["end"]))) for t in timestamps)
                method = "silero"
            except Exception:
                voiced, method = None, "energy"

        energy_frame = max(1, round(self.sr * 0.010))
        starts = np.arange(0, len(wave), energy_frame, dtype=np.int64)
        frame_rms = None
        if len(starts):
            squared = np.square(wave.astype(np.float64))
            counts = np.minimum(energy_frame, len(wave) - starts)
            sums = np.add.reduceat(squared, starts)
            frame_rms = np.sqrt(sums / np.maximum(counts, 1))

        if voiced is None:
            if frame_rms is not None and len(frame_rms):
                p90 = float(np.percentile(frame_rms, 90))
                threshold = max(1e-5, p90 * 0.10)
                voiced = merge_ranges((lo + int(start), min(hi, lo + int(start) + energy_frame)) for start, value in zip(starts, frame_rms) if value > threshold)
            else:
                voiced = []

        pauses = subtract([(lo, hi)], voiced)
        cuts = {lo: "segment", hi: "segment"}
        min_pause_samples = max(1, round(self.sr * 0.020))
        for a, b in pauses:
            if b - a >= min_pause_samples:
                cuts[(a + b) // 2] = method

        valley_frame = max(1, round(self.sr * 0.010))
        valley_hop = max(1, round(self.sr * 0.005))
        if len(wave) >= valley_frame * 3:
            valley_positions = np.arange(0, len(wave) - valley_frame + 1, valley_hop, dtype=np.int64)
            squared = np.square(wave.astype(np.float64))
            cumsum = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(squared, dtype=np.float64)))
            ends = valley_positions + valley_frame
            energies = cumsum[ends] - cumsum[valley_positions]
            valley_rms = np.sqrt(energies / valley_frame + 1e-12)

            if len(valley_rms) >= 3:
                global_ref = float(np.percentile(valley_rms, 75))
                absolute_floor = 1e-5
                neighbourhood_radius = 6
                candidates = []

                for i in range(1, len(valley_rms) - 1):
                    here = float(valley_rms[i])
                    if here > valley_rms[i - 1] or here > valley_rms[i + 1]:
                        continue

                    left_i = max(0, i - neighbourhood_radius)
                    right_i = min(len(valley_rms), i + neighbourhood_radius + 1)
                    neighbourhood = valley_rms[left_i:right_i]
                    if len(neighbourhood) < 3:
                        continue

                    local_ref = float(np.percentile(neighbourhood, 75))
                    if local_ref <= absolute_floor:
                        continue

                    valley_ratio = here / max(local_ref, absolute_floor)
                    if valley_ratio > 0.60:
                        continue
                    if global_ref > absolute_floor and here > global_ref * 0.55:
                        continue

                    point = lo + int(valley_positions[i] + valley_frame // 2)
                    edge_guard = max(self.sr // 100, valley_frame)
                    if point - lo < edge_guard or hi - point < edge_guard:
                        continue
                    if any(a <= point <= b and b - a >= min_pause_samples for a, b in pauses):
                        continue
                    candidates.append((valley_ratio, here, point))

                if candidates:
                    candidates.sort(key=lambda x: x[2])
                    cluster_distance = max(1, round(self.sr * 0.040))
                    clusters, current = [], []
                    for candidate in candidates:
                        if current and candidate[2] - current[-1][2] > cluster_distance:
                            clusters.append(current)
                            current = []
                        current.append(candidate)
                    if current:
                        clusters.append(current)
                    for cluster in clusters:
                        _, _, point = min(cluster, key=lambda x: (x[0], x[1]))
                        cuts.setdefault(point, "word_gap")

        result = (cuts, voiced, method)
        self.cache[key] = result
        return result

    def quiet_edge(self, point):
        """Biên tính toán chỉ được cắt khi có khoảng năng lượng thấp quanh nó."""
        radius = max(1, round(0.02 * self.sr))
        lo, hi = max(0, point - radius), min(len(self.waveform), point + radius)
        nearby = self.waveform[max(0, point - self.sr // 2):min(len(self.waveform), point + self.sr // 2)]
        if hi <= lo or not len(nearby):
            return False
        quiet = float(np.sqrt(np.mean(self.waveform[lo:hi].astype(np.float64) ** 2)))
        reference = float(np.sqrt(np.mean(nearby.astype(np.float64) ** 2)))
        return quiet <= max(1e-5, reference * 0.1)


# Compatibility export for callers and old checkpoints. All active boundary
# decisions now use the shared implementation in acoustic_boundary.py.
AcousticCuts = AcousticBoundaryFinder


@dataclass(frozen=True)
class Piece:
    start: int
    end: int
    speaker: str
    source: str
    kind: str
    start_cut: str = "segment"
    end_cut: str = "segment"


@dataclass
class Window:
    audio: np.ndarray
    core: tuple
    probes: dict
    layout: dict


@dataclass
class WindowPlan:
    window: object
    core_source_samples: tuple
    reason: str
    detail: str
    actions: list


class WindowPlanner:
    """Dựng window theo thứ tự overlap, context liên tục, rồi clean padding."""

    def __init__(self, segments, pairs, waveform, sr, music_map=None, seams=(), vad=None, context_seconds=2.0, search_seconds=400.0):
        self.segments, self.pairs = segments, pairs
        self.waveform, self.sr = waveform, sr
        self.music_map = music_map
        self.seams = [int(t * sr) for t in seams]
        self.context = round(context_seconds * sr)
        self.search = search_seconds * sr
        self.target = round(15.0 * sr)
        self.fade = round(0.020 * sr)
        self.minimum = round(1.5 * sr)
        # 15s is a hard model limit. A centred core and 1-2s of continuous
        # context are preferences that may move when a file edge or SP3 blocks.
        self.cuts = AcousticBoundaryFinder(waveform, sr, vad)
        self.expander = ContextExpander(self.cuts)
        self.by_speaker = {}
        for s in segments:
            self.by_speaker.setdefault(s.speaker, []).append((int(s.start * sr), int(s.end * sr)))
        self.by_speaker = {s: merge_ranges(v) for s, v in self.by_speaker.items()}
        self.clean = clean_segments(segments)
        self.reason = "no_window"
        self.detail = ""
        self.actions = []
        self._support_cache = {}

    def _record(self, action, **details):
        self.actions.append({"step": len(self.actions) + 1,
                             "action": action, **details})

    def _solo(self, speaker, lo, hi):
        others = [r for s, rs in self.by_speaker.items() if s != speaker for r in rs]
        return subtract(intersect(self.by_speaker.get(speaker, []), lo, hi), others)

    def _voiced(self, spans):
        result = []
        for lo, hi in spans:
            result.extend(self.cuts.analyse(lo, hi)[1])
        return merge_ranges(result)

    def _support(self, speaker, centre):
        key = (speaker, centre)
        if key in self._support_cache:
            return self._support_cache[key]

        candidates = [s for s in self.clean if s.speaker == speaker and abs((s.start + s.end) * self.sr / 2 - centre) <= self.search]
        candidates.sort(key=lambda s: abs((s.start + s.end) * self.sr / 2 - centre))
        pieces = []

        for seg in candidates[:12]:
            spans = [(seg.start, seg.end)]
            if self.music_map:
                spans = self.music_map.clean_parts(seg.start, seg.end)

            for start, end in spans:
                lo, hi = max(0, int(start * self.sr)), min(len(self.waveform), int(end * self.sr))
                # Recording seams are not speech boundaries. Keeping one
                # continuous interval lets a clean support piece cross a seam.
                limits = [lo, hi]

                for a, b in zip(limits, limits[1:]):
                    cuts, voiced, _ = self.cuts.analyse(a, b)
                    cuts = dict(cuts)

                    for edge, original in ((a, int(seg.start * self.sr)), (b, int(seg.end * self.sr))):
                        if edge != original:
                            if self.cuts.quiet_edge(edge):
                                cuts[edge] = "energy"
                            else:
                                cuts.pop(edge, None)

                    points = sorted(cuts)
                    if len(points) > 64:
                        keep = {points[0], points[-1]}
                        keep.update(points[i] for i in np.linspace(0, len(points) - 1, 62, dtype=int))
                        points = sorted(keep)

                    for x, y in combinations(points, 2):
                        if self.minimum <= y - x <= self.target + self.fade:
                            voice = sum(v - u for u, v in intersect(voiced, x, y))
                            if voice >= self.sr:
                                pieces.append(Piece(x, y, speaker, str(seg.index), "support", cuts[x], cuts[y]))

        buckets = {}
        for p in pieces:
            bucket = round((p.end - p.start) / self.sr * 10)
            buckets.setdefault(bucket, []).append(p)

        result = []
        for bucket in sorted(buckets):
            result.extend(sorted(buckets[bucket], key=lambda p: abs((p.start + p.end) / 2 - centre))[:2])

        self._support_cache[key] = result
        return result

    def _trim_support(self, piece, minimum, maximum, centre):
        """Chỉ khi support nguyên bản không vừa budget mới cắt đoạn dài tại safe cut."""
        if maximum < minimum or maximum <= 0:
            return []

        cuts, voiced, _ = self.cuts.analyse(piece.start, piece.end)
        cuts = dict(cuts)
        cuts[piece.start], cuts[piece.end] = piece.start_cut, piece.end_cut
        points = sorted(cuts)

        if len(points) > 64:
            keep = {points[0], points[-1]}
            keep.update(points[i] for i in np.linspace(0, len(points) - 1, 62, dtype=int))
            points = sorted(keep)

        result = []
        for x, y in combinations(points, 2):
            added = y - x - self.fade
            if y - x < self.minimum or not (minimum <= added <= maximum):
                continue
            voice = sum(v - u for u, v in intersect(voiced, x, y))
            if voice < self.sr:
                continue
            trimmed = Piece(x, y, piece.speaker, piece.source, piece.kind, cuts[x], cuts[y])
            rank = (maximum - added, abs((x + y) / 2 - centre))
            result.append((rank, trimmed))

        result.sort(key=lambda item: item[0])
        return [p for _, p in result]

    def _rank_sequences(self, options, centre):
        buckets = {}
        for seq in options:
            added = sum(p.end - p.start - self.fade for p in seq)
            bucket = round(added / self.sr * 10)
            rank = (len(seq), -added, sum(abs((p.start + p.end) / 2 - centre) for p in seq))
            if bucket not in buckets or rank < buckets[bucket][0]:
                buckets[bucket] = (rank, seq)
        return [item[1] for item in buckets.values()]

    def _sequences(self, speaker, centre, minimum, maximum, base,
                   floor=0, ceiling=None):
        """Ưu tiên support nguyên bản vừa budget; chỉ khi không có mới trim support dài."""
        if maximum < minimum or maximum < 0:
            return []

        ceiling = len(self.waveform) if ceiling is None else ceiling
        available = [
            p for p in self._support(speaker, centre)
            if floor <= p.start and p.end <= ceiling
            and (p.end <= base.start or p.start >= base.end)
        ]
        empty = [()] if minimum <= 0 else []
        native = []

        for p in available:
            added = p.end - p.start - self.fade
            if minimum <= added <= maximum:
                native.append((p,))

        for i, p in enumerate(available):
            ap = p.end - p.start - self.fade
            if ap > maximum:
                continue
            for q in available[i + 1:]:
                if p.source == q.source:
                    continue
                aq = q.end - q.start - self.fade
                if minimum <= ap + aq <= maximum:
                    native.append(tuple(sorted((p, q), key=lambda z: z.start)))

        if native:
            return self._rank_sequences(empty + native, centre)

        trimmed = []
        for p in available:
            if p.end - p.start - self.fade <= maximum:
                continue
            for q in self._trim_support(p, minimum, maximum, centre)[:6]:
                trimmed.append((q,))

        if trimmed:
            return self._rank_sequences(empty + trimmed, centre)
        return empty



    def _assemble(self, prefix, base, suffix, speakers, core_lo, core_hi, solos):
        pieces = list(prefix) + [base] + list(suffix)
        output = None
        maps, probes = [], {s: [] for s in speakers}
        base_offset = None

        for k, p in enumerate(pieces):
            chunk = self.waveform[p.start:p.end].astype(np.float32, copy=True)
            start = 0 if output is None else len(output) - self.fade

            if output is None:
                output = chunk
            else:
                ramp = np.linspace(0, 1, self.fade, dtype=np.float32)
                output[-self.fade:] = output[-self.fade:] * (1 - ramp) + chunk[:self.fade] * ramp
                output = np.concatenate((output, chunk[self.fade:]))

            end = start + len(chunk)
            safe_lo = p.start + (self.fade if k else 0)
            safe_hi = p.end - (self.fade if k + 1 < len(pieces) else 0)

            for speaker in speakers:
                ranges = solos[speaker] if p.kind == "base" else (self.cuts.analyse(p.start, p.end)[1] if p.speaker == speaker else [])
                for a, b in intersect(ranges, safe_lo, safe_hi):
                    probes[speaker].append((start + a - p.start, start + b - p.start))

            if p.kind == "base":
                base_offset = start

            maps.append({"kind": p.kind, "speaker": p.speaker, "source_segment": p.source, "source_samples": [p.start, p.end], "window_samples": [start, end], "cuts": [p.start_cut, p.end_cut]})

        core = (base_offset + core_lo - base.start, base_offset + core_hi - base.start)
        return Window(output, core, probes, {
            "sample_rate": self.sr,
            "duration_seconds": len(output) / self.sr,
            "core_samples": list(core),
            "core_source_samples": [core_lo, core_hi],
            "core_start_seconds": core[0] / self.sr,
            "pieces": maps,
            "crossfade_samples": self.fade,
            "probe_samples": {s: [list(r) for r in rs] for s, rs in probes.items()},
        })
    def _piece_voice_samples(self, piece):
        """Ước lượng số sample có tiếng nói trong một support piece sạch."""
        _, voiced, _ = self.cuts.analyse(piece.start, piece.end)
        return sum(b - a for a, b in intersect(voiced, piece.start, piece.end))

    def _sequence_added_samples(self, seq):
        """Số sample thực sự được thêm sau crossfade khi seq đứng trước/sau base."""
        return sum(max(0, p.end - p.start - self.fade) for p in seq)

    def _sequence_voice_samples(self, seq):
        return sum(self._piece_voice_samples(p) for p in seq)

    def _window_samples(self, prefix, base, suffix):
        pieces = list(prefix) + [base] + list(suffix)
        if not pieces:
            return 0
        return sum(p.end - p.start for p in pieces) - self.fade * (len(pieces) - 1)

    def _safe_cut_outward(self, desired, lo, hi, side):
        """Nới timestamp ra acoustic cut gần nhất, tuyệt đối không cắt vào trong."""
        if hi <= lo:
            return desired, "timestamp_bound"
        decision = self.cuts.find_cut(
            desired, direction=side, search_min=lo, search_max=hi,
            hard_bounds=(lo, hi),
        )
        return decision.sample, decision.method

    def _expand_context(self, start, end, floor, ceiling, left_need, right_need):
        """Expand both sides through the shared, budget-aware primitive."""
        room = min(
            max(0, self.target - (end - start)),
            max(0, int(left_need)) + max(0, int(right_need)),
        )
        if room <= 0:
            return start, end, "none", "none"

        left_take = min(max(0, int(left_need)), max(0, start - floor), room)
        room -= left_take
        right_take = min(max(0, int(right_need)), max(0, ceiling - end), room)
        room -= right_take

        # Nếu một phía bị file/SP3 chặn, dồn budget sang phía còn lại.
        if room > 0:
            extra_left = min(max(0, start - floor - left_take), room)
            left_take += extra_left
            room -= extra_left
        if room > 0:
            extra_right = min(max(0, ceiling - end - right_take), room)
            right_take += extra_right

        left = self.expander.expand(
            start, "left", left_take / self.sr,
            minimum_seconds=max(0.0, left_take / self.sr - 0.5),
            maximum_seconds=left_take / self.sr,
            hard_bound=floor,
        )
        right = self.expander.expand(
            end, "right", right_take / self.sr,
            minimum_seconds=max(0.0, right_take / self.sr - 0.5),
            maximum_seconds=right_take / self.sr,
            hard_bound=ceiling,
        )
        return left.boundary_sample, right.boundary_sample, left.method, right.method

    def _split_core_bounds(self, core_lo, core_hi):
        """Partition an overlap into quality-sized cores using ranked cuts."""
        hard_limit = self.target
        quality_limit = round(12.0 * self.sr)
        min_piece = min(round(2.5 * self.sr), max(1, (core_hi - core_lo) // 3))
        pending = [(core_lo, core_hi)]
        result = []

        while pending:
            lo, hi = pending.pop(0)
            width = hi - lo
            mandatory = width > hard_limit
            optional = quality_limit < width <= hard_limit
            if not mandatory and not optional:
                result.append((lo, hi))
                continue

            midpoint = (lo + hi) // 2
            radius = min(round(2.0 * self.sr), max(1, width // 3))
            search_lo = max(lo + min_piece, midpoint - radius)
            search_hi = min(hi - min_piece, midpoint + radius)
            candidates = self.cuts.find_candidates(
                midpoint, direction="both", search_min=search_lo,
                search_max=search_hi, hard_bounds=(search_lo, search_hi),
                include_fallback=True,
            )
            ranked = []
            for candidate in candidates:
                left, right = candidate.sample - lo, hi - candidate.sample
                if left < min_piece or right < min_piece:
                    continue
                overflow = max(0, max(left, right) - hard_limit)
                ranked.append(((
                    overflow,
                    candidate.method == "timestamp_bound",
                    -candidate.confidence,
                    abs(left - right),
                ), candidate))
            ranked.sort(key=lambda item: item[0])
            chosen = ranked[0][1] if ranked else None

            # A quality split below 15s is accepted only with real acoustic
            # evidence. A mandatory split always falls back to the midpoint.
            if optional and (
                chosen is None
                or chosen.method == "timestamp_bound"
                or chosen.confidence < 0.65
            ):
                result.append((lo, hi))
                continue
            cut = chosen.sample if chosen is not None else midpoint
            cut = min(max(cut, lo + 1), hi - 1)
            self._record(
                "split_overlap_core",
                mandatory=mandatory,
                source_samples=[lo, hi],
                split_sample=cut,
                split_seconds=cut / self.sr,
                method=(chosen.method if chosen is not None else "timestamp_bound"),
                confidence=(chosen.confidence if chosen is not None else 0.0),
            )
            pending[0:0] = [(lo, cut), (cut, hi)]

        return sorted(result)

    def build_many(self, group):
        """Build every quality-sized core, retaining failures per core."""
        self.actions = []
        if not group:
            self.reason, self.detail = "no_window", "invalid_overlap_group"
            self._record("reject", reason=self.detail)
            return [WindowPlan(None, (0, 0), self.reason, self.detail,
                               list(self.actions))]

        core_lo = max(0, int(np.floor(
            min(pair["overlap_start"] for pair in group) * self.sr
        )))
        core_hi = min(len(self.waveform), int(np.ceil(
            max(pair["overlap_end"] for pair in group) * self.sr
        )))
        split_actions = []
        bounds = self._split_core_bounds(core_lo, core_hi)
        split_actions.extend(self.actions)
        plans = []
        for index, bound in enumerate(bounds):
            initial = list(split_actions)
            initial.append({
                "step": len(initial) + 1,
                "action": "select_core_chunk",
                "chunk_index": index,
                "chunk_count": len(bounds),
                "source_samples": list(bound),
            })
            window = self.build(group, core_bounds=bound, initial_actions=initial)
            plans.append(WindowPlan(
                window=window,
                core_source_samples=bound,
                reason=self.reason,
                detail=self.detail,
                actions=list(self.actions),
            ))
        return plans

    def build(self, group, core_bounds=None, initial_actions=None):
        """Dựng best-effort window; chỉ speaker thứ ba trong core mới chặn."""
        self.reason, self.detail = "no_window", "invalid_overlap_group"
        self.actions = list(initial_actions or [])
        if not group:
            self._record("reject", reason=self.detail)
            return None

        first = min(group, key=lambda p: (p["overlap_start"], p["overlap_end"]))
        speaker_a = first["seg1"]["speaker"]
        speaker_b = first["seg2"]["speaker"]
        if speaker_a == speaker_b:
            self.detail = "same_speaker_overlap"
            self._record("reject", reason=self.detail, speaker=str(speaker_a))
            return None

        speakers = tuple(sorted((speaker_a, speaker_b), key=lambda value: str(value)))
        target_set = frozenset(speakers)
        for pair in group:
            pair_set = frozenset((pair["seg1"]["speaker"], pair["seg2"]["speaker"]))
            if pair_set != target_set:
                self.detail = "mixed_speaker_pairs"
                self._record("reject", reason=self.detail)
                return None

        n_samples = len(self.waveform)
        if core_bounds is None:
            core_lo = max(0, int(np.floor(min(
                p["overlap_start"] for p in group
            ) * self.sr)))
            core_hi = min(n_samples, int(np.ceil(max(
                p["overlap_end"] for p in group
            ) * self.sr)))
        else:
            core_lo = max(0, int(core_bounds[0]))
            core_hi = min(n_samples, int(core_bounds[1]))
        if core_hi <= core_lo:
            self.detail = "overlap_shorter_than_one_sample"
            self._record("reject", reason=self.detail)
            return None
        if core_hi - core_lo > self.target:
            self.detail = "core_exceeds_15s_use_build_many"
            self._record("reject", reason=self.detail)
            return None

        self._record(
            "keep_overlap_core",
            speakers=[str(s) for s in speakers],
            pair_count=len(group),
            source_samples=[core_lo, core_hi],
            source_seconds=[core_lo / self.sr, core_hi / self.sr],
        )

        seg_indices = sorted({str(p[side]["index"]) for p in group for side in ("seg1", "seg2")})
        seg_starts = [float(p[side]["start"]) for p in group for side in ("seg1", "seg2")]
        seg_ends = [float(p[side]["end"]) for p in group for side in ("seg1", "seg2")]
        segment_envelope_raw = [
            max(0, int(np.floor(min(seg_starts) * self.sr))),
            min(n_samples, int(np.ceil(max(seg_ends) * self.sr))),
        ]
        # The overlap core is mandatory. The full diarization envelope is useful
        # evidence, but it may be much wider than the model input and therefore
        # cannot itself be a hard window boundary.
        timestamp_lo_raw, timestamp_hi_raw = core_lo, core_hi
        self._record(
            "observe_segment_envelope",
            segment_indices=seg_indices,
            source_samples=segment_envelope_raw,
            source_seconds=[value / self.sr for value in segment_envelope_raw],
            mandatory=False,
        )

        # SP3 là hard boundary duy nhất. Seam và overlap khác của A/B không chặn.
        floor, ceiling = 0, n_samples
        third_speaker_bounds = []
        for speaker, ranges in self.by_speaker.items():
            if speaker in target_set:
                continue
            for a, b in ranges:
                if a < core_hi and b > core_lo:
                    self.reason = "multi_speaker"
                    self.detail = f"third_speaker_in_core:{speaker}"
                    self._record(
                        "reject",
                        reason=self.detail,
                        third_speaker=str(speaker),
                        blocker_samples=[a, b],
                    )
                    return None
                if b <= core_lo and b > floor:
                    floor = b
                    third_speaker_bounds.append(("left", str(speaker), a, b))
                elif a >= core_hi and a < ceiling:
                    ceiling = a
                    third_speaker_bounds.append(("right", str(speaker), a, b))

        timestamp_lo = max(floor, timestamp_lo_raw)
        timestamp_hi = min(ceiling, timestamp_hi_raw)
        self._record(
            "apply_third_speaker_boundaries",
            floor_ceiling_samples=[floor, ceiling],
            envelope_before=[timestamp_lo_raw, timestamp_hi_raw],
            envelope_after=[timestamp_lo, timestamp_hi],
            context_reduced=(
                timestamp_lo != timestamp_lo_raw
                or timestamp_hi != timestamp_hi_raw
            ),
        )
        if timestamp_lo > core_lo or timestamp_hi < core_hi:
            self.reason = "multi_speaker"
            self.detail = "third_speaker_clips_core_context"
            self._record("reject", reason=self.detail)
            return None

        core_width = core_hi - core_lo
        left_room = max(0, (self.target - core_width) // 2)
        right_room = max(0, self.target - core_width - left_room)

        def preferred_context(room):
            if room < self.sr:
                return room
            if room <= 3 * self.sr:
                return min(room, self.sr, self.context)
            return min(room, self.context)

        left_preferred = min(preferred_context(left_room), core_lo - floor)
        right_preferred = min(preferred_context(right_room), ceiling - core_hi)
        tolerance = round(0.35 * self.sr)
        left = self.expander.expand(
            core_lo, "left", left_preferred / self.sr,
            minimum_seconds=max(0.0, (left_preferred - round(0.5 * self.sr)) / self.sr),
            maximum_seconds=min(left_room, left_preferred + tolerance) / self.sr,
            hard_bound=floor,
        )
        right = self.expander.expand(
            core_hi, "right", right_preferred / self.sr,
            minimum_seconds=max(0.0, (right_preferred - round(0.5 * self.sr)) / self.sr),
            maximum_seconds=min(right_room, right_preferred + tolerance) / self.sr,
            hard_bound=ceiling,
        )
        base_start, start_cut = left.boundary_sample, left.method
        base_end, end_cut = right.boundary_sample, right.method
        if not (floor <= base_start <= core_lo < core_hi <= base_end <= ceiling):
            base_start, base_end = core_lo, core_hi
            start_cut = end_cut = "timestamp_bound"
        self._record(
            "expand_preferred_context",
            core_samples=[core_lo, core_hi],
            base_samples=[base_start, base_end],
            cut_methods=[start_cut, end_cut],
            cut_confidence=[left.confidence, right.confidence],
            side_budget_seconds=[left_room / self.sr, right_room / self.sr],
            preferred_seconds=[left_preferred / self.sr, right_preferred / self.sr],
            actual_seconds=[
                (core_lo - base_start) / self.sr,
                (base_end - core_hi) / self.sr,
            ],
        )

        def make_base(start, end, left_cut, right_cut):
            return Piece(
                start=start,
                end=end,
                speaker="+".join(map(str, speakers)),
                source="+".join(seg_indices),
                kind="base",
                start_cut=left_cut,
                end_cut=right_cut,
            )

        def base_evidence(piece):
            voice_by_speaker = {
                s: intersect(self.by_speaker.get(s, []), piece.start, piece.end)
                for s in speakers
            }
            voiced_by_speaker = {
                s: [
                    r
                    for a, b in voice_by_speaker[s]
                    for r in intersect(self.cuts.analyse(a, b)[1], a, b)
                ]
                for s in speakers
            }
            solo_ranges = {
                s: self._voiced(self._solo(s, piece.start, piece.end))
                for s in speakers
            }
            voice_samples = {
                s: sum(y - x for x, y in intersect(
                    voiced_by_speaker[s], piece.start, piece.end
                ))
                for s in speakers
            }
            return solo_ranges, voice_samples

        def choose_padding(piece, base_voice):
            budget = max(0, self.target - (piece.end - piece.start))
            if budget <= 0:
                return (), (), None, None, dict(base_voice)

            centre = (core_lo + core_hi) // 2
            base_core_pos = core_lo - piece.start
            desired_core_start = max(0, (self.target - (core_hi - core_lo)) // 2)
            desired_right_room = (
                self.target - (core_hi - core_lo) - desired_core_start
            )
            prefix_budget = min(
                budget, max(0, desired_core_start - base_core_pos)
            )
            base_right_context = piece.end - core_hi
            suffix_budget = min(
                budget, max(0, desired_right_room - base_right_context)
            )
            candidates = []
            for prefix_speaker, suffix_speaker in (
                (speakers[0], speakers[1]),
                (speakers[1], speakers[0]),
            ):
                prefix_options = self._sequences(
                    prefix_speaker, centre, 0, prefix_budget, piece,
                    floor, ceiling,
                ) or [()]

                for prefix_seq in prefix_options:
                    prefix_added = self._sequence_added_samples(prefix_seq)
                    remaining = min(
                        max(0, budget - prefix_added), suffix_budget
                    )
                    suffix_options = self._sequences(
                        suffix_speaker, centre, 0, remaining, piece,
                        floor, ceiling,
                    ) or [()]
                    for suffix_seq in suffix_options:
                        predicted = self._window_samples(
                            prefix_seq, piece, suffix_seq
                        )
                        if predicted > self.target:
                            continue
                        final_core_start = prefix_added + base_core_pos
                        anchor_penalty = abs(final_core_start - desired_core_start)

                        total_voice = dict(base_voice)
                        total_voice[prefix_speaker] += self._sequence_voice_samples(prefix_seq)
                        total_voice[suffix_speaker] += self._sequence_voice_samples(suffix_seq)
                        va, vb = total_voice[speakers[0]], total_voice[speakers[1]]
                        balance_penalty = abs(np.log((va + 1.0) / (vb + 1.0)))
                        rank = (
                            anchor_penalty,
                            int(not prefix_seq) + int(not suffix_seq),
                            balance_penalty,
                            -predicted,
                        )
                        candidates.append((
                            rank, tuple(prefix_seq), tuple(suffix_seq),
                            prefix_speaker, suffix_speaker, total_voice,
                        ))

            if not candidates:
                return (), (), None, None, dict(base_voice)
            candidates.sort(key=lambda item: item[0])
            _, prefix_seq, suffix_seq, prefix_speaker, suffix_speaker, total_voice = candidates[0]
            return prefix_seq, suffix_seq, prefix_speaker, suffix_speaker, total_voice

        base = make_base(base_start, base_end, start_cut, end_cut)
        solos, base_voice = base_evidence(base)
        prefix, suffix, prefix_speaker, suffix_speaker, total_voice = choose_padding(
            base, base_voice
        )
        self._record(
            "select_clean_padding",
            phase="initial",
            base_seconds=(base.end - base.start) / self.sr,
            soft_target_seconds=self.target / self.sr,
            available_budget_seconds=max(
                0.0, (self.target - (base.end - base.start)) / self.sr
            ),
            padding_dropped_for_context=(base.end - base.start) >= self.target,
            prefix_speaker=None if prefix_speaker is None else str(prefix_speaker),
            suffix_speaker=None if suffix_speaker is None else str(suffix_speaker),
            prefix_source_samples=[[p.start, p.end] for p in prefix],
            suffix_source_samples=[[p.start, p.end] for p in suffix],
            prefix_seconds=self._sequence_added_samples(prefix) / self.sr,
            suffix_seconds=self._sequence_added_samples(suffix) / self.sr,
        )

        # Clean pad không bắt buộc. Nếu layout vẫn ngắn, lấy context nền liên
        # tục để bù, ưu tiên giữ midpoint của core tại giây 7.5.
        # Chạy tối đa hai lượt vì context mới có thể nuốt một support piece cũ.
        for pass_index in range(2):
            current = self._window_samples(prefix, base, suffix)
            remaining = max(0, self.target - current)
            if remaining <= 0:
                break
            final_core_start = (
                self._sequence_added_samples(prefix) + core_lo - base.start
            )
            desired_core_start = max(0, (self.target - (core_hi - core_lo)) // 2)
            left_need = min(remaining, max(0, desired_core_start - final_core_start))
            final_core_end_room = (
                self._sequence_added_samples(suffix) + base.end - core_hi
            )
            desired_right = self.target - (core_hi - core_lo) - desired_core_start
            right_need = min(
                remaining - left_need,
                max(0, desired_right - final_core_end_room),
            )
            if left_need + right_need < remaining:
                right_need += remaining - left_need - right_need
            new_start, new_end, new_start_cut, new_end_cut = self._expand_context(
                base.start, base.end, floor, ceiling, left_need, right_need
            )
            if new_start == base.start and new_end == base.end:
                self._record(
                    "context_fallback_blocked",
                    pass_index=pass_index + 1,
                    requested_seconds=remaining / self.sr,
                    floor_ceiling_samples=[floor, ceiling],
                )
                break
            old_base = [base.start, base.end]
            base = make_base(new_start, new_end, new_start_cut, new_end_cut)
            self._record(
                "expand_background_context",
                pass_index=pass_index + 1,
                requested_left_seconds=left_need / self.sr,
                requested_right_seconds=right_need / self.sr,
                base_before=old_base,
                base_after=[new_start, new_end],
                cut_methods=[new_start_cut, new_end_cut],
            )
            solos, base_voice = base_evidence(base)
            prefix, suffix, prefix_speaker, suffix_speaker, total_voice = choose_padding(
                base, base_voice
            )
            self._record(
                "reselect_clean_padding",
                pass_index=pass_index + 1,
                prefix_speaker=None if prefix_speaker is None else str(prefix_speaker),
                suffix_speaker=None if suffix_speaker is None else str(suffix_speaker),
                prefix_source_samples=[[p.start, p.end] for p in prefix],
                suffix_source_samples=[[p.start, p.end] for p in suffix],
                prefix_seconds=self._sequence_added_samples(prefix) / self.sr,
                suffix_seconds=self._sequence_added_samples(suffix) / self.sr,
            )

        result = self._assemble(
            prefix, base, suffix, speakers, core_lo, core_hi, solos
        )
        final_voice_seconds = {s: total_voice[s] / self.sr for s in speakers}
        min_voice = min(total_voice.values()) if total_voice else 0
        max_voice = max(total_voice.values()) if total_voice else 0
        ratio = max_voice / max(1, min_voice) if max_voice else 1.0
        prefix_added = self._sequence_added_samples(prefix)
        suffix_added = self._sequence_added_samples(suffix)
        left_context = max(0, core_lo - base.start)
        right_context = max(0, base.end - core_hi)
        mandatory_context_preserved = (
            base.start <= timestamp_lo_raw and base.end >= timestamp_hi_raw
        )
        self._record(
            "finalize_window",
            duration_seconds=len(result.audio) / self.sr,
            target_seconds=self.target / self.sr,
            target_exceeded=len(result.audio) > self.target,
            core_window_samples=list(result.core),
            base_source_samples=[base.start, base.end],
            mandatory_context_preserved=mandatory_context_preserved,
            context_was_never_reduced_for_budget=True,
            prefix_pad_seconds=prefix_added / self.sr,
            suffix_pad_seconds=suffix_added / self.sr,
        )

        result.layout.update({
            "policy": "overlap>ranked-context>clean-pad;missing-pad=>context;sp3-hard-bound",
            "policy_version": POLICY_VERSION,
            "involved_segments": seg_indices,
            "host_segment": "+".join(seg_indices),
            "host_source_samples": [base.start, base.end],
            "timestamp_source_samples": [timestamp_lo, timestamp_hi],
            "timestamp_source_samples_raw": [timestamp_lo_raw, timestamp_hi_raw],
            "mandatory_context_preserved": mandatory_context_preserved,
            "third_speaker_floor_ceiling": [floor, ceiling],
            "third_speaker_bounds_seen": third_speaker_bounds,
            "cut_method": [base.start_cut, base.end_cut],
            "estimated_voice_seconds": {
                str(s): final_voice_seconds[s] for s in speakers
            },
            "base_voice_seconds": {
                str(s): base_voice[s] / self.sr for s in speakers
            },
            "ratio": ratio,
            "core_position_seconds": result.core[0] / self.sr,
            "core_midpoint_seconds": (
                (result.core[0] + result.core[1]) / (2.0 * self.sr)
            ),
            "base_width_seconds": (base.end - base.start) / self.sr,
            "base_core_position_seconds": (core_lo - base.start) / self.sr,
            "context_seconds": [left_context / self.sr, right_context / self.sr],
            "context_shortfall_seconds": [
                max(0.0, (left_preferred - left_context) / self.sr),
                max(0.0, (right_preferred - right_context) / self.sr),
            ],
            "core_expanded": base.start < timestamp_lo or base.end > timestamp_hi,
            "full_overlap": timestamp_lo == core_lo and timestamp_hi == core_hi,
            "target_is_soft": False,
            "prefix_speaker": None if prefix_speaker is None else str(prefix_speaker),
            "suffix_speaker": None if suffix_speaker is None else str(suffix_speaker),
            "pad_seconds": {
                "prefix": prefix_added / self.sr,
                "suffix": suffix_added / self.sr,
            },
            "overlaps": [
                [p["overlap_start"], p["overlap_end"]] for p in group
            ],
            "actions": list(self.actions),
        })
        self.reason, self.detail = "ok", "best_effort_window"
        return result
