"""Dựng cửa sổ hai người nói, giữ nguồn và ánh xạ theo mẫu âm thanh.

Mô-đun chỉ cần NumPy. Silero được truyền vào từ model đã tải; khi không có
Silero hoặc suy luận lỗi, dùng khoảng năng lượng thấp và khe giữa từ/âm tiết
để tìm điểm cắt an toàn.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

POLICY_VERSION = "priority-balanced-15s-v7"


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


class AcousticCuts:
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


class WindowPlanner:
    """Tìm base/support cho cửa sổ <=15s; base rộng 3-10s và core cuối ở 5-8s."""

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
        # How far the base may start before the core. The floor was 3.0s, which
        # combined with the 5-8s anchor to force every window's left context to
        # 5-8s: with no prefix piece the core's position in the window just is
        # its position in the base, so satisfying the anchor meant starting the
        # base that far back. The right side then got whatever the 15s budget
        # had left, which measured 1.39s on average against 5.58s on the left.
        # At 1.5s the base can sit close to the core and a prefix support piece
        # carries the core out to the anchor instead.
        # base occupies window seconds 3-10 (7s budget).
        # base_core_min: minimum left context before core inside base.
        # At 1.5s this is the normal floor; the edge-case logic in build()
        # may go as low as BASE_CORE_EDGE_MIN when the core sits near the
        # host boundary and there is no room on that side.
        self.base_core_min = round(1.5 * sr)
        self.base_core_max = round(10.0 * sr)
        # Absolute minimum base length (3s). Below this the separator has
        # too little host context to anchor its estimate.
        self.base_min_total = round(3.0 * sr)
        # When the core span is shorter than this, expand ±short_core_pad
        # on each side before searching for base cuts, so Sidon sees enough
        # audio around the overlap itself.
        # Only truly micro overlaps (boundary jitter, 19-60ms from the old
        # VAD bug) need the pad. Real backchannels start at 0.24s; anything
        # above 0.2s is genuine speech and Sidon can anchor on it without
        # extra padding.
        self.short_core_threshold = round(0.2 * sr)
        self.short_core_pad = round(2.0 * sr)
        # The base is budgeted roughly window seconds 3-10; past that, extra
        # host is not worth having.
        self.base_target = round(7.0 * sr)
        self.final_core_min = round(5.0 * sr)
        self.final_core_max = round(8.0 * sr)
        self.cuts = AcousticCuts(waveform, sr, vad)
        self.by_speaker = {}
        for s in segments:
            self.by_speaker.setdefault(s.speaker, []).append((int(s.start * sr), int(s.end * sr)))
        self.by_speaker = {s: merge_ranges(v) for s, v in self.by_speaker.items()}
        self.clean = clean_segments(segments)
        self.reason = "no_window"
        self.detail = ""
        self._support_cache = {}

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
                limits = [lo] + [s for s in self.seams if lo < s < hi] + [hi]

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

    def _sequences(self, speaker, centre, minimum, maximum, base):
        """Ưu tiên support nguyên bản vừa budget; chỉ khi không có mới trim support dài."""
        if maximum < minimum or maximum < 0:
            return []

        available = [p for p in self._support(speaker, centre) if p.end <= base.start or p.start >= base.end]
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

    def _safe_cut_near(self, desired, lo, hi, side, core_lo, core_hi):
        """Tìm acoustic cut gần desired; timestamp chỉ là fallback.

        Base được phép nới nhẹ ra ngoài timestamp để tránh cắt cụt, nhưng tuyệt
        đối không vượt floor/ceiling do speaker thứ ba tạo ra.
        """
        if hi <= lo:
            return desired, "timestamp_bound"

        cuts, _, _ = self.cuts.analyse(lo, hi)
        candidates = []
        for point, kind in cuts.items():
            # Hai đầu do analyse() tự chèn không phải acoustic evidence thật.
            if kind == "segment":
                continue
            if side == "left" and point <= core_lo:
                candidates.append((point, kind))
            elif side == "right" and point >= core_hi:
                candidates.append((point, kind))

        if not candidates:
            return desired, "timestamp_bound"

        point, kind = min(candidates, key=lambda item: abs(item[0] - desired))
        return point, kind

    def build(self, group):
        """Dựng cửa sổ <=15s theo thứ tự ưu tiên:

        1) overlap core: không bao giờ cắt;
        2) context thật theo timestamp của hai segment;
        3) pad/support sạch để cân bằng hai speaker và đưa core về khoảng 5-8s;
        4) phần mở rộng ngoài timestamp chỉ dùng để tránh cửa sổ quá ngắn/cắt cụt.

        Speaker thứ ba là hard boundary duy nhất của phần base. Nếu speaker thứ
        ba chồng trực tiếp vào overlap core thì window bị loại.
        """
        self.reason, self.detail = "no_window", "no_safe_layout"
        if not group:
            return None

        # ------------------------------------------------------------------
        # 1. Xác định đúng hai speaker và overlap core bắt buộc phải giữ nguyên.
        # ------------------------------------------------------------------
        first = min(group, key=lambda p: (p["overlap_start"], p["overlap_end"]))
        speaker_a = first["seg1"]["speaker"]
        speaker_b = first["seg2"]["speaker"]
        if speaker_a == speaker_b:
            self.detail = "same_speaker_overlap"
            return None

        speakers = tuple(sorted((speaker_a, speaker_b), key=lambda value: str(value)))
        target_set = frozenset(speakers)

        # Một group chỉ được chứa overlap của cùng một cặp speaker.
        for pair in group:
            pair_set = frozenset((pair["seg1"]["speaker"], pair["seg2"]["speaker"]))
            if pair_set != target_set:
                self.detail = "mixed_speaker_pairs"
                return None

        raw_overlap_lo = min(float(p["overlap_start"]) for p in group)
        raw_overlap_hi = max(float(p["overlap_end"]) for p in group)
        n_samples = len(self.waveform)
        core_lo = max(0, int(np.floor(raw_overlap_lo * self.sr)))
        core_hi = min(n_samples, int(np.ceil(raw_overlap_hi * self.sr)))

        if core_hi <= core_lo:
            self.detail = "overlap_shorter_than_one_sample"
            return None
        if core_hi - core_lo > self.target:
            self.detail = "overlap_core_exceeds_15s"
            return None

        seg_starts = [float(p[side]["start"]) for p in group for side in ("seg1", "seg2")]
        seg_ends = [float(p[side]["end"]) for p in group for side in ("seg1", "seg2")]
        seg_indices = sorted({str(p[side]["index"]) for p in group for side in ("seg1", "seg2")})

        timestamp_lo_raw = max(0, int(np.floor(min(seg_starts) * self.sr)))
        timestamp_hi_raw = min(n_samples, int(np.ceil(max(seg_ends) * self.sr)))

        # ------------------------------------------------------------------
        # 2. Speaker thứ ba tạo floor / ceiling. Không dùng seam làm hard bound.
        # ------------------------------------------------------------------
        floor, ceiling = 0, n_samples
        third_speaker_bounds = []
        for speaker, ranges in self.by_speaker.items():
            if speaker in target_set:
                continue
            for a, b in ranges:
                # Không thể dựng cửa sổ hai người sạch nếu người thứ ba nằm ngay core.
                if a < core_hi and b > core_lo:
                    self.detail = f"third_speaker_in_core:{speaker}"
                    return None
                if b <= core_lo:
                    if b > floor:
                        floor = b
                        third_speaker_bounds.append(("left", str(speaker), a, b))
                elif a >= core_hi:
                    if a < ceiling:
                        ceiling = a
                        third_speaker_bounds.append(("right", str(speaker), a, b))

        timestamp_lo = max(floor, timestamp_lo_raw)
        timestamp_hi = min(ceiling, timestamp_hi_raw)

        if timestamp_lo > core_lo or timestamp_hi < core_hi:
            self.detail = "third_speaker_clips_core_context"
            return None

        # Full-overlap: không có solo timestamp context ở hai bên. Theo policy,
        # giữ nguyên toàn bộ overlap/timestamp và không ép phải có pad.
        edge_eps = round(0.05 * self.sr)
        full_overlap = (
            core_lo - timestamp_lo <= edge_eps
            and timestamp_hi - core_hi <= edge_eps
        )

        # ------------------------------------------------------------------
        # 3. Chọn phần timestamp context của base.
        #    Base thường ~7s để còn budget cho pad, nhưng overlap luôn thắng.
        # ------------------------------------------------------------------
        core_len = core_hi - core_lo
        left_available = max(0, core_lo - timestamp_lo)
        right_available = max(0, timestamp_hi - core_hi)
        timestamp_len = timestamp_hi - timestamp_lo

        if timestamp_len <= 0:
            self.detail = "empty_timestamp_union"
            return None

        if full_overlap:
            desired_lo, desired_hi = timestamp_lo, timestamp_hi
        else:
            min_left = min(left_available, self.base_core_min)
            min_right = min(right_available, self.base_core_min)

            # Giữ ít nhất context gần core nếu có; base_target=7s là soft target.
            desired_len = max(
                core_len + min_left + min_right,
                min(self.base_target, timestamp_len),
            )
            desired_len = min(timestamp_len, self.target, desired_len)

            context_budget = max(0, desired_len - core_len)

            # Chia context hai bên gần cân đối, rồi dồn phần thừa sang bên còn chỗ.
            left_take = min(left_available, context_budget // 2)
            right_take = min(right_available, context_budget - left_take)
            remaining = context_budget - left_take - right_take
            if remaining > 0:
                add_left = min(left_available - left_take, remaining)
                left_take += add_left
                remaining -= add_left
            if remaining > 0:
                add_right = min(right_available - right_take, remaining)
                right_take += add_right
                remaining -= add_right

            # Nếu chia đôi làm một bên hụt minimum context, chuyển budget từ bên kia.
            if left_take < min_left:
                need = min_left - left_take
                give = min(need, max(0, right_take - min_right))
                left_take += give
                right_take -= give
            if right_take < min_right:
                need = min_right - right_take
                give = min(need, max(0, left_take - min_left))
                right_take += give
                left_take -= give

            desired_lo = core_lo - left_take
            desired_hi = core_hi + right_take

            # Nếu timestamp thật quá ngắn (<3s), mới cho phép mở rộng ra ngoài
            # timestamp. Đây là ưu tiên thấp hơn pad/context và chỉ bị chặn bởi
            # speaker thứ ba hoặc biên file.
            current_len = desired_hi - desired_lo
            if current_len < self.base_min_total:
                need = min(self.base_min_total - current_len,
                           self.target - current_len)
                left_room = max(0, desired_lo - floor)
                right_room = max(0, ceiling - desired_hi)

                add_left = min(left_room, need // 2)
                add_right = min(right_room, need - add_left)
                remaining = need - add_left - add_right
                if remaining > 0:
                    extra = min(left_room - add_left, remaining)
                    add_left += extra
                    remaining -= extra
                if remaining > 0:
                    extra = min(right_room - add_right, remaining)
                    add_right += extra
                    remaining -= extra

                desired_lo -= add_left
                desired_hi += add_right

        if desired_hi <= desired_lo or not (desired_lo <= core_lo < core_hi <= desired_hi):
            self.detail = "invalid_base_before_cut"
            return None

        # ------------------------------------------------------------------
        # 4. Nới/cắt nhẹ quanh hai mép để tránh cắt giữa từ. Không bị seam chặn.
        # ------------------------------------------------------------------
        if full_overlap:
            # Full-overlap là ngoại lệ: giữ đúng union timestamp, không nới mép
            # và cũng không cần pad chỉ để đạt anchor 5-8s.
            best_a, best_b = desired_lo, desired_hi
            start_cut = end_cut = "timestamp_bound"
        else:
            cut_radius = round(0.5 * self.sr)
            left_search_lo = max(floor, desired_lo - cut_radius)
            left_search_hi = min(core_lo, desired_lo + cut_radius)
            right_search_lo = max(core_hi, desired_hi - cut_radius)
            right_search_hi = min(ceiling, desired_hi + cut_radius)

            best_a, start_cut = self._safe_cut_near(
                desired_lo, left_search_lo, left_search_hi,
                "left", core_lo, core_hi,
            )
            best_b, end_cut = self._safe_cut_near(
                desired_hi, right_search_lo, right_search_hi,
                "right", core_lo, core_hi,
            )

            # Acoustic cut chỉ là refinement. Nếu nó làm vỡ invariant hoặc vượt 15s,
            # quay lại timestamp boundary đã tính ở trên.
            if not (floor <= best_a <= core_lo < core_hi <= best_b <= ceiling):
                best_a, best_b = desired_lo, desired_hi
                start_cut = end_cut = "timestamp_bound"
            if best_b - best_a > self.target:
                best_a, best_b = desired_lo, desired_hi
                start_cut = end_cut = "timestamp_bound"

        if best_b <= best_a or best_b - best_a > self.target:
            self.detail = "base_exceeds_budget"
            return None

        base = Piece(
            start=best_a,
            end=best_b,
            speaker="+".join(map(str, speakers)),
            source="+".join(seg_indices),
            kind="base",
            start_cut=start_cut,
            end_cut=end_cut,
        )

        # Solo/voiced trong base dùng để biết speaker nào đang thiếu clean evidence.
        voice_by_speaker = {
            s: intersect(self.by_speaker.get(s, []), best_a, best_b)
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
        solos = {s: self._voiced(self._solo(s, best_a, best_b)) for s in speakers}
        base_voice = {
            s: sum(y - x for x, y in intersect(voiced_by_speaker[s], best_a, best_b))
            for s in speakers
        }

        # ------------------------------------------------------------------
        # 5. Full-overlap là ngoại lệ: overlap thắng, không bắt buộc pad.
        # ------------------------------------------------------------------
        if full_overlap:
            result = self._assemble((), base, (), speakers, core_lo, core_hi, solos)
            total_voice = dict(base_voice)
            prefix, suffix = (), ()
            prefix_speaker = suffix_speaker = None
        else:
            # --------------------------------------------------------------
            # 6. Phân budget pad. Một speaker ở prefix, speaker kia ở suffix.
            #    Thử cả hai orientation và chọn layout:
            #      - core gần 5-8s nhất;
            #      - clean voice A/B gần 1:1 nhất;
            #      - tận dụng pad khi còn budget.
            # --------------------------------------------------------------
            centre = (core_lo + core_hi) // 2
            base_core_pos = core_lo - base.start
            base_len = base.end - base.start
            total_pad_budget = max(0, self.target - base_len)

            candidates = []
            for prefix_speaker, suffix_speaker in (
                (speakers[0], speakers[1]),
                (speakers[1], speakers[0]),
            ):
                # _sequences tính "added" đúng theo crossfade của prefix.
                strict_min = max(0, self.final_core_min - base_core_pos)
                strict_max = max(0, self.final_core_max - base_core_pos)
                strict_max = min(strict_max, total_pad_budget)

                prefix_options = []
                if strict_min <= strict_max:
                    prefix_options = self._sequences(
                        prefix_speaker, centre, strict_min, strict_max, base
                    )

                # Không tìm được pad đặt core đúng 5-8s thì mới nới constraint.
                if not prefix_options:
                    prefix_options = self._sequences(
                        prefix_speaker, centre, 0, total_pad_budget, base
                    )
                if not prefix_options:
                    prefix_options = [()]

                for prefix_seq in prefix_options:
                    prefix_added = self._sequence_added_samples(prefix_seq)
                    remaining = max(0, total_pad_budget - prefix_added)
                    suffix_options = self._sequences(
                        suffix_speaker, centre, 0, remaining, base
                    ) or [()]

                    for suffix_seq in suffix_options:
                        predicted = self._window_samples(prefix_seq, base, suffix_seq)
                        if predicted > self.target:
                            continue

                        final_core_start = prefix_added + base_core_pos
                        if self.final_core_min <= final_core_start <= self.final_core_max:
                            anchor_penalty = 0
                        else:
                            anchor_penalty = min(
                                abs(final_core_start - self.final_core_min),
                                abs(final_core_start - self.final_core_max),
                            )

                        total_voice = dict(base_voice)
                        total_voice[prefix_speaker] += self._sequence_voice_samples(prefix_seq)
                        total_voice[suffix_speaker] += self._sequence_voice_samples(suffix_seq)

                        va = total_voice[speakers[0]]
                        vb = total_voice[speakers[1]]
                        # Log-ratio đối xứng: 2:1 và 1:2 bị phạt như nhau.
                        balance_penalty = abs(np.log((va + 1.0) / (vb + 1.0)))
                        missing_pad = int(not prefix_seq) + int(not suffix_seq)

                        # Anchor là hard preference; nếu có thể thì giữ đúng
                        # một pad cho mỗi speaker, sau đó mới tối ưu gần 1:1.
                        rank = (
                            anchor_penalty,
                            missing_pad,
                            balance_penalty,
                            -predicted,
                        )
                        candidates.append((
                            rank,
                            tuple(prefix_seq),
                            tuple(suffix_seq),
                            prefix_speaker,
                            suffix_speaker,
                            total_voice,
                        ))

            if candidates:
                candidates.sort(key=lambda item: item[0])
                _, prefix, suffix, prefix_speaker, suffix_speaker, total_voice = candidates[0]
            else:
                prefix = suffix = ()
                prefix_speaker = suffix_speaker = None
                total_voice = dict(base_voice)

            result = self._assemble(prefix, base, suffix, speakers, core_lo, core_hi, solos)

        # ------------------------------------------------------------------
        # 7. Metadata để đọc được vì sao planner chọn layout này.
        # ------------------------------------------------------------------
        base_width_s = (base.end - base.start) / self.sr
        final_voice_seconds = {s: total_voice[s] / self.sr for s in speakers}
        min_voice = min(total_voice.values()) if total_voice else 0
        max_voice = max(total_voice.values()) if total_voice else 0
        ratio = (max_voice / max(1, min_voice)) if max_voice else 1.0

        prefix_added = self._sequence_added_samples(prefix)
        suffix_added = self._sequence_added_samples(suffix)
        left_context = max(0, core_lo - base.start)
        right_context = max(0, base.end - core_hi)
        context_floor = min(self.base_core_min, max(left_available, right_available))

        result.layout.update({
            "policy": "overlap>timestamp>balanced-pad>extension",
            "policy_version": POLICY_VERSION,
            "involved_segments": seg_indices,
            "host_segment": "+".join(seg_indices),
            "host_source_samples": [base.start, base.end],
            "timestamp_source_samples": [timestamp_lo, timestamp_hi],
            "third_speaker_floor_ceiling": [floor, ceiling],
            "third_speaker_bounds_seen": third_speaker_bounds,
            "cut_method": [base.start_cut, base.end_cut],
            "estimated_voice_seconds": {str(s): final_voice_seconds[s] for s in speakers},
            "base_voice_seconds": {str(s): base_voice[s] / self.sr for s in speakers},
            "ratio": ratio,
            "core_position_seconds": result.core[0] / self.sr,
            "base_width_seconds": base_width_s,
            "base_core_position_seconds": (core_lo - base.start) / self.sr,
            "context_seconds": [left_context / self.sr, right_context / self.sr],
            "context_shortfall_seconds": [
                max(0.0, (self.base_core_min - left_context) / self.sr),
                max(0.0, (self.base_core_min - right_context) / self.sr),
            ],
            "core_expanded": base.start < timestamp_lo or base.end > timestamp_hi,
            "full_overlap": full_overlap,
            "prefix_speaker": None if prefix_speaker is None else str(prefix_speaker),
            "suffix_speaker": None if suffix_speaker is None else str(suffix_speaker),
            "pad_seconds": {
                "prefix": prefix_added / self.sr,
                "suffix": suffix_added / self.sr,
            },
            "overlaps": [[p["overlap_start"], p["overlap_end"]] for p in group],
        })

        self.reason, self.detail = "ok", "balanced_window"
        return result
