"""Dựng cửa sổ hai người nói, giữ nguồn và ánh xạ theo mẫu âm thanh.

Mô-đun chỉ cần NumPy. Silero được truyền vào từ model đã tải; khi không có
Silero hoặc suy luận lỗi, dùng khoảng năng lượng thấp và khe giữa từ/âm tiết
để tìm điểm cắt an toàn.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

POLICY_VERSION = "connected-balanced-15s-v5"


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

    def build(self, group):
        self.reason, self.detail = "no_window", "no_safe_layout"
        first = min(group, key=lambda p: (p["overlap_start"], p["overlap_end"]))
        speakers = tuple(sorted((first["seg1"]["speaker"], first["seg2"]["speaker"])))
        core_lo = int(min(p["overlap_start"] for p in group) * self.sr)
        core_hi = int(max(p["overlap_end"] for p in group) * self.sr)

        if core_hi <= core_lo:
            self.detail = "overlap_shorter_than_one_sample"
            return None
        if core_hi - core_lo > self.target - self.final_core_min:
            self.detail = "overlap_does_not_fit_15s"
            return None

        sources = {str(p[side]["index"]) for p in group for side in ("seg1", "seg2")}
        hosts = [s for s in self.segments if str(s.index) in sources]
        host = max(hosts, key=lambda s: (s.end - s.start, -s.start))
        host_lo, host_hi = int(host.start * self.sr), int(host.end * self.sr)

        if any(a < core_hi and b > core_lo for s, rs in self.by_speaker.items() if s not in speakers for a, b in rs):
            self.reason, self.detail = "multi_speaker", "third_speaker_in_connected_overlap"
            return None

        extra = [p for p in group if (p["overlap_start"], p["overlap_end"]) != (first["overlap_start"], first["overlap_end"])]
        margin = self.sr
        if any(int(p["overlap_start"] * self.sr) - host_lo < margin or host_hi - int(p["overlap_end"] * self.sr) < margin for p in extra):
            self.detail = "secondary_overlap_near_original_edge"
            return None

        floor, ceiling = 0, len(self.waveform)
        # floor and ceiling mark how far the base may extend left and right.
        #
        # What the base may cross:
        #   - segment boundaries of the two speakers in the window
        #   - recording seams and music beds
        # What it must not cross:
        #   - a segment of a third speaker (would corrupt the extraction)
        #   - an overlap of the two window speakers from a different pair
        #     (the separator would see two voices competing at that point)
        #
        # The old code also blocked on every segment of the non-host speaker
        # outside host_lo..host_hi, and on seams. Both restrictions are lifted.

        # Third-speaker segments: must not enter base.
        for a, b in (r for s, rs in self.by_speaker.items()
                     if s not in speakers for r in rs):
            if a < core_hi and b > core_lo:
                self.detail = "foreign_overlap_in_target"
                return None
            if b <= core_lo:
                floor = max(floor, b)
            if a >= core_hi:
                ceiling = min(ceiling, a)

        # Overlaps of the two window speakers from OTHER pairs: also block.
        # A segment of one speaker alone is fine; a cross-speaker overlap
        # means the separator would face two voices at the cut point.
        group_ids = {id(p) for p in group}
        for p in self.pairs:
            if id(p) in group_ids:
                continue
            s1, s2 = p["seg1"]["speaker"], p["seg2"]["speaker"]
            if s1 not in speakers or s2 not in speakers:
                continue
            a, b = int(p["overlap_start"] * self.sr), int(p["overlap_end"] * self.sr)
            if a < core_hi and b > core_lo:
                self.detail = "foreign_overlap_in_target"
                return None
            if b <= core_lo:
                floor = max(floor, b)
            if a >= core_hi:
                ceiling = min(ceiling, a)

        # --- core expansion for short overlaps ----------------------------
        # When the overlap itself is very short (< 1s), Sidon barely sees
        # the competing voice. Expand the effective core region by ±2s so
        # the base always carries real context around the overlap, stopping
        # at floor/ceiling so no third speaker leaks in.
        #
        # The pad must leave base_core_min of room between the expanded core
        # and floor/ceiling. Clamping straight to floor/ceiling let the pad
        # eat the very context the base needs: with a third speaker ending
        # 0.5s before the overlap, eff_core_lo landed on floor, no left cut
        # could sit base_core_min earlier, and the window failed outright --
        # while the same geometry with a longer overlap, which is not padded
        # at all, built fine. Where that room does not exist the core simply
        # is not expanded on that side.
        core_span = core_hi - core_lo
        if core_span < self.short_core_threshold:
            pad = self.short_core_pad
            eff_core_lo = min(core_lo, max(core_lo - pad, floor + self.base_core_min))
            eff_core_hi = max(core_hi, min(core_hi + pad, ceiling - self.base_core_min))
        else:
            eff_core_lo = core_lo
            eff_core_hi = core_hi

        # --- base search range -----------------------------------------
        # Base is bounded to window seconds 3-10 (base_target = 7s).
        # The left cut must be at least base_core_min before eff_core_lo
        # (normal case: 1.5s). When the overlap sits near the host edge
        # there may not be 1.5s available on one side; in that case we
        # allow the cut to go as close as the fade (0.02s), provided the
        # total base still reaches base_min_total (3s). This is the "edge
        # expansion" case: the base grows toward the other side to make up
        # the deficit, still capped at base_core_max.
        room_left  = eff_core_lo - floor
        room_right = ceiling - eff_core_hi

        # How much left context does base need at minimum?
        # Normal: base_core_min. Edge: whatever fits, ≥ fade.
        if room_left <= self.base_core_min:
            # Near left edge -- relax the left floor to fade, require
            # right side to compensate so total >= base_min_total.
            # The comparison has to include equality. At exactly base_core_min
            # the only admissible left cut is floor itself, and that one is
            # subject to the quiet_edge() veto, so the usual outcome of the
            # strict form was no_safe_left_cut on a window that had room.
            left_min = self.fade
        else:
            left_min = self.base_core_min

        lo = max(floor, eff_core_lo - self.base_core_max)
        hi = min(ceiling, eff_core_hi + self.base_core_max)
        cuts, voiced, method = self.cuts.analyse(lo, hi)
        cuts = dict(cuts)

        for edge in (lo, hi):
            if edge not in (host_lo, host_hi, 0, len(self.waveform)):
                if self.cuts.quiet_edge(edge):
                    cuts[edge] = "energy"
                else:
                    cuts.pop(edge, None)

        for edge in (host_lo, host_hi):
            if lo <= edge <= hi:
                cuts[edge] = "segment"

        lefts = [a for a in cuts
                 if eff_core_lo - self.base_core_max <= a <= eff_core_lo - left_min
                 and eff_core_lo - a >= self.fade]
        if not lefts:
            self.detail = "no_safe_left_cut_for_3_10s_base"
            return None

        right_pool = [b for b in cuts if b >= eff_core_hi + self.fade]
        if not right_pool:
            self.detail = "no_safe_right_cut"
            return None

        voice_by_speaker = {s: intersect(self.by_speaker[s], lo, hi) for s in speakers}
        voiced_by_speaker = {s: [r for a, b in voice_by_speaker[s] for r in intersect(voiced, a, b)] for s in speakers}
        solos = {s: self._voiced(self._solo(s, lo, hi)) for s in speakers}
        support_voice = {}

        def voice_count(piece):
            key = (piece.start, piece.end)
            if key not in support_voice:
                support_voice[key] = sum(b - a for a, b in self.cuts.analyse(*key)[1])
            return support_voice[key]

        best = None

        for a in sorted(lefts):
            raw_core_pos = eff_core_lo - a
            if raw_core_pos > self.final_core_max:
                continue

            left_spk = host.speaker
            at_left = [s for s in speakers if any(x <= a < y for x, y in self.by_speaker[s])]
            if len(at_left) == 1:
                left_spk = at_left[0]

            prefix_min = max(0, self.final_core_min - raw_core_pos)
            prefix_max = self.final_core_max - raw_core_pos
            if prefix_max < 0:
                continue

            temp_end = eff_core_hi + self.fade
            if temp_end > ceiling:
                continue
            temp_base = Piece(a, temp_end, host.speaker, str(host.index), "base", cuts[a], "temporary")
            prefixes = self._sequences(left_spk, eff_core_lo, prefix_min, prefix_max, temp_base)

            for prefix in prefixes:
                pre = sum(p.end - p.start - self.fade for p in prefix)
                core_pos = pre + raw_core_pos
                if not (self.final_core_min <= core_pos <= self.final_core_max):
                    continue

                # base is capped at base_core_max (10s) from its left cut;
                # beyond that any extra host goes into the pad piece instead.
                max_base_end = min(ceiling,
                                   a + self.base_core_max,
                                   a + self.target - pre)
                # base must reach at least eff_core_hi and total >= base_min_total
                min_base_end = max(eff_core_hi + self.fade,
                                   a + self.base_min_total)
                if max_base_end < min_base_end:
                    continue

                valid_rights = [b for b in right_pool if min_base_end <= b <= max_base_end]
                if not valid_rights:
                    continue

                for b in sorted(valid_rights, reverse=True):
                    if pre + (b - a) > self.target:
                        continue
                    if any(int(p["overlap_start"] * self.sr) - a < margin or b - int(p["overlap_end"] * self.sr) < margin for p in extra):
                        continue

                    base = Piece(a, b, host.speaker, str(host.index), "base", cuts[a], cuts[b])
                    if any(min(p.end, b) > max(p.start, a) for p in prefix):
                        continue

                    base_voice = {s: sum(y - x for x, y in intersect(voiced_by_speaker[s], a, b)) for s in speakers}
                    base_solo = {s: sum(y - x for x, y in intersect(solos[s], a + self.fade, b - self.fade)) for s in speakers}
                    counts = dict(base_voice)

                    for p in prefix:
                        counts[p.speaker] += max(0, voice_count(p) - 2 * self.fade)

                    used = pre + (b - a)
                    available = self.target - used
                    if available < 0:
                        continue

                    missing = min(speakers, key=lambda s: counts[s])
                    suffixes = self._sequences(missing, eff_core_hi, 0, available, base)

                    for suffix in suffixes:
                        support = prefix + suffix
                        if len({p.source for p in support}) != len(support):
                            continue
                        if any(min(p.end, q.end) > max(p.start, q.start) for p, q in combinations(support, 2)):
                            continue
                        if any(min(p.end, b) > max(p.start, a) for p in support):
                            continue

                        total_voice, total_solo = dict(base_voice), dict(base_solo)
                        for p in support:
                            amount = max(0, voice_count(p) - 2 * self.fade)
                            total_voice[p.speaker] += amount
                            total_solo[p.speaker] += amount

                        if min(total_solo.values()) < self.sr:
                            continue

                        suffix_len = sum(p.end - p.start - self.fade for p in suffix)
                        duration = b - a + pre + suffix_len
                        if duration > self.target:
                            continue

                        ratio_error = abs(total_voice[speakers[0]] - total_voice[speakers[1]]) / max(1, sum(total_voice.values()))
                        retained = max(0, min(b, host_hi) - max(a, host_lo))
                        budget = min(host_hi - host_lo, self.base_target)
                        loss = 1 - min(retained, budget) / max(1, budget)
                        # With no prefix the core's position in the window just
                        # is its left context, and every extra second there is
                        # a second the right cannot have. Aiming at 6.5 -- the
                        # middle of the legal 5-8s band -- therefore starved the
                        # right by construction. 5.0 is the bottom of the same
                        # band: still legal, and it leaves the budget for the
                        # side that carries the evidence.
                        anchor_error = max(0, core_pos / self.sr - 5.0) / 3.0
                        # self.context is the target on BOTH sides and was only
                        # ever reported, never scored -- so the right side cost
                        # nothing. The host resuming after the backchannel is
                        # what shows the blip belonged to the other speaker, and
                        # that evidence is entirely on the right.
                        context_error = (max(0, self.context - (eff_core_lo - a))
                                         + max(0, self.context - (b - eff_core_hi))) / (2 * self.context)
                        score = (0.28 * loss + 0.28 * ratio_error
                                 + 0.08 * (self.target - duration) / self.target
                                 + 0.10 * anchor_error + 0.25 * context_error
                                 + 0.010 * len(support))
                        rank = (score, len(support), -retained)

                        if best is None or rank < best[0]:
                            best = (rank, prefix, base, suffix, total_voice, method, core_pos)

        if best is None:
            # --- pad-less fallback: stretch base toward both edges ----------
            # No support material was found for the non-host speaker. Rather
            # than failing, widen the base on both sides (2-10s each) so
            # Sidon at least sees the transition region. The base may cross
            # the other speaker's segments freely; it only stops at a third-
            # speaker boundary (floor / ceiling).
            # Both sides are walked nearest-cut-first, so the first pair that
            # qualifies is the tightest one -- roughly self.context on each
            # side, which is what the scored path aims for anyway.
            BASE_FALLBACK_MIN = round(2.0 * self.sr)
            BASE_FALLBACK_MAX = round(10.0 * self.sr)

            fb_lefts = sorted(
                [a for a in cuts
                 if eff_core_lo - BASE_FALLBACK_MAX <= a <= eff_core_lo - BASE_FALLBACK_MIN
                 and eff_core_lo - a >= self.fade],
                reverse=True  # largest a first → nearest the core → tightest base
            )
            fb_rights = sorted(
                [b for b in cuts
                 if eff_core_hi + BASE_FALLBACK_MIN <= b <= eff_core_hi + BASE_FALLBACK_MAX
                 and b - eff_core_hi >= self.fade],
            )  # nearest right first → keep base tight

            fb_best, fb_over_budget = None, False
            for fa in fb_lefts:
                for fb in fb_rights:
                    if fa >= fb:
                        continue
                    # The fallback is a relaxation of the layout rules, not of
                    # the window budget. Where clean cuts only exist far from
                    # the core this used to hand the separator a 27s window
                    # against a 15s design point; refusing the job is the
                    # honest outcome, and it is recorded as a failed span
                    # rather than passed off as clean mixture.
                    if fb - fa > self.target:
                        fb_over_budget = True
                        continue
                    # In fallback mode we accept any pair that gives each
                    # speaker at least one voiced frame -- the separator will
                    # do its best with whatever is here; failing is worse.
                    fb_voice = {s: sum(y - x for x, y in intersect(voiced_by_speaker[s], fa, fb))
                                for s in speakers}
                    if min(fb_voice.values()) < 1:
                        continue
                    fb_best = (fa, fb, fb_voice)
                    break
                if fb_best:
                    break

            if fb_best is None:
                # Separating the two says whether the recording had no usable
                # support at all, or only cuts too far out to stay inside the
                # budget -- different problems, different fixes.
                self.detail = ("fallback_base_exceeds_15s" if fb_over_budget
                               else "no_layout_with_clean_support_and_5_8s_anchor")
                return None

            fa, fb, fb_voice = fb_best
            fb_base = Piece(fa, fb, host.speaker, str(host.index), "base",
                            cuts.get(fa, "silero"), cuts.get(fb, "silero"))
            fb_core_pos = eff_core_lo - fa
            fb_result = self._assemble((), fb_base, (), speakers,
                                       eff_core_lo, eff_core_hi, solos)
            fb_result.layout.update({
                "policy": POLICY_VERSION,
                "host_segment": str(host.index),
                "host_source_samples": [host_lo, host_hi],
                "cut_method": method,
                "estimated_voice_seconds": {s: v / self.sr for s, v in fb_voice.items()},
                "ratio": max(fb_voice.values()) / max(1, min(fb_voice.values())),
                "core_position_seconds": fb_core_pos / self.sr,
                "base_core_position_seconds": (core_lo - fa) / self.sr,
                "context_seconds": [(eff_core_lo - fa) / self.sr,
                                    (fb - eff_core_hi) / self.sr],
                "context_shortfall_seconds": [
                    max(0, self.context - (eff_core_lo - fa)) / self.sr,
                    max(0, self.context - (fb - eff_core_hi)) / self.sr,
                ],
                "core_expanded": core_span < self.short_core_threshold,
                "pad_fallback": True,
                "overlaps": [[p["overlap_start"], p["overlap_end"]] for p in group],
            })
            return fb_result

        _, prefix, base, suffix, voice, method, core_pos = best
        result = self._assemble(prefix, base, suffix, speakers, eff_core_lo, eff_core_hi, solos)
        result.layout.update({
            "policy": POLICY_VERSION,
            "host_segment": str(host.index),
            "host_source_samples": [host_lo, host_hi],
            "cut_method": method,
            "estimated_voice_seconds": {s: v / self.sr for s, v in voice.items()},
            "ratio": max(voice.values()) / max(1, min(voice.values())),
            "core_position_seconds": core_pos / self.sr,
            "base_core_position_seconds": (core_lo - base.start) / self.sr,
            "context_seconds": [(eff_core_lo - base.start) / self.sr, (base.end - eff_core_hi) / self.sr],
            "context_shortfall_seconds": [max(0, self.context - (eff_core_lo - base.start)) / self.sr, max(0, self.context - (base.end - eff_core_hi)) / self.sr],
            "core_expanded": core_span < self.short_core_threshold,
            "overlaps": [[p["overlap_start"], p["overlap_end"]] for p in group],
        })
        return result

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
