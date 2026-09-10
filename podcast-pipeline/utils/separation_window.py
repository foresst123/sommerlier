"""Dựng cửa sổ hai người nói, giữ nguồn và ánh xạ theo mẫu âm thanh.

Mô-đun chỉ cần NumPy. Silero được truyền vào từ model đã tải; khi không có
Silero hoặc suy luận lỗi, dùng khoảng năng lượng thấp để tìm chỗ ngắt.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

POLICY_VERSION = "connected-balanced-15s-v1"


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
    """Lưu VAD theo vùng; chỉ chấp nhận điểm cắt nội bộ ở khoảng nghỉ.

    Không coi điểm qua zero đơn lẻ là hết chữ. Nhánh năng lượng yêu cầu một
    khoảng thấp liên tục ít nhất 40 ms. Biên segment vẫn là ứng viên có nguồn
    từ diarization; thông tin này được lưu riêng với điểm ngắt đo từ âm thanh.
    """

    def __init__(self, waveform, sr, vad=None):
        self.waveform, self.sr, self.vad = waveform, sr, vad
        self.cache = {}

    def analyse(self, lo, hi):
        key = (int(lo), int(hi))
        if key in self.cache:
            return self.cache[key]
        lo, hi = key
        wave = self.waveform[lo:hi]
        voiced, method = None, "energy"
        if self.vad is not None:
            try:
                timestamps = self.vad.get_speech_timestamps(
                    wave, sampling_rate=self.sr)
                voiced = merge_ranges((lo + max(0, int(t["start"])),
                                       min(hi, lo + int(t["end"])))
                                      for t in timestamps)
                method = "silero"
            except Exception:
                # Lỗi VAD không cho phép cắt tùy ý; chuyển sang khoảng nghỉ
                # đo bằng năng lượng và ghi rõ phương pháp trong layout.
                voiced = None
        if voiced is None:
            step = max(1, round(self.sr * 0.01))
            starts = np.arange(0, len(wave), step)
            if len(starts):
                squared = np.square(wave.astype(np.float64))
                counts = np.minimum(step, len(wave) - starts)
                rms = np.sqrt(np.add.reduceat(squared, starts) / counts)
                threshold = max(1e-5, float(np.percentile(rms, 90)) * 0.10)
                voiced = merge_ranges((lo + int(i), min(hi, lo + int(i) + step))
                                      for i, value in zip(starts, rms)
                                      if value > threshold)
            else:
                voiced = []
        pauses = subtract([(lo, hi)], voiced)
        cuts = {lo: "segment", hi: "segment"}
        for a, b in pauses:
            if b - a >= round(self.sr * 0.04):
                cuts[(a + b) // 2] = method
        self.cache[key] = (cuts, voiced, method)
        return self.cache[key]

    def quiet_edge(self, point):
        """Biên tính toán chỉ được cắt khi có khoảng năng lượng thấp quanh nó."""
        radius = max(1, round(0.02*self.sr))
        lo, hi = max(0, point-radius), min(len(self.waveform), point+radius)
        nearby = self.waveform[max(0,point-self.sr//2):min(len(self.waveform),point+self.sr//2)]
        if hi <= lo or not len(nearby):
            return False
        quiet = float(np.sqrt(np.mean(self.waveform[lo:hi].astype(np.float64)**2)))
        reference = float(np.sqrt(np.mean(nearby.astype(np.float64)**2)))
        return quiet <= max(1e-5, reference*0.1)


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
    """Tìm tổ hợp nền/support; không thêm im lặng hay quay về cửa sổ cũ."""

    def __init__(self, segments, pairs, waveform, sr, music_map=None,
                 seams=(), vad=None, context_seconds=2.0, search_seconds=400.0):
        self.segments, self.pairs = segments, pairs
        self.waveform, self.sr = waveform, sr
        self.music_map = music_map
        self.seams = [int(t * sr) for t in seams]
        self.context = round(context_seconds * sr)
        self.search = search_seconds * sr
        self.target = round(15.0 * sr)
        self.fade = round(0.020 * sr)
        self.minimum = round(1.5 * sr)
        self.cuts = AcousticCuts(waveform, sr, vad)
        self.by_speaker = {}
        for s in segments:
            self.by_speaker.setdefault(s.speaker, []).append(
                (int(s.start * sr), int(s.end * sr)))
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
        candidates = [s for s in self.clean if s.speaker == speaker
                      and abs((s.start + s.end) * self.sr / 2 - centre) <= self.search]
        candidates.sort(key=lambda s: abs((s.start + s.end) * self.sr / 2 - centre))
        pieces = []
        for seg in candidates[:12]:
            spans = [(seg.start, seg.end)]
            if self.music_map:
                spans = self.music_map.clean_parts(seg.start, seg.end)
            for start, end in spans:
                lo, hi = max(0, int(start * self.sr)), min(len(self.waveform), int(end * self.sr))
                # Một support không được vượt mối nối vốn có của bản ghi.
                limits = [lo] + [s for s in self.seams if lo < s < hi] + [hi]
                for a, b in zip(limits, limits[1:]):
                    cuts, voiced, _ = self.cuts.analyse(a, b)
                    cuts = dict(cuts)
                    # Biên do cắt nhạc/mối nối tạo ra không phải biên câu gốc.
                    for edge, original in ((a, int(seg.start*self.sr)), (b, int(seg.end*self.sr))):
                        if edge != original:
                            if self.cuts.quiet_edge(edge):
                                cuts[edge] = "energy"
                            else:
                                cuts.pop(edge, None)
                    points = sorted(cuts)
                    # Giới hạn số ứng viên theo thời gian, vẫn giữ hai biên.
                    if len(points) > 48:
                        points = [points[i] for i in np.linspace(0, len(points)-1, 48, dtype=int)]
                    for x, y in combinations(points, 2):
                        if self.minimum <= y - x <= round(8.04 * self.sr):
                            voice = sum(v-u for u, v in intersect(voiced, x, y))
                            if voice >= self.sr:
                                pieces.append(Piece(x, y, speaker, str(seg.index),
                                                    "support", cuts[x], cuts[y]))
        # Giữ ứng viên ở nhiều mức thời lượng, tránh toàn bộ danh sách chỉ
        # gồm các đoạn ngắn nhất hoặc cùng một segment gần nhất.
        buckets = {}
        for p in pieces:
            bucket = round((p.end-p.start) / self.sr * 10)
            buckets.setdefault(bucket, []).append(p)
        result = []
        for bucket in sorted(buckets):
            result.extend(sorted(buckets[bucket], key=lambda p: abs((p.start+p.end)/2-centre))[:2])
        self._support_cache[key] = result
        return result

    def _sequences(self, speaker, centre, minimum, maximum, base):
        """Tối đa hai support mỗi phía; mỗi đoạn độc lập phải đạt 1.5 s."""
        options = [()] if minimum <= 0 else []
        available = [p for p in self._support(speaker, centre)
                     if p.end <= base.start or p.start >= base.end]
        # Mỗi đoạn nối vào nền làm tăng thời lượng bằng độ dài trừ crossfade.
        for p in available:
            added = p.end-p.start-self.fade
            if minimum <= added <= maximum:
                options.append((p,))
        # Dùng hai segment nguồn khác nhau khi không có một đoạn phù hợp.
        short = [p for p in available if p.end-p.start-self.fade <= maximum]
        for i, p in enumerate(short):
            for q in short[i+1:]:
                if p.source == q.source:
                    continue
                added = p.end-p.start+q.end-q.start-2*self.fade
                if minimum <= added <= maximum:
                    options.append(tuple(sorted((p, q), key=lambda z: z.start)))
        # Giữ đa dạng thời lượng và ưu tiên ít mối nối/nguồn gần.
        buckets = {}
        for seq in options:
            added = sum(p.end-p.start-self.fade for p in seq)
            bucket = round(added / self.sr * 10)
            rank = (len(seq), sum(abs((p.start+p.end)/2-centre) for p in seq))
            if bucket not in buckets or rank < buckets[bucket][0]:
                buckets[bucket] = (rank, seq)
        return [item[1] for item in buckets.values()]

    def build(self, group):
        self.reason, self.detail = "no_window", "no_safe_layout"
        first = min(group, key=lambda p: (p["overlap_start"], p["overlap_end"]))
        speakers = tuple(sorted((first["seg1"]["speaker"], first["seg2"]["speaker"])))
        core_lo = int(min(p["overlap_start"] for p in group) * self.sr)
        core_hi = int(max(p["overlap_end"] for p in group) * self.sr)
        if core_hi <= core_lo:
            self.detail = "overlap_shorter_than_one_sample"
            return None
        if core_hi-core_lo > self.target-round(6*self.sr):
            self.detail = "overlap_does_not_fit_15s"
            return None
        sources = {str(p[side]["index"]) for p in group for side in ("seg1", "seg2")}
        hosts = [s for s in self.segments if str(s.index) in sources]
        host = max(hosts, key=lambda s: (s.end-s.start, -s.start))
        host_lo, host_hi = int(host.start*self.sr), int(host.end*self.sr)
        if any(a < core_hi and b > core_lo for s, rs in self.by_speaker.items()
               if s not in speakers for a, b in rs):
            self.reason, self.detail = "multi_speaker", "third_speaker_in_connected_overlap"
            return None

        # Overlap phụ phải nằm sâu ít nhất 1 s trong segment gốc đã chọn,
        # đồng thời trong phần nền cuối cùng. Không lấy support để giả biên này.
        extra = [p for p in group if
                 (p["overlap_start"], p["overlap_end"]) !=
                 (first["overlap_start"], first["overlap_end"])]
        margin = self.sr
        if any(int(p["overlap_start"]*self.sr)-host_lo < margin or
               host_hi-int(p["overlap_end"]*self.sr) < margin for p in extra):
            self.detail = "secondary_overlap_near_original_edge"
            return None

        floor, ceiling = 0, len(self.waveform)
        blockers = []
        group_ids = {id(p) for p in group}
        for p in self.pairs:
            if id(p) not in group_ids:
                a, b = int(p["overlap_start"]*self.sr), int(p["overlap_end"]*self.sr)
                # Cặp cùng speaker là nhãn trùng, không tạo người thứ ba.
                if p["seg1"]["speaker"] != p["seg2"]["speaker"]:
                    blockers.append((a, b))
        blockers.extend(r for s, rs in self.by_speaker.items() if s not in speakers for r in rs)
        # Ra ngoài segment nền thì dừng khi gặp speaker khác, kể cả đối tác.
        for s, rs in self.by_speaker.items():
            if s != host.speaker:
                for a, b in rs:
                    blockers.extend(intersect([(a, b)], 0, min(host_lo, core_lo)))
                    blockers.extend(intersect([(a, b)], max(host_hi, core_hi), len(self.waveform)))
        for a, b in blockers:
            if a < core_hi and b > core_lo:
                self.detail = "foreign_overlap_in_target"
                return None
            if b <= core_lo:
                floor = max(floor, b)
            if a >= core_hi:
                ceiling = min(ceiling, a)
        for seam in self.seams:
            if core_lo < seam < core_hi:
                self.detail = "target_crosses_recording_seam"
                return None
            if seam <= core_lo:
                floor = max(floor, seam)
            if seam >= core_hi:
                ceiling = min(ceiling, seam)

        # Cho phép tìm điểm ngắt hơi quá 2 s (tối đa 0.5 s) khi mở rộng.
        reach = self.context + round(0.5*self.sr)
        lo = max(floor, min(host_lo, core_lo-reach))
        hi = min(ceiling, max(host_hi, core_hi+reach))
        cuts, voiced, method = self.cuts.analyse(lo, hi)
        cuts = dict(cuts)
        # lo/hi có thể chỉ là giới hạn tìm kiếm, không phải chỗ hết lời.
        for edge in (lo, hi):
            if edge not in (host_lo, host_hi, 0, len(self.waveform)):
                if self.cuts.quiet_edge(edge):
                    cuts[edge] = "energy"
                else:
                    cuts.pop(edge, None)
        for edge in (host_lo, host_hi):
            if lo <= edge <= hi:
                cuts[edge] = "segment"
        left_need = min(self.context, max(0, core_lo-floor))
        right_need = min(self.context, max(0, ceiling-core_hi))
        lefts = [a for a in cuts if core_lo-8*self.sr <= a <= core_lo-left_need]
        rights = [b for b in cuts if core_hi+right_need <= b <= core_lo+9*self.sr]
        # Khi bị chặn sát overlap, cần ít nhất đủ chỗ bảo vệ crossfade.
        lefts = [a for a in lefts if core_lo-a >= self.fade]
        rights = [b for b in rights if b-core_hi >= self.fade]
        if not lefts or not rights:
            self.detail = "no_safe_context_cut"
            return None
        voice_by_speaker = {s: intersect(self.by_speaker[s], lo, hi) for s in speakers}
        voiced_by_speaker = {s: [r for a, b in voice_by_speaker[s]
                                 for r in intersect(voiced, a, b)] for s in speakers}
        solos = {s: self._voiced(self._solo(s, lo, hi)) for s in speakers}
        support_voice = {}

        def voice_count(piece):
            key = (piece.start, piece.end)
            if key not in support_voice:
                support_voice[key] = sum(b-a for a, b in self.cuts.analyse(*key)[1])
            return support_voice[key]

        best = None
        for a in sorted(lefts):
            for b in sorted(rights):
                if b-a > self.target:
                    continue
                if any(int(p["overlap_start"]*self.sr)-a < margin or
                       b-int(p["overlap_end"]*self.sr) < margin for p in extra):
                    continue
                base = Piece(a, b, host.speaker, str(host.index), "base", cuts[a], cuts[b])
                # Speaker sát mép trái được ưu tiên khi có nhãn chắc chắn.
                left_spk = host.speaker
                at_left = [s for s in speakers if any(x <= a < y for x, y in self.by_speaker[s])]
                if len(at_left) == 1:
                    left_spk = at_left[0]
                prefixes = self._sequences(left_spk, core_lo,
                    max(0, 6*self.sr-(core_lo-a)), 8*self.sr-(core_lo-a), base)
                base_voice = {s: sum(y-x for x, y in intersect(voiced_by_speaker[s], a, b))
                              for s in speakers}
                base_solo = {s: sum(y-x for x, y in intersect(solos[s], a+self.fade, b-self.fade))
                             for s in speakers}
                for prefix in prefixes:
                    pre = sum(p.end-p.start-self.fade for p in prefix)
                    available = self.target-(b-a)-pre
                    if available < 0:
                        continue
                    counts = dict(base_voice)
                    for p in prefix:
                        counts[p.speaker] += max(0, voice_count(p)-2*self.fade)
                    missing = min(speakers, key=lambda s: counts[s])
                    suffixes = self._sequences(missing, core_hi, 0, available, base)
                    for suffix in suffixes:
                        support = prefix + suffix
                        if len({p.source for p in support}) != len(support):
                            continue
                        # Không dùng hai lát trùng âm thanh, kể cả nhãn nguồn khác.
                        if any(min(p.end,q.end)>max(p.start,q.start) for p,q in combinations(support,2)):
                            continue
                        total_voice, total_solo = dict(base_voice), dict(base_solo)
                        for p in support:
                            amount = max(0, voice_count(p)-2*self.fade)
                            total_voice[p.speaker] += amount
                            total_solo[p.speaker] += amount
                        if min(total_solo.values()) < self.sr:
                            continue
                        duration = b-a+pre+sum(p.end-p.start-self.fade for p in suffix)
                        ratio_error = abs(total_voice[speakers[0]]-total_voice[speakers[1]]) / max(1, sum(total_voice.values()))
                        retained = max(0, min(b,host_hi)-max(a,host_lo))
                        # Giữ nền có trọng số cao; cân bằng vẫn có thể thắng
                        # khi chỉ cần đổi một ít ngữ cảnh. Không ép đúng 1:1.
                        loss = 1-retained/max(1, min(host_hi-host_lo,self.target))
                        score = (0.45*loss + 0.40*ratio_error +
                                 0.12*(self.target-duration)/self.target + 0.015*len(support))
                        rank = (score, len(support), -retained)
                        if best is None or rank < best[0]:
                            best = (rank, prefix, base, suffix, total_voice, method)
        if best is None:
            self.detail = "no_layout_with_clean_support_and_6_8s_anchor"
            return None
        _, prefix, base, suffix, voice, method = best
        result = self._assemble(prefix, base, suffix, speakers, core_lo, core_hi, solos)
        result.layout.update({"policy": POLICY_VERSION, "host_segment": str(host.index),
            "host_source_samples": [host_lo,host_hi], "cut_method": method,
            "estimated_voice_seconds": {s: v/self.sr for s,v in voice.items()},
            "ratio": max(voice.values())/max(1,min(voice.values())),
            "context_seconds": [(core_lo-base.start)/self.sr,(base.end-core_hi)/self.sr],
            "context_shortfall_seconds": [max(0,self.context-(core_lo-base.start))/self.sr,
                                          max(0,self.context-(base.end-core_hi))/self.sr],
            "overlaps": [[p["overlap_start"],p["overlap_end"]] for p in group]})
        return result

    def _assemble(self, prefix, base, suffix, speakers, core_lo, core_hi, solos):
        pieces = list(prefix)+( [base] )+list(suffix)
        output = None
        maps, probes = [], {s: [] for s in speakers}
        base_offset = None
        for k,p in enumerate(pieces):
            chunk = self.waveform[p.start:p.end].astype(np.float32,copy=True)
            start = 0 if output is None else len(output)-self.fade
            if output is None:
                output = chunk
            else:
                ramp = np.linspace(0,1,self.fade,dtype=np.float32)
                output[-self.fade:] = output[-self.fade:]*(1-ramp)+chunk[:self.fade]*ramp
                output = np.concatenate((output,chunk[self.fade:]))
            end = start+len(chunk)
            safe_lo = p.start+(self.fade if k else 0)
            safe_hi = p.end-(self.fade if k+1<len(pieces) else 0)
            for speaker in speakers:
                ranges = solos[speaker] if p.kind == "base" else (
                    self.cuts.analyse(p.start,p.end)[1] if p.speaker == speaker else [])
                for a,b in intersect(ranges,safe_lo,safe_hi):
                    probes[speaker].append((start+a-p.start,start+b-p.start))
            if p.kind == "base":
                base_offset = start
            maps.append({"kind":p.kind,"speaker":p.speaker,"source_segment":p.source,
                         "source_samples":[p.start,p.end],"window_samples":[start,end],
                         "cuts":[p.start_cut,p.end_cut]})
        core = (base_offset+core_lo-base.start,base_offset+core_hi-base.start)
        return Window(output,core,probes,{
            "sample_rate":self.sr,"duration_seconds":len(output)/self.sr,
            "core_samples":list(core),"core_source_samples":[core_lo,core_hi],
            "core_start_seconds":core[0]/self.sr,"pieces":maps,
            "crossfade_samples":self.fade,
            "probe_samples":{s:[list(r) for r in rs] for s,rs in probes.items()}})
