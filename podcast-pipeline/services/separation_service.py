import bisect
import collections
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from schemas.audio import AudioData
from schemas.segment import Segment, EnhancedSegment
from algorithms.diarization.overlap import detect_overlapping_segments

# --- Window ---------------------------------------------------------------
# Sidon processes 20s chunks (CHUNK_SECONDS in sidon_infer.py). Start there and
# only grow when the window does not yet contain enough solo speech to score.
TSE_WINDOW_TARGET = float(os.environ.get("TSE_WINDOW_TARGET", "20.0"))
# 40s left a real case 3.3s short of the interrupting speaker's nearest
# turn. Sidon chunks internally at 20s, so a longer window costs stitching
# passes rather than memory.
TSE_WINDOW_MAX = float(os.environ.get("TSE_WINDOW_MAX", "48.0"))
TSE_WINDOW_GROW = float(os.environ.get("TSE_WINDOW_GROW", "4.0"))
# Voiced solo audio one speaker needs for ECAPA to produce a usable embedding.
TSE_MIN_SOLO = float(os.environ.get("TSE_MIN_SOLO", "2.0"))

# --- Job grouping ---------------------------------------------------------
TSE_JOB_MERGE_GAP = float(os.environ.get("TSE_JOB_MERGE_GAP", "8.0"))
TSE_JOB_MAX_SPAN = float(os.environ.get("TSE_JOB_MAX_SPAN", "120.0"))
TSE_RETRY_SPLIT = os.environ.get("TSE_RETRY_SPLIT", "1") not in ("0", "false", "False")

# --- Enrollment -----------------------------------------------------------
TSE_ENROLL_BUDGET = float(os.environ.get("TSE_ENROLL_BUDGET", "8.0"))
TSE_ENROLL_MIN_CLIP = float(os.environ.get("TSE_ENROLL_MIN_CLIP", "0.35"))
TSE_ENROLL_MIN_TOTAL = float(os.environ.get("TSE_ENROLL_MIN_TOTAL", "1.5"))

# --- QC -------------------------------------------------------------------
# NOT CALIBRATED. DialogueSidon resynthesises speech through a VAE decoder
# rather than masking the mixture, so ECAPA cosine against a natural enrollment
# sits in a lower, unmeasured range. Read the sim percentiles in the [TSE] log
# line and the clips under separation/failed/ before trusting either number.
TSE_QC_SIM_THRESHOLD = float(os.environ.get("TSE_QC_SIM_THRESHOLD", "0.25"))
TSE_NOT_A_MARGIN = float(os.environ.get("TSE_NOT_A_MARGIN", "0.15"))
TSE_SILENCE_RMS = float(os.environ.get("TSE_SILENCE_RMS", "0.002"))

TSE_DUMP_FAILED = os.environ.get("TSE_DUMP_FAILED", "1") not in ("0", "false", "False")

# Closed vocabulary of failure reasons, so every discarded overlap can be
# counted and grepped rather than vanishing into a bare `continue`.
REASONS = (
    "no_enroll",        # speaker lacks enough clean audio for an enrollment
    "no_window",        # no window up to TSE_WINDOW_MAX satisfies the criteria
    "multi_speaker",    # >2 speakers in the window; Sidon is a 2-source model
    "qc_sim",           # track scored below TSE_QC_SIM_THRESHOLD
    "unscorable",       # too little voiced audio to judge (not a failure to separate)
    "not_a_fail",       # the "not-A" relative test did not pass
    "already_spliced",  # another job already wrote this span
    "short_track",      # separator returned too few samples
)


class TargetExtractionService:
    """Isolate overlapping speech with blind separation plus ECAPA assignment.

    Every overlap ends up in exactly one of two places on its EnhancedSegment:
    tse_spans (separated) or tse_failed_spans (not, with a reason). Nothing is
    dropped silently -- test_every_overlap_is_accounted_for enforces that.
    """

    def __init__(self, tse_model, logger=None, dump_dir: Optional[str] = None):
        self.tse_model = tse_model
        self.logger = logger
        self.dump_dir = dump_dir
        self.stats = collections.Counter()
        self.sims = []
        self.overlap_durations = []
        self.failures = []          # (start, end, speaker, reason, detail)
        self._dump_warned = False

    def reset_stats(self):
        """Clear per-file counters.

        The service instance is reused across a batch, so without this the
        second file's [TSE] summary and _tse_report.json would report the
        running total rather than that file.
        """
        self.stats = collections.Counter()
        self.sims = []
        self.overlap_durations = []
        self.failures = []

    # ------------------------------------------------------------------
    def _fail(self, enh_seg, start, end, reason, detail=""):
        """Record a span that stayed as raw mixture, and why."""
        assert reason in REASONS, f"unknown reason {reason!r}"
        self.stats[f"fail_{reason}"] += 1
        if enh_seg is not None:
            enh_seg.tse_failed_spans.append((start, end, reason, detail))
        self.failures.append((start, end, getattr(enh_seg, "speaker", "?"), reason, detail))
        if self.logger:
            self.logger.debug(f"[TSE] {reason} @ {start:.2f}s {detail}")

    def _report_stats(self):
        if not self.logger:
            return
        s = self.stats
        self.logger.info(
            f"[TSE] jobs={s['jobs']} pairs={s['pairs']} spliced={s['spliced']} "
            f"retried={s['retried']}"
        )
        fails = {r: s[f"fail_{r}"] for r in REASONS if s[f"fail_{r}"]}
        self.logger.info(f"[TSE] failures: {fails or 'none'}")
        if self.overlap_durations:
            d = np.array(self.overlap_durations)
            self.logger.info(
                f"[TSE] overlap dur: min={d.min():.2f}s p50={np.median(d):.2f}s "
                f"max={d.max():.2f}s | <0.5s={int((d < 0.5).sum())}"
            )
        if self.sims:
            a = np.array(self.sims)
            self.logger.info(
                f"[TSE] ECAPA sim: p10={np.percentile(a, 10):.2f} "
                f"p50={np.percentile(a, 50):.2f} p90={np.percentile(a, 90):.2f} "
                f"max={a.max():.2f} (threshold {TSE_QC_SIM_THRESHOLD}, NOT calibrated)"
            )
        else:
            self.logger.info("[TSE] no similarity was computed at all")

    def write_report(self, save_dir: str, audio_name: str):
        """Dump per-span audit so failures can be inspected instead of guessed at."""
        path = os.path.join(save_dir, f"{audio_name}_tse_report.json")
        payload = {
            "thresholds": {
                "qc_sim": TSE_QC_SIM_THRESHOLD, "not_a_margin": TSE_NOT_A_MARGIN,
                "min_solo": TSE_MIN_SOLO, "window_target": TSE_WINDOW_TARGET,
                "window_max": TSE_WINDOW_MAX,
            },
            "stats": dict(self.stats),
            "sim_percentiles": (
                {p: float(np.percentile(self.sims, p)) for p in (10, 50, 90)}
                if self.sims else None
            ),
            "failures": [
                {"start": a, "end": b, "speaker": spk, "reason": r, "detail": d}
                for a, b, spk, r, d in self.failures
            ],
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to write TSE report: {e}")

    def mine_enrollments(self, segments: List[Segment], audio: AudioData, min_dur: float = 2.0, top_k: int = 5) -> Dict[str, List[np.ndarray]]:
        """
        Component 1: Extract clean, non-overlapping segments for each speaker to serve as TSE enrollments.
        """
        if self.logger: self.logger.info("Mining enrollments for Target Speaker Extraction...")
        
        # Convert to dicts for overlap detection
        seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "index": s.index} for s in segments]
        
        # We need to find segments that DO NOT overlap with anything else.
        # A simple way is to use detect_overlapping_segments and exclude them.
        overlaps = detect_overlapping_segments(seg_dicts, overlap_threshold=0.1)
        overlap_ranges = [(o["overlap_start"], o["overlap_end"]) for o in overlaps]
        
        enrollments = {}
        sr = audio.sample_rate
        waveform = audio.waveform

        # Sorting once lets a single sweep find each segment's intersecting
        # overlaps, instead of rescanning the full overlap list per segment.
        overlap_ranges.sort(key=lambda x: x[0])
        overlap_starts = [o[0] for o in overlap_ranges]

        # The clean sub-segments of a segment do not depend on the duration
        # threshold, only the filter over them does. Compute them once.
        clean_by_speaker: Dict[str, List[Tuple[float, float]]] = {}
        for s in segments:
            curr_start = s.start
            sub_segments = []

            # Overlaps starting at or after s.end cannot intersect it.
            end_idx = bisect.bisect_left(overlap_starts, s.end)
            for o_start, o_end in overlap_ranges[:end_idx]:
                if o_end <= s.start:
                    continue
                if curr_start < o_start:
                    sub_segments.append((curr_start, o_start))
                curr_start = max(curr_start, o_end)

            if curr_start < s.end:
                sub_segments.append((curr_start, s.end))

            clean_by_speaker.setdefault(s.speaker, []).extend(sub_segments)

        for spk in set(s.speaker for s in segments):
            candidates = clean_by_speaker.get(spk, [])

            # The old fallback ladder ended at 0.1s. A 0.1s ECAPA enrollment is
            # not "slightly worse", it is noise -- and it failed silently, because
            # the bad embedding surfaced later as a low similarity that looked
            # like a separation failure. Gather clips up to a time budget instead
            # and refuse outright when there is not enough.
            candidates = [c for c in candidates if (c[1] - c[0]) >= TSE_ENROLL_MIN_CLIP]
            candidates.sort(key=lambda c: c[1] - c[0], reverse=True)

            picked, total = [], 0.0
            for start, end in candidates:
                if total >= TSE_ENROLL_BUDGET:
                    break
                picked.append(waveform[int(start * sr):int(end * sr)].copy())
                total += end - start

            if total < TSE_ENROLL_MIN_TOTAL:
                if self.logger:
                    self.logger.warning(
                        f"Speaker {spk}: only {total:.2f}s of clean audio "
                        f"(need >={TSE_ENROLL_MIN_TOTAL}s); enrollment would be unreliable, "
                        "so every overlap involving this speaker is skipped."
                    )
                enrollments[spk] = []
            else:
                enrollments[spk] = picked

        return enrollments

    def _cross_fade(self, orig_audio: np.ndarray, new_audio: np.ndarray, fade_samples: int) -> np.ndarray:
        """Replace a portion of orig_audio with new_audio, crossfading the joins.

        The ramps are equal-power (sin/cos) rather than linear. Two uncorrelated
        signals -- which separated speech and the original mixture are -- sum in
        power, not amplitude, so complementary linear ramps drop the level to
        0.71 amplitude / 0.51 power halfway through the fade. That audible dip
        lands on a fixed 20ms at each end, which is 16.7% of a 0.24s backchannel
        and leaves an already-short clip quieter exactly where ASR needs it.

        The fade also shrinks with the clip so a short splice is not mostly
        ramp: a 0.24s core keeps at least three quarters of its length at full
        strength.
        """
        result = orig_audio.copy()
        limit = min(len(orig_audio), len(new_audio))
        if limit == 0:
            return result

        # Never spend more than an eighth of the splice on each ramp.
        fade_samples = min(fade_samples, limit // 8)
        if fade_samples <= 0:
            result[:limit] = new_audio[:limit]
            return result

        t = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)
        ramp_in = np.sin(t * (np.pi / 2.0))
        ramp_out = np.cos(t * (np.pi / 2.0))

        # Fade from the original into the separated track.
        result[:fade_samples] = (orig_audio[:fade_samples] * ramp_out
                                 + new_audio[:fade_samples] * ramp_in)

        # Full-strength separated audio in the middle.
        mid_limit = limit - fade_samples
        result[fade_samples:mid_limit] = new_audio[fade_samples:mid_limit]

        # And back out to the original.
        result[mid_limit:limit] = (orig_audio[mid_limit:limit] * ramp_in
                                   + new_audio[mid_limit:limit] * ramp_out)
        return result
            # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------
    @staticmethod
    def _intervals_by_speaker(segments) -> Dict[str, List[Tuple[float, float]]]:
        by_spk: Dict[str, List[Tuple[float, float]]] = {}
        for s in segments:
            by_spk.setdefault(s.speaker, []).append((s.start, s.end))
        for spk in by_spk:
            by_spk[spk].sort()
        return by_spk

    @staticmethod
    def _solo_spans(by_spk, spk, lo, hi) -> List[Tuple[float, float]]:
        """Parts of [lo, hi] where `spk` speaks and nobody else does."""
        own = [(max(a, lo), min(b, hi)) for a, b in by_spk.get(spk, []) if b > lo and a < hi]
        own = [(a, b) for a, b in own if b > a]
        if not own:
            return []
        others = []
        for other, ivals in by_spk.items():
            if other == spk:
                continue
            for a, b in ivals:
                if b > lo and a < hi:
                    others.append((max(a, lo), min(b, hi)))
        others.sort()

        solo = []
        for a, b in own:
            cur = a
            for oa, ob in others:
                if ob <= cur:
                    continue
                if oa >= b:
                    break
                if cur < oa:
                    solo.append((cur, min(oa, b)))
                cur = max(cur, ob)
                if cur >= b:
                    break
            if cur < b:
                solo.append((cur, b))
        return [(a, b) for a, b in solo if b - a > 1e-6]

    @staticmethod
    def _speakers_in(by_spk, lo, hi):
        return {spk for spk, ivals in by_spk.items()
                if any(b > lo and a < hi for a, b in ivals)}

    def _build_window(self, by_spk, spk_a, spk_b, ov_lo, ov_hi, total_dur):
        """Grow a window around the overlap until one speaker can be scored.

        Returns (lo, hi, solo_a, solo_b, anchor) on success, or (None, reason).

        Only ONE speaker needs solo audio. Channel assignment is a 2x2 decision:
        if track 1 matches A strongly where A speaks alone and track 2 does not,
        the assignment is settled without knowing anything about B. Requiring
        both would reject the common case here -- A talking for 28s with B only
        interjecting.
        """
        pad = max(0.0, (TSE_WINDOW_TARGET - (ov_hi - ov_lo)) / 2.0)
        lo, hi = max(0.0, ov_lo - pad), min(total_dur, ov_hi + pad)

        # Borrow from the other side when clamped at a file edge, so a window
        # near 0.0 still reaches its target length.
        if hi - lo < TSE_WINDOW_TARGET and total_dur >= TSE_WINDOW_TARGET:
            if lo <= 0.0:
                hi = min(total_dur, lo + TSE_WINDOW_TARGET)
            elif hi >= total_dur:
                lo = max(0.0, hi - TSE_WINDOW_TARGET)

        best_single = None
        for _ in range(24):
            present = self._speakers_in(by_spk, lo, hi)
            if len(present) > 2:
                # Sidon emits exactly two sources; a third voice would be folded
                # into one of them. Growing the window only makes that worse.
                return None, "multi_speaker"

            solo_a = self._solo_spans(by_spk, spk_a, lo, hi)
            solo_b = self._solo_spans(by_spk, spk_b, lo, hi)
            dur_a = sum(b - a for a, b in solo_a)
            dur_b = sum(b - a for a, b in solo_b)

            # Both scorable is strictly better: each speaker then gets a real
            # cosine instead of the weaker relative "not-A" test. Keep growing
            # for it, but remember the one-speaker window as a fallback so a
            # reachable anchor is never thrown away.
            if dur_a >= TSE_MIN_SOLO and dur_b >= TSE_MIN_SOLO:
                anchor = spk_a if dur_a >= dur_b else spk_b
                return (lo, hi, solo_a, solo_b, anchor), None
            # Keep the *best* single-speaker window, not the first one found.
            # Recording only the first froze the initial 20s window and made
            # every subsequent growth step pointless, which is why widening the
            # search never actually recovered the second speaker.
            if dur_a >= TSE_MIN_SOLO or dur_b >= TSE_MIN_SOLO:
                weaker = min(dur_a, dur_b)
                if best_single is None or weaker > best_single[0]:
                    best_single = (
                        weaker,
                        (lo, hi, solo_a, solo_b, spk_a if dur_a >= dur_b else spk_b),
                    )

            if (hi - lo) >= TSE_WINDOW_MAX or (lo <= 0.0 and hi >= total_dur):
                break

            # Grow toward whichever speaker still lacks solo audio, instead of
            # padding both sides equally. A backchannel sits inside a long turn
            # by the other speaker, so the symmetric window spends its whole
            # budget on audio that speaker already has plenty of: on this
            # corpus, 40s centred gave 0.00s of solo for the interrupting
            # speaker, while the same 40s pushed toward their nearest turn gave
            # 18.94s.
            need = None
            if dur_a < TSE_MIN_SOLO and dur_b >= TSE_MIN_SOLO:
                need = spk_a
            elif dur_b < TSE_MIN_SOLO and dur_a >= TSE_MIN_SOLO:
                need = spk_b
            elif dur_a < TSE_MIN_SOLO and dur_b < TSE_MIN_SOLO:
                need = spk_a if dur_a <= dur_b else spk_b

            budget = min(2.0 * TSE_WINDOW_GROW, TSE_WINDOW_MAX - (hi - lo))
            left_room, right_room = lo, total_dur - hi

            # bias is the share of the step to spend on the right: 0.0 means
            # grow left only, 1.0 right only, 0.5 split evenly.
            bias = self._growth_bias(by_spk, need, lo, hi) if need else 0.5
            take_right = min(right_room, budget * bias)
            take_left = min(left_room, budget * (1.0 - bias))
            # Only spill onto the other side once the preferred one is exhausted
            # (it hit the file edge). Splitting the remainder unconditionally
            # would spend budget away from the speaker we are trying to reach.
            spare = budget - take_left - take_right
            if spare > 0:
                if bias >= 0.5:
                    take_left = min(left_room, take_left + spare)
                else:
                    take_right = min(right_room, take_right + spare)

            if take_left <= 0.0 and take_right <= 0.0:
                break
            lo, hi = max(0.0, lo - take_left), min(total_dur, hi + take_right)

        if best_single is not None:
            return best_single[1], None
        return None, "no_window"

    @staticmethod
    def _growth_bias(by_spk, spk, lo, hi) -> float:
        """How strongly to grow right rather than left to reach `spk`.

        Returns 1.0 to spend the whole step on the right, 0.0 on the left, and
        0.5 when neither side is closer (or the speaker is unreachable).
        """
        ivals = by_spk.get(spk) or []
        left = [b for a, b in ivals if b <= lo]
        right = [a for a, b in ivals if a >= hi]
        dist_left = lo - max(left) if left else float("inf")
        dist_right = min(right) - hi if right else float("inf")

        if dist_left == float("inf") and dist_right == float("inf"):
            return 0.5
        if dist_right < dist_left:
            return 1.0
        if dist_left < dist_right:
            return 0.0
        return 0.5

    @staticmethod
    def _to_samples(spans, window_lo, sr, n_samples):
        out = []
        for a, b in spans:
            i, j = int((a - window_lo) * sr), int((b - window_lo) * sr)
            i, j = max(0, i), min(n_samples, j)
            if j > i:
                out.append((i, j))
        return out

    def _group_jobs(self, pairs):
        """One separation call per (speaker pair, neighbourhood).

        Two overlaps a few seconds apart would otherwise each get their own
        window covering nearly the same audio: twice the diffusion cost, and two
        independent separations spliced over each other in the shared region.
        TSE_JOB_MAX_SPAN caps the chain so a backchannel every 5s cannot silently
        merge the whole file into one job.
        """
        buckets: Dict[frozenset, list] = {}
        for p in pairs:
            buckets.setdefault(frozenset((p["seg1"]["speaker"], p["seg2"]["speaker"])), []).append(p)

        jobs = []
        for spk_pair, plist in buckets.items():
            if len(spk_pair) != 2:
                continue
            spk_a, spk_b = sorted(spk_pair)
            plist.sort(key=lambda p: p["overlap_start"])
            current = []
            for p in plist:
                if current:
                    gap = p["overlap_start"] - current[-1]["overlap_end"]
                    span = p["overlap_end"] - current[0]["overlap_start"]
                    if gap > TSE_JOB_MERGE_GAP or span > TSE_JOB_MAX_SPAN:
                        jobs.append((spk_a, spk_b, current))
                        current = []
                current.append(p)
            if current:
                jobs.append((spk_a, spk_b, current))
        return jobs

    # ------------------------------------------------------------------
    def _dump_tracks(self, subdir, tag, mixture, track_1, track_2, sr):
        """Write the mixture and both separated tracks for one job.

        Listening to these is the only way to tell a real separation failure
        from a mis-calibrated QC threshold, so successful jobs are dumped too,
        not just failures.
        """
        if not (TSE_DUMP_FAILED and self.dump_dir):
            return
        try:
            import soundfile as sf
            d = os.path.join(self.dump_dir, subdir)
            os.makedirs(d, exist_ok=True)
            sf.write(os.path.join(d, f"{tag}_mix.wav"), mixture, sr)
            if track_1 is not None:
                sf.write(os.path.join(d, f"{tag}_trackA.wav"), track_1, sr)
                sf.write(os.path.join(d, f"{tag}_trackB.wav"), track_2, sr)
        except Exception as e:
            # Warn once rather than per clip: a missing soundfile or a read-only
            # output dir would otherwise disable the audit trail in silence.
            if self.logger and not self._dump_warned:
                self._dump_warned = True
                self.logger.warning(
                    f"[TSE] track dumps disabled: {type(e).__name__}: {e}"
                )

    def _dump_failed(self, tag, mixture, track_1, track_2, sr):
        self._dump_tracks("failed", tag, mixture, track_1, track_2, sr)

    # ------------------------------------------------------------------
    def process_overlaps(self, segments: List[Segment], audio: AudioData, overlap_threshold: float = 0.1) -> List[EnhancedSegment]:
        if not self.tse_model:
            return [EnhancedSegment(**s.__dict__) for s in segments]

        if self.logger:
            self.logger.info("Processing overlaps with Target Speaker Extraction (TSE)")

        # với mỗi thành phần trong segments, tạo một dictionary chứa thông tin start, end, speaker và index
        seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "index": s.index} for s in segments]

        # lọc ra những bộ key valuf có overlap >= overlap_threshold
        pairs = detect_overlapping_segments(seg_dicts, overlap_threshold=overlap_threshold, logger=self.logger)

        # Tạo danh sách các EnhancedSegment từ danh sách segments
        enhanced = [EnhancedSegment(**s.__dict__) for s in segments]
        # Lấy sample rate và waveform từ audio
        sr = audio.sample_rate
        waveform = audio.waveform
        total_dur = len(waveform) / sr
        for e in enhanced:
            e.enhanced_audio = waveform[int(e.start * sr):int(e.end * sr)].copy()

        if not pairs:
            if self.logger:
                self.logger.info(
                    f"[TSE] no overlap >= {overlap_threshold}s among {len(segments)} segments"
                )
            return enhanced

        enrollments = self.mine_enrollments(segments, audio)
        by_spk = self._intervals_by_speaker(segments)
        seg_by_index = {s.index: s for s in enhanced}

        queue = self._group_jobs(pairs)
        if self.logger:
            self.logger.info(f"[TSE] {len(pairs)} overlap pairs -> {len(queue)} separation jobs")

        from tqdm import tqdm
        pbar = tqdm(total=len(queue), desc="[TSE Extractor]", leave=True)
        while queue:
            spk_a, spk_b, plist = queue.pop(0)
            pbar.total = pbar.n + len(queue) + 1
            pbar.update(1)
            is_retry = len(plist) == 1 and self.stats["jobs"] >= len(pairs)
            self.stats["jobs"] += 1
            self.stats["pairs"] += len(plist)
            for p in plist:
                self.overlap_durations.append(p["overlap_end"] - p["overlap_start"])

            def fail_all(reason, detail=""):
                for p in plist:
                    for sd in (p["seg1"], p["seg2"]):
                        self._fail(seg_by_index.get(sd["index"]),
                                   p["overlap_start"], p["overlap_end"], reason, detail)

            if not enrollments.get(spk_a) or not enrollments.get(spk_b):
                missing = spk_a if not enrollments.get(spk_a) else spk_b
                fail_all("no_enroll", f"speaker={missing}")
                continue

            job_lo = min(p["overlap_start"] for p in plist)
            job_hi = max(p["overlap_end"] for p in plist)
            built, reason = self._build_window(by_spk, spk_a, spk_b, job_lo, job_hi, total_dur)
            if built is None:
                fail_all(reason, f"span={job_hi - job_lo:.2f}s")
                continue

            win_lo, win_hi, solo_a, solo_b, anchor = built
            lo_f, hi_f = int(win_lo * sr), int(win_hi * sr)
            window_audio = waveform[lo_f:hi_f].copy()
            n = len(window_audio)
            core = (int((job_lo - win_lo) * sr), int((job_hi - win_lo) * sr))

            track_A, track_B, sim_A, sim_B, diag = self.tse_model.separate_two_speakers(
                window_audio,
                enroll_A=enrollments[spk_a], enroll_B=enrollments[spk_b],
                sample_rate=sr, id_A=spk_a, id_B=spk_b,
                probe_A=self._to_samples(solo_a, win_lo, sr, n),
                probe_B=self._to_samples(solo_b, win_lo, sr, n),
                core_range=core,
            )
            for sim in (sim_A, sim_B):
                if sim is not None:
                    self.sims.append(sim)

            # Per-track gating. Discarding both whenever one fails threw away a
            # clean extraction of the dominant speaker because a 0.3s
            # backchannel scored badly.
            accepted, rejected = {}, {}
            for spk, track, sim in ((spk_a, track_A, sim_A), (spk_b, track_B, sim_B)):
                if sim is not None:
                    if sim >= TSE_QC_SIM_THRESHOLD:
                        accepted[spk] = (track, sim)
                    else:
                        rejected[spk] = ("qc_sim", f"sim={sim:.2f} th={TSE_QC_SIM_THRESHOLD}")
                    continue

                # No solo region for this speaker. "Is this track B?" is
                # unanswerable on a sub-second core, but "is this track just a
                # copy of the anchor?" only needs a relative comparison.
                own, other, rms = diag["anchor_self"], diag["anchor_other"], diag["other_rms"]
                if rms is not None and rms < TSE_SILENCE_RMS:
                    rejected[spk] = ("unscorable", f"rms={rms:.5f}")
                elif own is None or other is None:
                    rejected[spk] = ("unscorable", "no core embedding")
                elif (own - other) > TSE_NOT_A_MARGIN:
                    accepted[spk] = (track, None)
                    self.stats["accept_not_a"] += 1
                else:
                    rejected[spk] = ("not_a_fail",
                                     f"margin={own - other:.2f} th={TSE_NOT_A_MARGIN}")

            if not accepted:
                if len(plist) > 1 and TSE_RETRY_SPLIT:
                    # A grouped job failing as a whole must not condemn every
                    # overlap in it; retry each one with its own window.
                    self.stats["retried"] += 1
                    self.stats["pairs"] -= len(plist)
                    self.stats["jobs"] -= 1
                    queue.extend([(spk_a, spk_b, [p]) for p in plist])
                    continue
                for p in plist:
                    for sd in (p["seg1"], p["seg2"]):
                        r, d = rejected.get(sd["speaker"], ("unscorable", "no verdict"))
                        self._fail(seg_by_index.get(sd["index"]),
                                   p["overlap_start"], p["overlap_end"], r, d)
                self._dump_failed(f"{job_lo:.2f}_{spk_a}_{spk_b}", window_audio, track_A, track_B, sr)
                continue

            # Audit trail for the jobs that succeeded: what the separator was
            # given, what it returned, and how each track scored.
            if self.logger:
                who = ", ".join(
                    f"{spk}=" + (f"{sim:.2f}" if sim is not None else "not-A")
                    for spk, (_, sim) in accepted.items()
                )
                self.logger.info(
                    f"[TSE:sep] {win_lo:.2f}-{win_hi:.2f}s ({win_hi - win_lo:.1f}s window) "
                    f"anchor={anchor} solo_a={sum(b - a for a, b in solo_a):.1f}s "
                    f"solo_b={sum(b - a for a, b in solo_b):.1f}s | accepted: {who}"
                    + (f" | rejected: {sorted(rejected)}" if rejected else "")
                )
            self._dump_tracks("separated", f"{job_lo:.2f}_{spk_a}_{spk_b}",
                              window_audio, track_A, track_B, sr)

            fade_samples = int(0.02 * sr)
            for p in plist:
                ov_lo, ov_hi = p["overlap_start"], p["overlap_end"]
                for sd in (p["seg1"], p["seg2"]):
                    spk = sd["speaker"]
                    enh = seg_by_index.get(sd["index"])
                    if enh is None:
                        continue
                    if spk not in accepted:
                        r, d = rejected.get(spk, ("unscorable", "no verdict"))
                        self._fail(enh, ov_lo, ov_hi, r, d)
                        continue

                    track, sim = accepted[spk]
                    src = int((ov_lo - win_lo) * sr)
                    dst = int((ov_lo - enh.start) * sr)
                    if src < 0 or dst < 0:
                        self._fail(enh, ov_lo, ov_hi, "short_track", "negative offset")
                        continue
                    limit = min(int((ov_hi - ov_lo) * sr), len(track) - src,
                                len(enh.enhanced_audio) - dst)
                    if limit <= 0:
                        self._fail(enh, ov_lo, ov_hi, "short_track", f"limit={limit}")
                        continue
                    if any(not (ov_hi <= a or ov_lo >= b) for a, b, _ in enh.tse_spans):
                        self._fail(enh, ov_lo, ov_hi, "already_spliced", "")
                        continue

                    enh.enhanced_audio[dst:dst + limit] = self._cross_fade(
                        enh.enhanced_audio[dst:dst + limit], track[src:src + limit], fade_samples)
                    enh.tse = True
                    enh.tse_spans.append((ov_lo, ov_lo + limit / sr,
                                          float(sim) if sim is not None else -1.0))
                    self.stats["spliced"] += 1
                    if self.logger:
                        self.logger.info(
                            f"[TSE:splice] seg {sd['index']} spk={spk} "
                            f"{ov_lo:.2f}-{ov_lo + limit / sr:.2f}s "
                            f"({limit / sr:.2f}s) sim="
                            + (f"{sim:.2f}" if sim is not None else "not-A")
                        )

        pbar.close()
        self._report_stats()
        return enhanced

    # ------------------------------------------------------------------
    def export_sdlm_dual_channel(self, enhanced_segments: List[EnhancedSegment], audio_duration: float, sr: int,
                                 strict: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct two continuous tracks for SDLM / full-duplex training.

        strict=True zeroes every span recorded in tse_failed_spans. Those spans
        still contain the interfering speaker, so writing them would train track
        0 on audio containing speaker 1 -- label contamination that is far more
        expensive to discover later than a shorter corpus is now.
        """
        total_samples = int(audio_duration * sr)
        track_0 = np.zeros(total_samples, dtype=np.float32)
        track_1 = np.zeros(total_samples, dtype=np.float32)

        speakers = sorted({s.speaker for s in enhanced_segments})
        if not speakers:
            return track_0, track_1
        if len(speakers) > 2 and self.logger:
            self.logger.warning(
                f"export_sdlm_dual_channel: expected 2 speakers, found {len(speakers)}; "
                "the rest are ignored."
            )
        spk_0 = speakers[0]
        spk_1 = speakers[1] if len(speakers) > 1 else None

        written = dropped = 0
        for seg in enhanced_segments:
            if seg.speaker not in (spk_0, spk_1) or seg.enhanced_audio is None:
                continue
            start_idx = int(seg.start * sr)
            end_idx = min(total_samples, start_idx + len(seg.enhanced_audio))
            if end_idx <= start_idx:
                continue
            chunk = seg.enhanced_audio[: end_idx - start_idx].copy()

            if strict:
                keep = np.ones(end_idx - start_idx, dtype=bool)
                # getattr, not seg.tse_failed_spans: a checkpoint pickled before
                # this field existed restores an object without it, and pickle
                # does not backfill dataclass defaults.
                for a, b, _reason, _detail in getattr(seg, "tse_failed_spans", ()):
                    i = max(start_idx, int(a * sr)) - start_idx
                    j = min(end_idx, int(b * sr)) - start_idx
                    if j > i:
                        keep[i:j] = False
                dropped += int((~keep).sum())
                written += int(keep.sum())
                chunk = chunk * keep
            else:
                written += end_idx - start_idx

            if seg.speaker == spk_0:
                track_0[start_idx:end_idx] += chunk
            else:
                track_1[start_idx:end_idx] += chunk

        if strict and self.logger:
            total = written + dropped
            pct = (100.0 * dropped / total) if total else 0.0
            self.logger.info(
                f"[SDLM] wrote {written / sr:.1f}s, zeroed {dropped / sr:.1f}s ({pct:.1f}%) "
                "of un-separated overlap to avoid cross-speaker leakage."
            )
            if pct > 5.0:
                self.logger.warning(
                    f"[SDLM] {pct:.1f}% of speech dropped as contaminated -- that is the "
                    "TSE failure rate landing in your dataset. Check the [TSE] counters."
                )

        np.clip(track_0, -1.0, 1.0, out=track_0)
        np.clip(track_1, -1.0, 1.0, out=track_1)
        return track_0, track_1
