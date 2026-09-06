import bisect
import collections
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from schemas.audio import AudioData
from schemas.segment import Segment, EnhancedSegment
from algorithms.diarization.overlap import detect_overlapping_segments
from utils.audio_normalize import match_splice_level, safe_limit
from utils.mixture_window import window_for
from utils.music_map import MusicMap
from utils.enrollment_memory import EnrollmentMemory, ENABLED as TSE_MEMORY

# Stand-in for a service constructed without one, so the separation path does
# not have to branch on its presence.
_NO_MEMORY = EnrollmentMemory(enabled=False)


# Crossfade over each seam so the joins are not step discontinuities the
# decoder would read as acoustic events.
TSE_STITCH_FADE = float(os.environ.get("TSE_STITCH_FADE", "0.02"))
# Silence padded around the overlap so seam artifacts cannot bleed into the one
# span actually spliced back.
TSE_STITCH_GUARD = float(os.environ.get("TSE_STITCH_GUARD", "0.25"))

# --- Job grouping ---------------------------------------------------------
TSE_JOB_MERGE_GAP = float(os.environ.get("TSE_JOB_MERGE_GAP", "8.0"))
# Overlaps this close together are treated as one span rather than as a job of
# several. The stitched window -- the thing that puts both voices in front of
# the separator in equal measure -- only builds for a single overlap, because
# its assembled timeline cannot locate two of them at different offsets. So a
# pair 0.4s apart lost the balancing and fell back to a plain window, where the
# measured cost is real: across this file the spans whose window held under 40%
# target audio scored 0.518 similarity against 0.613 for the rest.
#
# Merging them instead splices the sliver between the two overlaps as well. That
# sliver is single-speaker audio the separator handles trivially, and 0.5s of it
# is a smaller price than separating the pair unbalanced.
TSE_OVERLAP_FUSE_GAP = float(os.environ.get("TSE_OVERLAP_FUSE_GAP", "0.6"))
TSE_JOB_MAX_SPAN = float(os.environ.get("TSE_JOB_MAX_SPAN", "120.0"))
TSE_RETRY_SPLIT = os.environ.get("TSE_RETRY_SPLIT", "1") not in ("0", "false", "False")

# --- Enrollment -----------------------------------------------------------
TSE_ENROLL_BUDGET = float(os.environ.get("TSE_ENROLL_BUDGET", "8.0"))
TSE_ENROLL_MIN_CLIP = float(os.environ.get("TSE_ENROLL_MIN_CLIP", "0.35"))
TSE_ENROLL_MIN_TOTAL = float(os.environ.get("TSE_ENROLL_MIN_TOTAL", "1.5"))

# --- QC -------------------------------------------------------------------
# NOT CALIBRATED. Read the sim percentiles in the [TSE] log
# line and the clips under separation/failed/ before trusting either number.
# ECAPA scores might sit lower than they would on natural speech. The default
# is set relatively low.
TSE_QC_SIM_THRESHOLD = float(os.environ.get("TSE_QC_SIM_THRESHOLD", "0.20"))
TSE_NOT_A_MARGIN = float(os.environ.get("TSE_NOT_A_MARGIN", "0.15"))
TSE_SILENCE_RMS = float(os.environ.get("TSE_SILENCE_RMS", "0.002"))

TSE_DUMP_FAILED = os.environ.get("TSE_DUMP_FAILED", "1") not in ("0", "false", "False")

# Closed vocabulary of failure reasons, so every discarded overlap can be
# counted and grepped rather than vanishing into a bare `continue`.
REASONS = (
    "no_enroll",        # speaker lacks enough clean audio for an enrollment
    "no_window",        # no window up to TSE_WINDOW_MAX satisfies the criteria
    "multi_speaker",    # >2 speakers in the window; the extractor is a 2-source model
    "qc_sim",           # track scored below TSE_QC_SIM_THRESHOLD
    "unscorable",       # too little voiced audio to judge (not a failure to separate)
    "not_a_fail",       # the "not-A" relative test did not pass
    "already_spliced",  # another job already wrote this span
    "short_track",      # separator returned too few samples
    "empty_track",      # track is silent exactly where the mixture has speech
    "same_speaker",     # both sides are the same speaker; nothing to separate
)


class TargetExtractionService:
    """Isolate overlapping speech with blind separation plus ECAPA assignment.

    Every overlap ends up in exactly one of two places on its EnhancedSegment:
    tse_spans (separated) or tse_failed_spans (not, with a reason). Nothing is
    dropped silently -- test_every_overlap_is_accounted_for enforces that.
    """

    def __init__(self, tse_model=None, logger=None, dump_dir: Optional[str] = None,
                 model_loader=None):
        self._tse_model = tse_model
        self.model_loader = model_loader
        self.logger = logger
        self.dump_dir = dump_dir
        self.stats = collections.Counter()
        self.sims = []
        self.overlap_durations = []
        self.failures = []          # (start, end, speaker, reason, detail)
        self._dump_warned = False
        # Set by the pipeline after the post-diarization music sweep. Empty
        # means "no reason to avoid anything", which is the right default both
        # when there is no music and when the detector never ran.
        self.music_map = MusicMap()
        # Speaker labels are per-file, so this is cleared in reset_stats() along
        # with the counters -- carrying it across files would attach one file's
        # voice to another file's speaker "1".
        self.memory = EnrollmentMemory(logger=logger)

    # The model is fetched from the loader on use, not captured at
    # construction. PipelineService loads each stage's models when that stage
    # runs, so a reference taken here would be None for every stage that had
    # not loaded yet -- and would stay None after it did.
    @property
    def tse_model(self):
        if self._tse_model is not None:
            return self._tse_model
        return getattr(self, "model_loader", None) and self.model_loader.get("separator")

    @tse_model.setter
    def tse_model(self, model):
        """Assigning the model directly still works, which is how callers that
        build the service by hand -- the tests among them -- supply one."""
        self._tse_model = model

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
        # Same reason the separator's own cache is cleared below: speaker "1"
        # in the next file is a different person, and a memory carried across
        # would enrol them on this file's voice. getattr because a service built
        # for a narrower purpose -- a test, a script -- has no memory to clear,
        # and that is not an error.
        memory = getattr(self, "memory", None)
        if memory is not None:
            memory.reset()
        # The separator caches enrollment embeddings under the diarizer's
        # speaker labels, and those restart at "1" for every file.
        reset = getattr(self.tse_model, "reset_speakers", None)
        if reset:
            reset()

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

    def report_payload(self) -> dict:
        """The audit dict, so callers can embed it instead of re-deriving it."""
        return self._report_payload()

    def write_report(self, save_dir: str, audio_name: str, payload: dict = None):
        """Dump per-span audit so failures can be inspected instead of guessed at.

        `payload` lets a caller supply counters captured earlier. Under
        stage-major execution the export happens in a later run() than the
        separation, and reset_stats() has cleared the counters by then -- so
        reading them here produced an empty report on every batch run. Passing
        the payload the separation stage checkpointed is what keeps it filled.
        """
        path = os.path.join(save_dir, f"{audio_name}_tse_report.json")
        if payload is None:
            payload = self._report_payload()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to write TSE report: {e}")
        return path

    def _report_payload(self) -> dict:
        payload = {
            "thresholds": {
                "qc_sim": TSE_QC_SIM_THRESHOLD, "not_a_margin": TSE_NOT_A_MARGIN,
                "min_solo": TSE_MIN_SOLO, "window_target": TSE_WINDOW_TARGET,
                "window_max": TSE_WINDOW_MAX,
            },
            # Recorded per run because it changes what the audio sounds like:
            # a report from a run without it is not comparable to one with it.
            # How much of the recording the enrollment search had to avoid. A
            # run with music but an empty map means the sweep did not happen.
            "enrollment_memory": (self.memory.summary()
                                  if TSE_MEMORY and getattr(self, "memory", None)
                                  else None),
            "music_map": self.music_map.summary(),
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
        return payload

    def mine_enrollments(self, segments: List[Segment], audio: AudioData, min_dur: float = 2.0, top_k: int = 5) -> Dict[str, List[np.ndarray]]:
        """
        Component 1: Extract clean, non-overlapping segments for each speaker to serve as TSE enrollments.
        """
        if self.logger: self.logger.info("Mining enrollments for Target Speaker Extraction...")
        
        # Convert to dicts for overlap detection
        # với mỗi thành phần trong segments, tạo một dictionary chứa thông tin start, end, speaker và index
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

        # Drop anything playing over music. This stage runs before the
        # per-segment music pass, so a "solo" span is only solo in the sense
        # that one person is talking -- the bed underneath is still there, and
        # an enrollment taken from it describes the speaker plus the backing
        # track. USEF is conditioned on this audio and nothing else, so a
        # contaminated enrollment is a contaminated extraction.
        #
        # getattr, not self.music_map: the attribute is set in __init__ and
        # again by the pipeline, but a service built for a narrower purpose --
        # a test, a one-off script -- has neither, and a missing map must read
        # as "nothing to avoid" rather than raise.
        music_map = getattr(self, "music_map", None)
        if music_map:
            for spk, spans in list(clean_by_speaker.items()):
                kept = []
                for a, b in spans:
                    kept.extend(music_map.clean_parts(a, b))
                clean_by_speaker[spk] = kept

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

    @staticmethod
    def _track_has_speech(host: np.ndarray, track: np.ndarray,
                          frame_sec: float = 0.02, sr_hint: int = 24000) -> bool:
        """True when `track` holds speech wherever `host` (the mixture) does.

        Compares short-frame energy rather than whole-clip RMS. A track that is
        silent over the first two thirds of a backchannel and then carries the
        other speaker's tail still passes an RMS threshold comfortably -- that
        is exactly the failure this exists to catch -- but it is obvious frame
        by frame.
        """
        n = min(len(host), len(track))
        if n == 0:
            return False
        frame = max(1, int(frame_sec * sr_hint))
        if n < frame * 2:
            # Too short to profile; fall back to "is there anything at all".
            return float(np.sqrt(np.mean(track[:n] ** 2) + 1e-12)) >= TSE_SILENCE_RMS

        m = n // frame
        h = np.sqrt((host[:m * frame].reshape(m, frame) ** 2).mean(axis=1) + 1e-12)
        t = np.sqrt((track[:m * frame].reshape(m, frame) ** 2).mean(axis=1) + 1e-12)

        # Frames where the mixture clearly has speech, relative to its own peak
        # so this holds at any recording level.
        voiced = h >= max(h.max() * 0.25, TSE_SILENCE_RMS)
        if not voiced.any():
            return True          # nothing to preserve here; not the track's fault

        # The track may legitimately be quieter -- the interferer is gone -- so
        # judge it against its own scale, not the mixture's.
        alive = t >= max(t.max() * 0.15, TSE_SILENCE_RMS * 0.5)
        return float(np.mean(alive[voiced])) >= 0.35

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







    def seams(self):
        """Where the recording was spliced, in the clock this stage works in.

        Excising sung and standalone-music stretches leaves points where two
        parts of the recording that were never adjacent now touch. Widening a
        mixture window across one would hand the separator audio from
        somewhere else entirely, whose speakers have nothing to do with the
        overlap being separated.

        Read through getattr for the same reason `music_map` is: the pipeline
        sets it, but a service built for a test or a one-off script has none,
        and no timeline must read as "nothing was cut".
        """
        timeline = getattr(self, "timeline", None)
        return timeline.seams() if timeline else []


    @staticmethod
    def _fuse_adjacent(plist, gap=None):
        """Join overlaps separated by less than `gap` into single spans.

        Two overlaps half a second apart are one interruption as far as the
        separator is concerned, but as two entries they make a job of several --
        and a multi-overlap job cannot use the stitched window, which is what
        balances the two voices. Fusing them keeps the balancing at the cost of
        splicing the short single-speaker sliver between them.

        Only the span is widened; the pair keeps its first entry's speakers and
        segments, which is what the caller reads.
        """
        gap = TSE_OVERLAP_FUSE_GAP if gap is None else gap
        if gap <= 0 or len(plist) < 2:
            return plist

        # Sorted here as well as by the caller. The merge walks forward and
        # compares each entry with the last kept one, so an out-of-order list
        # silently drops whatever precedes its predecessor -- losing an overlap
        # rather than failing, which is the worst way for this to go wrong.
        plist = sorted(plist, key=lambda p: p["overlap_start"])

        fused = [dict(plist[0])]
        for p in plist[1:]:
            if p["overlap_start"] - fused[-1]["overlap_end"] <= gap:
                fused[-1]["overlap_end"] = max(fused[-1]["overlap_end"],
                                               p["overlap_end"])
                fused[-1]["overlap_duration"] = (fused[-1]["overlap_end"]
                                                 - fused[-1]["overlap_start"])
            else:
                fused.append(dict(p))
        return fused

    def _group_jobs(self, pairs):
        """Overlaps that need no separation are collected, not dropped."""
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
                # A speaker overlapping themselves. Diarization can produce
                # this on its own, and merging a ghost speaker into a
                # neighbour creates more of it. There is nothing to separate --
                # one voice is already one source -- but it must not vanish
                # silently: every overlap ends up in tse_spans or
                # tse_failed_spans, and the caller records these from here.
                if not hasattr(self, "_same_speaker_pairs"):
                    self._same_speaker_pairs = []
                self._same_speaker_pairs.extend(plist)
                continue
            spk_a, spk_b = sorted(spk_pair)
            plist.sort(key=lambda p: p["overlap_start"])
            plist = self._fuse_adjacent(plist)
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
    def passthrough(self, segments, audio):
        """EnhancedSegments carrying the mixture, with nothing separated.

        What the separation stage returns when the profile turns it off. The
        shape has to match `process_overlaps` exactly -- ASR and the exporters
        read `enhanced_audio` either way -- so the difference is only that no
        overlap is replaced.
        """
        sr = audio.sample_rate
        enhanced = [EnhancedSegment(**s.__dict__) for s in segments]
        for e in enhanced:
            e.enhanced_audio = audio.waveform[int(e.start * sr):int(e.end * sr)].copy()
        return enhanced

    def process_overlaps(self, segments: List[Segment], audio: AudioData, overlap_threshold: float = 0.1) -> List[EnhancedSegment]:
        if not self.tse_model:
            return [EnhancedSegment(**s.__dict__) for s in segments]

        if self.logger:
            self.logger.info("Processing overlaps with Target Speaker Extraction (TSE)")

        seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "index": s.index} for s in segments]
        # lọc ra những bộ key value có overlap >= overlap_threshold
        pairs = detect_overlapping_segments(seg_dicts, overlap_threshold=overlap_threshold, logger=self.logger)

        # Tạo danh sách các EnhancedSegment từ danh sách segments
        enhanced = [EnhancedSegment(**s.__dict__) for s in segments]
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

        self._same_speaker_pairs = []
        queue = self._group_jobs(pairs)
        # Record them against both segments, the same way a real failure is.
        by_index = {e.index: e for e in enhanced}
        for p in self._same_speaker_pairs:
            for side in ("seg1", "seg2"):
                self._fail(by_index.get(p[side].get("index")),
                           p["overlap_start"], p["overlap_end"],
                           "same_speaker", f"speaker={p[side].get('speaker')}")
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

            # A stitched window only makes sense for a single overlap: the
            # assembled timeline no longer matches the recording, so several
            # overlaps at different offsets could not all be located in it.
            # USEF is told who to extract by its 8s enrollment, so the mixture
            # only has to carry the overlap itself. A target-conditioned masker has no
            # such failure as blind extractors, and the solo audio it was fed was
            # audio it did not need.
            #
            # What is left is the model's own 2s window: overlaps longer than
            # that are chunked internally, shorter ones are widened to reach it.
            win_lo, win_hi = window_for(job_lo, job_hi, self.seams(), total_dur)
            lo_f, hi_f = int(win_lo * sr), int(win_hi * sr)
            window_audio = waveform[lo_f:hi_f].copy()
            core = (int((job_lo - win_lo) * sr), int((job_hi - win_lo) * sr))

            # No solo speech in the window any more, so nothing to probe with.
            # `separate_two_speakers` reads an empty probe as "score the whole
            # track", which is the right question to ask of a track that is
            # supposed to be one speaker from end to end.
            solo_a = solo_b = ()
            probe_a_s = probe_b_s = []
            anchor = spk_a
            if self.logger:
                self.logger.info(
                    f"[TSE] {job_lo:.2f}-{job_hi:.2f}s ({job_hi - job_lo:.2f}s) "
                    f"-> {win_hi - win_lo:.2f}s window"
                    + ("" if win_hi - win_lo >= 2.0 - 1e-6
                       else "  (walled in by a cut; the model pads the rest)"))

            # Enrolments grown from earlier well-separated spans of this same
            # conversation, when the memory is on. The mined clips stay at the
            # front; this only appends.
            memory = getattr(self, "memory", None) or _NO_MEMORY
            enroll_a = memory.extend(spk_a, enrollments[spk_a], sr)
            enroll_b = memory.extend(spk_b, enrollments[spk_b], sr)

            track_A, track_B, sim_A, sim_B, diag = self.tse_model.separate_two_speakers(
                window_audio,
                enroll_A=enroll_a, enroll_B=enroll_b,
                sample_rate=sr, id_A=spk_a, id_B=spk_b,
                probe_A=probe_a_s,
                probe_B=probe_b_s,
                core_range=core,
            )

            for sim in (sim_A, sim_B):
                if sim is not None:
                    self.sims.append(sim)

            # Offer both tracks to the memory. Only ones well above the QC
            # threshold are kept, so a mis-assigned track cannot teach a speaker
            # its own mistake.
            for spk, track, sim in ((spk_a, track_A, sim_A), (spk_b, track_B, sim_B)):
                memory.offer(spk, track, sim, sr)

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

                    # Verify the track carries speech where the mixture does,
                    # on the span about to be written. Every other check scores
                    # a speaker's solo region, which can sit many seconds away:
                    # one case scored sim=0.67 from solo audio 19s earlier while
                    # the track was flat silence across the backchannel itself,
                    # and that silence went into the dataset labelled as speech.
                    host = enh.enhanced_audio[dst:dst + limit]
                    if not self._track_has_speech(host, track[src:src + limit]):
                        self._fail(enh, ov_lo, ov_hi, "empty_track",
                                   "silent where mixture has speech")
                        continue

                    # The separated track is quieter than the mixture it
                    # replaces -- the interfering speaker and the background
                    # are gone, which is the whole point -- so splicing it in
                    # raw leaves a level step at each join. On a backchannel of
                    # a few hundred milliseconds that step spans most of the
                    # clip. Matching RMS to the audio being replaced closes it
                    # with a scalar, before the crossfade smooths the edges.
                    patch = match_splice_level(
                        enh.enhanced_audio[dst:dst + limit], track[src:src + limit])
                    enh.enhanced_audio[dst:dst + limit] = self._cross_fade(
                        enh.enhanced_audio[dst:dst + limit], patch, fade_samples)
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

        # Segments are summed into these tracks, so overlapping same-speaker
        # spans can push the total past full scale. Hard clipping would fold
        # the waveform and spread broadband distortion across the spectrum --
        # the unnatural kind of artifact that costs ASR accuracy and that a
        # listener hears as crackle. Scaling instead keeps every relative level
        # intact and removes nothing.
        track_0, g0 = safe_limit(track_0)
        track_1, g1 = safe_limit(track_1)
        if self.logger and (g0 < 1.0 or g1 < 1.0):
            self.logger.info(
                f"[SDLM] limiter applied: track_0 x{g0:.3f}, track_1 x{g1:.3f} "
                "(summed segments exceeded full scale)")
        return track_0, track_1
