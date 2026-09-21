"""Find stretches of a recording worth cutting out as a two-person conversation.

This is `conversation_dataset_plan.md` (sections 2-5 and the crop rules of 7)
turned into code, on the transcript after speaker relabel. Nothing here touches
audio or a model: it reads segments, the cut timeline and the SSLAM noise and
music maps, and returns candidate spans with the numbers that justify them.

What makes a candidate:

  * Exactly two speakers, for the whole stretch. A third voice, even a "ừ",
    ends the block -- the corpus wants dialogue, not a panel.
  * No join. The audio is cut from the recording with its music removed, so a
    stretch that spans a cut is two pieces of the recording glued together.
  * No lasting noise. The SSLAM noise track is read per group, and a group
    staying over its level for `noisy_run_seconds` ends the block there, so a
    conversation is split around a lawnmower instead of being thrown away whole.
    Voices in the background are held to the strictest level: they are the ones
    that break diarization and put words in the transcript nobody said.
  * 60-240 seconds, starting and ending on a turn or a pause.

This module is a filter, not a repair: nothing is denoised, because a denoiser
alters the recording (see utils/noise_map.py). What noise cannot be excluded is
scored, so a cleaner stretch outranks a dirtier one.

Where the plan says 300 seconds and a duration score peaking at 90, this uses a
240 second ceiling and a plateau over 90-180, because the target here is one to
four minutes. The plan's total of 100 is kept: the SSLAM cleanliness term is
paid for by lowering turn-taking, duration and interaction (and dropping the
five points for transcript quality). Weights are configurable.
"""

import bisect
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from algorithms.asr.hallucination import is_hallucination
from utils.excise import TimelineMap
from utils.noise_map import dominant_kind

NORMAL, BACKCHANNEL, INVALID = "NORMAL", "BACKCHANNEL", "INVALID"

# Listener noises the plan (section 2) keeps but does not count as a change of
# turn. Compared after lower-casing and dropping punctuation.
BACKCHANNELS = frozenset({
    "ừ", "ừm", "ờ", "ờm", "à", "ạ", "dạ", "vâng", "dạ vâng", "vâng ạ",
    "đúng rồi", "đúng vậy", "thế à", "vậy à", "thế ạ", "vậy hả", "ừ đúng rồi",
})

BACKCHANNEL_MAX_SECONDS = 1.0
TOO_SHORT_SECONDS = 0.3

DEFAULT_WEIGHTS = {
    "turn_taking": 15.0,
    "duration": 10.0,
    "balance": 15.0,
    "interaction": 10.0,
    "continuity": 10.0,
    "speaker_correctness": 20.0,
    "cleanliness": 20.0,
}

TIERS = ((85.0, "S"), (70.0, "A"), (55.0, "B"), (40.0, "C"))


@dataclass
class ConversationSelectionConfig:
    """Every knob used to select a conversation excerpt."""

    # -- shape of one exported conversation item
    min_seconds: float = 60.0
    max_seconds: float = 240.0
    plateau_min: float = 90.0            # duration scores 1.0 between these two
    plateau_max: float = 180.0
    max_silence: float = 3.0             # a longer pause ends the block
    max_monologue: float = 45.0          # one person without an answer this long ends it; 0 = no limit
    boundary_gap: float = 0.6            # a pause this long is a place to start or stop
    min_turns_each: int = 2              # each of the two must have real turns
    ends_per_start: int = 3
    pad_seconds: float = 0.2

    # -- what to keep
    min_score: float = 55.0
    min_semantic: int = 4                # the model's self-contained score, 1-5
    require_semantic: bool = True        # no verdict is not a yes; off accepts un-judged items
    # A selected training item must carry timestamps for the final, refined
    # text, not stale Whisper words from before the text was edited.
    require_word_alignment: bool = False
    max_candidates: int = 24             # sent to the model per file
    shortlist_overlap: float = 0.5
    # How the model is asked. With thinking on a model that has the mode (Qwen3)
    # reasons before it answers, and the reasoning comes out of max_new_tokens;
    # the service never lets the budget fall below what reasoning needs.
    thinking: bool = False
    max_new_tokens: int = 256

    # -- the SSLAM signals. Provisional: the levels below come from
    # noise_map.NOTICEABLE, measured on three indoor recordings, and have not
    # seen outdoor audio. Calibrate on a real batch before trusting them.
    require_noise: bool = True           # unmeasured is not clean
    noisy_run_seconds: float = 2.0
    noise_speech_max: float = 0.10
    noise_env_max: float = 0.15
    noise_room_max: float = 0.10
    noise_max: float = 0.15              # ceiling on the combined p90
    noisy_share_max: float = 0.20        # share of frames over, where cleanliness hits 0
    music_patched_max: float = 0.15      # share of the item whose music bed was replaced

    # -- where an excerpt is filed (used by the service)
    # An excerpt in which the two spoke at once is the hardest thing to get and the
    # most useful, so it is filed apart from the rest: in `overlap_good` when the
    # separator pulled the two voices apart cleanly (each ear holds one voice), in
    # `overlap_bad` when it did not. Off, everything is filed by score alone.
    overlap_first: bool = True
    overlap_min_seconds: float = 0.3     # how much overlap makes an excerpt "with overlap"
    overlap_good_min_similarity: float = 0.6   # the weakest separated span still counts as clean
    overlap_good_min_coverage: float = 0.9     # share of the overlap that must have been separated

    # -- writing the audio (used by the service)
    fade_ms: float = 50.0
    zero_cross_ms: float = 5.0
    # Beside the mono recording, a two-channel file of the same length: the first
    # speaker on the left, the second on the right, so each can be heard in one ear.
    stereo: bool = True
    gate_margin_ms: float = 30.0         # a channel stays open this far around its turn
    gate_fade_ms: float = 10.0           # ramp at each opening and closing, so it cannot click

    weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        unknown = set(self.weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown conversation-export weight(s): {sorted(unknown)}")
        if self.min_seconds > self.max_seconds:
            raise ValueError("min_seconds is above max_seconds")

    @property
    def resolved_weights(self) -> Dict[str, float]:
        return {**DEFAULT_WEIGHTS, **{k: float(v) for k, v in self.weights.items()}}

    @property
    def limits(self) -> Dict[str, float]:
        """The level at which each noise group is worth noticing."""
        return {"noise_speech": self.noise_speech_max,
                "noise_env": self.noise_env_max,
                "noise_room": self.noise_room_max}

    @classmethod
    def from_settings(cls, settings: Optional[dict]) -> "ConversationSelectionConfig":
        return cls(**dict(settings or {}))


def tier_of(score: float) -> str:
    for floor, name in TIERS:
        if score >= floor:
            return name
    return "Reject"


# ---------------------------------------------------------------------------
# One segment
# ---------------------------------------------------------------------------

def _fold(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def classify(seg) -> Tuple[str, Optional[str]]:
    """(state, reason). Plan section 2: NORMAL, BACKCHANNEL, or INVALID with why."""
    text = getattr(seg, "text", "") or ""
    folded = _fold(text)
    duration = float(seg.end) - float(seg.start)
    if not folded:
        return INVALID, "no_speech"
    if is_hallucination(text, duration):
        return INVALID, "hallucination"
    if duration < BACKCHANNEL_MAX_SECONDS and folded in BACKCHANNELS:
        return BACKCHANNEL, None
    if duration < TOO_SHORT_SECONDS:
        return INVALID, "too_short"
    return NORMAL, None


def _union_seconds(intervals) -> float:
    total, edge = 0.0, None
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if edge is None or start > edge:
            total += end - start
            edge = end
        elif end > edge:
            total += end - edge
            edge = end
    return total


def _unseparated_seconds(seg) -> float:
    spans = getattr(seg, "unseparated", None)
    if not spans:
        return 0.0
    duration = float(seg.end) - float(seg.start)
    total = 0.0
    for span in spans:
        try:
            total += max(0.0, float(span["end"]) - float(span["start"]))
        except (KeyError, TypeError, ValueError):
            return duration     # cannot read it; the whole segment is in doubt
    return min(total, duration)


def has_complete_word_alignment(seg) -> bool:
    """Every space-delimited final-text word has a valid monotonic timestamp."""
    expected = len((getattr(seg, "text", "") or "").split())
    words = getattr(seg, "words", None) or []
    if expected == 0 or len(words) != expected:
        return False
    lower, upper = float(seg.start), float(seg.end)
    previous = lower
    for word in words:
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError):
            return False
        if start < lower - 0.001 or end > upper + 0.001 or end < start:
            return False
        if start < previous - 0.001:
            return False
        previous = start
    return True


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    first: int                      # positions in the finder's segment list, inclusive
    last: int
    start: float                    # the turns themselves, in the cut timeline
    end: float
    pad_start: float                # what to cut: the same, padded where that is safe
    pad_end: float
    speakers: Tuple[str, str]       # in order of first appearance
    metrics: dict
    noise: dict
    music_patched_share: Optional[float]
    components: dict
    score: float
    tier: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def overlap_fraction(a: Candidate, b: Candidate) -> float:
    shared = min(a.end, b.end) - max(a.start, b.start)
    if shared <= 0:
        return 0.0
    return shared / max(1e-9, min(a.duration, b.duration))


def shortlist(candidates: List[Candidate], k: int,
              max_overlap: float = 0.5) -> List[Candidate]:
    """The `k` best candidates that are not mostly the same stretch.

    A block yields many windows sliding over it, so the top of a plain ranking is
    one good stretch repeated with slightly different edges. Skipping any that
    overlap an already-chosen one too much spends the model's attention on
    different parts of the recording.
    """
    chosen: List[Candidate] = []
    for cand in sorted(candidates, key=lambda c: (-c.score, c.start)):
        if all(overlap_fraction(cand, other) <= max_overlap for other in chosen):
            chosen.append(cand)
            if len(chosen) >= k:
                break
    return chosen


def pick_non_overlapping(candidates: List[Candidate]) -> List[Candidate]:
    """The set of non-overlapping candidates with the highest total score.

    Weighted interval scheduling. Two exported items may touch but not overlap,
    so no stretch of the recording is duplicated.
    """
    ordered = sorted(candidates, key=lambda c: (c.end, c.start))
    if not ordered:
        return []
    ends = [c.end for c in ordered]
    best = [0.0] * (len(ordered) + 1)
    take = [False] * len(ordered)
    prev = [0] * len(ordered)
    for i, cand in enumerate(ordered):
        prev[i] = bisect.bisect_right(ends, cand.start, 0, i)
        with_it = cand.score + best[prev[i]]
        if with_it > best[i]:
            best[i + 1], take[i] = with_it, True
        else:
            best[i + 1] = best[i]
    picked, i = [], len(ordered)
    while i > 0:
        if take[i - 1]:
            picked.append(ordered[i - 1])
            i = prev[i - 1]
        else:
            i -= 1
    return sorted(picked, key=lambda c: c.start)


# ---------------------------------------------------------------------------
# The finder
# ---------------------------------------------------------------------------

class ConversationSelectionFinder:
    """Blocks, candidates and their scores for one recording.

    `segments` are the relabelled transcript segments in the cut timeline;
    `timeline` is the TimelineMap that produced it; `noise` the NoiseTrack (in
    the ORIGINAL timeline); `music_map` the MusicMap already expressed in the
    cut timeline (or None if music was not analysed).
    """

    def __init__(self, segments, timeline, noise, music_map, config: ConversationSelectionConfig):
        self.cfg = config
        self.timeline = timeline or TimelineMap()   # nothing cut: an empty map
        self.noise = noise
        self.music_map = music_map
        self.segs = sorted(segments, key=lambda s: (float(s.start), float(s.end)))
        self.stats = {"segments": len(self.segs), "invalid": Counter(),
                      "breaks": Counter(), "blocks": 0, "blocks_two_speakers": 0,
                      "blocks_other_speaker_count": 0, "blocks_short": 0,
                      "candidates_generated": 0, "candidates_rejected": Counter()}

        self.noise_measured = bool(noise) if noise is not None else False
        if config.require_noise and not self.noise_measured:
            self.stats["candidates_rejected"]["noise_not_measured"] += 1

        self._sustained = (noise.sustained_spans(config.limits, config.noisy_run_seconds)
                           if self.noise_measured else [])
        self._sus_starts = [a for a, _ in self._sustained]
        self._sus_ends = [b for _, b in self._sustained]

        self.state, self.reason = [], []
        for seg in self.segs:
            state, reason = classify(seg)
            self.state.append(state)
            self.reason.append(reason)
            if state == INVALID:
                self.stats["invalid"][reason] += 1
        self.block_of = [-1] * len(self.segs)
        self.blocks: List[Tuple[int, int]] = []
        self._build_blocks()

    # -- noise ----------------------------------------------------------------
    def _originals(self, start, end):
        return self.timeline.spans_to_original(start, end)

    def _in_lasting_noise(self, start, end) -> bool:
        """Whether [start, end) of the cut timeline touches a lasting noisy run."""
        if not self._sustained:
            return False
        for a, b in self._originals(start, end):
            k = bisect.bisect_right(self._sus_ends, a)
            if k < len(self._sustained) and self._sus_starts[k] < b:
                return True
        return False

    # -- blocks ---------------------------------------------------------------
    def _ineligible(self, i) -> Optional[str]:
        seg = self.segs[i]
        if self.state[i] == INVALID:
            return f"invalid_{self.reason[i]}"
        if self.cfg.require_word_alignment and not has_complete_word_alignment(seg):
            return "word_alignment_missing"
        if self.timeline.crosses_cut(float(seg.start), float(seg.end)):
            return "seam_inside_segment"
        if self._in_lasting_noise(float(seg.start), float(seg.end)):
            return "lasting_noise"
        return None

    def _build_blocks(self):
        segs, cfg = self.segs, self.cfg
        current: List[int] = []
        speakers: set = set()
        last_normal, run_start = None, None

        def close():
            nonlocal current, speakers, last_normal, run_start
            if current:
                self.blocks.append((current[0], current[-1]))
            current, speakers, last_normal, run_start = [], set(), None, None

        for i, seg in enumerate(segs):
            why = self._ineligible(i)
            if why:
                self.stats["breaks"][why] += 1
                close()
                continue
            if current:
                prev = segs[current[-1]]
                gap = float(seg.start) - float(prev.end)
                brk = None
                if self.timeline.cut_between(float(prev.end), float(seg.start)):
                    brk = "seam"
                elif gap > cfg.max_silence:
                    brk = "silence"
                elif seg.speaker not in speakers and len(speakers) >= 2:
                    brk = "third_speaker"
                elif (cfg.max_monologue > 0 and self.state[i] == NORMAL
                      and seg.speaker == last_normal
                      and float(seg.end) - run_start > cfg.max_monologue):
                    brk = "monologue"
                elif self._in_lasting_noise(float(prev.end), float(seg.start)) and gap > 0:
                    brk = "lasting_noise_between"
                if brk:
                    self.stats["breaks"][brk] += 1
                    close()
            current.append(i)
            speakers.add(seg.speaker)
            if self.state[i] == NORMAL and seg.speaker != last_normal:
                last_normal, run_start = seg.speaker, float(seg.start)
        close()

        for number, (first, last) in enumerate(self.blocks):
            for pos in range(first, last + 1):
                self.block_of[pos] = number
            self.stats["blocks"] += 1
            kinds = {segs[p].speaker for p in range(first, last + 1)}
            if len(kinds) == 2:
                self.stats["blocks_two_speakers"] += 1
            else:
                self.stats["blocks_other_speaker_count"] += 1
            if self._span(first, last)[1] - float(segs[first].start) < cfg.min_seconds:
                self.stats["blocks_short"] += 1

    def _span(self, i, j) -> Tuple[float, float]:
        return (float(self.segs[i].start),
                max(float(s.end) for s in self.segs[i:j + 1]))

    # -- padding ----------------------------------------------------------------
    def _padded(self, i, j) -> Tuple[float, float]:
        """The span to cut: the turns plus a little air, where that is safe.

        Padding never steps into a neighbouring turn, never crosses a join, and
        never reaches into a lasting noisy run -- the point of the pad is the
        breath before a word, not more of what surrounds it.
        """
        start, end = self._span(i, j)
        pad = self.cfg.pad_seconds
        lo, hi = start, end
        if pad > 0:
            want = start - pad
            if i > 0:
                want = max(want, float(self.segs[i - 1].end))
            want = max(0.0, want)
            if (want < start and not self.timeline.cut_between(want, start)
                    and not self._in_lasting_noise(want, start)):
                lo = want
            want = end + pad
            if j + 1 < len(self.segs):
                want = min(want, float(self.segs[j + 1].start))
            if (want > end and not self.timeline.cut_between(end, want)
                    and not self._in_lasting_noise(end, want)):
                hi = want
        return lo, hi

    # -- metrics -----------------------------------------------------------------
    def _metrics(self, i, j, start, end) -> Tuple[dict, List[str]]:
        """The plan's section-4 numbers for positions i..j, and the speakers in order."""
        segs = self.segs[i:j + 1]
        duration = end - start
        speakers: List[str] = []
        for s in segs:
            if s.speaker not in speakers:
                speakers.append(s.speaker)
        speaking = {spk: 0.0 for spk in speakers}
        for s in segs:
            speaking[s.speaker] += max(0.0, float(s.end) - float(s.start))

        runs: List[List] = []           # [speaker, first_start, last_end]
        for pos in range(i, j + 1):
            if self.state[pos] != NORMAL:
                continue
            s = self.segs[pos]
            if runs and runs[-1][0] == s.speaker:
                runs[-1][2] = max(runs[-1][2], float(s.end))
            else:
                runs.append([s.speaker, float(s.start), float(s.end)])
        turns_each = Counter(r[0] for r in runs)

        latencies = [runs[k + 1][1] - runs[k][2] for k in range(len(runs) - 1)]
        union = _union_seconds([(float(s.start), float(s.end)) for s in segs])
        words = sum(len((s.text or "").split()) for s in segs)
        big, small = max(speaking.values()), min(speaking.values())
        unseparated = sum(_unseparated_seconds(s) for s in segs)
        gaps = [float(segs[k + 1].start) - float(segs[k].end) for k in range(len(segs) - 1)]

        return {
            "duration": round(duration, 3),
            "turn_count": len(runs),
            "turns_each": dict(turns_each),
            "speaker_count": len(speakers),
            "speaker_balance": round(small / big, 4) if big > 0 else 0.0,
            "switch_count": max(0, len(runs) - 1),
            "max_monologue": round(max((r[2] - r[1] for r in runs), default=0.0), 3),
            "silence_ratio": round(max(0.0, 1.0 - union / duration), 4) if duration > 0 else 1.0,
            "overlap_count": sum(1 for g in gaps if g < -0.05),
            "separated_count": sum(1 for s in segs if getattr(s, "bss", False)),
            "backchannel_count": sum(1 for pos in range(i, j + 1)
                                     if self.state[pos] == BACKCHANNEL),
            "pacing_wpm": round(words / (union / 60.0), 1) if union > 0 else 0.0,
            "turn_latency_avg": (round(sum(latencies) / len(latencies), 3)
                                 if latencies else None),
            "unseparated_share": round(unseparated / duration, 4) if duration > 0 else 0.0,
        }, speakers

    # -- noise and music of one candidate -------------------------------------------
    def _noise_of(self, lo, hi) -> Optional[dict]:
        if not self.noise_measured:
            return None
        spans = self._originals(lo, hi)
        combined = self.noise.score_spans(spans)
        if combined is None:
            return None
        by_kind = self.noise.breakdown(spans)
        limits = self.cfg.limits
        return {
            "combined_p90": combined,
            "by_kind": by_kind,
            "noisy_frame_share": self.noise.share_over(spans, limits),
            "longest_noisy_run_seconds": self.noise.longest_run_over(spans, limits),
            "dominant_kind": dominant_kind(by_kind),
        }

    def _patched_share(self, lo, hi) -> Optional[float]:
        if self.music_map is None or hi <= lo:
            return None
        from utils.music_map import MUSIC
        clean = sum(b - a for a, b in self.music_map.clean_parts(lo, hi, MUSIC))
        return round(max(0.0, 1.0 - clean / (hi - lo)), 4)

    # -- scoring -------------------------------------------------------------------------
    def _components(self, metrics, noise, patched) -> Dict[str, float]:
        cfg = self.cfg
        latency = metrics["turn_latency_avg"]
        if latency is None:
            turn_taking = 0.0
        elif 0.1 <= latency <= 1.0:
            turn_taking = 1.0
        elif latency > 1.0:
            turn_taking = max(0.0, 1.0 - (latency - 1.0) / 2.0)
        else:
            turn_taking = max(0.0, 1.0 - (0.1 - latency) / 0.6)

        d = metrics["duration"]
        if cfg.plateau_min <= d <= cfg.plateau_max:
            duration = 1.0
        elif d < cfg.plateau_min:
            span = max(1e-9, cfg.plateau_min - cfg.min_seconds)
            duration = max(0.0, 0.7 + 0.3 * (d - cfg.min_seconds) / span)
        else:
            span = max(1e-9, cfg.max_seconds - cfg.plateau_max)
            duration = max(0.0, 1.0 - 0.3 * (d - cfg.plateau_max) / span)

        if noise is None:
            cleanliness = 0.0           # nobody measured it: not clean
        else:
            parts = [1.0 - min(1.0, noise["combined_p90"] / cfg.noise_max),
                     1.0 - min(1.0, (noise["noisy_frame_share"] or 0.0) / cfg.noisy_share_max)]
            if patched is not None:
                parts.append(1.0 - min(1.0, patched / cfg.music_patched_max))
            cleanliness = sum(parts) / len(parts)

        return {
            "turn_taking": turn_taking,
            "duration": duration,
            "balance": metrics["speaker_balance"],
            "interaction": min(1.0, (metrics["switch_count"] * 2
                                     + metrics["backchannel_count"]) / 15.0),
            "continuity": max(0.0, 1.0 - metrics["silence_ratio"] / 0.5),
            "speaker_correctness": max(0.0, 1.0 - 10.0 * metrics["unseparated_share"]),
            "cleanliness": cleanliness,
        }

    def _score(self, components) -> float:
        weights = self.cfg.resolved_weights
        total = sum(weights.values())
        if total <= 0:
            return 0.0
        return round(100.0 * sum(weights[k] * components[k] for k in weights) / total, 1)

    # -- evaluating one window -------------------------------------------------------------
    def evaluate(self, i: int, j: int) -> Tuple[Optional[Candidate], Optional[str]]:
        """(candidate, None) if positions i..j make an export item, else (None, why).

        The same checks whether the window came from the scan or from the model
        trimming a candidate, so a trim cannot smuggle in what the scan would
        have refused.
        """
        cfg = self.cfg
        if not (0 <= i <= j < len(self.segs)):
            return None, "out_of_range"
        block = self.block_of[i]
        if block < 0 or block != self.block_of[j]:
            return None, "not_one_block"
        if cfg.require_noise and not self.noise_measured:
            return None, "noise_not_measured"

        start, end = self._span(i, j)
        if not cfg.min_seconds <= end - start <= cfg.max_seconds:
            return None, "duration"
        metrics, speakers = self._metrics(i, j, start, end)
        if len(speakers) != 2:
            return None, "not_two_speakers"
        if min(metrics["turns_each"].get(spk, 0) for spk in speakers) < cfg.min_turns_each:
            return None, "too_few_turns"

        lo, hi = self._padded(i, j)
        noise = self._noise_of(lo, hi)
        if cfg.require_noise and noise is None:
            return None, "noise_not_measured"
        if noise is not None:
            for kind, limit in cfg.limits.items():
                value = noise["by_kind"].get(kind)
                if value is not None and value > limit:
                    return None, kind
            if noise["combined_p90"] > cfg.noise_max:
                return None, "noise_combined"
            run = noise["longest_noisy_run_seconds"]
            if run is not None and run >= cfg.noisy_run_seconds:
                return None, "noisy_run"
        patched = self._patched_share(lo, hi)
        if patched is not None and patched > cfg.music_patched_max:
            return None, "music_patched"

        components = self._components(metrics, noise, patched)
        score = self._score(components)
        if score < cfg.min_score:
            return None, "low_score"
        return Candidate(
            first=i, last=j, start=start, end=end, pad_start=lo, pad_end=hi,
            speakers=(speakers[0], speakers[1]), metrics=metrics,
            noise=noise or {}, music_patched_share=patched,
            components={k: round(v, 4) for k, v in components.items()},
            score=score, tier=tier_of(score)), None

    # -- the scan -----------------------------------------------------------------------------
    def _is_start(self, pos, first) -> bool:
        if pos == first:
            return True
        prev, cur = self.segs[pos - 1], self.segs[pos]
        return (cur.speaker != prev.speaker
                or float(cur.start) - float(prev.end) >= self.cfg.boundary_gap)

    def _is_stop(self, pos, last) -> bool:
        if pos == last:
            return True
        cur, nxt = self.segs[pos], self.segs[pos + 1]
        return (nxt.speaker != cur.speaker
                or float(nxt.start) - float(cur.end) >= self.cfg.boundary_gap)

    def candidates(self) -> List[Candidate]:
        """Every window that passes the gates and clears `min_score`."""
        cfg = self.cfg
        if cfg.require_noise and not self.noise_measured:
            # Not checked is not clean; scanning would only reject every window
            # for the same reason.
            return []
        ideal = (cfg.plateau_min + cfg.plateau_max) / 2.0
        found: List[Candidate] = []
        rejected = self.stats["candidates_rejected"]
        for first, last in self.blocks:
            if len({self.segs[p].speaker for p in range(first, last + 1)}) != 2:
                continue
            for i in range(first, last + 1):
                if not self._is_start(i, first):
                    continue
                valid, edge = [], float(self.segs[i].start)
                for j in range(i, last + 1):
                    edge = max(edge, float(self.segs[j].end))
                    span = edge - float(self.segs[i].start)
                    if span > cfg.max_seconds:
                        break
                    if span >= cfg.min_seconds and self._is_stop(j, last):
                        valid.append((j, span))
                if not valid:
                    continue
                # The end nearest the ideal length first, then the extremes, so a
                # small `ends_per_start` keeps the most useful ones.
                picks: List[int] = []
                for item in (min(valid, key=lambda it: abs(it[1] - ideal)),
                             valid[-1], valid[0]):
                    if item[0] not in picks:
                        picks.append(item[0])
                for j in picks[:max(1, cfg.ends_per_start)]:
                    self.stats["candidates_generated"] += 1
                    cand, why = self.evaluate(i, j)
                    if cand is None:
                        rejected[why] += 1
                    else:
                        found.append(cand)
        return found

    def report(self) -> dict:
        """The counts that say why a recording produced few items, or none."""
        out = dict(self.stats)
        for key in ("invalid", "breaks", "candidates_rejected"):
            out[key] = dict(self.stats[key])
        out["noise_measured"] = self.noise_measured
        out["lasting_noise_spans"] = len(self._sustained)
        return out
