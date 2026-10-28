"""Continuous two-channel audio for the full-duplex export: a user channel that keeps the room, an assistant channel that is clean.

The conversation export used to lay each speaker's separated segments on a track of digital zeros, so the user
channel stopped dead between its sentences. This module builds the two channels the way the dataset wants them:

  user       the real room all the time. Where nobody but the user (or nobody at all) is heard the recording itself is
             used (mode "raw_mask"); where the other speaker is heard (or the separation failed) the user's separated
             voice is laid on a "bed" made from stretches of the same recording in which nobody speaks. Never the
             mixture minus the other voice: that leaves a ghost of it.
  assistant  as clean as possible over the whole channel: no bed, digital silence between sentences, a soft mask
             around each sentence after the filter chain, an expander under the floor, levels matched to what went
             into the chain, short backchannels the filter erased restored.

All times are seconds on one clock named by each argument (`abs` = the pipeline's clock, `start_s` = the excerpt's
first sample on that clock). Arrays are mono float32 at `sr`. Nothing here raises for bad audio: callers rely on
`build_bed` always returning a bed (see the tiers) and on every function being total over its inputs.

Modes (AmbienceParams.mode):
  "none"       the old behaviour: tracks of zeros between the segments (this module is not used)
  "raw_mask"   M1: the recording where it is safe, voice + bed where it is not
  "bed_only"   M2: the bed everywhere with the user's voice on it (the voice should be denoised hard by the caller)
"""
from dataclasses import dataclass, field, fields
from typing import List, Optional, Sequence, Tuple

import numpy as np

Span = Tuple[float, float]

MODES = ("none", "raw_mask", "bed_only")
TIERS = ("local", "file", "synthetic", "pink")


@dataclass
class AmbienceParams:
    mode: str = "none"
    # -- quiet gaps (stretches of the recording in which nobody speaks)
    gap_margin_after_s: float = 0.25     # kept clear after anyone's speech ends (reverb tail, breath)
    gap_margin_before_s: float = 0.10    # kept clear before anyone's speech starts
    gap_min_s: float = 0.40              # shorter gaps are not used
    prefer_radius_s: float = 60.0        # gaps within this distance of the excerpt are the "local" tier
    energy_reject_db: float = 12.0       # a gap whose p90 frame level is this far above the recording's floor is dropped
    # -- the bed
    xfade_ms: float = 300.0              # equal-power crossfade between bed pieces (linear when the room hums)
    floor_window_s: float = 2.0          # window of the noise-floor estimate the bed follows
    pink_dbfs: float = -65.0             # last-resort noise level
    seed: int = 0                        # the bed is a pure function of (recording, excerpt, seed)
    # -- the safe mask (where the recording itself may be used for the user channel)
    safe_dilate_ms: float = 150.0        # around the other speaker's speech
    reverb_tail_max_ms: float = 400.0    # the measured tail is capped at this
    other_margin_ms: float = 100.0       # around any third speaker
    splice_fade_ms: float = 30.0         # equal-power crossfade at every raw<->bed boundary
    min_overlap_s: float = 0.10          # overlaps shorter than this are cut at their middle (fade 10 ms)
    # -- assistant post-processing
    island_pad_ms: float = 30.0          # the soft mask opens this far around each sentence
    island_fade_ms: float = 10.0
    expander_floor_dbfs: float = -62.0   # below this the assistant channel is exactly zero
    level_match_db: float = 6.0          # a sentence may be this far from its pre-chain level, no more
    restore_max_s: float = 1.0           # a sentence this short that the chain erased comes back from before the chain
    restore_ratio: float = 0.25          # "erased": RMS after the chain under this share of the RMS before it

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "AmbienceParams":
        """Unknown keys are ignored (profiles may carry notes); a bad mode raises ValueError here, not per excerpt."""
        known = {f.name for f in fields(cls)}
        params = cls(**{k: v for k, v in (data or {}).items() if k in known})
        if params.mode not in MODES:
            raise ValueError(f"user_ambience.mode must be one of {MODES}, got {params.mode!r}")
        return params

    @property
    def enabled(self) -> bool:
        return self.mode != "none"


@dataclass
class BedResult:
    audio: np.ndarray                    # float32, exactly the length asked for, never digital silence
    tier: str                            # one of TIERS: where the room came from
    pieces: int = 0                      # how many gap pieces were joined
    repeats: int = 0                     # how many times a gap sample range was used twice (0 is the goal)
    notes: List[str] = field(default_factory=list)


def find_quiet_gaps(speech_spans: Sequence[Span], lo_s: float, hi_s: float, params: AmbienceParams,
                    waveform: Optional[np.ndarray] = None, sr: Optional[int] = None) -> List[Span]:
    """Stretches inside [lo_s, hi_s] (abs clock) in which nobody speaks, cleared by the margins.

    `speech_spans` is the speech of EVERY speaker (all diarized people, not just the two of the excerpt), in any order,
    possibly overlapping. A gap is what is left between their union, shrunk by gap_margin_after_s after each span and
    gap_margin_before_s before the next, and kept when it is at least gap_min_s long. When `waveform` and `sr` are
    given, a gap whose p90 20 ms frame level is more than energy_reject_db above the median level of the quiet frames
    of the whole window is dropped (a cough, a door, a voice the diarization missed).
    """
    raise NotImplementedError


def noise_floor_track(waveform: np.ndarray, sr: int, speech_spans: Sequence[Span], start_s: float, end_s: float,
                      params: AmbienceParams) -> np.ndarray:
    """The level of the room (linear RMS) for every sample of [start_s, end_s) (abs clock).

    The median RMS of the 20 ms frames outside `speech_spans`, per floor_window_s window, interpolated between window
    centres and smoothed; windows without a quiet frame take the nearest window's value, and a recording with none
    at all takes its 5th-percentile frame level. Length = round((end_s - start_s) * sr). Never zero (floored at 1e-6).
    """
    raise NotImplementedError


def build_bed(waveform: np.ndarray, sr: int, gaps: Sequence[Span], speech_spans: Sequence[Span],
              start_s: float, end_s: float, params: AmbienceParams, focus_s: Optional[float] = None) -> BedResult:
    """A continuous room tone for [start_s, end_s) (abs clock), length round((end_s - start_s) * sr).

    Tiers, first that works:
      "local"      pieces of gaps within prefer_radius_s of `focus_s` (default: the middle of the excerpt)
      "file"       pieces of any gap of the recording
      "synthetic"  noise shaped by the spectrum of the quietest frames (minimum statistics), when no gap is long enough
      "pink"       pink noise at pink_dbfs, when nothing at all can be measured
    Pieces are random slices of the gaps (seeded by params.seed and start_s), joined by equal-power crossfades of
    xfade_ms (linear when the bed has a strong tonal hum), no sample range used twice while unused range is left, and
    each piece scaled to follow noise_floor_track. The result is never digital silence and never clips.
    """
    raise NotImplementedError


def safe_mask(n: int, sr: int, start_s: float, other_spans: Sequence[Span], third_spans: Sequence[Span],
              bad_spans: Sequence[Span], overlap_spans: Sequence[Span], params: AmbienceParams,
              tail_ms: Optional[float] = None) -> np.ndarray:
    """Boolean mask of length n: True where the recording itself may serve as the user channel.

    False (unsafe) within safe_dilate_ms (+ the measured reverb tail `tail_ms`, capped at reverb_tail_max_ms, after)
    of `other_spans` (the other speaker's speech), within other_margin_ms of `third_spans` (any third speaker), over
    `bad_spans` (failed separation) and over `overlap_spans` (both speak). All spans on the abs clock; `start_s` is
    the abs time of sample 0. Overlaps shorter than min_overlap_s are split at their middle: the first half belongs to
    whoever spoke first, so only the unsafe part is the longer one (callers pass them in `overlap_spans` as they are).
    """
    raise NotImplementedError


def compose_user_channel(raw: np.ndarray, user_track: np.ndarray, bed: np.ndarray, safe: np.ndarray, sr: int,
                         params: AmbienceParams) -> np.ndarray:
    """The user channel, len(raw) samples.

    raw_mask:  raw where `safe`; user_track + bed where not safe. Every raw<->bed boundary is an equal-power
               crossfade of splice_fade_ms, centred on the boundary. The bed is NOT added to raw (that would be two rooms).
    bed_only:  bed + user_track everywhere.
    `user_track` is the user's separated voice on the excerpt clock, zeros where the user is silent. The result never
    exceeds full scale (a gentle limiter, not a clip) and has no digital silence longer than 10 ms.
    """
    raise NotImplementedError


def postprocess_assistant(clean: np.ndarray, before: np.ndarray, sr: int, params: AmbienceParams,
                          islands: Optional[Sequence[Span]] = None) -> Tuple[np.ndarray, dict]:
    """The assistant channel after the filter chain, made cleaner.

    `before` is the channel that went into the chain (digital zeros between sentences), `clean` what came out.
    `islands` are the sentences' spans in seconds on this array's clock; None = the runs of non-zero samples in
    `before`, joined when closer than 150 ms. For each island: open a soft mask of island_pad_ms around it with
    island_fade_ms fades and zero everything outside; apply an expander that sends samples under expander_floor_dbfs
    to exactly 0 (smooth gain, no chatter); bring the island's RMS to within level_match_db of its RMS in `before`; and
    when the island is shorter than restore_max_s and its RMS after the chain is under restore_ratio of its RMS
    before, take the island from `before` instead (a backchannel the filter erased) and count it.
    Returns (audio of len(clean), {"islands": n, "restored": n, "level_adjusted": n, "zeroed_fraction": float}).
    """
    raise NotImplementedError


def quality_report(user: np.ndarray, assistant: np.ndarray, sr: int, user_spans: Sequence[Span],
                   assistant_spans: Sequence[Span], bed_tier: Optional[str] = None) -> dict:
    """Numbers to audit one exported pair (spans on the arrays' clock).

    {"user_digital_silence_s": longest run of exact zeros in the user channel outside user_spans (goal 0),
     "assistant_floor_dbfs": the 10th percentile frame level of the assistant outside assistant_spans,
     "user_rms_jump_db": the largest RMS step (50 ms before vs after) at a point where the user channel switches
                          source, as an upper bound taken over all points where the level changes by > 6 dB in 50 ms,
     "bed_tier": bed_tier}
    """
    raise NotImplementedError
