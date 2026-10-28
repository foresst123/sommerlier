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


_FRAME_S = 0.02
_FLOOR = 1e-6
_EPS = 1e-9


def _as_f64(x) -> np.ndarray:
    if x is None:
        return np.zeros(0)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _as_1d(x) -> np.ndarray:
    """The recording as given (no dtype change, no copy for an array): a long file is only ever sliced."""
    if x is None:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(x).reshape(-1)


def _fit(a, n: int) -> np.ndarray:
    a = _as_f64(a)
    if a.size >= n:
        return a[:n]
    return np.concatenate([a, np.zeros(n - a.size)])


def _odd(k: float) -> int:
    k = max(1, int(round(k)))
    return k if k % 2 else k + 1


_CHUNK = 1 << 20


def _rms(x: np.ndarray) -> float:
    """RMS with float64 accumulation, in blocks (no float64 copy of the whole input)."""
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return 0.0
    acc = 0.0
    for c in range(0, x.size, _CHUNK):
        b = np.asarray(x[c:c + _CHUNK], dtype=np.float64)
        acc += float(np.dot(b, b))
    return float(np.sqrt(acc / x.size))


def _clean_spans(spans, lo: Optional[float] = None, hi: Optional[float] = None) -> List[Span]:
    out: List[Span] = []
    for s in (spans if spans is not None else ()):
        try:
            a, b = float(s[0]), float(s[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if lo is not None:
            a = max(a, lo)
        if hi is not None:
            b = min(b, hi)
        if b > a:
            out.append((a, b))
    return out


def _union(spans: List[Span]) -> List[Span]:
    merged: List[Span] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _in_spans(t: np.ndarray, merged: List[Span]) -> np.ndarray:
    """Membership of times t in sorted, non-overlapping merged spans."""
    t = np.asarray(t, dtype=np.float64)
    if not merged:
        return np.zeros(t.shape, dtype=bool)
    starts = np.array([a for a, _ in merged])
    ends = np.array([b for _, b in merged])
    j = np.searchsorted(starts, t, side="right") - 1
    return (j >= 0) & (t < ends[np.clip(j, 0, None)])


def _frame_len(sr: int) -> int:
    return max(1, int(round(_FRAME_S * sr)))


_FRAME_CHUNK = 4096  # frames per block: bounded temporaries however long the window


def _frame_rms(x: np.ndarray, fl: int) -> np.ndarray:
    """RMS of consecutive frames of length fl, computed in float32 blocks with float64 accumulation."""
    nf = x.size // fl
    out = np.zeros(nf)
    for c in range(0, nf, _FRAME_CHUNK):
        m = min(_FRAME_CHUNK, nf - c)
        f = np.asarray(x[c * fl:(c + m) * fl], dtype=np.float32).reshape(m, fl)
        out[c:c + m] = np.sqrt(np.mean(f * f, axis=1, dtype=np.float64))
    return out


def _db(level):
    return 20.0 * np.log10(np.maximum(level, _EPS))


def _fft_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    size = a.size + b.size - 1
    nfft = 1 << max(0, size - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)[:size]


def _conv_centred(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Same-length convolution with an odd kernel; the edges are held, not faded."""
    h = k.size // 2
    return _fft_conv(np.pad(x, (h, h), mode="edge"), k)[2 * h: 2 * h + x.size]


def _smooth(x: np.ndarray, width: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    w = _odd(width)
    if x.size == 0 or w == 1:
        return x.copy()
    return _conv_centred(x, np.full(w, 1.0 / w))


def _hann_kernel(length: int) -> np.ndarray:
    k = np.hanning(length + 2)[1:-1]
    return k / k.sum()


def _mark(arr: np.ndarray, sr: int, origin: float, a: float, b: float) -> None:
    i0 = int(np.floor((a - origin) * sr + 1e-6))
    i1 = int(np.ceil((b - origin) * sr - 1e-6))
    i0, i1 = max(i0, 0), min(i1, arr.size)
    if i1 > i0:
        arr[i0:i1] = True


def _runs(mask: np.ndarray, join: int) -> List[Tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    out: List[Tuple[int, int]] = []
    for s, e in zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()):
        if out and s - out[-1][1] < join:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _longest_run(b: np.ndarray) -> int:
    if not b.any():
        return 0
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def _has_hum(seg: np.ndarray) -> bool:
    """A narrow spectral peak 20 dB over the median bin: a hum, where equal-power fades go comb-like."""
    if seg.size < 1024:
        return False
    p = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2
    p = p[1:]
    med = float(np.median(p))
    if med <= 1e-300:
        return False
    return 10.0 * np.log10(float(p.max()) / med) >= 20.0


def _xfade(ov: int, hum: bool) -> Tuple[np.ndarray, np.ndarray]:
    """(fade-in, fade-out) for ov samples: equal power, or linear for a hum."""
    t = (np.arange(ov) + 0.5) / ov
    if hum:
        return t, 1.0 - t
    return np.sin(t * np.pi / 2), np.cos(t * np.pi / 2)


def _finish_audio(a, fill=None) -> np.ndarray:
    """Never clips, never leaves a run of exact zeros (fill = the value to use there, default 1e-7)."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 0.98:
        a = a * (0.98 / peak)
    tiny = a == 0.0
    if tiny.any():
        f = np.full(a.shape, 1e-7) if fill is None else np.where(np.abs(fill) >= 1e-7, fill, 1e-7)
        a = np.where(tiny, f, a)
    return a.astype(np.float32)


def find_quiet_gaps(speech_spans: Sequence[Span], lo_s: float, hi_s: float, params: AmbienceParams,
                    waveform: Optional[np.ndarray] = None, sr: Optional[int] = None) -> List[Span]:
    """Stretches inside [lo_s, hi_s] (abs clock) in which nobody speaks, cleared by the margins.

    `speech_spans` is the speech of EVERY speaker (all diarized people, not just the two of the excerpt), in any order,
    possibly overlapping. A gap is what is left between their union, shrunk by gap_margin_after_s after each span and
    gap_margin_before_s before the next, and kept when it is at least gap_min_s long. When `waveform` and `sr` are
    given, a gap whose p90 20 ms frame level is more than energy_reject_db above the median level of the quiet frames
    of the whole window is dropped (a cough, a door, a voice the diarization missed).
    """
    lo, hi = float(lo_s), float(hi_s)
    if not hi > lo:
        return []
    merged = _union(_clean_spans(speech_spans, lo, hi))
    gaps: List[Span] = []
    start = lo
    for a, b in merged:
        end = a - params.gap_margin_before_s
        if end - start >= params.gap_min_s:
            gaps.append((start, end))
        start = b + params.gap_margin_after_s
    if hi - start >= params.gap_min_s:
        gaps.append((start, hi))
    if not gaps or waveform is None or sr is None or int(sr) <= 0:
        return gaps
    x = _as_1d(waveform)
    sr = int(sr)
    fl = _frame_len(sr)
    i_lo, i_hi = max(0, int(round(lo * sr))), min(x.size, int(round(hi * sr)))
    if i_hi - i_lo < fl:
        return gaps
    lev = _db(_frame_rms(x[i_lo:i_hi], fl))
    centers = i_lo / sr + (np.arange(lev.size) * fl + fl / 2) / sr
    loud = _in_spans(centers, merged)
    quiet = ~loud
    ref = float(np.median(lev[quiet])) if quiet.any() else float(np.percentile(lev, 5))
    kept: List[Span] = []
    for a, b in gaps:
        j0, j1 = max(0, int(round(a * sr))), min(x.size, int(round(b * sr)))
        g = _db(_frame_rms(x[j0:j1], fl))
        if g.size and float(np.percentile(g, 90)) - ref > params.energy_reject_db:
            continue
        kept.append((a, b))
    return kept


def noise_floor_track(waveform: np.ndarray, sr: int, speech_spans: Sequence[Span], start_s: float, end_s: float,
                      params: AmbienceParams) -> np.ndarray:
    """The level of the room (linear RMS) for every sample of [start_s, end_s) (abs clock).

    The median RMS of the 20 ms frames outside `speech_spans`, per floor_window_s window, interpolated between window
    centres and smoothed; windows without a quiet frame take the nearest window's value, and a recording with none
    at all takes its 5th-percentile frame level. Length = round((end_s - start_s) * sr). Never zero (floored at 1e-6).
    """
    sr = int(sr)
    n = max(0, int(round((float(end_s) - float(start_s)) * sr))) if sr > 0 else 0
    if n == 0:
        return np.zeros(0)
    x = _as_1d(waveform)
    fl = _frame_len(sr)
    nfile = x.size // fl
    if nfile == 0:
        return np.full(n, _FLOOR)
    W = max(fl, int(round(params.floor_window_s * sr))) / sr
    ks = np.arange(int(np.floor(float(start_s) / W - 0.5)) - 1, int(np.floor(float(end_s) / W - 0.5)) + 3)
    # only the frames of the windows in use are measured; they sit on the file's own 20 ms grid
    f0 = max(0, int(np.floor(ks[0] * W * sr / fl)))
    f1 = min(nfile, int(np.ceil((ks[-1] + 1) * W * sr / fl)))
    if f1 <= f0:  # the excerpt lies outside the file: borrow the nearest stretch of it
        m = int(np.ceil(ks.size * W * sr / fl))
        f0, f1 = (max(0, nfile - m), nfile) if f0 >= nfile else (0, min(nfile, m))
    fr = _frame_rms(x[f0 * fl:f1 * fl], fl)
    if fr.size == 0:
        return np.full(n, _FLOOR)
    fc = ((f0 + np.arange(fr.size)) * fl + fl / 2) / sr
    quiet = ~_in_spans(fc, _union(_clean_spans(speech_spans)))
    vals = np.full(ks.size, np.nan)
    for i, k in enumerate(ks):
        lo = np.searchsorted(fc, k * W, side="left")
        hi = np.searchsorted(fc, (k + 1) * W, side="left")
        sel = fr[lo:hi][quiet[lo:hi]]
        if sel.size:
            vals[i] = float(np.median(sel))
    valid = np.flatnonzero(np.isfinite(vals))
    if valid.size == 0:
        vals = np.full(ks.size, float(np.percentile(fr, 5)))
    else:
        idx = np.arange(ks.size)
        r = np.clip(np.searchsorted(valid, idx), 0, valid.size - 1)
        left = np.clip(r - 1, 0, valid.size - 1)
        pick = np.where(np.abs(valid[r] - idx) < np.abs(valid[left] - idx), r, left)
        vals = vals[valid[pick]]
    t = float(start_s) + np.arange(n) / sr
    track = np.interp(t, (ks + 0.5) * W, vals)
    track = _smooth(track, 0.05 * sr)
    return np.maximum(track, _FLOOR)


def build_bed(waveform: np.ndarray, sr: int, gaps: Sequence[Span], speech_spans: Sequence[Span],
              start_s: float, end_s: float, params: AmbienceParams, focus_s: Optional[float] = None) -> BedResult:
    """A continuous room tone for [start_s, end_s) (abs clock), length round((end_s - start_s) * sr).

    Tiers, first that works:
      "local"      pieces of gaps within prefer_radius_s of `focus_s` (default: the middle of the excerpt), when
                   those gaps add up to at least the excerpt's length
      "file"       the local gaps plus the nearest other gaps of the recording until they add up to the excerpt's
                   length (all gaps if they never do, then repeats); also when the local gaps are too short
      "synthetic"  noise shaped by the spectrum of the quietest frames (minimum statistics), when no gap is long enough
      "pink"       pink noise at pink_dbfs, when nothing at all can be measured
    Pieces are random slices of the gaps (seeded by params.seed and start_s), joined by equal-power crossfades of
    xfade_ms (linear when the bed has a strong tonal hum), no sample range used twice while unused range is left, and
    each piece scaled to follow noise_floor_track. The result is never digital silence and never clips.
    """
    sr = int(sr)
    start_s, end_s = float(start_s), float(end_s)
    n = max(0, int(round((end_s - start_s) * sr))) if sr > 0 else 0
    if n == 0:
        return BedResult(np.zeros(0, np.float32), "pink", notes=["empty range"])
    rng = np.random.default_rng([int(params.seed) & 0xFFFFFFFF, int(round(start_s * 1000)) & 0xFFFFFFFF])
    x = _as_1d(waveform)
    focus = 0.5 * (start_s + end_s) if focus_s is None else float(focus_s)
    floor = noise_floor_track(x, sr, speech_spans, start_s, end_s, params)
    min_len = max(2, int(round(params.gap_min_s * sr)))
    usable = []
    for a, b in _clean_spans(gaps):
        i0, i1 = max(0, int(round(a * sr))), min(x.size, int(round(b * sr)))
        if i1 - i0 < min_len:
            continue
        step = max(1, (i1 - i0) // 65536)  # silence test on a subsample: a long gap is not read whole
        if _rms(x[i0:i1:step]) < 1e-6:
            continue
        usable.append((a, b, i0, i1))
    local = [g for g in usable if _dist(g[0], g[1], focus) <= params.prefer_radius_s]
    if usable:
        need = _needed(n, sr, params)
        local_len = sum(g[3] - g[2] for g in local)
        notes: List[str] = []
        if local and local_len >= need:
            pool, tier = local, "local"
        else:
            # the local gaps do not cover the excerpt: add the nearest gaps of the file until they do
            order = sorted(range(len(usable)), key=lambda j: _dist(usable[j][0], usable[j][1], focus))
            take, total = set(), 0
            for j in order:
                if total >= need:
                    break
                take.add(j)
                total += usable[j][3] - usable[j][2]
            pool = [g for j, g in enumerate(usable) if j in take]
            tier = "file"
            if not local:
                notes.append("no gap within prefer_radius_s")
            elif local_len < need:
                notes.append(f"local gaps cover {local_len / sr:.1f} s of {need / sr:.1f} s; nearest file gaps added")
            if total < need:
                notes.append("gaps of the whole recording are shorter than the excerpt: repeats")
        audio, pieces, repeats = _assemble(x, sr, [(g[2], g[3]) for g in pool], floor, n, params, rng)
        return BedResult(_finish_audio(audio), tier, pieces, repeats, notes)
    syn = _synthetic(x, sr, floor, n, rng)
    if syn is not None:
        return BedResult(_finish_audio(syn), "synthetic", notes=["no gap long enough"])
    return BedResult(_finish_audio(_pink(n, params, rng)), "pink", notes=["nothing measurable"])


def _needed(n: int, sr: int, params: AmbienceParams) -> int:
    """Gap samples a bed of n samples needs: the output plus the crossfade overlap each piece gives back.

    A piece is 1.5-3 s, so there are at most about n / 1.5 s + 2 of them; each one eats xfade samples of the next.
    """
    xf = max(1, int(round(params.xfade_ms * sr / 1000.0)))
    return n + xf * (n // max(1, int(1.5 * sr)) + 2)


def _dist(a: float, b: float, focus: float) -> float:
    return 0.0 if a <= focus <= b else min(abs(a - focus), abs(b - focus))


def _take(track: np.ndarray, pos: int, length: int) -> np.ndarray:
    idx = np.clip(np.arange(pos, pos + length), 0, track.size - 1)
    return track[idx]


def _assemble(x: np.ndarray, sr: int, ranges: List[Tuple[int, int]], floor: np.ndarray, n: int,
              params: AmbienceParams, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """Random pieces of the gap sample ranges, equal-power joined, each scaled to the floor. Returns (bed, pieces, repeats)."""
    X = max(1, int(round(params.xfade_ms * sr / 1000.0)))
    lens = [b - a for a, b in ranges]
    min_piece, max_piece = min(lens), max(lens)
    hum = _has_hum(x[ranges[0][0]:ranges[0][1]][: 4 * sr])
    buf = np.zeros(n + max_piece + 1)
    all_free = [list(r) for r in ranges]
    free = [list(r) for r in ranges]
    reused = False
    repeats = pieces = 0
    acc = 0
    while acc < n:
        if not free:
            free = [list(r) for r in all_free]
            reused = True
        sizes = np.array([b - a for a, b in free], dtype=np.float64)
        k = int(rng.choice(len(free), p=sizes / sizes.sum()))
        a, b = free[k]
        length = b - a
        want = int(rng.integers(int(1.5 * sr), int(3.0 * sr) + 1))
        p = min(length, want)
        if length - p < min_piece:
            p = length
        s0 = a + int(rng.integers(0, length - p + 1))
        rest = []
        for j, (c, d) in enumerate(free):
            if j != k:
                rest.append([c, d])
                continue
            if s0 - c >= min_piece:
                rest.append([c, s0])
            if d - (s0 + p) >= min_piece:
                rest.append([s0 + p, d])
        free = rest
        pieces += 1
        repeats += int(reused)
        ov = 0 if acc == 0 else min(X, p // 2, acc // 2)
        pos = acc - ov
        piece = x[s0:s0 + p]
        piece = piece * _take(floor, pos, p) / max(_rms(piece), 1e-9)
        if ov == 0:
            buf[pos:pos + p] = piece
        else:
            w_in, w_out = _xfade(ov, hum)
            buf[pos:pos + ov] = buf[pos:pos + ov] * w_out + piece[:ov] * w_in
            buf[pos + ov:pos + p] = piece[ov:]
        acc = pos + p
    return buf[:n], pieces, repeats


def _synthetic(x: np.ndarray, sr: int, floor: np.ndarray, n: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    """Noise shaped like the quietest tenth of the recording's frames (minimum statistics), scaled to the floor."""
    N = 1024
    nf = x.size // N
    if nf == 0:
        return None
    win = np.hanning(N)
    energy = np.zeros(nf)
    for c in range(0, nf, _FRAME_CHUNK):
        m = min(_FRAME_CHUNK, nf - c)
        blk = np.asarray(x[c * N:(c + m) * N], dtype=np.float64).reshape(m, N)
        energy[c:c + m] = np.sum(blk * blk, axis=1)
    k = max(1, int(np.ceil(0.10 * nf)))
    quiet = np.stack([np.asarray(x[i * N:(i + 1) * N], dtype=np.float64) for i in np.argsort(energy)[:k]])
    P = np.mean(np.abs(np.fft.rfft(quiet * win, axis=1)) ** 2, axis=0)
    if not np.all(np.isfinite(P)) or P.sum() <= 1e-24:
        return None
    amp = np.sqrt(P)
    hop = N // 2
    nb = -(-n // hop) + 1
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N) / N)
    out = np.zeros(nb * hop + N)
    for c0 in range(0, nb, 2048):
        m = min(2048, nb - c0)
        g = (rng.standard_normal((m, amp.size)) + 1j * rng.standard_normal((m, amp.size))) * (amp / np.sqrt(2.0))
        blk = np.fft.irfft(g, n=N, axis=1) * hann
        for j in range(m):
            s = (c0 + j) * hop
            out[s:s + N] += blk[j]
    syn = out[:n]
    level = _rms(syn)
    if not level > 1e-12:
        return None
    return syn * floor / level


def _pink(n: int, params: AmbienceParams, rng: np.random.Generator) -> np.ndarray:
    S = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(S.size, dtype=np.float64)
    f[0] = 1.0
    y = np.fft.irfft(S / np.sqrt(f), n=n)
    level = _rms(y)
    target = 10.0 ** (params.pink_dbfs / 20.0)
    return y * (target / level) if level > 0 else np.full(n, target)


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
    n = max(0, int(n))
    sr = int(sr)
    if n == 0 or sr <= 0:
        return np.ones(n, dtype=bool)
    unsafe = np.zeros(n, dtype=bool)
    tail = 0.0 if tail_ms is None else min(max(float(tail_ms), 0.0), params.reverb_tail_max_ms)
    dil = params.safe_dilate_ms / 1000.0
    for a, b in _clean_spans(other_spans):
        _mark(unsafe, sr, start_s, a - dil, b + dil + tail / 1000.0)
    mar = params.other_margin_ms / 1000.0
    for a, b in _clean_spans(third_spans):
        _mark(unsafe, sr, start_s, a - mar, b + mar)
    for a, b in _clean_spans(bad_spans):
        _mark(unsafe, sr, start_s, a, b)
    for a, b in _clean_spans(overlap_spans):
        if b - a < params.min_overlap_s:
            mid = 0.5 * (a + b)
            _mark(unsafe, sr, start_s, mid - 0.05, mid + 0.05)
        else:
            _mark(unsafe, sr, start_s, a, b)
    return ~unsafe


def compose_user_channel(raw: np.ndarray, user_track: np.ndarray, bed: np.ndarray, safe: np.ndarray, sr: int,
                         params: AmbienceParams) -> np.ndarray:
    """The user channel, len(raw) samples.

    raw_mask:  raw where `safe`; user_track + bed where not safe. Every raw<->bed boundary is a crossfade of
               splice_fade_ms, centred on the boundary: linear for the voice (raw and user_track carry the same voice),
               equal-power for the bed. The bed is NOT added to raw (that would be two rooms).
    bed_only:  bed + user_track everywhere.
    `user_track` is the user's separated voice on the excerpt clock, zeros where the user is silent. The result never
    exceeds full scale (a gentle limiter, not a clip) and has no digital silence longer than 10 ms.
    """
    raw = _as_f64(raw)
    n = raw.size
    if n == 0:
        return np.zeros(0, np.float32)
    ut = _fit(user_track, n)
    bd = _fit(bed, n)
    sr = int(sr)
    if params.mode == "bed_only":
        out = bd + ut
    else:
        sf = np.zeros(n)
        m = np.asarray(safe, dtype=bool).reshape(-1)[:n]
        sf[: m.size] = m
        s = np.clip(_conv_centred(sf, _hann_kernel(_odd(params.splice_fade_ms * sr / 1000.0))), 0.0, 1.0)
        # the voice is the same signal in raw and in user_track: linear weights (sum 1) so it never adds up to +3 dB.
        # The bed is room noise, uncorrelated with raw: equal power on it.
        out = s * raw + (1.0 - s) * ut + np.sqrt(1.0 - s) * bd
    return _finish_audio(out, bd)


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
    c = _as_f64(clean)
    L = c.size
    sr = int(sr)
    out = np.zeros(L)
    if L == 0 or sr <= 0:
        return out.astype(np.float32), {"islands": 0, "restored": 0, "level_adjusted": 0, "zeroed_fraction": 0.0}
    b = _fit(before, L)
    if islands is None:
        isl = _runs(b != 0.0, int(round(0.15 * sr)))
    else:
        isl = []
        for s0, s1 in _clean_spans(islands):
            i0, i1 = min(max(int(round(s0 * sr)), 0), L), min(max(int(round(s1 * sr)), 0), L)
            if i1 > i0:
                isl.append((i0, i1))
    pad = int(round(params.island_pad_ms * sr / 1000.0))
    fade = max(1, int(round(params.island_fade_ms * sr / 1000.0)))
    floor_lin = 10.0 ** (params.expander_floor_dbfs / 20.0)
    env_w, gate_w = 0.005 * sr, 0.010 * sr
    restored = adjusted = 0
    for i0, i1 in isl:
        a, z = max(0, i0 - pad), min(L, i1 + pad)
        m = z - a
        mask = np.ones(m)
        r = min(fade, m // 2)
        if r > 0:
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
            mask[:r] = ramp
            mask[m - r:] = ramp[::-1]
        rms_b = _rms(b[i0:i1])
        rms_c = _rms(c[i0:i1])
        if (i1 - i0) / sr < params.restore_max_s and rms_b > 0 and rms_c < params.restore_ratio * rms_b:
            seg = b[a:z] * mask
            restored += 1
        else:
            seg_c = c[a:z]
            env = np.sqrt(np.maximum(_smooth(seg_c * seg_c, env_w), 0.0))
            ind = (env >= floor_lin).astype(np.float64)
            seg = seg_c * mask * (ind * _smooth(ind, gate_w))
            rms_out = _rms(seg[i0 - a:i1 - a])
            if rms_b > 0 and rms_out > 0:
                d = 20.0 * np.log10(rms_out / rms_b)
                if abs(d) > params.level_match_db:
                    seg = seg * 10.0 ** ((np.sign(d) * params.level_match_db - d) / 20.0)
                    adjusted += 1
        out[a:z] = np.where(np.abs(seg) > np.abs(out[a:z]), seg, out[a:z])
    report = {"islands": len(isl), "restored": restored, "level_adjusted": adjusted,
              "zeroed_fraction": float(np.mean(out == 0.0))}
    return out.astype(np.float32), report


def quality_report(user: np.ndarray, assistant: np.ndarray, sr: int, user_spans: Sequence[Span],
                   assistant_spans: Sequence[Span], bed_tier: Optional[str] = None) -> dict:
    """Numbers to audit one exported pair (spans on the arrays' clock).

    {"user_digital_silence_s": longest run of exact zeros in the user channel outside user_spans (goal 0),
     "assistant_floor_dbfs": the 10th percentile frame level of the assistant outside assistant_spans,
     "user_rms_jump_db": the largest RMS step (50 ms before vs after) at a point where the user channel switches
                          source, as an upper bound taken over all points where the level changes by > 6 dB in 50 ms,
     "bed_tier": bed_tier}
    """
    sr = max(1, int(sr))
    u = _as_f64(user)
    a = _as_f64(assistant)
    inside = np.zeros(u.size, dtype=bool)
    for s0, s1 in _clean_spans(user_spans):
        _mark(inside, sr, 0.0, s0, s1)
    silence_s = _longest_run((u == 0.0) & ~inside) / sr
    fl = _frame_len(sr)
    lev = _db(_frame_rms(a, fl))
    fc = (np.arange(lev.size) * fl + fl / 2) / sr
    outside = ~_in_spans(fc, _union(_clean_spans(assistant_spans)))
    pool = lev[outside] if outside.any() else lev
    floor_db = float(np.percentile(pool, 10)) if pool.size else -180.0
    return {
        "user_digital_silence_s": float(silence_s),
        "assistant_floor_dbfs": floor_db,
        "user_rms_jump_db": _max_rms_jump(u, sr),
        "bed_tier": bed_tier,
    }


def _max_rms_jump(u: np.ndarray, sr: int) -> float:
    """Largest 50 ms-before vs 50 ms-after RMS step, over the points where the level changes by more than 6 dB."""
    w = max(1, int(round(0.05 * sr)))
    if u.size < 2 * w:
        return 0.0
    cs = np.concatenate([[0.0], np.cumsum(u * u)])
    t = np.arange(w, u.size - w + 1, max(1, w // 5))
    before = np.sqrt(np.maximum((cs[t] - cs[t - w]) / w, 0.0))
    after = np.sqrt(np.maximum((cs[t + w] - cs[t]) / w, 0.0))
    step = np.abs(_db(after) - _db(before))
    changed = step[step > 6.0]
    return float(changed.max()) if changed.size else 0.0
