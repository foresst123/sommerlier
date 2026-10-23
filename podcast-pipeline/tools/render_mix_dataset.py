"""Render training mixtures that follow the overlaps measured in real recordings.

Every sample starts from a REAL overlap template (measure_real_overlap_stats.py writes
pairs.csv): the overlap length L, the pattern (full / contained / partial) and the level
difference. It is realised with clean single-speaker stretches of the same recording
(nobody else talks within --margin seconds, no music) and laid out by the shared rule in
utils/separation_layout.py, the same rule the inference window planner uses:

    [solo X][ctx left][CORE (L s, both speakers)][ctx right][solo Y]   exactly --window s

ctx continues the very stretch used for the core (contained: both sides from the holder A;
partial: left from A, right from B; full: silence). Solo pieces come from elsewhere in the
recording; which speaker's solo comes first is random. Every junction is a 20 ms
crossfade, never inside the core. Each speaker gets one gain over all its pieces.
Samples the rule cannot fill even through its fallback chain (cases 5/6 of the design
doc) produce no row; render_report.json counts them.

  <out>/<split>/<id>_mix.wav, _s1.wav, _s2.wav (+ _noise.wav)   --sr mono, PCM-16
  <out>/chunks.jsonl            one row per chunk, with `slots` and `layout` (source time of
                                every piece): the source of truth every manifest derives from
  <out>/render_report.json      tries, drops by reason, fallbacks, multi-overlap share
  <out>/manifests/{mossformer2,tiger,speechbrain}/   per-trainer lists (W per model)
  <out>/manifests/tse/{train,val}.jsonl              TSE: two rows per mix (target A, B)
  <out>/tse/<split>/...  and  <out>/tse_mixes.jsonl  TSE mixes (--tse-mix plain)
  <out>/enroll/<recording>/<speaker>.wav/.json       enrollment bank per recording+speaker

What is drawn per chunk (--mix real:strat:hard, default 0.7:0.2:0.1):
  real   a template from the SAME recording (its own measured overlaps)
  strat  a template from the split's pool, equal share per core-length bin
         (l_min-0.5, 0.5-1, 1-W s), so long overlaps are not starved
  hard   an overlap >= 1 s with a near-zero or very wide level difference
Templates with L < --l-min or L > --window are never drawn.
Noise: with --noise-prob, background from the recording's own non-speech gaps is added to
the MIXTURE (stems stay clean) and saved as _noise.wav, so mix = s1 + s2 + noise.

    python tools/render_mix_dataset.py --cache cache --stats overlap_stats \\
        --out mix_16k_w4 --n 20000 --window 4 --exclude-index overlap_samples/index.json
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_overlap_samples as cos  # noqa: E402  (also puts podcast-pipeline/ on sys.path)
import measure_real_overlap_stats as mrs  # noqa: E402
from utils import separation_layout as sl  # noqa: E402

BASE_DB = -23.0           # level of the holder's active speech
MIN_ACTIVE = 0.15         # each stem needs this much active speech in the chunk
FRAME = 0.01
READ_PAD = 0.05           # source read around each run before resampling
WHOLE_CLIP_P = 0.5        # contained: use a whole short clean clip of B as the core
WHOLE_CLIP_SLACK = 0.15   # ... when its length is within [L, L + this]
SNAP = 0.08               # cut points move at most this far to a quiet spot
SNAP_SHORT = 0.02         # ... for cores shorter than SHORT_CORE
SHORT_CORE = 0.4
ENROLL_SECONDS, ENROLL_MIN, ENROLL_K, ENROLL_CLIP_MAX, ENROLL_BANK_CLIPS = 4.0, 1.0, 8, 4.0, 40


# ----------------------------------------------------------------------------- templates
def load_templates(stats_dir: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    with open(os.path.join(stats_dir, "pairs.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["recording"]].append({
                "rec": r["recording"], "t": float(r.get("t") or 0.0), "overlap": float(r["overlap"]),
                "pattern": r["pattern"], "first_len": float(r["first_len"]),
                "second_len": float(r["second_len"]),
                "delta_db": float(r["delta_db"]) if r["delta_db"] != "" else None})
    return out


def load_rates(stats_dir: str) -> Dict[str, float]:
    """events per minute per recording (recordings.csv of measure_real_overlap_stats.py)."""
    p = os.path.join(stats_dir, "recordings.csv")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return {r["recording"]: float(r["events_per_min"] or 0) for r in csv.DictReader(f)}


def usable_templates(ts: List[dict], l_min: float, window: float) -> List[dict]:
    return [t for t in ts if l_min - 1e-9 <= t["overlap"] <= window + 1e-9]


def pick_template(rng: random.Random, kind: str, own: List[dict], pool: List[dict],
                  bins: List[Tuple[float, float]], window: float = 1e9):
    if kind == "real" and own:
        return rng.choice(own), "real"
    if kind == "strat" or (kind == "real" and not own):
        k = rng.randrange(len(bins))
        lo, hi = bins[k]
        cand = [t for t in pool if sl.in_bin(t["overlap"], lo, hi, k == len(bins) - 1)]
        if cand:
            return rng.choice(cand), "strat"
    cand = [t for t in pool if 1.0 <= t["overlap"] <= window] or pool
    if not cand:
        return None, "hard"
    t = dict(rng.choice(cand))
    t["delta_db"] = rng.uniform(-2, 2) if rng.random() < 0.5 else rng.choice([-1, 1]) * rng.uniform(6, 12)
    return t, "hard"


def multi_pair_share(templates: Dict[str, List[dict]], merge_gap: float) -> dict:
    """Pairs whose overlaps lie within `merge_gap` of each other form one event (case 4)."""
    pairs = events = multi = 0
    for ts in templates.values():
        spans = sorted((t["t"], t["t"] + t["overlap"]) for t in ts)
        end, k = None, 0
        for a, b in spans:
            if end is not None and a - end <= merge_gap:
                k += 1
                end = max(end, b)
                continue
            if end is not None:
                events += 1
                multi += k > 1
            end, k = b, 1
        if end is not None:
            events += 1
            multi += k > 1
        pairs += len(spans)
    return {"pairs": pairs, "events": events, "pairs_per_event": round(pairs / events, 4) if events else None,
            "multi_pair_event_share": round(multi / events, 4) if events else None}


# ----------------------------------------------------------------------------- audio helpers
def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    from scipy.signal import resample_poly
    g = math.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def read_run(audio, sr_src: int, sr: int, start_s: float, n_out: int) -> np.ndarray:
    """`n_out` samples at `sr` from source time `start_s`: the stretch is read with padding,
    resampled on its own and trimmed, so sample 0 is exactly `start_s` (on the source grid)."""
    g = math.gcd(sr_src, sr)
    up, down = sr // g, sr_src // g
    pad = down * int(math.ceil(READ_PAD * sr_src / down))
    k0 = int(round(start_s * sr_src))
    lo, hi = k0 - pad, k0 + int(math.ceil(n_out * down / up)) + pad
    x = np.zeros(hi - lo, np.float32)
    a, b = max(0, lo), min(len(audio), hi)
    if b > a:
        x[a - lo:b - lo] = np.asarray(audio[a:b], dtype=np.float32)
    y = resample(x, sr_src, sr)
    off = pad * up // down
    y = y[off:off + n_out]
    return y if len(y) == n_out else np.pad(y, (0, n_out - len(y)))


def level_db(x: np.ndarray, sr: int) -> float:
    lv = mrs.speech_level_db(x, sr)
    return lv if lv is not None else float(10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-12))


def active_seconds(x: np.ndarray, sr: int, floor_db: float = -45.0) -> float:
    n = int(FRAME * sr)
    if len(x) < n:
        return 0.0
    f = np.asarray(x[: len(x) // n * n], np.float64).reshape(-1, n)
    return float(np.sum(10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12) > floor_db)) * FRAME


def noise_bank(segs: List[dict], audio, sr: int, max_pieces: int = 30) -> Optional[np.ndarray]:
    cover = cos.merge_close([(s["start"], s["end"]) for s in segs], 0.0)
    gaps = [(cover[i][1], cover[i + 1][0]) for i in range(len(cover) - 1) if cover[i + 1][0] - cover[i][1] >= 0.8]
    parts = []
    for a, b in mrs._evenly(gaps, max_pieces):
        mid, half = (a + b) / 2, min((b - a) / 2, 1.5)
        parts.append(np.asarray(audio[int((mid - half) * sr):int((mid + half) * sr)], dtype=np.float32))
    bank = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return bank if len(bank) >= 2 * sr else None


def make_snapper(audio, sr_src: int):
    """Move a cut to the best acoustic boundary within [lo, hi] (AcousticBoundaryFinder,
    energy fallback without Silero), as the inference planner does."""
    from utils.acoustic_boundary import AcousticBoundaryFinder
    finder = AcousticBoundaryFinder(audio, sr_src)

    def snap(s: float, lo: float, hi: float):
        a, lo_i, hi_i = int(round(s * sr_src)), int(math.ceil(lo * sr_src)), int(math.floor(hi * sr_src))
        if hi_i <= lo_i:
            return s, "none"
        d = finder.find_cut(min(max(a, lo_i), hi_i), search_min=lo_i, search_max=hi_i)
        return (d.sample / sr_src, d.method) if d is not None else (s, "none")
    return snap


# ----------------------------------------------------------------------------- one chunk: plan
def _len(c: Tuple[float, float], sr: int) -> int:
    return int(math.floor((c[1] - c[0]) * sr + 1e-6))


def plan_chunk(rng: random.Random, t: dict, pools: Dict[str, list], n: int, cfg: sl.LayoutConfig,
               sr_src: int, total: Optional[float] = None, snap=None,
               whole_clip_p: float = WHOLE_CLIP_P):
    """Layout of one chunk from template `t` and clean clips `pools` {speaker: [(a, b) s]}.
    No audio is read (except by `snap`). Returns (layout, None) or (None, drop_reason)."""
    sr = cfg.sr
    spk = sorted(s for s, c in pools.items() if c)
    if len(spk) < 2:
        return None, "one_speaker"
    if not cfg.l_min - 1e-9 <= t["overlap"] <= n / sr + 1e-9:
        return None, "out_of_range"
    holder, other = rng.sample(spk, 2)
    pa, pb = pools[holder], pools[other]
    pattern = t["pattern"] if t["pattern"] in sl.CTX_OWNER else "partial"
    f = cfg.n(cfg.fade)
    core = min(n, max(1, cfg.n(t["overlap"])))
    clip_b, whole = None, False
    if pattern == "contained" and rng.random() < whole_clip_p:
        short = [c for c in pb if core <= _len(c, sr) <= min(n, core + cfg.n(WHOLE_CLIP_SLACK))]
        if short:
            clip_b, whole = rng.choice(short), True
            core = _len(clip_b, sr)
    b = sl.plan_budget(n, core, cfg)
    mn_l, mn_r = b.ctx_min

    def choose(clips, need, minimum):
        fits = [c for c in clips if _len(c, sr) >= need]
        if fits:
            return rng.choice(fits)
        ok = [c for c in clips if _len(c, sr) >= minimum]
        return max(ok, key=lambda c: c[1] - c[0]) if ok else None

    if pattern == "contained":
        clip_a = choose(pa, core + b.ctx_left + b.ctx_right + f, core + mn_l + mn_r + f)
        if clip_b is None:
            clip_b = choose(pb, core + 2 * f, core)
        room = None if clip_a is None else _len(clip_a, sr) - core - f
        avail = dict(ctx_left=room, ctx_right=room, ctx_total=room)
    elif pattern == "partial":
        clip_a = choose(pa, b.ctx_left + core + f, mn_l + core + f)
        clip_b = choose(pb, core + b.ctx_right + f, core + mn_r + f)
        avail = dict(ctx_left=None if clip_a is None else _len(clip_a, sr) - core - f,
                     ctx_right=None if clip_b is None else _len(clip_b, sr) - core - f)
    else:
        clip_a = choose(pa, core + 2 * f, core)
        clip_b = choose(pb, core + 2 * f, core)
        avail = {}
    if clip_a is None or clip_b is None:
        return None, "no_core_clip"
    rest = {"A": [c for c in pa if c != clip_a], "B": [c for c in pb if c != clip_b]}
    solo_av = tuple(max([_len(c, sr) - f for c in rest[w]] or [0]) for w in "AB")
    fb, reasons = sl.fallback(b, sl.Avail(solo=solo_av, **avail), cfg)
    if fb is None:
        return None, reasons
    order = ("A", "B") if rng.random() < 0.5 else ("B", "A")
    slots = sl.make_slots(fb, pattern, order)
    runs = sl.make_runs(slots, pattern, cfg)
    pieces, cuts = [], Counter()
    for k, run in enumerate(runs):
        who = run["who"]
        ns, ne = run["nominal"]
        ln, (el, er) = ne - ns, run["ext"]
        if run["role"] == "core":
            clip = clip_a if who == "A" else clip_b
        else:
            fits = [c for c in rest[who] if _len(c, sr) >= ln + f] or \
                [max(rest[who], key=lambda c: c[1] - c[0])]
            clip = rng.choice(fits)
        lo, hi = clip[0] + el / sr, clip[1] - (ln + er) / sr
        if hi < lo:
            lo, hi = clip[0], max(clip[0], clip[1] - ln / sr)
        if run["role"] == "core" and who == "B" and whole:
            s = clip[0]
        else:
            s = rng.uniform(lo, hi)
            if snap is not None:
                short = run["role"] == "core" and core < cfg.n(SHORT_CORE)
                shift = SNAP_SHORT if short else SNAP
                s, method = snap(s, max(lo, s - shift), min(hi, s + shift))
                cuts[method] += 1
        s0 = round((s - el / sr) * sr_src) / sr_src
        span = (run["end"] - run["start"]) / sr
        if total is not None:
            s0 = min(max(0.0, s0), max(0.0, math.floor((total - span) * sr_src) / sr_src))
        for p in run["pieces"]:
            a = s0 + (p["start"] - run["start"]) / sr
            pieces.append(dict(p, run=k, src_start=round(a, 6), src_end=round(a + (p["end"] - p["start"]) / sr, 6)))
    c0, c1 = sl.core_range(slots)
    reasons = list(dict.fromkeys(reasons))
    return {"n": slots[-1]["end"], "sr": sr, "pattern": pattern, "L": round((c1 - c0) / sr, 6), "core": [c0, c1],
            "slots": slots, "layout": pieces,
            "solo_order": [s["who"] for s in slots if s["kind"] == "solo"],
            "fallback": "+".join(reasons) if reasons else None, "whole_clip_core": whole,
            "holder": holder, "interrupter": other, "cuts": dict(cuts),
            "layout_cfg": asdict(cfg)}, None


# ----------------------------------------------------------------------------- one chunk: audio
def render_layout(lay: dict, audio, sr_src: int) -> Dict[str, np.ndarray]:
    """Raw stems {"A", "B"} (no gain) of a layout; each run is read and resampled once."""
    n, sr = lay["n"], lay["sr"]
    stems = {"A": np.zeros(n, np.float32), "B": np.zeros(n, np.float32)}
    runs: Dict[int, list] = defaultdict(list)
    for p in lay["layout"]:
        runs[p["run"]].append(p)
    for ps in runs.values():
        ps.sort(key=lambda p: p["start"])
        start, end = ps[0]["start"], ps[-1]["end"]
        x = read_run(audio, sr_src, sr, ps[0]["src_start"], end - start)
        fi, fo = ps[0]["fade_in"], ps[-1]["fade_out"]
        if fi:
            x[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)
        if fo:
            x[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
        stems[ps[0]["who"]][start:end] += x
    return stems


def realize(rng: random.Random, t: dict, pools: Dict[str, list], audio, sr_src: int, n: int,
            cfg: sl.LayoutConfig, noise: Optional[np.ndarray], noise_prob: float, delta: float,
            snap=None, total: Optional[float] = None):
    """Plan, render, level, noise. Returns (result, None) or (None, drop_reason)."""
    lay, why = plan_chunk(rng, t, pools, n, cfg, sr_src, total, snap)
    if lay is None:
        return None, why
    sr, n = cfg.sr, lay["n"]
    st = render_layout(lay, audio, sr_src)
    A = st["A"] * 10 ** ((BASE_DB - level_db(st["A"], sr)) / 20)
    B = st["B"] * 10 ** ((BASE_DB + delta - level_db(st["B"], sr)) / 20)
    if active_seconds(A, sr) < MIN_ACTIVE or active_seconds(B, sr) < MIN_ACTIVE:
        return None, "low_active"
    N, snr = None, None
    if noise is not None and rng.random() < noise_prob:
        snr = rng.uniform(10, 25) if rng.random() < 0.2 else rng.uniform(25, 45)
        m = int(round(n * sr_src / sr))
        start = rng.randint(0, max(0, len(noise) - m))
        seg = np.resize(noise[start:start + m] if len(noise) >= m else noise, m)
        seg = resample(seg, sr_src, sr)[:n]
        seg = np.pad(seg, (0, n - len(seg)))
        p = float(np.mean(seg.astype(np.float64) ** 2)) + 1e-12
        N = (seg * math.sqrt(10 ** ((BASE_DB - snr) / 10) / p)).astype(np.float32)
    M = A + B + (N if N is not None else 0)
    peak = max(float(np.max(np.abs(x))) for x in (A, B, M) + ((N,) if N is not None else ()))
    if peak > 0.95:
        k = 0.95 / peak
        A, B, M = A * k, B * k, M * k
        N = None if N is None else N * k
    lay.update(delta_db=round(delta, 2), noise_snr_db=None if snr is None else round(snr, 1))
    return {"A": A, "B": B, "M": M, "N": N, "layout": lay}, None


# ----------------------------------------------------------------------------- enrollment bank
def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", str(name))


def enroll_bank(segs, speakers, audio, sr_src: int, sr: int, margin: float, out_dir: str) -> Dict[str, dict]:
    """Per speaker: clean clips >= 1 s (spread over the recording, each <= 4 s) written once as
    one wav, with an index of offsets and source times. TSE rows point into it."""
    import soundfile as sf
    bank = {}
    os.makedirs(out_dir, exist_ok=True)
    for spk in speakers:
        clips = mrs._evenly(cos.clean_clips(segs, spk, margin, ENROLL_MIN), ENROLL_BANK_CLIPS)
        entries, parts, off = [], [], 0
        for a, b in clips:
            if b - a > ENROLL_CLIP_MAX:
                mid = (a + b) / 2
                a, b = mid - ENROLL_CLIP_MAX / 2, mid + ENROLL_CLIP_MAX / 2
            a = round(a * sr_src) / sr_src
            k = int((b - a) * sr)
            parts.append(read_run(audio, sr_src, sr, a, k))
            entries.append({"offset": off, "len": k, "src_start": round(a, 6), "src_end": round(a + k / sr, 6)})
            off += k
        if not entries:
            continue
        path = os.path.join(out_dir, f"{_safe(spk)}.wav")
        sf.write(path, np.concatenate(parts), sr, subtype="PCM_16")
        with open(path[:-4] + ".json", "w", encoding="utf-8") as fh:
            json.dump({"speaker": spk, "sr": sr, "clips": entries}, fh)
        bank[spk] = {"path": os.path.abspath(path), "clips": entries}
    return bank


def tse_rows(mix: dict, lay: dict, bank: Dict[str, dict], margin: float, stats: Counter) -> List[dict]:
    """Two TSE rows (target holder, target interrupter) for one mix; enrollment candidates are
    the nearest bank clips outside every source range of the mix (+- margin)."""
    exclude = [(p["src_start"], p["src_end"]) for p in lay["layout"]]
    out = []
    for who, role in (("A", "holder"), ("B", "interrupter")):
        spk = lay["holder"] if who == "A" else lay["interrupter"]
        if spk not in bank:
            stats["enroll_short"] += 1
            continue
        clips = [(e["src_start"], e["src_end"]) for e in bank[spk]["clips"]]
        core = next(p for p in lay["layout"] if p["kind"] == "core" and p["who"] == who)
        cand = sl.pick_enrollment(clips, exclude, (core["src_start"] + core["src_end"]) / 2, margin, ENROLL_K)
        default = sl.default_enrollment(clips, cand, ENROLL_SECONDS, ENROLL_MIN)
        if default is None:
            stats["enroll_short"] += 1
            continue
        out.append({"id": f"{mix['id']}_t{who}", "mix_id": mix["id"], "split": mix["split"],
                    "recording": mix["recording"], "mix": mix["mix"], "target": mix[who],
                    "target_role": role, "target_speaker": spk, "samples": lay["n"], "sr": lay["sr"],
                    "enroll_bank": bank[spk]["path"], "enroll_seconds": ENROLL_SECONDS,
                    "enroll_candidates": [dict(bank[spk]["clips"][i], index=i) for i in cand],
                    "enroll_default": [cand.index(i) for i in default]})
    return out


# ----------------------------------------------------------------------------- one recording
_JOBS: List[tuple] = []


def _write(sf, sdir, cid, cfg, pairs):
    paths = {}
    for tag, x in pairs:
        if x is None:
            paths[tag] = None
            continue
        p = os.path.join(sdir, f"{cid}_{tag}.wav")
        sf.write(p, x, cfg["sr"], subtype=cfg["subtype"])
        paths[tag] = os.path.abspath(p)
    return paths


def render_recording(job: tuple):
    import soundfile as sf
    rec, split, n_rows, own, pool, cfg, seed, out = job
    rng_t = random.Random(seed)        # template stream: the same for every --window
    segs = rec["segments"]()
    audio, sr_src = rec["audio"]()
    total = len(audio) / sr_src
    lcfg = sl.LayoutConfig(**cfg["layout"])
    n = sl.window_samples(cfg["window"], lcfg.sr, (sr_src,))
    pools = {}
    for s in sorted({x["speaker"] for x in segs}):
        c = cos.clean_clips(segs, s, cfg["margin"], cfg["min_clip"])
        if c:
            pools[s] = c
    noise = noise_bank(segs, audio, sr_src) if cfg["noise_prob"] > 0 else None
    snap = make_snapper(audio, sr_src) if cfg.get("snap", True) else None
    name = os.path.basename(rec["path"].rstrip("/"))
    own_u = usable_templates(own, lcfg.l_min, cfg["window"])
    bins = sl.len_bins(cfg["window"], lcfg.l_min)
    kinds, weights = ["real", "strat", "hard"], cfg["mix"]
    tse = cfg.get("tse_mix", "none")
    tcfg = replace(lcfg, solo=False)
    n_tse = sl.window_samples(cfg.get("tse_window") or cfg["window"], lcfg.sr, (sr_src,))
    bank = enroll_bank(segs, sorted(pools), audio, sr_src, lcfg.sr, cfg["margin"],
                       os.path.join(out, "enroll", _safe(name))) if tse != "none" else {}
    stats = Counter()
    rows, trows, mixes, tries = [], [], [], 0
    sdir = os.path.join(out, split)
    os.makedirs(sdir, exist_ok=True)
    while len(rows) < n_rows and tries < n_rows * 30:
        tries += 1
        t, kind = pick_template(rng_t, rng_t.choices(kinds, weights)[0], own_u, pool, bins, cfg["window"])
        attempt = rng_t.getrandbits(32)
        if t is None:
            stats["drop:no_template"] += 1
            continue
        delta = t["delta_db"] if kind == "hard" else (t.get("delta_db") or 0.0) + rng_t.gauss(0, 1.0)
        rng = random.Random(attempt)
        r, why = realize(rng, t, pools, audio, sr_src, n, lcfg, noise, cfg["noise_prob"], delta, snap, total)
        if r is None:
            stats[f"drop:{why}"] += 1
            continue
        lay = r["layout"]
        cid = f"{name}_{len(rows):04d}"
        flip = rng.random() < 0.5
        s1, s2 = (r["B"], r["A"]) if flip else (r["A"], r["B"])
        paths = _write(sf, sdir, cid, cfg, (("mix", r["M"]), ("s1", s1), ("s2", s2), ("noise", r["N"])))
        stats[f"fallback:{lay['fallback']}" if lay["fallback"] else "nominal"] += 1
        c0, c1 = lay["core"]
        row = {"id": cid, "split": split, "recording": name, "kind": kind,
               "window": cfg["window"], "sr": lcfg.sr, "samples": n, "src_sr": sr_src,
               "fade_samples": lcfg.n(lcfg.fade), "paths": paths,
               "target_order": "interrupter,holder" if flip else "holder,interrupter",
               "template_overlap": round(t["overlap"], 3), "first_len": t["first_len"],
               "second_len": t["second_len"], "template_t": t.get("t"),
               "overlap_in_window": lay["L"], "overlap_start_in_window": round(c0 / lcfg.sr, 4),
               **{k: v for k, v in lay.items() if k not in ("n",)}}
        rows.append(row)
        if tse == "none":
            continue
        if tse == "bss":
            mix = {"id": cid, "split": split, "recording": name, "mix": paths["mix"],
                   "A": paths["s2" if flip else "s1"], "B": paths["s1" if flip else "s2"]}
            trows += tse_rows(mix, lay, bank, cfg["margin"], stats)
            continue
        rt, why = realize(random.Random(attempt ^ 0x5EED), t, pools, audio, sr_src, n_tse, tcfg, noise,
                          cfg["noise_prob"], delta, snap, total)
        if rt is None:
            stats[f"tse_drop:{why}"] += 1
            continue
        tl = rt["layout"]
        tdir = os.path.join(out, "tse", split)
        os.makedirs(tdir, exist_ok=True)
        tid = f"{cid}_tse"
        tp = _write(sf, tdir, tid, cfg, (("mix", rt["M"]), ("A", rt["A"]), ("B", rt["B"]), ("noise", rt["N"])))
        mixes.append({"id": tid, "chunk": cid, "split": split, "recording": name, "window": tl["n"] / lcfg.sr,
                      "sr": lcfg.sr, "samples": tl["n"], "src_sr": sr_src, "paths": tp, "tse": True,
                      "fade_samples": lcfg.n(lcfg.fade), **{k: v for k, v in tl.items() if k != "n"}})
        trows += tse_rows({"id": tid, "split": split, "recording": name, "mix": tp["mix"], "A": tp["A"],
                           "B": tp["B"]}, tl, bank, cfg["margin"], stats)
    stats["tries"] += tries
    stats["rows"] += len(rows)
    stats["tse_rows"] += len(trows)
    return rows, trows, mixes, dict(stats)


def _run(i: int):
    return render_recording(_JOBS[i])


# ----------------------------------------------------------------------------- manifests
def write_manifests(out: str, rows: List[dict], tse: List[dict] = ()) -> None:
    import soundfile as sf
    base = os.path.join(out, "manifests")
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val")}

    def lines(path, items):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(items) + ("\n" if items else ""))

    for s, name in (("train", "tr_list.txt"), ("val", "cv_list.txt")):
        lines(os.path.join(base, "mossformer2", name),
              [f"{r['paths']['mix']} {r['paths']['s1']} {r['paths']['s2']}" for r in by[s]])
    for s, d in (("train", "train"), ("val", "val"), ("val", "test")):
        for tag in ("mix", "s1", "s2"):
            p = os.path.join(base, "tiger", d, f"{tag}.json")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump([[r["paths"][tag], r["samples"]] for r in by[s]], f)
    sb = os.path.join(base, "speechbrain")
    os.makedirs(sb, exist_ok=True)
    silence = {}
    for r in rows:      # noise-free chunks point at one silent file of their length
        if not r["paths"].get("noise") and r["samples"] not in silence:
            p = os.path.join(sb, f"silence_{r['samples']}.wav")
            sf.write(p, np.zeros(r["samples"], np.float32), r["sr"], subtype="PCM_16")
            silence[r["samples"]] = os.path.abspath(p)
    for s, name in (("train", "train.csv"), ("val", "valid.csv")):
        with open(os.path.join(sb, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ID", "duration", "mix_wav", "mix_wav_format", "mix_wav_opts",
                        "s1_wav", "s1_wav_format", "s1_wav_opts", "s2_wav", "s2_wav_format", "s2_wav_opts",
                        "noise_wav", "noise_wav_format", "noise_wav_opts"])
            for r in by[s]:
                w.writerow([r["id"], r["window"], r["paths"]["mix"], "wav", "",
                            r["paths"]["s1"], "wav", "", r["paths"]["s2"], "wav", "",
                            r["paths"].get("noise") or silence[r["samples"]], "wav", ""])
    if tse:
        for s in ("train", "val"):
            lines(os.path.join(base, "tse", f"{s}.jsonl"),
                  [json.dumps(r, ensure_ascii=False) for r in tse if r["split"] == s])


# ----------------------------------------------------------------------------- split
def split_recordings(names: List[str], rate: Dict[str, float], val_share: float, seed: int,
                     group_regex: Optional[str] = None) -> Dict[str, str]:
    """Val = one random group from each of n_val strata of groups sorted by event rate.
    A group is all recordings whose name gives the same `group_regex` match (show / host)."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for nm in names:
        m = re.search(group_regex, nm) if group_regex else None
        groups[(m.group(1) if m and m.groups() else m.group(0)) if m else nm].append(nm)
    keys = sorted(groups, key=lambda k: (float(np.mean([rate.get(x, 0.0) for x in groups[k]])), k))
    n_val = max(1, round(val_share * len(keys))) if val_share > 0 and len(keys) > 1 else 0
    rng = random.Random(seed)
    val = set()
    for st in (np.array_split(np.array(keys, dtype=object), n_val) if n_val else []):
        if len(st):
            val.add(rng.choice(list(st)))
    return {nm: ("val" if k in val else "train") for k in keys for nm in groups[k]}


# ----------------------------------------------------------------------------- driver
def build(recs: List[dict], templates: Dict[str, List[dict]], out: str, n: int = 20000,
          window: float = 4.0, sr: int = 16000, seed: int = 0, val_share: float = 0.08,
          mix=(0.7, 0.2, 0.1), noise_prob: float = 0.5, margin: float = 0.3, min_clip: float = 0.25,
          subtype: str = "PCM_16", workers: int = 1, per_rec_cap: int = 100, l_min: float = 0.25,
          tse_mix: str = "none", tse_window: Optional[float] = None, rates: Optional[Dict[str, float]] = None,
          group_regex: Optional[str] = None, merge_gap: float = 1.0, snap: bool = True,
          layout: Optional[dict] = None) -> List[dict]:
    global _JOBS
    if tse_mix not in ("none", "plain", "bss"):
        raise ValueError(f"tse_mix {tse_mix!r}")
    lcfg = sl.LayoutConfig(**dict(layout or {}, sr=sr, l_min=l_min))
    sl.window_samples(window, sr)
    names = {os.path.basename(r["path"].rstrip("/")): r for r in recs}
    usable = [nm for nm in names if templates.get(nm)]
    if not usable:
        raise SystemExit("no recording has measured overlap templates")
    rates = rates or {nm: float(len(templates[nm])) for nm in usable}
    split_of = split_recordings(sorted(usable), rates, val_share, seed, group_regex)
    pool_of = {s: usable_templates([t for nm in usable if split_of[nm] == s for t in templates[nm]], l_min, window)
               for s in ("train", "val")}
    w = {nm: min(len(templates[nm]), per_rec_cap) for nm in usable}
    total_w = sum(w.values())
    cfg = {"window": window, "mix": list(mix), "noise_prob": noise_prob, "margin": margin,
           "min_clip": min_clip, "subtype": subtype, "tse_mix": tse_mix, "tse_window": tse_window,
           "snap": snap, "sr": sr, "layout": asdict(lcfg)}
    jobs = []
    for k, nm in enumerate(sorted(usable)):
        cnt = max(1, round(n * w[nm] / total_w))
        jobs.append((names[nm], split_of[nm], cnt, templates[nm], pool_of[split_of[nm]], cfg,
                     seed * 100003 + k, out))
    _JOBS = jobs
    os.makedirs(out, exist_ok=True)
    rows, trows, mixes = [], [], []
    per_split = {s: Counter() for s in ("train", "val")}

    def take(job, res):
        rows.extend(res[0])
        trows.extend(res[1])
        mixes.extend(res[2])
        per_split[job[1]].update(res[3])

    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as p:
            for k, res in enumerate(p.imap(_run, range(len(jobs))), 1):
                take(jobs[k - 1], res)
                if k % 20 == 0 or k == len(jobs):
                    print(f"  rendered {k}/{len(jobs)} recordings, {len(rows)} chunks", flush=True)
    else:
        for k in range(len(jobs)):
            take(jobs[k], _run(k))
    with open(os.path.join(out, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if mixes:
        with open(os.path.join(out, "tse_mixes.jsonl"), "w", encoding="utf-8") as f:
            for r in mixes:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_manifests(out, rows, trows)
    warnings = []
    if len(pool_of["val"]) < 1000:
        warnings.append(f"val holds {len(pool_of['val'])} usable templates (< 1000)")
    report = {"window": window, "sr": sr, "l_min": l_min, "min_clip": min_clip, "margin": margin,
              "tse_mix": tse_mix, "layout": asdict(lcfg),
              "recordings": {s: sum(v == s for v in split_of.values()) for s in ("train", "val")},
              "usable_templates": {s: len(v) for s, v in pool_of.items()},
              "templates_out_of_range": sum(len(templates[nm]) for nm in usable) - sum(
                  len(v) for v in pool_of.values()),
              "multi_overlap": multi_pair_share({nm: templates[nm] for nm in usable}, merge_gap),
              "splits": {s: dict(c) for s, c in per_split.items()}, "warnings": warnings}
    with open(os.path.join(out, "render_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache")
    p.add_argument("--runs", nargs="+", default=[])
    p.add_argument("--stats", required=True, help="folder with pairs.csv from measure_real_overlap_stats.py")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20000, help="chunks in total")
    p.add_argument("--window", type=float, default=4.0,
                   help="seconds per chunk, W of the model being trained (one dataset per W; "
                        "see doc/cach_tao_du_lieu_tach_overlap.md section 8)")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--val-share", type=float, default=0.08,
                   help="share of recordings (or groups) kept for validation, stratified by event rate")
    p.add_argument("--group-regex", help="recordings whose names give the same match (group 1 if any) "
                                         "stay in one split, e.g. a show or host prefix")
    p.add_argument("--mix", default="0.7,0.2,0.1", help="real,strat,hard shares")
    p.add_argument("--noise-prob", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.3, help="nobody else may talk this close to a clean stretch")
    p.add_argument("--min-clip", type=float, default=0.25, help="shortest clean stretch kept in the pool")
    p.add_argument("--l-min", type=float, default=0.25, help="shortest overlap core used")
    p.add_argument("--merge-gap", type=float, default=1.0, help="for the multi-overlap share in render_report")
    p.add_argument("--tse-mix", default="none", choices=["none", "plain", "bss"],
                   help="TSE samples: plain = own mix without solo (design default), "
                        "bss = reuse the BSS chunk (ablation), none = no TSE")
    p.add_argument("--tse-window", type=float, help="W of the TSE model (default --window)")
    p.add_argument("--no-snap", action="store_true", help="do not move cuts to acoustic boundaries")
    p.add_argument("--subtype", default="PCM_16", choices=["PCM_16", "FLOAT"])
    p.add_argument("--exclude-index", help="index.json of collect_overlap_samples: its recordings are never used")
    p.add_argument("--files", type=int, help="use only N recordings, picked at random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = p.parse_args()
    recs = cos.stage_recordings(a.runs) + (cos.cache_recordings(a.cache) if a.cache else [])
    if a.exclude_index:
        with open(a.exclude_index, encoding="utf-8") as f:
            held = {os.path.basename(s["recording"].rstrip("/")) for s in json.load(f)["samples"]}
        recs = [r for r in recs if os.path.basename(r["path"].rstrip("/")) not in held]
        print(f"excluded {len(held)} test recording(s)", flush=True)
    if a.files and a.files < len(recs):
        recs = random.Random(a.seed).sample(recs, a.files)
    print(f"{len(recs)} recording(s)", flush=True)
    rows = build(recs, load_templates(a.stats), a.out, a.n, a.window, a.sr, a.seed, a.val_share,
                 tuple(float(x) for x in a.mix.split(",")), a.noise_prob, a.margin, a.min_clip, a.subtype,
                 a.workers, l_min=a.l_min, tse_mix=a.tse_mix, tse_window=a.tse_window,
                 rates=load_rates(a.stats) or None, group_regex=a.group_regex, merge_gap=a.merge_gap,
                 snap=not a.no_snap)
    split = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
    print(f"{len(rows)} chunk(s) -> {a.out}  {split}  ({len(rows) * a.window / 3600:.1f} h); "
          f"see {os.path.join(a.out, 'render_report.json')}")


if __name__ == "__main__":
    main()
