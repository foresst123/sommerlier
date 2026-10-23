"""Layout of one two-speaker separation window, shared by training data and inference.

Pure functions, integer samples, no audio I/O. The renderer of training mixtures
(tools/render_mix_dataset.py) uses it now; the inference window planner is meant to
import the same functions so that training windows and inference windows are built by
one rule (doc/cach_tao_du_lieu_tach_overlap.md, section 5).

A window of `n` samples is built around an overlap core of `core` samples:

    [pad][solo X][ctx left][CORE][ctx right][solo Y][pad]

Budget order: core -> `margin` of ctx on each side -> solo (up to `solo_max` per speaker,
none below `solo_min`) -> the rest to ctx, capped at `ctx_max` per side -> what the cap
leaves goes to extra solo. When the sources are short, `fallback` walks one chain: ctx
down to its minimum (shortfall -> extra solo) -> missing solo -> ctx -> other speaker's
solo -> silence pad; below the ctx minimum the window is dropped.

Pieces of one source stretch form a run. Where two runs meet at a slot boundary they
overlap by `fade` samples with complementary linear ramps (also when the speaker
changes, as the planner's single-channel crossfade does). A run that starts or ends at
the core edge fades entirely outside the core, so a fade never enters the core.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

INF = 1 << 60


@dataclass(frozen=True)
class LayoutConfig:
    sr: int = 16000
    margin: float = 0.25        # ctx kept on each side of the core when the room allows
    solo_max: float = 1.0       # nominal solo per speaker
    solo_min: float = 0.3       # shorter solo pieces are not used
    solo_full_at: float = 2.0   # room left after the margins at which each speaker gets solo_max
    ctx_max: float = 2.0        # ctx cap per side (= planner context_seconds)
    fade: float = 0.02          # crossfade at every junction
    l_min: float = 0.25         # shortest core used for training
    solo: bool = True           # False: TSE mix (no solo, window shrinks instead of padding)
    tse_min: float = 2.0        # TSE: shortest mix
    tse_room_min: float = 0.5   # TSE: mix at least core + this

    def n(self, seconds: float) -> int:
        return int(round(seconds * self.sr))


def window_samples(window: float, sr: int, other_rates: Sequence[int] = (24000,)) -> int:
    """round(window * sr), refusing a window that is not a whole number of samples at
    `sr` and at every rate in `other_rates` (the source rate)."""
    for r in (sr, *other_rates):
        x = window * r
        if abs(x - round(x)) > 1e-6:
            raise ValueError(f"window {window} s is not a whole number of samples at {r} Hz")
    return int(round(window * sr))


def len_bins(window: float, l_min: float) -> List[Tuple[float, float]]:
    """Strat bins of core length: l_min-0.5, 0.5-1, 1-W (last one closed at W)."""
    edges = [l_min] + [e for e in (0.5, 1.0) if l_min < e < window] + [window]
    out = [(lo, hi) for lo, hi in zip(edges, edges[1:]) if hi > lo]
    return out or [(l_min, window)]


def in_bin(x: float, lo: float, hi: float, last: bool) -> bool:
    return lo <= x <= hi + 1e-9 if last else lo <= x < hi


# ----------------------------------------------------------------------------- budget
@dataclass(frozen=True)
class Budget:
    n: int
    core: int
    ctx_left: int
    ctx_right: int
    solo: Tuple[int, int]               # (holder A, interrupter B)
    pad_left: int = 0
    pad_right: int = 0
    ctx_min: Tuple[int, int] = (0, 0)

    @property
    def total(self) -> int:
        return self.pad_left + sum(self.solo) + self.ctx_left + self.core + self.ctx_right + self.pad_right


def plan_budget(n: int, core: int, cfg: LayoutConfig = LayoutConfig()) -> Budget:
    if not 0 < core <= n:
        raise ValueError(f"core {core} outside (0, {n}]")
    room, cmax = n - core, cfg.n(cfg.ctx_max)
    if not cfg.solo:
        cl, cr = min(room // 2, cmax), min(room - room // 2, cmax)
        t_min = min(core + cl + cr, max(cfg.n(cfg.tse_min), core + cfg.n(cfg.tse_room_min)))
        need = max(0, t_min - core)
        return Budget(core + cl + cr, core, cl, cr, (0, 0), ctx_min=(need // 2, need - need // 2))
    m = cfg.n(cfg.margin)
    if room >= 2 * m:
        base_l = base_r = m
        rest = room - 2 * m
    else:
        base_l, base_r, rest = room // 2, room - room // 2, 0
    solo = 0
    if rest >= cfg.n(cfg.solo_full_at):
        solo = cfg.n(cfg.solo_max)
    elif rest >= 2 * cfg.n(cfg.solo_min):
        solo = rest // 2
    left = rest - 2 * solo
    cl, cr = base_l + left // 2, base_r + left - left // 2
    over = max(0, cl - cmax) + max(0, cr - cmax)
    cl, cr = min(cl, cmax), min(cr, cmax)
    sa = sb = solo
    pad_l = pad_r = 0
    if over and solo:
        sa, sb = solo + over // 2, solo + over - over // 2
    elif over:
        pad_l, pad_r = over // 2, over - over // 2
    return Budget(n, core, cl, cr, (sa, sb), pad_l, pad_r, (base_l, base_r))


# ----------------------------------------------------------------------------- fallback
@dataclass(frozen=True)
class Avail:
    """What the chosen sources allow, in samples. None = unlimited (silent ctx)."""
    ctx_left: Optional[int] = None
    ctx_right: Optional[int] = None
    ctx_total: Optional[int] = None     # joint limit when both ctx come from one stretch
    solo: Tuple[int, int] = (INF, INF)


def _split(amount: int, cap_a: int, cap_b: int) -> Tuple[int, int]:
    """Share `amount` between two sinks as evenly as their capacities allow."""
    a = min(cap_a, amount // 2)
    b = min(cap_b, amount - a)
    a = min(cap_a, amount - b)
    return max(0, a), max(0, b)


def fallback(b: Budget, a: Avail, cfg: LayoutConfig = LayoutConfig()):
    """Fit budget `b` to the sources. Returns (budget, reasons) or (None, drop_reason)."""
    cap_l = INF if a.ctx_left is None else a.ctx_left
    cap_r = INF if a.ctx_right is None else a.ctx_right
    cap_t = INF if a.ctx_total is None else a.ctx_total
    mn_l, mn_r = b.ctx_min
    if cap_l < mn_l or cap_r < mn_r or cap_t < mn_l + mn_r:
        return None, "ctx_short"
    reasons: List[str] = []
    cl, cr = min(b.ctx_left, cap_l), min(b.ctx_right, cap_r)
    if cl + cr > cap_t:
        t_l, t_r = _split(cl + cr - cap_t, cl - mn_l, cr - mn_r)
        cl, cr = cl - t_l, cr - t_r
    deficit = b.ctx_left + b.ctx_right - cl - cr
    if deficit:
        reasons.append("ctx_short")
    if not cfg.solo:
        out = Budget(b.core + cl + cr, b.core, cl, cr, (0, 0), ctx_min=b.ctx_min)
        return out, (["tse_shrunk"] if deficit else [])
    smin, cmax = cfg.n(cfg.solo_min), cfg.n(cfg.ctx_max)
    want = [b.solo[0] + deficit // 2, b.solo[1] + deficit - deficit // 2]
    got = [min(w, s) for w, s in zip(want, a.solo)]
    got = [g if g >= smin else 0 for g in got]
    for i, who in enumerate("AB"):
        if b.solo[i] > 0 and got[i] == 0:
            reasons.append(f"solo_missing_{who}")
        elif b.solo[i] > 0 and got[i] < b.solo[i]:
            reasons.append(f"solo_short_{who}")
    missing = sum(want) - sum(got)
    if missing:
        add_l, add_r = _split(missing, max(0, min(cmax, cap_l) - cl), max(0, min(cmax, cap_r) - cr))
        if add_l + add_r > cap_t - cl - cr:
            add_l, add_r = _split(max(0, cap_t - cl - cr), add_l, add_r)
        cl, cr, missing = cl + add_l, cr + add_r, missing - add_l - add_r
        if add_l + add_r:
            reasons.append("solo_to_ctx")
    if missing:
        spare = [a.solo[i] - got[i] if got[i] else 0 for i in range(2)]
        g0, g1 = _split(missing, max(0, spare[0]), max(0, spare[1]))
        got = [got[0] + g0, got[1] + g1]
        missing -= g0 + g1
        if g0 + g1:
            reasons.append("solo_other")
    pad_l, pad_r = b.pad_left, b.pad_right
    if missing:
        pad_l, pad_r = pad_l + missing // 2, pad_r + missing - missing // 2
        reasons.append("silence_pad")
    out = Budget(b.n, b.core, cl, cr, (got[0], got[1]), pad_l, pad_r, b.ctx_min)
    assert out.total == b.n, (out, b)
    return out, reasons


# ----------------------------------------------------------------------------- slots and runs
CTX_OWNER = {"contained": ("A", "A"), "partial": ("A", "B"), "full": (None, None)}


def make_slots(b: Budget, pattern: str, order: Tuple[str, str] = ("A", "B")) -> List[dict]:
    """Nominal slots, contiguous over [0, b.total). `order` = (speaker of solo X, of solo Y)."""
    owner_l, owner_r = CTX_OWNER[pattern]
    solo_len = {"A": b.solo[0], "B": b.solo[1]}
    seq = [("pad", None, b.pad_left), ("solo", order[0], solo_len[order[0]]),
           ("ctx", owner_l, b.ctx_left), ("core", None, b.core), ("ctx", owner_r, b.ctx_right),
           ("solo", order[1], solo_len[order[1]]), ("pad", None, b.pad_right)]
    out, t = [], 0
    for kind, who, ln in seq:
        if ln > 0:
            out.append({"kind": kind, "who": who, "start": t, "end": t + ln})
            t += ln
    return out


def core_range(slots: List[dict]) -> Tuple[int, int]:
    c = next(s for s in slots if s["kind"] == "core")
    return c["start"], c["end"]


def make_runs(slots: List[dict], pattern: str, cfg: LayoutConfig = LayoutConfig()) -> List[dict]:
    """Group slots into runs (one source stretch each) and set edge extensions and fades.

    Run: {"who", "role" (solo|core), "pieces": [{kind, who, start, end, fade_in, fade_out}],
    "start", "end"} with start/end including the extensions."""
    n = slots[-1]["end"]
    c0, c1 = core_range(slots)
    f = cfg.n(cfg.fade)
    h1, h2 = f // 2, f - f // 2
    core_i = next(i for i, s in enumerate(slots) if s["kind"] == "core")

    def ctx_of(side, who):
        j = core_i + side
        if 0 <= j < len(slots) and slots[j]["kind"] == "ctx" and slots[j]["who"] == who:
            return [slots[j]]
        return []

    core_a = {"kind": "core", "who": "A", "start": c0, "end": c1}
    core_b = {"kind": "core", "who": "B", "start": c0, "end": c1}
    groups = []
    for s in slots:
        if s["kind"] == "solo":
            groups.append(("solo", s["who"], [s]))
    if pattern == "contained":
        groups += [("core", "A", ctx_of(-1, "A") + [core_a] + ctx_of(1, "A")), ("core", "B", [core_b])]
    elif pattern == "partial":
        groups += [("core", "A", ctx_of(-1, "A") + [core_a]), ("core", "B", [core_b] + ctx_of(1, "B"))]
    else:
        groups += [("core", "A", [core_a]), ("core", "B", [core_b])]
    runs = []
    for role, who, ps in groups:
        start, end = ps[0]["start"], ps[-1]["end"]
        if start == 0:
            el, fi = 0, 0
        elif ps[0]["kind"] == "core":
            el = fi = min(f, start)
        else:
            el = min(h1, start)
            fi = el + h2
        if end == n:
            er, fo = 0, 0
        elif ps[-1]["kind"] == "core":
            er = fo = min(f, n - end)
        else:
            er = min(h2, n - end)
            fo = er + h1
        pieces = [dict(p, fade_in=0, fade_out=0) for p in ps]
        pieces[0]["start"] -= el
        pieces[0]["fade_in"] = fi
        pieces[-1]["end"] += er
        pieces[-1]["fade_out"] = fo
        runs.append({"who": who, "role": role, "pieces": pieces, "start": start - el, "end": end + er,
                     "ext": (el, er), "nominal": (start, end)})
    runs.sort(key=lambda r: (r["start"], r["who"]))
    return runs


# ----------------------------------------------------------------------------- checks
def config_of(row: dict) -> LayoutConfig:
    """The LayoutConfig a row was rendered with (stored as `layout_cfg`)."""
    return LayoutConfig(**row["layout_cfg"]) if row.get("layout_cfg") else LayoutConfig(sr=row["sr"])


def check_layout(row: dict, cfg: Optional[LayoutConfig] = None, tol: int = 2) -> List[str]:
    """Invariants of a chunks.jsonl row's `slots` / `layout` (no audio). Returns problems."""
    cfg = cfg or config_of(row)
    rid, n, sr = row.get("id", "?"), row["samples"], row["sr"]
    p: List[str] = []
    slots, pieces = row.get("slots") or [], row.get("layout") or []
    if not slots or not pieces:
        return [f"{rid}: no layout"]
    t = 0
    for s in slots:
        if s["start"] != t or s["end"] <= s["start"]:
            p.append(f"{rid}: slots not contiguous at {t}")
            break
        t = s["end"]
    if t != n:
        p.append(f"{rid}: slots cover {t} != {n} samples")
    kinds = [s["kind"] for s in slots]
    inner = kinds[:]
    while inner and inner[0] == "pad":
        inner.pop(0)
    while inner and inner[-1] == "pad":
        inner.pop()
    lead = inner[:1] == ["solo"]
    trail = inner[-1:] == ["solo"] and len(inner) > 1
    mid = inner[int(lead):len(inner) - int(trail)]
    order_ok = mid in (["core"], ["ctx", "core"], ["core", "ctx"], ["ctx", "core", "ctx"])
    if (lead or trail) and mid != ["ctx", "core", "ctx"]:
        order_ok = False
    if not order_ok:
        p.append(f"{rid}: slot order {kinds}")
    if "pad" in kinds and row.get("fallback") is None:
        p.append(f"{rid}: silence pad without fallback")
    if not cfg.solo and ("solo" in kinds or "pad" in kinds):
        p.append(f"{rid}: TSE mix holds solo or pad")
    solos = [s["who"] for s in slots if s["kind"] == "solo"]
    if len(solos) == 2 and solos[0] == solos[1]:
        p.append(f"{rid}: both solos are {solos[0]}")
    c0, c1 = row["core"]
    L = (c1 - c0) / sr
    if not cfg.l_min - 1e-6 <= L <= n / sr + 1e-6:
        p.append(f"{rid}: core {L:.3f} s outside [{cfg.l_min}, {n / sr}]")
    if abs(L - row.get("L", L)) > 1.5 / sr:
        p.append(f"{rid}: L field {row.get('L')} != core {L:.4f}")
    pat = row.get("pattern")
    ci = kinds.index("core") if "core" in kinds else -1
    if ci < 0 or (slots[ci]["start"], slots[ci]["end"]) != (c0, c1):
        p.append(f"{rid}: core slot != core field")
    for s in slots:
        if s["kind"] == "ctx" and pat in CTX_OWNER:
            side = 0 if s["end"] <= c0 else 1
            if s["who"] != CTX_OWNER[pat][side]:
                p.append(f"{rid}: ctx owner {s['who']} for {pat}")
    cores = [q for q in pieces if q["kind"] == "core"]
    if sorted(q["who"] for q in cores) != ["A", "B"]:
        p.append(f"{rid}: core pieces {[q['who'] for q in cores]}")
    f = int(round(cfg.fade * sr))
    # the pieces must be exactly what the shared rule derives from the slots: same runs,
    # extensions and fades, so every junction overlaps by one fade and no fade enters the core
    if "core" in kinds and pat in CTX_OWNER:
        fields = ("run", "kind", "who", "start", "end", "fade_in", "fade_out")
        try:
            want = [dict(q, run=k) for k, r in enumerate(make_runs(slots, pat, cfg)) for q in r["pieces"]]
        except Exception as e:      # malformed slots are reported above
            want = [{"error": str(e)}]
        got_p = sorted((tuple(q.get(k) for k in fields) for q in pieces))
        want_p = sorted((tuple(q.get(k) for k in fields) for q in want))
        if got_p != want_p:
            diff = next((g for g in got_p if g not in want_p), None) or next((w for w in want_p if w not in got_p), None)
            p.append(f"{rid}: pieces differ from the rule (make_runs) at {dict(zip(fields, diff or ()))}")
    for q in pieces:
        if q["start"] < 0 or q["end"] > n or q["end"] <= q["start"]:
            p.append(f"{rid}: piece {q['kind']}/{q['who']} outside window")
        if q["fade_in"] + q["fade_out"] > q["end"] - q["start"] or max(q["fade_in"], q["fade_out"]) > f:
            p.append(f"{rid}: bad fades on {q['kind']}/{q['who']}")
        if abs((q["end"] - q["start"]) - (q["src_end"] - q["src_start"]) * sr) > tol:
            p.append(f"{rid}: piece {q['kind']}/{q['who']} length != source length")
    for q in cores:
        if q["start"] > c0 or q["end"] < c1:
            p.append(f"{rid}: core piece of {q['who']} does not cover the core")
        if q["start"] + q["fade_in"] > c0 or q["end"] - q["fade_out"] < c1:
            p.append(f"{rid}: fade enters the core ({q['who']})")
    for i, q in enumerate(pieces):
        for r in pieces[i + 1:]:
            if q["run"] == r["run"]:
                continue
            ov = min(q["end"], r["end"]) - max(q["start"], r["start"])
            if ov > f and not (q["kind"] == r["kind"] == "core"):
                p.append(f"{rid}: {q['kind']}/{q['who']} and {r['kind']}/{r['who']} overlap {ov} > fade")
    # every voiced slot is covered by a piece of its speaker
    for s in slots:
        if s["kind"] in ("solo", "ctx") and s["who"] is not None:
            if not any(q["who"] == s["who"] and q["start"] <= s["start"] and q["end"] >= s["end"] for q in pieces):
                p.append(f"{rid}: {s['kind']} slot of {s['who']} has no piece")
    by_run: Dict[int, list] = {}
    for q in pieces:
        by_run.setdefault(q["run"], []).append(q)
    for run in by_run.values():
        run.sort(key=lambda q: q["start"])
        if len({q["who"] for q in run}) != 1:
            p.append(f"{rid}: run mixes speakers")
        for x, y in zip(run, run[1:]):
            if x["end"] != y["start"] or abs(x["src_end"] - y["src_start"]) > tol / sr:
                p.append(f"{rid}: run of {x['who']} not continuous at {x['end']}")
    per_spk: Dict[str, list] = {}
    for run in by_run.values():
        per_spk.setdefault(run[0]["who"], []).append((run[0]["src_start"], run[-1]["src_end"]))
    for who, v in per_spk.items():
        v.sort()
        for (a0, b0), (a1, b1) in zip(v, v[1:]):
            if a1 < b0 - tol / sr:
                p.append(f"{rid}: sources of {who} overlap in time")
    if row.get("fallback") is None and cfg.solo and not p:
        b = plan_budget(n, c1 - c0, cfg)
        got_solo = {s["who"]: s["end"] - s["start"] for s in slots if s["kind"] == "solo"}
        ctx_l = sum(s["end"] - s["start"] for s in slots if s["kind"] == "ctx" and s["end"] <= c0)
        ctx_r = sum(s["end"] - s["start"] for s in slots if s["kind"] == "ctx" and s["start"] >= c1)
        if (got_solo.get("A", 0), got_solo.get("B", 0)) != b.solo or (ctx_l, ctx_r) != (b.ctx_left, b.ctx_right):
            p.append(f"{rid}: slot lengths differ from the budget without a fallback")
    return p


# ----------------------------------------------------------------------------- enrollment
def pick_enrollment(clips: Sequence[Tuple[float, float]], exclude: Sequence[Tuple[float, float]],
                    anchor: float, margin: float = 0.3, k: int = 8) -> List[int]:
    """Indices of up to `k` clean clips nearest to `anchor` (seconds) whose span does not
    come within `margin` of any `exclude` range. Same rule in training and at inference."""
    ok = []
    for i, (a, b) in enumerate(clips):
        if any(a < y + margin and b > x - margin for x, y in exclude):
            continue
        ok.append((0.0 if a <= anchor <= b else min(abs(a - anchor), abs(b - anchor)), i))
    return [i for _, i in sorted(ok)[:k]]


def default_enrollment(clips: Sequence[Tuple[float, float]], order: Sequence[int],
                       max_seconds: float = 4.0, min_seconds: float = 1.0) -> Optional[List[int]]:
    """Nearest-first clips until `max_seconds`; None if they hold less than `min_seconds`."""
    out, acc = [], 0.0
    for i in order:
        if acc >= max_seconds:
            break
        out.append(i)
        acc += clips[i][1] - clips[i][0]
    return out if acc >= min_seconds - 1e-9 else None


# ----------------------------------------------------------------------------- metric
def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-9) -> float:
    est, ref = np.asarray(est, np.float64), np.asarray(ref, np.float64)
    est, ref = est - est.mean(), ref - ref.mean()
    s = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    return float(10 * np.log10((np.dot(s, s) + eps) / (np.dot(est - s, est - s) + eps)))


def core_si_sdri(est: np.ndarray, ref: np.ndarray, mix: np.ndarray, core: Tuple[int, int],
                 sr: int, pad: float = 0.1) -> float:
    """SI-SDR improvement over the core +- `pad` s only (the region that needs separating)."""
    lo, hi = max(0, core[0] - int(pad * sr)), min(len(ref), core[1] + int(pad * sr))
    return si_sdr(est[lo:hi], ref[lo:hi]) - si_sdr(mix[lo:hi], ref[lo:hi])
