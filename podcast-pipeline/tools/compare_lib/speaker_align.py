"""Speaker-label comparison of two diarizations on a 10 ms grid."""

from typing import Dict, List, Optional

import numpy as np

HOP = 0.01
KINDS = ("agree", "swap", "missing", "extra", "overlap")


def label_grid(segments: Optional[List[dict]], n_frames: int) -> Dict[str, np.ndarray]:
    """label -> bool array; a frame can be active for several labels (overlap)."""
    grid: Dict[str, np.ndarray] = {}
    for seg in segments or []:
        a = max(0, int(round(seg["start"] / HOP)))
        b = min(n_frames, int(round(seg["end"] / HOP)))
        if b <= a:
            continue
        arr = grid.setdefault(str(seg["speaker"]), np.zeros(n_frames, dtype=bool))
        arr[a:b] = True
    return grid


def best_mapping(base: Dict[str, np.ndarray], var: Dict[str, np.ndarray]) -> Dict[str, Optional[str]]:
    """variant label -> baseline label, maximising total overlapped time.

    One-to-one (Hungarian); variant labels left without a partner map to None.
    """
    vl, bl = sorted(var), sorted(base)
    mapping: Dict[str, Optional[str]] = {v: None for v in vl}
    if not vl or not bl:
        return mapping
    m = np.array([[np.count_nonzero(var[v] & base[b]) for b in bl] for v in vl], dtype=float)
    pairs = []
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-m)
        pairs = list(zip(rows.tolist(), cols.tolist()))
    except Exception:
        used_r, used_c = set(), set()
        for idx in sorted(np.ndindex(m.shape), key=lambda ij: (-m[ij], ij)):
            if idx[0] in used_r or idx[1] in used_c:
                continue
            used_r.add(idx[0])
            used_c.add(idx[1])
            pairs.append(idx)
    for r, c in pairs:
        if m[r, c] > 0:
            mapping[vl[r]] = bl[c]
    return mapping


def _names(mask: int, labels: List[str]) -> Optional[str]:
    names = [labels[i] for i in range(len(labels)) if mask >> i & 1]
    return "+".join(names) if names else None


def compare_speakers(base_segs: Optional[List[dict]], var_segs: Optional[List[dict]],
                     duration: Optional[float]) -> Optional[dict]:
    """Diarization differences of a variant against the baseline.

    der_vs_baseline = (missed + false alarm + confusion) / baseline speaker-seconds,
    all in speaker-seconds, so overlapped speech counts once per speaker.
    """
    if base_segs is None or var_segs is None:
        return None
    ends = [s["end"] for s in base_segs + var_segs]
    span = max([duration or 0.0] + ends)
    n = int(np.ceil(span / HOP)) + 1
    base = label_grid(base_segs, n)
    var = label_grid(var_segs, n)
    mapping = best_mapping(base, var)

    # Common label universe: baseline labels, plus variant labels with no partner
    # (they can never agree with anything, which counts as confusion).
    labels = sorted(base) + sorted("?" + v for v, b in mapping.items() if b is None)
    idx = {l: i for i, l in enumerate(labels)}
    mb = np.zeros(n, dtype=np.int64)
    mv = np.zeros(n, dtype=np.int64)
    for l, arr in base.items():
        mb |= arr.astype(np.int64) << idx[l]
    for v, arr in var.items():
        key = mapping[v] if mapping[v] is not None else "?" + v
        mv |= arr.astype(np.int64) << idx[key]

    def popcount(x):
        c = np.zeros(x.shape, dtype=np.int64)
        for i in range(len(labels)):
            c += (x >> i) & 1
        return c

    nb, nv, nc = popcount(mb), popcount(mv), popcount(mb & mv)
    kind = np.zeros(n, dtype=np.int8)  # index into KINDS
    both = (nb > 0) & (nv > 0)
    kind[(nb > 0) & (nv == 0)] = KINDS.index("missing")
    kind[(nb == 0) & (nv > 0)] = KINDS.index("extra")
    ov_diff = both & ((nb >= 2) != (nv >= 2))
    kind[ov_diff] = KINDS.index("overlap")
    kind[both & ~ov_diff & (mb != mv)] = KINDS.index("swap")

    bad = np.flatnonzero(kind > 0)
    ranges = []
    if bad.size:
        key = np.stack([kind[bad], mb[bad], mv[bad], bad - np.arange(bad.size)])
        cuts = np.flatnonzero(np.any(key[:, 1:] != key[:, :-1], axis=0)) + 1
        for chunk in np.split(np.arange(bad.size), cuts):
            f0, f1 = int(bad[chunk[0]]), int(bad[chunk[-1]]) + 1
            ranges.append({
                "start": round(f0 * HOP, 3), "end": round(f1 * HOP, 3),
                "baseline": _names(int(mb[f0]), labels), "variant": _names(int(mv[f0]), labels),
                "kind": KINDS[int(kind[f0])]})

    ref_total = int(nb.sum())
    err = int(np.maximum(nb - nv, 0).sum() + np.maximum(nv - nb, 0).sum()
              + (np.minimum(nb, nv) - nc).sum())
    return {
        "n_speakers": sum(1 for a in var.values() if a.any()),
        "speaker_seconds": {v: round(float(a.sum()) * HOP, 3) for v, a in sorted(var.items())},
        "label_map": mapping,
        "disagree_seconds": round(float(np.count_nonzero(kind)) * HOP, 3),
        "disagree_ranges": ranges,
        "der_vs_baseline": err / ref_total if ref_total else None,
    }
