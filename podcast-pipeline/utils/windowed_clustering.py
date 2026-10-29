"""Windowed Nemotron probabilities merged into global speakers, DiariZen style.

Pure numpy/scipy. The caller passes the model and the embedder as callbacks, so the logic runs
on fake data without GPU, NeMo or torch.

UNVERIFIED on GPU: the defaults (ahc_threshold, match_threshold, frame_s, active_thr) are untuned.

Per piece (about 4 min, from plan_pieces):
  1. sliding 30 s windows (hop 10 s); each gives frame probabilities, one column per local label
  2. clean speech of a label (no other label active) -> mean WeSpeaker embedding per window label
  3. constrained average-linkage clustering (two labels of one window never share a cluster),
     then per-window Hungarian reassignment against the centroids
  4. frame probability per cluster (a window that does not see a cluster counts as 0), median
     filter, hysteresis, then the short-segment and padding rules
Neighbouring pieces share 30 s of audio and are stitched by Hungarian matching of their centroids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment

ProbFn = Callable[[float, float], np.ndarray]   # (start, end) piece-relative s -> (frames, labels)
EmbedFn = Callable[[list], np.ndarray]          # [(start, end)] piece-relative s -> (n, dim)

# Same values as utils/nemotron_segments.POSTPROCESSING_BASE (a test checks them). Copied because
# that module cannot be imported by name inside NeMo's environment.
POSTPROCESSING = {"onset": 0.53, "offset": 0.49, "pad_onset": 0.23, "pad_offset": 0.01,
                  "min_duration_on": 0.42, "min_duration_off": 0.34}


@dataclass(frozen=True)
class Piece:
    """core_* is the span this piece's output keeps; audio_* is the audio the model sees."""

    core_start: float
    core_end: float
    audio_start: float
    audio_end: float

    @property
    def audio_duration(self) -> float:
        return self.audio_end - self.audio_start


@dataclass
class WindowFit:
    """One window after clustering. Arrays are indexed by the original local column."""

    start: float
    end: float
    cluster_of: np.ndarray   # (labels,) global cluster id, -1 when the column was dropped
    embs: np.ndarray         # (labels, dim) unit vectors, zero rows without an embedding
    has_emb: np.ndarray      # (labels,) bool


@dataclass
class PieceResult:
    segments: list           # piece-relative {"start", "end", "cluster"}
    windows: list            # WindowFit per window
    n_clusters: int
    stats: dict


@dataclass
class _Window:
    start: float
    end: float
    f0: int                  # first frame index on the piece's frame grid
    n_frames: int
    n_orig: int              # local columns before dropping silent ones
    cols: np.ndarray         # original column index of each kept label
    probs: np.ndarray        # (n_frames, kept)


def piece_bounds(piece_s: float) -> tuple[float, float, float]:
    """(target, min, max) seconds for a piece length; 240 gives the 180/240/300 of the design."""
    return float(piece_s), 0.75 * float(piece_s), 1.25 * float(piece_s)


def _default_vad_fn():
    from utils.nemotron_two_pass import build_vad_windows
    return build_vad_windows


def plan_pieces(duration: float, speech, *, target_s: float = 240.0, min_s: float = 180.0,
                max_s: float = 300.0, overlap_s: float = 30.0, vad_fn: Optional[Callable] = None) -> list[Piece]:
    """Cut [0, duration] at silences near target_s; neighbours share overlap_s seconds of audio.

    vad_fn is utils.nemotron_two_pass.build_vad_windows; it is imported by default, but the NeMo-side
    script passes it in because `utils` there is NeMo's package.
    """
    fn = vad_fn or _default_vad_fn()
    windows = fn(duration, speech, target_seconds=target_s, min_seconds=min_s, max_seconds=max_s,
                 context_seconds=overlap_s / 2.0)
    return [Piece(w.core_start, w.core_end, w.context_start, w.context_end) for w in windows]


def window_spans(duration: float, window_s: float = 30.0, hop_s: float = 10.0) -> list[tuple[float, float]]:
    """Sliding windows over [0, duration]; the last window ends exactly at duration."""
    if duration <= 0:
        return []
    if duration <= window_s:
        return [(0.0, float(duration))]
    last = duration - window_s
    starts, s = [], 0.0
    while s < last - 1e-9:
        starts.append(s)
        s += hop_s
    starts.append(last)
    return [(round(x, 6), round(x + window_s, 6)) for x in starts]


def assign_labels(sim: np.ndarray) -> np.ndarray:
    """Give each row (a local label) its own column (a cluster), maximising total similarity.

    Hungarian, so a greedy pick cannot steal a column another row needs. Rows left over when there
    are more rows than columns take their best column (the caller counts that as a violation).
    """
    sim = np.asarray(sim, dtype=float)
    if sim.shape[1] == 0:
        return np.full(sim.shape[0], -1, dtype=int)
    out = np.argmax(sim, axis=1).astype(int)
    rows, cols = linear_sum_assignment(-sim)
    out[rows] = cols
    return out


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def _unit_strict(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(rows).all() or (norms <= 0).any():
        raise ValueError("embedding is zero or not finite")
    return rows / norms


def constrained_ahc(x: np.ndarray, win_of: np.ndarray, threshold: float,
                    min_k: int = 1, max_k: int = 8) -> np.ndarray:
    """Average-linkage on cosine distance. Two clusters that share a window never merge.

    Merging stops when the closest pair is at least `threshold` apart and the count is within
    max_k, or when the count reaches min_k. Above max_k it keeps merging the closest valid pair.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    labels = np.zeros(n, dtype=int)
    if n == 0:
        return labels
    win_of = np.asarray(win_of, dtype=int)
    sums = x.copy()
    counts = np.ones(n)
    members = [[i] for i in range(n)]
    inc = np.zeros((n, int(win_of.max()) + 1), dtype=np.int32)
    inc[np.arange(n), win_of] = 1
    while len(members) > max(min_k, 1):
        dist = 1.0 - (sums @ sums.T) / np.outer(counts, counts)
        dist[(inc @ inc.T) > 0] = np.inf
        np.fill_diagonal(dist, np.inf)
        i, j = (int(v) for v in np.unravel_index(int(np.argmin(dist)), dist.shape))
        d = dist[i, j]
        if not np.isfinite(d) or (d >= threshold and len(members) <= max_k):
            break
        i, j = min(i, j), max(i, j)
        sums[i] += sums[j]
        counts[i] += counts[j]
        inc[i] = np.maximum(inc[i], inc[j])
        members[i].extend(members[j])
        del members[j]
        sums = np.delete(sums, j, axis=0)
        counts = np.delete(counts, j)
        inc = np.delete(inc, j, axis=0)
    for c, mem in enumerate(members):
        labels[mem] = c
    return labels


def make_vbx_cluster_fn(cluster_vbx: Callable, xvec_tf: Callable, plda_tf: Callable, plda_psi, *,
                        ahc_threshold: float = 0.6, Fa: float = 0.07, Fb: float = 0.8, lda_dim: int = 128,
                        max_iters: int = 20, init_smoothing: float = 7.0) -> Callable:
    """cluster_fn for run_piece that runs VBx the way pyannote's VBxClustering (used by DiariZen) does.

    cluster_vbx, xvec_tf, plda_tf and plda_psi are diarizen.clustering.VBx's cluster_vbx and vbx_setup
    outputs, passed in so this module does not import diarizen. The AHC init is pyannote's: centroid
    linkage on the unit embeddings cut at ahc_threshold (euclidean). VBx is GMM-style (loop_prob 0), so the
    embeddings are an unordered set. A speaker is kept when its prior is above 1e-7; each embedding takes
    its largest responsibility among the kept speakers.

    run_piece hands it the raw (unnormalised) mean embeddings as `raw`: AHC runs on the unit vectors, the
    PLDA transform on the raw ones, as in pyannote.

    VBx does not enforce cannot-link: run_piece's per-window Hungarian reassignment does.
    If VBx keeps fewer than min_k or more than max_k speakers, the piece falls back to constrained_ahc with the cosine
    threshold ahc_threshold**2 / 2 (for unit vectors, euclidean d means cosine distance d**2 / 2).
    Each call sets cluster_fn.last_info (vbx_fallback, embeddings, ahc_init_clusters, vbx_clusters and the largest
    VBx priors), which run_piece copies into its stats: it says why a piece ended with one speaker.
    """
    phi = np.asarray(plda_psi, dtype=float)[:lda_dim]

    def cluster_fn(x, win_of, min_k, max_k, raw=None):
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n < 2:
            cluster_fn.last_info = {"vbx_fallback": False, "embeddings": n, "vbx_clusters": min(n, 1)}
            return np.zeros(n, dtype=int)
        normed = _unit(x)
        ahc = fcluster(linkage(normed, method="centroid", metric="euclidean"), ahc_threshold,
                       criterion="distance") - 1
        _, ahc_init = np.unique(ahc, return_inverse=True)
        # The PLDA transform subtracts a mean learned on the embedder's raw vectors, so it takes those.
        fea = plda_tf(xvec_tf(x if raw is None else np.asarray(raw, dtype=float)), lda_dim)
        q, sp = cluster_vbx(ahc_init, fea, phi, Fa=Fa, Fb=Fb, maxIters=max_iters, init_smoothing=init_smoothing)
        q, sp = np.asarray(q, dtype=float), np.asarray(sp, dtype=float)
        keep = np.flatnonzero(sp > 1e-7)
        if keep.size == 0:
            keep = np.array([int(np.argmax(sp))])
        hard = keep[np.argmax(q[:, keep], axis=1)]
        _, labels = np.unique(hard, return_inverse=True)
        k_vbx = int(labels.max()) + 1
        info = {"vbx_fallback": False, "embeddings": n, "ahc_init_clusters": int(ahc_init.max()) + 1,
                "vbx_clusters": k_vbx, "vbx_priors_top": [round(float(v), 4) for v in np.sort(sp)[::-1][:5]]}
        if k_vbx > max_k or k_vbx < min_k:
            info["vbx_fallback"] = True
            cluster_fn.last_info = info
            return constrained_ahc(normed, win_of, ahc_threshold ** 2 / 2.0, min_k, max_k)
        cluster_fn.last_info = info
        return labels.astype(int)

    cluster_fn.last_info = {}
    cluster_fn.wants_raw = True
    return cluster_fn


def _centroids(x: np.ndarray, labels: np.ndarray, k: int, prev: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.zeros((k, x.shape[1]))
    for c in range(k):
        m = labels == c
        if m.any():
            out[c] = _unit(x[m].sum(axis=0))
        elif prev is not None:
            out[c] = prev[c]
    return out


def _window_assign(x: np.ndarray, win_of: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    labels = np.full(len(x), -1, dtype=int)
    for w in np.unique(win_of):
        rows = np.flatnonzero(win_of == w)
        labels[rows] = assign_labels(x[rows] @ centroids.T)
    return labels


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs of True values."""
    padded = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _split_run(t0: float, t1: float, max_s: float = 3.0) -> list[tuple[float, float]]:
    """Equal parts of at most max_s; a run longer than max_s gives parts of at least 1.5 s."""
    d = t1 - t0
    if d <= max_s:
        return [(t0, t1)]
    n = int(math.ceil(d / max_s - 1e-9))
    step = d / n
    return [(t0 + i * step, t0 + (i + 1) * step) for i in range(n)]


def _trim_crops(crops, quantum: float, min_len: float) -> list[tuple[float, float]]:
    """Each crop cut down to a multiple of quantum seconds; crops shorter than min_len are dropped."""
    out = []
    for t0, t1 in crops:
        d = t1 - t0
        if quantum > 0:
            d = math.floor(d / quantum + 1e-9) * quantum
        if d >= min_len - 1e-9 and d > 0:
            out.append((t0, t0 + d))
    return out


def _fit_rows(raw: np.ndarray, n: int) -> np.ndarray:
    if raw.shape[0] >= n:
        return raw[:n]
    return np.vstack([raw, np.zeros((n - raw.shape[0], raw.shape[1]))])


def _cluster_frames(recs: Sequence[_Window], cluster_of: Sequence[np.ndarray], k: int,
                    n_frames: int) -> np.ndarray:
    """Mean probability per cluster and frame; a window that lacks a cluster contributes 0."""
    total = np.zeros((k, n_frames))
    cover = np.zeros(n_frames)
    for rec, cl in zip(recs, cluster_of):
        lo, hi = max(0, rec.f0), min(n_frames, rec.f0 + rec.n_frames)
        if hi <= lo or k == 0:
            continue
        a, b = lo - rec.f0, hi - rec.f0
        block = np.zeros((hi - lo, k))
        for j, c in enumerate(cl):
            if c >= 0:
                block[:, c] = np.maximum(block[:, c], rec.probs[a:b, j])
        total[:, lo:hi] += block.T
        cover[lo:hi] += 1
    return total / np.maximum(cover, 1.0)


def _hysteresis(x: np.ndarray, onset: float, offset: float) -> np.ndarray:
    out = np.zeros(len(x), dtype=bool)
    on = False
    for i, v in enumerate(x):
        on = (v >= offset) if on else (v >= onset)
        out[i] = on
    return out


def _postprocess(spans: list[tuple[float, float]], duration: float, post: dict) -> list[tuple[float, float]]:
    """Bridge short gaps, drop short segments, then pad the edges. The order is ours, not NeMo's."""
    merged: list[list[float]] = []
    for s, e in sorted(spans):
        if merged and s - merged[-1][1] < post["min_duration_off"]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    kept = [(s, e) for s, e in merged if e - s >= post["min_duration_on"]]
    return [(max(0.0, s - post["pad_onset"]), min(duration, e + post["pad_offset"])) for s, e in kept]


def run_piece(duration: float, prob_fn: ProbFn, embed_fn: EmbedFn, *,
              window_s: float = 30.0, hop_s: float = 10.0, frame_s: float = 0.08,
              active_thr: float = 0.5, min_clean_s: float = 1.0,
              ahc_threshold: float = 0.5, min_speakers: int = 1, max_speakers: int = 8,
              reassign_iters: int = 2, median_frames: int = 5,
              post: Optional[dict] = None,
              cluster_fn: Optional[Callable] = None,
              max_embed_s: float = 6.0, min_crop_s: float = 0.5,
              crop_quantum_s: float = 0.25) -> PieceResult:
    """Cluster one piece's windows into global speakers and binarise them into segments.

    max_embed_s caps the clean speech embedded per (window, label): the longest crops are kept until
    that many seconds are reached (0 means no cap). One embedding needs a few seconds, not all of them.
    Crops are trimmed down to a multiple of crop_quantum_s (so an embedder sees few distinct lengths) and
    crops under min_crop_s are dropped: very short audio gives unusable embeddings. A row of embed_fn that
    is empty or not finite is dropped too (stats["invalid_crops"]); its label may then end up weak.

    cluster_fn(x, win_of, min_k, max_k) -> labels replaces constrained_ahc when given (arm G uses VBx).
    The Hungarian reassignment below runs either way, so it enforces cannot-link for any cluster_fn.
    """
    post = dict(POSTPROCESSING, **(post or {}))
    spans = window_spans(duration, window_s, hop_s)
    n_frames = int(math.ceil(duration / frame_s - 1e-9))

    recs: list[_Window] = []
    for s, e in spans:
        n_w = max(1, int(round((e - s) / frame_s)))
        raw = np.asarray(prob_fn(s, e), dtype=float)
        if raw.ndim != 2:
            raise ValueError(f"prob_fn must return (frames, labels), got shape {raw.shape}")
        raw = _fit_rows(raw, n_w)
        keep = np.flatnonzero(raw.max(axis=0) >= active_thr) if raw.shape[1] else np.zeros(0, int)
        recs.append(_Window(s, e, int(round(s / frame_s)), n_w, raw.shape[1], keep, raw[:, keep]))

    # Clean speech per label -> embedding. A label with too little clean speech gets none (weak).
    requests = []
    for w, rec in enumerate(recs):
        act = rec.probs >= active_thr
        n_act = act.sum(axis=1)
        for j in range(act.shape[1]):
            clean = act[:, j] & ((n_act - act[:, j].astype(int)) == 0)
            runs = _runs(clean)
            if sum(b - a for a, b in runs) * frame_s < min_clean_s:
                continue
            crops = [crop for a, b in runs
                     for crop in _split_run((rec.f0 + a) * frame_s, (rec.f0 + b) * frame_s)]
            crops = _trim_crops(crops, crop_quantum_s, min_crop_s)
            if max_embed_s > 0:
                kept, total = [], 0.0
                for crop in sorted(crops, key=lambda c: c[0] - c[1]):    # longest first
                    if total >= max_embed_s:
                        break
                    kept.append(crop)
                    total += crop[1] - crop[0]
                crops = sorted(kept)
            for t0, t1 in crops:
                requests.append((w, j, t0, t1))

    emb: dict[tuple[int, int], np.ndarray] = {}
    emb_raw: dict[tuple[int, int], np.ndarray] = {}     # mean of the embedder's own vectors, norm kept
    dim = 0
    invalid_crops = 0
    if requests:
        raw = np.asarray(embed_fn([(t0, t1) for _, _, t0, t1 in requests]), dtype=float)
        if len(raw) != len(requests):
            raise ValueError("embed_fn returned a different number of rows")
        raw = raw.reshape(len(requests), -1)
        valid = np.isfinite(raw).all(axis=1) & (np.linalg.norm(raw, axis=1) > 0)
        invalid_crops = int((~valid).sum())
        grouped: dict[tuple[int, int], list] = {}
        grouped_raw: dict[tuple[int, int], list] = {}
        if valid.any():
            kept_requests = [r for r, ok in zip(requests, valid) if ok]
            raw = raw[valid]
            vecs = _unit_strict(raw)
            dim = vecs.shape[1]
            for (w, j, _, _), v, r in zip(kept_requests, vecs, raw):
                grouped.setdefault((w, j), []).append(v)
                grouped_raw.setdefault((w, j), []).append(r)
        for key, vs in grouped.items():
            emb[key] = _unit(np.mean(vs, axis=0))
            emb_raw[key] = np.mean(grouped_raw[key], axis=0)

    # Global clustering of the strong (embedded) labels.
    item_keys = sorted(emb)
    x = np.array([emb[k] for k in item_keys]).reshape(len(item_keys), dim)
    win_of = np.array([k[0] for k in item_keys], dtype=int)
    cluster_stats: dict = {}
    if not len(item_keys):
        labels = np.zeros(0, dtype=int)
    elif cluster_fn is None:
        labels = constrained_ahc(x, win_of, ahc_threshold, min_speakers, max_speakers)
    else:
        extra = {}
        if getattr(cluster_fn, "wants_raw", False):
            extra["raw"] = np.array([emb_raw[k] for k in item_keys]).reshape(len(item_keys), dim)
        labels = np.asarray(cluster_fn(x, win_of, min_speakers, max_speakers, **extra), dtype=int)
        if labels.shape != (len(item_keys),):
            raise ValueError("cluster_fn must return one label per embedded label")
        cluster_stats = dict(getattr(cluster_fn, "last_info", None) or {})
        cluster_stats["vbx_fallback"] = bool(cluster_stats.get("vbx_fallback", False))
    if len(item_keys):
        k0 = int(labels.max()) + 1
        cent = _centroids(x, labels, k0)
        for _ in range(reassign_iters):
            labels = _window_assign(x, win_of, cent)
            cent = _centroids(x, labels, k0, prev=cent)
        labels = _window_assign(x, win_of, cent)
    used = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(used)}
    k = len(used)

    cluster_of = [np.full(len(rec.cols), -1, dtype=int) for rec in recs]
    for (w, j), lab in zip(item_keys, labels):
        cluster_of[w][j] = remap[int(lab)]

    # Weak labels (no embedding): nearest cluster by overlap with the strong clusters' frame
    # probabilities, never one already used by another label of the same window.
    weak = fallback = 0
    if any((cl < 0).any() for cl in cluster_of):
        if k == 0:
            k = 1
            fallback += 1
        p0 = _cluster_frames(recs, cluster_of, k, n_frames)
        for w, rec in enumerate(recs):
            taken = {int(c) for c in cluster_of[w] if c >= 0}
            for j in range(len(rec.cols)):
                if cluster_of[w][j] >= 0:
                    continue
                lo, hi = rec.f0, min(n_frames, rec.f0 + rec.n_frames)
                if hi <= lo:
                    continue
                score = p0[:, lo:hi] @ rec.probs[: hi - lo, j]
                free = [c for c in range(k) if c not in taken]
                if free:
                    best = max(free, key=lambda c: (score[c], -c))
                else:
                    best = int(np.argmax(score))
                    fallback += 1
                cluster_of[w][j] = best
                taken.add(best)
                weak += 1

    # Frame-level output: smoothing, hysteresis, then the segment rules.
    frame_p = _cluster_frames(recs, cluster_of, k, n_frames)
    segments = []
    if k:
        smooth = median_filter(frame_p, size=(1, median_frames), mode="nearest") \
            if median_frames > 1 else frame_p
        for c in range(k):
            on = _hysteresis(smooth[c], post["onset"], post["offset"])
            raw_spans = [(a * frame_s, min(b * frame_s, duration)) for a, b in _runs(on)]
            for s, e in _postprocess(raw_spans, duration, post):
                segments.append({"start": s, "end": e, "cluster": c})
    segments.sort(key=lambda r: (r["start"], r["end"]))

    windows = []
    for w, rec in enumerate(recs):
        full_cl = np.full(rec.n_orig, -1, dtype=int)
        full_emb = np.zeros((rec.n_orig, dim))
        full_has = np.zeros(rec.n_orig, dtype=bool)
        for j, o in enumerate(rec.cols):
            full_cl[o] = cluster_of[w][j]
            if (w, j) in emb:
                full_emb[o] = emb[(w, j)]
                full_has[o] = True
        windows.append(WindowFit(rec.start, rec.end, full_cl, full_emb, full_has))

    violations = 0
    for cl in cluster_of:
        assigned = cl[cl >= 0]
        violations += len(assigned) - len(set(assigned.tolist()))
    stats = {"windows": len(recs), "strong_labels": len(item_keys), "weak_labels": weak,
             "fallback_assignments": fallback, "cannot_link_violations": violations,
             "embedded_requests": len(requests), "invalid_crops": invalid_crops, **cluster_stats}
    return PieceResult(segments=segments, windows=windows, n_clusters=k, stats=stats)


def _overlap_centroids(res: PieceResult, t0: float, t1: float) -> dict[int, np.ndarray]:
    """Unit centroid per cluster from the embedded labels of windows that touch [t0, t1)."""
    acc: dict[int, np.ndarray] = {}
    for w in res.windows:
        if w.end <= t0 or w.start >= t1:
            continue
        for j in np.flatnonzero(w.has_emb):
            c = int(w.cluster_of[j])
            if c >= 0:
                acc[c] = acc[c] + w.embs[j] if c in acc else w.embs[j].copy()
    return {c: _unit(v) for c, v in acc.items() if np.linalg.norm(v) > 0}


def _match(prev_c: dict[int, np.ndarray], prev_map: dict[int, int], cur_c: dict[int, np.ndarray],
           threshold: float) -> dict[int, int]:
    """Current cluster -> global id, one-to-one by centroid cosine; pairs under threshold stay new."""
    if not prev_c or not cur_c:
        return {}
    pk, ck = sorted(prev_c), sorted(cur_c)
    sim = np.array([[float(prev_c[p] @ cur_c[c]) for c in ck] for p in pk])
    rows, cols = linear_sum_assignment(-sim)
    return {ck[c]: prev_map[pk[r]] for r, c in zip(rows, cols) if sim[r, c] >= threshold}


def stitch_pieces(pieces: Sequence[Piece], results: Sequence[PieceResult], *,
                  match_threshold: float = 0.5) -> list[dict]:
    """Absolute-time segments with global SPEAKER_NN labels; each piece keeps only its core."""
    if len(pieces) != len(results):
        raise ValueError("pieces and results differ in length")
    out: list[dict] = []
    next_id = 0
    prev_map: dict[int, int] = {}
    for i, (piece, res) in enumerate(zip(pieces, results)):
        gmap: dict[int, int] = {}
        if i > 0:
            prev_piece, prev_res = pieces[i - 1], results[i - 1]
            lo, hi = piece.audio_start, prev_piece.audio_end
            if hi > lo:
                prev_c = _overlap_centroids(prev_res, lo - prev_piece.audio_start, hi - prev_piece.audio_start)
                cur_c = _overlap_centroids(res, 0.0, hi - piece.audio_start)
                gmap = _match(prev_c, prev_map, cur_c, match_threshold)
        for c in range(res.n_clusters):
            if c not in gmap:
                gmap[c] = next_id
                next_id += 1
        prev_map = gmap
        for seg in res.segments:
            s = max(piece.core_start, piece.audio_start + seg["start"])
            e = min(piece.core_end, piece.audio_start + seg["end"])
            if e > s:
                out.append({"start": s, "end": e, "speaker": f"SPEAKER_{gmap[seg['cluster']]:02d}"})
    out.sort(key=lambda r: (r["start"], r["end"]))
    merged: list[dict] = []
    for seg in out:
        if merged and merged[-1]["speaker"] == seg["speaker"] and seg["start"] - merged[-1]["end"] <= 1e-6:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return [{"start": round(r["start"], 3), "end": round(r["end"], 3), "speaker": r["speaker"]}
            for r in merged]


def save_piece_probs(path, piece: Piece, spans: Sequence[tuple[float, float]], probs_list: Sequence,
                     frame_s: float) -> None:
    """One piece's window probabilities to a compressed .npz (float64, so the round trip is exact)."""
    if len(spans) != len(probs_list):
        raise ValueError("spans and probs_list differ in length")
    arrays = {f"p{i}": np.asarray(p, dtype=float) for i, p in enumerate(probs_list)}
    np.savez_compressed(
        path,
        piece=np.array([piece.core_start, piece.core_end, piece.audio_start, piece.audio_end], dtype=float),
        spans=np.asarray(spans, dtype=float).reshape(-1, 2),
        frame_s=np.array(float(frame_s)),
        **arrays)


def load_piece_probs(path) -> tuple[Piece, list, list, float]:
    """Inverse of save_piece_probs: (piece, spans, probs_list, frame_s)."""
    with np.load(path) as z:
        c0, c1, a0, a1 = (float(v) for v in z["piece"])
        spans = [(float(s), float(e)) for s, e in z["spans"]]
        probs = [np.asarray(z[f"p{i}"], dtype=float) for i in range(len(spans))]
        frame_s = float(z["frame_s"])
    return Piece(c0, c1, a0, a1), spans, probs, frame_s


def run_file(duration: float, speech, *, make_prob_fn: Callable[[Piece], ProbFn],
             make_embed_fn: Callable[[Piece], EmbedFn], piece_s: float = 240.0, window_s: float = 30.0,
             hop_s: float = 10.0, overlap_s: float = 30.0, match_threshold: float = 0.5,
             vad_fn: Optional[Callable] = None, pieces: Optional[Sequence[Piece]] = None,
             piece_fn: Optional[Callable] = None, **piece_kwargs) -> dict:
    """Whole-file run: plan pieces, run_piece on each, stitch. piece_kwargs go to run_piece.

    pieces, when given, replaces the planning (speech and vad_fn are then unused). piece_fn replaces run_piece
    (same call shape; arm J passes utils.dz_pipeline.run_piece_dz).
    """
    if pieces is None:
        target, lo, hi = piece_bounds(piece_s)
        if speech is None:
            speech = [(0.0, float(duration))]
        pieces = plan_pieces(duration, speech, target_s=target, min_s=lo, max_s=hi, overlap_s=overlap_s,
                             vad_fn=vad_fn)
    pieces = list(pieces)
    run = piece_fn or run_piece
    results = [run(p.audio_duration, make_prob_fn(p), make_embed_fn(p), window_s=window_s,
                   hop_s=hop_s, **piece_kwargs) for p in pieces]
    segments = stitch_pieces(pieces, results, match_threshold=match_threshold)
    return {
        "segments": segments,
        "pieces": [{"core": [p.core_start, p.core_end], "audio": [p.audio_start, p.audio_end],
                    "clusters": r.n_clusters, "stats": r.stats} for p, r in zip(pieces, results)],
    }
