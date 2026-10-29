"""DiariZen's pipeline with Nemotron as the local segmentation (arm J).

Pure numpy/scipy; the embedder and VBx come in as callbacks. Every step follows DiariZen's code (BUTSpeechFIT/DiariZen:
diarizen/pipelines/inference.py and its pyannote-audio fork: speaker_diarization.py, clustering.py, utils/diarization.py,
core/inference.py, utils/signal.py), except the local segmentation, which is Nemotron's sigmoid output per slot
(DiariZen's is a 4-speaker powerset):

  1. each window's slot probabilities are binarised at `thr`, then median-filtered along time (DiariZen: size 11 frames
     on its own 20 ms frames; here `median_frames`);
  2. per-frame speaker count = round(mean over the windows covering the frame of the number of active slots);
  3. one embedding per (window, active slot): the WeSpeaker ResNet sees the whole window and `weights` = the frames where
     only that slot is active (the whole slot mask when it has no more than `min_num_frames` such frames);
  4. clustering set = embeddings with at least `min_frames_ratio` of the window's frames active alone (all active ones
     when fewer than two qualify); AHC (centroid linkage, euclid on unit vectors, cut at ahc_threshold), then VBx;
  5. centroids = VBx responsibilities x raw embeddings; every (window, slot) scores `2 - cosine distance` to each centroid
     and a Hungarian assignment per window gives each slot its own cluster; the clusters are renumbered with np.unique
     as DiariZen does (a slot left without a cluster ends up in the shifted id 0);
  6. per window and cluster the activation is the max over its slots; clusters are summed over the windows, and each frame
     keeps the `count` clusters with the largest sums; the result is cut where it crosses 0.5 at frame middles (pyannote's
     Binarize with onset = offset = 0.5, no padding, no minimum durations).

Unlike DiariZen, `min_speakers` and `max_speakers` do not touch the clustering (its VBx ignores them); `max_speakers`
only caps the per-frame count. Not reproduced: a count above the number of clusters (DiariZen pads extra empty speakers).

UNVERIFIED on a GPU: only fake probabilities and embeddings, and the reference VBx.py, were run.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment

from utils.windowed_clustering import PieceResult, WindowFit, _fit_rows, window_spans

EmbedMaskFn = Callable[[tuple, np.ndarray], Optional[np.ndarray]]   # ((start, end), frame mask) -> vector or None
VbxFn = Callable[[np.ndarray], tuple]                               # train embeddings -> (q, sp)


def binarise_window(raw: np.ndarray, thr: float = 0.5, median_frames: int = 3) -> np.ndarray:
    """Slot probabilities (frames, slots) -> 0/1, median-filtered along time (DiariZen: size (1, 11, 1), reflect)."""
    binary = (np.asarray(raw, dtype=float) >= thr).astype(float)
    if median_frames > 1 and binary.size:
        binary = median_filter(binary, size=(median_frames, 1), mode="reflect")
    return binary


def speaker_count(placed: list, n_frames: int, max_speakers: int = 20) -> np.ndarray:
    """Per-frame count: mean over the covering windows of the active slots, rounded (warm_up 0, no Hamming window)."""
    total = np.zeros(n_frames)
    cover = np.zeros(n_frames)
    for f0, binary in placed:
        hi = min(n_frames, f0 + len(binary))
        if hi <= f0:
            continue
        total[f0:hi] += binary[: hi - f0].sum(axis=1)
        cover[f0:hi] += 1
    count = np.rint(total / np.maximum(cover, 1e-12))
    return np.minimum(count, max_speakers).astype(int)


def make_dz_vbx_fn(cluster_vbx: Callable, xvec_tf: Callable, plda_tf: Callable, plda_psi, *,
                   ahc_threshold: float = 0.6, ahc_criterion: str = "distance", Fa: float = 0.07, Fb: float = 0.8,
                   lda_dim: int = 128, max_iters: int = 20) -> VbxFn:
    """train embeddings (n, dim, raw) -> (q, sp) exactly as pyannote's VBxClustering.__call__ computes them.

    The VBx functions are diarizen.clustering.VBx's cluster_vbx and vbx_setup outputs. fn.last_info keeps the counts.
    """
    phi = np.asarray(plda_psi, dtype=float)[:lda_dim]

    def fn(train):
        train = np.asarray(train, dtype=float)
        normed = train / np.linalg.norm(train, axis=1, keepdims=True)
        ahc = fcluster(linkage(normed, method="centroid", metric="euclidean"), ahc_threshold,
                       criterion=ahc_criterion) - 1
        _, ahc = np.unique(ahc, return_inverse=True)
        fea = plda_tf(xvec_tf(train), lda_dim=lda_dim)
        q, sp = cluster_vbx(ahc, fea, phi, Fa=Fa, Fb=Fb, maxIters=max_iters)
        q, sp = np.asarray(q, dtype=float), np.asarray(sp, dtype=float)
        fn.last_info = {"train_embeddings": int(len(train)), "ahc_init_clusters": int(ahc.max()) + 1,
                        "vbx_clusters": int((sp > 1e-7).sum()),
                        "vbx_priors_top": [round(float(v), 4) for v in np.sort(sp)[::-1][:5]]}
        return q, sp

    fn.last_info = {}
    return fn


def constrained_argmax(soft: np.ndarray) -> np.ndarray:
    """soft (windows, slots, clusters) -> hard (windows, slots); -2 for a slot that gets no cluster.

    nan scores take the smallest score, then each window is a Hungarian assignment that maximises the total score.
    """
    if not np.isfinite(soft).any():
        return -2 * np.ones(soft.shape[:2], dtype=int)
    soft = np.nan_to_num(soft, nan=float(np.nanmin(soft)))
    hard = -2 * np.ones(soft.shape[:2], dtype=int)
    for w, cost in enumerate(soft):
        slots, clusters = linear_sum_assignment(cost, maximize=True)
        hard[w, slots] = clusters
    return hard


def binarize_frames(values: np.ndarray, frame_s: float, offset_s: float = 0.0, onset: float = 0.5,
                    offset: float = 0.5) -> list[tuple[float, float]]:
    """pyannote's Binarize on one 0/1 track: regions run between frame middles; no padding, no minimum durations."""
    n = len(values)
    if n == 0:
        return []
    mids = [offset_s + (i + 0.5) * frame_s for i in range(n)]
    out = []
    start, active = mids[0], values[0] > onset
    t = mids[0]
    for t, y in zip(mids[1:], values[1:]):
        if active:
            if y < offset:
                out.append((start, t))
                start, active = t, False
        elif y > onset:
            start, active = t, True
    if active:
        out.append((start, t))
    return out


def run_piece_dz(duration: float, prob_fn: Callable, embed_fn: EmbedMaskFn, *, vbx_fn: VbxFn,
                 window_s: float = 16.0, hop_s: float = 0.8, frame_s: float = 0.08, thr: float = 0.5,
                 median_frames: int = 3, min_frames_ratio: float = 0.1, min_num_frames: int = 1,
                 max_speakers: int = 20, drop_empty_clusters: bool = True) -> PieceResult:
    """One piece through DiariZen's steps (see the module docstring). Returns a PieceResult like run_piece's.

    drop_empty_clusters: np.unique's shift leaves DiariZen with an empty cluster 0 whenever a slot got no cluster, and
    a frame whose count exceeds its non-empty clusters can then pick that empty one (a ghost speaker). True renumbers
    the clusters that are used; this is the one deliberate deviation from DiariZen. False keeps its ids.
    """
    spans = window_spans(duration, window_s, hop_s)
    n_frames = int(math.ceil(duration / frame_s - 1e-9))

    # 1. binarised local segmentation per window
    wins = []        # (span, f0, binary (frames, slots))
    for s, e in spans:
        n_w = max(1, int(round((e - s) / frame_s)))
        raw = np.asarray(prob_fn(s, e), dtype=float)
        if raw.ndim != 2:
            raise ValueError(f"prob_fn must return (frames, slots), got shape {raw.shape}")
        wins.append(((s, e), int(round(s / frame_s)), binarise_window(_fit_rows(raw, n_w), thr, median_frames)))
    placed = [(f0, b) for _, f0, b in wins]
    count = speaker_count(placed, n_frames, max_speakers)

    # 3. one embedding per (window, active slot)
    n_win = len(wins)
    n_slot = max((b.shape[1] for _, _, b in wins), default=0)
    active = np.zeros((n_win, n_slot), dtype=bool)
    clean_counts = np.zeros((n_win, n_slot))
    emb = [[None] * n_slot for _ in range(n_win)]
    for w, (span, _, binary) in enumerate(wins):
        n_active = binary.sum(axis=1)
        for j in range(binary.shape[1]):
            if binary[:, j].sum() <= 0:
                continue
            active[w, j] = True
            clean = binary[:, j] * (n_active < 2)
            clean_counts[w, j] = float((binary[:, j] * (n_active == 1)).sum())
            mask = clean if clean.sum() > min_num_frames else binary[:, j]
            vec = embed_fn(span, np.asarray(mask, dtype=np.float32))
            if vec is not None:
                vec = np.asarray(vec, dtype=float).reshape(-1)
                if np.isfinite(vec).all() and np.linalg.norm(vec) > 0:
                    emb[w][j] = vec
    valid = np.array([[emb[w][j] is not None for j in range(n_slot)] for w in range(n_win)], dtype=bool)
    dim = next((len(v) for row in emb for v in row if v is not None), 0)

    # 4. clustering set, AHC + VBx
    stats = {"windows": n_win, "active_slots": int(active.sum()), "embeddings": int(valid.sum()),
             "invalid_embeddings": int((active & ~valid).sum()), "count_max": int(count.max()) if n_frames else 0}
    min_frames = round(min_frames_ratio * max((len(b) for _, _, b in wins), default=0))
    sel = active & valid & (clean_counts >= min_frames)
    if sel.sum() < 2:
        sel = active & valid
    stats["train_embeddings"] = int(sel.sum())
    hard = -2 * np.ones((n_win, n_slot), dtype=int)
    if sel.sum() < 2:
        hard[:] = 0                                    # DiariZen: no clustering below 2 embeddings, one cluster
    else:
        train = np.array([emb[w][j] for w, j in zip(*np.where(sel))])
        q, sp = vbx_fn(train)
        stats.update(getattr(vbx_fn, "last_info", {}))
        centroids = q[:, sp > 1e-7].T @ train
        everything = np.array([[emb[w][j] if emb[w][j] is not None else np.full(dim, np.nan)
                                for j in range(n_slot)] for w in range(n_win)])
        flat = everything.reshape(-1, dim)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (flat @ centroids.T) / (np.linalg.norm(flat, axis=1, keepdims=True)
                                          * np.linalg.norm(centroids, axis=1)[None, :])
        soft = (1.0 + cos).reshape(n_win, n_slot, -1)       # 2 - (1 - cos)
        hard = constrained_argmax(soft)
        _, renumbered = np.unique(hard.ravel(), return_inverse=True)   # as DiariZen: -2 shifts every id by one
        hard = renumbered.reshape(n_win, n_slot).astype(int)
    hard[~active] = -2                                    # inactive slots are ignored from here on
    if drop_empty_clusters and (hard >= 0).any():
        _, compact = np.unique(hard[hard >= 0], return_inverse=True)
        hard[hard >= 0] = compact
    n_clusters = int(hard.max()) + 1 if (hard >= 0).any() else 0
    stats["clusters"] = n_clusters

    # 6. reconstruct: max over a cluster's slots per window, summed over windows, top-`count` per frame
    segments = []
    if n_clusters:
        acts = np.zeros((n_clusters, n_frames))
        for w, (_, f0, binary) in enumerate(wins):
            hi = min(n_frames, f0 + len(binary))
            if hi <= f0:
                continue
            for k in np.unique(hard[w][hard[w] >= 0]):
                cols = np.where(hard[w] == k)[0]
                acts[k, f0:hi] += binary[: hi - f0, cols].max(axis=1)
        top = np.argsort(-acts, axis=0)                  # same call as pyannote's to_diarization
        binary_out = np.zeros_like(acts)
        for t in range(n_frames):
            for i in range(min(int(count[t]), n_clusters)):
                binary_out[top[i, t], t] = 1.0
        for k in range(n_clusters):
            for s0, s1 in binarize_frames(binary_out[k], frame_s):
                segments.append({"start": s0, "end": s1, "cluster": k})
    segments.sort(key=lambda r: (r["start"], r["end"]))

    fits = []
    for w, ((s, e), _, binary) in enumerate(wins):
        cl = np.where(hard[w] >= 0, hard[w], -1)
        embs = np.zeros((n_slot, dim))
        for j in range(n_slot):
            if valid[w, j]:
                embs[j] = emb[w][j] / np.linalg.norm(emb[w][j])
        fits.append(WindowFit(s, e, cl, embs, valid[w].copy()))
    return PieceResult(segments=segments, windows=fits, n_clusters=n_clusters, stats=stats)
