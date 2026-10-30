#!/usr/bin/env python3
"""Re-estimate the x-vector LDA and the PLDA that DiariZen's VBx uses, for another embedder. CPU only (numpy/scipy).

    python tools/train_plda.py --train labelled.npz --out DIR --embedder-id wespeaker_r293 \
        [--lda-dim 128] [--allow-lower-dim] [--holdout-speakers 0.2] [--force]

INPUT (--train, NPZ, the only format): X float32/float64 (N, D) embeddings and y (N,) speaker labels, int or str.
    np.savez("labelled.npz", X=X, y=y)
Embeddings are NOT extracted here. Produce X and y elsewhere (the GPU and the model are needed for that).

OUTPUT (--out DIR, three files, read by utils/dz_pipeline.make_dz_vbx_fn and tools/windowed_vbx_stage.py --plda-dir):
    xvec_transform.npz   mean1 (D,), mean2 (L,), lda (D, L)
    plda.npz             mu (L,), tr (L, L), psi (L,)
    plda_meta.json       embedder_id, embedder_dim, lda_dim, n_speakers, n_embeddings, source, created, note
The files load with diarizen/clustering/VBx.py vbx_setup, which re-derives W and B from tr and psi.

FORMULAS (same preprocessing as DiariZen's xvec_tf / plda_tf in VBx.py):
    x-vector step:  z = sqrt(D) * l2(x - mean1)
                    p = lda.T z                    mean2 = mean of p over the training set
                    y = sqrt(L) * l2(p - mean2)    (the xvec_tf output that PLDA sees)
    LDA:            Sw, Sb = within/between scatter of z (Sb weights each class by its count, both divided by N),
                    lda = top-L eigenvectors of eigh(Sb, Sw + ridge*I). At most n_speakers - 1 are useful.
    PLDA (two-covariance model) on y:
                    mu = mean(y)
                    W  = sum_c sum_{i in c} (y_i - m_c)(y_i - m_c)^T / (N - C)        (within-class)
                    Bm = sum_c (m_c - mbar)(m_c - mbar)^T / (C - 1), mbar = mean of the m_c
                    B  = Bm - mean_c(1/n_c) * W                                        (between-class, unbiased-ish)
                    (vals, V) = eigh(B, W), columns of V normalised so that V^T W V = I
                    psi = vals sorted descending, floored at 1e-6; tr = V[:, order].T
                    so tr W tr^T = I and tr B tr^T = diag(psi): the whitened space has unit within-class variance.
    vbx_setup recomputes W = inv(tr.T tr) and B = inv((tr.T / psi) tr) from the files, which equal the W and B above,
    so its eigh gives the same whitening. Each PLDA dimension may come back with a flipped sign (eigh's choice).
    VBx's likelihood is symmetric in the sign of each dimension, so this does not change the clustering.

--holdout-speakers FRAC (default 0.2; 0 turns it off): a check, not part of the output. A second LDA+PLDA is fitted
without a random FRAC of the speakers. Its held-out PLDA log-likelihood ratio and cosine scores on held-out pairs give an
EER, which is printed. The output model itself is fitted on all speakers.

NOT VALIDATED: this does not test whether the re-estimated PLDA improves DER. The meta note says so.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
from scipy.linalg import eigh

PSI_FLOOR = 1e-6
RIDGE = 1e-8
MAX_TRIAL_PAIRS = 200_000
HOLDOUT_SEED = 0
NOTE = "re-estimated, not validated against DER"
OUT_FILES = ("xvec_transform.npz", "plda.npz", "plda_meta.json")


def l2_norm(vec_or_matrix):
    """Same as diarizen/clustering/VBx.py l2_norm: row-wise for 2-D input."""
    if len(vec_or_matrix.shape) == 1:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    return vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]


def xvec_apply(X, mean1, lda, mean2):
    """xvec_tf of vbx_setup: sqrt(L) * l2(lda.T @ (sqrt(D) * l2(x - mean1)).T).T - mean2) for rows of X."""
    return np.sqrt(lda.shape[1]) * l2_norm(lda.T.dot(np.sqrt(lda.shape[0]) * l2_norm(X - mean1).T).T - mean2)


def plda_apply(Y, mu, tr, lda_dim=None):
    """plda_tf of vbx_setup: ((y - mu) @ tr.T)[:, :lda_dim]."""
    if lda_dim is None:
        lda_dim = tr.shape[0]
    return ((Y - mu).dot(tr.T))[:, :lda_dim]


def _class_stats(Z, idx):
    C = int(idx.max()) + 1
    counts = np.bincount(idx, minlength=C).astype(float)
    sums = np.zeros((C, Z.shape[1]))
    np.add.at(sums, idx, Z)
    means = sums / counts[:, None]
    diff = Z - means[idx]
    return counts, means, diff.T.dot(diff)


def fisher_lda(Z, idx, L):
    """Top-L generalised eigenvectors of (Sb, Sw), returned as columns (D, L)."""
    N, D = Z.shape
    counts, means, Sw = _class_stats(Z, idx)
    d = means - Z.mean(0)
    Sb = (d * counts[:, None]).T.dot(d) / N
    Sw = Sw / N
    vals, V = eigh(Sb, Sw + RIDGE * np.trace(Sw) / D * np.eye(D))
    order = np.argsort(vals)[::-1][:L]
    return V[:, order]


def xvec_fit(X, idx, L):
    """mean1, lda, mean2 from labelled X (N, D) and speaker indices idx."""
    D = X.shape[1]
    mean1 = X.mean(0)
    Z = np.sqrt(D) * l2_norm(X - mean1)
    lda = fisher_lda(Z, idx, L)
    P = lda.T.dot(Z.T).T
    mean2 = P.mean(0)
    return mean1, lda, mean2


def plda_fit(Y, idx):
    """Two-covariance PLDA on y (N, L). Returns mu (L,), tr (L, L), psi (L,) as in the module docstring."""
    N = Y.shape[0]
    counts, means, Sw = _class_stats(Y, idx)
    C = len(counts)
    if N <= C:
        raise ValueError("PLDA needs speakers with at least 2 embeddings each")
    mu = Y.mean(0)
    W = Sw / (N - C)
    dm = means - means.mean(0)
    Bm = dm.T.dot(dm) / (C - 1)
    B = Bm - np.mean(1.0 / counts) * W
    vals, V = eigh(B, W)
    order = np.argsort(vals)[::-1]
    psi = np.maximum(vals[order], PSI_FLOOR)
    tr = V[:, order].T
    return mu, tr, psi


def fit_all(X, idx, L):
    """Full estimation for one labelled set. Returns a dict with the 3+3 arrays of the two npz files."""
    mean1, lda, mean2 = xvec_fit(X, idx, L)
    Y = xvec_apply(X, mean1, lda, mean2)
    mu, tr, psi = plda_fit(Y, idx)
    return {"mean1": mean1, "mean2": mean2, "lda": lda, "mu": mu, "tr": tr, "psi": psi}


def plda_llr(U, V, psi):
    """Two-covariance log-likelihood ratio, row-wise, for whitened vectors U, V (P, L).

    Per dimension: within variance 1, between variance psi, so the same-speaker joint covariance is
    [[t, p], [p, t]] with t = 1 + p and p = psi, and the different-speaker one is diag(t, t).
    """
    p = np.asarray(psi, dtype=float)[None, :]
    t = 1.0 + p
    uu = U * U + V * V
    uv = U * V
    q = (t * uu - 2.0 * p * uv) / (t * t - p * p)
    r = uu / t
    return (-0.5 * (np.log((t * t - p * p) / (t * t)) + q - r)).sum(axis=1)


def eer(scores, labels):
    """Equal error rate. labels: True/1 for target trials. Scores: higher means more likely the same speaker."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n_t, n_n = int(labels.sum()), int((~labels).sum())
    if n_t == 0 or n_n == 0:
        raise ValueError("EER needs both target and non-target trials")
    order = np.argsort(-scores, kind="mergesort")
    s, l = scores[order], labels[order]
    # one threshold per distinct score: accept the top k (k at the last index of each tie group)
    k = np.nonzero(np.r_[s[1:] != s[:-1], True])[0]
    tp = np.r_[0, np.cumsum(l)[k]] / n_t
    fp = np.r_[0, np.cumsum(~l)[k]] / n_n
    fnr = 1.0 - tp
    i = int(np.argmax(fnr <= fp))
    return float((fnr[i] + fp[i]) / 2.0)


def make_trials(idx, rng):
    """Index pairs (i, j) over the embeddings: all pairs if few, else MAX_TRIAL_PAIRS random pairs."""
    n = len(idx)
    if n * (n - 1) // 2 <= MAX_TRIAL_PAIRS:
        i, j = np.triu_indices(n, 1)
    else:
        i = rng.integers(0, n, MAX_TRIAL_PAIRS)
        j = rng.integers(0, n, MAX_TRIAL_PAIRS)
        keep = i != j
        i, j = i[keep], j[keep]
    return i, j, idx[i] == idx[j]


def resolve_lda_dim(requested, n_speakers, dim, allow_lower):
    """Return the LDA dimension to use. Raises ValueError when it cannot be used."""
    if requested < 1:
        raise ValueError(f"--lda-dim must be >= 1, got {requested}")
    if requested > dim:
        raise ValueError(f"--lda-dim {requested} is larger than the embedding dimension {dim}")
    useful = n_speakers - 1
    if requested > useful:
        if allow_lower:
            return useful
        raise ValueError(
            f"--lda-dim {requested} > n_speakers - 1 = {useful}: the LDA has at most {useful} useful directions. "
            "Lower --lda-dim, add speakers, or pass --allow-lower-dim.")
    return requested


def load_train(path):
    """NPZ with X (N, D) and y (N,). Returns X as float64 and y unchanged."""
    with np.load(path, allow_pickle=False) as z:
        if "X" not in z.files or "y" not in z.files:
            raise ValueError(f"{path} must hold arrays X (N, D) and y (N,)")
        X = np.asarray(z["X"], dtype=np.float64)
        y = np.asarray(z["y"])
    if X.ndim != 2 or y.ndim != 1 or len(y) != len(X):
        raise ValueError(f"X must be (N, D) and y (N,) with the same N; got {X.shape} and {y.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X has non-finite values")
    return X, y


def holdout_eer(X, idx, L, frac):
    """Fit without a random `frac` of the speakers, then score held-out pairs with PLDA and cosine."""
    S = int(idx.max()) + 1
    n_hold = max(2, int(round(frac * S)))
    if n_hold < 2 or S - n_hold < 3:
        return None, f"holdout check skipped ({S} speakers, {n_hold} held out; need >= 2 held out and >= 3 kept)"
    rng = np.random.default_rng(HOLDOUT_SEED)
    perm = rng.permutation(S)
    held = np.isin(idx, perm[:n_hold])
    tr_idx = np.unique(idx[~held], return_inverse=True)[1]
    ho_idx = np.unique(idx[held], return_inverse=True)[1]
    L_check = min(L, (S - n_hold) - 1)
    m = fit_all(X[~held], tr_idx, L_check)
    Xh = X[held]
    Yh = xvec_apply(Xh, m["mean1"], m["lda"], m["mean2"])
    Zh = plda_apply(Yh, m["mu"], m["tr"])
    i, j, target = make_trials(ho_idx, rng)
    if target.sum() == 0 or (~target).sum() == 0:
        return None, "holdout check skipped (no target or non-target pairs among the held-out speakers)"
    plda_scores = plda_llr(Zh[i], Zh[j], m["psi"])
    Xn = l2_norm(Xh)
    cos_scores = np.sum(Xn[i] * Xn[j], axis=1)
    res = {"eer_plda": eer(plda_scores, target), "eer_cosine": eer(cos_scores, target),
           "n_held_speakers": n_hold, "n_train_speakers": S - n_hold, "lda_dim_check": L_check,
           "n_target": int(target.sum()), "n_nontarget": int((~target).sum())}
    return res, None


def _refuse_overwrite(out_dir, force):
    existing = [f for f in OUT_FILES if os.path.exists(os.path.join(out_dir, f))]
    if existing and not force:
        raise ValueError(f"{out_dir} already has {', '.join(existing)}; pass --force to overwrite")


def write_model(out_dir, model, meta):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "xvec_transform.npz"),
             mean1=model["mean1"], mean2=model["mean2"], lda=model["lda"])
    np.savez(os.path.join(out_dir, "plda.npz"), mu=model["mu"], tr=model["tr"], psi=model["psi"])
    with open(os.path.join(out_dir, "plda_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", required=True, help="NPZ with X (N, D) embeddings and y (N,) speaker labels")
    ap.add_argument("--out", required=True, help="output folder for xvec_transform.npz, plda.npz, plda_meta.json")
    ap.add_argument("--embedder-id", required=True, help="embedder name stored in plda_meta.json")
    ap.add_argument("--lda-dim", type=int, default=128)
    ap.add_argument("--allow-lower-dim", action="store_true",
                    help="use n_speakers - 1 LDA dimensions when --lda-dim is larger")
    ap.add_argument("--holdout-speakers", type=float, default=0.2,
                    help="fraction of speakers held out for an EER check (0 to skip)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing model in --out")
    return ap


def run(a):
    if not 0.0 <= a.holdout_speakers < 1.0:
        raise ValueError("--holdout-speakers must be in [0, 1)")
    _refuse_overwrite(a.out, a.force)
    X, y = load_train(a.train)
    uniq, idx = np.unique(y, return_inverse=True)
    idx = idx.reshape(-1)
    S = len(uniq)
    if S < 3:
        raise ValueError(f"need at least 3 speakers, got {S}")
    L = resolve_lda_dim(a.lda_dim, S, X.shape[1], a.allow_lower_dim)
    model = fit_all(X, idx, L)
    meta = {"embedder_id": a.embedder_id, "embedder_dim": int(X.shape[1]), "lda_dim": int(L),
            "n_speakers": int(S), "n_embeddings": int(len(X)), "source": os.path.abspath(a.train),
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "note": NOTE, "requested_lda_dim": int(a.lda_dim)}
    write_model(a.out, model, meta)
    print(f"wrote {a.out}: embedder={a.embedder_id} D={X.shape[1]} L={L} speakers={S} embeddings={len(X)}")
    if L != a.lda_dim:
        print(f"note: --lda-dim {a.lda_dim} lowered to {L} = n_speakers - 1 (--allow-lower-dim)")
    if a.holdout_speakers > 0:
        res, why = holdout_eer(X, idx, L, a.holdout_speakers)
        if res is None:
            print(why)
        else:
            print(f"held-out EER ({res['n_held_speakers']} speakers held out, {res['n_train_speakers']} kept, "
                  f"lda_dim {res['lda_dim_check']}, {res['n_target']} target / {res['n_nontarget']} non-target pairs): "
                  f"PLDA {res['eer_plda']:.4f}  cosine {res['eer_cosine']:.4f}")
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return run(a)
    except ValueError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    sys.exit(main())
