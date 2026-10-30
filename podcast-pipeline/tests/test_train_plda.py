import json
import os
import sys

import numpy as np
import pytest
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import train_plda as T  # noqa: E402


def _synth(S=8, n=30, D=24, seed=0, spread=3.0, noise=0.5):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(S, D)) * spread
    y = np.repeat(np.array([f"spk{i:02d}" for i in range(S)]), n)
    X = centres[np.repeat(np.arange(S), n)] + noise * rng.normal(size=(S * n, D))
    return X.astype(np.float32), y


def _save_npz(path, X, y):
    np.savez(path, X=X, y=y)
    return str(path)


# ---- copy of diarizen/clustering/VBx.py l2_norm and vbx_setup (do not import diarizen) ----
def l2_norm(vec_or_matrix):
    if len(vec_or_matrix.shape) == 1:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    elif len(vec_or_matrix.shape) == 2:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]
    else:
        raise ValueError('Wrong number of dimensions, 1 or 2 is supported, not %i.' % len(vec_or_matrix.shape))


def vbx_setup(tf_dir):
    x = np.load(f'{tf_dir}/xvec_transform.npz')
    mean1, mean2, lda = x['mean1'], x['mean2'], x['lda']
    p = np.load(f'{tf_dir}/plda.npz')
    plda_mu, plda_tr, plda_psi = p['mu'], p['tr'], p['psi']
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]
    xvec_tf = lambda x: np.sqrt(lda.shape[1]) * l2_norm(
        lda.T.dot(
            np.sqrt(lda.shape[0]) * l2_norm(x - mean1).T
        ).T - mean2
    )
    plda_tf = lambda x0, lda_dim=lda.shape[1]: (x0 - plda_mu).dot(plda_tr.T)[:, :lda_dim]
    return xvec_tf, plda_tf, plda_psi


def _align_sign(a, b):
    """Flip columns of a to the sign of b (eigh picks each eigenvector's sign)."""
    s = np.sign(np.sum(a * b, axis=0))
    s[s == 0] = 1.0
    return a * s


# (a) the copied vbx_setup loads the files and reproduces the transforms
def test_files_load_through_vbx_setup_and_reproduce_transforms(tmp_path):
    X, y = _synth()
    npz = _save_npz(tmp_path / "train.npz", X, y)
    out = tmp_path / "model"
    assert T.main(["--train", npz, "--out", str(out), "--embedder-id", "toy", "--lda-dim", "5",
                   "--holdout-speakers", "0"]) == 0

    xt = np.load(out / "xvec_transform.npz")
    pl = np.load(out / "plda.npz")
    assert xt["mean1"].shape == (24,) and xt["mean2"].shape == (5,) and xt["lda"].shape == (24, 5)
    assert pl["mu"].shape == (5,) and pl["tr"].shape == (5, 5) and pl["psi"].shape == (5,)

    xvec_tf, plda_tf, plda_psi = vbx_setup(str(out))
    Xd = X.astype(np.float64)
    Y_ref = T.xvec_apply(Xd, xt["mean1"], xt["lda"], xt["mean2"])
    # xvec_tf is the same formula with the same numbers, so it must match exactly
    np.testing.assert_allclose(xvec_tf(Xd), Y_ref, rtol=1e-10, atol=1e-12)
    # plda_tf: vbx_setup re-derives the whitening, so each dimension may come back sign-flipped
    Z_vbx = plda_tf(xvec_tf(Xd), lda_dim=5)
    Z_ref = T.plda_apply(Y_ref, pl["mu"], pl["tr"])
    np.testing.assert_allclose(_align_sign(Z_vbx, Z_ref), Z_ref, rtol=1e-6, atol=1e-8)
    # psi comes back in descending order and equals the stored psi
    np.testing.assert_allclose(plda_psi, pl["psi"], rtol=1e-6)
    assert np.all(np.diff(plda_psi) <= 0)


# (b) LDA dimension refusal, and the --allow-lower-dim path
def test_lda_dim_refused_with_too_few_speakers(tmp_path):
    X, y = _synth(S=4, n=20, D=16)  # 4 speakers: at most 3 useful LDA directions
    npz = _save_npz(tmp_path / "train.npz", X, y)
    with pytest.raises(SystemExit, match="n_speakers - 1"):
        T.main(["--train", npz, "--out", str(tmp_path / "m"), "--embedder-id", "toy",
                "--lda-dim", "4", "--holdout-speakers", "0"])
    assert not (tmp_path / "m").exists()


def test_allow_lower_dim_uses_speakers_minus_one_and_records_it(tmp_path):
    X, y = _synth(S=4, n=20, D=16)
    npz = _save_npz(tmp_path / "train.npz", X, y)
    out = tmp_path / "m"
    T.main(["--train", npz, "--out", str(out), "--embedder-id", "toy", "--lda-dim", "4",
            "--allow-lower-dim", "--holdout-speakers", "0"])
    meta = json.loads((out / "plda_meta.json").read_text())
    assert meta["lda_dim"] == 3 and meta["requested_lda_dim"] == 4
    assert np.load(out / "xvec_transform.npz")["lda"].shape == (16, 3)
    assert np.load(out / "plda.npz")["tr"].shape == (3, 3)


def test_fewer_than_three_speakers_refused(tmp_path):
    X, y = _synth(S=2, n=20, D=8)
    npz = _save_npz(tmp_path / "train.npz", X, y)
    with pytest.raises(SystemExit, match="at least 3 speakers"):
        T.main(["--train", npz, "--out", str(tmp_path / "m"), "--embedder-id", "toy", "--lda-dim", "1"])


# (c) within-class whitening, and between-class variances equal psi
def test_within_class_whitened_and_between_equals_psi():
    X, y = _synth(S=8, n=30, D=24, seed=3, spread=4.0)
    idx = np.unique(y, return_inverse=True)[1]
    L = 5
    mean1, lda, mean2 = T.xvec_fit(X.astype(np.float64), idx, L)
    Y = T.xvec_apply(X.astype(np.float64), mean1, lda, mean2)
    mu, tr, psi = T.plda_fit(Y, idx)
    Z = T.plda_apply(Y, mu, tr)

    counts = np.bincount(idx).astype(float)
    C, N = len(counts), len(idx)
    means = np.array([Z[idx == c].mean(0) for c in range(C)])
    Sw = sum((Z[idx == c] - means[c]).T @ (Z[idx == c] - means[c]) for c in range(C))
    within = Sw / (N - C)
    np.testing.assert_allclose(within, np.eye(L), atol=1e-8)

    dm = means - means.mean(0)
    Bm = dm.T @ dm / (C - 1)
    k = np.mean(1.0 / counts)
    Bz = Bm - k * np.eye(L)  # the between-class covariance the fit used, in the whitened space
    np.testing.assert_allclose(Bz, np.diag(psi), atol=1e-8)
    assert np.all(psi > 0) and np.all(np.diff(psi) <= 0)


# (d) EER
def test_eer_perfect_separation_is_zero():
    rng = np.random.default_rng(1)
    tgt = 2.0 + rng.random(200)
    non = rng.random(200)
    scores = np.r_[tgt, non]
    labels = np.r_[np.ones(200, bool), np.zeros(200, bool)]
    assert T.eer(scores, labels) == 0.0


def test_eer_identical_distributions_is_about_half():
    rng = np.random.default_rng(2)
    n = 4000
    scores = rng.normal(size=2 * n)
    labels = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    assert abs(T.eer(scores, labels) - 0.5) < 0.03


def test_eer_needs_both_classes():
    with pytest.raises(ValueError):
        T.eer(np.array([1.0, 2.0]), np.array([True, True]))


def test_plda_llr_prefers_same_speaker_on_synthetic():
    X, y = _synth(S=8, n=30, D=24, seed=5)
    idx = np.unique(y, return_inverse=True)[1]
    m = T.fit_all(X.astype(np.float64), idx, 5)
    Z = T.plda_apply(T.xvec_apply(X.astype(np.float64), m["mean1"], m["lda"], m["mean2"]), m["mu"], m["tr"])
    i, j, target = T.make_trials(idx, np.random.default_rng(0))
    s = T.plda_llr(Z[i], Z[j], m["psi"])
    assert T.eer(s, target) < 0.05


# (e) CLI: three files, overwrite refused without --force
def test_cli_writes_three_files_and_refuses_overwrite(tmp_path, capsys):
    X, y = _synth(S=6, n=20, D=16, seed=7)
    npz = _save_npz(tmp_path / "train.npz", X, y)
    out = tmp_path / "model"
    args = ["--train", npz, "--out", str(out), "--embedder-id", "toy_emb", "--lda-dim", "4"]
    assert T.main(args) == 0
    for name in ("xvec_transform.npz", "plda.npz", "plda_meta.json"):
        assert (out / name).is_file()
    meta = json.loads((out / "plda_meta.json").read_text())
    assert meta["embedder_id"] == "toy_emb" and meta["embedder_dim"] == 16 and meta["lda_dim"] == 4
    assert meta["n_speakers"] == 6 and meta["n_embeddings"] == 120
    assert meta["source"] == os.path.abspath(npz) and meta["note"] == "re-estimated, not validated against DER"
    assert "created" in meta
    out_text = capsys.readouterr().out
    assert "held-out EER" in out_text  # default holdout check runs and prints

    with pytest.raises(SystemExit, match="--force"):
        T.main(args)
    assert T.main(args + ["--force"]) == 0
