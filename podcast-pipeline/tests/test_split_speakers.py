import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from tools import split_speakers as S  # noqa: E402
from tools import train_plda as T  # noqa: E402


def toy(rng, speakers=10, per=8, dim=6):
    centres = rng.standard_normal((speakers, dim)) * 3.0
    X = np.vstack([centres[s] + 0.3 * rng.standard_normal((per, dim)) for s in range(speakers)])
    y = np.repeat(np.array([f"id{s:05d}" for s in range(speakers)]), per)
    return X, y


def test_split_is_speaker_disjoint_and_covers_all():
    rng = np.random.default_rng(0)
    _, y = toy(rng)
    held = S.split_by_speaker(y, 0.5, seed=0)
    assert set(np.unique(y[held])) & set(np.unique(y[~held])) == set()
    assert set(np.unique(y[held])) | set(np.unique(y[~held])) == set(np.unique(y))
    assert len(np.unique(y[held])) == 5


def test_split_is_reproducible_and_seed_dependent():
    rng = np.random.default_rng(0)
    _, y = toy(rng)
    a = S.split_by_speaker(y, 0.5, seed=0)
    assert np.array_equal(a, S.split_by_speaker(y, 0.5, seed=0))
    assert not np.array_equal(a, S.split_by_speaker(y, 0.5, seed=7))


def test_split_refuses_bad_fraction_and_too_few_speakers():
    rng = np.random.default_rng(0)
    _, y = toy(rng)
    with pytest.raises(ValueError):
        S.split_by_speaker(y, 1.0, seed=0)
    with pytest.raises(ValueError):
        S.split_by_speaker(np.array(["a", "a", "b", "b"]), 0.5, seed=0)


def test_split_command_writes_two_files_and_a_summary(tmp_path):
    rng = np.random.default_rng(1)
    X, y = toy(rng)
    src = tmp_path / "train.npz"
    np.savez(src, X=X, y=y)
    out = tmp_path / "half"
    assert S.main(["split", "--train", str(src), "--out-dir", str(out), "--frac", "0.5", "--seed", "0"]) == 0
    with np.load(out / "train_half.npz", allow_pickle=False) as z:
        tr_y = z["y"]
    with np.load(out / "heldout_half.npz", allow_pickle=False) as z:
        ho_y = z["y"]
    assert not set(tr_y) & set(ho_y)
    assert len(tr_y) + len(ho_y) == len(y)
    doc = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert doc["train"]["speakers"] == 5 and doc["heldout"]["speakers"] == 5


def test_eval_on_heldout_with_cosine_and_plda(tmp_path):
    rng = np.random.default_rng(2)
    X, y = toy(rng, speakers=8, per=10)
    ho = tmp_path / "heldout.npz"
    np.savez(ho, X=X, y=y)
    res_cos = tmp_path / "cos.json"
    assert S.main(["eval", "--heldout", str(ho), "--out", str(res_cos)]) == 0
    doc = json.loads(res_cos.read_text(encoding="utf-8"))
    assert doc["eer_cosine"] == pytest.approx(0.0, abs=1e-9)   # well separated speakers
    assert doc["eer_plda"] is None and doc["n_speakers"] == 8

    # PLDA trained on other speakers (not these), then scored on the held-out speakers
    Xt, yt = toy(np.random.default_rng(3), speakers=8, per=10)
    model = T.fit_all(Xt, np.unique(yt, return_inverse=True)[1].reshape(-1), 3)
    plda_dir = tmp_path / "plda"
    T.write_model(str(plda_dir), model, {"embedder_id": "toy"})
    res_plda = tmp_path / "plda.json"
    assert S.main(["eval", "--heldout", str(ho), "--plda-dir", str(plda_dir), "--out", str(res_plda)]) == 0
    doc2 = json.loads(res_plda.read_text(encoding="utf-8"))
    assert doc2["eer_plda"] is not None and 0.0 <= doc2["eer_plda"] <= 1.0
