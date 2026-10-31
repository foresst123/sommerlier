#!/usr/bin/env python3
"""Chia tập embedding VoxCeleb2 theo speaker thành hai nửa không trùng người, rồi đánh giá nửa còn lại.

    # 1. chia: nửa train để huấn luyện PLDA, nửa held để đánh giá
    python tools/split_speakers.py split --train $PLDA_ROOT/r34lm_vox2/train.npz --out-dir $PLDA_ROOT/r34lm_half \
        --frac 0.5 --seed 0
    # 2. huấn luyện PLDA chỉ trên nửa train
    python tools/train_plda.py --train $PLDA_ROOT/r34lm_half/train_half.npz --out $PLDA_ROOT/plda_r34lm_half \
        --embedder-id r34lm --lda-dim 128 --holdout-speakers 0
    # 3. EER trên nửa held (cặp lấy từ speaker chưa thấy khi huấn luyện)
    python tools/split_speakers.py eval --heldout $PLDA_ROOT/r34lm_half/heldout_half.npz \
        --plda-dir $PLDA_ROOT/plda_r34lm_half --out $PLDA_ROOT/eer_heldout_r34lm.json

Chia theo speaker: mỗi speaker chỉ nằm ở một nửa. Hạt giống cố định cho cùng kết quả. Lệnh split ghi split.json
(seed, frac, số speaker và embedding mỗi nửa) để bạn kiểm tra. Lệnh eval dùng cùng cách tạo cặp như train_plda.py
(MAX_TRIAL_PAIRS, hạt giống cố định) và chấm bằng tools/eval_verification.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_verification as V  # noqa: E402
import train_plda as T  # noqa: E402


def split_by_speaker(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean mask: True for embeddings of the held-out half. Whole speakers only."""
    if not 0.0 < frac < 1.0:
        raise ValueError("--frac phải nằm trong (0, 1)")
    speakers = np.unique(y)
    if len(speakers) < 4:
        raise ValueError(f"cần ít nhất 4 speaker để chia hai nửa, có {len(speakers)}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(speakers))
    n_held = max(2, int(round(frac * len(speakers))))
    held_names = speakers[perm[:n_held]]
    return np.isin(y, held_names)


def cmd_split(a) -> int:
    X, y = T.load_train(a.train)
    held = split_by_speaker(y, a.frac, a.seed)
    train_mask = ~held
    if set(np.unique(y[train_mask])) & set(np.unique(y[held])):
        raise RuntimeError("bug: speaker xuất hiện ở cả hai nửa")
    os.makedirs(a.out_dir, exist_ok=True)
    tr_path = os.path.join(a.out_dir, "train_half.npz")
    ho_path = os.path.join(a.out_dir, "heldout_half.npz")
    np.savez(tr_path + ".tmp.npz", X=X[train_mask], y=y[train_mask])
    os.replace(tr_path + ".tmp.npz", tr_path)
    np.savez(ho_path + ".tmp.npz", X=X[held], y=y[held])
    os.replace(ho_path + ".tmp.npz", ho_path)
    doc = {"seed": a.seed, "frac": a.frac, "source": os.path.abspath(a.train),
           "train": {"speakers": int(len(np.unique(y[train_mask]))), "embeddings": int(train_mask.sum())},
           "heldout": {"speakers": int(len(np.unique(y[held]))), "embeddings": int(held.sum())}}
    with open(os.path.join(a.out_dir, "split.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"train: {doc['train']}  heldout: {doc['heldout']}  → {a.out_dir}")
    return 0


def heldout_trials(y: np.ndarray, seed: int) -> list[tuple[int, str, str]]:
    """Cặp (nhãn, i, j) trên các embedding held-out, cùng cách tạo cặp như train_plda.make_trials."""
    _, idx = np.unique(y, return_inverse=True)
    idx = idx.reshape(-1)
    rng = np.random.default_rng(seed)
    i, j, target = T.make_trials(idx, rng)
    return [(int(t), str(a), str(b)) for t, a, b in zip(target, i, j)]


def cmd_eval(a) -> int:
    X, y = T.load_train(a.heldout)
    trials = heldout_trials(y, a.seed)
    vecs = {str(k): X[k] for k in range(len(X))}
    plda = V.load_plda(a.plda_dir) if a.plda_dir else None
    res = V.evaluate(trials, vecs, plda)
    res.update({"heldout": os.path.abspath(a.heldout), "plda_dir": a.plda_dir, "n_speakers": int(len(np.unique(y))),
                "n_embeddings": int(len(X))})
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--train", required=True, help="train.npz của tools/extract_voxceleb_embeddings.py")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--frac", type=float, default=0.5, help="tỉ lệ speaker đưa vào nửa held-out")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_split)
    e = sub.add_parser("eval")
    e.add_argument("--heldout", required=True, help="heldout_half.npz")
    e.add_argument("--plda-dir", default=None, help="thư mục PLDA huấn luyện trên nửa train (bỏ trống = chỉ cosine)")
    e.add_argument("--seed", type=int, default=T.HOLDOUT_SEED)
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_eval)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
