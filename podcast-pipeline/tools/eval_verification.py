#!/usr/bin/env python3
"""EER của embedder trên một danh sách cặp xác minh (VoxCeleb1 test: veri_test2.txt), bằng cosine và PLDA.

    python tools/eval_verification.py --trials /data/vox1/veri_test2.txt --audio-root /data/vox1/wav \
        --embedder r34lm --device cuda [--plda-dir $PLDA_ROOT/plda_r34lm] --out eer_r34lm.json

Mỗi dòng của danh sách: "<nhãn> <đường_dẫn_a> <đường_dẫn_b>", nhãn 1 là cùng người, 0 là khác người. Đường dẫn tương
đối với --audio-root. Mỗi file được trích một lần (kể cả khi nhiều cặp dùng cùng file).

Điểm cosine: tích vô hướng của hai vector đã chuẩn hoá. Điểm PLDA: log-likelihood ratio của tools/train_plda.py, dùng
xvec_transform.npz và plda.npz trong --plda-dir (phải huấn luyện cho đúng embedder, số chiều phải khớp).
Cặp có file không đọc được hoặc bị embedder từ chối bị bỏ và được đếm, không âm thầm bỏ qua.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import embedders as E  # noqa: E402
import extract_voxceleb_embeddings as X  # noqa: E402
import train_plda as T  # noqa: E402


def read_trials(path: str) -> list[tuple[int, str, str]]:
    trials = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3 or parts[0] not in ("0", "1"):
                raise ValueError(f"{path}:{n}: cần 'nhãn đường_dẫn_a đường_dẫn_b' với nhãn 0 hoặc 1, có {line!r}")
            trials.append((int(parts[0]), parts[1], parts[2]))
    return trials


def embed_files(files: list[str], audio_root: str, adapter) -> tuple[dict[str, np.ndarray], list[str]]:
    """Embedding của từng file. Trả về (file -> vector, danh sách file lỗi)."""
    out: dict[str, np.ndarray] = {}
    failed: list[str] = []
    for rel in files:
        try:
            wave = X.decode(os.path.join(audio_root, rel))
            vec = np.asarray(adapter.embed(wave), dtype=np.float64).reshape(-1)
            if not np.isfinite(vec).all():
                raise ValueError("non-finite embedding")
            out[rel] = vec
        except (RuntimeError, ValueError):
            failed.append(rel)
    return out, failed


def load_plda(plda_dir: str) -> dict:
    with np.load(os.path.join(plda_dir, "xvec_transform.npz"), allow_pickle=False) as z:
        xt = {k: z[k] for k in ("mean1", "mean2", "lda")}
    with np.load(os.path.join(plda_dir, "plda.npz"), allow_pickle=False) as z:
        pl = {k: z[k] for k in ("mu", "tr", "psi")}
    with open(os.path.join(plda_dir, "plda_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return {**xt, **pl, "meta": meta}


def score_trials(trials, vecs: dict[str, np.ndarray], plda: dict | None):
    """Điểm cosine (và PLDA nếu có) cho các cặp có đủ hai vector. Trả về (nhãn, cosine, plda hoặc None, số cặp bỏ)."""
    keep = [(t, a, b) for t, a, b in trials if a in vecs and b in vecs]
    dropped = len(trials) - len(keep)
    labels = np.array([t for t, _, _ in keep], dtype=bool)
    if not keep:
        return labels, np.array([]), None, dropped
    A = np.vstack([vecs[a] for _, a, _ in keep])
    B = np.vstack([vecs[b] for _, _, b in keep])
    An, Bn = T.l2_norm(A), T.l2_norm(B)
    cos = np.sum(An * Bn, axis=1)
    if plda is None:
        return labels, cos, None, dropped
    if A.shape[1] != plda["mean1"].shape[0]:
        raise ValueError(f"số chiều embedding {A.shape[1]} khác mô hình PLDA {plda['mean1'].shape[0]}: "
                         f"PLDA này huấn luyện cho embedder khác")
    Ya = T.xvec_apply(A, plda["mean1"], plda["lda"], plda["mean2"])
    Yb = T.xvec_apply(B, plda["mean1"], plda["lda"], plda["mean2"])
    Za = T.plda_apply(Ya, plda["mu"], plda["tr"])
    Zb = T.plda_apply(Yb, plda["mu"], plda["tr"])
    llr = T.plda_llr(Za, Zb, plda["psi"])
    return labels, cos, llr, dropped


def evaluate(trials, vecs, plda=None) -> dict:
    labels, cos, llr, dropped = score_trials(trials, vecs, plda)
    res = {"trials": len(trials), "scored": int(len(labels)), "dropped": int(dropped),
           "targets": int(labels.sum()) if len(labels) else 0,
           "nontargets": int((~labels).sum()) if len(labels) else 0}
    if res["targets"] == 0 or res["nontargets"] == 0:
        res["eer_cosine"] = res["eer_plda"] = None
        res["error"] = "EER cần cả cặp cùng người và cặp khác người"
        return res
    res["eer_cosine"] = T.eer(cos, labels)
    res["eer_plda"] = T.eer(llr, labels) if llr is not None else None
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--embedder", required=True, choices=E.KINDS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plda-dir", default=None)
    ap.add_argument("--r293-onnx")
    ap.add_argument("--simam-onnx")
    ap.add_argument("--simam-dir")
    ap.add_argument("--espnet-tag", default=E.DEFAULT_ESPNET_TAG)
    ap.add_argument("--out", default=None, help="ghi kết quả JSON vào đây")
    a = ap.parse_args(argv)

    trials = read_trials(a.trials)
    files = sorted({p for _, x, y in trials for p in (x, y)})
    adapter = E.make_embedder(a.embedder, device=a.device, r293_onnx=a.r293_onnx, simam_onnx=a.simam_onnx,
                              simam_dir=a.simam_dir, espnet_tag=a.espnet_tag)
    adapter.load()
    vecs, failed = embed_files(files, a.audio_root, adapter)
    plda = load_plda(a.plda_dir) if a.plda_dir else None
    res = evaluate(trials, vecs, plda)
    res.update({"embedder": a.embedder, "files": len(files), "files_failed": len(failed),
                "plda_dir": a.plda_dir, "failed_examples": failed[:10]})
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
