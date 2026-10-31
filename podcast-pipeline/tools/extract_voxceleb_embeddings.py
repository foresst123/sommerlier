#!/usr/bin/env python3
"""Trích embedding VoxCeleb2 cho huấn luyện PLDA (giai đoạn 1 của kế hoạch PLDA → full).

    python tools/extract_voxceleb_embeddings.py --audio-root /data/voxceleb2/aac --embedder r34lm \
        --out /data/plda/r34lm_vox2 [--per-speaker 50] [--min-s 0.5] [--limit-speakers 20] [--device cuda]
    python tools/extract_voxceleb_embeddings.py --out /data/plda/r34lm_vox2 --embedder r34lm --merge

Layout VoxCeleb2: <audio-root>/<speaker>/<video>/<utterance>.m4a. Mỗi speaker lấy tối đa --per-speaker utterance,
trải đều trên các video (round-robin), chọn ngẫu nhiên có hạt giống cố định (--seed).

Đầu ra trong --out:
  chunks/<speaker>.npz   embedding của một speaker (X, y). Chạy lại sẽ bỏ qua speaker đã có.
  progress.jsonl         một dòng mỗi speaker: số utterance dùng, quá ngắn, bị từ chối.
  train.npz              sau --merge: X (N, D) và y (N,), đúng định dạng tools/train_plda.py đọc.
  extract_meta.json      sau --merge: embedder, D, số speaker, số embedding.

Chỉ dùng embedder có sẵn trong tools/embedders.py. SimAM và ESPnet còn UNVERIFIED, chưa nên dùng để huấn luyện.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import embedders as E  # noqa: E402

SR = 16000
DEFAULT_PER_SPEAKER = 50
DEFAULT_MIN_S = 0.5


def list_speakers(audio_root: str, ext: str) -> dict[str, list[str]]:
    """speaker -> sorted utterance paths, from the VoxCeleb layout <speaker>/<video>/<utterance>.<ext>."""
    out: dict[str, list[str]] = defaultdict(list)
    suffix = "." + ext.lower().lstrip(".")
    for spk in sorted(os.listdir(audio_root)):
        sdir = os.path.join(audio_root, spk)
        if not os.path.isdir(sdir):
            continue
        for dirpath, _, files in os.walk(sdir):
            for name in files:
                if name.lower().endswith(suffix):
                    out[spk].append(os.path.join(dirpath, name))
    return {spk: sorted(paths) for spk, paths in out.items()}


def pick_utterances(paths: list[str], k: int, rng: random.Random) -> list[str]:
    """Up to k paths of one speaker, spread over its videos (the parent folder) by round-robin.
    k <= 0 keeps every path."""
    if k <= 0 or len(paths) <= k:
        return list(paths)
    by_video: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        by_video[os.path.dirname(p)].append(p)
    groups = [rng.sample(v, len(v)) for _, v in sorted(by_video.items())]
    rng.shuffle(groups)
    chosen: list[str] = []
    depth = 0
    while len(chosen) < k:
        progressed = False
        for g in groups:
            if depth < len(g):
                chosen.append(g[depth])
                progressed = True
                if len(chosen) == k:
                    break
        if not progressed:
            break
        depth += 1
    return chosen


def decode(path: str, sr: int = SR) -> np.ndarray:
    """Mono float32 waveform at sr, decoded by ffmpeg (m4a, wav, ...)."""
    proc = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-ac", "1", "-ar", str(sr),
                           "-f", "f32le", "-"], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def embed_paths(paths: list[str], adapter, *, min_s: float, sr: int = SR,
                decode_fn: Callable[[str, int], np.ndarray] = decode, workers: int = 1) -> tuple[np.ndarray, dict]:
    """Embeddings of the utterances that pass. A clip shorter than min_s, or one the adapter refuses with a
    ValueError or a non-finite vector, is counted and left out. Any other error stops the run.

    workers > 1 decodes the clips in parallel threads (ffmpeg runs in its own process, so the threads wait
    on it without holding the GIL). The GPU then gets the next clip while the others are still decoding."""
    vecs: list[np.ndarray] = []
    stats = {"offered": len(paths), "used": 0, "too_short": 0, "refused": 0}
    if workers > 1 and len(paths) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            waves = list(pool.map(lambda p: decode_fn(p, sr), paths))
    else:
        waves = (decode_fn(p, sr) for p in paths)
    for wave in waves:
        if len(wave) < min_s * sr:
            stats["too_short"] += 1
            continue
        try:
            vec = np.asarray(adapter.embed(wave), dtype=np.float64).reshape(-1)
        except ValueError:
            stats["refused"] += 1
            continue
        if not np.isfinite(vec).all():
            stats["refused"] += 1
            continue
        vecs.append(vec)
        stats["used"] += 1
    X = np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float64)
    return X, stats


def _atomic_savez(path: str, **arrays) -> None:
    tmp = path + ".tmp.npz"                 # np.savez appends .npz only when it is missing
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def chunk_path(out: str, speaker: str) -> str:
    return os.path.join(out, "chunks", f"{speaker}.npz")


def parse_shard(text: str) -> tuple[int, int]:
    """'K/N' -> (K, N) with 0 <= K < N. Shard K takes every N-th speaker starting at K."""
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise ValueError(f"--shard phải có dạng K/N, có {text!r}") from None
    if n < 1 or not 0 <= k < n:
        raise ValueError(f"--shard {text}: cần 0 <= K < N")
    return k, n


def extract(audio_root: str, out: str, adapter, *, ext: str, per_speaker: int, min_s: float, seed: int,
            limit_speakers: int = 0, decode_fn: Callable[[str, int], np.ndarray] = decode,
            workers: int = 1, shard: tuple[int, int] = (0, 1), log: Callable[[str], None] = print) -> dict:
    """Trích từng speaker, bỏ qua speaker đã có chunk. Trả về số liệu tổng.

    shard (K, N): chỉ làm speaker thứ K, K+N, K+2N, ... Nhiều tiến trình với các K khác nhau trên cùng GPU
    chạy song song mà không trùng speaker; các chunk đều vào cùng một thư mục."""
    os.makedirs(os.path.join(out, "chunks"), exist_ok=True)
    speakers = list_speakers(audio_root, ext)
    names = sorted(speakers)
    if limit_speakers > 0:
        names = names[:limit_speakers]
    if not names:
        raise SystemExit(f"không tìm thấy file .{ext} nào dưới {audio_root}")
    k, n = shard
    names = names[k::n]
    totals = {"speakers": len(names), "done_before": 0, "done_now": 0, "embeddings": 0, "dim": None}
    progress_path = os.path.join(out, "progress.jsonl")
    t0 = time.time()
    with open(progress_path, "a", encoding="utf-8") as prog:
        for i, spk in enumerate(names, 1):
            cpath = chunk_path(out, spk)
            if os.path.exists(cpath):
                totals["done_before"] += 1
                continue
            rng = random.Random(f"{seed}:{spk}")
            paths = pick_utterances(speakers[spk], per_speaker, rng)
            X, stats = embed_paths(paths, adapter, min_s=min_s, decode_fn=decode_fn, workers=workers)
            if len(X):
                if totals["dim"] is None:
                    totals["dim"] = int(X.shape[1])
                elif X.shape[1] != totals["dim"]:
                    raise RuntimeError(f"{spk}: dimension {X.shape[1]} differs from {totals['dim']}")
                _atomic_savez(cpath, X=X, y=np.array([spk] * len(X)))
                totals["embeddings"] += len(X)
            prog.write(json.dumps({"speaker": spk, **stats}, ensure_ascii=False) + "\n")
            prog.flush()
            totals["done_now"] += 1
            if i % 25 == 0 or i == len(names):
                log(f"{i}/{len(names)} speaker, {totals['embeddings']} embedding mới, "
                    f"{time.time() - t0:.0f} s")
    return totals


def merge(out: str, embedder_id: str) -> dict:
    """Ghép chunks thành train.npz (X, y) và extract_meta.json. Từ chối nếu số chiều các chunk khác nhau."""
    files = sorted(f for f in os.listdir(os.path.join(out, "chunks")) if f.endswith(".npz"))
    if not files:
        raise SystemExit(f"chưa có chunk nào trong {out}/chunks")
    Xs, ys = [], []
    dim = None
    for f in files:
        with np.load(os.path.join(out, "chunks", f), allow_pickle=False) as z:
            X, y = z["X"], z["y"]
        if dim is None:
            dim = X.shape[1]
        elif X.shape[1] != dim:
            raise SystemExit(f"{f}: dimension {X.shape[1]} khác {dim}")
        Xs.append(X)
        ys.append(y)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    _atomic_savez(os.path.join(out, "train.npz"), X=X, y=y)
    meta = {"embedder_id": embedder_id, "embedder_dim": int(dim), "n_speakers": int(len(files)),
            "n_embeddings": int(len(X)),
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(out, "extract_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-root", help="thư mục gốc VoxCeleb2 (có các thư mục speaker)")
    ap.add_argument("--out", required=True, help="thư mục đầu ra (chunks, train.npz)")
    ap.add_argument("--embedder", required=True, choices=E.KINDS)
    ap.add_argument("--embedder-id", default=None, help="tên ghi vào extract_meta.json (mặc định: --embedder)")
    ap.add_argument("--ext", default="m4a", help="đuôi file (VoxCeleb2 gốc là m4a)")
    ap.add_argument("--per-speaker", type=int, default=DEFAULT_PER_SPEAKER, help="utterance mỗi speaker, 0 = tất cả")
    ap.add_argument("--min-s", type=float, default=DEFAULT_MIN_S, help="bỏ utterance ngắn hơn (giây)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-speakers", type=int, default=0, help="chỉ speaker đầu theo thứ tự tên (để thử)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=8, help="luồng giải mã song song (ffmpeg)")
    ap.add_argument("--shard", default="0/1", help="K/N: tiến trình thứ K trong N tiến trình song song")
    ap.add_argument("--r293-onnx")
    ap.add_argument("--simam-onnx")
    ap.add_argument("--simam-dir")
    ap.add_argument("--espnet-tag", default=E.DEFAULT_ESPNET_TAG)
    ap.add_argument("--merge", action="store_true", help="chỉ ghép chunks thành train.npz")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    embedder_id = a.embedder_id or a.embedder
    if a.merge:
        meta = merge(a.out, embedder_id)
        print(f"wrote {a.out}/train.npz: embedder={embedder_id} D={meta['embedder_dim']} "
              f"speakers={meta['n_speakers']} embeddings={meta['n_embeddings']}")
        return 0
    if not a.audio_root:
        raise SystemExit("--audio-root là bắt buộc khi trích (không dùng --merge)")
    adapter = E.make_embedder(a.embedder, device=a.device, r293_onnx=a.r293_onnx, simam_onnx=a.simam_onnx,
                              simam_dir=a.simam_dir, espnet_tag=a.espnet_tag)
    adapter.load()
    totals = extract(a.audio_root, a.out, adapter, ext=a.ext, per_speaker=a.per_speaker, min_s=a.min_s,
                      seed=a.seed, limit_speakers=a.limit_speakers, workers=a.workers,
                      shard=parse_shard(a.shard))
    print(f"xong: {totals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
