#!/usr/bin/env python3
"""Trích embedding VoxCeleb2 cho huấn luyện PLDA. Mỗi utterance được giải mã MỘT lần, rồi đưa cho mọi embedder.

    # một embedder
    python tools/extract_voxceleb_embeddings.py --audio-root /data/voxceleb2/aac --embedder r34lm \
        --out /data/plda/r34lm_train --per-speaker 50 --device cuda --workers 6 --shard 0/4
    # nhiều embedder trong cùng một tiến trình (một lần giải mã cho tất cả)
    python tools/extract_voxceleb_embeddings.py --audio-root /data/voxceleb2/aac \
        --embedder r34lm,simam100,espnet \
        --out r34lm=/data/plda/r34lm_train,simam100=/data/plda/simam_train,espnet=/data/plda/espnet_train \
        --simam-dir /home/lamkd2/models/simam100 --device cuda --workers 6 --shard 0/4
    # ghép chunk thành train.npz (một embedder)
    python tools/extract_voxceleb_embeddings.py --out /data/plda/r34lm_train --embedder r34lm --merge

Layout VoxCeleb2: <audio-root>/<speaker>/<video>/<utterance>.m4a. Mỗi speaker lấy tối đa --per-speaker utterance,
trải đều trên các video, chọn ngẫu nhiên có hạt giống cố định (--seed). Chọn utterance không phụ thuộc embedder,
nên mọi embedder dùng cùng một tập.

Đầu ra cho mỗi embedder (thư mục --out của nó):
  chunks/<speaker>.npz   embedding của một speaker (X, y). Chạy lại sẽ bỏ qua speaker đã có chunk của embedder đó.
  progress.jsonl         một dòng mỗi speaker: số utterance dùng, quá ngắn, bị từ chối.
  train.npz              sau --merge: X (N, D) và y (N,), đúng định dạng tools/train_plda.py đọc.
  extract_meta.json      sau --merge: embedder, D, số speaker, số embedding.

--shard K/N: tiến trình thứ K trong N tiến trình song song, mỗi tiến trình lấy speaker K, K+N, ... Các chunk
không trùng nhau nên các shard không ghi đè lên nhau.
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


def embed_paths_multi(paths: list[str], adapters: dict, *, min_s: float, sr: int = SR,
                      decode_fn: Callable[[str, int], np.ndarray] = decode,
                      workers: int = 1) -> dict[str, tuple[np.ndarray, dict]]:
    """Embeddings of the utterances for every adapter, decoding each utterance once.

    adapters: {kind: adapter}. Returns {kind: (X, stats)}. A clip shorter than min_s is left out for every
    adapter. For one adapter, a clip the adapter refuses (ValueError, or a non-finite vector) is counted and
    left out; any other error stops the run.

    workers > 1 decodes the clips in parallel threads (ffmpeg runs in its own process, so the threads wait on it
    without holding the GIL). The adapters take the clips in order."""
    kinds = list(adapters)
    vecs = {k: [] for k in kinds}
    stats = {k: {"offered": len(paths), "used": 0, "too_short": 0, "refused": 0} for k in kinds}
    if workers > 1 and len(paths) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            waves = list(pool.map(lambda p: decode_fn(p, sr), paths))
    else:
        waves = (decode_fn(p, sr) for p in paths)
    for wave in waves:
        if len(wave) < min_s * sr:
            for k in kinds:
                stats[k]["too_short"] += 1
            continue
        for k in kinds:
            try:
                vec = np.asarray(adapters[k].embed(wave), dtype=np.float64).reshape(-1)
            except ValueError:
                stats[k]["refused"] += 1
                continue
            if not np.isfinite(vec).all():
                stats[k]["refused"] += 1
                continue
            vecs[k].append(vec)
            stats[k]["used"] += 1
    out = {}
    for k in kinds:
        X = np.vstack(vecs[k]) if vecs[k] else np.zeros((0, 0), dtype=np.float64)
        out[k] = (X, stats[k])
    return out


def embed_paths(paths: list[str], adapter, *, min_s: float, sr: int = SR,
                decode_fn: Callable[[str, int], np.ndarray] = decode, workers: int = 1) -> tuple[np.ndarray, dict]:
    """One adapter. Same rules as embed_paths_multi."""
    res = embed_paths_multi(paths, {"one": adapter}, min_s=min_s, sr=sr, decode_fn=decode_fn, workers=workers)
    return res["one"]


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


def extract_multi(audio_root: str, outs: dict[str, str], adapters: dict, *, ext: str, per_speaker: int,
                  min_s: float, seed: int, limit_speakers: int = 0,
                  decode_fn: Callable[[str, int], np.ndarray] = decode, workers: int = 1,
                  shard: tuple[int, int] = (0, 1), log: Callable[[str], None] = print) -> dict:
    """Trích cho mọi embedder cùng lúc. outs và adapters có cùng các khoá (kind).

    Một speaker được giải mã chỉ khi còn ít nhất một embedder chưa có chunk của speaker đó."""
    kinds = list(adapters)
    for k in kinds:
        os.makedirs(os.path.join(outs[k], "chunks"), exist_ok=True)
    speakers = list_speakers(audio_root, ext)
    names = sorted(speakers)
    if limit_speakers > 0:
        names = names[:limit_speakers]
    if not names:
        raise SystemExit(f"không tìm thấy file .{ext} nào dưới {audio_root}")
    k_shard, n_shard = shard
    names = names[k_shard::n_shard]
    totals = {k: {"done_before": 0, "done_now": 0, "embeddings": 0, "dim": None} for k in kinds}
    t0 = time.time()
    progress = {k: open(os.path.join(outs[k], "progress.jsonl"), "a", encoding="utf-8") for k in kinds}
    try:
        for i, spk in enumerate(names, 1):
            todo = []
            for k in kinds:
                if os.path.exists(chunk_path(outs[k], spk)):
                    totals[k]["done_before"] += 1
                else:
                    todo.append(k)
            if not todo:
                continue
            rng = random.Random(f"{seed}:{spk}")
            paths = pick_utterances(speakers[spk], per_speaker, rng)
            results = embed_paths_multi(paths, {k: adapters[k] for k in todo}, min_s=min_s,
                                        decode_fn=decode_fn, workers=workers)
            for k in todo:
                X, stats = results[k]
                if len(X):
                    dim = totals[k]["dim"]
                    if dim is None:
                        totals[k]["dim"] = int(X.shape[1])
                    elif X.shape[1] != dim:
                        raise RuntimeError(f"{k} {spk}: dimension {X.shape[1]} differs from {dim}")
                    _atomic_savez(chunk_path(outs[k], spk), X=X, y=np.array([spk] * len(X)))
                    totals[k]["embeddings"] += len(X)
                progress[k].write(json.dumps({"speaker": spk, **stats}, ensure_ascii=False) + "\n")
                progress[k].flush()
                totals[k]["done_now"] += 1
            if i % 25 == 0 or i == len(names):
                log(f"{i}/{len(names)} speaker, {time.time() - t0:.0f} s")
    finally:
        for f in progress.values():
            f.close()
    return totals


def extract(audio_root: str, out: str, adapter, *, ext: str, per_speaker: int, min_s: float, seed: int,
            limit_speakers: int = 0, decode_fn: Callable[[str, int], np.ndarray] = decode,
            workers: int = 1, shard: tuple[int, int] = (0, 1), log: Callable[[str], None] = print) -> dict:
    """Một embedder, một thư mục ra. Trả về số liệu của embedder đó."""
    totals = extract_multi(audio_root, {"one": out}, {"one": adapter}, ext=ext, per_speaker=per_speaker,
                           min_s=min_s, seed=seed, limit_speakers=limit_speakers, decode_fn=decode_fn,
                           workers=workers, shard=shard, log=log)
    t = totals["one"]
    return {"speakers": t["done_before"] + t["done_now"], "done_before": t["done_before"],
            "done_now": t["done_now"], "embeddings": t["embeddings"], "dim": t["dim"]}


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


def parse_outs(text: str, kinds: list[str]) -> dict[str, str]:
    """Một embedder: '--out DIR' cũng được. Nhiều: 'kind=DIR,kind=DIR'. Phải có đúng một thư mục cho mỗi kind."""
    if "=" not in text:
        if len(kinds) != 1:
            raise ValueError("với nhiều embedder, --out phải có dạng kind=DIR,kind=DIR")
        return {kinds[0]: text}
    outs = {}
    for item in text.split(","):
        kind, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--out {item!r}: cần kind=DIR")
        if kind in outs:
            raise ValueError(f"--out lặp embedder {kind!r}")
        outs[kind] = path
    if set(outs) != set(kinds):
        raise ValueError(f"--out có {sorted(outs)} nhưng --embedder là {kinds}")
    return outs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-root", help="thư mục gốc VoxCeleb2 (có các thư mục speaker)")
    ap.add_argument("--out", required=True, help="DIR (một embedder) hoặc kind=DIR,kind=DIR (nhiều)")
    ap.add_argument("--embedder", required=True, help="một hoặc nhiều tên, cách nhau bằng dấu phẩy: "
                                                      + ", ".join(E.KINDS))
    ap.add_argument("--embedder-id", default=None, help="tên ghi vào extract_meta.json (chỉ với một embedder)")
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
    ap.add_argument("--merge", action="store_true", help="chỉ ghép chunks thành train.npz (một embedder)")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    kinds = [x.strip() for x in a.embedder.split(",") if x.strip()]
    for k in kinds:
        if k not in E.KINDS:
            raise SystemExit(f"embedder {k!r} không có trong {E.KINDS}")
    if a.merge:
        if len(kinds) != 1 or "=" in a.out:
            raise SystemExit("--merge cần đúng một embedder và --out là một thư mục")
        meta = merge(a.out, a.embedder_id or kinds[0])
        print(f"wrote {a.out}/train.npz: embedder={meta['embedder_id']} D={meta['embedder_dim']} "
              f"speakers={meta['n_speakers']} embeddings={meta['n_embeddings']}")
        return 0
    if not a.audio_root:
        raise SystemExit("--audio-root là bắt buộc khi trích (không dùng --merge)")
    outs = parse_outs(a.out, kinds)
    adapters = {}
    for k in kinds:
        kw = {}
        if k == "wespeaker_r293":
            kw["r293_onnx"] = a.r293_onnx
        if k == "wespeaker_simam100":
            kw = {"simam_onnx": a.simam_onnx, "simam_dir": a.simam_dir}
        if k == "espnet_spk":
            kw["espnet_tag"] = a.espnet_tag
        adapters[k] = E.make_embedder(k, device=a.device, **kw)
    for ad in adapters.values():
        ad.load()
    totals = extract_multi(a.audio_root, outs, adapters, ext=a.ext, per_speaker=a.per_speaker, min_s=a.min_s,
                           seed=a.seed, limit_speakers=a.limit_speakers, workers=a.workers,
                           shard=parse_shard(a.shard))
    for k in kinds:
        print(f"xong {k}: {totals[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
