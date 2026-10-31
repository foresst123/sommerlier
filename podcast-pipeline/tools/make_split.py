#!/usr/bin/env python3
"""Chia các mẫu có nhãn thành hai nửa cố định (giai đoạn 6): nửa A để chọn tham số, nửa B để báo cáo.

    python tools/make_split.py OUT/diar_arms --seed 0 --out OUT/diar_arms/split_seed0.json
    python tools/analyze_arms_detail.py OUT/diar_arms --arms ... --split OUT/diar_arms/split_seed0.json --half B

Chia ngẫu nhiên có hạt giống trên danh sách đã sắp xếp, nên cùng hạt giống cho cùng kết quả. Nửa A lấy ceil(n/2) mẫu.
File đầu ra là JSON {"seed", "A": [...], "B": [...]}. Không ghi đè file cũ nếu không có --force.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys


def labelled_samples(root: str) -> list[str]:
    return sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isdir(d) and os.path.isfile(os.path.join(d, "labels.json")))


def make_split(names: list[str], seed: int) -> dict[str, list[str]]:
    if len(names) < 2:
        raise ValueError(f"cần ít nhất 2 mẫu để chia, có {len(names)}")
    shuffled = sorted(names)
    random.Random(seed).shuffle(shuffled)
    cut = (len(shuffled) + 1) // 2
    return {"A": sorted(shuffled[:cut]), "B": sorted(shuffled[cut:])}


def load_split(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if set(doc) & {"A", "B"} != {"A", "B"} or set(doc["A"]) & set(doc["B"]):
        raise ValueError(f"{path}: cần hai khoá A và B không trùng nhau")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    if os.path.exists(a.out) and not a.force:
        print(f"{a.out} đã có; dùng --force để ghi đè (một khi đã chọn tham số trên nửa A thì không nên đổi)",
              file=sys.stderr)
        return 1
    names = labelled_samples(a.root)
    split = make_split(names, a.seed)
    doc = {"seed": a.seed, "n": len(names), **split}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"A: {len(split['A'])} mẫu, B: {len(split['B'])} mẫu → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
