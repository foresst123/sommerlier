"""Score separated estimates of a rendered dataset, on the core region first.

SI-SDRi over the whole window is mostly decided by the solo and ctx parts, where only one
speaker talks, and it rises with W. This script reports the core-region SI-SDRi (the
overlap +- 0.1 s, read from each row's `layout`; utils/separation_layout.core_si_sdri)
next to the full-window number, split by pattern, core-length bin and role (holder /
interrupter), so models with different W can be compared on what they must learn.

BSS: the estimates of row <id> are --est-pattern with k = 1, 2 (default <id>_est1.wav,
<id>_est2.wav) under --est. The permutation is chosen on the full window (PIT), then kept
for the core score. TSE (--tse): one estimate per manifests/tse row, --tse-pattern
(default <id>.wav); no permutation.

    python tools/eval_separation.py --data mix_16k_w4 --est est_mossformer2 --out eval.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_mix_dataset as val  # noqa: E402
from utils import separation_layout as sl  # noqa: E402

ROLES = {"holder,interrupter": ("holder", "interrupter"), "interrupter,holder": ("interrupter", "holder")}


def _read(path: str, n: int) -> np.ndarray:
    import soundfile as sf
    x, _ = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def l_bin(L: float, window: float, l_min: float) -> str:
    bins = sl.len_bins(window, l_min)
    for k, (lo, hi) in enumerate(bins):
        if sl.in_bin(L, lo, hi, k == len(bins) - 1):
            return f"{lo:g}-{hi:g}"
    return "out"


def evaluate_bss(rows: List[dict], est_path: Callable[[str, int], str], pad: float = 0.1) -> List[dict]:
    out = []
    for r in rows:
        n, sr = r["samples"], r["sr"]
        p1, p2 = est_path(r["id"], 1), est_path(r["id"], 2)
        if not (os.path.exists(p1) and os.path.exists(p2)):
            continue
        mix = _read(r["paths"]["mix"], n)
        ref = [_read(r["paths"]["s1"], n), _read(r["paths"]["s2"], n)]
        est = [_read(p1, n), _read(p2, n)]
        keep = sl.si_sdr(est[0], ref[0]) + sl.si_sdr(est[1], ref[1])
        swap = sl.si_sdr(est[1], ref[0]) + sl.si_sdr(est[0], ref[1])
        if swap > keep:
            est = est[::-1]
        cfg = r.get("layout_cfg") or {}
        for k in range(2):
            out.append({"id": r["id"], "split": r["split"], "pattern": r["pattern"], "kind": r.get("kind"),
                        "L": r["L"], "L_bin": l_bin(r["L"], r["window"], cfg.get("l_min", 0.25)),
                        "role": ROLES.get(r.get("target_order"), ("s1", "s2"))[k],
                        "fallback": r.get("fallback"),
                        "full_si_sdri": sl.si_sdr(est[k], ref[k]) - sl.si_sdr(mix, ref[k]),
                        "core_si_sdri": sl.core_si_sdri(est[k], ref[k], mix, tuple(r["core"]), sr, pad)})
    return out


def evaluate_tse(trows: List[dict], mixes: Dict[str, dict], est_path: Callable[[str], str],
                 pad: float = 0.1) -> List[dict]:
    out = []
    for t in trows:
        m = mixes.get(t["mix_id"])
        p = est_path(t["id"])
        if m is None or not os.path.exists(p):
            continue
        n, sr = t["samples"], t["sr"]
        mix, ref, est = _read(t["mix"], n), _read(t["target"], n), _read(p, n)
        w = m.get("window", n / sr)
        out.append({"id": t["id"], "split": t["split"], "pattern": m["pattern"], "kind": m.get("kind"),
                    "L": m["L"], "L_bin": l_bin(m["L"], w, (m.get("layout_cfg") or {}).get("l_min", 0.25)),
                    "role": t["target_role"], "fallback": m.get("fallback"),
                    "full_si_sdri": sl.si_sdr(est, ref) - sl.si_sdr(mix, ref),
                    "core_si_sdri": sl.core_si_sdri(est, ref, mix, tuple(m["core"]), sr, pad)})
    return out


def summarize(res: List[dict], by=("pattern", "L_bin", "role")) -> str:
    lines = ["| group | n | core SI-SDRi mean | core median | full SI-SDRi mean |", "|---|---|---|---|---|"]

    def row(name, items):
        c = np.array([x["core_si_sdri"] for x in items])
        f = np.array([x["full_si_sdri"] for x in items])
        lines.append(f"| {name} | {len(items)} | {c.mean():.2f} | {np.median(c):.2f} | {f.mean():.2f} |")

    if not res:
        return "no estimates found"
    row("all", res)
    for key in by:
        groups = defaultdict(list)
        for x in res:
            groups[x[key]].append(x)
        for g in sorted(groups, key=str):
            row(f"{key}={g}", groups[g])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="rendered dataset (chunks.jsonl)")
    p.add_argument("--est", required=True, help="folder with the estimates")
    p.add_argument("--est-pattern", default="{id}_est{k}.wav")
    p.add_argument("--tse", action="store_true", help="score manifests/tse rows instead of BSS chunks")
    p.add_argument("--tse-pattern", default="{id}.wav")
    p.add_argument("--split", default="val", help="train, val or all")
    p.add_argument("--pad", type=float, default=0.1, help="seconds around the core in the core score")
    p.add_argument("--out", help="CSV with one line per scored source")
    a = p.parse_args()
    rows = val.load_rows(a.data)
    if a.tse:
        trows = [t for t in val.load_tse(a.data) if a.split == "all" or t["split"] == a.split]
        mixes = {r["id"]: r for r in rows} | {m["id"]: m for m in val.load_rows(a.data, "tse_mixes.jsonl")}
        res = evaluate_tse(trows, mixes, lambda i: os.path.join(a.est, a.tse_pattern.format(id=i)), a.pad)
    else:
        rows = [r for r in rows if a.split == "all" or r["split"] == a.split]
        res = evaluate_bss(rows, lambda i, k: os.path.join(a.est, a.est_pattern.format(id=i, k=k)), a.pad)
    print(summarize(res))
    if a.out and res:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0]))
            w.writeheader()
            w.writerows(res)


if __name__ == "__main__":
    main()
