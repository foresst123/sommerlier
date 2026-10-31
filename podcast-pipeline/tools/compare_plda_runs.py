#!/usr/bin/env python3
"""So DER theo từng mẫu của cùng một arm giữa hai lần chạy (giai đoạn 3: PLDA gốc và PLDA tính lại).

    python tools/compare_plda_runs.py OUT_BASELINE OUT_CANDIDATE --arm nemotron_dz \
        [--max-mean-diff 0.5] [--csv per_sample_plda.csv] [--postprocess]

Cả hai thư mục là diar_arms đã chạy với nhãn tay (labels.json). Mẫu chỉ có ở một bên bị bỏ và được báo.
Hiệu = candidate − baseline, dương nghĩa là lần mới sai hơn.
Đạt khi |trung bình hiệu| ≤ --max-mean-diff (mặc định 0,5 điểm DER, theo kế hoạch). Thoát 0 nếu đạt, 1 nếu không.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_arms as ca  # noqa: E402


def der_by_sample(root: str, arm: str, postprocess: bool) -> dict[str, float]:
    rows = ca.run(root, "labels", [arm], None, postprocess)
    return {r["sample"]: float(r["der"]) for r in rows if not math.isnan(float(r["der"]))}


def compare(base: dict[str, float], cand: dict[str, float]) -> dict:
    common = sorted(set(base) & set(cand))
    diffs = {s: cand[s] - base[s] for s in common}
    values = list(diffs.values())
    if not values:
        return {"n": 0, "common": [], "only_base": sorted(set(base) - set(cand)),
                "only_cand": sorted(set(cand) - set(base)), "diffs": {}}
    return {
        "n": len(values),
        "common": common,
        "base_mean": st.mean(base[s] for s in common),
        "cand_mean": st.mean(cand[s] for s in common),
        "mean": st.mean(values),
        "median": st.median(values),
        "max_abs": max(abs(v) for v in values),
        "n_worse_over_1": sum(v > 1.0 for v in values),
        "n_better_over_1": sum(v < -1.0 for v in values),
        "only_base": sorted(set(base) - set(cand)),
        "only_cand": sorted(set(cand) - set(base)),
        "diffs": diffs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--max-mean-diff", type=float, default=0.5)
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)

    base = der_by_sample(a.baseline, a.arm, a.postprocess)
    cand = der_by_sample(a.candidate, a.arm, a.postprocess)
    res = compare(base, cand)
    if res["n"] == 0:
        print("không có mẫu chung nào có nhãn và có arm này trong cả hai thư mục", file=sys.stderr)
        return 1

    print(f"arm {a.arm}: {res['n']} mẫu chung")
    print(f"  DER TB: baseline={res['base_mean']:.3f}  candidate={res['cand_mean']:.3f}")
    print(f"  hiệu (candidate − baseline): TB={res['mean']:+.3f}  TV={res['median']:+.3f}  "
          f"max|hiệu|={res['max_abs']:.2f}")
    print(f"  mẫu xấu đi quá 1 điểm: {res['n_worse_over_1']}   mẫu tốt lên quá 1 điểm: {res['n_better_over_1']}")
    if res["only_base"]:
        print(f"  chỉ có ở baseline (bỏ qua): {', '.join(res['only_base'])}")
    if res["only_cand"]:
        print(f"  chỉ có ở candidate (bỏ qua): {', '.join(res['only_cand'])}")
    for s, d in sorted(res["diffs"].items(), key=lambda kv: -abs(kv[1]))[:5]:
        print(f"    {s:28s} baseline={base[s]:6.2f} candidate={cand[s]:6.2f} hiệu={d:+.2f}")

    ok = abs(res["mean"]) <= a.max_mean_diff
    print(f"{'ĐẠT' if ok else 'KHÔNG ĐẠT'}: |TB hiệu| {abs(res['mean']):.3f} "
          f"{'≤' if ok else '>'} {a.max_mean_diff}")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sample", "baseline_der", "candidate_der", "diff"])
            for s in res["common"]:
                w.writerow([s, round(base[s], 3), round(cand[s], 3), round(res["diffs"][s], 3)])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
