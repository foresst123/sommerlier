"""Check a rendered mixture dataset and compare it with the measured real overlaps.

Hard checks (exit code 1 if any fails): every file exists, sample rate and length match
(a BSS chunk is exactly window * sr samples), nothing is NaN or clipped, mix == s1 + s2
(+ noise when a noise stem was saved; a noisy chunk without one is skipped), each stem
holds speech, the `slots` / `layout` of every row follow the shared layout rule
(utils/separation_layout.check_layout), no recording appears in both splits, no excluded
test recording was used, and every TSE row's enrollment candidates come from the same
recording, outside every source range of its mix, in the mix's split.
Then it prints the rendered distribution next to the real one (pair-level overlap from
pairs.csv, pattern, level and noise from summary.json), so a mismatch is visible before
any training.

    python tools/validate_mix_dataset.py --data mix_16k_w4 --stats overlap_stats \\
        --exclude-index overlap_samples/index.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_mix_dataset as rmd  # noqa: E402
from utils import separation_layout as sl  # noqa: E402


def load_rows(data: str, name: str = "chunks.jsonl"):
    p = os.path.join(data, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tse(data: str):
    rows = []
    for s in ("train", "val"):
        rows += load_rows(os.path.join(data, "manifests", "tse"), f"{s}.jsonl")
    return rows


def check_audio(r: dict, stems=("s1", "s2")) -> list:
    import soundfile as sf
    problems, sig = [], {}
    for tag in ("mix",) + tuple(stems) + ("noise",):
        p = r["paths"].get(tag)
        if p is None and tag == "noise":
            continue
        if not p or not os.path.exists(p):
            problems.append(f"{r['id']}: missing {tag}")
            return problems
        x, sr = sf.read(p, dtype="float64")
        if sr != r["sr"] or len(x) != r["samples"]:
            problems.append(f"{r['id']}: {tag} sr/length {sr}/{len(x)} != {r['sr']}/{r['samples']}")
            return problems
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) >= 0.9999:
            problems.append(f"{r['id']}: {tag} NaN or clipped")
        sig[tag] = x
    total = sig[stems[0]] + sig[stems[1]]
    if "noise" in sig:
        total = total + sig["noise"]
    if ("noise" in sig or r.get("noise_snr_db") is None) and np.max(np.abs(sig["mix"] - total)) > 1e-3:
        problems.append(f"{r['id']}: mix != sum of stems" + (" + noise" if "noise" in sig else ""))
    for tag in stems:
        if rmd.active_seconds(sig[tag], r["sr"]) < rmd.MIN_ACTIVE - 1e-6:
            problems.append(f"{r['id']}: {tag} has no speech")
    return problems


def check_tse(trows, mixes: dict) -> list:
    """Enrollment candidates: same recording, outside the mix's source ranges, same split."""
    problems = []
    for t in trows:
        m = mixes.get(t["mix_id"])
        if m is None:
            problems.append(f"{t['id']}: mix {t['mix_id']} not found")
            continue
        if t["split"] != m["split"] or t["recording"] != m["recording"]:
            problems.append(f"{t['id']}: split/recording differ from its mix")
        if os.path.basename(os.path.dirname(t["enroll_bank"])) != rmd._safe(m["recording"]):
            problems.append(f"{t['id']}: enrollment bank from another recording")
        spans = [(p["src_start"], p["src_end"]) for p in m["layout"]]
        for c in t["enroll_candidates"]:
            if any(c["src_start"] < b and c["src_end"] > a for a, b in spans):
                problems.append(f"{t['id']}: enrollment candidate {c['src_start']}-{c['src_end']} inside the mix")
        cand = t["enroll_candidates"]
        if not t["enroll_default"] or any(not 0 <= i < len(cand) for i in t["enroll_default"]):
            problems.append(f"{t['id']}: bad enroll_default")
        elif sum(cand[i]["len"] for i in t["enroll_default"]) < rmd.ENROLL_MIN * t["sr"] - 1:
            problems.append(f"{t['id']}: enrollment shorter than {rmd.ENROLL_MIN} s")
    pairs = {}
    for t in trows:
        pairs.setdefault(t["mix_id"], []).append(t["target_role"])
    for k, v in pairs.items():
        if len(v) != len(set(v)) or len(v) > 2:
            problems.append(f"{k}: TSE rows {v}")
    return problems


def check(rows, check_n: int = 400, seed: int = 0, held=frozenset(), tse_rows=(), tse_mixes=()):
    problems = []
    rng = np.random.default_rng(seed)
    for r in rows:
        if "window" in r and r["samples"] != int(round(r["window"] * r["sr"])):
            problems.append(f"{r['id']}: {r['samples']} samples != window {r['window']} s * {r['sr']}")
        if r.get("layout"):
            problems += sl.check_layout(r)
    for m in tse_mixes:
        problems += sl.check_layout(m)
    pick = rows if len(rows) <= check_n else [rows[i] for i in rng.choice(len(rows), check_n, replace=False)]
    for r in pick:
        problems += check_audio(r)
    for m in list(tse_mixes)[:check_n]:
        problems += check_audio(m, ("A", "B"))
    by = {}
    for r in rows:
        by.setdefault(r["recording"], set()).add(r["split"])
    both = [k for k, v in by.items() if len(v) > 1]
    if both:
        problems.append(f"recordings in both splits: {both[:5]}")
    used = held & set(by)
    if used:
        problems.append(f"excluded test recordings were used: {sorted(used)[:5]}")
    if tse_rows:
        mixes = {r["id"]: r for r in rows} | {m["id"]: m for m in tse_mixes}
        problems += check_tse(tse_rows, mixes)
    return problems


def load_pair_overlaps(stats: str):
    p = os.path.join(stats, "pairs.csv")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return np.array([float(r["overlap"]) for r in csv.DictReader(f)], float)


def compare(rows, real: dict, pair_overlap=None) -> str:
    q = lambda a, p: round(float(np.percentile(a, p)), 2) if len(a) else None  # noqa: E731
    L = np.array([r.get("L", r["overlap_in_window"]) for r in rows], float)
    tmpl = np.array([r["template_overlap"] for r in rows], float)
    share = np.array([(r["core"][1] - r["core"][0]) / r["samples"] for r in rows if r.get("core")], float)
    pat = {p: float(np.mean([r["pattern"] == p for r in rows])) for p in ("full", "contained", "partial")}
    delta = np.abs([r["delta_db"] for r in rows if r["delta_db"] is not None])
    snr = [r["noise_snr_db"] for r in rows if r["noise_snr_db"] is not None]
    fb = float(np.mean([r.get("fallback") is not None for r in rows]))
    solo = float(np.mean([len(r.get("solo_order") or []) == 2 for r in rows]))
    W = rows[0]["window"]
    l_min = (rows[0].get("layout_cfg") or {}).get("l_min", 0.25)
    if pair_overlap is not None and len(pair_overlap):
        kept = pair_overlap[(pair_overlap >= l_min) & (pair_overlap <= W)]
        real_med, real_p90 = f"{q(pair_overlap, 50)} (in [{l_min}, {W}]: {q(kept, 50)})", \
            f"{q(pair_overlap, 90)} (in range: {q(kept, 90)})"
        dropped = f"{float(np.mean((pair_overlap < l_min) | (pair_overlap > W))):.3f}"
    else:
        d = real.get("event_duration_s", {})
        real_med, real_p90, dropped = f"{d.get('p50')} (events)", f"{d.get('p90')} (events)", "n/a"
    dl = real.get("later_minus_earlier_speaker_db", {})
    lines = [f"chunks {len(rows)} ({len(rows) * W / 3600:.1f} h, W = {W} s); kinds " +
             ", ".join(f"{k} {sum(r['kind'] == k for r in rows)}" for k in ("real", "strat", "hard")),
             "| measure | rendered | real |", "|---|---|---|",
             f"| overlap length (pair), median (s) | {q(L, 50)} (template {q(tmpl, 50)}) | {real_med} |",
             f"| overlap length (pair), p90 (s) | {q(L, 90)} (template {q(tmpl, 90)}) | {real_p90} |",
             f"| real pairs outside [l_min, W] (never drawn) | - | {dropped} |",
             f"| core share of the window, median | {q(share, 50)} | n/a |",
             f"| chunks kept through a fallback | {fb:.3f} | n/a |",
             f"| chunks with solo of both speakers | {solo:.3f} | n/a |"]
    for p in ("full", "contained", "partial"):
        lines.append(f"| pattern {p} | {pat[p]:.3f} | {real.get('pattern', {}).get(p, {}).get('share')} |")
    lines += [f"| abs level difference, median (dB) | {q(delta, 50)} | {dl.get('abs_median')} |",
              f"| noise added in | {len(snr) / len(rows):.2f} of chunks, SNR median {q(snr, 50)} dB | "
              f"speech-to-noise {real.get('speech_to_noise_db', {}).get('p50')} dB per recording |"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--exclude-index")
    p.add_argument("--check-n", type=int, default=400, help="chunks whose audio is read and checked")
    a = p.parse_args()
    rows = load_rows(a.data)
    if not rows:
        print(f"no chunks in {a.data}")
        sys.exit(1)
    held = frozenset()
    if a.exclude_index:
        with open(a.exclude_index, encoding="utf-8") as f:
            held = frozenset(os.path.basename(s["recording"].rstrip("/")) for s in json.load(f)["samples"])
    problems = check(rows, a.check_n, held=held, tse_rows=load_tse(a.data),
                     tse_mixes=load_rows(a.data, "tse_mixes.jsonl"))
    real = {}
    if os.path.exists(os.path.join(a.stats, "summary.json")):
        with open(os.path.join(a.stats, "summary.json"), encoding="utf-8") as f:
            real = json.load(f)
    print(compare(rows, real, load_pair_overlaps(a.stats)))
    rp = os.path.join(a.data, "render_report.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            rep = json.load(f)
        print("\nrender report:", json.dumps({k: rep.get(k) for k in ("splits", "multi_overlap", "warnings")},
                                             ensure_ascii=False))
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for x in problems[:30]:
            print(" -", x)
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
