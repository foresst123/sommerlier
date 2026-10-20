"""Check a rendered mixture dataset and compare it with the measured real overlaps.

Hard checks (exit code 1 if any fails): every file exists, sample rate and length match,
nothing is NaN or clipped, a clean chunk (no noise) has mix == s1 + s2, each stem holds
speech, no recording appears in both splits, no excluded test recording was used.
Then it prints the rendered distribution next to the real one from
measure_real_overlap_stats.py, so a mismatch is visible before any training.

    python tools/validate_mix_dataset.py --data mix_16k_w4 --stats overlap_stats \\
        --exclude-index overlap_samples/index.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_mix_dataset as rmd  # noqa: E402


def load_rows(data: str):
    with open(os.path.join(data, "chunks.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check(rows, check_n: int = 400, seed: int = 0, held=frozenset()):
    import soundfile as sf
    problems = []
    rng = np.random.default_rng(seed)
    pick = rows if len(rows) <= check_n else [rows[i] for i in rng.choice(len(rows), check_n, replace=False)]
    for r in pick:
        sig = {}
        for tag in ("mix", "s1", "s2"):
            p = r["paths"][tag]
            if not os.path.exists(p):
                problems.append(f"{r['id']}: missing {tag}")
                break
            x, sr = sf.read(p, dtype="float64")
            if sr != r["sr"] or len(x) != r["samples"]:
                problems.append(f"{r['id']}: {tag} sr/length {sr}/{len(x)} != {r['sr']}/{r['samples']}")
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) >= 0.9999:
                problems.append(f"{r['id']}: {tag} NaN or clipped")
            sig[tag] = x
        if len(sig) == 3:
            if r["noise_snr_db"] is None and np.max(np.abs(sig["mix"] - sig["s1"] - sig["s2"])) > 1e-3:
                problems.append(f"{r['id']}: clean chunk but mix != s1 + s2")
            for tag in ("s1", "s2"):
                if rmd.active_seconds(sig[tag], r["sr"]) < rmd.MIN_ACTIVE - 1e-6:
                    problems.append(f"{r['id']}: {tag} has no speech")
    by = {}
    for r in rows:
        by.setdefault(r["recording"], set()).add(r["split"])
    both = [k for k, v in by.items() if len(v) > 1]
    if both:
        problems.append(f"recordings in both splits: {both[:5]}")
    used = held & set(by)
    if used:
        problems.append(f"excluded test recordings were used: {sorted(used)[:5]}")
    return problems


def compare(rows, real: dict) -> str:
    ov = np.array([r["overlap_in_window"] for r in rows], float)
    full = np.array([r["template_overlap"] for r in rows], float)
    pat = {p: float(np.mean([r["pattern"] == p for r in rows])) for p in ("full", "contained", "partial")}
    delta = np.abs([r["delta_db"] for r in rows if r["delta_db"] is not None])
    snr = [r["noise_snr_db"] for r in rows if r["noise_snr_db"] is not None]
    q = lambda a, p: round(float(np.percentile(a, p)), 2) if len(a) else None  # noqa: E731
    d, dl = real["event_duration_s"], real["later_minus_earlier_speaker_db"]
    lines = [f"chunks {len(rows)} ({len(rows) * rows[0]['window'] / 3600:.1f} h); kinds " +
             ", ".join(f"{k} {sum(r['kind'] == k for r in rows)}" for k in ("real", "strat", "hard")),
             "| measure | rendered | real |", "|---|---|---|",
             f"| overlap length, median (s) | {q(full, 50)} | {d.get('p50')} |",
             f"| overlap length, p90 (s) | {q(full, 90)} | {d.get('p90')} |",
             f"| overlap kept in window, median (s) | {q(ov, 50)} | n/a |"]
    for p in ("full", "contained", "partial"):
        lines.append(f"| pattern {p} | {pat[p]:.3f} | {real['pattern'][p]['share']} |")
    lines += [f"| abs level difference, median (dB) | {q(delta, 50)} | {dl.get('abs_median')} |",
              f"| noise added in | {len(snr) / len(rows):.2f} of chunks, SNR median {q(snr, 50)} dB | "
              f"speech-to-noise {real['speech_to_noise_db'].get('p50')} dB per recording |"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--exclude-index")
    p.add_argument("--check-n", type=int, default=400, help="chunks whose audio is read and checked")
    a = p.parse_args()
    rows = load_rows(a.data)
    held = frozenset()
    if a.exclude_index:
        with open(a.exclude_index, encoding="utf-8") as f:
            held = frozenset(os.path.basename(s["recording"].rstrip("/")) for s in json.load(f)["samples"])
    problems = check(rows, a.check_n, held=held)
    with open(os.path.join(a.stats, "summary.json"), encoding="utf-8") as f:
        print(compare(rows, json.load(f)))
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for x in problems[:30]:
            print(" -", x)
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
