#!/usr/bin/env python3
"""Append measurements to metrics.jsonl, or rescore historical window dumps."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.matrix_axes import POLISH_SUFFIX, STAGE_DIR, fingerprint
from tools.matrix_io import read_json
from tools.metrics.audio import read_audio, selected_windows, sidecar, probe_spans
from tools.metrics.dnsmos_scorer import DNSMOSScorer, WARNINGS
from tools.metrics import f0, spectral


def record_key(row):
    return tuple(row.get(k) for k in ("combo", "stem", "tag", "track", "polish"))


def latest_records(path):
    rows = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                row = json.loads(line)
                rows[record_key(row)] = row
            except json.JSONDecodeError:
                # A killed writer may have left an incomplete final line.
                continue
    return rows


def window_jobs(matrix, dump_root=None, windows=4):
    if dump_root:
        for directory in sorted({p.parent for p in Path(dump_root).rglob("*_mix.wav")
                                 if p.parent.name == "separated"}):
            for mix in selected_windows(directory, windows):
                yield "legacy", {}, directory.parents[3].name, "raw", mix
        return
    manifest = read_json(matrix / "manifest.json")
    if not manifest:
        raise ValueError("No matrix manifest. Use --dump-root for historical dumps.")
    for cell in manifest["cells"]:
        for stem in manifest["inputs"]:
            directory = matrix / "out" / cell["jobs"]["separation"] / stem / STAGE_DIR["separation"] / "audio/raw/separated"
            for mix in selected_windows(directory, windows):
                yield cell["id"], cell["axes"], stem, cell["axes"]["F"], mix


def score_track(path, mix, raw_path, meta, track, dns, ecapa, executor=None):
    reference, sr = read_audio(mix, 24000)
    output, native_sr = read_audio(path)
    if native_sr != sr:
        output, _ = read_audio(path, sr)
    if abs(len(output) - len(reference)) > sr * 0.02:
        raise ValueError("Output duration differs from mixture; probe alignment is invalid")
    n = min(len(output), len(reference))
    output, reference = output[:n], reference[:n]
    spans = probe_spans(meta, track, sr, n) if track in ("A", "B") else None
    result = {"native_sr": native_sr, "comparison_sr": sr,
              "probe_scope": "solo_probes" if meta and track != "mix" else "legacy_full_window_unpaired",
              "spectral": spectral.score(reference, output, sr, spans),
              "dnsmos": None, "f0": None, "ecapa": None}
    pitch = None
    if spans:
        pitch = (executor.submit(f0.score, reference, output, sr, spans) if executor
                 else f0.score(reference, output, sr, spans))
    if dns:
        result["dnsmos"] = dns(output, sr)
        baseline = dns(reference, sr)
        result["dnsmos_mix"] = baseline
        result["dnsmos_delta"] = {k: result["dnsmos"][k] - baseline[k] for k in ("SIG", "BAK", "OVRL")}
    if ecapa and track in ("A", "B"):
        own = ecapa.embed(output, sr)
        raw, _ = read_audio(raw_path, sr)
        suffix = str(path).removeprefix(str(raw_path).removesuffix(".wav"))
        other_track = "B" if track == "A" else "A"
        other_path = Path(str(mix).removesuffix("_mix.wav") + f"_track{other_track}" + suffix)
        other = ecapa.embed(read_audio(other_path, sr)[0], sr) if other_path.exists() else None
        enroll_path = Path(str(mix).removesuffix("_mix.wav") + f"_enroll{track}.wav")
        enroll = None
        if meta and enroll_path.exists():
            enrollment, enrollment_sr = read_audio(enroll_path)
            lengths = meta.get("enrollment_lengths", [[], []])[0 if track == "A" else 1]
            enroll = ecapa.enrollment(enrollment, enrollment_sr, lengths)
        probe = np.concatenate([output[a:b] for a, b in spans]) if spans else None
        result["ecapa"] = {
            "sim_enroll": ecapa.similarity(ecapa.embed(probe, sr) if probe is not None else None, enroll),
            "sim_self": ecapa.similarity(own, ecapa.embed(raw, sr)),
            "sim_cross": ecapa.similarity(own, other)}
    if pitch is not None:
        result["f0"] = pitch.result() if executor else pitch
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dump-root", type=Path)
    parser.add_argument("--windows", type=int, default=4, help="Per prefix and recording; 0 means all")
    parser.add_argument("--dnsmos-model")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--env", default="a100")
    parser.add_argument("--skip-dnsmos", action="store_true")
    parser.add_argument("--skip-ecapa", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args(argv)
    matrix = args.matrix.resolve()
    matrix.mkdir(parents=True, exist_ok=True)
    for message in WARNINGS:
        print(message, file=sys.stderr)
    profile = read_json(args.config, {}).get("environments", {}).get(args.env, {})
    dns = None if args.skip_dnsmos else DNSMOSScorer(args.dnsmos_model, profile)
    ecapa = None
    if not args.skip_ecapa:
        from tools.metrics.ecapa import ECAPAScorer
        ecapa = ECAPAScorer()
    settings = fingerprint(dict(version=1, dns=not args.skip_dnsmos,
                                ecapa=not args.skip_ecapa, model=args.dnsmos_model, profile=profile))
    path = matrix / "metrics.jsonl"
    previous = latest_records(path)
    jobs = list(window_jobs(matrix, args.dump_root, args.windows))
    completed, failures = 0, 0
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
            for combo, axes, stem, polish, mix in jobs:
                tag = mix.name.removesuffix("_mix.wav")
                for track in ("mix", "A", "B"):
                    base = str(mix).removesuffix("_mix.wav")
                    raw_path = mix if track == "mix" else Path(base + f"_track{track}.wav")
                    target = mix if track == "mix" else Path(base + f"_track{track}{POLISH_SUFFIX[polish]}.wav")
                    row = dict(combo=combo, axes=axes, stem=stem, tag=tag, track=track,
                               polish=polish, path=str(target), settings=settings)
                    row["input_stamp"] = [target.stat().st_mtime_ns, target.stat().st_size] if target.exists() else None
                    old = previous.get(record_key(row), {})
                    if not args.rescore and old.get("status") == "done" and old.get("settings") == settings and old.get("input_stamp") == row["input_stamp"]:
                        continue
                    try:
                        row.update(score_track(target, mix, raw_path, sidecar(mix), track, dns, ecapa, executor))
                        row["status"] = "done"
                        completed += 1
                    except Exception as exc:
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                        failures += 1
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                print(f"{combo} / {stem} / {tag}: {completed} scored, {failures} failed", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    print(json.dumps(dict(windows=len(jobs), scored=completed, failed=failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
