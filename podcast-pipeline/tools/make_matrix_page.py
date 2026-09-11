#!/usr/bin/env python3
"""Build a portable matrix report, including unmeasured cells and listening notes."""

import argparse
import base64
from collections import defaultdict
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.matrix_axes import AXES, POLISH_SUFFIX, STAGE_DIR
from tools.matrix_io import read_json
from tools.metrics.aggregate import METRICS, aggregate_records, marginal_effects, metric_values
from tools.metrics.dnsmos_scorer import WARNINGS
from tools.metrics.run_metrics import latest_records


def build_payload(matrix):
    records = list(latest_records(matrix / "metrics.jsonl").values())
    by_file, summaries = aggregate_records(records)
    manifest = read_json(matrix / "manifest.json", {})
    cells = manifest.get("cells") or [dict(id=combo, axes={}, jobs={}) for combo in sorted({r["combo"] for r in records})]
    state = read_json(matrix / "state.json", {})
    rows = []
    for cell in cells:
        subset = [r for r in records if r["combo"] == cell["id"]]
        errors = [r.get("error", "Measurement missing") for r in subset if r.get("status") != "done"]
        files = []
        stages_done = True
        for stem in manifest.get("inputs", {}):
            sep_job, diar_job = cell["jobs"]["separation"], cell["jobs"]["diarization"]
            status = state.get(f"{sep_job}/{stem}", {})
            if status.get("status") != "done":
                stages_done = False
                errors.append(f"{stem}: {status.get('error') or status.get('status', 'pending')}")
            directory = matrix / "out" / sep_job / stem / STAGE_DIR["separation"]
            files.append(dict(stem=stem,
                diarization=read_json(matrix / "out" / diar_job / stem / STAGE_DIR["diarization"] / "stats.json", {}),
                separation=read_json(directory / "report.json", {}),
                stage_stats=read_json(directory / "stats.json", {})))
        windows = defaultdict(list)
        for record in subset:
            windows[(record["stem"], record["tag"])].append(record)
        clips = []
        for (stem, tag), group in sorted(windows.items()):
            paths = {r["track"]: r["path"] for r in group}
            mix = paths.get("mix")
            audio = {"mix": mix}
            if mix:
                base = mix.removesuffix("_mix.wav")
                audio.update({"raw A": base + "_trackA.wav", "raw B": base + "_trackB.wav"})
            if cell.get("axes", {}).get("F", "raw") != "raw":
                audio.update({"polish A": paths.get("A"), "polish B": paths.get("B")})
            clips.append(dict(id=f"{cell['id']}|{stem}|{tag}", stem=stem, tag=tag,
                              audio=audio, metrics={r["track"]: metric_values(r) for r in group},
                              errors=[r.get("error") for r in group if r.get("error")]))
        measured = bool(subset) and any(r.get("track") in ("A", "B") and r.get("status") == "done" for r in subset)
        status = "measured" if measured and stages_done and not errors else "partial" if measured else "pending"
        rows.append(dict(id=cell["id"], axes=cell["axes"], metrics=summaries.get(cell["id"], {}),
                         status=status, reasons=errors or ([] if measured else ["No measurements yet"]),
                         files=files, clips=clips))
    return dict(title="Sommelier Matrix", axes=AXES, metrics={k: v[1] for k, v in METRICS.items()},
                warnings=WARNINGS, rows=rows, effects=marginal_effects(by_file, records),
                by_file=by_file, notes={}, media={})


def embed_audio(payload, output, max_bytes):
    from pydub import AudioSegment
    available = max(0, max_bytes)
    media = payload["media"]
    # One item per cell per round: every combination has a chance at the same
    # budget. Shared raw/mix files are encoded and embedded just once.
    queues = []
    for row in payload["rows"]:
        queues.append([path for clip in row["clips"] for path in clip["audio"].values() if path])
    for index in range(max((len(q) for q in queues), default=0)):
        for paths in queues:
            if index >= len(paths) or paths[index] in media:
                continue
            path = Path(paths[index])
            entry = dict(src=os.path.relpath(path, output.parent), embedded=False)
            if path.is_file() and available > 0:
                try:
                    buffer = io.BytesIO()
                    AudioSegment.from_file(path).set_channels(1).export(buffer, format="mp3", bitrate="48k")
                    uri = "data:audio/mpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                    if len(uri) <= available:
                        entry = dict(src=uri, embedded=True)
                        available -= len(uri)
                except Exception as exc:
                    entry["error"] = str(exc)
            elif not path.is_file():
                entry["error"] = "Audio not produced"
            media[str(path)] = entry


def build_report(matrix, output, max_mb=200):
    payload = build_payload(matrix)
    template = (ROOT / "tools/matrix_report.html").read_text()
    # JSON/markup count toward the limit too, including fallback paths.
    reserve = len(template.encode()) + len(json.dumps(payload).encode()) * 2 + 65536
    embed_audio(payload, output, max(0, int(max_mb * 1024 * 1024) - reserve))
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = template.replace("__MATRIX_PAYLOAD__", encoded)
    if len(html.encode()) > max_mb * 1024 * 1024:
        raise ValueError("Report metadata exceeds --max-mb; increase the budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-mb", type=float, default=200)
    args = parser.parse_args(argv)
    matrix = args.matrix.resolve()
    output = (args.out or matrix / "matrix_report.html").resolve()
    payload = build_report(matrix, output, args.max_mb)
    print(f"{output}: {len(payload['rows'])} cells, {output.stat().st_size / 1048576:.2f} MiB")


if __name__ == "__main__":
    main()
