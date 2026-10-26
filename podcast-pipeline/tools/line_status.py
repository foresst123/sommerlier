#!/usr/bin/env python3
"""Where is each variant in the 6-process line?  python tools/line_status.py OUT_DIR

One row per variant, one column per hand-over of the line (the .ready_<stage> files the processes write for
each other); `ok` = handed over, `FAIL` = the stage failed (reason below the table), `.` = not yet. The last
line of each process's log is shown underneath, so a process that is waiting says what it waits for.
"""
import json
import sys
import time
from pathlib import Path

STAGES = ("denoise", "diarization", "denoise_pre_separation", "separation", "denoise_pre_asr")
HEAD = ("music+denoise1", "diarization", "denoise2", "separation", "denoise3", "asr")


def age(path):
    s = max(0, time.time() - path.stat().st_mtime)
    return f"{s / 60:.0f}m" if s >= 90 else f"{s:.0f}s"


def variant_ids(out):
    manifest = out / "variants_manifest.json"
    ids = []
    if manifest.is_file():
        try:
            ids = [v["id"] for v in json.loads(manifest.read_text()).get("variants", [])]
        except (ValueError, KeyError, TypeError):
            ids = []
    for d in sorted(p.name for p in out.iterdir() if p.is_dir() and not p.name.startswith(("_", "lanes"))):
        if d not in ids:
            ids.append(d)
    return ids


def cell(vdir, stage):
    ready, failed = vdir / f".ready_{stage}", vdir / f".failed_{stage}"
    if failed.is_file():
        return "FAIL", f"{stage}: " + _reason(failed)
    return (f"ok {age(ready)}", None) if ready.is_file() else (".", None)


def _reason(path):
    try:
        return str(json.loads(path.read_text()).get("reason", ""))[:160]
    except (ValueError, OSError):
        return "?"


def asr_cell(vdir):
    done = vdir / ".done"
    return f"ok {age(done)}" if done.is_file() else "."


def last_lines(out):
    rows = []
    for log in sorted((out / "_resident" / "logs").glob("*.log")):
        text = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
        rows.append((log.stem, f"{age(log)} ago", text[-1][:140] if text else "(empty)"))
    return rows


def render(out):
    lines = []
    ids = variant_ids(out)
    width = max([len(i) for i in ids] + [7])
    lines.append(f"{'variant':<{width}}  " + "  ".join(f"{h:<14}" for h in HEAD))
    problems = []
    for vid in ids:
        vdir = out / vid
        cells = []
        for st in STAGES:
            c, why = cell(vdir, st)
            cells.append(c)
            if why:
                problems.append((vid, why))
        cells.append(asr_cell(vdir))
        lines.append(f"{vid:<{width}}  " + "  ".join(f"{c:<14}" for c in cells))
    if problems:
        lines.append("\nfailures:")
        lines += [f"  {v}: {why}" for v, why in problems]
    lines.append("\nlast log line of each process:")
    for name, when, text in last_lines(out):
        lines.append(f"  {name:<18} {when:<9} {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
        sys.exit(__doc__)
    print(render(Path(sys.argv[1])))
