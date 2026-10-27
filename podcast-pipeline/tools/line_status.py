#!/usr/bin/env python3
"""Where is each variant in the line of processes?  python tools/line_status.py OUT_DIR

One row per variant, one column per hand-over of the line (the .ready_<stage> files the processes write for
each other); `ok` = handed over, `FAIL` = the stage failed (reason below the table), `.` = not yet, `-` = nothing
to do for this variant at that stage (no filter chain there). With --per-file the markers are per file
(.ready_<stage>.<stem>): `ok` when every file has it, `2/3` while only some do. The last column `done` is the
final process (--stop-after all) writing .done. Files a copy of a stage holds (--claim) are listed under `claims`
with the host and pid of the copy. The last line of each process's log is shown underneath, so a process that is
waiting says what it waits for.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

STAGES = ("denoise", "diarization", "denoise_pre_separation", "separation_plan", "separation", "denoise_pre_asr",
          "asr", "conversation_selection")
HEAD = ("music+denoise1", "diarization", "denoise2", "separation_plan", "separation", "denoise3", "asr", "llm",
        "done")


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
    for d in sorted(p.name for p in out.iterdir() if p.is_dir() and not p.name.startswith(("_", ".", "lanes"))):
        if d not in ids:
            ids.append(d)
    return ids


def _resident():
    spec = importlib.util.spec_from_file_location("resident_for_status", Path(__file__).resolve().parents[1]
                                                  / "run_variants_resident.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def file_stems(out):
    """Stems the line works on: the files of the newest input dir a process linked."""
    dirs = sorted((out / "_resident").glob("input_*"), key=lambda p: p.stat().st_mtime)
    return sorted({f.stem for f in dirs[-1].iterdir()}) if dirs else []


def cell(vdir, stage, stems=()):
    """(text, failure note). Per-file markers (.ready_<stage>.<stem>) when there are stems and no variant-level one."""
    ready, failed = vdir / f".ready_{stage}", vdir / f".failed_{stage}"
    if failed.is_file():
        return "FAIL", f"{stage}: " + _reason(failed)
    if ready.is_file():
        return f"ok {age(ready)}", None
    marks = [vdir / f".ready_{stage}.{st}" for st in stems]
    bad = [vdir / f".failed_{stage}.{st}" for st in stems if (vdir / f".failed_{stage}.{st}").is_file()]
    if bad:
        return "FAIL", "; ".join(f"{stage} {b.name.split('.', 2)[-1]}: {_reason(b)}" for b in bad)
    have = [m for m in marks if m.is_file()]
    if not have:
        return ".", None
    if len(have) == len(marks):
        return f"ok {age(max(have, key=lambda m: m.stat().st_mtime))}", None
    return f"{len(have)}/{len(marks)}", None


def _reason(path):
    try:
        return str(json.loads(path.read_text()).get("reason", ""))[:160]
    except (ValueError, OSError):
        return "?"


def asr_cell(vdir, stems=()):
    """The asr process signals .ready_asr (per file or whole); a run that stopped at asr writes .done.asr."""
    done = vdir / ".done.asr"
    if done.is_file():
        return f"ok {age(done)}", None
    return cell(vdir, "asr", stems)


def done_cell(vdir):
    done = vdir / ".done"
    return f"ok {age(done)}" if done.is_file() else "."


def claims(out):
    """(variant, stage, stem, holder text or None when finished) for every .claim_<stage>.<stem> file."""
    rows = []
    for vdir in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith(("_", ".", "lanes"))):
        for c in sorted(vdir.glob(".claim_*")):
            stage, _, stem = c.name[len(".claim_"):].partition(".")
            if (vdir / f".ready_{stage}.{stem}").is_file() or (vdir / f".failed_{stage}.{stem}").is_file():
                rows.append((vdir.name, stage, stem, None))
                continue
            try:
                who = json.loads(c.read_text())
                holder = f"{who.get('host', '?')} pid {who.get('pid', '?')}"
            except (OSError, ValueError):
                holder = "(writing)"
            rows.append((vdir.name, stage, stem, f"held by {holder}, {age(c)} ago"))
    return rows


def last_lines(out):
    rows = []
    for log in sorted((out / "_resident" / "logs").glob("*.log")):
        text = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
        rows.append((log.stem, f"{age(log)} ago", text[-1][:140] if text else "(empty)"))
    return rows


def manifest_entries(out):
    try:
        return {v["id"]: v for v in json.loads((out / "variants_manifest.json").read_text()).get("variants", [])}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def render(out):
    lines = []
    ids = variant_ids(out)
    stems = file_stems(out)
    entries = manifest_entries(out)
    music = "profile" if (out / "_resident" / "music_profile").exists() else "off"
    rr = _resident()
    width = max([len(i) for i in ids] + [7])
    lines.append(f"{'variant':<{width}}  " + "  ".join(f"{h:<14}" for h in HEAD))
    problems = []
    for vid in ids:
        vdir = out / vid
        cells = []
        entry = entries.get(vid)
        for st in STAGES:
            if st == "asr":
                c, why = asr_cell(vdir, stems)
            elif entry is not None and "chain" in entry and rr.is_noop(entry, st, music):
                c, why = "-", None
            else:
                c, why = cell(vdir, st, stems)
            cells.append(c)
            if why:
                problems.append((vid, why))
        cells.append(done_cell(vdir))
        lines.append(f"{vid:<{width}}  " + "  ".join(f"{c:<14}" for c in cells))
    if problems:
        lines.append("\nfailures:")
        lines += [f"  {v}: {why}" for v, why in problems]
    held = [r for r in claims(out) if r[3]]
    if held:
        lines.append("\nclaims (file held by a copy of its stage):")
        lines += [f"  {v:<{width}} {st:<14} {stem}: {who}" for v, st, stem, who in held]
    lines.append("\nlast log line of each process:")
    for name, when, text in last_lines(out):
        lines.append(f"  {name:<18} {when:<9} {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
        sys.exit(__doc__)
    print(render(Path(sys.argv[1])))
