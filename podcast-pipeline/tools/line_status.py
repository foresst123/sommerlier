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


STALE_CLAIM_MINUTES = 15       # a claim not stamped for this long: its copy is gone, or runs without heartbeats


def frozen_list(out):
    """(audio_dir, entries) of <out>/_resident/files.json; ("", []) when there is none or it is unreadable."""
    try:
        data = json.loads((out / "_resident" / "files.json").read_text(encoding="utf-8"))
        return str(data.get("audio_dir") or ""), [e for e in data.get("files") or [] if isinstance(e, dict)]
    except (OSError, ValueError, AttributeError):
        return "", []


def frozen_warning(out, rr):
    """A line saying the audio dir changed since files.json was frozen (what to delete), or None."""
    audio, entries = frozen_list(out)
    path = out / "_resident" / "files.json"
    if not entries or not audio or not Path(audio).is_dir():
        return None
    exts = rr.rv().AUDIO_EXTS
    now = {p.name: p.stat().st_size for p in Path(audio).iterdir() if p.is_file() and p.suffix.lower() in exts}
    frozen = {e.get("name"): e.get("size") for e in entries}
    added = sorted(set(now) - set(frozen))
    changed = sorted(n for n in frozen if n in now and now[n] != frozen[n])
    missing = sorted(set(frozen) - set(now))
    if not (added or changed or missing):
        return None
    return (f"WARNING: audio-dir changed since {path.name} was frozen (added: {', '.join(added) or '-'}; "
            f"size changed: {', '.join(changed) or '-'}; gone: {', '.join(missing) or '-'}). The frozen list is used; "
            f"delete {path} and run the three jobs again to apply it.")


def file_stems(out):
    """Stems the line works on: the frozen list, else the files of the newest input dir a process linked."""
    _, entries = frozen_list(out)
    if entries:
        return sorted(Path(e.get("name", "")).stem for e in entries if e.get("name"))
    dirs = sorted((out / "_resident").glob("input_*"), key=lambda p: p.stat().st_mtime)
    return sorted({f.stem for f in dirs[-1].iterdir()}) if dirs else []


def cell(vdir, stage, stems=()):
    """(text, failure note). Per-file markers (.ready_<stage>.<stem>) when there are stems and no variant-level one."""
    ready, failed = vdir / f".ready_{stage}", vdir / f".failed_{stage}"
    if failed.is_file():
        return "FAIL", f"{stage}: {_reason(failed)} (failed {age(failed)} ago)"
    if ready.is_file():
        return f"ok {age(ready)}", None
    marks = [vdir / f".ready_{stage}.{st}" for st in stems]
    bad = [vdir / f".failed_{stage}.{st}" for st in stems if (vdir / f".failed_{stage}.{st}").is_file()]
    if bad:
        return "FAIL", "; ".join(f"{stage} {b.name.split('.', 2)[-1]}: {_reason(b)} (failed {age(b)} ago)"
                                 for b in bad)
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


def claims(out, rr=None):
    """(variant, stage, stem, holder text or None when finished) for every .claim_<stage>.<stem> file.

    A claim of a copy on this host whose pid is gone is marked: the next --claim copy here removes it. A claim not
    stamped for STALE_CLAIM_MINUTES is marked too (no heartbeat: the copy is gone, or runs without heartbeats).
    """
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
            note = ""
            if rr is not None and rr.dead_claim_here(c):
                note = "; pid gone on this host: the next --claim copy here removes it"
            else:
                try:
                    idle = max(0.0, time.time() - c.stat().st_mtime) / 60
                except OSError:
                    idle = 0.0
                if idle >= STALE_CLAIM_MINUTES:
                    note = f"; no heartbeat for {idle:.0f} min (gone, or a copy without heartbeats)"
            rows.append((vdir.name, stage, stem, f"held by {holder}, {age(c)} ago{note}"))
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
    rr = _resident()
    warn = frozen_warning(out, rr)
    if warn:
        lines.append(warn + "\n")
    ids = variant_ids(out)
    stems = file_stems(out)
    entries = manifest_entries(out)
    music = "profile" if (out / "_resident" / "music_profile").exists() else "off"
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
    held = [r for r in claims(out, rr) if r[3]]
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
