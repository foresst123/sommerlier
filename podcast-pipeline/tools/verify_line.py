#!/usr/bin/env python3
"""Did every variant really finish, for every file?   python tools/verify_line.py --audio-dir DIR --out-root OUT

Checks, per (variant, file), from what is on disk:
  - 02_diarization/segments.json, 03_separation/segments.json and 05_asr/transcripts.json exist and hold segments
  - every denoise point the variant uses has its denoise.json, with no errors (an error means that filter fell back
    to the unfiltered audio, so the variant says nothing about the filter)
  - the variant's status in variants_manifest.json is ok
For a variant the line has run through (it has any .ready_ marker or the final .done):
  - the hand-over of the new stages exists for every file: separation_plan and conversation_selection
    (.ready_<stage>.<stem>, or the stage's whole .ready_<stage>), and no .failed_ marker is left behind
  - 07d_conversation_selection/selection.json exists and holds the selection
  - the final process (--stop-after all) wrote .done
Exit code 0 only when all of it holds; otherwise the problems are listed. A variant dropped on purpose
(--music off makes `raw` a copy of `none`) is reported as skipped, not as missing.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# The variants the production line runs (tools/job_common.sh uses the same file by default).
DEFAULT_VARIANTS = ROOT / "tools" / "variants_production.json"
STAGE_FILES = {"diarization": "02_diarization/segments.json",
               "separation": "03_separation/segments.json",
               "asr": "05_asr/transcripts.json"}
# outputs of the stages after asr, checked for a variant the line has run through
LATE_FILES = {"conversation_selection": "07d_conversation_selection/selection.json"}
# stages whose hand-over (markers) the line must show for every file
HANDOVER_STAGES = ("separation_plan", "conversation_selection")
POINTS = (("chain", "01b_denoise", "pre_diarization"),
          ("chain_pre_separation", "02b_denoise_pre_separation", "pre_separation"),
          ("chain_pre_asr", "04b_denoise_pre_asr", "pre_asr"))
NOOP_CORR = 0.99999


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def n_rows(data):
    if isinstance(data, dict):
        data = data.get("segments", data.get("transcripts", data.get("selections")))
    return len(data) if isinstance(data, list) else 0


def check_file(fdir, variant):
    """(problems, warnings) for one variant's output folder of one file."""
    problems, warnings = [], []
    if not fdir.is_dir():
        return ["no output folder"], warnings
    for stage, rel in STAGE_FILES.items():
        data = read_json(fdir / rel)
        if data is None:
            problems.append(f"{stage}: {rel} missing or unreadable")
        elif n_rows(data) == 0:
            problems.append(f"{stage}: {rel} holds no segments")
    for key, sub, point in POINTS:
        if not variant.get(key):
            continue
        info = read_json(fdir / sub / "denoise.json")
        if info is None:
            problems.append(f"{point}: {sub}/denoise.json missing (the filter did not run)")
            continue
        if info.get("errors"):
            problems.append(f"{point}: filter fell back to the input: {str(info['errors'][0])[:120]}")
        elif (info.get("corr_with_input") or 0) >= NOOP_CORR and (info.get("steps") or True):
            warnings.append(f"{point}: output identical to the input (corr {info['corr_with_input']:.6f})")
    return problems, warnings


def line_has_run(vdir):
    """A variant the line has touched: a hand-over marker or the final .done is on disk."""
    return (vdir / ".done").is_file() or any(vdir.glob(".ready_*"))


def check_line(vdir, stems):
    """Problems of the line's hand-overs for one variant (only for a variant the line has run through)."""
    problems = []
    if not (vdir / ".done").is_file():
        problems.append("final process did not finish: .done (--stop-after all) is missing")
    for stage in HANDOVER_STAGES:
        total = (vdir / f".ready_{stage}").is_file()
        for stem in stems:
            if (vdir / f".failed_{stage}.{stem}").is_file() or (vdir / f".failed_{stage}").is_file():
                problems.append(f"{stem}: {stage} failed")
            elif not (total or (vdir / f".ready_{stage}.{stem}").is_file()):
                problems.append(f"{stem}: {stage} hand-over missing (no .ready_{stage}.{stem})")
    for stem in stems:
        for stage, rel in LATE_FILES.items():
            if read_json(vdir / stem / rel) is None:
                problems.append(f"{stem}: {stage}: {rel} missing or unreadable")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--variants", default=str(DEFAULT_VARIANTS))
    ap.add_argument("--music", choices=["off", "profile"], default="off")
    a = ap.parse_args(argv)

    rv = load("rv_verify", HERE / "run_variants.py")
    rr = load("rr_verify", ROOT / "run_variants_resident.py")
    out = Path(a.out_root)
    variants = rv.load_variants(a.variants)
    stems = [f.stem for f in rv.collect_files(a.audio_dir, None)]
    manifest = {v["id"]: v for v in (read_json(out / "variants_manifest.json") or {}).get("variants", [])}

    bad, warned, skipped, complete = {}, {}, [], 0
    for v in variants:
        vid = v["id"]
        if a.music == "off" and rr.is_music_off_copy(v):
            skipped.append(vid)
            continue
        problems = []
        status = (manifest.get(vid) or {}).get("status")
        if status != "ok":
            problems.append(f"manifest status is {status!r}, not 'ok'")
        for stem in stems:
            p, w = check_file(out / vid / stem, v)
            problems += [f"{stem}: {x}" for x in p]
            warned.setdefault(vid, []).extend(f"{stem}: {x}" for x in w)
        if line_has_run(out / vid):
            problems += check_line(out / vid, stems)
        if problems:
            bad[vid] = problems
        else:
            complete += 1

    expected = len(variants) - len(skipped)
    print(f"{complete}/{expected} variants complete for all {len(stems)} file(s)"
          + (f"   (skipped on purpose: {', '.join(skipped)})" if skipped else ""))
    for vid, ps in bad.items():
        print(f"\nNOT COMPLETE  {vid}")
        for p in ps[:8]:
            print(f"   - {p}")
        if len(ps) > 8:
            print(f"   ... and {len(ps) - 8} more")
    shown = {k: v for k, v in warned.items() if v and k not in bad}
    for vid, ws in shown.items():
        print(f"\nwarning  {vid}")
        for w in ws[:4]:
            print(f"   - {w}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
