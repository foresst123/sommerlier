#!/usr/bin/env python3
"""Run prefix-shared matrix stages in isolated, resumable subprocesses."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.matrix_axes import (AXES, STAGES, STAGE_DIR, MUSIC_MODELS, DENOISE_MODEL,
    DIARIZEN_MODELS, cells, combo_id, fingerprint, job_id, parents, prefix_axes)
from tools.matrix_io import read_json, write_json


def cell_config(base, profile_name, cell):
    cfg = copy.deepcopy(base)
    profile = cfg["environments"][profile_name]
    profile.setdefault("pipeline", {}).update(by_stage=False, prefetch_workers=False,
        review_page=False, keep_models=False, dia3=False, no_stage_output=False)
    profile.setdefault("steps", {}).update(music_analysis=True, music_removal=True,
        cut_music=cell["C"] == "cut", noise_removal=cell["B"] == "aufr33",
        ina_scan=cell["G"] == "ina", diarization=True, separation=True,
        asr=False, captioning=False, refinement=False, export=False)
    models = profile.setdefault("models", {})
    diar = models.setdefault("diarizen", {})
    diar.update(model=DIARIZEN_MODELS[cell["D"]], min_speakers=1, max_speakers=4)
    # Use each checkpoint's own clusterer and threshold; overriding these can
    # silently turn the MLC treatment back into MD-v2's settings.
    for key in ("clustering_method", "ahc_threshold", "seg_duration"):
        diar.pop(key, None)
    if cell["D"] == "mlc":
        diar.update(min_speakers=2, max_speakers=2)
    models.setdefault("sidon", {})["num_steps"] = 100
    models.setdefault("bss", {}).update(separator="sidon", enrollment_memory=cell["E"] == "mem")
    models["denoise"] = dict(models.get("denoise", {}), model=DENOISE_MODEL,
                             stem="dry", hi_res=False, chunk_duration=300,
                             normalization_threshold=1.0)
    return cfg


def cell_environment(cell):
    env = os.environ.copy()
    env.update(DENOISE_MODEL=DENOISE_MODEL, DIARIZEN_MODEL=DIARIZEN_MODELS[cell["D"]],
        MUSIC_MAP_EXCISE_SINGING=str(int(cell["C"] == "cut")),
        NOISE_EXCISE=str(int(cell["C"] == "cut")),
        BSS_MEMORY=str(int(cell["E"] == "mem")), BSS_DUMP_FAILED="1",
        BSS_SEPARATOR="sidon", SOMMELIER_AUDIO_CACHE="1", PYTHONHASHSEED="0")
    return env


def jobs_for(selected, sources):
    jobs, seen = [], set()
    for stage in STAGES:
        for cell in selected:
            job = job_id(cell, stage)
            for source in sources:
                key = f"{job}/{source.stem}"
                if key not in seen:
                    jobs.append(dict(key=key, job=job, cell=cell, stage=stage,
                                     stem=source.stem, source=str(source)))
                    seen.add(key)
    return jobs


def assert_axes(matrix, job, stem, expected, signature):
    path = matrix / "cache" / f"{job}_{stem}" / "axes.json"
    data = read_json(path)
    if data is None or data.get("prefix") != expected or data.get("signature") != signature:
        raise RuntimeError(f"Checkpoint prefix mismatch or missing metadata: {path}")


def validate_log(log):
    for line in log.splitlines():
        if "Music settings have changed since this file was checkpointed" in line:
            raise RuntimeError("Parent music settings changed")
        if "config_ignored" in line and any(key in line for key in ("clustering_method", "max_speakers", "min_speakers")):
            raise RuntimeError(f"Diarization treatment ignored: {line.strip()}")
        if "Music analysis wants to cut" in line:
            raise RuntimeError("Axis C ineffective: CUT_SHARE_LIMIT vetoed the requested cuts")
        if "Music sweep failed:" in line:
            raise RuntimeError("SSLAM sweep failed; refusing an unmeasured empty music map")


def run_job(matrix, job, base, profile, python, state, signature, timeout):
    stage, cell, stem, jid = job["stage"], job["cell"], job["stem"], job["job"]
    for parent_stage in STAGES[:STAGES.index(stage)]:
        pid = job_id(cell, parent_stage)
        if state.get(f"{pid}/{stem}", {}).get("status") != "done":
            raise RuntimeError(f"Parent incomplete: {pid}/{stem}")
        assert_axes(matrix, pid, stem, prefix_axes(cell, parent_stage), signature)
    scope = matrix / "cache" / f"{jid}_{stem}"
    axes_path = scope / "axes.json"
    if axes_path.exists():
        assert_axes(matrix, jid, stem, prefix_axes(cell, stage), signature)
    write_json(axes_path, dict(prefix=prefix_axes(cell, stage), axes=cell,
                               stage=stage, signature=signature))
    config_path = matrix / "configs" / f"{jid}.json"
    write_json(config_path, cell_config(base, profile, cell))
    input_dir = matrix / "inputs" / stem
    input_dir.mkdir(parents=True, exist_ok=True)
    source = Path(job["source"])
    link = input_dir / source.name
    if not link.exists():
        link.symlink_to(source)
    elif link.resolve() != source.resolve():
        raise RuntimeError(f"Input changed: {link}")
    record = state[job["key"]]
    ledger = matrix / "ledgers" / jid / f"{stem}-{record['attempts']}.json"
    cmd = [python, "-u", str(ROOT / "main.py"), "--audio_dir", str(input_dir),
           "--config", str(config_path), "--env", profile, "--job_id", jid,
           "--cache_dir", str(matrix / "cache"), "--save_path", str(matrix / "out" / jid),
           "--ledger", str(ledger), "--stop_after", stage, "--only_batch", "0",
           "--music_separator", MUSIC_MODELS[cell["A"]], "--bss", "--music", "--vad",
           "--no_review_page", "--cache_parents", *parents(cell, stage)]
    log_path = matrix / record["log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"command": cmd, "axes": cell}) + "\n")
        log.flush()
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
            cwd=ROOT, env=cell_environment(cell), start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    output = log_path.read_text(errors="replace")
    record["tail"] = output.splitlines()[-50:]
    validate_log(output)
    progress = read_json(ledger, {})
    if code or source.name not in progress.get("done", {}) or progress.get("failed"):
        raise RuntimeError(f"Pipeline exit={code}, ledger does not confirm completion: {progress.get('failed', {})}")
    stage_dir = matrix / "out" / jid / stem / STAGE_DIR[stage]
    artifact = "music_map.json" if stage == "music" else "stats.json"
    if not (stage_dir / artifact).is_file():
        raise RuntimeError(f"Missing stage result: {stage_dir / artifact}")
    # main's failed-file cleanup may remove a child cache. Metadata is restored
    # only after successful computation and artifact validation.
    write_json(axes_path, dict(prefix=prefix_axes(cell, stage), axes=cell,
                               stage=stage, signature=signature, completed=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--env", default="a100")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--axis", action="append", default=[], metavar="A=mel3005,bsr368")
    parser.add_argument("--include", nargs="*", help="Exact recording stems to include")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot-g", action="store_true", help="Compare SSLAM and INA on lm8 before a full matrix")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--max-hours", type=float, default=11.0)
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    matrix = args.matrix.resolve()
    selection = dict(item.split("=", 1) for item in args.axis)
    selected = cells({k: v.split(",") for k, v in selection.items()}, args.smoke)
    sources = sorted(p.resolve() for p in args.audio_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"))
    if args.include:
        sources = [p for p in sources if p.stem in args.include]
        if {p.stem for p in sources} != set(args.include):
            parser.error("Some --include recordings are missing")
    if args.smoke:
        sources = sources[:1]
    if not sources or len({p.stem for p in sources}) != len(sources):
        parser.error("Need audio files with unique stems")
    if args.pilot_g:
        sources = [p for p in sources if "lm8" in p.stem.lower()]
        if len(sources) != 1:
            parser.error("G pilot requires exactly one lm8 recording")
        selected = cells({k: (v if k == "G" else v[:1]) for k, v in AXES.items()})
    jobs = jobs_for(selected, sources)
    if args.pilot_g:
        jobs = [job for job in jobs if job["stage"] == "music"]
    if args.dry_run:
        print(json.dumps({"cells": len(selected), "files": len(sources),
            "jobs": len(jobs), "stages": {s: sum(j["stage"] == s for j in jobs) for s in STAGES}}, indent=2))
        return 0
    if not args.pilot_g and any(c["G"] == "ina" for c in selected):
        pilot = read_json(matrix / "ina_pilot.json", {})
        if not pilot.get("completed"):
            parser.error("Run --pilot-g first (or explicitly select --axis G=nog)")
        if not pilot.get("keep_axis"):
            selected = [c for c in selected if c["G"] == "nog"]
            jobs = jobs_for(selected, sources)
    base = read_json(args.config)
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("services", "models", "utils", "algorithms", "schemas")
            for p in (ROOT / folder).rglob("*.py")}
    for p in [ROOT / "main.py", ROOT / "diarizen_worker.py", Path(__file__), ROOT / "tools/matrix_axes.py"]:
        code[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    fixed_env = {k: v for k, v in os.environ.items() if k.startswith(("BSS_", "MUSIC_MAP_", "NOISE_", "SSLAM_"))}
    signature = fingerprint(dict(config=base, code=code, env=fixed_env))
    identity = {p.stem: dict(path=str(p), size=p.stat().st_size,
                             sha256=hashlib.file_digest(p.open("rb"), "sha256").hexdigest()) for p in sources}
    previous = read_json(matrix / "manifest.json")
    if previous and previous["signature"] != signature:
        parser.error("Code/config/environment changed; choose a new matrix directory")
    for stem, source in identity.items():
        if previous and stem in previous["inputs"] and previous["inputs"][stem] != source:
            parser.error(f"Input changed: {stem}; choose a new matrix directory")
    if not args.pilot_g:
        write_json(matrix / "manifest.json", dict(signature=signature, inputs=identity,
            axes=AXES, cells=[dict(id=combo_id(c), axes=c, jobs={s: job_id(c, s) for s in STAGES}) for c in selected]))
    state = read_json(matrix / "state.json", {})
    for job in jobs:
        state.setdefault(job["key"], dict(status="pending", attempts=0, stage=job["stage"],
                                         stem=job["stem"], axes=job["cell"]))
    start, executed = time.monotonic(), 0
    for job in jobs:
        record = state[job["key"]]
        if record["status"] == "done":
            assert_axes(matrix, job["job"], job["stem"], prefix_axes(job["cell"], job["stage"]), signature)
            continue
        if record["status"] == "failed" and record["attempts"] >= 1 + args.retries:
            continue
        if (args.max_jobs is not None and executed >= args.max_jobs) or time.monotonic() - start >= args.max_hours * 3600:
            break
        record.update(status="running", attempts=record["attempts"] + 1,
                      log=f"logs/{job['job']}/{job['stem']}-{record['attempts'] + 1}.log")
        write_json(matrix / "state.json", state)
        began = time.monotonic()
        print(f"[{executed + 1}] {job['stage']} {job['stem']} {job['job']}", flush=True)
        try:
            run_job(matrix, job, base, args.env, args.python, state, signature, args.timeout)
            record.update(status="done", error=None)
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            log = matrix / record["log"]
            if log.exists():
                record["tail"] = log.read_text(errors="replace").splitlines()[-50:]
            print(record["error"], file=sys.stderr, flush=True)
        finally:
            record["seconds"] = round(time.monotonic() - began, 2)
            write_json(matrix / "state.json", state)
        executed += 1
    if args.pilot_g and all(state[j["key"]]["status"] == "done" for j in jobs):
        treatment = next(j for j in jobs if j["cell"]["G"] == "ina")
        diagnostic = read_json(matrix / "cache" / f"{treatment['job']}_{treatment['stem']}" / "ina_diagnostics/result.json", {})
        added = diagnostic.get("added_music_seconds", 0) + diagnostic.get("added_noise_seconds", 0)
        write_json(matrix / "ina_pilot.json", dict(completed=True, keep_axis=added > 0,
            diagnostic=diagnostic, signature=signature, inputs=identity))
    summary = {s: sum(state[j["key"]]["status"] == s for j in jobs) for s in ("done", "failed", "pending", "running")}
    print(json.dumps(summary))
    return 1 if summary["failed"] else (0 if summary["done"] == len(jobs) else 2)


if __name__ == "__main__":
    raise SystemExit(main())
