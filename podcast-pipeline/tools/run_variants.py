#!/usr/bin/env python3
"""Run the pipeline once per noise-filter variant and record where the outputs are.

Cross product: files x variants. The baseline ("none") is always run, even when the
variants file omits it. Each variant gets its own output dir, job_id and private input
dir (symlinks to the chosen files), so caches, progress ledgers and --resume state of
different variants never collide and nothing is written into the user's audio dir.
The baseline command carries no --denoise, so it is the pipeline's old behaviour.

variants.json: {"baseline": "none", "variants": [{"chain": "none"}, {"chain": "dfn3"},
                {"chain": "dfn3>mf2se48", "label": "DFN3 then MF2SE"}]}

Output: OUT/<variant id>/<file stem>/{01_music,01b_denoise,02_diarization,...},
OUT/logs/<variant id>.log, OUT/<variant id>/.done, OUT/variants_manifest.json.

Example on the DGX (two GPUs, variants in parallel, safe to re-run with --resume):
    cd podcast-pipeline
    python tools/run_variants.py --audio-dir /data/vi_podcasts_20 \\
        --variants tools/variants.example.json --out-root /data/denoise_cmp \\
        --env a100 --stop-after asr --gpus 0,1 --parallel 2 --resume \\
        --reference-dir /data/vi_podcasts_20/ref
    python tools/run_variants.py ... --dry-run     # print commands only

Note: main.py's output flag is --save_path (there is no --output_dir); the runner
passes --save_path. A failed variant is logged and recorded as "failed"; the others go on.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".opus", ".ogg", ".aac")
OUT_FLAG = "--save_path"
DONE_MARKER = ".done"
MANIFEST_NAME = "variants_manifest.json"


# ---------------------------------------------------------------- variant ids
def _fallback_parse_chain(spec):
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        tokens = [str(t).strip().lower() for t in spec]
    else:
        text = str(spec).strip()
        if text.lower() in ("", "none"):
            return []
        tokens = [t.strip().lower() for t in text.split(">")]
    if any(not t for t in tokens):
        raise ValueError(f"empty token in chain spec {spec!r}")
    return [] if tokens == ["none"] else tokens


def _fallback_chain_id(spec):
    chain = _fallback_parse_chain(spec)
    if not chain:
        return "none"
    return re.sub(r"[^A-Za-z0-9._+-]", "_", "+".join(chain))


def _denoise_helpers():
    try:
        sys.path.insert(0, str(ROOT))
        from models.denoise import parse_chain, chain_id
        return parse_chain, chain_id
    except Exception:
        return _fallback_parse_chain, _fallback_chain_id
    finally:
        if sys.path and sys.path[0] == str(ROOT):
            sys.path.pop(0)


def load_variants(path):
    """Return variants [{id, chain, label, spec}], baseline first, deduplicated."""
    parse_chain, chain_id = _denoise_helpers()
    data = {}
    if path:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    entries = list(data.get("variants") or [])
    entries.insert(0, {"chain": data.get("baseline", "none")})
    out, seen = [], set()
    for entry in entries:
        if isinstance(entry, str):
            entry = {"chain": entry}
        chain = parse_chain(entry.get("chain"))
        vid = chain_id(chain)
        if vid in seen:
            continue
        seen.add(vid)
        out.append({"id": vid, "chain": list(chain),
                    "label": entry.get("label") or vid,
                    "spec": ">".join(chain)})
    if out[0]["id"] != "none":
        out.insert(0, {"id": "none", "chain": [], "label": "none", "spec": ""})
    return out


# ---------------------------------------------------------------------- files
def collect_files(audio_dir, files):
    paths = []
    if audio_dir:
        for p in sorted(Path(audio_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                paths.append(p.resolve())
    for f in files or []:
        paths.append(Path(f).resolve())
    stems = {}
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"audio file not found: {p}")
        stems.setdefault(p.stem, []).append(p)
    clash = {s: v for s, v in stems.items() if len(v) > 1 and len(set(v)) > 1}
    if clash:
        raise SystemExit(f"files share a stem and would overwrite each other: {sorted(clash)}")
    if not stems:
        raise SystemExit("no audio files selected")
    return sorted({p for p in paths}, key=lambda p: p.name)


def link_inputs(input_dir, files):
    input_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f.name for f in files}
    for old in input_dir.iterdir():  # drop links to files no longer selected
        if old.name not in wanted and (old.is_symlink() or old.is_file()):
            old.unlink()
    for f in files:
        dst = input_dir / f.name
        if dst.is_symlink() and Path(os.readlink(dst)) == f:
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(f, dst)
        except OSError:
            shutil.copy2(f, dst)


# ------------------------------------------------------------------- commands
def build_command(variant, files, args, input_dir=None):
    vdir = Path(args.out_root) / variant["id"]
    cmd = [args.python, str(args.main),
           "--audio_dir", str(input_dir or vdir / "input"),
           OUT_FLAG, str(vdir),
           "--job_id", f"exp-{variant['id']}",
           "--env", args.env,
           "--stop_after", args.stop_after]
    if variant["chain"]:
        cmd += ["--denoise", variant["spec"]]
    if args.cache_dir:
        cmd += ["--cache_dir", str(args.cache_dir)]
    return cmd


def run_hash(cmd, files):
    h = hashlib.sha256(json.dumps(cmd).encode())
    for f in files:
        st = f.stat()
        h.update(f"{f}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


# ------------------------------------------------------------------ discovery
def discover_dirs(vdir, stems):
    """Map file stem -> per-file output dir (holds 02_diarization and/or 05_asr)."""
    found = {}
    if not vdir.is_dir():
        return found
    for root, dirs, _ in os.walk(vdir):
        if Path(root) == vdir:
            dirs[:] = [d for d in dirs if d != "input"]
        has = set(dirs)
        if Path(root).name in stems and ({"02_diarization", "05_asr"} & has):
            found.setdefault(Path(root).name, str(Path(root).resolve()))
            dirs[:] = []
    return found


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_manifest(out_root, variants, stems, status, reference):
    entries = []
    for v in variants:
        vdir = Path(out_root) / v["id"]
        entries.append({
            "id": v["id"], "chain": v["chain"], "label": v["label"],
            "out_dir": str(vdir.resolve()), "job_id": f"exp-{v['id']}",
            "status": status.get(v["id"], "missing"),
            "files": discover_dirs(vdir, stems)})
    write_json_atomic(Path(out_root) / MANIFEST_NAME, {
        "version": 1, "baseline": "none",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "variants": entries, "reference": reference})


def build_reference(stems, reference_dir, pairs):
    ref = {s: None for s in stems}
    if reference_dir:
        for s in stems:
            for ext in (".json",):
                cand = Path(reference_dir) / f"{s}{ext}"
                if cand.is_file():
                    ref[s] = str(cand.resolve())
    for pair in pairs or []:
        stem, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"--reference expects stem=path, got {pair!r}")
        ref[stem] = str(Path(path).resolve())
    return ref


# --------------------------------------------------------------------- runner
class Runner:
    def __init__(self, args, variants, files):
        self.args, self.variants, self.files = args, variants, files
        self.stems = [f.stem for f in files]
        self.out_root = Path(args.out_root)
        self.status = {}
        self.procs = set()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        gpus = [g for g in (args.gpus or "").split(",") if g.strip()]
        self.gpus = gpus
        self.reference = build_reference(self.stems, args.reference_dir, args.reference)

    def _manifest(self):
        with self.lock:
            write_manifest(self.out_root, self.variants, set(self.stems),
                           self.status, self.reference)

    def terminate_all(self, *_):
        self.stop.set()
        with self.lock:
            procs = list(self.procs)
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    def run_one(self, index, variant):
        a = self.args
        vdir = self.out_root / variant["id"]
        if self.stop.is_set():
            return
        input_dir = vdir / "input"
        link_inputs(input_dir, self.files)
        cmd = build_command(variant, self.files, a)
        digest = run_hash(cmd, self.files)
        marker = vdir / DONE_MARKER
        if a.resume and marker.is_file():
            try:
                if json.loads(marker.read_text()).get("hash") == digest:
                    print(f"[{variant['id']}] up to date, skipped", flush=True)
                    self.status[variant["id"]] = "ok"
                    self._manifest()
                    return
            except (ValueError, OSError):
                pass
        marker.unlink(missing_ok=True)
        env = os.environ.copy()
        if self.gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.gpus[index % len(self.gpus)]
        log_path = self.out_root / "logs" / f"{variant['id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t0, code = time.time(), None
        print(f"[{variant['id']}] start (gpu={env.get('CUDA_VISIBLE_DEVICES', '-')}) "
              f"log={log_path}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            try:
                proc = subprocess.Popen(cmd, cwd=a.cwd, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as exc:
                log.write(f"could not start: {exc}\n")
                self.status[variant["id"]] = "failed"
                self._manifest()
                return
            with self.lock:
                self.procs.add(proc)
            try:
                code = proc.wait(timeout=a.timeout or None)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                code = "timeout"
            finally:
                with self.lock:
                    self.procs.discard(proc)
            log.write(f"\n[run_variants] exit status: {code} after {time.time() - t0:.0f}s\n")
        ok = code == 0 and not self.stop.is_set()
        if ok:
            marker.write_text(json.dumps({"hash": digest, "exit": 0}))
        self.status[variant["id"]] = "ok" if ok else "failed"
        print(f"[{variant['id']}] {'ok' if ok else 'FAILED'} (exit {code}, "
              f"{time.time() - t0:.0f}s)", flush=True)
        self._manifest()

    def run(self):
        self.out_root.mkdir(parents=True, exist_ok=True)
        self._manifest()
        old = {s: signal.signal(s, self.terminate_all)
               for s in (signal.SIGINT, signal.SIGTERM)} \
            if threading.current_thread() is threading.main_thread() else {}
        try:
            with ThreadPoolExecutor(max_workers=max(1, self.args.parallel)) as pool:
                futs = [pool.submit(self.safe_run, i, v) for i, v in enumerate(self.variants)]
                for f in futs:
                    f.result()
        finally:
            for s, h in old.items():
                signal.signal(s, h)
        self._manifest()
        return self.status

    def safe_run(self, index, variant):
        try:
            self.run_one(index, variant)
        except Exception as exc:  # one variant must never stop the others
            print(f"[{variant['id']}] runner error: {exc!r}", flush=True)
            self.status[variant["id"]] = "failed"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the pipeline for each noise-filter variant.")
    p.add_argument("--audio-dir")
    p.add_argument("--files", nargs="*")
    p.add_argument("--variants", help="variants JSON (default: baseline only)")
    p.add_argument("--out-root", required=True)
    p.add_argument("--env", default="a100")
    p.add_argument("--stop-after", default="asr")
    p.add_argument("--gpus", default="", help="comma list, round-robin per variant")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--main", default=str(ROOT / "main.py"))
    p.add_argument("--cwd", default=None, help="working dir for main (default: its folder)")
    p.add_argument("--reference-dir")
    p.add_argument("--reference", nargs="*", metavar="STEM=PATH")
    p.add_argument("--cache-dir")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--timeout", type=float, default=0, help="seconds per variant, 0 = none")
    args = p.parse_args(argv)
    if not args.audio_dir and not args.files:
        p.error("one of --audio-dir or --files is required")
    args.main = Path(args.main).resolve()
    args.cwd = args.cwd or str(args.main.parent)
    args.out_root = str(Path(args.out_root).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    variants = load_variants(args.variants)
    files = collect_files(args.audio_dir, args.files)
    if args.dry_run:
        for i, v in enumerate(variants):
            gpu = args.gpus.split(",")[i % len(args.gpus.split(","))] if args.gpus else "-"
            print(f"[{v['id']}] CUDA_VISIBLE_DEVICES={gpu} " +
                  shlex.join(build_command(v, files, args)))
        return 0
    status = Runner(args, variants, files).run()
    print(json.dumps(status))
    print(f"manifest: {Path(args.out_root) / MANIFEST_NAME}")
    return 0 if all(s == "ok" for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
