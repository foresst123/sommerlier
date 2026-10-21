#!/usr/bin/env python3
"""Run the pipeline once per noise-filter variant and record where the outputs are.

Cross product: files x variants. The baseline ("none") is always run, even when the
variants file omits it. Each variant gets its own output dir, job_id and private input
dir (symlinks to the chosen files), so caches, progress ledgers and --resume state of
different variants never collide and nothing is written into the user's audio dir.
The baseline command carries no --denoise, so it is the pipeline's old behaviour.

variants.json: {"baseline": "none", "variants": [{"chain": "none"}, {"chain": "dfn3"},
                {"chain": "dfn3>mf2se48", "label": "DFN3 then MF2SE", "group": "chains",
                 "note": "free text shown as the chip tooltip"}]}

Per-variant overrides (all optional; a variant that has any is never the baseline, even with chain
none, so "music stage OFF" is a second control):
    {"chain": "none", "tag": "music stage OFF", "set": {"steps.music_removal": false},
     "music_separator": "model.ckpt", "extra_args": ["--bss"], "id": "my-id"}
`set` becomes repeated `--set KEY=VALUE` (VALUE as JSON), `music_separator` --music_separator,
`extra_args` raw arguments. Without `id` it is derived: chain id, plus ".<short tag>" when overrides exist.
Ids must be unique and filename-safe. Pick work with --groups, --only-ids a,b, --max-variants N
(counts the baseline) with --shuffle-seed S; --list prints the plan and exits.

--only-available skips (never silently) every variant whose chain names a backend that
models.denoise.available_backends() reports unusable here: status "skipped" plus the reason in
variants_manifest.json and in the console summary. --groups a,b runs only variants whose
"group" is listed (the baseline always runs).

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

Silent fallbacks: when a denoise backend cannot load or run, DenoiseService keeps the unfiltered audio and
only writes the error into <file dir>/01b_denoise/denoise.json. After each variant the runner reads that
file for every file and gives a verdict: ok, fallback (errors, chain dropped), missing (non-none chain but
no denoise.json) or suspect_noop (no errors but output identical to input). fallback/missing make the
variant status "degraded" (subprocess exit 0, but the filter did not run): no .done marker (so --resume
re-runs it) unless --accept-degraded, listed in the DEGRADED summary, and --strict exits 3. A degraded
variant says nothing about what the filter does.
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
DEFAULT_MAIN = (ROOT / "main.py").resolve()
DONE_MARKER = ".done"
MANIFEST_NAME = "variants_manifest.json"
NOOP_CORR = 0.99999  # corr_with_input at or above this = the filter changed nothing


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


ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._+-]*$")
PSEUDO_RE = re.compile(r"^blend\d{1,3}$")


def _backend_tokens(chain):
    """Chain tokens that need a real backend (pseudo-steps like blend50 do not)."""
    try:
        sys.path.insert(0, str(ROOT))
        from models.denoise.pseudo import backend_tokens
        return list(backend_tokens(list(chain)))
    except Exception:
        return [t for t in chain if not PSEUDO_RE.match(t)]
    finally:
        if sys.path and sys.path[0] == str(ROOT):
            sys.path.pop(0)


def _norm_set(raw):
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit(f"variant 'set' must be an object of dotted keys, got {raw!r}")
    for k in raw:
        if not isinstance(k, str) or not k.strip():
            raise SystemExit(f"variant 'set' has a bad key {k!r}")
    return {k.strip(): v for k, v in raw.items()}


def _norm_extra(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        return shlex.split(raw)
    if not isinstance(raw, (list, tuple)):
        raise SystemExit(f"variant 'extra_args' must be a list, got {raw!r}")
    return [str(x) for x in raw]


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()


def override_tag(overrides, music_separator=None, extra_args=()):
    """Short deterministic id suffix for an override set: 'music-removal-off' or a hash8."""
    parts = []
    for key, val in overrides.items():
        leaf = _slug(key.split(".")[-1])
        if isinstance(val, bool):
            parts.append(f"{leaf}-{'on' if val else 'off'}")
        else:
            parts.append(f"{leaf}-{_slug(json.dumps(val))}")
    if music_separator:
        parts.append("sep-" + _slug(os.path.splitext(str(music_separator))[0]))
    if extra_args:
        parts.append("x-" + _slug(" ".join(extra_args)))
    tag = "_".join(parts)
    if not tag or len(tag) > 40 or not ID_RE.match(tag):
        canon = json.dumps([overrides, music_separator, list(extra_args)], sort_keys=True)
        tag = hashlib.sha1(canon.encode()).hexdigest()[:8]
    return tag


def load_variants(path):
    """Return variants [{id, chain, label, spec, group, note, tag, set, music_separator,
    extra_args}], baseline first. Identical entries merge; clashing ids are an error."""
    parse_chain, chain_id = _denoise_helpers()
    data = {}
    if path:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    entries = list(data.get("variants") or [])
    entries.insert(0, {"chain": data.get("baseline", "none")})
    out, by_id, dups = [], {}, []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"chain": entry}
        chain = parse_chain(entry.get("chain"))
        overrides = _norm_set(entry.get("set"))
        sep = entry.get("music_separator") or None
        extra = _norm_extra(entry.get("extra_args"))
        has_over = bool(overrides or sep or extra)
        vid = str(entry.get("id") or "") or (
            f"{chain_id(chain)}.{override_tag(overrides, sep, extra)}" if has_over else chain_id(chain))
        v = {"id": vid, "chain": list(chain),
             "label": entry.get("label") or vid,
             "spec": ">".join(chain),
             "group": entry.get("group") or None,
             "note": entry.get("note") or None,
             "tag": entry.get("tag") or None,
             "set": overrides, "music_separator": sep, "extra_args": extra}
        if not ID_RE.match(vid):
            raise SystemExit(f"variant id {vid!r} is not filename-safe (use letters, digits, . _ + -)")
        if vid in by_id:
            same = all(by_id[vid][k] == v[k] for k in ("chain", "set", "music_separator", "extra_args"))
            if not same:
                dups.append(vid)
            continue
        by_id[vid] = v
        out.append(v)
    if dups:
        raise SystemExit(f"duplicate variant id(s) with different settings: {sorted(set(dups))}; "
                         "give each variant a unique 'id'")
    if out[0]["id"] != "none":
        out.insert(0, {"id": "none", "chain": [], "label": "none", "spec": "", "group": None,
                       "note": None, "tag": None, "set": {}, "music_separator": None,
                       "extra_args": []})
    return out


def pick_variants(variants, only_ids=None, max_variants=None, shuffle_seed=None):
    """Keep the baseline plus the chosen variants, in file order.

    only_ids: explicit ids. max_variants: total kept, baseline included; the first N in file
    order, or a random subset when shuffle_seed is given (same seed, same subset).
    """
    base = [v for v in variants if v["id"] == "none"]
    rest = [v for v in variants if v["id"] != "none"]
    ids = [i.strip() for i in (only_ids or []) if i.strip()]
    if ids:
        unknown = [i for i in ids if i not in {v["id"] for v in rest} | {"none"}]
        if unknown:
            raise SystemExit(f"--only-ids: unknown id(s) {unknown}; known: {[v['id'] for v in variants]}")
        rest = [v for v in rest if v["id"] in ids]
    if max_variants is not None:
        room = max(0, max_variants - len(base))
        if shuffle_seed is not None:
            import random
            keep = {v["id"] for v in random.Random(shuffle_seed).sample(rest, min(room, len(rest)))}
            rest = [v for v in rest if v["id"] in keep]
        else:
            rest = rest[:room]
    return base + rest


def filter_groups(variants, groups):
    """Keep the baseline and variants whose group is in `groups` (a list of names); no groups = keep all."""
    wanted = {g.strip() for g in (groups or []) if g.strip()}
    if not wanted:
        return variants
    known = {v["group"] for v in variants if v.get("group")}
    unknown = wanted - known
    if unknown:
        raise SystemExit(f"unknown group(s) {sorted(unknown)}; the variants file has {sorted(known)}")
    return [v for v in variants if v["id"] == "none" or v.get("group") in wanted]


def check_availability(variants, cfg=None):
    """Return {variant id: reason} for variants whose chain has a backend that is unusable here."""
    try:
        sys.path.insert(0, str(ROOT))
        from models.denoise import available_backends
        avail = available_backends(cfg)
    except Exception as exc:
        raise SystemExit(f"--only-available needs models.denoise to be importable: {exc!r}")
    finally:
        if sys.path and sys.path[0] == str(ROOT):
            sys.path.pop(0)
    skipped = {}
    for v in variants:
        reasons = []
        for name in _backend_tokens(v["chain"]):
            ok, why = avail.get(name, (False, "unknown backend (not in the registry)"))
            if not ok:
                reasons.append(f"{name}: {why}")
        if reasons:
            skipped[v["id"]] = "; ".join(reasons)
    return skipped


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
    if variant.get("music_separator"):
        cmd += ["--music_separator", str(variant["music_separator"])]
    # A comparison run is not a dataset export: keep it out of the shared clean-data folder.
    # Placed before the variant's own settings, so a variant can still override it.
    cmd += ["--set", 'outputs.clean_two_channel_dir=""']
    for key, val in (variant.get("set") or {}).items():
        cmd += ["--set", f"{key}={json.dumps(val, ensure_ascii=False)}"]
    cmd += list(variant.get("extra_args") or [])
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


# --------------------------------------------------------------- denoise report
def _first_line(text, limit=200):
    lines = str(text).strip().splitlines()
    return (lines[0] if lines else "")[:limit]


def _has_real_filter(chain):
    """True when the chain runs a real backend (not blend pseudo-steps, not the identity test backend)."""
    return any(t != "identity" for t in _backend_tokens(chain))


def file_verdict(chain, info, exists=True):
    """Verdict for one file from its parsed denoise.json (None = absent/corrupt).

    Returns {"verdict","errors","corr_with_input","lag_samples"}; see the module docstring.
    """
    errors = [str(e) for e in (info or {}).get("errors") or []] if isinstance(info, dict) else []
    corr = info.get("corr_with_input") if isinstance(info, dict) else None
    lag = info.get("lag_samples") if isinstance(info, dict) else None
    if not chain:
        verdict = "ok"
    elif not isinstance(info, dict):
        verdict = "missing"
        errors = errors or ["01b_denoise/denoise.json is " + ("unreadable" if exists else "absent")]
    elif errors:
        verdict = "fallback"
    elif isinstance(corr, (int, float)) and corr >= NOOP_CORR and _has_real_filter(chain):
        verdict = "suspect_noop"
    else:
        verdict = "ok"
    return {"verdict": verdict, "errors": errors, "corr_with_input": corr, "lag_samples": lag}


def denoise_report(chain, file_dirs, stems):
    """{file stem: file_verdict} for a finished variant; a stem with no output dir counts as missing."""
    report = {}
    for stem in sorted(stems):
        d = file_dirs.get(stem)
        if d is None:
            if chain:
                report[stem] = file_verdict(chain, None, exists=False)
                report[stem]["errors"] = ["no output directory for this file"]
            continue
        path = Path(d) / "01b_denoise" / "denoise.json"
        exists = path.is_file()
        try:
            info = json.loads(path.read_text(encoding="utf-8")) if exists else None
        except (OSError, ValueError):
            info = None
        report[stem] = file_verdict(chain, info, exists)
    return report


def report_counts(report, total):
    bad = sum(1 for r in report.values() if r["verdict"] in ("fallback", "missing"))
    noop = sum(1 for r in report.values() if r["verdict"] == "suspect_noop")
    return {"degraded_files": bad, "suspect_files": noop, "total_files": total}


def write_manifest(out_root, variants, stems, status, reference, reasons=None, reports=None):
    entries = []
    for v in variants:
        vdir = Path(out_root) / v["id"]
        rep = (reports or {}).get(v["id"]) or {}
        entries.append({
            "id": v["id"], "chain": v["chain"], "label": v["label"],
            "out_dir": str(vdir.resolve()), "job_id": f"exp-{v['id']}",
            "status": status.get(v["id"], "missing"),
            "group": v.get("group"), "note": v.get("note"),
            "tag": v.get("tag"), "set": v.get("set") or {},
            "music_separator": v.get("music_separator"),
            "extra_args": v.get("extra_args") or [],
            "reason": (reasons or {}).get(v["id"]),
            "denoise_report": rep.get("denoise_report", {}),
            "degraded_files": rep.get("degraded_files", 0),
            "suspect_files": rep.get("suspect_files", 0),
            "total_files": rep.get("total_files", 0),
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
    def __init__(self, args, variants, files, skipped=None):
        self.args, self.variants, self.files = args, variants, files
        self.skipped = dict(skipped or {})
        self.stems = [f.stem for f in files]
        self.out_root = Path(args.out_root)
        self.status = {vid: "skipped" for vid in self.skipped}
        self.reports = {}
        self.procs = set()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        gpus = [g for g in (args.gpus or "").split(",") if g.strip()]
        self.gpus = gpus
        self.reference = build_reference(self.stems, args.reference_dir, args.reference)

    def _manifest(self):
        with self.lock:
            write_manifest(self.out_root, self.variants, set(self.stems),
                           self.status, self.reference, self.skipped, self.reports)

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
        if self.stop.is_set() or variant["id"] in self.skipped:
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
                    self.status[variant["id"]] = self._assess(variant)
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
        grp = f" group={variant['group']}" if variant.get("group") else ""
        print(f"[{variant['id']}] start{grp} (gpu={env.get('CUDA_VISIBLE_DEVICES', '-')}) "
              f"log={log_path}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            try:
                # stdin=DEVNULL: main.py asks for folders on a terminal (utils/env_bootstrap.ask_paths)
                # and would otherwise wait for Enter on ours while the log shows only the prompt.
                proc = subprocess.Popen(cmd, cwd=a.cwd, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
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
        state = self._assess(variant) if ok else "failed"
        if state == "ok" or (state == "degraded" and a.accept_degraded):
            marker.write_text(json.dumps({"hash": digest, "exit": 0, "status": state}))
        self.status[variant["id"]] = state
        print(f"[{variant['id']}] {state.upper() if state != 'ok' else 'ok'} (exit {code}, "
              f"{time.time() - t0:.0f}s)", flush=True)
        self._manifest()

    def _assess(self, variant):
        """Read every file's denoise.json; record the report; "degraded" or "ok"."""
        vdir = self.out_root / variant["id"]
        report = denoise_report(variant["chain"], discover_dirs(vdir, set(self.stems)), self.stems)
        counts = report_counts(report, len(self.stems))
        self.reports[variant["id"]] = {"denoise_report": report, **counts}
        if counts["suspect_files"]:
            print(f"[{variant['id']}] WARNING: output identical to input in {counts['suspect_files']}/"
                  f"{counts['total_files']} file(s) (suspect_noop)", flush=True)
        if counts["degraded_files"]:
            print(f"[{variant['id']}] DEGRADED: {degraded_reason(report, counts)}", flush=True)
            return "degraded"
        return "ok"

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


def degraded_reason(report, counts):
    first = next((r for r in report.values() if r["verdict"] in ("fallback", "missing")), None)
    err = _first_line(first["errors"][0]) if first and first["errors"] else "no error recorded"
    return (f"{counts['degraded_files']}/{counts['total_files']} file(s) fell back to unfiltered audio "
            f"or have no denoise.json; first error: {err}")


def print_degraded_summary(reports, status):
    rows = [(vid, r) for vid, r in reports.items() if status.get(vid) == "degraded"]
    if not rows:
        return
    print("\n" + "=" * 70)
    print(f"DEGRADED: {len(rows)} variant(s) ran but the filter did not apply; "
          "their outputs are the unfiltered audio, NOT 'the filter has no effect'")
    for vid, r in rows:
        print(f"  {vid}: {degraded_reason(r['denoise_report'], r)}")
    print("Fix the environment and re-run with --resume (degraded variants have no .done marker), "
          "or pass --accept-degraded to keep them as they are.")
    print("=" * 70)


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
    p.add_argument("--only-available", action="store_true",
                   help="skip (and record as skipped, with the reason) variants whose chain has an unusable backend")
    p.add_argument("--groups", default="", help="comma list of variant groups to run (baseline always runs)")
    p.add_argument("--only-ids", default="", help="comma list of variant ids to run (baseline always runs)")
    p.add_argument("--max-variants", type=int, default=None,
                   help="keep at most N variants in total, baseline included")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="with --max-variants: random (reproducible) subset instead of the first N")
    p.add_argument("--accept-degraded", action="store_true",
                   help="write the .done marker for a degraded variant too (so --resume keeps it)")
    p.add_argument("--strict", action="store_true",
                   help="exit with code 3 when any variant is degraded or failed")
    p.add_argument("--list", action="store_true", help="print the planned variants and work, then exit")
    p.add_argument("--minutes-per-file-run", type=float, default=20.0,
                   help="ASSUMPTION for the --list time estimate: minutes per (variant, file) run")
    p.add_argument("--main", default=str(ROOT / "main.py"))
    p.add_argument("--cwd", default=None, help="working dir for main (default: its folder)")
    p.add_argument("--reference-dir")
    p.add_argument("--reference", nargs="*", metavar="STEM=PATH")
    p.add_argument("--cache-dir")
    p.add_argument("--python", default=sys.executable,
                   help="interpreter that runs main.py (default: the one running this tool; "
                        "it must be the pipeline's environment, not another project's)")
    p.add_argument("--no-preflight", action="store_true",
                   help="skip the import check of --python before the first variant starts")
    p.add_argument("--preflight-modules", default="torch,torchaudio",
                   help="modules the interpreter must import cleanly (comma list)")
    p.add_argument("--timeout", type=float, default=0, help="seconds per variant, 0 = none")
    args = p.parse_args(argv)
    if not args.audio_dir and not args.files:
        p.error("one of --audio-dir or --files is required")
    args.main = Path(args.main).resolve()
    args.cwd = args.cwd or str(args.main.parent)
    args.out_root = str(Path(args.out_root).resolve())
    return args


def preflight(python, modules, timeout=240):
    """Import the heavy modules in the interpreter that will run main.py. -> (ok, message).

    A mismatched environment (e.g. torchaudio built for another CUDA than torch) otherwise only
    shows up minutes into the first variant, as a traceback in its log.
    """
    mods = [m.strip() for m in modules.split(",") if m.strip()]
    code = ("import sys, importlib\nout = []\n"
            f"for m in {mods!r}:\n"
            "    mod = importlib.import_module(m)\n"
            "    out.append(m + '=' + str(getattr(mod, '__version__', '?')))\n"
            "print(sys.executable, ' '.join(out))")
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not run {python}: {exc}"
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-6:])
        return False, f"{python} cannot import {', '.join(mods)}:\n{tail}"
    return True, r.stdout.strip()


def print_plan(variants, files, skipped, args):
    rows = [("id", "group", "chain", "tag", "overrides", "available")]
    for v in variants:
        over = [f"{k}={json.dumps(x)}" for k, x in (v.get("set") or {}).items()]
        if v.get("music_separator"):
            over.append(f"music_separator={v['music_separator']}")
        over += v.get("extra_args") or []
        avail = ("SKIPPED: " + skipped[v["id"]]) if v["id"] in skipped else (
            "yes" if args.only_available else "-")
        rows.append((v["id"], v.get("group") or "-", v["spec"] or "none", v.get("tag") or "-",
                     " ".join(over) or "-", avail))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
    runnable = len(variants) - len(skipped)
    runs = runnable * len(files)
    print(f"\nvariants: {len(variants)} ({len(skipped)} skipped, {runnable} to run)")
    print(f"files: {len(files)}")
    print(f"variant-file runs: {runs}")
    mins = runs * args.minutes_per_file_run
    print(f"sequential estimate: {mins:.0f} min ({mins / 60:.1f} h) "
          f"ASSUMING {args.minutes_per_file_run:g} min per variant-file run; --parallel/--gpus divide it")


def main(argv=None):
    args = parse_args(argv)
    variants = filter_groups(load_variants(args.variants), args.groups.split(","))
    variants = pick_variants(variants, args.only_ids.split(","), args.max_variants, args.shuffle_seed)
    files = collect_files(args.audio_dir, args.files)
    skipped = check_availability(variants) if args.only_available else {}
    if args.list:
        print_plan(variants, files, skipped, args)
        return 0
    for vid, why in skipped.items():
        print(f"[{vid}] SKIPPED (not available here): {why}", flush=True)
    if args.dry_run:
        for i, v in enumerate(variants):
            if v["id"] in skipped:
                continue
            gpu = args.gpus.split(",")[i % len(args.gpus.split(","))] if args.gpus else "-"
            print(f"[{v['id']}] CUDA_VISIBLE_DEVICES={gpu} " +
                  shlex.join(build_command(v, files, args)))
        return 0
    if not args.no_preflight and args.main == DEFAULT_MAIN:
        ok, msg = preflight(args.python, args.preflight_modules)
        if not ok:
            print(f"[preflight] FAILED: {msg}\n"
                  f"Pass --python <path to the pipeline's environment>/bin/python (the one where "
                  f"`python -c 'import torch, torchaudio'` works), or --no-preflight to skip this check.",
                  file=sys.stderr, flush=True)
            return 2
        print(f"[preflight] ok: {msg}", flush=True)
    runner = Runner(args, variants, files, skipped)
    status = runner.run()
    print(json.dumps(status))
    print_degraded_summary(runner.reports, status)
    if skipped:
        print(f"skipped {len(skipped)} variant(s): {', '.join(skipped)} (reasons in the manifest)")
    print(f"manifest: {Path(args.out_root) / MANIFEST_NAME}")
    if args.strict and any(s in ("degraded", "failed") for s in status.values()):
        return 3
    return 0 if all(s in ("ok", "skipped", "degraded") for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
