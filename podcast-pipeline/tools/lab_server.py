#!/usr/bin/env python3
"""Backend for the Denoise Lab: serves the page, browses audio on this server and runs real jobs.

    python tools/lab_server.py --audio-root /home/lamkd2/data --out-root /home/lamkd2/data/denoise_lab --jobid 17405
    # on your laptop:  ssh -L 8000:localhost:8000 <dgx>   then open http://localhost:8000

The page's "Chạy" button posts the recipe to /api/experiment/run. The server turns the selected trial into a
variants.json, then starts tools/run_variants.py inside the GPU allocation:

    srun --jobid=<jobid> --overlap bash -lc '<conda activate som>; cd <pipeline>; python tools/run_variants.py ...'

Every run gets its own directory OUT_ROOT/lab_runs/<time>_<run id>/ (variants.json, run.log, status.json, out/), so
nothing is ever overwritten. Standard library only.

It binds to 127.0.0.1 by default and starts jobs on your behalf: there is no login, so pass --host 0.0.0.0 only on a
network you trust. File access is limited to --audio-root (and --out-root, read-only for playback).
"""
import argparse
import json
import mimetypes
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

TOOLS = Path(__file__).resolve().parent
PIPELINE = TOOLS.parent
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}
ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._+-]{0,80}$")
TOKEN_RE = re.compile(r"^[a-z0-9_]+$")
PSEUDO_RE = re.compile(r"^(blend|dry)(\d{1,3})$")
PARAM_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
WORD_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
MAX_BODY = 1 << 20
MAX_LIST = 2000
LOG_TAIL = 6000
CONDA_SH = ('"$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" '
            '"$HOME/miniforge3/etc/profile.d/conda.sh" "$HOME/mambaforge/etc/profile.d/conda.sh" '
            '/opt/conda/etc/profile.d/conda.sh')


class BadRequest(Exception):
    pass


class Config:
    def __init__(self, audio_roots, out_root, jobid, launcher, conda_env, conda_sh, python, dry_run):
        self.audio_roots = [Path(p).expanduser().resolve() for p in audio_roots]
        self.out_root = Path(out_root).expanduser().resolve()
        self.jobid, self.launcher = str(jobid or ""), launcher
        self.conda_env, self.conda_sh, self.python, self.dry_run = conda_env, conda_sh, python, dry_run
        self.runs = {}  # run id -> run dir

    @property
    def read_roots(self):
        return self.audio_roots + [self.out_root]


def inside(path, roots):
    return any(path == r or r in path.parents for r in roots)


def safe_path(raw, roots):
    """Resolve raw (symlinks included) and require it to sit under one of roots."""
    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise BadRequest("đường dẫn không hợp lệ")
    p = Path(raw).expanduser().resolve()
    if not inside(p, roots):
        raise BadRequest(f"ngoài thư mục được phép: {raw}")
    return p


def list_dir(cfg, raw):
    if not raw:
        return {"path": "", "parent": None, "dirs": [str(r) for r in cfg.audio_roots], "files": [], "roots": True}
    p = safe_path(raw, cfg.audio_roots)
    if not p.is_dir():
        raise BadRequest(f"không phải thư mục: {raw}")
    dirs, files = [], []
    for e in sorted(p.iterdir(), key=lambda x: x.name.lower()):
        if e.name.startswith("."):
            continue
        try:
            if e.is_dir():
                dirs.append(e.name)
            elif e.suffix.lower() in AUDIO_EXT:
                files.append({"name": e.name, "size": e.stat().st_size})
        except OSError:
            continue
        if len(dirs) + len(files) >= MAX_LIST:
            break
    parent = str(p.parent) if p not in cfg.audio_roots and inside(p.parent, cfg.audio_roots) else ""
    return {"path": str(p), "parent": parent, "dirs": dirs, "files": files, "roots": False}


def number(value):
    """'15' -> 15, '0.5' -> 0.5, anything else stays a string (the UI sends select values as text)."""
    if isinstance(value, bool) or value is None:
        raise BadRequest("giá trị tham số không hợp lệ")
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if not WORD_RE.match(s):
        raise BadRequest(f"giá trị tham số không hợp lệ: {s[:30]!r}")
    return s


def clean_chain(chain):
    if not isinstance(chain, list) or not chain or len(chain) > 12:
        raise BadRequest("chuỗi model rỗng hoặc quá dài")
    out = []
    for t in chain:
        t = str(t).strip().lower()
        m = PSEUDO_RE.match(t)
        if m and int(m.group(2)) > 100:
            raise BadRequest(f"{t}: phần trăm phải trong 0..100")
        if not (m or TOKEN_RE.match(t)):
            raise BadRequest(f"bước không hợp lệ: {t!r}")
        out.append(t)
    return out


def step_overrides(step_params):
    """stepParams -> {'denoise.<model>.<param>': value}. One parameter set per model: the pipeline keeps one sub-dict per backend."""
    out = {}
    for step in step_params or []:
        if not isinstance(step, dict):
            continue
        model = str(step.get("model") or "").strip().lower()
        if not model or PSEUDO_RE.match(model) or model in ("blend", "dry") or not TOKEN_RE.match(model):
            continue
        for k, v in step.items():
            if k == "model":
                continue
            if not PARAM_RE.match(str(k)):
                raise BadRequest(f"tên tham số không hợp lệ: {k!r}")
            key, val = f"denoise.{model}.{k}", number(v)
            if key in out and out[key] != val:
                raise BadRequest(f"{model} xuất hiện nhiều lần với {k} khác nhau; pipeline chỉ nhận một bộ tham số cho mỗi model")
            out[key] = val
    return out


def sanitize(text, fallback="run"):
    s = re.sub(r"[^A-Za-z0-9._+-]+", "-", str(text or "")).strip("-.")[:60]
    return s or fallback


def build_run(cfg, payload):
    """Validate the UI payload. -> (variants.json dict, run_variants argv tail, summary). Raises BadRequest."""
    exp = payload.get("experiment")
    if not isinstance(exp, dict):
        raise BadRequest("thiếu công thức (experiment)")
    variant = next((v for v in exp.get("variants") or [] if isinstance(v, dict) and v.get("id") == payload.get("trialId")), None)
    if variant is None:
        raise BadRequest("không tìm thấy lượt chạy (trialId)")
    s = exp.get("settings") or {}
    chain = clean_chain(variant.get("chain"))
    overrides = step_overrides(variant.get("stepParams"))
    try:
        gate = float(variant.get("threshold", s.get("dnsmos_threshold", 0.3)))
    except (TypeError, ValueError):
        raise BadRequest("ngưỡng DNSMOS không hợp lệ")
    if not 0 <= gate <= 1:
        raise BadRequest("ngưỡng DNSMOS phải trong 0..1")
    vid = sanitize(variant.get("id"), "variant")
    entry = {"id": vid, "chain": chain, "label": str(variant.get("label") or vid)[:120], "group": "lab",
             "set": {"audio_quality.min_normalized_quality": round(gate, 3), **overrides}}
    tail = []
    files = s.get("input_files")
    if files:
        if not isinstance(files, list) or len(files) > 200:
            raise BadRequest("danh sách file đầu vào không hợp lệ")
        paths = [safe_path(f, cfg.audio_roots) for f in files]
        bad = [str(p) for p in paths if not p.is_file() or p.suffix.lower() not in AUDIO_EXT]
        if bad:
            raise BadRequest(f"không phải file audio: {bad[0]}")
        tail += ["--files", *map(str, paths)]
        source = f"{len(paths)} file"
    else:
        d = safe_path(s.get("audio_dir"), cfg.audio_roots)
        if not d.is_dir():
            raise BadRequest(f"thư mục audio không tồn tại: {d}")
        tail += ["--audio-dir", str(d)]
        source = str(d)
    for key, pat in (("env", WORD_RE), ("stop_after", WORD_RE)):
        if not pat.match(str(s.get(key) or "")):
            raise BadRequest(f"thiếu hoặc sai {key}")
    chunk = s.get("chunk_s")  # run_variants has no global --set: shared defaults go in the variant's own `set`
    if chunk not in (None, ""):
        entry["set"] = {"denoise.chunk_s": number(chunk), **entry["set"]}
    argv = ["--env", str(s["env"]), "--stop-after", str(s["stop_after"]), "--parallel", "1"]
    if s.get("only_available", True):
        argv.append("--only-available")
    if s.get("resume", True):
        argv.append("--resume")
    return {"baseline": "none", "variants": [entry]}, argv + tail, {"chain": ">".join(chain), "source": source, "id": vid}


def conda_prelude(cfg):
    first = f'"{cfg.conda_sh}" ' if cfg.conda_sh else ""
    return (f'for f in {first}{CONDA_SH}; do [ -f "$f" ] && . "$f" && break; done\n'
            f'command -v conda >/dev/null 2>&1 || {{ echo "conda không có trong PATH trên node GPU" >&2; exit 96; }}\n'
            f'conda activate {shlex.quote(cfg.conda_env)} || exit 97\n')


def launch_argv(cfg, argv_tail, variants_path, out_dir):
    cmd = [cfg.python, str(TOOLS / "run_variants.py"), "--variants", str(variants_path), "--out-root", str(out_dir), *argv_tail]
    script = conda_prelude(cfg) + f"cd {shlex.quote(str(PIPELINE))} || exit 98\nexec {shlex.join(cmd)}\n"
    return (shlex.split(cfg.launcher.format(jobid=cfg.jobid)) if cfg.launcher else []) + ["bash", "-lc", script], cmd


def write_status(run_dir, **fields):
    p = run_dir / "status.json"
    try:
        cur = json.loads(p.read_text())
    except (OSError, ValueError):
        cur = {}
    cur.update(fields)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1))
    os.replace(tmp, p)


def reap(proc, run_dir):
    rc = proc.wait()
    write_status(run_dir, state="finished" if rc == 0 else "failed", returncode=rc, finishedAt=datetime.now().isoformat(timespec="seconds"))


def start_run(cfg, payload):
    variants, tail, info = build_run(cfg, payload)
    rid = sanitize(payload.get("runId"), "run")
    run_dir = cfg.out_root / "lab_runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{rid}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "variants.json").write_text(json.dumps(variants, ensure_ascii=False, indent=1))
    (run_dir / "payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    full, plain = launch_argv(cfg, tail, run_dir / "variants.json", run_dir / "out")
    base = {"runId": rid, "runDir": str(run_dir), "log": str(run_dir / "run.log"), "out": str(run_dir / "out"),
            "command": shlex.join(plain), "launcher": full[:-1] if not cfg.dry_run else full[:-1], **info}
    cfg.runs[rid] = run_dir
    if cfg.dry_run:
        write_status(run_dir, state="dry-run", **base)
        return {**base, "status": "dry-run", "note": "--dry-run: chưa chạy, chỉ ghi file", "script": full[-1]}
    log = open(run_dir / "run.log", "ab")
    try:
        proc = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, cwd=str(PIPELINE))
    except OSError as e:
        write_status(run_dir, state="failed", error=str(e), **base)
        raise BadRequest(f"không khởi động được lệnh ({full[0]}): {e}")
    finally:
        log.close()
    write_status(run_dir, state="running", pid=proc.pid, startedAt=datetime.now().isoformat(timespec="seconds"), **base)
    threading.Thread(target=reap, args=(proc, run_dir), daemon=True).start()
    print(f"[lab] run {rid} pid={proc.pid} dir={run_dir}\n[lab]   {base['command']}", flush=True)
    return {**base, "status": "running", "pid": proc.pid}


def find_run_dir(cfg, rid):
    rid = sanitize(rid, "")
    if rid in cfg.runs:
        return cfg.runs[rid]
    found = sorted((cfg.out_root / "lab_runs").glob(f"*_{rid}")) if rid else []
    return found[-1] if found else None


def run_status(cfg, rid):
    run_dir = find_run_dir(cfg, rid)
    if run_dir is None:
        raise BadRequest("không tìm thấy lượt chạy")
    try:
        st = json.loads((run_dir / "status.json").read_text())
    except (OSError, ValueError):
        st = {"state": "unknown"}
    pid = st.get("pid")
    if st.get("state") == "running" and pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            st["state"] = "ended"  # the lab server restarted while the job ran; exit code unknown
        except PermissionError:
            pass
    try:
        with open(run_dir / "run.log", "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - LOG_TAIL))
            st["logTail"] = f.read().decode("utf-8", "replace")
    except OSError:
        st["logTail"] = ""
    st["outputs"] = sorted(str(p) for p in (run_dir / "out").glob("*/*/01b_denoise/after_denoise.wav"))[:200]
    return st


def make_handler(cfg, page):
    class H(BaseHTTPRequestHandler):
        server_version = "DenoiseLab/1"

        def log_message(self, fmt, *a):
            if "/api/experiment/status" not in (a[0] if a else ""):
                super().log_message(fmt, *a)

        def send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def fail(self, e, code=400):
            self.send_json({"error": str(e)}, code)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    body = page.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif u.path == "/api/config":
                    self.send_json({"audioRoots": [str(r) for r in cfg.audio_roots], "outRoot": str(cfg.out_root),
                                    "jobid": cfg.jobid, "dryRun": cfg.dry_run, "condaEnv": cfg.conda_env})
                elif u.path == "/api/files":
                    self.send_json(list_dir(cfg, q.get("path", "")))
                elif u.path == "/api/audio":
                    self.send_audio(safe_path(q.get("path"), cfg.read_roots))
                elif u.path == "/api/experiment/status":
                    self.send_json(run_status(cfg, q.get("runId")))
                else:
                    self.fail("not found", 404)
            except BadRequest as e:
                self.fail(e)
            except OSError as e:
                self.fail(e, 500)

        def send_audio(self, p):
            if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
                raise BadRequest("không phải file audio")
            size = p.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(p, "rb") as f:
                while chunk := f.read(1 << 20):
                    self.wfile.write(chunk)

        def do_POST(self):
            if urlparse(self.path).path != "/api/experiment/run":
                return self.fail("not found", 404)
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                return self.fail("origin không khớp", 403)
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self.fail("cần Content-Type: application/json", 415)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if not 0 < n <= MAX_BODY:
                    raise BadRequest("nội dung rỗng hoặc quá lớn")
                payload = json.loads(self.rfile.read(n))
                if not isinstance(payload, dict):
                    raise BadRequest("payload phải là object")
                self.send_json(start_run(cfg, payload), 202)
            except (BadRequest, ValueError) as e:
                self.fail(e)
            except OSError as e:
                self.fail(e, 500)

    return H


def load_page():
    sys.path.insert(0, str(TOOLS))
    import make_experiment_lab
    return make_experiment_lab.PAGE


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audio-root", action="append", required=True, help="folder the page may browse for input audio (repeatable)")
    ap.add_argument("--out-root", required=True, help="every run gets OUT_ROOT/lab_runs/<time>_<id>/")
    ap.add_argument("--jobid", default=os.environ.get("LAB_JOBID", ""), help="Slurm job id of your GPU allocation (srun --jobid)")
    ap.add_argument("--launcher", default="srun --jobid={jobid} --overlap",
                    help="prefix that runs the command on the GPU; '' runs it on this machine")
    ap.add_argument("--conda-env", default="som")
    ap.add_argument("--conda-sh", default="", help="path to conda.sh if it is not under ~/miniconda3 etc.")
    ap.add_argument("--python", default="python", help="interpreter name/path inside the conda env")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true", help="write the run directory and show the command, but start nothing")
    a = ap.parse_args(argv)
    if "{jobid}" in a.launcher and not a.jobid:
        ap.error("--jobid is required by the default --launcher (or pass --launcher '')")
    cfg = Config(a.audio_root, a.out_root, a.jobid, a.launcher, a.conda_env, a.conda_sh, a.python, a.dry_run)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(cfg, load_page()))
    print(f"[lab] http://{a.host}:{a.port}  audio={[str(r) for r in cfg.audio_roots]}  out={cfg.out_root}  "
          f"launcher={a.launcher.format(jobid=a.jobid) or '(local)'}  env={a.conda_env}{'  DRY-RUN' if a.dry_run else ''}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
