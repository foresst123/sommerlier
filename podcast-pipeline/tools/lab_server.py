#!/usr/bin/env python3
"""Backend for the Denoise Lab: serves the page, browses audio on this server and runs real jobs.

    python tools/lab_server.py --audio-root /home/lamkd2/data --out-root /home/lamkd2/data/denoise_lab --jobid 17405
    # per-backend env vars (REUSE_PYTHON, AUDIO_SEPARATOR_MODEL_DIR, ...) are resolved from tools/lab_env.py, --set-env overrides
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
import array
import json
import mimetypes
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import wave
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
ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,60}$")
MAX_BODY = 1 << 20
MAX_LIST = 2000
MAX_MANIFEST_BYTES = 2 << 20
MAX_ASR_BYTES = 8 << 20
MAX_ASR_SEGMENTS = 5000
MAX_CONVERSATION_BYTES = 2 << 20
MAX_CONVERSATIONS = 500
MAX_TURNS = 1000
CONVERSATION_TRACKS = {"mixture", "assistant", "user", "stereo_2ch"}
DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 100
MAX_QUERY_LENGTH = 120
LOG_TAIL = 6000
WAVEFORM_BINS = 4096
CONDA_SH = ('"$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" '
            '"$HOME/miniforge3/etc/profile.d/conda.sh" "$HOME/mambaforge/etc/profile.d/conda.sh" '
            '/opt/conda/etc/profile.d/conda.sh')


class BadRequest(Exception):
    pass


class ASRArtifactError(BadRequest):
    """An output artifact exists but cannot be safely decoded."""

    code = 422


def _page_args(limit=None, cursor=None, search=None, review=None):
    """Validate bounded list controls shared by the ASR/conversation APIs."""
    if limit in (None, ""):
        page_limit = DEFAULT_PAGE_LIMIT
    else:
        try:
            page_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise BadRequest("limit không hợp lệ") from exc
        if str(page_limit) != str(limit).strip():
            raise BadRequest("limit không hợp lệ")
    if not 1 <= page_limit <= MAX_PAGE_LIMIT:
        raise BadRequest("limit phải trong 1..100")
    if cursor in (None, ""):
        offset = 0
    else:
        try:
            offset = int(cursor)
        except (TypeError, ValueError) as exc:
            raise BadRequest("cursor không hợp lệ") from exc
        if str(offset) != str(cursor).strip() or offset < 0:
            raise BadRequest("cursor không hợp lệ")
    term = "" if search is None else str(search)
    if len(term) > MAX_QUERY_LENGTH:
        raise BadRequest("search quá dài")
    if review not in (None, "", "all", "flagged"):
        raise BadRequest("review không hợp lệ")
    return page_limit, offset, term.casefold(), review or "all"


def _page(items, limit, offset):
    """Return a bounded page plus stable integer cursors (never artifact paths)."""
    total = len(items)
    offset = min(offset, total)
    rows = items[offset:offset + limit]
    return rows, {"resultCount": total, "limit": limit,
                  "cursor": str(offset) if offset else None,
                  "prevCursor": str(max(0, offset - limit)) if offset else None,
                  "nextCursor": str(offset + limit) if offset + limit < total else None}


def _ffprobe(path):
    """Return duration/sample-rate without sending the audio through the browser."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=sample_rate",
             "-of", "json", str(path)], stderr=subprocess.DEVNULL, text=True, timeout=30)
        data = json.loads(out)
        duration = float((data.get("format") or {}).get("duration") or 0)
        stream = next((s for s in data.get("streams", []) if s.get("sample_rate")), {})
        rate = int(stream.get("sample_rate") or 0)
        if duration > 0:
            return duration, rate or 48000
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as src:
                return src.getnframes() / max(1, src.getframerate()), src.getframerate()
        except (OSError, wave.Error):
            pass
    raise BadRequest("không đọc được metadata audio (cần ffprobe cho định dạng này)")


def audio_summary(cfg, path):
    """Build/cache compact waveform peaks server-side, never allocating decoded PCM."""
    p = safe_path(path, cfg.read_roots)
    if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
        raise BadRequest("không phải file audio")
    stat = p.stat()
    key = f"{p}:{stat.st_size}:{stat.st_mtime_ns}"
    import hashlib
    cache = cfg.out_root / ".lab_waveforms" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    try:
        cached = json.loads(cache.read_text())
        if cached.get("version") == 1:
            return cached
    except (OSError, ValueError):
        pass
    duration, rate = _ffprobe(p)
    total = max(1, int(duration * rate))
    peaks = [0.0] * WAVEFORM_BINS
    try:
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-f", "f32le", "-"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise BadRequest("không tạo được waveform (cần ffmpeg trên server)") from exc
    pos = 0
    assert proc.stdout is not None
    try:
        while chunk := proc.stdout.read(1 << 20):
            usable = len(chunk) - len(chunk) % 4
            values = array.array("f")
            values.frombytes(chunk[:usable])
            if sys.byteorder != "little":
                values.byteswap()
            for value in values:
                bucket = min(WAVEFORM_BINS - 1, pos * WAVEFORM_BINS // total)
                amp = abs(value)
                if amp > peaks[bucket]:
                    peaks[bucket] = amp
                pos += 1
        _, err = proc.communicate(timeout=30)
    except (OSError, subprocess.SubprocessError):
        proc.kill()
        raise BadRequest("ffmpeg không đọc được audio để tạo waveform")
    if proc.returncode:
        raise BadRequest("ffmpeg không đọc được audio để tạo waveform")
    result = {"version": 1, "duration": duration, "sampleRate": rate, "bins": WAVEFORM_BINS,
              "peaks": peaks, "name": p.name, "size": stat.st_size}
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, separators=(",", ":")))
        os.replace(tmp, cache)
    except OSError:
        pass
    return result


class Config:
    def __init__(self, audio_roots, out_root, jobid, launcher, conda_env, conda_sh, python, dry_run, extra_env=None):
        # --audio-root must be a folder; given a file, browse the folder that holds it
        self.audio_roots = [(r.parent if r.is_file() else r) for r in (Path(p).expanduser().resolve() for p in audio_roots)]
        self.out_root = Path(out_root).expanduser().resolve()
        self.jobid, self.launcher = str(jobid or ""), launcher
        self.conda_env, self.conda_sh, self.python, self.dry_run = conda_env, conda_sh, python, dry_run
        self.extra_env = dict(extra_env or {})  # exported inside the job: REUSE_PYTHON, REUSE_DIR, ...
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
    if p.is_file():
        p = p.parent  # a file path lists the folder it sits in
    if not p.is_dir():
        raise BadRequest(f"thư mục không tồn tại: {raw}")
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
    """'15' -> 15, '0.5' -> 0.5, 'true'/'false' -> bool, anything else stays a string (the UI sends select values as text)."""
    if isinstance(value, bool):
        return value
    if value is None:
        raise BadRequest("giá trị tham số không hợp lệ")
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
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
            pass  # latest step param wins
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
    vid = sanitize("+".join(chain), "variant")  # named after what it runs, not after the UI trial id (a trial called dfn3 may run re_use)
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
    if s.get("music_removal") is False:  # BS-RoFormer music stage off, baseline included (an absent key keeps the pipeline default)
        argv += ["--set", "steps.music_removal=false"]
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
    # stdout goes to a file, where Python would otherwise buffer it and the log stays empty while the job runs
    py = shlex.quote(cfg.python)
    diag = ('echo "[lab] host=$(hostname) job=${SLURM_JOB_ID-?} step=${SLURM_STEP_ID-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"\n'
            'nvidia-smi -L 2>&1 | head -8\n'
            f'{py} -c "import torch; print(\'[lab] torch\', torch.__version__, \'cuda_available=\', torch.cuda.is_available(), \'devices=\', torch.cuda.device_count())" 2>&1 | tail -2\n')
    exports = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in cfg.extra_env.items())
    if cfg.extra_env:  # names only: a value may be a token
        diag += f'echo "[lab] env set: {" ".join(cfg.extra_env)}"\n'
    script = (conda_prelude(cfg) + "export PYTHONUNBUFFERED=1\n" + exports + diag
              + f"cd {shlex.quote(str(PIPELINE))} || exit 98\nexec {shlex.join(cmd)}\n")
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


def manifest_variants(run_dir):
    try:
        data = json.loads((run_dir / "out" / "variants_manifest.json").read_text())
    except (OSError, ValueError):
        return []
    return [{"id": v.get("id"), "status": v.get("status"), "reason": v.get("reason")}
            for v in data.get("variants", []) if isinstance(v, dict)]


def manifest_problem(run_dir, vid):
    """run_variants exits 0 for a skipped or degraded variant (no GPU work happened): say so instead of 'finished'."""
    for v in manifest_variants(run_dir):
        if v["id"] == vid and v["status"] != "ok":
            return f"variant {vid}: {v['status']}" + (f" - {v['reason']}" if v.get("reason") else "")
    return None


def reap(proc, run_dir):
    rc = proc.wait()
    vid = read_status(run_dir).get("id")
    problem = manifest_problem(run_dir, vid) if rc == 0 and vid else None
    write_status(run_dir, state="failed" if rc or problem else "finished", returncode=rc, problem=problem,
                 finishedAt=datetime.now().isoformat(timespec="seconds"), finishedEpoch=time.time())


def start_run(cfg, payload):
    variants, tail, info = build_run(cfg, payload)
    rid = sanitize(payload.get("runId"), "run")
    run_dir = cfg.out_root / "lab_runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{rid}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "variants.json").write_text(json.dumps(variants, ensure_ascii=False, indent=1))
    (run_dir / "payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    full, plain = launch_argv(cfg, tail, run_dir / "variants.json", run_dir / "out")
    base = {"runId": rid, "startedEpoch": time.time(), "runDir": str(run_dir), "log": str(run_dir / "run.log"), "out": str(run_dir / "out"),
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


def _run_out(cfg, run_id):
    """Resolve a server-owned run output directory, never a client path."""
    run_dir = find_run_dir(cfg, run_id)
    if run_dir is None or run_dir.is_symlink() or not run_dir.is_dir():
        raise BadRequest("không tìm thấy lượt chạy")
    root = (cfg.out_root / "lab_runs").resolve()
    resolved = run_dir.resolve()
    if not inside(resolved, [root]):
        raise BadRequest("lượt chạy không hợp lệ")
    out = run_dir / "out"
    if out.is_symlink() or not out.is_dir() or out.resolve() != out:
        raise BadRequest("thư mục output không hợp lệ")
    return run_dir, out


def _safe_manifest_file(out, raw):
    """Allow only a regular, non-symlink file physically below run/out."""
    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise ASRArtifactError("đường dẫn artifact không hợp lệ")
    out = out.resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = out / candidate
    try:
        rel = candidate.resolve(strict=False).relative_to(out)
    except ValueError as exc:
        raise ASRArtifactError("artifact nằm ngoài output của lượt chạy") from exc
    # Reject symlink components, including a symlinked file whose resolved path
    # happens to land back inside out.
    cur = out
    for part in rel.parts:
        cur = cur / part
        try:
            if cur.is_symlink():
                raise ASRArtifactError("artifact symlink không được phép")
        except OSError as exc:
            raise ASRArtifactError("không thể kiểm tra artifact") from exc
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ASRArtifactError("artifact không tồn tại") from exc
    if not inside(resolved, [out]) or not resolved.is_dir():
        raise ASRArtifactError("artifact nằm ngoài output của lượt chạy")
    return resolved


def _safe_artifact_file(out, raw, label="artifact"):
    """Resolve a regular local artifact below run/out; reject symlink components."""
    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise ASRArtifactError(f"{label} không hợp lệ")
    root = out.resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        rel = candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ASRArtifactError(f"{label} nằm ngoài output") from exc
    cur = root
    for part in rel.parts:
        cur /= part
        if cur.is_symlink():
            raise ASRArtifactError(f"{label} symlink không được phép")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ASRArtifactError(f"{label} không tồn tại") from exc
    if not inside(resolved, [root]) or not resolved.is_file():
        raise ASRArtifactError(f"{label} không hợp lệ")
    return resolved


def _safe_manifest_probe(out, raw):
    """Resolve a known output path for readiness checks, allowing it to be absent."""
    root = out.resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        rel = candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ASRArtifactError("artifact nằm ngoài output") from exc
    cur = root
    for part in rel.parts:
        cur /= part
        try:
            if cur.is_symlink():
                raise ASRArtifactError("artifact symlink không được phép")
        except OSError as exc:
            raise ASRArtifactError("không thể kiểm tra artifact") from exc
    return candidate


def _read_json_limited(path, limit, label):
    try:
        if path.stat().st_size > limit:
            raise ASRArtifactError(f"{label} quá lớn")
        return json.loads(path.read_text(encoding="utf-8"))
    except ASRArtifactError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise ASRArtifactError(f"{label} bị hỏng") from exc


def workspace_asr(cfg, run_id, variant_id=None, recording=None, limit=None, cursor=None, search=None):
    """Read one ASR recording selected solely from its run-owned manifest."""
    run_dir, out = _run_out(cfg, run_id)
    manifest_path = out / "variants_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return {"runId": str(run_id), "status": "pending", "variants": []}
    data = _read_json_limited(manifest_path, MAX_MANIFEST_BYTES, "manifest")
    variants = data.get("variants") if isinstance(data, dict) else None
    if not isinstance(variants, list):
        raise ASRArtifactError("manifest không hợp lệ")
    safe_variants = [v for v in variants if isinstance(v, dict) and isinstance(v.get("id"), str)]
    if variant_id is None:
        return {"runId": str(run_id), "status": "ready", "variants": [
            {"id": v["id"], "label": v.get("label") or v["id"], "status": v.get("status", "unknown"),
             "recordings": sorted((v.get("files") or {}).keys()) if isinstance(v.get("files"), dict) else []}
            for v in safe_variants]}
    selected = next((v for v in safe_variants if v["id"] == str(variant_id)), None)
    if selected is None:
        raise BadRequest("không tìm thấy variant")
    files = selected.get("files")
    if not isinstance(files, dict):
        return {"runId": str(run_id), "variantId": str(variant_id), "status": "pending", "recordings": []}
    names = sorted(str(k) for k in files if isinstance(k, str) and ID_RE.match(k))
    if recording is None:
        return {"runId": str(run_id), "variantId": str(variant_id), "status": "ready", "recordings": names}
    if recording not in files or recording not in names:
        raise BadRequest("không tìm thấy recording")
    page_limit, offset, term, _review = _page_args(limit, cursor, search)
    base = _safe_manifest_file(out, files[recording])
    asr_path = base / "05_asr" / "transcripts.json"
    # Check every output component again: a manifest may point at a legitimate
    # directory while a later stage swaps in a symlink.
    _safe_manifest_file(out, str(asr_path.parent))
    if asr_path.is_symlink() or not asr_path.is_file():
        return {"runId": str(run_id), "variantId": str(variant_id), "recording": recording,
                "status": "pending", "segments": []}
    transcript = _read_json_limited(asr_path, MAX_ASR_BYTES, "ASR artifact")
    if not isinstance(transcript, list):
        raise ASRArtifactError("ASR artifact không phải danh sách")
    if len(transcript) > MAX_ASR_SEGMENTS:
        raise ASRArtifactError("ASR artifact có quá nhiều đoạn")
    segments = [x for x in transcript if isinstance(x, dict)]
    if term:
        segments = [x for x in segments if term in json.dumps(x, ensure_ascii=False).casefold()]
    segments, paging = _page(segments, page_limit, offset)
    return {"runId": str(run_id), "variantId": str(variant_id), "recording": recording,
            "status": "empty" if not segments else "ready", "segments": segments, **paging}


def workspace_conversations(cfg, run_id, variant_id=None, recording=None, conversation_id=None,
                            limit=None, cursor=None, search=None, review=None):
    """Read conversation exports selected through the same run manifest as ASR."""
    _run_dir, out = _run_out(cfg, run_id)
    manifest_path = out / "variants_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return {"runId": str(run_id), "status": "pending", "variants": []}
    data = _read_json_limited(manifest_path, MAX_MANIFEST_BYTES, "manifest")
    variants = data.get("variants") if isinstance(data, dict) else None
    if not isinstance(variants, list):
        raise ASRArtifactError("manifest không hợp lệ")
    safe_variants = [v for v in variants if isinstance(v, dict) and isinstance(v.get("id"), str)]
    if variant_id is None:
        return {"runId": str(run_id), "status": "ready", "variants": [
            {"id": v["id"], "label": v.get("label") or v["id"], "status": v.get("status", "unknown"),
             "recordings": sorted((v.get("files") or {}).keys()) if isinstance(v.get("files"), dict) else []}
            for v in safe_variants]}
    selected = next((v for v in safe_variants if v["id"] == str(variant_id)), None)
    if selected is None:
        raise BadRequest("không tìm thấy variant")
    files = selected.get("files")
    if not isinstance(files, dict):
        return {"runId": str(run_id), "variantId": str(variant_id), "status": "pending", "recordings": []}
    names = sorted(str(k) for k in files if isinstance(k, str) and ID_RE.match(k))
    if recording is None:
        return {"runId": str(run_id), "variantId": str(variant_id), "status": "ready", "recordings": names}
    if recording not in files or recording not in names:
        raise BadRequest("không tìm thấy recording")
    page_limit, offset, term, review_mode = _page_args(limit, cursor, search, review)
    base = _safe_manifest_file(out, files[recording])
    export_dir = base / "09_conversation_exports"
    _safe_manifest_file(out, str(export_dir))
    exports_path = export_dir / "exports.json"
    if exports_path.is_symlink() or not exports_path.is_file():
        return {"runId": str(run_id), "variantId": str(variant_id), "recording": recording,
                "status": "pending", "conversations": []}
    exports = _read_json_limited(exports_path, MAX_CONVERSATION_BYTES, "conversation exports")
    if not isinstance(exports, list):
        raise ASRArtifactError("conversation exports không phải danh sách")
    if len(exports) > MAX_CONVERSATIONS:
        raise ASRArtifactError("quá nhiều conversation")
    rows = [x for x in exports if isinstance(x, dict) and isinstance(x.get("id"), str)]
    if conversation_id is None:
        if review_mode == "flagged":
            # Canonical review state is deliberately narrow: only the export
            # row's boolean quarantine/verification fields can flag an item.
            rows = [x for x in rows if x.get("quarantined") is True or x.get("verified") is False]
        if term:
            rows = [x for x in rows if term in json.dumps(x, ensure_ascii=False).casefold()]
        rows, paging = _page(rows, page_limit, offset)
        return {"runId": str(run_id), "variantId": str(variant_id), "recording": recording,
                "status": "empty" if not rows else "ready", "conversations": [
                    {k: x.get(k) for k in ("id", "category", "quarantined", "verified", "duration", "topic", "summary", "score")}
                    for x in rows], **paging}
    row = next((x for x in rows if x["id"] == str(conversation_id)), None)
    if row is None:
        raise BadRequest("không tìm thấy conversation")
    folder = row.get("folder")
    item = _safe_manifest_file(out, str(export_dir / folder)) if isinstance(folder, str) else None
    if item is None:
        raise ASRArtifactError("conversation folder không hợp lệ")
    meta_path = item / "conversation.json"
    meta = _read_json_limited(meta_path, MAX_CONVERSATION_BYTES, "conversation")
    if not isinstance(meta, dict):
        raise ASRArtifactError("conversation không hợp lệ")
    turns = meta.get("conversation") if isinstance(meta.get("conversation"), list) else []
    if len(turns) > MAX_TURNS:
        raise ASRArtifactError("quá nhiều lượt thoại")
    files_meta = meta.get("files") if isinstance(meta.get("files"), dict) else {}
    tracks = {k: v for k, v in files_meta.items() if k in CONVERSATION_TRACKS and isinstance(v, str)}
    audio = {track: "ready" for track, rel in tracks.items() if _safe_artifact_file(item, rel, "audio").suffix.lower() in AUDIO_EXT}
    return {"runId": str(run_id), "variantId": str(variant_id), "recording": recording,
            "conversationId": str(conversation_id), "status": "ready", "row": row,
            "meta": {k: meta.get(k) for k in ("id", "category", "intended_category", "tags", "events", "quality", "verification", "topic", "summary", "duration", "role", "overlap_quality")},
            "turns": turns, "audio": audio, "warnings": [w for w in (
                "Quarantine item: review before use" if row.get("quarantined") or meta.get("category") == "_quarantine" else None,
                "Verification failed; audio is not trusted" if row.get("verified") is False or (meta.get("verification") or {}).get("ok") is False else None,
            ) if w]}


def workspace_observe(cfg, run_id):
    """Return cheap run/artifact readiness without reading artifact contents.

    The observer intentionally only reads status.json and the run-owned manifest,
    then checks the two known output paths for each manifest entry.  It never
    walks an output tree or decodes ASR/conversation JSON.
    """
    run_dir, out = _run_out(cfg, run_id)
    state = read_status(run_dir).get("state", "unknown")
    result = {"runId": str(run_id), "status": state,
              "asr": {"available": False, "status": "pending"},
              "conversations": {"available": False, "status": "pending"}}
    manifest_path = out / "variants_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return result
    data = _read_json_limited(manifest_path, MAX_MANIFEST_BYTES, "manifest")
    variants = data.get("variants") if isinstance(data, dict) else None
    if not isinstance(variants, list):
        raise ASRArtifactError("manifest không hợp lệ")
    for variant in variants:
        if not isinstance(variant, dict) or not isinstance(variant.get("files"), dict):
            continue
        for raw_base in variant["files"].values():
            if not isinstance(raw_base, str):
                continue
            base = _safe_manifest_probe(out, raw_base)
            asr_path = base / "05_asr" / "transcripts.json"
            conv_path = base / "09_conversation_exports" / "exports.json"
            # Validate parent paths without opening or enumerating artifacts.
            _safe_manifest_probe(out, str(asr_path.parent))
            _safe_manifest_probe(out, str(conv_path.parent))
            if asr_path.is_file() and not asr_path.is_symlink():
                result["asr"] = {"available": True, "status": "ready"}
            if conv_path.is_file() and not conv_path.is_symlink():
                result["conversations"] = {"available": True, "status": "ready"}
            if result["asr"]["available"] and result["conversations"]["available"]:
                return result
    return result


def conversation_audio(cfg, run_id, variant_id, recording, conversation_id, track):
    if track not in CONVERSATION_TRACKS:
        raise BadRequest("track không được phép")
    detail = workspace_conversations(cfg, run_id, variant_id, recording, conversation_id)
    if detail.get("status") != "ready":
        raise BadRequest("conversation chưa sẵn sàng")
    _run_dir, out = _run_out(cfg, run_id)
    data = json.loads((out / "variants_manifest.json").read_text(encoding="utf-8"))
    selected = next(v for v in data["variants"] if v.get("id") == str(variant_id))
    base = _safe_manifest_file(out, selected["files"][recording])
    export_dir = base / "09_conversation_exports"
    exports = json.loads((export_dir / "exports.json").read_text(encoding="utf-8"))
    row = next(x for x in exports if x.get("id") == str(conversation_id))
    item = _safe_manifest_file(out, str(export_dir / row["folder"]))
    meta = _read_json_limited(item / "conversation.json", MAX_CONVERSATION_BYTES, "conversation")
    rel = (meta.get("files") or {}).get(track)
    if not isinstance(rel, str):
        raise BadRequest("audio track không khả dụng")
    return _safe_artifact_file(item, rel, "audio")


def tail(path, n=LOG_TAIL):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def tail_head(path, n=4000):
    try:
        with open(path, "rb") as f:
            return f.read(n).decode("utf-8", "replace")
    except OSError:
        return ""


def read_status(run_dir):
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
    return st


def elapsed(st):
    start = st.get("startedEpoch")
    if not start:
        return 0
    end = st.get("finishedEpoch") or (time.time() if st.get("state") == "running" else start)
    return max(0, round(end - start, 1))


def run_status(cfg, rid):
    run_dir = find_run_dir(cfg, rid)
    if run_dir is None:
        raise BadRequest("không tìm thấy lượt chạy")
    st = read_status(run_dir)
    st["elapsedSeconds"] = elapsed(st)
    st["logTail"] = tail(run_dir / "run.log")
    # the pipeline's own progress lives in out/logs/<variant>.log, not in the runner's log
    logs = sorted((run_dir / "out" / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime) if (run_dir / "out" / "logs").is_dir() else []
    head = tail_head(run_dir / "run.log")
    cuda = re.search(r"cuda_available=\s*(True|False)", head)
    st["cuda"] = None if not cuda else cuda.group(1) == "True"
    st["gpuInfo"] = [ln.strip() for ln in head.splitlines() if ln.startswith(("[lab]", "GPU "))][:6]
    st["variants"] = manifest_variants(run_dir)
    # while running, a lab variant that is already skipped/failed/degraded will never use the GPU: warn now, not at the end
    st["warning"] = None
    if st.get("state") == "running" and st.get("id"):
        bad = next((v for v in st["variants"] if v["id"] == st["id"] and v["status"] in ("skipped", "failed", "degraded")), None)
        if bad:
            st["warning"] = f"variant {bad['id']}: {bad['status']}" + (f" - {bad['reason']}" if bad.get("reason") else "")
    st["pipelineLog"] = logs[-1].name if logs else ""
    st["pipelineLogTail"] = tail(logs[-1]) if logs else ""
    st["outputs"] = sorted(str(p) for p in (run_dir / "out").glob("*/*/01b_denoise/after_denoise.wav"))[:200]
    return st


def list_runs(cfg, limit=10):
    root = cfg.out_root / "lab_runs"
    dirs = sorted((d for d in root.glob("*") if d.is_dir()), key=lambda d: d.name, reverse=True)[:max(1, min(limit, 50))] if root.is_dir() else []
    out = []
    for d in dirs:
        st = read_status(d)
        out.append({"runId": d.name.split("_", 1)[-1], "state": st.get("state"), "runDir": str(d), "chain": st.get("chain"),
                    "source": st.get("source"), "elapsedSeconds": elapsed(st), "startedEpoch": st.get("startedEpoch")})
    return {"runs": out}


def list_workspace_runs(cfg, limit=50):
    """Return cheap, read-only run metadata from the server-owned lab_runs root.

    This endpoint deliberately does not accept a path and never walks output
    artifacts.  Routine workspace polling therefore only reads each run's
    small status manifest and ignores symlinks/non-directories.
    """
    root = (cfg.out_root / "lab_runs").resolve()
    if not root.is_dir():
        return {"runs": []}
    try:
        candidates = [d for d in root.iterdir() if d.is_dir() and not d.is_symlink()]
    except OSError:
        return {"runs": []}
    candidates.sort(key=lambda d: d.name, reverse=True)
    out = []
    for d in candidates[:max(1, min(int(limit or 50), 100))]:
        st = read_status(d)
        run_id = d.name.split("_", 1)[-1] if "_" in d.name else d.name
        out.append({
            "runId": run_id,
            "state": st.get("state", "unknown"),
            "startedEpoch": st.get("startedEpoch"),
            "finishedEpoch": st.get("finishedEpoch"),
            "chain": st.get("chain") or [],
            "source": st.get("source"),
            "variantId": st.get("id"),
        })
    return {"runs": out}


def make_handler(cfg, page):
    class H(BaseHTTPRequestHandler):
        server_version = "DenoiseLab/1"

        def log_message(self, fmt, *a):
            if "/api/experiment/status" not in (a[0] if a else ""):
                super().log_message(fmt, *a)

        def send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def fail(self, e, code=400):
            print(f"[lab] HTTP {code}: {e}", flush=True)
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
                elif u.path == "/api/audio/summary":
                    self.send_json(audio_summary(cfg, q.get("path")))
                elif u.path == "/api/experiment/runs":
                    self.send_json(list_runs(cfg, int(q.get("limit", 10) or 10)))
                elif u.path == "/api/workspace/runs":
                    self.send_json(list_workspace_runs(cfg, int(q.get("limit", 50) or 50)))
                elif u.path == "/api/workspace/observe":
                    self.send_json(workspace_observe(cfg, q.get("runId")))
                elif u.path == "/api/workspace/asr":
                    self.send_json(workspace_asr(cfg, q.get("runId"), q.get("variantId"), q.get("recording"),
                                                 q.get("limit"), q.get("cursor"), q.get("search")))
                elif u.path == "/api/workspace/conversations":
                    self.send_json(workspace_conversations(cfg, q.get("runId"), q.get("variantId"), q.get("recording"),
                                                          q.get("conversationId"), q.get("limit"), q.get("cursor"),
                                                          q.get("search"), q.get("review")))
                elif u.path == "/api/workspace/conversation-audio":
                    self.send_audio(conversation_audio(cfg, q.get("runId"), q.get("variantId"), q.get("recording"), q.get("conversationId"), q.get("track")))
                elif u.path == "/api/experiment/status":
                    self.send_json(run_status(cfg, q.get("runId")))
                else:
                    self.fail("not found", 404)
            except ASRArtifactError as e:
                self.fail(e, e.code)
            except (BadRequest, ValueError) as e:
                self.fail(e)
            except (BrokenPipeError, ConnectionResetError):
                # A browser cancelling a range request is normal; the response
                # is already gone, so never attempt to write a second error.
                return
            except OSError as e:
                self.fail(e, 500)

        def send_audio(self, p):
            if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
                raise BadRequest("không phải file audio")
            size = p.stat().st_size
            start, end, code = 0, size - 1, 200
            raw_range = self.headers.get("Range")
            if raw_range:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", raw_range.strip())
                if not match:
                    self.send_error(416)
                    return
                start = int(match.group(1) or 0)
                end = int(match.group(2) or size - 1)
                if start >= size or start > end:
                    self.send_error(416)
                    return
                end, code = min(end, size - 1), 206
            self.send_response(code)
            self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(p, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining and (chunk := f.read(min(1 << 20, remaining))):
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

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
            except (BrokenPipeError, ConnectionResetError):
                return
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
    ap.add_argument("--set-env", action="append", default=[], metavar="KEY=VALUE",
                    help="export this variable inside the job (repeatable); beats tools/lab_env.py and the shell")
    ap.add_argument("--no-lab-env", action="store_true", help="do not read tools/lab_env.py (export only --set-env)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true", help="write the run directory and show the command, but start nothing")
    a = ap.parse_args(argv)
    if "{jobid}" in a.launcher and not a.jobid:
        ap.error("--jobid is required by the default --launcher (or pass --launcher '')")
    extra = {}
    for item in a.set_env:
        key, sep, val = item.partition("=")
        if not sep or not ENV_RE.match(key) or "\0" in val or "\n" in val:
            ap.error(f"--set-env expects KEY=VALUE with KEY like REUSE_PYTHON, got {item!r}")
        extra[key] = os.path.expanduser(val)
    report = []
    if not a.no_lab_env:  # tools/lab_env.py: where each backend's env/weights live on this machine
        sys.path.insert(0, str(TOOLS))
        import lab_env
        extra, report = lab_env.resolve(extra)
    cfg = Config(a.audio_root, a.out_root, a.jobid, a.launcher, a.conda_env, a.conda_sh, a.python, a.dry_run, extra)
    for r in cfg.audio_roots:  # resolve() accepts paths that do not exist, so say so here instead of failing in the browser
        if not r.is_dir():
            ap.error(f"--audio-root không phải thư mục tồn tại: {r}"
                     + (" (symlink hỏng?)" if r.is_symlink() else "") + "  -> kiểm tra: ls -ld <đường dẫn>")
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(cfg, load_page()))
    print(f"[lab] http://{a.host}:{a.port}  audio={[str(r) for r in cfg.audio_roots]}  out={cfg.out_root}  "
          f"launcher={a.launcher.format(jobid=a.jobid) or '(local)'}  env={a.conda_env}{'  DRY-RUN' if a.dry_run else ''}", flush=True)
    for line in report:
        print(f"[lab] env {line}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
