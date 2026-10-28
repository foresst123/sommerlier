"""Backend of the diarization lab: run only the diarization stage on a chosen stretch of audio,
look at the result, run again with other parameters and compare two results.

``handle`` (GET) and ``handle_post`` (POST) serve these routes and return False for any other
path, so the dashboard can run its own routes after them:

    GET  /diar                          the page (tools/diar_lab_page.html, next to this file)
    GET  /api/diar/audio?q=&limit=      audio under audio_dir (files up to 2 subfolder levels deep)
    GET  /api/diar/params               parameter schema; defaults read from config.json (env profile)
    GET  /api/diar/gpus                 nvidia-smi
    POST /api/diar/run                  {audio,start,end,params,gpu,label} -> 201 {"id"}
    GET  /api/diar/runs                 runs, newest first
    GET  /api/diar/run?id=              state, log tail, params, summary
    GET  /api/diar/result?id=           segments and stats of a finished run (409 before that)
    GET  /api/diar/audio_file?id=       the run's input.wav, with Range
    GET  /api/diar/compare?a=&b=        two finished runs compared on a 10 ms grid
    POST /api/diar/stop                 {id}: stop a queued or running run
    POST /api/diar/delete               {id, confirm: true}: delete a run that is not active

``ctx`` (optional dict): ``audio_dir`` (None or missing gives an empty list), ``lab_dir`` (default
``out/diar_lab``, where ``out`` defaults to pipeline_dir/output), ``pipeline_dir`` (default: the
parent of this file's folder), ``python`` (default sys.executable), ``env_name`` (default "a100"),
``config_path`` (default pipeline_dir/config.json). ``final`` is not used.

Runs go through one background thread that takes them in order, so at most one run is active
and no two runs share a GPU. Each run lives in ``lab_dir/runs/<id>/``: input.wav, params.json,
status.json, run.log, out/ (main.py's --save_path), cache/.

Compare assumptions: both results are quantized to 10 ms frames. For each frame, A's and B's
active speaker sets are A and B. The one-to-one speaker mapping maximizes the total frames in
which the two labels are active together (Hungarian algorithm, exact for any label count). Then
per frame, with na = |A| and nb = |B| and "correct" = labels of A mapped into B, the error
slots are miss = max(0, na-nb), false alarm = max(0, nb-na), confusion = min(na,nb)-correct.
Times are speaker-seconds (overlapping speakers count twice), so der_like = (miss+fa+conf) /
speaker-seconds of A, the usual DER form without a collar. Lane kind precedence: conf, miss, fa.

Standard library only; soundfile and numpy are used when they import, ffmpeg otherwise.
"""
import importlib
import json
import math
import os
import re
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from collections import Counter, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE_PATH = HERE / "diar_lab_page.html"
PIPELINE_DIR = HERE.parent
DEFAULT_ENV = "a100"
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac")
SCAN_LEVELS = 2             # subfolder levels below audio_dir that are scanned
LIST_TTL = 60.0
LIST_DEFAULT = 200
LIST_MAX = 1000
MAX_BODY = 64 * 1024
MAX_RUN_SECONDS = 1800.0
SAMPLE_RATE = 16000
INPUT_NAME = "input.wav"
STEM = "input"              # main.py writes its outputs to <save_path>/<stem of --audio>/
RESULT_DIR = Path("out") / STEM / "02_diarization"
RUNS = "runs"
RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TAIL_LINES = 200
TAIL_BYTES = 256 * 1024
STOP_KILL_AFTER = 5.0       # seconds after TERM before KILL
STOP_WAIT = 15.0
GRID = 0.01                 # compare grid, seconds
LANES_MAX = 5000
CHUNK = 1 << 20
MAX_GPU = 15
_RANGE = re.compile(r"bytes\s*=\s*([0-9]*)\s*-\s*([0-9]*)", re.IGNORECASE)
_MISSING = object()

_LOCK = threading.Lock()
_LIST_CACHE = {}       # audio root -> (monotonic time, entries)
_DUR_CACHE = {}        # real path -> (stamp, seconds)
_RESULTS = {}          # segments.json path -> (stamp, parsed result)
_MODS = {}             # module name -> module or None

# Run queue. Everything below is guarded by _CV.
_CV = threading.Condition()
_QUEUE = deque()       # (run id, run dir, settings)
_PENDING = set()       # ids being created (input cut in progress)
_CURRENT = None        # id of the run the worker is executing
_PROCS = {}            # run id -> Popen while it runs
_STOPS = set()         # run ids the user asked to stop
_WORKER = None


# -- parameter schema ---------------------------------------------------------------------
# (key, label, type, spec, help). spec: (min, max, step) for int/float, choices for choice.
# A key missing from the env profile of config.json is left out of the schema.
_GROUPS = (
    ("backend", "Backend & chế độ", (
        ("models.diarization_backend", "Backend", "choice", ("nemotron", "diarizen"),
         "nemotron: mô hình streaming của NVIDIA (dùng models.nemotron.*). diarizen: DiariZen "
         "(dùng models.diarizen.*). Hai backend cho kết quả khác hẳn nhau, nên so sánh chúng "
         "trên cùng một đoạn audio."),
        ("models.nemotron.two_pass", "Hai lượt", "bool", None,
         "Bật: chạy hai lượt (VAD Silero cắt cửa sổ, rồi xử lý từng cửa sổ có ngữ cảnh). Tắt: "
         "chạy đường một lượt."),
        ("models.nemotron.legacy_one_pass", "Một lượt cũ", "bool", None,
         "Ép đường một lượt cũ, bỏ qua Hai lượt. Dùng để đối chiếu với kết quả trước khi có "
         "hai lượt."),
        ("models.nemotron.batch_size", "Batch size", "int", (1, 64, 1),
         "Số cửa sổ xử lý cùng lúc trên GPU. Chủ yếu ảnh hưởng tốc độ và VRAM; giảm nếu hết "
         "bộ nhớ GPU."),
    )),
    ("vad", "VAD Silero (nemotron)", (
        ("models.nemotron.vad_threshold", "Ngưỡng VAD", "float", (0.05, 0.95, 0.01),
         "Xác suất giọng nói tối thiểu của Silero VAD. Tăng: cắt chặt hơn, dễ bỏ sót lời nhỏ "
         "hoặc thì thầm. Giảm: giữ nhiều tạp âm thành đoạn nói hơn."),
        ("models.nemotron.vad_min_speech_ms", "VAD: đoạn nói tối thiểu (ms)", "int", (0, 1000, 10),
         "Đoạn VAD ngắn hơn giá trị này bị bỏ. Tăng để loại tiếng 'ừ', 'à' ngắn, nhưng dễ mất "
         "backchannel."),
        ("models.nemotron.vad_min_silence_ms", "VAD: khoảng lặng tối thiểu (ms)", "int", (0, 2000, 10),
         "Khoảng lặng phải dài ít nhất này mới tách hai đoạn nói. Tăng để nối các đoạn ngắt "
         "quãng; giảm để tách chặt hơn."),
        ("models.nemotron.vad_pad_ms", "VAD: nới hai đầu (ms)", "int", (0, 1000, 10),
         "Nới thêm ms vào đầu và cuối mỗi đoạn VAD, để không cắt mất đầu hoặc cuối lời. Đổi "
         "lại có thể lấn sang lời người khác."),
    )),
    ("window", "Cửa sổ (nemotron)", (
        ("models.nemotron.chunk_target_seconds", "Cửa sổ: độ dài mong muốn (s)", "float", (5, 60, 0.5),
         "Độ dài mục tiêu của một cửa sổ xử lý. Dài hơn: ít mép cắt hơn, nhưng tốn VRAM hơn."),
        ("models.nemotron.chunk_min_seconds", "Cửa sổ: tối thiểu (s)", "float", (1, 60, 0.5),
         "Cửa sổ ngắn hơn giá trị này được gộp với cửa sổ bên cạnh. Phải nhỏ hơn hoặc bằng độ dài mục tiêu."),
        ("models.nemotron.chunk_max_seconds", "Cửa sổ: tối đa (s)", "float", (5, 120, 0.5),
         "Cửa sổ không dài quá giá trị này. Phải lớn hơn hoặc bằng độ dài mục tiêu."),
        ("models.nemotron.context_seconds", "Ngữ cảnh hai bên (s)", "float", (0, 10, 0.5),
         "Số giây ngữ cảnh thêm ở hai bên cửa sổ, để nhãn ổn định ở mép. Tăng: ổn định hơn "
         "nhưng chậm hơn."),
    )),
    ("merge", "Gộp người nói & số người nói", (
        ("models.nemotron.wespeaker_fallback", "Gộp bằng WeSpeaker", "bool", None,
         "Khi nhãn của các cửa sổ không khớp nhau, dùng embedding WeSpeaker để ghép người nói "
         "toàn cục. Tắt: không ghép, một người có thể bị tách thành nhiều nhãn."),
        ("models.nemotron.wespeaker_min_clean_speech_seconds", "WeSpeaker: giây nói sạch tối thiểu (s)",
         "float", (0, 10, 0.1),
         "Số giây nói sạch tối thiểu của một người để tin embedding khi ghép. Tăng: ghép thận "
         "trọng hơn, nhiều người không được ghép."),
        ("models.diarizen.ahc_threshold", "DiariZen: ngưỡng gom cụm", "float", (0, 2, 0.01),
         "Ngưỡng dừng gom cụm của DiariZen. Tăng: gom mạnh hơn, ít người nói hơn. Giảm: tách "
         "nhiều người hơn. Chỉ có tác dụng với backend diarizen."),
        ("models.diarizen.min_speakers", "DiariZen: số người nói tối thiểu", "int", (1, 20, 1),
         "Đặt khi biết trước số người nói tối thiểu. Chỉ diarizen dùng; nemotron có đầu ra cố định 8 khe."),
        ("models.diarizen.max_speakers", "DiariZen: số người nói tối đa", "int", (1, 20, 1),
         "Đặt khi biết trước số người nói tối đa, để không tách quá nhiều nhãn. Chỉ diarizen dùng."),
    )),
    ("post", "Hậu xử lý (nemotron)", (
        ("models.nemotron.postprocessing.onset", "Ngưỡng bật (onset)", "float", (0, 1, 0.01),
         "Xác suất để bật một đoạn nói. Cao hơn: chặt hơn, dễ mất đầu lời."),
        ("models.nemotron.postprocessing.offset", "Ngưỡng tắt (offset)", "float", (0, 1, 0.01),
         "Xác suất để tắt một đoạn nói. Thấp hơn: kéo đoạn nói dài ra sau."),
        ("models.nemotron.postprocessing.pad_onset", "Nới trước đoạn (s)", "float", (0, 1, 0.01),
         "Nới thêm giây trước mỗi đoạn nói."),
        ("models.nemotron.postprocessing.pad_offset", "Nới sau đoạn (s)", "float", (0, 1, 0.01),
         "Nới thêm giây sau mỗi đoạn nói."),
        ("models.nemotron.postprocessing.min_duration_on", "Đoạn nói tối thiểu (s)", "float", (0, 2, 0.01),
         "Đoạn nói ngắn hơn giá trị này (giây) bị bỏ."),
        ("models.nemotron.postprocessing.min_duration_off", "Khoảng lặng tối thiểu (s)", "float", (0, 2, 0.01),
         "Khoảng lặng ngắn hơn giá trị này (giây) được lấp, nối hai đoạn nói."),
        ("models.nemotron.merge_gap", "Gộp hai đoạn cùng người (s)", "float", (0, 5, 0.05),
         "Hai đoạn của cùng một người cách nhau không quá giá trị này (giây) được gộp thành một."),
        ("models.nemotron.min_duration", "Bỏ đoạn quá ngắn (s)", "float", (0, 5, 0.05),
         "Sau khi gộp, đoạn ngắn hơn giá trị này (giây) bị bỏ."),
    )),
    ("pre", "Tiền xử lý", (
        ("pipeline.denoise", "Lọc nhiễu", "choice", ("none",),
         "Bộ lọc nhiễu trước diarization. none: không lọc. Giá trị còn lại là chuỗi lọc đang "
         "có trong config.json."),
    )),
)


class _HTTPError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _get(profile, dotted):
    node = profile
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _profile(config_path, env):
    try:
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
        prof = data["environments"][env]
    except Exception:
        return {}
    return prof if isinstance(prof, dict) else {}


def _unique(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _param(prof, row):
    key, label, kind, spec, help_text = row
    default = _get(prof, key)
    if default is _MISSING:
        return None
    item = {"key": key, "label": label, "type": kind, "default": default,
            "min": None, "max": None, "step": None, "choices": None, "help": help_text}
    if kind == "choice":
        if not isinstance(default, str):
            return None
        item["choices"] = _unique([default, *spec])
    elif kind == "bool":
        if not isinstance(default, bool):
            return None
    else:
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            return None
        lo, hi, step = spec
        if kind == "int":
            if float(default) != int(default):
                return None
            default = int(default)
            lo, hi = int(lo), int(hi)
        else:
            default = float(default)
        item.update(default=default, min=lo, max=hi, step=step)
    return item


def _schema(cfg):
    prof = _profile(cfg["config_path"], cfg["env_name"])
    groups = []
    for gid, title, rows in _GROUPS:
        params = [p for p in (_param(prof, row) for row in rows) if p is not None]
        if params:
            groups.append({"id": gid, "title": title, "params": params})
    return {"env": cfg["env_name"], "groups": groups, "_profile_found": bool(prof)}


def _defaults(schema):
    return {item["key"]: item["default"] for g in schema["groups"] for item in g["params"]}


def _coerce(item, value):
    key, kind = item["key"], item["type"]
    if kind == "bool":
        if not isinstance(value, bool):
            raise _HTTPError(400, f"{key}: cần true hoặc false")
        return value
    if kind == "choice":
        if not isinstance(value, str) or value not in item["choices"]:
            raise _HTTPError(400, f"{key}: chọn một trong {item['choices']}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _HTTPError(400, f"{key}: cần một số")
    if kind == "int":
        if float(value) != int(value):
            raise _HTTPError(400, f"{key}: cần số nguyên")
        value = int(value)
    else:
        value = float(value)
    if value < item["min"] or value > item["max"]:
        raise _HTTPError(400, f"{key}: {value} ngoài khoảng {item['min']}..{item['max']}")
    return value


def _check_combo(eff):
    lo = eff.get("models.nemotron.chunk_min_seconds")
    tg = eff.get("models.nemotron.chunk_target_seconds")
    hi = eff.get("models.nemotron.chunk_max_seconds")
    if None not in (lo, tg, hi) and not (lo <= tg <= hi):
        raise _HTTPError(400, "cần chunk_min_seconds <= chunk_target_seconds <= chunk_max_seconds")
    mn = eff.get("models.diarizen.min_speakers")
    mx = eff.get("models.diarizen.max_speakers")
    if None not in (mn, mx) and mn > mx:
        raise _HTTPError(400, "min_speakers không được lớn hơn max_speakers")


# -- entry points -------------------------------------------------------------------------
def handle(h, path, query, ctx=None) -> bool:
    """Serve one GET if ``path`` is one of ours. True when a response was sent."""
    try:
        fn = _GET_ROUTES.get(path)
        if fn is None:
            return False
        cfg = _settings(ctx)
        _guard(h, fn, query or {}, cfg)
        return True
    except Exception:
        _safe_json(h, 500, {"error": "internal error"})
        return True


def handle_post(h, path, ctx=None) -> bool:
    """Serve one POST if ``path`` is one of ours. True when a response was sent."""
    try:
        fn = _POST_ROUTES.get(path)
        if fn is None:
            return False
        cfg = _settings(ctx)
        _guard(h, fn, cfg)
        return True
    except Exception:
        _safe_json(h, 500, {"error": "internal error"})
        return True


def _guard(h, fn, *args):
    try:
        fn(h, *args)
    except _HTTPError as exc:
        _safe_json(h, exc.code, {"error": str(exc)})
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        pass
    except Exception as exc:
        _safe_json(h, 500, {"error": f"internal error: {type(exc).__name__}"})


def _safe_json(h, code, obj):
    try:
        _json(h, code, obj)
    except Exception:
        pass


# -- settings -----------------------------------------------------------------------------
def _settings(ctx):
    ctx = ctx if isinstance(ctx, dict) else {}
    pipeline = Path(ctx.get("pipeline_dir") or PIPELINE_DIR).expanduser().resolve()
    out = Path(ctx.get("out") or (pipeline / "output")).expanduser()
    lab = Path(ctx.get("lab_dir") or (out / "diar_lab")).expanduser()
    audio = ctx.get("audio_dir")
    audio_root = None
    if audio not in (None, ""):
        audio_root = os.path.realpath(os.fspath(audio))
    return {
        "audio_root": audio_root,
        "lab_dir": lab,
        "runs_dir": lab / RUNS,
        "pipeline_dir": pipeline,
        "python": str(ctx.get("python") or sys.executable),
        "env_name": str(ctx.get("env_name") or DEFAULT_ENV),
        "config_path": Path(ctx.get("config_path") or (pipeline / "config.json")).expanduser(),
    }


# -- responses ----------------------------------------------------------------------------
def _head(h, code, ctype, length, extra=()):
    h.send_response(code)
    h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(length))
    h.send_header("Cache-Control", "no-store")
    for key, value in extra:
        h.send_header(key, value)
    h.end_headers()


def _send(h, code, body, ctype, extra=()):
    _head(h, code, ctype, len(body), extra)
    if body:
        h.wfile.write(body)


def _clean(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _json(h, code, obj):
    body = json.dumps(_clean(obj), ensure_ascii=False, allow_nan=False).encode("utf-8")
    _send(h, code, body, "application/json; charset=utf-8")


def _arg(query, key) -> str:
    value = query.get(key)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return "" if value is None else str(value)


def _int_arg(raw, default, lo, hi):
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise _HTTPError(400, "số nguyên không hợp lệ")
    return max(lo, min(hi, value))


def _body(h):
    try:
        length = int(h.headers.get("Content-Length"))
    except (TypeError, ValueError):
        length = -1
    if length < 0:
        raise _HTTPError(400, "cần Content-Length")
    if length > MAX_BODY:
        h.close_connection = True
        h.rfile.read(min(length, 1 << 20))      # drain so the client does not see a reset
        raise _HTTPError(413, "thân yêu cầu quá lớn (tối đa 64 KB)")
    raw = h.rfile.read(length) if length else b""
    if len(raw) != length:
        raise _HTTPError(400, "thân yêu cầu chưa đủ")
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise _HTTPError(400, "JSON không hợp lệ")
    if not isinstance(data, dict):
        raise _HTTPError(400, "thân yêu cầu phải là object JSON")
    return data


def _now():
    """Local time to the millisecond, so runs started in the same second still sort."""
    now = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _mod(name):
    if name not in _MODS:
        try:
            _MODS[name] = importlib.import_module(name)
        except Exception:
            _MODS[name] = None
    return _MODS[name]


def _plain(text) -> str:
    text = str(text or "").replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _fold(text) -> str:
    return " ".join(_plain(text).casefold().split())


def _inside(root, path) -> bool:
    try:
        return path != root and os.path.commonpath([root, path]) == root
    except ValueError:
        return False


# -- GET /diar and /api/diar/audio --------------------------------------------------------
def _page(h, query, cfg):
    try:
        body = Path(PAGE_PATH).read_bytes()
    except OSError:
        _send(h, 404, f"diar lab page missing: {PAGE_PATH} does not exist\n".encode("utf-8"),
              "text/plain; charset=utf-8")
        return
    _send(h, 200, body, "text/html; charset=utf-8")


def _audio_seconds(path):
    path = Path(path)
    sf = _mod("soundfile")
    if sf is not None:
        try:
            value = float(sf.info(str(path)).duration)
            if math.isfinite(value) and value >= 0:
                return value
        except Exception:
            pass
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                rate = w.getframerate()
                return w.getnframes() / rate if rate else None
        except Exception:
            return None
    return None


def _duration(real):
    try:
        st = os.stat(real)
    except OSError:
        return None
    stamp = (st.st_mtime_ns, st.st_size)
    with _LOCK:
        hit = _DUR_CACHE.get(real)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    secs = _audio_seconds(real)
    with _LOCK:
        _DUR_CACHE[real] = (stamp, secs)
    return secs


def _scan_audio(root):
    now = time.monotonic()
    with _LOCK:
        hit = _LIST_CACHE.get(root)
    if hit is not None and now - hit[0] < LIST_TTL:
        return hit[1]
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        level = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".")) if level < SCAN_LEVELS else []
        for name in sorted(filenames):
            if name.startswith(".") or Path(name).suffix.lower() not in AUDIO_EXT:
                continue
            full = os.path.join(dirpath, name)
            real = os.path.realpath(full)
            if not _inside(root, real) or not os.path.isfile(real):
                continue
            try:
                size = os.stat(real).st_size
            except OSError:
                continue
            rel = "/".join(os.path.relpath(full, root).split(os.sep))
            out.append({"path": rel, "name": name, "size": size, "real": real})
    out.sort(key=lambda e: e["path"])
    with _LOCK:
        _LIST_CACHE[root] = (time.monotonic(), out)
    return out


def _audio_list(h, query, cfg):
    root = cfg["audio_root"]
    if root is None:
        _json(h, 200, {"files": [], "total": 0, "note": "chưa cấu hình audio_dir"})
        return
    if not os.path.isdir(root):
        _json(h, 200, {"files": [], "total": 0, "note": f"thư mục audio không tồn tại: {root}"})
        return
    needle = _fold(_arg(query, "q"))
    limit = _int_arg(_arg(query, "limit"), LIST_DEFAULT, 1, LIST_MAX)
    entries = _scan_audio(root)
    if needle:
        entries = [e for e in entries if needle in _fold(e["path"])]
    files = [{"path": e["path"], "name": e["name"], "size": e["size"], "duration": _duration(e["real"])}
             for e in entries[:limit]]
    _json(h, 200, {"files": files, "total": len(entries)})


# -- GET /api/diar/params, /api/diar/gpus -------------------------------------------------
def _params(h, query, cfg):
    schema = _schema(cfg)
    if not schema["_profile_found"]:
        schema = dict(schema, note=f"không đọc được profile {cfg['env_name']!r} trong {cfg['config_path']}")
    schema.pop("_profile_found", None)
    _json(h, 200, schema)


def _num(text):
    text = text.strip()
    try:
        return int(float(text))
    except ValueError:
        return None


def _gpu_rows():
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        index = _num(parts[0])
        if index is None:
            continue
        rows.append({"index": index, "name": ",".join(parts[1:-3]),
                     "used_mb": _num(parts[-3]), "total_mb": _num(parts[-2]), "util": _num(parts[-1])})
    return rows


def _gpus(h, query, cfg):
    _json(h, 200, {"gpus": _gpu_rows()})


# -- runs: paths, status, queue -----------------------------------------------------------
def _run_dir(cfg, rid):
    if not isinstance(rid, str) or not RUN_ID.fullmatch(rid):
        raise _HTTPError(400, "id không hợp lệ")
    path = cfg["runs_dir"] / rid
    if not path.is_dir():
        raise _HTTPError(404, "không có lần chạy này")
    return path


def _is_live_locked(rid):
    return rid == _CURRENT or rid in _PENDING or any(item[0] == rid for item in _QUEUE)


def _status(run_dir, rid, *, reconcile=True):
    st = _read_json(run_dir / "status.json")
    if not isinstance(st, dict):
        raise _HTTPError(404, "không có lần chạy này")
    if reconcile and st.get("state") in ("queued", "running"):
        with _CV:
            live = _is_live_locked(rid)
        if not live:
            st = dict(st, state="failed", finished=_now(),
                      error="lần chạy không còn tiến trình (có thể server đã khởi động lại)")
            _write_json(run_dir / "status.json", st)
    return st


def _update_status(run_dir, **fields):
    st = _read_json(run_dir / "status.json") or {}
    st.update(fields)
    _write_json(run_dir / "status.json", st)


def _fmt_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return str(value)


def build_cmd(run_dir, params_json, ctx):
    """argv of one diarization run (main.py, stopped after diarization) for the run folder."""
    cfg = _settings(ctx)
    run_dir = Path(run_dir).resolve()
    values = (params_json or {}).get("params") or {}
    cmd = [cfg["python"], "main.py",
           "--audio", str(run_dir / INPUT_NAME),
           "--env", cfg["env_name"],
           "--stop_after", "diarization",
           "--save_path", str(run_dir / "out"),
           "--cache_dir", str(run_dir / "cache"),
           "--job_id", run_dir.name,
           "--no_stage_audio",
           "--gpu_1", "0", "--gpu_2", "0",
           "--config", str(cfg["config_path"])]
    for key, value in values.items():
        cmd += ["--set", f"{key}={_fmt_value(value)}"]
    return cmd


def _load_result(run_dir):
    seg = run_dir / RESULT_DIR / "segments.json"
    try:
        st = os.stat(seg)
    except OSError:
        return None
    key, stamp = str(seg), (st.st_mtime_ns, st.st_size)
    with _LOCK:
        hit = _RESULTS.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    doc = _read_json(seg)
    if doc is None:
        return None
    rows = _rows(doc)
    raw_doc = _read_json(run_dir / RESULT_DIR / "segments_raw.json")
    file_stats = _read_json(run_dir / RESULT_DIR / "stats.json")
    dur = _audio_seconds(run_dir / INPUT_NAME)
    if dur is None:
        dur = max((r["end"] for r in rows), default=0.0)
    stats = dict(file_stats) if isinstance(file_stats, dict) else {}
    stats.update(_segment_stats(rows, dur))
    data = {"rows": rows, "raw": _rows(raw_doc) if raw_doc is not None else None,
            "stats": stats, "duration": dur, "speakers": _speakers(rows)}
    with _LOCK:
        _RESULTS[key] = (stamp, data)
    return data


def _rows(doc):
    out = []
    for item in doc if isinstance(doc, list) else []:
        if not isinstance(item, dict):
            continue
        s, e = item.get("start"), item.get("end")
        if isinstance(s, bool) or isinstance(e, bool) or not isinstance(s, (int, float)) \
                or not isinstance(e, (int, float)) or not (math.isfinite(s) and math.isfinite(e)):
            continue
        if e <= s:
            continue
        out.append({"start": float(s), "end": float(e), "speaker": str(item.get("speaker", "?"))})
    return out


def _speakers(rows):
    seen = []
    for r in sorted(rows, key=lambda r: r["start"]):
        if r["speaker"] not in seen:
            seen.append(r["speaker"])
    return seen


def _union(spans):
    total, start, end = 0.0, None, None
    for s, e in sorted(spans):
        if end is None or s > end:
            if end is not None:
                total += end - start
            start, end = s, e
        else:
            end = max(end, e)
    if end is not None:
        total += end - start
    return total


def _segment_stats(rows, duration):
    durs = [r["end"] - r["start"] for r in rows]
    per = {}
    for r in rows:
        p = per.setdefault(r["speaker"], {"s": 0.0, "n": 0})
        p["s"] += r["end"] - r["start"]
        p["n"] += 1
    ordered = sorted(rows, key=lambda r: r["start"])
    overlap = 0.0
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b["start"] >= a["end"]:
                break
            if b["speaker"] != a["speaker"]:
                overlap += min(a["end"], b["end"]) - b["start"]
    return {
        "n_segments": len(rows),
        "n_speakers": len(per),
        "speech_s": round(_union([(r["start"], r["end"]) for r in rows]), 3),
        "overlap_s": round(overlap, 3),
        "median_segment_s": round(statistics.median(durs), 3) if durs else None,
        "short_lt_0_5s": sum(1 for d in durs if d < 0.5),
        "per_speaker": {k: {"s": round(v["s"], 3), "n": v["n"]} for k, v in per.items()},
        "duration": round(duration, 3) if duration is not None else None,
    }


def _summary(res):
    st = res["stats"]
    return {k: st.get(k) for k in ("n_speakers", "speech_s", "n_segments", "overlap_s", "median_segment_s")}


def _changed(values, defaults):
    return sum(1 for k, v in values.items() if k not in defaults or v != defaults[k])


def _log_tail(path):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - TAIL_BYTES))
            raw = fh.read()
    except OSError:
        return []
    lines = raw.decode("utf-8", errors="replace").split("\n")
    if size > TAIL_BYTES and lines:
        lines = lines[1:]
    out = []
    for line in lines[-TAIL_LINES:]:
        line = ANSI.sub("", line).rstrip("\r")
        if "\r" in line:
            line = line.rsplit("\r", 1)[-1]
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return out


def _run_get(h, query, cfg):
    rid = _arg(query, "id")
    d = _run_dir(cfg, rid)
    st = _status(d, rid)
    doc = _read_json(d / "params.json") or {}
    summary = None
    if st.get("state") == "done":
        res = _load_result(d)
        if res is not None:
            summary = _summary(res)
    _json(h, 200, {
        "id": rid, "label": doc.get("label", ""), "state": st.get("state"),
        "created": st.get("created"), "started": st.get("started"), "finished": st.get("finished"),
        "returncode": st.get("returncode"), "error": st.get("error"), "cmd": st.get("cmd"),
        "audio": doc.get("audio"), "start": doc.get("start"), "end": doc.get("end"),
        "gpu": doc.get("gpu"), "env": doc.get("env"), "params": doc.get("params", {}),
        "log_tail": _log_tail(d / "run.log"), "summary": summary,
    })


def _runs_get(h, query, cfg):
    base = cfg["runs_dir"]
    rows = []
    defaults = _defaults(_schema(cfg))
    try:
        entries = [e for e in os.scandir(base) if e.is_dir() and RUN_ID.fullmatch(e.name)]
    except OSError:
        entries = []
    for entry in entries:
        d = Path(entry.path)
        try:
            st = _status(d, entry.name)
        except _HTTPError:
            continue
        doc = _read_json(d / "params.json") or {}
        values = doc.get("params") or {}
        summary = None
        if st.get("state") == "done":
            res = _load_result(d)
            if res is not None:
                summary = res["stats"]
        rows.append({
            "id": entry.name, "label": doc.get("label", ""), "state": st.get("state"),
            "created": st.get("created"), "audio": doc.get("audio"), "start": doc.get("start"),
            "end": doc.get("end"), "gpu": doc.get("gpu"),
            "n_speakers": summary.get("n_speakers") if summary else None,
            "speech_s": summary.get("speech_s") if summary else None,
            "changed": _changed(values, defaults),
        })
    rows.sort(key=lambda r: (r.get("created") or "", r["id"]), reverse=True)
    _json(h, 200, {"runs": rows})


def _done_result(cfg, rid):
    d = _run_dir(cfg, rid)
    st = _status(d, rid)
    if st.get("state") != "done":
        raise _HTTPError(409, f"lần chạy {rid} chưa xong (state: {st.get('state')})")
    res = _load_result(d)
    if res is None:
        raise _HTTPError(409, f"lần chạy {rid} không có segments.json")
    return d, res


def _result_get(h, query, cfg):
    rid = _arg(query, "id")
    d, res = _done_result(cfg, rid)
    doc = _read_json(d / "params.json") or {}
    raw = res["raw"]
    _json(h, 200, {
        "id": rid,
        "duration": res["duration"],
        "audio_url": f"/api/diar/audio_file?id={rid}",
        "speakers": res["speakers"],
        "segments": [_out_row(r) for r in res["rows"]],
        "raw_segments": [_out_row(r) for r in raw] if raw is not None else None,
        "stats": res["stats"],
        "params": doc.get("params", {}),
    })


def _out_row(r):
    return {"start": round(r["start"], 3), "end": round(r["end"], 3), "speaker": r["speaker"]}


def _audio_file_get(h, query, cfg):
    d = _run_dir(cfg, _arg(query, "id"))
    path = d / INPUT_NAME
    try:
        fh = open(path, "rb")
    except OSError:
        raise _HTTPError(404, "lần chạy này chưa có input.wav")
    with fh:
        size = os.fstat(fh.fileno()).st_size
        raw_range = (h.headers.get("Range") or "").strip()
        if not raw_range:
            start, end, code = 0, size - 1, 200
        else:
            span = _parse_range(raw_range, size)
            if span is None:
                _send(h, 416, b"", "text/plain; charset=utf-8",
                      [("Content-Range", f"bytes */{size}"), ("Accept-Ranges", "bytes")])
                return
            (start, end), code = span, 206
        length = max(0, end - start + 1)
        extra = [("Accept-Ranges", "bytes")]
        if code == 206:
            extra.append(("Content-Range", f"bytes {start}-{end}/{size}"))
        _head(h, code, "audio/wav", length, extra)
        if length:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(CHUNK, left))
                if not chunk:
                    break
                h.wfile.write(chunk)
                left -= len(chunk)


def _parse_range(value, size):
    match = _RANGE.fullmatch(value.strip())
    if match is None:
        return None
    first, last = match.group(1), match.group(2)
    if not first and not last:
        return None
    if not first:
        n = int(last)
        if n == 0 or size == 0:
            return None
        return max(0, size - n), size - 1
    start = int(first)
    if start >= size:
        return None
    if not last:
        return start, size - 1
    end = int(last)
    if end < start:
        return None
    return start, min(end, size - 1)


# -- compare ------------------------------------------------------------------------------
def _quantized(rows):
    spans = []
    for r in rows:
        f0, f1 = int(round(r["start"] / GRID)), int(round(r["end"] / GRID))
        if f1 > f0:
            spans.append((f0, f1, r["speaker"]))
    return spans


def _sweep(spans_a, spans_b):
    """Elementary frame intervals: (length in frames, active labels of a, active labels of b)."""
    events = {}
    for side, spans in ((0, spans_a), (1, spans_b)):
        for f0, f1, label in spans:
            events.setdefault(f0, []).append((side, label, 1))
            events.setdefault(f1, []).append((side, label, -1))
    counts = (Counter(), Counter())
    frames = sorted(events)
    out = []
    for i, f in enumerate(frames[:-1]):
        for side, label, delta in events[f]:
            counts[side][label] += delta
            if counts[side][label] == 0:
                del counts[side][label]
        active_a = frozenset(k for k, v in counts[0].items() if v > 0)
        active_b = frozenset(k for k, v in counts[1].items() if v > 0)
        if active_a or active_b:
            out.append((frames[i + 1] - f, active_a, active_b))
    return out


def _assign(cost, n, m):
    """Min-cost assignment of n rows to distinct columns (n <= m): Kuhn-Munkres with potentials."""
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]


def _best_pairs(rows, cols, weight):
    if not rows or not cols:
        return []
    if len(rows) <= len(cols):
        pairs = _assign([[-weight(r, c) for c in cols] for r in rows], len(rows), len(cols))
        return [(rows[i], cols[j]) for i, j in pairs]
    pairs = _assign([[-weight(r, c) for r in rows] for c in cols], len(cols), len(rows))
    return [(rows[j], cols[i]) for i, j in pairs]


def _mapping(intervals):
    co = {}
    for length, active_a, active_b in intervals:
        for la in active_a:
            row = co.setdefault(la, {})
            for lb in active_b:
                row[lb] = row.get(lb, 0) + length
    rows = sorted(co)
    cols = sorted({lb for row in co.values() for lb in row})

    def weight(la, lb):
        return co.get(la, {}).get(lb, 0)

    return {la: lb for la, lb in _best_pairs(rows, cols, weight) if weight(la, lb) > 0}


def compare_rows(rows_a, rows_b):
    """Compare two segment lists (see the module docstring for the definitions)."""
    intervals = _sweep(_quantized(rows_a), _quantized(rows_b))
    mapping = _mapping(intervals)
    miss = fa = conf = ref = dis = 0
    lanes, truncated, cursor = [], False, 0
    for length, active_a, active_b in intervals:
        f0 = cursor
        cursor += length
        correct = sum(1 for x in active_a if mapping.get(x) in active_b)
        na, nb = len(active_a), len(active_b)
        m = max(0, na - nb) * length
        f = max(0, nb - na) * length
        c = (min(na, nb) - correct) * length
        miss += m
        fa += f
        conf += c
        ref += na * length
        kind = "conf" if c else "miss" if m else "fa" if f else None
        if kind is None:
            continue
        dis += length
        if lanes and lanes[-1]["kind"] == kind and lanes[-1]["_f1"] == f0:
            lanes[-1]["_f1"] = cursor
        elif len(lanes) < LANES_MAX:
            lanes.append({"_f0": f0, "_f1": cursor, "kind": kind})
        else:
            truncated = True
    return {
        "disagreement_s": round(dis * GRID, 3),
        "miss_s": round(miss * GRID, 3),
        "false_alarm_s": round(fa * GRID, 3),
        "confusion_s": round(conf * GRID, 3),
        "ref_speaker_s": round(ref * GRID, 3),
        "der_like": round((miss + fa + conf) / ref, 4) if ref else None,
        "mapping": {str(k): str(v) for k, v in mapping.items()},
        "lanes": [{"start": round(l["_f0"] * GRID, 3), "end": round(l["_f1"] * GRID, 3), "kind": l["kind"]}
                  for l in lanes],
        "lanes_truncated": truncated,
    }


def _compare_get(h, query, cfg):
    a_id, b_id = _arg(query, "a"), _arg(query, "b")
    _, res_a = _done_result(cfg, a_id)
    _, res_b = _done_result(cfg, b_id)
    out = compare_rows(res_a["rows"], res_b["rows"])
    out["a"] = {"id": a_id, "n_speakers": res_a["stats"]["n_speakers"],
                "speech_s": res_a["stats"]["speech_s"],
                "speaker_s": round(sum(r["end"] - r["start"] for r in res_a["rows"]), 3)}
    out["b"] = {"id": b_id, "n_speakers": res_b["stats"]["n_speakers"],
                "speech_s": res_b["stats"]["speech_s"],
                "speaker_s": round(sum(r["end"] - r["start"] for r in res_b["rows"]), 3)}
    _json(h, 200, out)


# -- input cut and validation -------------------------------------------------------------
def _resample(x, src_rate, np):
    if src_rate % SAMPLE_RATE == 0:
        k = src_rate // SAMPLE_RATE
        n = (x.shape[0] // k) * k
        return x[:n].reshape(-1, k).mean(axis=1)
    n_out = int(round(x.shape[0] * SAMPLE_RATE / src_rate))
    t_out = np.arange(n_out) * (src_rate / SAMPLE_RATE)
    return np.interp(t_out, np.arange(x.shape[0]), x).astype("float32")


def _cut(src, dst, start, end):
    """Write [start, end) of src to dst as mono 16 kHz PCM16. Returns the length in seconds."""
    sf, np = _mod("soundfile"), _mod("numpy")
    if sf is not None and np is not None:
        rate = int(sf.info(str(src)).samplerate)
        data, got = sf.read(str(src), start=int(round(start * rate)), stop=int(round(end * rate)),
                            dtype="float32", always_2d=True)
        if data.shape[0] == 0:
            raise _HTTPError(400, "đoạn nằm ngoài file audio")
        mono = data.mean(axis=1)
        if got != SAMPLE_RATE:
            mono = _resample(mono, got, np)
        sf.write(str(dst), mono, SAMPLE_RATE, subtype="PCM_16")
        return mono.shape[0] / SAMPLE_RATE
    if shutil.which("ffmpeg") is None:
        raise _HTTPError(500, "thiếu công cụ cắt audio: cần soundfile+numpy hoặc ffmpeg")
    args = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
            "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise _HTTPError(500, f"ffmpeg không chạy được: {exc}")
    if proc.returncode:
        raise _HTTPError(500, "ffmpeg không cắt được audio: " + proc.stderr.strip()[-300:])
    return _audio_seconds(dst) or 0.0


def _seconds(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise _HTTPError(400, f"{name}: cần số giây không âm")
    return float(value)


def _audio_file(root, rel):
    if root is None:
        raise _HTTPError(400, "chưa cấu hình audio_dir")
    if not isinstance(rel, str) or not rel or len(rel) > 1024 or "\x00" in rel:
        raise _HTTPError(400, "đường dẫn audio không hợp lệ")
    if rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        raise _HTTPError(400, "đường dẫn audio phải tương đối với audio_dir")
    if any(part in ("", ".", "..") for part in rel.split("/")):
        raise _HTTPError(400, "đường dẫn audio không hợp lệ")
    real = os.path.realpath(os.path.join(root, *rel.split("/")))
    if not _inside(root, real):
        raise _HTTPError(400, "đường dẫn nằm ngoài audio_dir")
    if not os.path.isfile(real):
        raise _HTTPError(404, "không có file audio này")
    if Path(real).suffix.lower() not in AUDIO_EXT:
        raise _HTTPError(400, "định dạng audio không được hỗ trợ")
    return real


def _validate_run(body, cfg):
    src = _audio_file(cfg["audio_root"], body.get("audio"))
    rel = body["audio"]
    dur = _audio_seconds(src)
    start = _seconds(body.get("start"), "start")
    end = _seconds(body.get("end"), "end")
    start_v = 0.0 if start is None else start
    if end is None:
        if dur is None:
            raise _HTTPError(400, "không đọc được độ dài file; hãy nhập end")
        end = dur
    if dur is not None and end > dur + 0.01:
        raise _HTTPError(400, f"end vượt quá độ dài file ({dur:.2f} s)")
    if end <= start_v:
        raise _HTTPError(400, "end phải lớn hơn start")
    if end - start_v > MAX_RUN_SECONDS:
        raise _HTTPError(400, f"đoạn tối đa {int(MAX_RUN_SECONDS)} s")
    gpu = body.get("gpu", 0)
    if gpu is None:
        gpu = 0
    if isinstance(gpu, bool) or not isinstance(gpu, int) or not 0 <= gpu <= MAX_GPU:
        raise _HTTPError(400, f"gpu: cần số nguyên 0..{MAX_GPU}")
    label = body.get("label") or ""
    if not isinstance(label, str) or len(label.strip()) > 80:
        raise _HTTPError(400, "label: chuỗi tối đa 80 ký tự")
    values = body.get("params") or {}
    if not isinstance(values, dict):
        raise _HTTPError(400, "params phải là object")
    schema = _schema(cfg)
    index = {item["key"]: item for g in schema["groups"] for item in g["params"]}
    params = {}
    for key, value in values.items():
        item = index.get(key)
        if item is None:
            raise _HTTPError(400, f"tham số không biết: {key}")
        params[key] = _coerce(item, value)
    effective = _defaults(schema)
    effective.update(params)
    _check_combo(effective)
    return {"src": src, "audio": rel, "start": start_v, "end": end, "gpu": gpu,
            "label": label.strip(), "params": params}


def _new_run_dir(runs_dir):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for _ in range(50):
        rid = f"{stamp}_{secrets.token_hex(3)}"
        path = runs_dir / rid
        try:
            path.mkdir(parents=True, exist_ok=False)
            return rid, path
        except FileExistsError:
            continue
    raise _HTTPError(500, "không tạo được thư mục lần chạy")


def _run_post(h, cfg):
    spec = _validate_run(_body(h), cfg)
    cfg["runs_dir"].mkdir(parents=True, exist_ok=True)
    rid, run_dir = _new_run_dir(cfg["runs_dir"])
    with _CV:
        _PENDING.add(rid)
    try:
        seconds = _cut(spec["src"], run_dir / INPUT_NAME, spec["start"], spec["end"])
        _write_json(run_dir / "params.json", {
            "audio": spec["audio"], "start": spec["start"], "end": spec["end"],
            "input_seconds": round(seconds, 3), "params": spec["params"], "gpu": spec["gpu"],
            "label": spec["label"], "env": cfg["env_name"], "created": _now(),
        })
        (run_dir / "run.log").touch()
        _write_json(run_dir / "status.json", {
            "state": "queued", "created": _now(), "started": None, "finished": None,
            "returncode": None, "cmd": None, "error": None,
        })
    except BaseException:
        with _CV:
            _PENDING.discard(rid)
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    with _CV:
        _PENDING.discard(rid)
        _QUEUE.append((rid, run_dir, dict(cfg)))
        _ensure_worker()
        _CV.notify_all()
    _json(h, 201, {"id": rid})


def _stop_post(h, cfg):
    body = _body(h)
    rid = body.get("id")
    run_dir = _run_dir(cfg, rid)
    with _CV:
        queued = next((item for item in _QUEUE if item[0] == rid), None)
        if queued is not None:
            _QUEUE.remove(queued)
        elif rid == _CURRENT:
            _STOPS.add(rid)
            proc = _PROCS.get(rid)
            if proc is not None:
                _signal(proc, signal.SIGTERM)
        else:
            raise _HTTPError(409, "lần chạy này không đang chạy hoặc đang chờ")
    if queued is not None:
        _update_status(run_dir, state="stopped", finished=_now(), error="dừng khi đang chờ")
        _json(h, 200, {"id": rid, "state": "stopped"})
        return
    t0 = time.monotonic()
    while True:
        with _CV:
            if _CURRENT != rid:
                break
            proc = _PROCS.get(rid)
        elapsed = time.monotonic() - t0
        if proc is not None and elapsed > STOP_KILL_AFTER and proc.poll() is None:
            _signal(proc, signal.SIGKILL)
        if elapsed > STOP_WAIT:
            break
        time.sleep(0.1)
    _json(h, 200, {"id": rid, "state": _status(run_dir, rid, reconcile=False).get("state")})


def _delete_post(h, cfg):
    body = _body(h)
    if body.get("confirm") is not True:
        raise _HTTPError(400, "cần confirm: true")
    rid = body.get("id")
    run_dir = _run_dir(cfg, rid)
    with _CV:
        if _is_live_locked(rid):
            raise _HTTPError(409, "lần chạy đang chạy hoặc đang chờ; hãy dừng trước")
    shutil.rmtree(run_dir)
    with _LOCK:
        _RESULTS.pop(str(run_dir / RESULT_DIR / "segments.json"), None)
    _json(h, 200, {"id": rid, "deleted": True})


def _signal(proc, sig):
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


# -- worker -------------------------------------------------------------------------------
def _ensure_worker():
    global _WORKER
    if _WORKER is None or not _WORKER.is_alive():
        _WORKER = threading.Thread(target=_worker_loop, name="diar-lab-worker", daemon=True)
        _WORKER.start()


def _worker_loop():
    global _CURRENT
    while True:
        with _CV:
            while not _QUEUE:
                _CV.wait()
            rid, run_dir, cfg = _QUEUE.popleft()
            _CURRENT = rid
        try:
            _execute(rid, run_dir, cfg)
        except Exception as exc:
            try:
                _update_status(run_dir, state="failed", finished=_now(),
                               error=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
        finally:
            with _CV:
                _CURRENT = None
                _PROCS.pop(rid, None)
                _STOPS.discard(rid)
                _CV.notify_all()


def _execute(rid, run_dir, cfg):
    doc = _read_json(run_dir / "params.json") or {}
    cmd = build_cmd(run_dir, doc, cfg)
    with _CV:
        stopped_before = rid in _STOPS
    if stopped_before:
        _update_status(run_dir, state="stopped", finished=_now(), error="dừng trước khi chạy")
        return
    gpu = doc.get("gpu", 0)
    _update_status(run_dir, state="running", started=_now(), cmd=cmd, error=None)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    with open(run_dir / "run.log", "ab") as log:
        proc = subprocess.Popen(cmd, cwd=str(cfg["pipeline_dir"]), env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with _CV:
            _PROCS[rid] = proc
            if rid in _STOPS:
                _signal(proc, signal.SIGTERM)
        returncode = proc.wait()
    with _CV:
        stopped = rid in _STOPS
    if stopped:
        state, error = "stopped", "dừng theo yêu cầu"
    elif returncode == 0 and (run_dir / RESULT_DIR / "segments.json").is_file():
        state, error = "done", None
    elif returncode == 0:
        state, error = "failed", "main.py kết thúc nhưng không có segments.json"
    else:
        state, error = "failed", f"main.py thoát với mã {returncode}"
    _update_status(run_dir, state=state, finished=_now(), returncode=returncode, error=error)


def _stop_all_for_tests():  # pragma: no cover - used by the test suite only
    with _CV:
        _QUEUE.clear()
        procs = list(_PROCS.items())
    for _, proc in procs:
        _signal(proc, signal.SIGKILL)


_GET_ROUTES = {
    "/diar": _page,
    "/api/diar/audio": _audio_list,
    "/api/diar/params": _params,
    "/api/diar/gpus": _gpus,
    "/api/diar/runs": _runs_get,
    "/api/diar/run": _run_get,
    "/api/diar/result": _result_get,
    "/api/diar/audio_file": _audio_file_get,
    "/api/diar/compare": _compare_get,
}
_POST_ROUTES = {
    "/api/diar/run": _run_post,
    "/api/diar/stop": _stop_post,
    "/api/diar/delete": _delete_post,
}
