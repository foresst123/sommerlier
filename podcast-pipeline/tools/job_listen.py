"""Backend of the conversation player: list and detail as JSON, audio with HTTP Range.

Stdlib only. ``handle`` serves four routes and returns False for any other path,
so the dashboard can run its own routes after it:

    GET /listen                          the page (tools/listen_page.html)
    GET /api/conversations               filtered, sorted, paged list
    GET /api/conversation?id=<id>        one conversation's conversation.json and audio URLs
    GET /audio/<id>/<file>.wav           audio with Range support, confined to the export root

Errors inside a route never escape ``handle``.
"""
import json
import math
import os
import pathlib
import re
import threading
import time
import unicodedata
from urllib.parse import quote, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_PATH = os.path.join(HERE, "listen_page.html")
CATEGORIES = ("interrupt", "backchannel", "turn_by_turn", "_quarantine")
JSON_NAME = "conversation.json"
MAX_DEPTH = 4
SCAN_TTL = 3.0          # seconds a directory scan is reused
CHUNK = 1 << 20
DEFAULT_LIMIT = 200
MAX_LIMIT = 1000
SORTS = ("new", "long", "short", "name")
AUDIO_FILES = {
    "stereo_clean": "stereo_2ch_clean.wav",
    "stereo": "stereo_2ch.wav",
    "assistant": "assistant.wav",
    "user": "user.wav",
    "assistant_clean": "assistant_clean.wav",
    "user_clean": "user_clean.wav",
    "mixture": "mixture.wav",
}
_RANGE = re.compile(r"bytes\s*=\s*([0-9]*)\s*-\s*([0-9]*)", re.IGNORECASE)

_lock = threading.Lock()
_scan_cache = {}   # str(final) -> (monotonic time, entries)
_meta_cache = {}   # conversation.json path -> ((mtime_ns, size), parsed dict or None)


# -- entry point -------------------------------------------------------------------------
def handle(h, path, query, final) -> bool:
    """Serve one GET if ``path`` is one of ours. True when a response was sent."""
    try:
        root = os.path.abspath(os.fspath(final))
        return _route(h, path, query or {}, root)
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        return True   # the client stopped reading (a seek in the player); nothing to do
    except Exception:
        try:
            _send_json(h, 500, {"error": "internal error"})
        except Exception:
            pass
        return True


def _route(h, path, query, final):
    if path == "/listen":
        return _listen(h)
    if path == "/api/conversations":
        _list(h, query, final)
        return True
    if path == "/api/conversation":
        _detail(h, query, final)
        return True
    if path.startswith("/audio/"):
        return _audio(h, path, final)
    return False


# -- responses ---------------------------------------------------------------------------
def _head(h, code, ctype, length, extra=()):
    h.send_response(code)
    h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(length))
    h.send_header("Cache-Control", "no-store")
    for key, value in extra:
        h.send_header(key, value)
    h.end_headers()


def _send(h, code, body: bytes, ctype: str):
    _head(h, code, ctype, len(body))
    if body:
        h.wfile.write(body)


def _send_json(h, code, obj):
    try:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except ValueError:
        body = json.dumps(_finite(obj), ensure_ascii=False).encode("utf-8")
    _send(h, code, body, "application/json; charset=utf-8")


def _finite(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def _not_found(h):
    _send_json(h, 404, {"error": "not found"})
    return True


def _listen(h):
    try:
        with open(PAGE_PATH, "rb") as fh:
            body = fh.read()
    except FileNotFoundError:
        _send(h, 404, f"listen page missing: {PAGE_PATH} does not exist\n".encode("utf-8"),
              "text/plain; charset=utf-8")
        return True
    _send(h, 200, body, "text/html; charset=utf-8")
    return True


# -- helpers -----------------------------------------------------------------------------
def _fold(text) -> str:
    """Lowercase, no Vietnamese tone marks, single spaces: the form search compares in."""
    text = str(text or "").replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.casefold().split())


def _inside(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


def _arg(query, key) -> str:
    value = query.get(key)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return "" if value is None else str(value)


def _int(raw, default, lo, hi):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _duration(meta):
    value = meta.get("duration")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _text(value):
    return value if isinstance(value, str) else None


# -- scanning and reading conversation.json ----------------------------------------------
def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _scan(final: str) -> list:
    now = time.monotonic()
    with _lock:
        hit = _scan_cache.get(final)
    if hit is not None and now - hit[0] < SCAN_TTL:
        return hit[1]
    entries = _walk(final)
    with _lock:
        _scan_cache[final] = (time.monotonic(), entries)
    return entries


def _walk(final: str) -> list:
    real_final = os.path.realpath(final)
    out = []
    stack = [()]
    while stack:
        parts = stack.pop()
        here = os.path.join(final, *parts) if parts else final
        json_path = os.path.join(here, JSON_NAME)
        if parts and os.path.isfile(json_path):
            entry = _entry(real_final, parts, here, json_path)
            if entry is not None:
                out.append(entry)
            continue
        if len(parts) >= MAX_DEPTH:
            continue
        try:
            with os.scandir(here) as it:
                names = [e.name for e in it if _is_subdir(e)]
        except OSError:
            continue
        for name in names:
            if name.startswith(".") or name == "lhotse":
                continue
            stack.append(parts + (name,))
    out.sort(key=lambda e: e["id"])
    return out


def _is_subdir(entry) -> bool:
    try:
        return entry.is_dir(follow_symlinks=False)
    except OSError:
        return False


def _entry(real_final, parts, here, json_path):
    try:
        if not _inside(real_final, os.path.realpath(json_path)):
            return None
        st = os.stat(json_path)
    except OSError:
        return None
    return {
        "id": "/".join(parts),
        "category": next((p for p in parts if p in CATEGORIES), parts[0]),
        "folder": parts[-1],
        "dir": here,
        "json": json_path,
        "stamp": (st.st_mtime_ns, st.st_size),
        "mtime": st.st_mtime,
        "has_clean": os.path.isfile(os.path.join(here, AUDIO_FILES["stereo_clean"])),
    }


def _load_meta(json_path, stamp):
    """Parsed conversation.json, reread only when (mtime_ns, size) changed. None if unreadable."""
    with _lock:
        hit = _meta_cache.get(json_path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        meta = _read_json(json_path)
    except Exception:
        meta = None
    if not isinstance(meta, dict):
        meta = None
    with _lock:
        _meta_cache[json_path] = (stamp, meta)
    return meta


# -- /api/conversations ------------------------------------------------------------------
def _matches(meta, entry, needle: str) -> bool:
    fields = (entry["folder"], meta.get("topic"), meta.get("summary"), meta.get("source_stem"))
    return needle in " | ".join(_fold(f) for f in fields)


def _sort_key(sort):
    if sort == "new":
        return lambda r: (-r[0]["mtime"], r[0]["id"])
    if sort == "name":
        return lambda r: (r[0]["folder"], r[0]["id"])
    if sort == "long":
        return lambda r: (_duration(r[1]) is None, -(_duration(r[1]) or 0.0), r[0]["id"])
    return lambda r: (_duration(r[1]) is None, _duration(r[1]) or 0.0, r[0]["id"])


def _item(entry, meta):
    conv = meta.get("conversation")
    return {
        "id": entry["id"],
        "category": entry["category"],
        "folder": entry["folder"],
        "duration": _duration(meta),
        "topic": _text(meta.get("topic")),
        "summary": _text(meta.get("summary")),
        "turns": len(conv) if isinstance(conv, list) else 0,
        "has_clean": entry["has_clean"],
        "overlap_quality": meta.get("overlap_quality"),
        "source_stem": meta.get("source_stem"),
        "mtime": entry["mtime"],
    }


def _list(h, query, final):
    category = _arg(query, "category")
    needle = _fold(_arg(query, "q"))
    sort = _arg(query, "sort") or "new"
    if sort not in SORTS:
        sort = "new"
    limit = _int(_arg(query, "limit"), DEFAULT_LIMIT, 1, MAX_LIMIT)
    offset = _int(_arg(query, "offset"), 0, 0, None)

    rows = []
    for entry in _scan(final):
        meta = _load_meta(entry["json"], entry["stamp"])
        if meta is None:
            continue
        if needle and not _matches(meta, entry, needle):
            continue
        rows.append((entry, meta))

    counts = {}
    for entry, _ in rows:
        counts[entry["category"]] = counts.get(entry["category"], 0) + 1
    picked = [r for r in rows if not category or r[0]["category"] == category]
    picked.sort(key=_sort_key(sort))
    items = [_item(e, m) for e, m in picked[offset:offset + limit]]
    _send_json(h, 200, {"total": len(picked), "counts": counts, "items": items})


# -- /api/conversation -------------------------------------------------------------------
def _resolve(final: str, rel: str):
    """Directory of the conversation with id ``rel``, or None if it is not one."""
    if not rel or "\x00" in rel or rel.startswith("/"):
        return None
    parts = rel.split("/")
    if len(parts) > MAX_DEPTH:
        return None
    for part in parts:
        if part in ("", ".", "..") or part.startswith(".") or part == "lhotse":
            return None
    here = os.path.join(final, *parts)
    json_path = os.path.join(here, JSON_NAME)
    if not os.path.isfile(json_path):
        return None
    if not _inside(os.path.realpath(final), os.path.realpath(json_path)):
        return None
    return here


def _detail(h, query, final):
    rel = _arg(query, "id")
    here = _resolve(final, rel)
    if here is None:
        _send_json(h, 404, {"error": "unknown conversation id"})
        return
    json_path = os.path.join(here, JSON_NAME)
    try:
        st = os.stat(json_path)
    except OSError:
        _send_json(h, 404, {"error": "unknown conversation id"})
        return
    meta = _load_meta(json_path, (st.st_mtime_ns, st.st_size))
    real_final = os.path.realpath(final)
    files = {}
    for key, name in AUDIO_FILES.items():
        real = os.path.realpath(os.path.join(here, name))
        if os.path.isfile(real) and _inside(real_final, real):
            files[key] = "/audio/" + quote(f"{rel}/{name}", safe="/")
    _send_json(h, 200, {"id": rel, "meta": meta, "files": files})


# -- /audio/<id>/<file>.wav --------------------------------------------------------------
def _parse_range(value: str, size: int):
    """(start, end) inclusive for one bytes range, or None if it is malformed or out of reach."""
    match = _RANGE.fullmatch(value.strip())
    if match is None:
        return None
    first, last = match.group(1), match.group(2)
    if not first and not last:
        return None
    if not first:                       # bytes=-n: the last n bytes
        n = int(last)
        if n == 0 or size == 0:
            return None
        return max(0, size - n), size - 1
    start = int(first)
    if start >= size:
        return None
    if not last:                        # bytes=a-
        return start, size - 1
    end = int(last)
    if end < start:
        return None
    return start, min(end, size - 1)


def _audio(h, path, final):
    decoded = unquote(path[len("/audio/"):])
    parts = decoded.split("/")
    if "\x00" in decoded or len(parts) < 2:
        return _not_found(h)
    *ids, name = parts
    if any(p in ("", ".", "..") for p in ids) or name in ("", ".", "..") or not name.endswith(".wav"):
        return _not_found(h)
    real_final = os.path.realpath(final)
    real = os.path.realpath(os.path.join(final, *ids, name))
    if real == real_final or not _inside(real_final, real) or not os.path.isfile(real):
        return _not_found(h)
    try:
        fh = open(real, "rb")
    except OSError:
        return _not_found(h)
    with fh:
        size = os.fstat(fh.fileno()).st_size
        raw_range = (h.headers.get("Range") or "").strip()
        if not raw_range:
            start, end, code = 0, size - 1, 200
        else:
            span = _parse_range(raw_range, size)
            if span is None:
                _head(h, 416, "text/plain; charset=utf-8", 0,
                      [("Content-Range", f"bytes */{size}"), ("Accept-Ranges", "bytes")])
                return True
            (start, end), code = span, 206
        length = end - start + 1
        extra = [("Accept-Ranges", "bytes")]
        if code == 206:
            extra.append(("Content-Range", f"bytes {start}-{end}/{size}"))
        _head(h, code, "audio/wav", length, extra)
        if length > 0:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(CHUNK, left))
                if not chunk:
                    break
                h.wfile.write(chunk)
                left -= len(chunk)
    return True
