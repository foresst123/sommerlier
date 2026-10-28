"""Backend of the conversation player and reviewer: list, detail, audio, labels, sample, stats.

Stdlib only; soundfile is used for durations when it can be imported. ``handle`` (GET) and
``handle_post`` (POST) serve these routes and return False for any other path, so the
dashboard can run its own routes after them:

    GET  /listen                          the page (tools/listen_page.html)
    GET  /api/conversations               filtered, sorted, paged list
    GET  /api/conversation?id=<id>        one conversation's conversation.json and audio URLs
    GET  /audio/<id>/<file>.wav           audio with Range support, confined to the export root
    GET  /api/review?set=                 labels of one review set and their summary
    POST /api/review                      set a label (verdict ok|bad, optional "set") or clear it
    GET  /api/review.csv?set=             one review set as CSV (UTF-8 with BOM)
    GET  /api/review_sets                 every review set, most recently updated first
    POST /api/review_sets                 create {name, from?} or delete {name, delete, confirm?}
    GET  /api/sample?n=&seed=&category=&set=&new=
                                          a seeded sample; with set, saved (new=1) or repeated
    GET  /api/stats                       input hours, output hours and their yield

``ctx`` (optional dict) carries ``audio_dir`` (the original recordings, Path or str),
``total_recordings`` (int) and ``review_dir`` (where review sets live; default
``<final>/../review_sets``, outside the export so that clearing ``final`` keeps the labels). Errors inside a route never escape ``handle``/``handle_post``.
"""
import csv
import io
import json
import math
import os
import pathlib
import random
import re
import tempfile
import threading
import time
import unicodedata
import wave
from datetime import datetime, timezone
from urllib.parse import quote, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_PATH = os.path.join(HERE, "listen_page.html")
CATEGORIES = ("interrupt", "backchannel", "turn_by_turn", "_quarantine")
ROTATION = ("interrupt", "backchannel", "turn_by_turn")   # order the sample cycles through
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
REVIEW_DIR = "_review"                # legacy labels: read only, for the default set
LABELS_NAME = "labels.json"
REVIEW_SETS_DIR = "review_sets"
DEFAULT_SET = "default"
SET_MAX = 40
_SET_CHAR = re.compile(r"[^A-Za-z0-9_-]")
VERDICTS = ("ok", "bad")
MAX_BODY = 64 * 1024
DRAIN_LIMIT = 1 << 20   # a rejected body up to this size is read off the socket before the 413
TAG_MAX = 8
TAG_LEN = 40
NOTE_MAX = 500
SAMPLE_N = 50
SAMPLE_MAX = 500
FILM_UNKNOWN = "(không rõ)"
FILM_SPLITS = (" - ", " | ", "｜", "|", ":", "：", "—", " Tập", " Tap", " Ep", " EP", " Episode", "#")
FILMS_TOP = 40
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3")
AUDIO_TTL = 60.0        # seconds the scan of the original recordings is reused
_RANGE = re.compile(r"bytes\s*=\s*([0-9]*)\s*-\s*([0-9]*)", re.IGNORECASE)
_ID_PREFIX = re.compile(r"^[A-Za-z0-9_-]{11}_")
_FILM_SPLIT = re.compile("|".join(re.escape(s) for s in FILM_SPLITS), re.IGNORECASE)
_EDGE_CHARS = frozenset("‎‏‪‫‬‭‮")

_lock = threading.Lock()
_review_lock = threading.Lock()   # one reader or writer at a time for the review sets
_scan_cache = {}   # str(final) -> (monotonic time, entries)
_meta_cache = {}   # conversation.json path -> ((mtime_ns, size), parsed dict or None)
_audio_scan = {}   # str(audio_dir) -> (monotonic time, index or None)
_dur_cache = {}    # audio file path -> ((mtime_ns, size), seconds or None)
_sf_state = {}     # "mod" -> soundfile module or None (import tried once)


# -- entry points ------------------------------------------------------------------------
def handle(h, path, query, final, ctx=None) -> bool:
    """Serve one GET if ``path`` is one of ours. True when a response was sent."""
    try:
        root = os.path.abspath(os.fspath(final))
        return _route(h, path, query or {}, root, ctx if isinstance(ctx, dict) else {})
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        return True   # the client stopped reading (a seek in the player); nothing to do
    except Exception:
        try:
            _send_json(h, 500, {"error": "internal error"})
        except Exception:
            pass
        return True


def handle_post(h, path, final, ctx=None) -> bool:
    """Serve one POST if ``path`` is one of ours. True when a response was sent."""
    try:
        if path not in ("/api/review", "/api/review_sets"):
            return False
        root = os.path.abspath(os.fspath(final))
        rdir = _review_dir(ctx if isinstance(ctx, dict) else {}, root)
        if path == "/api/review":
            _review_post(h, root, rdir)
        else:
            _sets_post(h, root, rdir)
        return True
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        return True
    except Exception:
        try:
            _send_json(h, 500, {"error": "internal error"})
        except Exception:
            pass
        return True


def _route(h, path, query, final, ctx):
    if path == "/listen":
        return _listen(h)
    if path == "/api/conversations":
        _list(h, query, final)
        return True
    if path == "/api/conversation":
        _detail(h, query, final)
        return True
    if path == "/api/review":
        _review_get(h, query, final, _review_dir(ctx, final))
        return True
    if path == "/api/review.csv":
        _review_csv(h, query, final, _review_dir(ctx, final))
        return True
    if path == "/api/review_sets":
        _sets_get(h, final, _review_dir(ctx, final))
        return True
    if path == "/api/sample":
        _sample(h, query, final, _review_dir(ctx, final))
        return True
    if path == "/api/stats":
        _send_json(h, 200, _stats_payload(final, ctx))
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
def _plain(text) -> str:
    """No Vietnamese tone marks (d-stroke folded to d)."""
    text = str(text or "").replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _fold(text) -> str:
    """Lowercase, no Vietnamese tone marks, single spaces: the form search compares in."""
    return " ".join(_plain(text).casefold().split())


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


def _count(value):
    """An int >= 0 from a context value, or None."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _duration(meta):
    value = meta.get("duration")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _text(value):
    return value if isinstance(value, str) else None


def _skipped(name, top) -> bool:
    """Folder names that never hold a conversation: hidden, lhotse, and at the top any
    underscore folder except the categories (so _review is never a category)."""
    if name.startswith(".") or name == "lhotse":
        return True
    return top and name.startswith("_") and name not in CATEGORIES


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
            if _skipped(name, top=not parts):
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
    for index, part in enumerate(parts):
        if part in ("", ".", "..") or _skipped(part, top=index == 0):
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


# -- /api/review -------------------------------------------------------------------------
def _labels_path(final):
    return os.path.join(final, REVIEW_DIR, LABELS_NAME)


def _read_raw(path):
    """The stored dict: {} when there is no file, None when it exists but is not a readable dict."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _clean_label(value):
    if not isinstance(value, dict) or value.get("verdict") not in VERDICTS:
        return None
    tags = value.get("tags")
    tags = [t for t in tags if isinstance(t, str)][:TAG_MAX] if isinstance(tags, list) else []
    note = value.get("note")
    return {"verdict": value["verdict"], "tags": tags,
            "note": note if isinstance(note, str) else "", "time": value.get("time")}


def _labels_view(raw):
    labels = {}
    for key, value in raw.items():
        clean = _clean_label(value)
        if isinstance(key, str) and clean is not None:
            labels[key] = clean
    return labels


def _summary(labels):
    ok = bad = 0
    by_tag = {}
    for label in labels.values():
        if label["verdict"] == "bad":
            bad += 1
            for tag in set(label["tags"]):
                by_tag[tag] = by_tag.get(tag, 0) + 1
        else:
            ok += 1
    total = ok + bad
    return {"total": total, "ok": ok, "bad": bad,
            "bad_rate": bad / total if total else None, "by_tag": by_tag}


def _review_payload(labels):
    return {"labels": labels, "summary": _summary(labels)}


def _parse_review(data):
    """(id, verdict, tags, note) from a POST body, or (None, error message)."""
    if not isinstance(data, dict):
        return None, "body must be a JSON object"
    rel = data.get("id")
    if not isinstance(rel, str):
        return None, "id must be a string"
    verdict = data.get("verdict")
    if verdict is not None and verdict not in VERDICTS:
        return None, "verdict must be ok, bad or null"
    raw_tags = data.get("tags")
    if raw_tags is None:
        raw_tags = []
    if not isinstance(raw_tags, list) or not all(isinstance(t, str) for t in raw_tags):
        return None, "tags must be a list of strings"
    tags = []
    for raw_tag in raw_tags:
        tag = raw_tag.strip()[:TAG_LEN].rstrip()
        if tag and tag not in tags:
            tags.append(tag)
        if len(tags) == TAG_MAX:
            break
    note = data.get("note")
    if note is None:
        note = ""
    if not isinstance(note, str):
        return None, "note must be a string"
    return (rel, verdict, tags, note[:NOTE_MAX]), None


def _drain(h, length):
    h.close_connection = True
    left = min(length, DRAIN_LIMIT)
    while left > 0:
        chunk = h.rfile.read(min(CHUNK, left))
        if not chunk:
            break
        left -= len(chunk)


def _read_body(h):
    """The request body as bytes, or None after the error response has been sent."""
    try:
        length = int(h.headers.get("Content-Length"))
    except (TypeError, ValueError):
        length = -1
    if length < 0:
        _send_json(h, 400, {"error": "Content-Length required"})
        return None
    if length > MAX_BODY:
        _drain(h, length)
        _send_json(h, 413, {"error": "body too large"})
        return None
    body = h.rfile.read(length) if length else b""
    if len(body) != length:
        _send_json(h, 400, {"error": "body incomplete"})
        return None
    return body


def _write_json_atomic(path, data):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".review-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# -- review sets: one <name>.json per set in the review folder, never inside the export --
CSV_HEADER = ("id", "film", "category", "verdict", "tags", "note", "time", "source_stem", "duration")


def _review_dir(ctx, final):
    raw = ctx.get("review_dir")
    if isinstance(raw, (str, os.PathLike)) and os.fspath(raw) != "":
        return os.path.abspath(os.fspath(raw))
    return os.path.join(os.path.dirname(final), REVIEW_SETS_DIR)


def _set_name(raw):
    """A set name from user input: tone marks dropped, other characters made "_". None when it
    is not a string or is not 1 to SET_MAX characters long afterwards."""
    if not isinstance(raw, str):
        return None
    name = _SET_CHAR.sub("_", _plain(raw.strip()))
    return name if 1 <= len(name) <= SET_MAX else None


def _query_set(query):
    """The set named in a query: DEFAULT_SET when absent or empty, None when invalid."""
    raw = _arg(query, "set")
    return DEFAULT_SET if raw == "" else _set_name(raw)


def _set_path(rdir, name):
    return os.path.join(rdir, name + ".json")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_doc(path):
    """(exists, data): data is None when the file exists but is not a JSON object."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return False, None
    except Exception:
        return True, None
    return True, (data if isinstance(data, dict) else None)


def _sample_view(value):
    if not isinstance(value, dict) or not isinstance(value.get("ids"), list):
        return None
    ids = [v for v in value["ids"] if isinstance(v, str)]
    seed, n, category = value.get("seed"), value.get("n"), value.get("category")
    return {"n": n if _count(n) is not None else len(ids),
            "seed": seed if isinstance(seed, int) and not isinstance(seed, bool) else 0,
            "category": category if isinstance(category, str) and category else None,
            "ids": ids}


def _read_set(rdir, final, name):
    """(known, damaged, doc) for one set. A set without a file is known only as DEFAULT_SET,
    whose labels are then the old _review/labels.json (read, never changed). A damaged file
    gives an empty doc with damaged True."""
    exists, data = _read_doc(_set_path(rdir, name))
    if exists and data is not None:
        labels = data.get("labels")
        return True, False, {"name": name, "created": _text(data.get("created")),
                             "updated": _text(data.get("updated")),
                             "sample": _sample_view(data.get("sample")),
                             "labels": _labels_view(labels if isinstance(labels, dict) else {})}
    doc = {"name": name, "created": None, "updated": None, "sample": None, "labels": {}}
    if exists:
        return True, True, doc
    if name == DEFAULT_SET:
        doc["labels"] = _labels_view(_read_raw(_labels_path(final)) or {})
        return True, False, doc
    return False, False, doc


def _store(rdir, doc, damaged):
    """Write one set atomically. A damaged file is moved aside first, never deleted."""
    path = _set_path(rdir, doc["name"])
    if damaged:
        os.replace(path, f"{path}.bad-{time.time_ns()}")
    now = _now()
    doc["created"] = doc["created"] or now
    doc["updated"] = now
    _write_json_atomic(path, doc)


def _set_row(doc):
    summary = _summary(doc["labels"])
    sample = doc["sample"]
    return {"name": doc["name"], "created": doc["created"], "updated": doc["updated"],
            "labeled": summary["total"], "ok": summary["ok"], "bad": summary["bad"],
            "bad_rate": summary["bad_rate"], "n_sample": len(sample["ids"]) if sample else 0}


def _names_on_disk(rdir):
    names = {DEFAULT_SET}
    try:
        files = os.listdir(rdir)
    except OSError:
        return names
    for fn in files:
        stem = fn[:-len(".json")] if fn.endswith(".json") else ""
        if stem and _set_name(stem) == stem:
            names.add(stem)
    return names


def _review_get(h, query, final, rdir):
    name = _query_set(query)
    if name is None:
        _send_json(h, 400, {"error": "invalid set name"})
        return
    with _review_lock:
        known, _, doc = _read_set(rdir, final, name)
    if not known:
        _send_json(h, 404, {"error": "unknown set"})
        return
    _send_json(h, 200, {"set": name, **_review_payload(doc["labels"])})


def _review_post(h, final, rdir):
    body = _read_body(h)
    if body is None:
        return
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        _send_json(h, 400, {"error": "body is not JSON"})
        return
    fields, error = _parse_review(data)
    if error is not None:
        _send_json(h, 400, {"error": error})
        return
    rel, verdict, tags, note = fields
    name = DEFAULT_SET if data.get("set") is None else _set_name(data.get("set"))
    if name is None:
        _send_json(h, 400, {"error": "set must be a valid set name"})
        return
    if _resolve(final, rel) is None:
        _send_json(h, 404, {"error": "unknown conversation id"})
        return
    with _review_lock:
        known, damaged, doc = _read_set(rdir, final, name)
        if not known:
            code, out = 404, {"error": "unknown set"}
        else:
            # a label never edits the sample: an id outside sample.ids is recorded as it is
            if verdict is None:
                doc["labels"].pop(rel, None)
            else:
                doc["labels"][rel] = {"verdict": verdict, "tags": tags, "note": note,
                                      "time": round(time.time(), 3)}
            _store(rdir, doc, damaged)
            code, out = 200, {"set": name, **_review_payload(doc["labels"])}
    _send_json(h, code, out)


def _sets_get(h, final, rdir):
    with _review_lock:
        rows = [_set_row(_read_set(rdir, final, name)[2]) for name in _names_on_disk(rdir)]
    rows.sort(key=lambda r: (r["updated"] or "", r["name"]), reverse=True)
    _send_json(h, 200, {"sets": rows, "dir": rdir})


def _delete_set(rdir, final, name, confirm):
    if name == DEFAULT_SET:
        return 400, {"error": "the default set cannot be deleted"}
    known, damaged, doc = _read_set(rdir, final, name)
    if not known:
        return 404, {"error": "unknown set"}
    labeled = len(doc["labels"])
    if labeled and not confirm:
        return 409, {"error": f"set has {labeled} labels; send confirm: true to delete it",
                     "labeled": labeled}
    path = _set_path(rdir, name)
    if damaged:   # unreadable: keep it under another name
        os.replace(path, f"{path}.bad-{time.time_ns()}")
    else:
        os.remove(path)
    return 200, {"deleted": name}


def _create_set(rdir, final, name, source):
    if name == DEFAULT_SET or os.path.exists(_set_path(rdir, name)):
        return 409, {"error": "set already exists"}
    sample = None
    if source is not None:
        src = _set_name(source)
        if src is None:
            return 400, {"error": "from must be a valid set name"}
        known, _, src_doc = _read_set(rdir, final, src)
        if not known:
            return 404, {"error": "unknown source set"}
        sample = src_doc["sample"]   # the sample is copied, the labels are not
    now = _now()
    doc = {"name": name, "created": now, "updated": now, "sample": sample, "labels": {}}
    _write_json_atomic(_set_path(rdir, name), doc)
    return 201, _set_row(doc)


def _sets_post(h, final, rdir):
    body = _read_body(h)
    if body is None:
        return
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        _send_json(h, 400, {"error": "body is not JSON"})
        return
    if not isinstance(data, dict):
        _send_json(h, 400, {"error": "body must be a JSON object"})
        return
    name = _set_name(data.get("name"))
    if name is None:
        _send_json(h, 400, {"error": "name must be 1 to 40 characters: letters without tone "
                                     "marks, digits, _ or -"})
        return
    with _review_lock:
        if data.get("delete") is True:
            code, out = _delete_set(rdir, final, name, data.get("confirm") is True)
        else:
            code, out = _create_set(rdir, final, name, data.get("from"))
    _send_json(h, code, out)


def _csv_cell(value):
    """Cell text; a value a spreadsheet would read as a formula gets a quote in front."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _iso_seconds(stamp):
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        return ""
    try:
        return datetime.fromtimestamp(float(stamp), timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return ""


def _review_csv(h, query, final, rdir):
    name = _query_set(query)
    if name is None:
        _send_json(h, 400, {"error": "invalid set name"})
        return
    with _review_lock:
        known, _, doc = _read_set(rdir, final, name)
    if not known:
        _send_json(h, 404, {"error": "unknown set"})
        return
    labels = doc["labels"]
    ids = doc["sample"]["ids"] if doc["sample"] is not None else list(labels)
    entries = {e["id"]: e for e in _scan(final)}
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow(CSV_HEADER)
    for rel in ids:
        entry = entries.get(rel)
        meta = _load_meta(entry["json"], entry["stamp"]) if entry is not None else None
        label = labels.get(rel) or {}
        parts = rel.split("/")
        category = next((p for p in parts if p in CATEGORIES), parts[0])
        src = meta.get("source_stem") if meta is not None else None
        duration = _duration(meta) if meta is not None else None
        row = [rel,
               _film_name(src) if meta is not None else "",
               category,
               label.get("verdict") or "",
               ";".join(label.get("tags") or []),
               label.get("note") or "",
               _iso_seconds(label.get("time")),
               src if isinstance(src, str) else "",
               "" if duration is None else str(duration)]
        writer.writerow([_csv_cell(c) for c in row])
    body = ("﻿" + out.getvalue()).encode("utf-8")
    _head(h, 200, "text/csv; charset=utf-8", len(body),
          [("Content-Disposition", f'attachment; filename="review_{name}.csv"')])
    h.wfile.write(body)


# -- /api/sample -------------------------------------------------------------------------
def _trim(text: str) -> str:
    """Strip whitespace, control characters and the bidi marks at both ends."""
    def edge(ch):
        return ch.isspace() or ch in _EDGE_CHARS or unicodedata.category(ch) == "Cc"
    start, end = 0, len(text)
    while start < end and edge(text[start]):
        start += 1
    while end > start and edge(text[end - 1]):
        end -= 1
    return text[start:end]


def _film_name(stem) -> str:
    """The film a recording belongs to, read from its source_stem (title before the first separator)."""
    text = _trim(stem if isinstance(stem, str) else "")
    text = _ID_PREFIX.sub("", text, count=1)
    found = _FILM_SPLIT.search(text)
    if found:
        text = text[:found.start()]
    return _trim(text) or FILM_UNKNOWN


def _take(state, want):
    """Next (entry, meta) of one film: an unused recording first, a category match if possible."""
    live = [s for s in state["order"] if state["left"][s]]
    fresh = [s for s in live if s not in state["used"]]
    for pool in (fresh, [s for s in live if s in state["used"]]):
        if not pool:
            continue
        for src in pool:
            convs = state["left"][src]
            for i, pair in enumerate(convs):
                if pair[0]["category"] == want:
                    del convs[i]
                    state["used"].add(src)
                    return pair
        src = pool[0]
        state["used"].add(src)
        return state["left"][src].pop(0)
    return None


def _sample_item(entry, meta, labels, film):
    item = _item(entry, meta)
    item["film"] = film
    item["labeled"] = (labels.get(entry["id"]) or {}).get("verdict")
    return item


def _canon_films(final):
    """One display name per film (folded key -> name), the first one met in id order over the
    whole library, so that a sample and its restored copy name a film the same way."""
    names = {}
    for entry in _scan(final):
        if entry["category"] == "_quarantine":
            continue
        meta = _load_meta(entry["json"], entry["stamp"])
        if meta is None:
            continue
        name = _film_name(meta.get("source_stem"))
        names.setdefault(_fold(name), name)
    return names


def _draw(final, n, seed, category, labels):
    """The seeded, film-spread sample of up to n conversations."""
    canon = _canon_films(final)
    films = {}
    cats = set()
    for entry in _scan(final):
        if entry["category"] == "_quarantine" or (category and entry["category"] != category):
            continue
        meta = _load_meta(entry["json"], entry["stamp"])
        if meta is None:
            continue
        cats.add(entry["category"])
        src = meta.get("source_stem") if isinstance(meta.get("source_stem"), str) else ""
        key = _fold(_film_name(src))
        name = canon[key]
        film = films.setdefault(key, {"name": name, "srcs": {}})
        film["srcs"].setdefault(src, []).append((entry, meta))
    if not films:
        return []

    rng = random.Random(seed)
    keys = sorted(films)
    rng.shuffle(keys)
    states = []
    for offset, key in enumerate(keys):
        film = films[key]
        order = sorted(film["srcs"])
        rng.shuffle(order)
        states.append({"name": film["name"], "offset": offset, "turn": 0, "order": order,
                       "left": {s: list(film["srcs"][s]) for s in order}, "used": set()})

    rot = [c for c in ROTATION if c in cats] + sorted(cats - set(ROTATION))
    picks = []
    active = states
    while len(picks) < n and active and rot:
        still = []
        for state in active:
            if len(picks) >= n:
                break
            want = rot[(state["turn"] + state["offset"]) % len(rot)]
            state["turn"] += 1
            pair = _take(state, want)
            if pair is None:
                continue
            picks.append((pair, state["name"]))
            still.append(state)
        active = still
    return [_sample_item(entry, meta, labels, name) for (entry, meta), name in picks]


def _restore(h, final, name, ids, seed, labels, extra=None):
    """The saved sample again, in its saved order; ids no longer in final are counted as missing."""
    found = {e["id"]: e for e in _scan(final) if e["category"] != "_quarantine"}
    canon = _canon_films(final)
    items = []
    missing = 0
    for rel in ids:
        entry = found.get(rel)
        meta = _load_meta(entry["json"], entry["stamp"]) if entry is not None else None
        if meta is None:
            missing += 1
            continue
        film = _film_name(meta.get("source_stem"))
        items.append(_sample_item(entry, meta, labels, canon.get(_fold(film), film)))
    body = {"n": len(items), "films": len({_fold(i["film"]) for i in items}), "seed": seed,
            "items": items, "set": name, "missing": missing}
    body.update(extra or {})
    _send_json(h, 200, body)


def _labeled_order(labels):
    """Labeled ids in the order they were labeled (undated ones last, then by id)."""
    def key(rel):
        stamp = labels[rel].get("time")
        known = isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
        return (0 if known else 1, float(stamp) if known else 0.0, rel)
    return sorted(labels, key=key)


def _sample(h, query, final, rdir):
    n = _int(_arg(query, "n"), SAMPLE_N, 1, SAMPLE_MAX)
    seed = _int(_arg(query, "seed"), 0, -(1 << 63), (1 << 63) - 1)
    category = _arg(query, "category")
    raw_set = _arg(query, "set")
    name = None
    if raw_set != "":
        name = _set_name(raw_set)
        if name is None:
            _send_json(h, 400, {"error": "invalid set name"})
            return
    new = _arg(query, "new") in ("1", "true")
    with _review_lock:
        known, _, doc = _read_set(rdir, final, name or DEFAULT_SET)

    if name is not None and not new:
        if not known:
            _send_json(h, 404, {"error": "unknown set"})
            return
        if doc["sample"] is not None:
            # the saved sample, never a new draw: the same ids in the same order every time
            _restore(h, final, name, doc["sample"]["ids"], doc["sample"]["seed"], doc["labels"])
            return
        if doc["labels"]:
            # a set with labels but no sample (the old labels file): its labeled ids become the
            # sample, saved once so that later reads agree
            ids = _labeled_order(doc["labels"])
            with _review_lock:
                _, damaged, cur = _read_set(rdir, final, name)
                cur["sample"] = {"n": len(ids), "seed": 0, "category": None, "ids": ids}
                _store(rdir, cur, damaged)
            _restore(h, final, name, ids, 0, doc["labels"], {"restored_from_labels": True})
            return
        _send_json(h, 409, {"error": "set has no sample yet; call with new=1 to draw one"})
        return

    items = _draw(final, n, seed, category, doc["labels"])
    body = {"n": len(items), "films": len({_fold(i["film"]) for i in items}), "seed": seed,
            "items": items}
    if name is not None:
        body["set"] = name
        with _review_lock:   # new=1: the drawn ids replace the saved sample; labels stay
            _, damaged, cur = _read_set(rdir, final, name)
            cur["sample"] = {"n": n, "seed": seed, "category": category or None,
                             "ids": [i["id"] for i in items]}
            _store(rdir, cur, damaged)
    _send_json(h, 200, body)


# -- /api/stats --------------------------------------------------------------------------
def _audio_dir(ctx):
    raw = ctx.get("audio_dir")
    if raw is None or raw == "":
        return None
    try:
        return os.path.abspath(os.fspath(raw))
    except TypeError:
        return None


def _soundfile():
    if "mod" not in _sf_state:
        try:
            import soundfile as module
        except Exception:
            module = None
        _sf_state["mod"] = module
    return _sf_state["mod"]


def _read_seconds(path):
    try:
        sf = _soundfile()
        if sf is not None:
            secs = float(sf.info(path).duration)
        elif path.lower().endswith(".wav"):
            with wave.open(path, "rb") as w:
                rate = w.getframerate()
                secs = w.getnframes() / rate if rate else None
        else:
            return None
    except Exception:
        return None
    if secs is None or not math.isfinite(secs) or secs < 0:
        return None
    return secs


def _file_seconds(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    stamp = (st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _dur_cache.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    secs = _read_seconds(path)
    with _lock:
        _dur_cache[path] = (stamp, secs)
    return secs


def _scan_audio(audio_dir):
    """Hours of the audio under audio_dir and per file stem. None when it is not a directory."""
    if not os.path.isdir(audio_dir):
        return None
    total = 0.0
    found = readable = 0
    by_stem = {}
    for dirpath, dirnames, filenames in os.walk(audio_dir):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            stem, ext = os.path.splitext(name)
            if name.startswith(".") or ext.lower() not in AUDIO_EXT:
                continue
            found += 1
            secs = _file_seconds(os.path.join(dirpath, name))
            if secs is None:
                continue
            readable += 1
            total += secs
            by_stem.setdefault(stem, secs / 3600)
    if found and not readable:
        return {"hours": None, "by_stem": {}}
    return {"hours": total / 3600, "by_stem": by_stem}


def _audio_index(audio_dir):
    """_scan_audio, reused for AUDIO_TTL seconds. None when it cannot be read."""
    try:
        now = time.monotonic()
        with _lock:
            hit = _audio_scan.get(audio_dir)
        if hit is not None and now - hit[0] < AUDIO_TTL:
            return hit[1]
        index = _scan_audio(audio_dir)
        with _lock:
            _audio_scan[audio_dir] = (time.monotonic(), index)
        return index
    except Exception:
        return None


def _recording_hours(index, src):
    """Hours of the original recording for ``src``: its stem as is, or with characters outside
    [A-Za-z0-9._-] replaced by underscores. None when no file matches."""
    by_stem = index["by_stem"]
    if src in by_stem:
        return by_stem[src]
    return by_stem.get(re.sub(r"[^A-Za-z0-9._-]", "_", src))


def _stats_payload(final, ctx):
    cands = []
    for entry in _scan(final):
        if entry["category"] == "_quarantine":
            continue
        meta = _load_meta(entry["json"], entry["stamp"])
        if meta is not None:
            cands.append((entry, meta))

    audio_dir = _audio_dir(ctx)
    index = _audio_index(audio_dir) if audio_dir else None

    total_s = 0.0
    by_cat = {}
    films = {}
    srcs = set()
    for entry, meta in cands:
        secs = _duration(meta) or 0.0
        total_s += secs
        bucket = by_cat.setdefault(entry["category"], {"n": 0, "seconds": 0.0})
        bucket["n"] += 1
        bucket["seconds"] += secs
        name = _film_name(meta.get("source_stem"))
        film = films.setdefault(_fold(name), {"film": name, "recordings": set(),
                                              "conversations": 0, "seconds": 0.0})
        film["conversations"] += 1
        film["seconds"] += secs
        src = meta.get("source_stem")
        if isinstance(src, str) and src:
            srcs.add(src)
            film["recordings"].add(src)

    hours_total = index["hours"] if index is not None else None
    rec_hours = None
    if index is not None:
        found = [h for h in (_recording_hours(index, s) for s in srcs) if h is not None]
        rec_hours = sum(found) if found else None

    out_hours = total_s / 3600
    rows = sorted(films.values(), key=lambda f: (-f["seconds"], f["film"]))[:FILMS_TOP]
    return {
        "input": {
            "recordings_total": _count(ctx.get("total_recordings")),
            "recordings_with_conversations": len(srcs),
            "hours_with_conversations": rec_hours,
            "hours_total": hours_total,
        },
        "output": {
            "conversations": len(cands),
            "hours": out_hours,
            "by_category": {c: {"n": b["n"], "hours": b["seconds"] / 3600}
                            for c, b in sorted(by_cat.items())},
            "films": len(films),
        },
        "yield": {
            "hours_ratio": out_hours / hours_total if hours_total else None,
            "conversations_per_recording": len(cands) / len(srcs) if srcs else None,
        },
        "films": [{"film": f["film"], "recordings": len(f["recordings"]),
                   "conversations": f["conversations"], "hours": f["seconds"] / 3600}
                  for f in rows],
    }


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
