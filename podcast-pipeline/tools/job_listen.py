"""Backend of the conversation player and reviewer: list, detail, audio, labels, sample, stats.

Stdlib only; soundfile is used for durations when it can be imported. ``handle`` (GET) and
``handle_post`` (POST) serve these routes and return False for any other path, so the
dashboard can run its own routes after them:

    GET  /listen                          the page (tools/listen_page.html)
    GET  /api/conversations               filtered, sorted, paged list
    GET  /api/conversation?id=<id>        one conversation's conversation.json and audio URLs
    GET  /audio/<id>/<file>.wav           audio with Range support, confined to the export root
    GET  /api/review                      listening labels and their summary
    POST /api/review                      set a label (verdict ok|bad) or clear it (verdict null)
    GET  /api/sample?n=&seed=&category=   a varied, seeded sample of conversations
    GET  /api/stats                       input hours, output hours and their yield

``ctx`` (optional dict) carries ``audio_dir`` (the original recordings, Path or str) and
``total_recordings`` (int). Errors inside a route never escape ``handle``/``handle_post``.
"""
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
REVIEW_DIR = "_review"
LABELS_NAME = "labels.json"
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
_review_lock = threading.Lock()   # one writer at a time for _review/labels.json
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


def handle_post(h, path, final) -> bool:
    """Serve one POST if ``path`` is one of ours. True when a response was sent."""
    try:
        if path != "/api/review":
            return False
        _review_post(h, os.path.abspath(os.fspath(final)))
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
        _send_json(h, 200, _review_view(final))
        return True
    if path == "/api/sample":
        _sample(h, query, final)
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


def _review_payload(raw):
    labels = _labels_view(raw)
    return {"labels": labels, "summary": _summary(labels)}


def _review_view(final):
    with _review_lock:
        raw = _read_raw(_labels_path(final))
    return _review_payload(raw or {})


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
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".labels-", suffix=".tmp")
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


def _review_post(h, final):
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
    if _resolve(final, rel) is None:
        _send_json(h, 404, {"error": "unknown conversation id"})
        return
    path = _labels_path(final)
    with _review_lock:
        raw = _read_raw(path)
        if raw is None:
            # unreadable or damaged: keep it under another name, never delete it
            os.replace(path, f"{path}.bad-{time.time_ns()}")
            raw = {}
        if verdict is None:
            raw.pop(rel, None)
        else:
            raw[rel] = {"verdict": verdict, "tags": tags, "note": note,
                        "time": round(time.time(), 3)}
        _write_json_atomic(path, raw)
        view = _review_payload(raw)
    _send_json(h, 200, view)


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


def _sample(h, query, final):
    n = _int(_arg(query, "n"), SAMPLE_N, 1, SAMPLE_MAX)
    seed = _int(_arg(query, "seed"), 0, -(1 << 63), (1 << 63) - 1)
    category = _arg(query, "category")
    labels = _labels_view(_read_raw(_labels_path(final)) or {})

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
        name = _film_name(src)
        film = films.setdefault(_fold(name), {"name": name, "srcs": {}})
        film["srcs"].setdefault(src, []).append((entry, meta))
    if not films:
        _send_json(h, 200, {"n": 0, "films": 0, "seed": seed, "items": []})
        return

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

    items = []
    sampled_films = set()
    for (entry, meta), name in picks:
        item = _item(entry, meta)
        item["film"] = name
        item["labeled"] = (labels.get(entry["id"]) or {}).get("verdict")
        items.append(item)
        sampled_films.add(_fold(name))
    _send_json(h, 200, {"n": len(items), "films": len(sampled_films), "seed": seed, "items": items})


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
