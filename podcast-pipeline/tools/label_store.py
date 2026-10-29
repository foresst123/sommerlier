"""Hand labels of the arm-comparison clips: validation, saving with history, listing. Standard library only.

One file per sample, <arms_dir>/<sample>/labels.json:

    {"version": 1, "sample": "01_MOI_TINH_AU", "rev": 7, "updated": "2026-10-29T08:12:03Z",
     "tolerance_s": 0.25,                                  # timing slack the scorer leaves around every label edge
     "speakers": [{"id": "S1", "name": "Chi Pu"}, ...],
     "segments": [{"start": 1.2, "end": 3.4, "speaker": "S1"}, ...],     # may overlap: simultaneous speech
     "ignore":   [{"start": 12.0, "end": 14.5, "note": "not sure"}],     # not scored at all
     "notes": ""}

Every save keeps the previous file under labels_history/ (the last HISTORY_KEEP), so a label can always be taken back.
Saving checks `rev`: a client that read revision 4 cannot silently overwrite revision 5 written from another tab.
"""

from __future__ import annotations

import glob
import json
import os
import re
import tempfile
import time

VERSION = 1
HISTORY_KEEP = 30
MAX_BYTES = 2_000_000
MAX_SEGMENTS = 5000
MAX_SECONDS = 4 * 3600
SAMPLE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
DEFAULT_TOLERANCE_S = 0.25


class Conflict(Exception):
    """The label file moved on since the client read it; .current is the file on disk."""

    def __init__(self, current: dict):
        super().__init__("labels changed since they were read")
        self.current = current


def empty_doc(sample: str) -> dict:
    return {"version": VERSION, "sample": sample, "rev": 0, "updated": None, "tolerance_s": DEFAULT_TOLERANCE_S,
            "speakers": [], "segments": [], "ignore": [], "notes": ""}


def _span(item, kind: str) -> dict:
    try:
        s, e = float(item["start"]), float(item["end"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"a {kind} needs numeric start and end") from None
    if not (0.0 <= s < e <= MAX_SECONDS):
        raise ValueError(f"bad {kind} span {s}-{e}")
    return {"start": round(s, 3), "end": round(e, 3)}


def validate(doc, sample: str) -> dict:
    """The document in canonical form, or ValueError. Speakers used by a segment but not declared are added."""
    if not isinstance(doc, dict):
        raise ValueError("labels must be a JSON object")
    out = empty_doc(sample)
    try:
        tol = float(doc.get("tolerance_s", DEFAULT_TOLERANCE_S))
    except (TypeError, ValueError):
        raise ValueError("tolerance_s must be a number") from None
    if not (0.0 <= tol <= 5.0):
        raise ValueError("tolerance_s must be between 0 and 5 seconds")
    out["tolerance_s"] = round(tol, 3)
    out["notes"] = str(doc.get("notes", ""))[:20000]
    speakers = []
    seen = set()
    for sp in doc.get("speakers") or []:
        if not isinstance(sp, dict) or not str(sp.get("id", "")).strip():
            raise ValueError("a speaker needs an id")
        sid = str(sp["id"]).strip()
        if sid in seen:
            raise ValueError(f"speaker {sid} is declared twice")
        seen.add(sid)
        speakers.append({"id": sid, "name": str(sp.get("name", sid))[:80]})
    segments = doc.get("segments") or []
    if not isinstance(segments, list) or len(segments) > MAX_SEGMENTS:
        raise ValueError(f"segments must be a list of at most {MAX_SEGMENTS}")
    clean = []
    for seg in segments:
        span = _span(seg, "segment")
        sid = str(seg.get("speaker", "")).strip()
        if not sid:
            raise ValueError("a segment needs a speaker")
        if sid not in seen:
            seen.add(sid)
            speakers.append({"id": sid, "name": sid})
        clean.append(dict(span, speaker=sid))
    ignore = doc.get("ignore") or []
    if not isinstance(ignore, list) or len(ignore) > MAX_SEGMENTS:
        raise ValueError("ignore must be a list")
    out["ignore"] = [dict(_span(i, "ignore region"), note=str(i.get("note", ""))[:200]) for i in ignore]
    out["speakers"] = speakers
    out["segments"] = sorted(clean, key=lambda r: (r["start"], r["end"], r["speaker"]))
    return out


def sample_dir(arms_dir: str, sample: str) -> str:
    if not SAMPLE_RE.match(sample or "") or sample in (".", ".."):
        raise ValueError("bad sample name")
    path = os.path.join(arms_dir, sample)
    if not os.path.isdir(path):
        raise FileNotFoundError(sample)
    return path


def load(arms_dir: str, sample: str) -> dict:
    """The saved labels, or an empty document (rev 0) when none were saved yet."""
    path = os.path.join(sample_dir(arms_dir, sample), "labels.json")
    if not os.path.isfile(path):
        return empty_doc(sample)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save(arms_dir: str, sample: str, doc, base_rev=None, force: bool = False, now=None) -> dict:
    """Validate and write the labels; the previous file goes to labels_history/. Raises Conflict on a stale base_rev."""
    folder = sample_dir(arms_dir, sample)
    clean = validate(doc, sample)
    current = load(arms_dir, sample)
    if not force and base_rev is not None and int(base_rev) != int(current.get("rev", 0)):
        raise Conflict(current)
    clean["rev"] = int(current.get("rev", 0)) + 1
    clean["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now if now is not None else time.time()))
    data = json.dumps(clean, ensure_ascii=False, indent=1)
    if len(data.encode("utf-8")) > MAX_BYTES:
        raise ValueError("labels are too large")
    path = os.path.join(folder, "labels.json")
    if os.path.isfile(path):
        hist = os.path.join(folder, "labels_history")
        os.makedirs(hist, exist_ok=True)
        stamp = f"{current.get('rev', 0):06d}"
        os.replace(path, os.path.join(hist, f"labels_{stamp}.json"))
        old = sorted(glob.glob(os.path.join(hist, "labels_*.json")))
        for extra in old[:-HISTORY_KEEP]:
            os.remove(extra)
    fd, tmp = tempfile.mkstemp(prefix=".labels.", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return clean


def list_samples(arms_dir: str) -> list[dict]:
    """Every sample folder with a clip.wav: its arms (segments_<arm>.json) and a summary of its labels."""
    out = []
    for folder in sorted(glob.glob(os.path.join(arms_dir, "*"))):
        name = os.path.basename(folder)
        if not os.path.isdir(folder) or not SAMPLE_RE.match(name) or not os.path.isfile(os.path.join(folder, "clip.wav")):
            continue
        arms = sorted(os.path.basename(f)[len("segments_"):-len(".json")]
                      for f in glob.glob(os.path.join(folder, "segments_*.json")))
        doc = load(arms_dir, name)
        out.append({"name": name, "arms": arms, "rev": doc.get("rev", 0), "updated": doc.get("updated"),
                    "segments": len(doc.get("segments", [])), "speakers": len(doc.get("speakers", []))})
    return out
