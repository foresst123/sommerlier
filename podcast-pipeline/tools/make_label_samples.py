"""Pick about 30 clips to hand-label, and write a samples JSON for tools/diar_arms_experiment.py.

Two sources:
  A) export excerpts from the manifest (manifest.jsonl, optionally the quarantine manifest). Round-robin over the
     categories present, at most one clip per source recording, a seeded random window inside each excerpt.
  B) random speech clips from recordings not used in A and not in the existing samples. A seeded random start,
     accepted only when enough 20 ms frames are above -45 dBFS.

Each output entry carries an explicit "audio" path and explicit start/end seconds, so resolve_span() takes the
explicit start and diar_arms_experiment.py does not need --audio-dir. The export's excerpt spans come from the
export's own conversation.json (orig_spans: the span in the ORIGINAL recording, after music cuts), so the window
is cut from the raw recording the same way the rest of the experiment does it.

Standard library plus numpy (already a dependency). ffmpeg/ffprobe are used only by the default probe and decode
functions; tests inject fakes for all three callables (probe, decode, find).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import diar_arms_experiment as X  # noqa: E402

DEFAULT_MANIFEST = "/home/lamkd2/data/data_output/manifest.jsonl"
DEFAULT_OUT = "samples_label30.json"
CLIP_MIN_S, CLIP_MAX_S = 30.0, 90.0
MIN_EXCERPT_S = 30.0          # excerpts shorter than this are skipped
RANDOM_TRIES = 8              # starts tried per random recording
FRAME_S = 0.02                # speech gate frame length
FLOOR_DBFS = -45.0            # a frame above this counts as speech
CATEGORY_ORDER = ("interrupt", "backchannel", "turn_by_turn", "_quarantine")
KIND_OF = {"interrupt": "Interrupt", "backchannel": "Backchannel", "turn_by_turn": "Turn_by_turn",
           "_quarantine": "Quarantine", "random": "Random"}
MANIFEST_NAME = "manifest.jsonl"
QUARANTINE_NAME = "manifest_quarantine.jsonl"
CONVERSATION_FILE = "conversation.json"

# Skip reasons (codes); REASON_VI is what the summary prints.
REASON_VI = {
    "excluded": "đã là mẫu có sẵn (--exclude-samples)",
    "recording-used": "file nguồn đã dùng cho một clip khác",
    "no-category": "dòng manifest thiếu category",
    "no-conversation": "không có conversation.json cạnh file xuất",
    "no-orig-spans": "conversation.json thiếu orig_spans",
    "cut-inside": "đoạn xuất bị cắt nhạc ở giữa (nhiều mảnh)",
    "short-excerpt": "đoạn xuất ngắn hơn 30 s",
    "source-not-found": "không tìm thấy file nguồn",
    "source-ambiguous": "tên file nguồn khớp nhiều file",
    "outside-recording": "cửa sổ nằm ngoài độ dài file",
    "probe-failed": "không đọc được độ dài file (ffprobe)",
    "random-too-short": "file ngắn hơn clip-s",
    "random-no-speech": "không có cửa sổ đủ tiếng nói sau 8 lần thử",
    "decode-failed": "lỗi giải mã audio",
    "duplicate-recording": "trùng tên file nguồn giữa các thư mục",
    "resolve-mismatch": "không qua resolve_span() không đổi",
}


class SourceNotFound(Exception):
    """No audio file under the audio roots has this stem."""


class Skip(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


Probe = Callable[[str], float]
Decode = Callable[[str, float, float], np.ndarray]
Find = Callable[[str], str]


# ---------- helpers -------------------------------------------------------------------------------------------

def rec_key(stem: str) -> str:
    """The recording's identity: the stem without a folder suffix, sanitised, lower case."""
    return X.safe_stem(X.split_folder_suffix(stem)[0]).lower()


def fmt_mmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def rng_for(seed: int, tag: str) -> random.Random:
    """A generator that depends only on (seed, tag): the same seed gives the same draws in every run."""
    digest = hashlib.sha256(f"{seed}:{tag}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def speech_ratio(samples: np.ndarray, sr: int = X.SR) -> float:
    """Share of 20 ms frames whose RMS is above FLOOR_DBFS (full scale = 1.0)."""
    n = int(round(sr * FRAME_S))
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size < n:
        return 0.0
    frames = x[: (x.size // n) * n].reshape(-1, n)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    return float(np.mean(db > FLOOR_DBFS))


def decode_window_ffmpeg(path: str, start: float, dur: float) -> np.ndarray:
    """[start, start + dur) of path as 16 kHz mono float32 in [-1, 1]."""
    proc = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                           "-i", path, "-ac", "1", "-ar", str(X.SR), "-f", "s16le", "-"],
                          capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def find_in_roots(roots: Sequence[str], stem: str) -> str:
    """The one audio file for stem across the audio roots (find_source per root)."""
    hits = []
    for root in roots:
        try:
            hits.append(X.find_source(root, stem))
        except FileNotFoundError:
            continue
    if not hits:
        raise SourceNotFound(stem)
    if len(hits) > 1:
        raise ValueError(f"{stem} matches several files: {hits}")
    return hits[0]


def list_recordings(roots: Sequence[str]) -> Tuple[List[dict], List[str]]:
    """Every audio file under the roots: [{key, stem, path}], plus the stems dropped as duplicates."""
    found: Dict[str, dict] = {}
    dupes: List[str] = []
    for root in roots:
        for cur, dirs, files in os.walk(root):
            dirs.sort()
            for name in sorted(files):
                base, ext = os.path.splitext(name)
                if ext.lower() not in X.AUDIO_EXT:
                    continue
                key = rec_key(base)
                path = os.path.abspath(os.path.join(cur, name))
                if key in found:
                    dupes.append(base)
                    continue
                found[key] = {"key": key, "stem": base, "path": path}
    return [found[k] for k in sorted(found)], dupes


def load_manifest(path: str) -> Tuple[List[dict], int]:
    """Rows of a JSON-lines manifest, and the number of unreadable lines."""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad += 1
    return rows, bad


def load_excluded(paths: Iterable[str]) -> set:
    """Recording keys of the existing samples (by stem, and by the basename of an explicit audio path)."""
    keys = set()
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            samples = json.load(fh)
        if not isinstance(samples, list):
            raise ValueError(f"{path} must hold a JSON list")
        for s in samples:
            stem = s.get("stem") or ""
            if not stem and s.get("audio"):
                stem = os.path.splitext(os.path.basename(str(s["audio"])))[0]
            if stem:
                keys.add(rec_key(str(stem)))
    return keys


def excerpt_span(out_dir: str, row: dict) -> Tuple[float, float]:
    """(start, end) of the excerpt in the original recording, from its conversation.json (orig_spans)."""
    conv = os.path.join(out_dir, os.path.dirname(str(row["path"])), CONVERSATION_FILE)
    if not os.path.isfile(conv):
        raise Skip("no-conversation")
    with open(conv, encoding="utf-8") as fh:
        meta = json.load(fh)
    spans = meta.get("orig_spans") or []
    if not spans:
        raise Skip("no-orig-spans")
    if len(spans) > 1:
        raise Skip("cut-inside")
    return float(spans[0]["start"]), float(spans[0]["end"])


def make_entry(kind: str, source: str, stem: str, audio: str, start: float, end: float, note: str) -> dict:
    return {"name": f"{kind} {stem} {int(start)}s", "kind": kind, "turns": None,
            "duration": fmt_mmss(end - start), "stem": stem, "audio": os.path.abspath(audio),
            "start": round(start, 3), "end": round(end, 3), "note": note, "source": source}


# ---------- A: export excerpts --------------------------------------------------------------------------------

def pick_exports(rows: Sequence[dict], *, out_dir: str, seed: int, n: int, clip_s: float, excluded: set,
                 find: Find, probe: Probe, skips: collections.Counter, missing: List[str]) -> List[dict]:
    queues: Dict[str, List[dict]] = collections.defaultdict(list)
    for row in sorted(rows, key=lambda r: str(r.get("path", ""))):
        queues[str(row.get("category") or "")].append(row)
    for cat, lst in queues.items():
        rng_for(seed, f"order:{cat}").shuffle(lst)
    cats = [c for c in CATEGORY_ORDER if c in queues] + sorted(c for c in queues if c and c not in CATEGORY_ORDER)
    window_rng = rng_for(seed, "export-window")
    used: set = set()
    entries: List[dict] = []
    while len(entries) < n and any(queues.values()):
        for cat in cats:
            if len(entries) >= n:
                break
            queue = queues.get(cat, [])
            while queue:
                row = queue.pop(0)
                entry = _export_candidate(row, cat, out_dir=out_dir, seed=seed, clip_s=clip_s, excluded=excluded,
                                          used=used, find=find, probe=probe, rng=window_rng,
                                          skips=skips, missing=missing)
                if entry is not None:
                    entries.append(entry)
                    break
    return entries


def _export_candidate(row: dict, cat: str, *, out_dir, seed, clip_s, excluded, used, find, probe, rng,
                      skips, missing) -> Optional[dict]:
    if not cat:
        skips["no-category"] += 1
        return None
    stem = str(row.get("source_stem") or "")
    key = rec_key(stem) if stem else ""
    if not key:
        skips["no-category"] += 1
        return None
    if key in excluded:
        skips["excluded"] += 1
        return None
    if key in used:
        skips["recording-used"] += 1
        return None
    try:
        lo, hi = excerpt_span(out_dir, row)
    except Skip as exc:
        skips[exc.reason] += 1
        return None
    length = hi - lo
    if length < MIN_EXCERPT_S:
        skips["short-excerpt"] += 1
        return None
    try:
        src = find(stem)
    except SourceNotFound:
        skips["source-not-found"] += 1
        if stem not in missing:
            missing.append(stem)
        return None
    except ValueError:
        skips["source-ambiguous"] += 1
        return None
    try:
        dur = float(probe(src))
    except Exception:
        skips["probe-failed"] += 1
        return None
    if length <= clip_s:
        start, end, note_tail = lo, hi, "whole excerpt"
    else:
        offset = round(rng.uniform(0.0, length - clip_s), 2)
        start, end = lo + offset, lo + offset + clip_s
        note_tail = f"offset {offset:.2f}s"
    start, end = round(start, 3), round(end, 3)
    if start < 0 or end > dur:
        skips["outside-recording"] += 1
        return None
    used.add(key)
    note = (f"{cat} seed={seed} {note_tail} in excerpt {lo:.2f}-{hi:.2f}s of {stem} "
            f"(recording {dur:.1f}s)")
    return make_entry(KIND_OF.get(cat, cat), "export", stem, src, start, end, note)


# ---------- B: random speech clips -----------------------------------------------------------------------------

def pick_randoms(recordings: Sequence[dict], *, seed: int, n: int, clip_s: float, min_ratio: float,
                 excluded: set, used: set, probe: Probe, decode: Decode, skips: collections.Counter) -> List[dict]:
    pool = [r for r in recordings if r["key"] not in excluded and r["key"] not in used]
    rng_for(seed, "random-order").shuffle(pool)
    entries: List[dict] = []
    for rec in pool:
        if len(entries) >= n:
            break
        try:
            dur = float(probe(rec["path"]))
        except Exception:
            skips["probe-failed"] += 1
            continue
        if dur < clip_s:
            skips["random-too-short"] += 1
            continue
        rng = rng_for(seed, f"random:{rec['key']}")
        accepted = None
        failed = False
        for attempt in range(1, RANDOM_TRIES + 1):
            start = round(rng.uniform(0.0, dur - clip_s), 3)
            try:
                samples = decode(rec["path"], start, clip_s)
            except Exception:
                failed = True
                break
            ratio = speech_ratio(samples)
            if ratio >= min_ratio:
                accepted = (start, ratio, attempt)
                break
        if accepted is None:
            skips["decode-failed" if failed else "random-no-speech"] += 1
            continue
        start, ratio, attempt = accepted
        note = (f"random seed={seed} offset {start:.2f}s speech={ratio:.2f} try {attempt} "
                f"(recording {dur:.1f}s)")
        entries.append(make_entry("Random", "random", rec["stem"], rec["path"], start, start + clip_s, note))
        used.add(rec["key"])
    return entries


# ---------- build and write ------------------------------------------------------------------------------------

def build(args, *, probe: Probe, decode: Decode, find: Find) -> dict:
    """Run both sources. Returns {entries, skips, missing, counts, dupes, bad_lines, manifest_rows}."""
    out_dir = os.path.dirname(os.path.abspath(args.manifest))
    excluded = load_excluded(args.exclude_samples)
    rows, bad = load_manifest(args.manifest)
    if args.include_quarantine:
        qpath = os.path.join(out_dir, QUARANTINE_NAME)
        if os.path.isfile(qpath):
            qrows, qbad = load_manifest(qpath)
            rows += qrows
            bad += qbad
    skips: collections.Counter = collections.Counter()
    missing: List[str] = []
    exports = pick_exports(rows, out_dir=out_dir, seed=args.seed, n=args.n_exports, clip_s=args.clip_s,
                           excluded=excluded, find=find, probe=probe, skips=skips, missing=missing)
    used = {rec_key(e["stem"]) for e in exports}
    recordings, dupes = list_recordings(args.audio_dir)
    if dupes:
        skips["duplicate-recording"] += len(dupes)
    randoms = pick_randoms(recordings, seed=args.seed, n=args.n_random, clip_s=args.clip_s,
                           min_ratio=args.min_speech_ratio, excluded=excluded, used=used, probe=probe,
                           decode=decode, skips=skips)
    entries = []
    for e in exports + randoms:
        try:
            if X.resolve_span(e, None)[:2] != (e["start"], e["end"]):
                raise ValueError("span changed")
        except ValueError:
            skips["resolve-mismatch"] += 1
            continue
        entries.append(e)
    seen = collections.Counter()
    for e in entries:
        seen[e["name"]] += 1
        if seen[e["name"]] > 1:
            e["name"] = f"{e['name']} #{seen[e['name']]}"
    counts = collections.Counter(e["kind"] for e in entries)
    return {"entries": entries, "skips": skips, "missing": missing, "counts": counts, "dupes": dupes,
            "bad_lines": bad, "manifest_rows": len(rows)}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="export manifest.jsonl")
    ap.add_argument("--audio-dir", action="append", default=None,
                    help="folder with the source recordings (repeatable)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="samples JSON to write")
    ap.add_argument("--n-exports", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--clip-s", type=float, default=50.0, help="clip length in seconds (30-90)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-samples", action="append", default=None,
                    help="existing samples JSON whose recordings are left out (repeatable; "
                         "default tools/diar_arms_samples.json)")
    ap.add_argument("--include-quarantine", action="store_true",
                    help="also draw from manifest_quarantine.jsonl (off by default)")
    ap.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    ap.add_argument("--min-speech-ratio", type=float, default=0.5,
                    help="share of 20 ms frames above -45 dBFS a random clip needs")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    return ap


def main(argv: Optional[Sequence[str]] = None, *, probe: Optional[Probe] = None,
         decode: Optional[Decode] = None, find: Optional[Find] = None) -> int:
    ap = _parser()
    a = ap.parse_args(argv)
    if not a.audio_dir:
        ap.error("pass at least one --audio-dir (the recordings for the random clips and the exports)")
    if not os.path.isfile(a.manifest):
        ap.error(f"no manifest at {a.manifest}")
    if not (CLIP_MIN_S <= a.clip_s <= CLIP_MAX_S):
        ap.error(f"--clip-s must be between {CLIP_MIN_S:g} and {CLIP_MAX_S:g}, got {a.clip_s}")
    if not (0.0 <= a.min_speech_ratio <= 1.0):
        ap.error("--min-speech-ratio must be between 0 and 1")
    if a.n_exports < 0 or a.n_random < 0:
        ap.error("--n-exports and --n-random must be 0 or more")
    if a.exclude_samples is None:
        a.exclude_samples = [X.TEMPLATE]
    if not a.dry_run and os.path.exists(a.out) and not a.force:
        print(f"refusing to overwrite {a.out} (pass --force, or --dry-run to look first)", file=sys.stderr)
        return 2
    roots = list(a.audio_dir)
    find_fn = find or (lambda stem: find_in_roots(roots, stem))
    probe_fn = probe or X.probe_duration
    decode_fn = decode or decode_window_ffmpeg
    res = build(a, probe=probe_fn, decode=decode_fn, find=find_fn)
    _print_summary(a, res)
    if a.dry_run:
        print("dry-run: nothing written")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res["entries"], fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"-> {os.path.abspath(a.out)}")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "diar_arms_label30")
    print("Lệnh tiếp theo:")
    print(f"  python tools/diar_arms_experiment.py --samples {os.path.abspath(a.out)} --out {out_dir} "
          f"--only-arms A,B,J")
    return 0


def _print_summary(a, res: dict) -> None:
    counts = res["counts"]
    total = sum(counts.values())
    print(f"Đã chọn {total} mẫu (xuất {a.n_exports}, ngẫu nhiên {a.n_random}, clip {a.clip_s:g} s, seed {a.seed}).")
    for kind in ("Interrupt", "Backchannel", "Turn_by_turn", "Quarantine", "Random"):
        if counts.get(kind):
            print(f"  {kind}: {counts[kind]}")
    print(f"Manifest: {a.manifest} ({res['manifest_rows']} dòng, {res['bad_lines']} dòng lỗi).")
    if res["skips"]:
        print("Bỏ qua:")
        for reason, count in res["skips"].most_common():
            print(f"  {REASON_VI.get(reason, reason)}: {count}")
    if res["missing"]:
        print(f"Không tìm thấy file nguồn ({len(res['missing'])}):")
        for stem in res["missing"]:
            print(f"  {stem}")
    if res["dupes"]:
        print(f"Trùng tên file nguồn đã bỏ qua: {len(res['dupes'])}")


if __name__ == "__main__":
    sys.exit(main())
