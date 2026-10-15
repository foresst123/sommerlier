"""Convert conversation exports into a SpeechWorld full-duplex Lhotse manifest (spec v2.2.0).

One export item folder (conversation.json + stereo_2ch.wav) becomes one MultiCut dict;
a manifest is one JSON line per cut. Everything here is plain dicts: lhotse is not
imported. Channel 0 is the user, channel 1 the assistant.

Decisions made where the spec is silent:
  * only `strict_separation_tracks` items without failed separation spans are used;
  * a segment that has text but no word timing ("untimed_segment") rejects the item,
    because alignment is needed for the containment and coverage checks;
  * a real tag in a turn ([Nhac], <...>) rejects the item; lone brackets are stripped;
  * an untimed backchannel candidate is dropped (noted), it does not reject the item;
  * a merged_backchannel phrase is cut out of its host segment (text and words);
  * speaker_name is the diarization speaker id; age and dislect are null.
"""
from __future__ import annotations

import base64
import dataclasses
import gzip
import hashlib
import importlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

CONVERSATION_FILE = "conversation.json"
QUARANTINE_DIR = "_quarantine"
LHOTSE_DIR = "lhotse"             # per-recording manifests + swapped audio live under <root>/lhotse
_SKIP_DIRS = (QUARANTINE_DIR, LHOTSE_DIR)
STRICT_TRACKS = "strict_separation_tracks"
DUPLEX_FILE = "stereo_duplex.wav"
CHANNEL_OF_ROLE = {"user": 0, "assistant": 1}
ROUND = 6
MAX_BACKCHANNEL_ROUNDS = 3
SAME_CHANNEL_TRIM = 0.05          # same-channel overlap up to this is trimmed, more rejects

_TAG = re.compile(r"\[[^\]]+\]|<[^>]+>")
_STRIP = re.compile(r"[<>\[\]]")


@dataclass
class LhotseManifestParams:
    language: str = "vi"
    emotion_min_conf: float = 0.6
    coverage_check: bool = True
    coverage_max_uncovered: float = 0.30
    # What counts as "speech" on a channel for that check: raw energy over this RMS for at least
    # this many 20 ms frames. Breaths, laughs, noise and separation bleed pass that test too, so
    # a stricter pair keeps real speech and stops them from sinking a whole conversation.
    coverage_rms_threshold: float = 0.005
    coverage_min_speech_frames: int = 3
    sampling_rate_check: bool = True
    # Same-speaker pieces further apart than this stay separate turns; a longer
    # silence inside one turn would read as that person talking over the other.
    max_turn_silence: float = 1.5
    # Where channel-swapped copies of stereo_2ch.wav are written. None puts them next
    # to the item; build_manifest defaults it to <manifest folder>/audio so the export
    # tree stays read-only.
    audio_dir: Optional[str] = None


@dataclass
class ItemResult:
    cut: Optional[dict]
    reasons: List[str]
    item_dir: str


@dataclass
class ManifestReport:
    written: int
    skipped: List[dict]
    notes: List[dict]
    output_path: str


class _Reject(Exception):
    pass


def _r(x: float) -> float:
    return round(float(x), ROUND)


def short_uuid(value: Optional[uuid.UUID] = None) -> str:
    value = value or uuid.uuid4()
    return base64.urlsafe_b64encode(value.bytes).rstrip(b"=").decode("ascii")


def _duplex():
    return importlib.import_module("utils.lhotse_duplex")


def _clean(text: str) -> str:
    return " ".join(_STRIP.sub("", text or "").split())


# -- words --------------------------------------------------------------------------

def _clean_words(words: Sequence[dict]) -> List[list]:
    """[[word, start, end, score], ...] with tag characters removed; raises on missing times."""
    out = []
    for w in words:
        if w.get("start") is None or w.get("end") is None:
            raise _Reject("untimed")
        token = _STRIP.sub("", str(w.get("word", ""))).strip()
        if not token:
            continue
        s, e = float(w["start"]), float(w["end"])
        out.append([token, s, max(e, s), w.get("score")])
    out.sort(key=lambda x: x[1])
    return out


def _alignment(words: Sequence[list], lo: float, hi: float) -> List[list]:
    """Alignment items [word, start, duration, score] inside [lo, hi], increasing, no overlap."""
    items, prev_end = [], lo
    for token, s, e, score in words:
        s = min(max(s, prev_end, lo), hi)
        e = min(max(e, s), hi)
        items.append([token, _r(s), _r(e - s), score])
        prev_end = _r(s) + _r(e - s)
    return items


# -- segments -----------------------------------------------------------------------

def _segment(row: dict, words: List[dict], text: str, emotion: Optional[dict], stem: str,
             key: str) -> dict:
    return {"key": key, "role": row["role"], "channel": CHANNEL_OF_ROLE[row["role"]],
            "speaker_id": str(row.get("speaker_id", row["role"])),
            "speaker": f"{stem}_{row.get('speaker_id', row['role'])}",
            "start": float(row["start"]), "end": float(row["end"]),
            "raw_text": row.get("text") or "", "text": _clean(text), "words": words,
            "emotion": emotion}


def _emotion_label(row: dict, min_conf: float) -> Optional[str]:
    emo = row.get("emotion")
    if isinstance(emo, dict) and emo.get("label") is not None:
        try:
            if float(emo.get("conf", 0.0)) >= min_conf:
                return str(emo["label"])
        except (TypeError, ValueError):
            return None
    return None


def _without_phrases(text: str, words: Sequence[dict], phrases: Sequence[Optional[str]]) -> str:
    """Host text minus the listener phrase; falls back to the remaining words."""
    out = text
    for phrase in phrases:
        if not phrase:
            continue
        cut = re.sub(re.escape(str(phrase)), " ", out, count=1, flags=re.I)
        if cut == out:
            return " ".join(str(w.get("word", "")) for w in words)
        out = cut
    return out


def _split_rows(meta: dict, params: LhotseManifestParams):
    """(segments, candidate segments, reasons). Raises _Reject for item-level problems."""
    stem = str(meta.get("source_stem") or "")
    rows = list(meta.get("conversation") or [])
    reasons: List[str] = []
    events = meta.get("events") or []
    ev_bc = {str(e.get("segment")) for e in events if e.get("type") == "backchannel"}
    merged_events = [e for e in events if e.get("type") == "merged_backchannel"]
    listener_id = {}
    for row in rows:
        listener_id.setdefault(row.get("role"), row.get("speaker_id"))

    by_index = {str(r.get("index")): r for r in rows}
    cand_segs: Dict[str, dict] = {}
    cand_source: Dict[str, str] = {}
    merged_cuts: List[Tuple[dict, dict, str]] = []     # (event, host row, key)

    for r in rows:
        if r.get("role") not in CHANNEL_OF_ROLE:
            raise _Reject(f"unknown_role:seg{r.get('index')}")
        if _TAG.search(r.get("text") or ""):
            raise _Reject(f"tag_in_text:seg{r.get('index')}")
    for n, ev in enumerate(merged_events):
        host = by_index.get(str(ev.get("segment")))
        if host is not None and ev.get("by") in CHANNEL_OF_ROLE:
            merged_cuts.append((ev, host, f"merged{ev.get('segment')}_{n}"))

    # phrase words moved out of their host segment
    moved: Dict[str, List[dict]] = {}
    for ev, host, key in merged_cuts:
        lo, hi = float(ev["start"]) - 0.02, float(ev["end"]) + 0.02
        taken = [w for w in host.get("words") or []
                 if w.get("start") is not None and w.get("end") is not None
                 and lo <= (float(w["start"]) + float(w["end"])) / 2 <= hi]
        if not taken:
            continue
        moved.setdefault(str(host["index"]), []).extend(taken)
        phrase_row = {"role": ev["by"], "speaker_id": listener_id.get(ev["by"], ev["by"]),
                      "start": ev["start"], "end": ev["end"], "text": ev.get("text") or ""}
        seg = _segment(phrase_row, taken, phrase_row["text"], None, stem, key)
        cand_segs[key] = seg
        cand_source[key] = "merged_backchannel"

    segments: List[dict] = []
    for r in rows:
        idx = str(r.get("index"))
        key = f"seg{idx}"
        words = list(r.get("words") or [])
        text = r.get("text") or ""
        if idx in moved:
            gone = {id(w) for w in moved[idx]}
            words = [w for w in words if id(w) not in gone]
            text = _without_phrases(text, words, [ev.get("text") for ev, host, _ in merged_cuts
                                                  if host is r])
            if not words:
                continue
        label = _emotion_label(r, params.emotion_min_conf)
        seg = _segment(r, words, text, {"label": label} if label else None, stem, key)
        if r.get("state") == "BACKCHANNEL" or idx in ev_bc:
            cand_segs[key] = seg
            cand_source[key] = "event" if idx in ev_bc else "state"
        else:
            segments.append(seg)
    for key, seg in cand_segs.items():
        seg["source"] = cand_source[key]
    return segments, cand_segs, reasons


def _prepare_segment(seg: dict) -> bool:
    """Clean words in place. False when it has no speech text (skip). Raises on untimed."""
    if not seg["text"]:
        return False
    if not seg["words"]:
        raise _Reject(f"untimed_segment:{seg['key']}")
    try:
        seg["words"] = _clean_words(seg["words"])
    except _Reject:
        raise _Reject(f"untimed_segment:{seg['key']}")
    if not seg["words"]:
        raise _Reject(f"untimed_segment:{seg['key']}")
    return True


def _merge_adjacent(segs: Sequence[dict], max_silence: float = float("inf")) -> List[dict]:
    out: List[dict] = []
    for s in sorted(segs, key=lambda s: (s["start"], s["end"])):
        p = out[-1] if out else None
        if (p and p["channel"] == s["channel"] and p["speaker_id"] == s["speaker_id"]
                and s["words"][0][1] - p["words"][-1][2] <= max_silence):
            p["end"] = max(p["end"], s["end"])
            p["text"] = f"{p['text']} {s['text']}".strip()
            p["raw_text"] = f"{p['raw_text']} {s['raw_text']}".strip()
            p["words"] = p["words"] + s["words"]
            p["emotion"] = p["emotion"] or s["emotion"]
        else:
            out.append({**s, "words": list(s["words"])})
    return out


def _build_turns(segs: Sequence[dict], rec_id: str, rec_duration: float,
                 max_silence: float = float("inf")) -> List[dict]:
    turns = _merge_adjacent(segs, max_silence)
    for t in turns:
        words = [list(w) for w in t["words"]]
        for w in words:
            w[1] = min(max(w[1], 0.0), rec_duration)
            w[2] = min(max(w[2], w[1]), rec_duration)
        t["words"] = words
        t["start"] = min(max(0.0, min(t["start"], words[0][1])), rec_duration)
    turns.sort(key=lambda t: (t["start"], t["end"]))
    # same-channel neighbours must not overlap
    for ch in (0, 1):
        own = [t for t in turns if t["channel"] == ch]
        for a, b in zip(own, own[1:]):
            over = max(w[2] for w in a["words"]) - b["start"]
            if over > SAME_CHANNEL_TRIM:
                raise _Reject(f"same_channel_overlap:{a['key']}")
            if over > 0:
                for w in a["words"]:
                    w[2] = min(w[2], b["start"])
                    w[1] = min(w[1], w[2])
    pos = {0: 0, 1: 0}
    for i, t in enumerate(turns):
        t["speech_start"] = t["words"][0][1]
        t["speech_end"] = max(w[2] for w in t["words"])
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        if nxt is None:
            end = rec_duration
        else:
            end = nxt["start"]
        if nxt is None or nxt["channel"] != t["channel"]:
            end = max(end, t["speech_end"])
        t["duration"] = max(end - t["start"], 0.0)
        t["position"] = pos[t["channel"]]
        pos[t["channel"]] += 1
        t["id"] = f"{rec_id}-turn_{t['channel']}_{t['position']}"
    return turns


def _turn_view(t: dict) -> dict:
    return {"id": t["id"], "channel": t["channel"], "start": t["start"],
            "duration": t["duration"], "speech_start": t["speech_start"],
            "speech_end": t["speech_end"], "words": [list(w[:1]) + [w[1], w[2] - w[1], w[3]]
                                                      for w in t["words"]]}


def _cand_view(seg: dict) -> dict:
    ws = [list(w[:1]) + [w[1], w[2] - w[1], w[3]] for w in seg["words"]]
    return {"key": seg["key"], "channel": seg["channel"], "role": seg["role"],
            "speaker": seg["speaker"], "start": seg["words"][0][1],
            "end": max(w[2] for w in seg["words"]), "text": seg["text"], "words": ws,
            "source": seg.get("source", "event")}


# -- audio --------------------------------------------------------------------------

def _prepare_wav(item_dir: str, meta: dict, params: LhotseManifestParams) -> Tuple[str, sf.SoundFile]:
    name = (meta.get("files") or {}).get("stereo_2ch") or "stereo_2ch.wav"
    path = os.path.abspath(os.path.join(item_dir, name))
    if not os.path.isfile(path):
        raise _Reject("missing_wav")
    info = sf.info(path)
    if info.channels != 2:
        raise _Reject(f"wav_channels:{info.channels}")
    if params.sampling_rate_check and meta.get("sample_rate") and \
            int(meta["sample_rate"]) != int(info.samplerate):
        raise _Reject(f"sampling_rate_mismatch:{info.samplerate}")
    left = (meta.get("channels") or {}).get("left")
    if left == "assistant":
        if params.audio_dir:
            tag = hashlib.sha1(os.path.abspath(item_dir).encode("utf-8")).hexdigest()[:8]
            out = os.path.abspath(os.path.join(
                params.audio_dir, f"{os.path.basename(os.path.normpath(item_dir))}_{tag}.wav"))
            os.makedirs(os.path.dirname(out), exist_ok=True)
        else:
            out = os.path.abspath(os.path.join(item_dir, DUPLEX_FILE))
        if not os.path.isfile(out) or os.path.getmtime(out) < os.path.getmtime(path):
            data, sr = sf.read(path, dtype="int16", always_2d=True)
            sf.write(out, data[:, ::-1], sr, subtype="PCM_16")
        path = out
        info = sf.info(path)
    elif left != "user":
        raise _Reject("unknown_channel_mapping")
    return path, info


# -- convert ------------------------------------------------------------------------

def _supervision_base(rec_id: str, sup_id: str, start: float, duration: float, channel: int,
                      text: str, params: LhotseManifestParams, speaker: str,
                      alignment: List[list], custom: dict) -> dict:
    return {"id": sup_id, "recording_id": rec_id, "start": _r(start), "duration": _r(duration),
            "channel": channel, "text": text, "language": params.language, "speaker": speaker,
            "alignment": {"word": alignment}, "custom": custom}


def convert_item(item_dir: str, batch_id: str, topic_id: str, params: LhotseManifestParams,
                 index: int = 0, uuid_factory: Optional[Callable[[], uuid.UUID]] = None
                 ) -> ItemResult:
    reasons: List[str] = []
    try:
        cut = _convert(item_dir, batch_id, topic_id, params, index, reasons, uuid_factory)
        return ItemResult(cut, reasons, item_dir)
    except _Reject as exc:
        return ItemResult(None, [str(exc)] + reasons, item_dir)


def _convert(item_dir, batch_id, topic_id, params, index, reasons, uuid_factory) -> dict:
    path = os.path.join(item_dir, CONVERSATION_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError) as exc:
        raise _Reject(f"unreadable_conversation_json:{exc}")
    if meta.get("category") == QUARANTINE_DIR or \
            (meta.get("quality") or {}).get("quarantine_reason") is not None:
        raise _Reject("quarantine")
    channels = meta.get("channels") or {}
    if channels.get("method") != STRICT_TRACKS:
        raise _Reject(f"method:{channels.get('method')}")
    if meta.get("failed_separation_spans"):
        raise _Reject(f"failed_separation_spans:{len(meta['failed_separation_spans'])}")

    wav, info = _prepare_wav(item_dir, meta, params)
    rec_duration = info.frames / float(info.samplerate)
    conv_id = os.path.splitext(os.path.basename(
        (meta.get("files") or {}).get("stereo_2ch") or "stereo_2ch.wav"))[0]
    uid = short_uuid(uuid_factory() if uuid_factory else None)
    rec_id = f"{batch_id}-{topic_id}-{conv_id}-{uid}"

    segments, cand_segs, _ = _split_rows(meta, params)
    segments = [s for s in segments if _prepare_segment(s)]
    cands: Dict[str, dict] = {}
    for key, seg in cand_segs.items():
        try:
            if _prepare_segment(seg):
                cands[key] = seg
        except _Reject:
            reasons.append(f"backchannel_dropped:{key}:untimed_segment")
    if not segments:
        raise _Reject("no_turns")

    dx = _duplex()
    live = dict(cands)
    attached: List[dict] = []
    for round_no in range(MAX_BACKCHANNEL_ROUNDS + 1):
        turns = _build_turns(segments, rec_id, rec_duration, params.max_turn_silence)
        res = dx.resolve_backchannels([_turn_view(t) for t in turns],
                                      [_cand_view(c) for c in live.values()])
        attached = res.attached
        for d in res.dropped:
            reasons.append(f"backchannel_dropped:{d['candidate']['key']}:{d['reason']}")
        for key in {d["candidate"]["key"] for d in res.dropped}:
            live.pop(key, None)
        if not res.demoted:
            break
        if round_no == MAX_BACKCHANNEL_ROUNDS:
            for d in res.demoted:
                reasons.append(f"backchannel_dropped:{d['key']}:demotion_did_not_settle")
            break
        for d in res.demoted:
            reasons.append(f"backchannel_demoted:{d['key']}")
            segments.append(live.pop(d["key"]))
    else:                                                 # pragma: no cover
        raise _Reject("backchannel_loop")

    bc_ids = dx.anchor_backchannel_ids(attached)
    by_turn = {t["id"]: t for t in turns}
    sups: List[dict] = []
    prev_end = None
    for t in turns:
        start, dur = t["start"], t["duration"]
        ali = _alignment(t["words"], start, start + dur)
        if prev_end is None:
            trans = {"type": "gap", "duration": 0.0}
        elif start < prev_end:
            trans = {"type": "overlap", "duration": _r(prev_end - start)}
        else:
            trans = {"type": "gap", "duration": _r(start - prev_end)}
        prev_end = start + dur
        if not ali:
            continue
        custom = {"role": t["role"], "event_type": "turn", "raw_text": t["raw_text"]}
        desc = meta.get("summary") or meta.get("topic")
        if desc:
            custom["conversation_description"] = desc
        custom.update({"speaker_name": t["speaker_id"], "age": None, "dislect": None,
                       "assistance_voice": t["role"] == "assistant",
                       "turn_channel_position": t["position"], "emotion":
                       (t["emotion"] or {}).get("label"), "transition": trans,
                       "backchannel_ids": list(bc_ids.get(t["id"], []))})
        sups.append(_supervision_base(rec_id, t["id"], start, dur, t["channel"], t["text"],
                                      params, t["speaker"], ali, custom))

    spans: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}
    for t in turns:
        spans[t["channel"]].append((t["start"], t["start"] + t["duration"]))
    for a in attached:
        anchor = by_turn[a["anchor_id"]]
        seg = live[a["key"]]
        lo = max(seg["words"][0][1], anchor["start"])
        hi = min(max(w[2] for w in seg["words"]), anchor["start"] + anchor["duration"])
        ch = 1 - anchor["channel"]
        ali = _alignment(seg["words"], lo, hi)
        if not seg["text"] or not ali:
            reasons.append(f"backchannel_dropped:{a['key']}:empty_text")
            continue
        custom = {"role": seg["role"], "event_type": "backchannel", "anchor_id": a["anchor_id"],
                  "anchor_word_index": a["anchor_word_index"], "speaker_name": seg["speaker_id"],
                  "age": None, "dislect": None, "assistance_voice": seg["role"] == "assistant"}
        sups.append(_supervision_base(rec_id, a["id"], lo, hi - lo, ch, seg["text"], params,
                                      seg["speaker"], ali, custom))
        spans[ch].append((lo, hi))
    kept_bc = {s["id"] for s in sups}
    for s in sups:
        if s["custom"]["event_type"] == "turn":
            s["custom"]["backchannel_ids"] = [i for i in s["custom"]["backchannel_ids"]
                                              if i in kept_bc]
    if not any(s["custom"]["event_type"] == "turn" for s in sups):
        raise _Reject("no_turns")

    if params.coverage_check:
        cov = dx.check_item_coverage(
            wav, spans, dx.CoverageParams(
                max_uncovered_seconds=params.coverage_max_uncovered,
                rms_threshold=params.coverage_rms_threshold,
                min_speech_frames=params.coverage_min_speech_frames))
        if cov:
            raise _Reject("coverage:" + ";".join(
                f"ch{v['channel']} {v['start']:.2f}-{v['end']:.2f}" for v in cov))

    recording = {"id": rec_id,
                 "sources": [{"type": "file", "channels": [0, 1], "source": wav}],
                 "sampling_rate": int(info.samplerate), "num_samples": int(info.frames),
                 "duration": _r(rec_duration), "channel_ids": [0, 1]}
    return {"id": f"{rec_id}-{index}", "start": 0.0, "duration": _r(rec_duration),
            "channel": [0, 1], "recording": recording, "supervisions": sups, "type": "MultiCut"}


# -- manifest -----------------------------------------------------------------------

def _round_floats(obj):
    if isinstance(obj, float):
        return round(obj, ROUND)
    if isinstance(obj, list):
        return [_round_floats(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    return obj


def find_items(root: str) -> List[Tuple[str, str, str]]:
    """(item_dir, batch_id, topic_id) for every folder holding a conversation.json.

    root/category/item (depth 2): batch = root's folder name, topic = category.
    Deeper trees (root/batch/topic/category/item) keep batch/topic from the path.
    `_quarantine` and `lhotse` folders are never entered.
    """
    root = os.path.abspath(root)
    found = []
    for cur, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        if CONVERSATION_FILE in files:
            parts = os.path.relpath(cur, root).split(os.sep)
            if len(parts) == 2:
                batch, topic = os.path.basename(root), parts[0]
            else:
                batch = parts[0] if len(parts) >= 2 else os.path.basename(root)
                topic = parts[1] if len(parts) >= 3 else "none"
            found.append((cur, batch, topic))
            dirs[:] = []
    return found


def build_manifest(root: str, output_path: str,
                   params: Optional[LhotseManifestParams] = None,
                   item_filter: Optional[Callable[[str], bool]] = None) -> ManifestReport:
    params = params or LhotseManifestParams()
    if params.audio_dir is None:
        params = dataclasses.replace(
            params, audio_dir=os.path.join(os.path.dirname(os.path.abspath(output_path)), "audio"))
    opener = gzip.open if output_path.endswith(".gz") else open
    skipped: List[dict] = []
    notes: List[dict] = []
    written = 0
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    with opener(output_path, "wt", encoding="utf-8") as fh:
        for item_dir, batch, topic in find_items(root):
            if item_filter is not None and not item_filter(item_dir):
                continue
            res = convert_item(item_dir, batch, topic, params, index=written)
            if res.cut is None:
                skipped.append({"item": item_dir, "reasons": res.reasons})
                continue
            if res.reasons:
                notes.append({"item": item_dir, "reasons": res.reasons})
            fh.write(json.dumps(_round_floats(res.cut), ensure_ascii=False) + "\n")
            written += 1
    return ManifestReport(written, skipped, notes, output_path)


def _is_recording_item(item_dir: str, stem: str) -> bool:
    """True when the item came from recording `stem` (metadata source_stem, else name prefix)."""
    try:
        with open(os.path.join(item_dir, CONVERSATION_FILE), encoding="utf-8") as fh:
            found = json.load(fh).get("source_stem")
    except (OSError, ValueError):
        found = None
    if found is not None:
        return found == stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    # Same rule as conversation_export_service._item_of: `ep1` must not take `ep1_part2`.
    return re.fullmatch(re.escape(safe) + r"_\d+s(_b)*",
                        os.path.basename(item_dir)) is not None


def build_recording_manifest(root: str, source_stem: str,
                             params: Optional[LhotseManifestParams] = None) -> ManifestReport:
    """Rewrite <root>/lhotse/<stem>.jsonl.gz from this recording's items only (idempotent).

    Swapped WAVs go to <root>/lhotse/audio. The file is built aside and moved in place.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", source_stem)
    out_dir = os.path.join(os.path.abspath(root), LHOTSE_DIR)
    final = os.path.join(out_dir, f"{safe}.jsonl.gz")
    tmp = os.path.join(out_dir, f".tmp-{os.getpid()}-{safe}.jsonl.gz")
    params = dataclasses.replace(params or LhotseManifestParams(),
                                 audio_dir=os.path.join(out_dir, "audio"))
    os.makedirs(out_dir, exist_ok=True)
    try:
        report = build_manifest(root, tmp, params,
                                item_filter=lambda d: _is_recording_item(d, source_stem))
        os.replace(tmp, final)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    report.output_path = final
    return report
