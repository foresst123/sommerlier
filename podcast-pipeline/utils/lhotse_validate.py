"""Validate SpeechWorld full-duplex Lhotse manifests (spec v2.2.0, section 9 + related clauses).

Works on plain dicts (one JSON line = one MultiCut), so lhotse is not needed. Codes are
INV01..INV11, one per invariant of section 9:

 01 cut/recording shape (MultiCut, channel [0,1], start 0, duration, ids, channel_ids)
 02 supervision ids unique across the manifest
 03 supervision inside the cut
 04 alignment absolute, ordered, inside the supervision
 05 alignment items are [symbol, start, DURATION, score|null]
 06 same-channel turns do not overlap
 07 backchannel: opposite channel, inside the anchor, id format, anchor_word_index
 08 anchor_id <-> backchannel_ids agree both ways
 09 event_type in the five-value set
 10 text has none of < > [ ]
 11 role / event_type / speaker / assistance_voice / gender rules

Further MUST clauses outside the section 9 table are reported as EXT12..EXT17:

 12 prompt event starts at 0 (4.5)
 13 function_call / function_response: anchor_id, anchor_word_index, JSON text (4.3, 4.5)
 14 language present on every supervision (4.2)
 15 backchannel ids use k = 0, 1, 2... per anchor turn (4.4)
 16 no top-level supervision field is null (2)
 17 turn: backchannel_ids is a list, transition is {"type": gap|overlap, "duration": s} (4.3)

Tolerance is 1 ms, 10 ms along alignment chains.

CLI:  python -m utils.lhotse_validate manifest.jsonl[.gz] [--max-print N]    (exit 1 on any violation)
"""
import argparse
import gzip
import json
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

EVENT_TYPES = ("turn", "backchannel", "prompt", "function_call", "function_response")
ROLES = ("system", "user", "assistant", "tool")
# (role, event_type) -> (has audio, speaker required)
VALID_PAIRS: Dict[Tuple[str, str], Tuple[bool, bool]] = {
    ("user", "turn"): (True, True),
    ("user", "backchannel"): (True, True),
    ("assistant", "turn"): (True, True),
    ("assistant", "backchannel"): (True, True),
    ("assistant", "function_call"): (False, False),
    ("tool", "function_response"): (False, False),
    ("system", "prompt"): (False, False),
}
FORBIDDEN_TEXT = "<>[]"
DEFAULT_TOL = 0.001
DEFAULT_ALI_TOL = 0.010


@dataclass(frozen=True)
class Violation:
    invariant: int
    code: str
    cut_id: str
    supervision_id: Optional[str]
    message: str

    def __str__(self) -> str:
        where = f"{self.cut_id}/{self.supervision_id}" if self.supervision_id else self.cut_id
        return f"{self.code} {where}: {self.message}"


def _num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _event_type(sup: dict) -> str:
    return (sup.get("custom") or {}).get("event_type") or "turn"


def _is_audio_turn(sup: dict) -> bool:
    return _event_type(sup) == "turn" and _num(sup.get("start")) and _num(sup.get("duration"))


def validate_cut(cut: dict, tol: float = DEFAULT_TOL, ali_tol: float = DEFAULT_ALI_TOL) -> List[Violation]:
    """Violations of invariants 1 and 3-11 in one cut (invariant 2 needs the whole manifest)."""
    cut_id = str(cut.get("id"))
    out: List[Violation] = []

    def bad(inv: int, sup: Optional[dict], message: str) -> None:
        sid = None if sup is None else str(sup.get("id"))
        out.append(Violation(inv, f"INV{inv:02d}" if inv <= 11 else f"EXT{inv:02d}", cut_id, sid, message))

    recording = cut.get("recording") if isinstance(cut.get("recording"), dict) else {}
    sups = [s for s in (cut.get("supervisions") or []) if isinstance(s, dict)]
    rec_duration = recording.get("duration")

    # 1: cut and recording
    if cut.get("type") != "MultiCut":
        bad(1, None, f"type is {cut.get('type')!r}, expected 'MultiCut'")
    if cut.get("channel") != [0, 1]:
        bad(1, None, f"channel is {cut.get('channel')!r}, expected [0, 1]")
    if not _num(cut.get("start")) or abs(cut["start"]) > tol:
        bad(1, None, f"start is {cut.get('start')!r}, expected 0")
    if not _num(cut.get("duration")) or not _num(rec_duration) or abs(cut["duration"] - rec_duration) > tol:
        bad(1, None, f"duration {cut.get('duration')!r} != recording.duration {rec_duration!r}")
    if recording.get("channel_ids") != [0, 1]:
        bad(1, None, f"recording.channel_ids is {recording.get('channel_ids')!r}, expected [0, 1]")
    rec_id = recording.get("id")
    if not isinstance(rec_id, str) or not rec_id:
        bad(1, None, "recording.id missing")
    elif not cut_id.startswith(rec_id):
        bad(1, None, f"cut.id does not start with recording.id {rec_id!r}")
    for sup in sups:
        if sup.get("recording_id") != rec_id:
            bad(1, sup, f"recording_id {sup.get('recording_id')!r} != recording.id {rec_id!r}")

    cut_duration = cut.get("duration") if _num(cut.get("duration")) else rec_duration
    by_id = {s.get("id"): s for s in sups}

    for sup in sups:
        custom = sup.get("custom") if isinstance(sup.get("custom"), dict) else {}
        start, dur = sup.get("start"), sup.get("duration")
        timed = _num(start) and _num(dur)

        # 3: inside the cut
        if not timed:
            bad(3, sup, f"start/duration not numeric: {start!r}, {dur!r}")
        else:
            if start < -tol:
                bad(3, sup, f"start {start} < 0")
            if dur < -tol:
                bad(3, sup, f"duration {dur} < 0")
            if _num(cut_duration) and start + dur > cut_duration + tol:
                bad(3, sup, f"ends at {start + dur:.3f} past cut duration {cut_duration}")

        _check_alignment(sup, timed, tol, ali_tol, bad)

        # 10: text
        text = sup.get("text")
        if not isinstance(text, str):
            if text is not None or _event_type(sup) in ("turn", "backchannel"):
                bad(10, sup, "text missing or not a string")
        else:
            found = sorted({c for c in text if c in FORBIDDEN_TEXT})
            if found:
                bad(10, sup, f"text contains {''.join(found)}")

        # 9 and 11
        etype = custom.get("event_type")
        if etype is not None and etype not in EVENT_TYPES:
            bad(9, sup, f"event_type {etype!r} not in {EVENT_TYPES}")
        _check_role(sup, custom, timed, tol, bad)

        if _event_type(sup) == "backchannel":
            _check_backchannel(sup, custom, by_id, timed, tol, bad)
        _check_extra(sup, custom, by_id, timed, tol, bad)

    # 8: anchors list their backchannels
    for sup in sups:
        custom = sup.get("custom") or {}
        ids = custom.get("backchannel_ids")
        if ids is None:
            continue
        if not isinstance(ids, list):
            bad(8, sup, "backchannel_ids is not a list")
            continue
        for bid in ids:
            child = by_id.get(bid)
            if child is None:
                bad(8, sup, f"backchannel_ids names {bid!r}, which is not in the cut")
            elif (child.get("custom") or {}).get("anchor_id") != sup.get("id"):
                bad(8, sup, f"{bid!r} does not point back to this turn via anchor_id")

    _check_backchannel_numbering(sups, bad)

    # 6: same-channel turns do not overlap
    per_channel: Dict[object, List[dict]] = {}
    for sup in sups:
        if _is_audio_turn(sup) and isinstance(sup.get("channel"), int) and (sup.get("custom") or {}).get("role") in ("user", "assistant"):
            per_channel.setdefault(sup["channel"], []).append(sup)
    for channel, turns in per_channel.items():
        turns.sort(key=lambda s: s["start"])
        for prev, nxt in zip(turns, turns[1:]):
            over = prev["start"] + prev["duration"] - nxt["start"]
            if over > tol:
                bad(6, nxt, f"overlaps {prev.get('id')!r} on channel {channel} by {over:.3f}s")
    return out


def _check_alignment(sup: dict, timed: bool, tol: float, ali_tol: float, bad) -> None:
    alignment = sup.get("alignment")
    if alignment is None:
        return
    if not isinstance(alignment, dict):
        bad(5, sup, "alignment is not an object keyed by type")
        return
    for kind, items in alignment.items():
        if not isinstance(items, list):
            bad(5, sup, f"alignment[{kind!r}] is not a list")
            continue
        clean: List[list] = []
        for k, item in enumerate(items):
            if not (isinstance(item, (list, tuple)) and len(item) == 4):
                bad(5, sup, f"alignment[{kind!r}][{k}] is not [symbol, start, duration, score]")
                continue
            symbol, a_start, a_dur, score = item
            if not isinstance(symbol, str) or not _num(a_start) or not _num(a_dur):
                bad(5, sup, f"alignment[{kind!r}][{k}] has a non-numeric start/duration or non-string symbol")
                continue
            if a_dur < -ali_tol:
                bad(5, sup, f"alignment[{kind!r}][{k}] duration {a_dur} < 0")
            elif timed and a_dur > sup["duration"] + ali_tol:
                bad(5, sup, f"alignment[{kind!r}][{k}] duration {a_dur} exceeds the supervision: an end time?")
            if score is not None and not _num(score):
                bad(5, sup, f"alignment[{kind!r}][{k}] score {score!r} is neither null nor a number")
            clean.append(list(item))
        if not clean:
            continue
        if timed:
            first, last = clean[0], clean[-1]
            if first[1] < sup["start"] - ali_tol:
                bad(4, sup, f"alignment[{kind!r}] starts at {first[1]} before the supervision ({sup['start']})")
            if last[1] + last[2] > sup["start"] + sup["duration"] + ali_tol:
                bad(4, sup, f"alignment[{kind!r}] ends at {last[1] + last[2]:.3f} after the supervision "
                            f"({sup['start'] + sup['duration']:.3f})")
        for prev, nxt in zip(clean, clean[1:]):
            if prev[1] + prev[2] > nxt[1] + ali_tol:
                bad(4, sup, f"alignment[{kind!r}] {prev[0]!r} ends at {prev[1] + prev[2]:.3f} after "
                            f"{nxt[0]!r} starts at {nxt[1]:.3f}")


def _check_role(sup: dict, custom: dict, timed: bool, tol: float, bad) -> None:
    role = custom.get("role")
    if role is None:
        bad(11, sup, "custom.role missing")
        return
    etype = _event_type(sup)
    pair = VALID_PAIRS.get((role, etype))
    if pair is None:
        if role not in ROLES:
            bad(11, sup, f"role {role!r} not in {ROLES}")
        elif etype in EVENT_TYPES:
            bad(11, sup, f"(role, event_type) = ({role!r}, {etype!r}) is not a valid pair")
        speaker_ok = True
    else:
        audio, needs_speaker = pair
        has_speaker = bool(sup.get("speaker"))
        if needs_speaker and not has_speaker:
            bad(11, sup, f"speaker is required for ({role}, {etype})")
        if not needs_speaker and "speaker" in sup and sup.get("speaker") is not None:
            bad(11, sup, f"speaker must not exist for ({role}, {etype})")
        channel = sup.get("channel")
        if audio and channel not in (0, 1):
            bad(11, sup, f"channel {channel!r} must be 0 or 1 for ({role}, {etype})")
        if not audio:
            if channel != [0, 1]:
                bad(11, sup, f"channel {channel!r} must be [0, 1] for event ({role}, {etype})")
            if timed and abs(sup["duration"]) > 1e-9:
                bad(11, sup, f"event ({role}, {etype}) must have duration 0, got {sup['duration']}")
        speaker_ok = True
    if "assistance_voice" in custom and custom["assistance_voice"] is not None \
            and custom["assistance_voice"] != (role == "assistant"):
        bad(11, sup, f"assistance_voice {custom['assistance_voice']!r} does not match role {role!r}")
    gender = sup.get("gender")
    if isinstance(gender, str) and gender.strip().lower() == "null":
        bad(11, sup, 'gender is the string "null"; omit the key instead')
    if speaker_ok and sup.get("speaker") == "Backchannel":
        bad(11, sup, 'speaker is the label "Backchannel", not the real listener')


def _check_backchannel(sup: dict, custom: dict, by_id: dict, timed: bool, tol: float, bad) -> None:
    anchor_id = custom.get("anchor_id")
    sid = sup.get("id")
    anchor = by_id.get(anchor_id)
    if anchor is None:
        bad(7, sup, f"anchor_id {anchor_id!r} is not a supervision of this cut")
        return
    if isinstance(sid, str) and isinstance(anchor_id, str):
        head, sep, tail = sid.rpartition("-bc")
        if not (head == anchor_id and sep and tail.isdigit()):
            bad(7, sup, f"id should be '{anchor_id}-bc<k>'")
    if sup.get("channel") not in (0, 1) or anchor.get("channel") not in (0, 1) \
            or sup.get("channel") != 1 - anchor.get("channel"):
        bad(7, sup, f"channel {sup.get('channel')!r} is not the opposite of the anchor's {anchor.get('channel')!r}")
    if timed and _num(anchor.get("start")) and _num(anchor.get("duration")):
        if sup["start"] < anchor["start"] - tol \
                or sup["start"] + sup["duration"] > anchor["start"] + anchor["duration"] + tol:
            bad(7, sup, f"[{sup['start']:.3f}, {sup['start'] + sup['duration']:.3f}] is not inside the anchor "
                        f"[{anchor['start']:.3f}, {anchor['start'] + anchor['duration']:.3f}]")
    index = custom.get("anchor_word_index")
    words = len(str(anchor.get("text") or "").split())
    if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= max(words, 1) or words == 0:
        bad(7, sup, f"anchor_word_index {index!r} is not an integer in [1, {words}]")
    if sid not in ((anchor.get("custom") or {}).get("backchannel_ids") or []):
        bad(8, sup, f"anchor {anchor_id!r} does not list this backchannel in backchannel_ids")


def _check_extra(sup: dict, custom: dict, by_id: dict, timed: bool, tol: float, bad) -> None:
    etype = _event_type(sup)
    # 14: language
    if not isinstance(sup.get("language"), str) or not sup["language"]:
        bad(14, sup, "language missing")
    # 16: null top-level fields are dropped by the dataclass serializer
    for key, value in sup.items():
        if value is None:
            bad(16, sup, f"top-level field {key!r} is null; omit the key (null is only kept inside custom)")
    # 12: prompt at the start of the conversation
    if etype == "prompt" and timed and abs(sup["start"]) > tol:
        bad(12, sup, f"prompt must start at 0, got {sup['start']}")
    # 13: function events hang on a word of an anchor turn and carry a JSON payload
    if etype in ("function_call", "function_response"):
        anchor = by_id.get(custom.get("anchor_id"))
        if anchor is None:
            bad(13, sup, f"anchor_id {custom.get('anchor_id')!r} is not a supervision of this cut")
        else:
            index = custom.get("anchor_word_index")
            words = len(str(anchor.get("text") or "").split())
            if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= words:
                bad(13, sup, f"anchor_word_index {index!r} is not an integer in [1, {words}]")
            if timed and _num(anchor.get("start")) and _num(anchor.get("duration")) and not (
                    anchor["start"] - tol <= sup["start"] <= anchor["start"] + anchor["duration"] + tol):
                bad(13, sup, f"start {sup['start']} is outside the anchor turn")
        keys = ("name", "arguments") if etype == "function_call" else ("name", "content")
        try:
            payload = json.loads(sup.get("text"))
        except (TypeError, ValueError):
            payload = None
        if not isinstance(payload, dict) or not all(k in payload for k in keys):
            bad(13, sup, f"text must be a JSON object with keys {keys}")
    # 17: turn bookkeeping
    if etype == "turn":
        if not isinstance(custom.get("backchannel_ids"), list):
            bad(17, sup, "custom.backchannel_ids must be a list ([] when there is none)")
        transition = custom.get("transition")
        if transition is not None and not (
                isinstance(transition, dict) and transition.get("type") in ("gap", "overlap")
                and _num(transition.get("duration"))):
            bad(17, sup, f'custom.transition {transition!r} is not {{"type": "gap"|"overlap", "duration": s}}')


def _check_backchannel_numbering(sups: List[dict], bad) -> None:
    """15: backchannels of one anchor are {anchor_id}-bc0, -bc1, ... without gaps."""
    ks: Dict[object, List[int]] = {}
    for sup in sups:
        custom = sup.get("custom") or {}
        sid, anchor_id = sup.get("id"), custom.get("anchor_id")
        if _event_type(sup) != "backchannel" or not isinstance(sid, str) or not isinstance(anchor_id, str):
            continue
        head, _, tail = sid.rpartition("-bc")
        if head == anchor_id and tail.isdigit():
            ks.setdefault(anchor_id, []).append(int(tail))
    for anchor_id, found in ks.items():
        if sorted(found) != list(range(len(found))):
            bad(15, None, f"backchannel numbers of {anchor_id!r} are {sorted(found)}, expected 0..{len(found) - 1}")


def validate_cuts(cuts: Iterable[dict], tol: float = DEFAULT_TOL, ali_tol: float = DEFAULT_ALI_TOL) -> List[Violation]:
    """validate_cut on every cut, plus invariant 2 (supervision ids unique across the manifest)."""
    out: List[Violation] = []
    seen: Dict[str, str] = {}
    for cut in cuts:
        out.extend(validate_cut(cut, tol, ali_tol))
        for sup in cut.get("supervisions") or []:
            sid = sup.get("id") if isinstance(sup, dict) else None
            if sid is None:
                continue
            if sid in seen:
                out.append(Violation(2, "INV02", str(cut.get("id")), str(sid),
                                     f"supervision id already used in cut {seen[sid]!r}"))
            else:
                seen[sid] = str(cut.get("id"))
    return out


def iter_cuts(path: str) -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate_file(path: str) -> List[Violation]:
    """validate_cuts over a .jsonl / .jsonl.gz manifest."""
    return validate_cuts(iter_cuts(path))


def validate_with_lhotse(path: str) -> List[Violation]:
    """Load the manifest with lhotse (CutSet.from_file) and report what it rejects; [] without lhotse."""
    try:
        from lhotse import CutSet
    except ImportError:
        return []
    try:
        cuts = CutSet.from_file(path)
        for cut in cuts:
            for sup in cut.supervisions:
                _ = (sup.id, sup.start, sup.duration, sup.channel)
    except Exception as exc:
        return [Violation(1, "INV01", "<file>", None, f"lhotse cannot load the manifest: {exc}")]
    return []


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a SpeechWorld full-duplex Lhotse manifest (spec 2.2.0).")
    ap.add_argument("manifest", help=".jsonl or .jsonl.gz")
    ap.add_argument("--max-print", type=int, default=50, help="violations to print (default 50, 0 = all)")
    ap.add_argument("--no-lhotse", action="store_true", help="skip the lhotse CutSet.from_file load check")
    args = ap.parse_args(argv)
    cuts = [0]

    def counted():
        for cut in iter_cuts(args.manifest):
            cuts[0] += 1
            yield cut

    violations = validate_cuts(counted())
    if not args.no_lhotse:
        violations += validate_with_lhotse(args.manifest)
    shown = violations if args.max_print == 0 else violations[:args.max_print]
    for v in shown:
        print(v)
    if len(shown) < len(violations):
        print(f"... {len(violations) - len(shown)} more (use --max-print 0)")
    bad_cuts = len({v.cut_id for v in violations})
    print(f"\n{cuts[0]} cuts checked, {len(violations)} violations in {bad_cuts} cuts")
    for code, n in sorted(Counter(v.code for v in violations).items()):
        print(f"  {code}: {n}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
