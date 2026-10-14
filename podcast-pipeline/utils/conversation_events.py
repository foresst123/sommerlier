"""Full-duplex events and the assistant/user role check for one exported excerpt.

Pure functions over the selected, word-aligned segments of one two-person
conversation: no audio, no LLM, no files. Times come in on the pipeline clock and
leave on the excerpt clock (`label_events`).

Definitions follow the full-duplex literature where there is one: an inter-pausal
unit (IPU) is one speaker's speech split at a silence longer than `ipu_gap`
(dGSLM); a backchannel is a short listener word while the other speaker talks on
(VAP, Full-Duplex-Bench). The interruption / yield rules have no published
precedent and are heuristics; every threshold lives in `EventParams` and is
configured through `ConversationSelectionConfig`.

Event dicts carry speaker ids in `by` / `of` / `during` until `label_events`
swaps them for "assistant" / "user".
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from utils.conversation_selection import BACKCHANNELS, _fold

INTERRUPTION_INTENTS = ("topic_shift", "clarify", "disagree", "complete")
# The event types that make an excerpt worth filing as full duplex.
TAG_TYPES = ("interruption", "backchannel", "merged_backchannel", "competitive_overlap")

# Short replies that are really questions: they ask the holder something, so they
# are never filed as a backchannel on text alone; what the holder does decides.
QUESTION_BACKCHANNELS = frozenset({"à", "thế à", "vậy à", "vậy hả", "thế ạ"})

# Only used to guess the assistant when the finder gave none and nobody was interrupted.
_QUESTION_WORDS = frozenset({"không", "chưa", "nhỉ", "hả", "gì", "sao", "nào", "đâu", "ai"})


@dataclass(frozen=True)
class EventParams:
    ipu_gap: float = 0.2
    overlap_min: float = 0.2
    backchannel_max_seconds: float = 1.0
    backchannel_max_words: int = 2
    backchannel_context: float = 1.0
    yield_within_seconds: float = 1.5
    cutter_min_seconds: float = 2.0      # a cut-in shorter than this is a comment, not taking the floor
    interrupt_min_overlap: float = 0.5   # the holder must still be talking this long over the cutter
    resume_gap_seconds: float = 0.5      # the holder coming back this soon after the cutter ends: no yield
    min_backchannels: int = 1
    half_duplex_max_overlap: float = 0.2
    role_min_conf: float = 0.7

    @classmethod
    def from_config(cls, cfg) -> "EventParams":
        return cls(**{name: getattr(cfg, name) for name in cls.__dataclass_fields__
                      if hasattr(cfg, name)})


@dataclass
class Unit:
    """One IPU: a speaker's continuous speech."""
    speaker: str
    start: float
    end: float
    text: str
    n_words: int
    keys: List[str] = field(default_factory=list)   # indexes of the segments it came from
    backchannel: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Analysis:
    units: List[Unit]
    events: List[dict]                       # pipeline clock, speaker ids
    overlaps: List[Tuple[float, float]]      # every cross-speaker overlap, merged
    gate_spans: List[Tuple[float, float]]    # where separation must have worked
    max_overlap: float                       # longest single cross-speaker overlap


@dataclass
class RoleResult:
    assistant: str
    user: str
    role: str                                # "locked" | "symmetric"
    role_conf: Optional[float]
    violations: List[str]
    swappable: bool
    source: str                              # "llm" | "behaviour"


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def _timed_words(seg) -> List[Tuple[float, float, str]]:
    """Word times clamped into the segment; a word wholly outside it is ignored."""
    lo, hi = float(seg.start), float(seg.end)
    out = []
    for word in getattr(seg, "words", None) or []:
        try:
            start, end = float(word["start"]), float(word["end"])
            text = str(word.get("word") or word.get("text") or "")
        except (KeyError, TypeError, ValueError):
            return []                        # a half-timed segment counts as untimed
        if end <= lo or start >= hi:
            continue
        out.append((max(start, lo), min(max(end, start), hi), text))
    return sorted(out)


def _speech_pieces(seg) -> List[Tuple[float, float, List[str]]]:
    """(start, end, words) pieces of one segment: its words, or the segment as a whole."""
    words = _timed_words(seg)
    if words:
        return [(s, e, [w]) for s, e, w in words]
    tokens = str(getattr(seg, "text", "") or "").split()
    return [(float(seg.start), float(seg.end), tokens)]


def speech_units(segments, params: EventParams) -> List[Unit]:
    """IPUs of every speaker, by start time."""
    by_speaker: Dict[str, List[Tuple[float, float, List[str], str]]] = {}
    for seg in segments:
        for start, end, tokens in _speech_pieces(seg):
            by_speaker.setdefault(str(seg.speaker), []).append(
                (start, end, tokens, str(seg.index)))
    units: List[Unit] = []
    for speaker, pieces in by_speaker.items():
        pieces.sort(key=lambda p: (p[0], p[1]))
        current: Optional[Unit] = None
        for start, end, tokens, key in pieces:
            if current is not None and start - current.end <= params.ipu_gap:
                current.end = max(current.end, end)
                current.text = (current.text + " " + " ".join(tokens)).strip()
                current.n_words += len(tokens)
                if key not in current.keys:
                    current.keys.append(key)
                continue
            current = Unit(speaker, start, end, " ".join(tokens), len(tokens), [key])
            units.append(current)
    units.sort(key=lambda u: (u.start, u.end))
    return units


def _active(units: Sequence[Unit], speaker: str, lo: float, hi: float) -> bool:
    return any(u.speaker == speaker and u.start < hi and u.end > lo for u in units)


def _mark_backchannels(units: List[Unit], params: EventParams) -> None:
    for unit in units:
        if (unit.duration > params.backchannel_max_seconds
                or unit.n_words > params.backchannel_max_words
                or _fold(unit.text) not in BACKCHANNELS
                or _fold(unit.text) in QUESTION_BACKCHANNELS):
            continue
        others = [u for u in units if u.speaker != unit.speaker and not u.backchannel]
        before = any(o.start < unit.start and o.end > unit.start - params.backchannel_context
                     for o in others)
        # "Continues": the other voice is still there after the listener has finished.
        after = any(o.end > unit.end and o.start < unit.end + params.backchannel_context
                    for o in others)
        unit.backchannel = before and after


def _shared(a: Unit, b: Unit) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _merge(spans):
    merged: List[Tuple[float, float]] = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def _phrase_times(seg, phrase: str) -> Optional[Tuple[float, float]]:
    """Start and end of the words that spell `phrase` inside `seg`, or None."""
    words = _timed_words(seg)
    target = _fold(phrase).split()
    if not words or not target:
        return None
    folded = [_fold(w) for _s, _e, w in words]
    for i in range(len(folded) - len(target) + 1):
        if folded[i:i + len(target)] == target:
            return words[i][0], words[i + len(target) - 1][1]
    return None


def analyze(segments, params: EventParams,
            marked: Optional[Dict[str, str]] = None) -> Analysis:
    """Events of one excerpt. `marked` maps a segment index to the finder's intent for it."""
    marked = marked or {}
    segments = list(segments)
    units = speech_units(segments, params)
    _mark_backchannels(units, params)
    speakers = sorted({u.speaker for u in units})
    events: List[dict] = []

    def other_of(speaker: str) -> Optional[str]:
        rest = [s for s in speakers if s != speaker]
        return rest[0] if len(rest) == 1 else None

    # backchannels
    for unit in units:
        if not unit.backchannel:
            continue
        shared = sum(_shared(unit, o) for o in units
                     if o.speaker != unit.speaker and not o.backchannel)
        # It counts towards a category only if it overlaps the other voice or sits in
        # a pause inside the other's turn (their speech within ipu_gap on both sides).
        others = [o for o in units if o.speaker != unit.speaker and not o.backchannel]
        inside = (any(o.end >= unit.start - params.ipu_gap and o.start < unit.start
                      for o in others)
                  and any(o.start <= unit.end + params.ipu_gap and o.end > unit.end
                          for o in others))
        events.append({"type": "backchannel", "by": unit.speaker,
                       "during": other_of(unit.speaker), "start": unit.start,
                       "end": unit.end, "overlap": min(shared, unit.duration),
                       "text": unit.text, "segment": unit.keys[0],
                       "in_turn": shared > 0 or inside})

    # Interruptions and competitive overlaps: one event per cut-in episode. A listener
    # who pauses and goes on over the same holder is still one cut-in, and the holder's
    # own resumed speech inside it is never a second, reversed event.
    stays = params.yield_within_seconds
    claimed: set = set()
    talk = [u for u in units if not u.backchannel]
    for unit in talk:
        if id(unit) in claimed:
            continue
        holders = [h for h in talk if h.speaker != unit.speaker and id(h) not in claimed
                   and h.start < unit.start < h.end]
        if not holders:
            continue
        holder = min(holders, key=lambda h: h.start)
        episode = [unit]
        for nxt in talk:
            last = episode[-1]
            if (nxt.speaker == unit.speaker and nxt.start >= last.end
                    and nxt.start - last.end <= stays and id(nxt) not in claimed
                    and nxt is not last and nxt not in episode):
                if any(_shared(nxt, h) > 0 for h in talk if h.speaker != unit.speaker):
                    episode.append(nxt)
        start, end = unit.start, episode[-1].end
        theirs = [h for h in talk if h.speaker != unit.speaker
                  and any(_shared(h, e) > 0 for e in episode)]
        overlap = sum(_shared(e, h) for e in episode for h in theirs)
        if overlap < params.overlap_min:
            continue
        claimed.update(id(e) for e in episode)
        claimed.update(id(h) for h in theirs if h is not holder)   # resumed speech, not a cut-in
        # The holder really stops only if no speech of theirs starts while the cutter is still on.
        resumed = any(h.start > holder.end and h.start < end for h in theirs)
        holder_stop = float("inf") if resumed else holder.end - start
        base = {"by": unit.speaker, "of": holder.speaker, "start": start, "end": end,
                "overlap": overlap, "text": " ".join(e.text for e in episode),
                "segment": unit.keys[0]}
        if holder_stop > stays and end - start > stays:
            events.append({"type": "competitive_overlap", **base, "intent": None,
                           "response": None, "yield_after": None})
            continue
        intent = next((marked[k] for e in episode for k in e.keys if k in marked), None)
        response = "continue"
        # The holder stops for real only if the cutter held the floor for a while, the
        # holder was still talking over them, and does not simply take the turn back.
        back_soon = any(h.speaker == holder.speaker and end <= h.start <= end + params.resume_gap_seconds
                        for h in talk)
        held = (end - start >= params.cutter_min_seconds
                and overlap >= params.interrupt_min_overlap and not back_soon)
        if holder_stop <= stays and (held or intent is not None):
            response = "yield"
            next_turn = next((h for h in talk if h.speaker == holder.speaker
                              and h.start >= end - 0.05), None)
            if intent is not None and next_turn is not None:
                response = "adapt"
        events.append({"type": "interruption", **base, "intent": intent,
                       "response": response,
                       "yield_after": holder_stop if holder_stop <= stays else None})

    # words of the other speaker merged into a turn by the diarizer
    for seg in segments:
        for flag in getattr(seg, "review_flags", None) or []:
            if not isinstance(flag, dict) or flag.get("kind") != "speaker_phrase":
                continue
            phrase = str(flag.get("phrase") or "").strip()
            times = _phrase_times(seg, phrase) if phrase else None
            other = other_of(str(seg.speaker))
            if times is None or other is None:
                continue
            events.append({"type": "merged_backchannel", "by": other,
                           "during": str(seg.speaker), "start": times[0], "end": times[1],
                           "overlap": 0.0, "text": phrase, "segment": str(seg.index)})

    # speaker changes with no overlap at all
    reach = None
    previous: Optional[Unit] = None
    for unit in units:
        if unit.backchannel:
            continue
        if (previous is not None and unit.speaker != previous.speaker
                and reach is not None and unit.start >= reach):
            events.append({"type": "turn_switch", "by": unit.speaker, "of": previous.speaker,
                           "start": previous.end, "end": unit.start,
                           "gap": unit.start - previous.end, "overlap": 0.0})
        reach = unit.end if reach is None else max(reach, unit.end)
        previous = unit

    # overlaps overall, for the half-duplex test and for the separation gate
    overlaps, gate, longest = [], [], 0.0
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            if b.start >= a.end:
                break
            if a.speaker == b.speaker:
                continue
            lo, hi = max(a.start, b.start), min(a.end, b.end)
            if hi <= lo:
                continue
            longest = max(longest, hi - lo)
            if hi - lo >= params.overlap_min:
                overlaps.append((lo, hi))
                gate.append((lo, hi))
            elif a.backchannel or b.backchannel:
                gate.append((lo, hi))
    for unit in units:
        if unit.backchannel:
            gate += [(max(unit.start, o.start), min(unit.end, o.end)) for o in units
                     if o.speaker != unit.speaker and _shared(unit, o) > 0]
    events.sort(key=lambda e: (e["start"], e["type"]))
    return Analysis(units, events, _merge(overlaps), _merge(gate), longest)


def event_counts(events: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    return counts


def tags_of(events: Sequence[dict]) -> List[str]:
    present = {e["type"] for e in events}
    return [kind for kind in TAG_TYPES if kind in present]


INTERRUPT, BACKCHANNEL, TURN_BY_TURN = "interrupt", "backchannel", "turn_by_turn"
CATEGORY_NAMES = (INTERRUPT, BACKCHANNEL, TURN_BY_TURN)


def _is_stray_backchannel(unit: Unit) -> bool:
    """A listener word ("ừ", "dạ") that stands alone and was not marked as a backchannel."""
    return (not unit.backchannel and unit.n_words <= 2
            and _fold(unit.text) in BACKCHANNELS)


def categorize(events: Sequence[dict], max_overlap: float, params: EventParams,
               units: Optional[Sequence[Unit]] = None) -> Tuple[Optional[str], Optional[str]]:
    """(category, reason) for one excerpt.

    interrupt     the listener cuts in and the holder stops or changes course.
    backchannel   a short cut-in or listener word while the holder keeps the floor.
    turn_by_turn  speakers alternate; no overlap event and no listener word.
    Every excerpt gets a folder: overlap that is not an interruption is a backchannel.
    """
    if any(e["type"] == "interruption" and e.get("response") in ("yield", "adapt")
           for e in events):
        return INTERRUPT, None
    keeps_floor = sum(
        1 for e in events
        if (e["type"] == "interruption" and e.get("response") == "continue")
        or (e["type"] == "backchannel" and e.get("in_turn", True)))
    merged = sum(1 for e in events if e["type"] == "merged_backchannel")
    if keeps_floor + merged >= params.min_backchannels:
        return BACKCHANNEL, None
    # Any overlap that is not an interruption, and any listener word, is a backchannel;
    # turn_by_turn is strictly one voice at a time.
    if (any(e["type"] in ("backchannel", "competitive_overlap") for e in events)
            or max_overlap >= params.half_duplex_max_overlap
            or (units is not None and any(_is_stray_backchannel(u) for u in units))):
        return BACKCHANNEL, None
    return TURN_BY_TURN, None


def category_is_plausible(category: str, events: Sequence[dict], max_overlap: float,
                          params: EventParams,
                          units: Optional[Sequence[Unit]] = None) -> bool:
    """Whether the measurements leave room for the folder the model named.

    Only a guard against a label the numbers contradict: interrupt needs a measured
    cut-in (an overlap that is only a listener word is not one), backchannel needs
    some overlap or listener word, turn_by_turn needs none of either.
    """
    counts = event_counts(events)
    if category == INTERRUPT:
        return bool(counts.get("interruption"))
    if category == BACKCHANNEL:
        return (any(counts.get(k) for k in ("backchannel", "interruption", "merged_backchannel",
                                            "competitive_overlap"))
                or max_overlap > 0
                or (units is not None and any(_is_stray_backchannel(u) for u in units)))
    if category == TURN_BY_TURN:
        return categorize(events, max_overlap, params, units)[0] == TURN_BY_TURN
    return False


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

def _answers(units: Sequence[Unit], speaker: str) -> int:
    """Turns of `speaker` that directly follow a question of the other speaker."""
    count, last = 0, None
    for unit in units:
        if unit.backchannel:
            continue
        if last is not None and last.speaker != speaker and unit.speaker == speaker:
            tokens = last.text.lower().split()
            if "?" in last.text or (tokens and tokens[-1].strip("?.,") in _QUESTION_WORDS):
                count += 1
        last = unit
    return count


def decide_roles(events: Sequence[dict], speakers: Tuple[str, str],
                 assistant: Optional[str], role_conf: Optional[float],
                 params: EventParams, units: Optional[Sequence[Unit]] = None) -> RoleResult:
    """Lock the finder's assistant when its behaviour agrees, else call the roles symmetric."""
    first, second = speakers

    def yields(speaker: str) -> int:
        return sum(1 for e in events if e["type"] == "interruption" and e.get("of") == speaker
                   and e.get("response") in ("yield", "adapt"))

    if assistant not in speakers:
        a, b = yields(first), yields(second)
        if a == b and units is not None:
            a, b = _answers(units, first), _answers(units, second)
        chosen = first if a > b else second
        return RoleResult(chosen, first if chosen == second else second, "symmetric",
                          None, [], True, "behaviour")

    user = second if assistant == first else first
    violations: List[str] = []
    held_on = any(e["type"] == "interruption" and e.get("of") == assistant
                  and e.get("response") == "continue" for e in events)
    if held_on and yields(user):
        violations.append("assistant_holds_on_user_yields")
    shifts = sum(1 for e in events if e["type"] == "interruption" and e.get("by") == assistant
                 and e.get("intent") == "topic_shift")
    if shifts >= 2:
        violations.append("assistant_shifts_topic")
    conf = None if role_conf is None else max(0.0, float(role_conf) - 0.2 * len(violations))
    locked = conf is not None and conf >= params.role_min_conf and not violations
    return RoleResult(assistant, user, "locked" if locked else "symmetric", conf,
                      violations, not locked, "llm")


def label_events(events: Sequence[dict], roles: Dict[str, str], origin: float = 0.0) -> List[dict]:
    """Events with roles instead of speaker ids and times from the start of the excerpt."""
    out = []
    for event in events:
        row = dict(event)
        for key in ("by", "of", "during"):
            if row.get(key) is not None:
                row[key] = roles.get(row[key], row[key])
        for key in ("start", "end"):
            row[key] = round(float(row[key]) - origin, 3)
        for key in ("overlap", "gap", "yield_after"):
            if row.get(key) is not None:
                row[key] = round(float(row[key]), 3)
        out.append(row)
    return out
