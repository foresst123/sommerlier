"""Persona profile stage (core): per-speaker habits of one recording.

Every number comes from code over fixed, versioned dictionaries. The LLM only
writes `style.description` and proposes extra forms; neither enters a number.
See doc/specs/2026-10-12-persona-profile-design.md.
"""
import hashlib
import json
import math
import re
import unicodedata
from itertools import combinations
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Tuple

from schemas.persona_profile import (
    AddressTerm, BackchannelProfile, Dist, Filler, InterruptionProfile, LexiconItem, Particle,
    PersonaProfile, PersonaStats, Rate, Ratio, SpeakerProfile, Style, TurnProfile)
from utils.conversation_events import EventParams, _mark_backchannels, _merge, analyze, speech_units
from utils.conversation_selection import (
    _fold, _unseparated_seconds, _union_seconds, foreign_seconds, hard_review_kind,
    soft_review_count)
from utils.llm_batches import ask_in_batches, reply_budget
from utils.llm_json import is_readable, objects_in
from utils.noise_map import dominant_kind
from utils.provenance import annotate

SCHEMA_VERSION = 1
# Hand-written. Change it whenever a constant, dictionary, EventParams default or
# formula below changes a number (tests/test_persona_profile.py guards this).
PARAMS_VERSION = "pp-2026.10.3"
PROFILE_PROMPT_VERSION = "persona-v1"

MIN_DIST_N = 20                     # distribution needs this many samples
MIN_RATIO_N = 8                     # ratio needs this denominator
MIN_EXPOSURE_MIN = 5.0              # minutes of opportunity for a frequency / style lists
MIN_EXPOSURE_INTERRUPTION_MIN = 20.0
MIN_LEXICON_N = 5
RATE_MIN_WORDS = 3
RATE_MIN_UNIT_S = 1.0
RATE_MIN_WPS = 0.5
RATE_MAX_WPS = 9.0
SENTENCE_MIN_WORDS = 2
SENTENCE_MAX_WORDS = 80
LATENCY_FLOOR_S = -3.0
LATENCY_CAP_S = 6.0
EDGE_MARGIN_S = 1.5                 # >= yield_within_seconds
NOISE_FLOOR = 0.10                  # the stage's own cut; never read NOISE_NOTICEABLE
LLM_WINDOW_CHARS = 6000
MAX_CANDIDATE_FORMS = 20
DESCRIPTION_MAX_CHARS = 400

# Conservative: only forms that almost always play that role (spec B4.4).
FILLERS = ("ừm", "ờm", "ờ", "ưm", "hừm", "đại loại")
PARTICLES = ("nhé", "nhỉ", "nha", "nhá", "nè", "nhể", "hả", "hử", "ạ", "chứ")
ADDRESS_TERMS = ("anh", "chị", "em", "mình", "bạn", "tớ", "cậu", "tôi")

EXCLUSION_KEYS = ("review_flag", "music", "noise", "unseparated_overlap", "empty_text")
EVENT_PARAMS = EventParams()
_NOISY_KINDS = frozenset({"noise_speech", "noise_env", "noise_room"})


def params_snapshot() -> dict:
    """Every constant that can change a number, for the PARAMS_VERSION guard."""
    return {
        "MIN_DIST_N": MIN_DIST_N, "MIN_RATIO_N": MIN_RATIO_N,
        "MIN_EXPOSURE_MIN": MIN_EXPOSURE_MIN,
        "MIN_EXPOSURE_INTERRUPTION_MIN": MIN_EXPOSURE_INTERRUPTION_MIN,
        "MIN_LEXICON_N": MIN_LEXICON_N, "RATE_MIN_WORDS": RATE_MIN_WORDS,
        "RATE_MIN_UNIT_S": RATE_MIN_UNIT_S, "RATE_MIN_WPS": RATE_MIN_WPS,
        "RATE_MAX_WPS": RATE_MAX_WPS, "SENTENCE_MIN_WORDS": SENTENCE_MIN_WORDS,
        "SENTENCE_MAX_WORDS": SENTENCE_MAX_WORDS, "LATENCY_FLOOR_S": LATENCY_FLOOR_S,
        "LATENCY_CAP_S": LATENCY_CAP_S, "EDGE_MARGIN_S": EDGE_MARGIN_S,
        "NOISE_FLOOR": NOISE_FLOOR, "LLM_WINDOW_CHARS": LLM_WINDOW_CHARS,
        "MAX_CANDIDATE_FORMS": MAX_CANDIDATE_FORMS,
        "DESCRIPTION_MAX_CHARS": DESCRIPTION_MAX_CHARS,
        "FILLERS": list(FILLERS), "PARTICLES": list(PARTICLES),
        "ADDRESS_TERMS": list(ADDRESS_TERMS), "EVENT_PARAMS": asdict(EVENT_PARAMS),
    }


def rnd(x: Optional[float]) -> Optional[float]:
    """Round to 3 places AFTER any threshold was applied; None stays None."""
    return None if x is None else round(float(x), 3)


def percentile(values: Iterable[float], q: float) -> float:
    """NumPy-style linear interpolation, in pure Python."""
    xs = sorted(values)
    k = q * (len(xs) - 1)
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (k - lo) * (xs[hi] - xs[lo])


def make_dist(values: List[float]) -> Dist:
    n = len(values)
    if n < MIN_DIST_N:
        return Dist(None, None, None, n)
    return Dist(rnd(percentile(values, 0.1)), rnd(percentile(values, 0.5)),
                rnd(percentile(values, 0.9)), n)


def make_rate(count: int, exposure_min: float, min_exposure: float, per: float = 1.0) -> Rate:
    """count per `exposure_min` minutes, times `per`; None under `min_exposure`."""
    if exposure_min < min_exposure or exposure_min <= 0:
        return Rate(None, count, rnd(exposure_min))
    return Rate(rnd(count / exposure_min * per), count, rnd(exposure_min))


def make_ratio(numerator: int, n: int) -> Ratio:
    if n < MIN_RATIO_N:
        return Ratio(None, n)
    return Ratio(rnd(numerator / n), n)


def nfc(text) -> str:
    return unicodedata.normalize("NFC", text or "")


def prepare_segments(segments, timeline, noise=None) -> list:
    """P0: working copies (no words, NFC text, sorted by start, provenance filled)."""
    ordered = sorted(segments, key=lambda s: float(s.start))      # stable
    copies = [replace(s, words=None, text=nfc(s.text)) for s in ordered]
    annotate(copies, timeline, noise=noise)
    return copies


def is_empty(seg) -> bool:
    return _fold(getattr(seg, "text", "") or "") == ""


def exclusion_reason(seg) -> Optional[str]:
    """The one reason `seg` adds no numbers, in the spec's priority order; None = usable."""
    if (hard_review_kind(seg) is not None or foreign_seconds(seg) > 0
            or soft_review_count(seg) > 0):
        return "review_flag"
    if getattr(seg, "has_music", False):
        return "music"
    if dominant_kind(getattr(seg, "noise_breakdown", None), floor=NOISE_FLOOR) in _NOISY_KINDS:
        return "noise"
    if _unseparated_seconds(seg) > 0:
        return "unseparated_overlap"
    if is_empty(seg):
        return "empty_text"
    return None


def usable(seg) -> bool:
    return exclusion_reason(seg) is None


def excluded_seconds(prepared) -> Dict[str, float]:
    out = {key: 0.0 for key in EXCLUSION_KEYS}
    for s in prepared:
        reason = exclusion_reason(s)
        if reason is not None:
            out[reason] += float(s.end) - float(s.start)
    return {key: round(v, 3) for key, v in out.items()}


def source_digest_of(prepared) -> str:
    """sha1 of every field P1/P2 read, per segment."""
    rows = [{
        "index": str(s.index), "start": s.start, "end": s.end, "speaker": str(s.speaker),
        "text": s.text, "has_music": bool(getattr(s, "has_music", False)),
        "unseparated": getattr(s, "unseparated", None),
        "noise_breakdown": getattr(s, "noise_breakdown", None),
        "crosses_cut": bool(getattr(s, "crosses_cut", False)),
        "gap_before": getattr(s, "gap_before", None),
        "review_flags": getattr(s, "review_flags", None),
    } for s in prepared]
    blob = json.dumps(rows, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


_SENT_END = re.compile(r"(?<!\d)[.!?…]+(?=\s|$)")


def split_sentences(text: str) -> List[str]:
    """Finished sentences only; the unfinished tail may be cut by a turn boundary."""
    out, last = [], 0
    for m in _SENT_END.finditer(text):
        out.append(text[last:m.end()])
        last = m.end()
    return out


def sentence_word_count(sentence: str) -> int:
    return sum(1 for t in sentence.split() if re.search(r"\w", t))


def fold_tokens(text: str) -> List[str]:
    return _fold(nfc(text)).split()


def count_form(tokens: List[str], form: List[str]) -> int:
    """Occurrences of `form` as a run of whole tokens (never a substring of a token)."""
    n = len(form)
    return sum(1 for i in range(len(tokens) - n + 1) if tokens[i:i + n] == form)


@dataclass
class StyleMeasure:
    style: Style
    speech_min: float
    n_sentences: int
    texts: List[str] = field(default_factory=list)    # clean, non-backchannel unit texts


def _turn_runs(units):
    """Consecutive non-backchannel units of one speaker, by start."""
    runs = []
    for u in units:
        if u.backchannel:
            continue
        if runs and runs[-1][0] == u.speaker:
            runs[-1][1].append(u)
        else:
            runs.append((u.speaker, [u]))
    return runs


def measure_style(prepared) -> Dict[str, StyleMeasure]:
    """Style of every speaker, once over the whole file (spec B4.4)."""
    speakers = sorted({str(s.speaker) for s in prepared})
    speech = [s for s in prepared if not is_empty(s)]
    clean_keys = {str(s.index) for s in speech if usable(s)}
    units = speech_units(speech, EVENT_PARAMS)
    _mark_backchannels(units, EVENT_PARAMS)
    is_clean = lambda u: all(k in clean_keys for k in u.keys)

    sentences: Dict[str, List[str]] = {sp: [] for sp in speakers}
    for speaker, run in _turn_runs(units):
        if all(is_clean(u) for u in run):
            sentences[speaker] += split_sentences(" ".join(u.text for u in run))

    out: Dict[str, StyleMeasure] = {}
    for sp in speakers:
        mine = [u for u in units if u.speaker == sp and not u.backchannel and is_clean(u)]
        speech_min = sum(u.duration for u in mine) / 60.0
        rates = [u.n_words / u.duration for u in mine
                 if u.n_words >= RATE_MIN_WORDS and u.duration >= RATE_MIN_UNIT_S
                 and RATE_MIN_WPS <= u.n_words / u.duration <= RATE_MAX_WPS]
        kept = [s for s in sentences[sp]
                if SENTENCE_MIN_WORDS <= sentence_word_count(s) <= SENTENCE_MAX_WORDS]
        lengths = [sentence_word_count(s) for s in kept]
        per_unit = [fold_tokens(u.text) for u in mine]

        def count(form: str) -> int:
            f = fold_tokens(form)
            return sum(count_form(t, f) for t in per_unit)

        enough = speech_min >= MIN_EXPOSURE_MIN and speech_min > 0
        fillers = ([Filler(f, rnd(count(f) / speech_min)) for f in FILLERS]
                   if enough else None)
        address = [AddressTerm(f, count(f)) for f in ADDRESS_TERMS] if enough else None
        particles = None
        if len(kept) >= MIN_DIST_N:
            ends = [t[-1] for t in map(fold_tokens, kept) if t]
            particles = [Particle(f, rnd(100.0 * sum(1 for e in ends if e == fold_tokens(f)[0])
                                         / len(kept))) for f in PARTICLES]
        style = Style(speech_rate_wps=make_dist(rates), words_per_sentence=make_dist(lengths),
                      fillers=fillers, particles=particles, address_terms=address,
                      llm_candidate_forms=None, description=None)
        out[sp] = StyleMeasure(style, speech_min, len(kept), [u.text for u in mine])
    return out


def split_regions(prepared) -> Tuple[List[list], int]:
    """Contiguous regions; a region starts at an unknown gap and a crosses_cut segment stands alone."""
    regions: List[list] = []
    current: list = []
    unknown = 0
    for pos, s in enumerate(prepared):
        new = False
        if pos > 0 and getattr(s, "gap_before", None) is None:
            unknown += 1
            new = True
        crossing = bool(getattr(s, "crosses_cut", False))
        if (new or crossing) and current:
            regions.append(current)
            current = []
        current.append(s)
        if crossing:
            regions.append(current)
            current = []
    if current:
        regions.append(current)
    return regions, unknown


@dataclass
class Accumulator:
    other_talk_s: float = 0.0
    backchannels: dict = field(default_factory=dict)     # (type, by, start, end) -> event
    interruptions: dict = field(default_factory=dict)    # events where this speaker cut in
    cut_ins: dict = field(default_factory=dict)          # events where this speaker was cut into
    turns: dict = field(default_factory=dict)            # (speaker, start, end) -> seconds
    latencies: dict = field(default_factory=dict)        # (speaker, u.start, u.end) -> seconds


def _hits(mask, lo, hi) -> bool:
    return any(a < hi and b > lo for a, b in mask)


def _free_seconds(spans, lo, hi, mask) -> float:
    """Seconds of the union of `spans` clipped to [lo, hi] and outside `mask`."""
    clipped = _merge([(max(a, lo), min(b, hi)) for a, b in spans if min(b, hi) > max(a, lo)])
    inside = [(max(a, x), min(b, y)) for a, b in clipped for x, y in mask
              if min(b, y) > max(a, x)]
    return _union_seconds(clipped) - _union_seconds(inside)


def _mask(members, pair, region_ids, speech, lo, hi):
    """Where a pair's numbers cannot be trusted: third voices, excluded lines, outside overlap."""
    pad = EVENT_PARAMS.ipu_gap
    spans = []
    for s in members:
        if str(s.speaker) not in pair or exclusion_reason(s) is not None:
            spans.append((s.start - pad, s.end + pad))
    for s in speech:
        if id(s) not in region_ids and s.start < hi and s.end > lo:
            spans.append((s.start - pad, s.end + pad))
    return _merge(spans)


def _measure_pair(accs, pair, pair_segs, mask, lo, s_end):
    analysis = analyze(pair_segs, EVENT_PARAMS)
    talk = [u for u in analysis.units if not u.backchannel]
    for me in pair:
        other = pair[1] if me == pair[0] else pair[0]
        accs[me].other_talk_s += _free_seconds(
            [(u.start, u.end) for u in talk if u.speaker == other], lo, s_end, mask)

    for ev in analysis.events:
        if ev["start"] > s_end or _hits(mask, ev["start"], ev["end"]):
            continue
        kind = ev["type"]
        key = (kind, ev["by"], ev["start"], ev["end"])
        if kind == "backchannel" and ev.get("in_turn"):
            accs[ev["by"]].backchannels[key] = ev
        elif kind == "interruption":
            accs[ev["by"]].interruptions[key] = ev
        if kind in ("interruption", "competitive_overlap"):
            accs[ev["of"]].cut_ins[key] = ev

    runs = []
    for u in talk:
        if runs and runs[-1][0] == u.speaker:
            runs[-1][1].append(u)
        else:
            runs.append((u.speaker, [u]))
    for k in range(1, len(runs) - 1):                 # first and last turn are cut by the pair
        speaker, units = runs[k]
        u = units[0]
        start, end = u.start, units[-1].end
        if start <= s_end and not _hits(mask, start, end):
            accs[speaker].turns[(speaker, start, end)] = end - start
        other = runs[k - 1][0]
        before = [v for v in talk if v.speaker == other and v.start <= u.start]
        if not before or u.start > s_end:
            continue
        p = max(before, key=lambda v: v.end)
        if p.end > u.end:
            continue
        mine_before = [v for v in talk if v.speaker == speaker and v.start < p.start]
        prev = max(mine_before, key=lambda v: v.start, default=None)
        if (prev is not None and prev.end - p.start >= EVENT_PARAMS.overlap_min
                and p.end <= prev.end + EVENT_PARAMS.yield_within_seconds):
            continue                                  # a short cut-in, not a turn hand-over
        if _hits(mask, p.start, u.end):
            continue
        value = u.start - p.end
        if LATENCY_FLOOR_S <= value <= LATENCY_CAP_S:
            accs[speaker].latencies[(speaker, u.start, u.end)] = value


def measure_interaction(prepared) -> Tuple[Dict[str, Accumulator], int]:
    """Per-pair events, turns and latencies over seam-free regions (spec B4.1, B4.5)."""
    accs = {str(s.speaker): Accumulator() for s in prepared}
    regions, unknown = split_regions(prepared)
    speech = [s for s in prepared if not is_empty(s)]
    for region in regions:
        members = [s for s in region if not is_empty(s)]
        speakers = sorted({str(s.speaker) for s in members})
        if len(speakers) < 2:
            continue
        lo = min(float(s.start) for s in members)
        hi = max(float(s.end) for s in members)
        s_end = hi - EDGE_MARGIN_S
        if s_end <= lo:
            continue
        ids = {id(s) for s in region}
        for pair in combinations(speakers, 2):
            pair_segs = [s for s in members if str(s.speaker) in pair]
            mask = _mask(members, pair, ids, speech, lo, hi)
            _measure_pair(accs, pair, pair_segs, mask, lo, s_end)
    return accs, unknown


LOWER_BOUND_WARNING = ("backchannel_and_interruption_are_lower_bounds: diarization drops "
                       "overlapped speech, so backchannel and interruption counts under-count")


@dataclass
class Core:
    profile: PersonaProfile
    stats: PersonaStats
    measures: Dict[str, StyleMeasure]


def base_warnings(prepared, usable_ratio: Optional[float]) -> List[str]:
    out = [LOWER_BOUND_WARNING]
    unreviewed = sum(1 for s in prepared if getattr(s, "review_flags", None) is None)
    if unreviewed:
        out.append(f"not_reviewed: {unreviewed} segment(s) have review_flags=None "
                   "and were treated as unflagged")
    unmeasured = sum(1 for s in prepared if getattr(s, "noise_breakdown", None) is None)
    if unmeasured:
        out.append(f"noise_unmeasured: {unmeasured} segment(s) have no noise measurement "
                   "and were not excluded")
    if usable_ratio is not None and usable_ratio < 0.5:
        out.append(f"low_usable_ratio: {usable_ratio}")
    return out


def _lexicon(acc: Accumulator) -> Optional[List[LexiconItem]]:
    forms = [_fold(e["text"]) for e in acc.backchannels.values()]
    if len(forms) < MIN_LEXICON_N:
        return None
    counts: Dict[str, int] = {}
    for f in forms:
        counts[f] = counts.get(f, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [LexiconItem(f, rnd(c / len(forms))) for f, c in ranked]


def _speaker_profile(speech_s: float, measure: StyleMeasure, acc: Accumulator) -> SpeakerProfile:
    exposure = acc.other_talk_s / 60.0
    bcs = list(acc.backchannels.values())
    cut_ins = list(acc.cut_ins.values())
    return SpeakerProfile(
        speech_s=rnd(speech_s),
        style=measure.style,
        backchannel=BackchannelProfile(
            per_min_of_other=make_rate(len(bcs), exposure, MIN_EXPOSURE_MIN),
            lexicon=_lexicon(acc),
            overlap_ratio=make_ratio(sum(1 for e in bcs if e["overlap"] > 0), len(bcs))),
        interruption=InterruptionProfile(
            per_hour_of_other=make_rate(len(acc.interruptions), exposure,
                                        MIN_EXPOSURE_INTERRUPTION_MIN, per=60.0),
            yield_rate=make_ratio(
                sum(1 for e in cut_ins
                    if e["type"] == "interruption" and e["response"] == "yield"),
                len(cut_ins))),
        turn=TurnProfile(duration_s=make_dist(list(acc.turns.values())),
                         response_latency_s=make_dist(list(acc.latencies.values()))))


def compute_core(prepared, recording_id: str) -> Core:
    """Everything except the two LLM-written fields (P1, P2, P4)."""
    measures = measure_style(prepared)
    accs, unknown = measure_interaction(prepared)
    speakers = sorted(measures)
    speech_s = {sp: _union_seconds([(float(s.start), float(s.end)) for s in prepared
                                    if str(s.speaker) == sp]) for sp in speakers}
    usable_s = sum(_union_seconds([(float(s.start), float(s.end)) for s in prepared
                                   if str(s.speaker) == sp and usable(s)]) for sp in speakers)
    total = sum(speech_s.values())
    ratio = rnd(usable_s / total) if total > 0 else None
    profile = PersonaProfile(
        schema_version=SCHEMA_VERSION, params_version=PARAMS_VERSION, source_type="real",
        recording_id=recording_id, usable_ratio=ratio,
        speakers={sp: _speaker_profile(speech_s[sp], measures[sp], accs[sp]) for sp in speakers})
    stats = PersonaStats(excluded_s=excluded_seconds(prepared), unknown_gaps_skipped=unknown,
                         llm_windows=0, llm_ok=0, warnings=base_warnings(prepared, ratio))
    return Core(profile, stats, measures)


_WINDOW_TOKEN_ESTIMATE = 4000          # tokens of one LLM_WINDOW_CHARS window

PERSONA_SYSTEM_PROMPT = (
    "Bạn phân tích cách nói của MỘT người trong một cuộc ghi âm tiếng Việt, "
    "chỉ dựa trên lời của người đó và các số liệu đã đo được đưa kèm.\n"
    "Trả về DUY NHẤT một đối tượng JSON: "
    '{"new_forms": [...], "description": "..."}.\n'
    "- new_forms: các từ hoặc cụm (1 đến 3 từ) là từ đệm, tiểu từ cuối câu hoặc từ xưng hô, "
    "xuất hiện NGUYÊN VĂN trong lời của người này và CHƯA có trong các danh sách đã cho. "
    "Không có thì để [].\n"
    "- description: 1 đến 2 câu tiếng Việt mô tả phong cách nói. Không bịa chi tiết ngoài lời và số liệu.")
_STRICT_JSON_SUFFIX = ("\n\nNHẮC LẠI: chỉ trả về một đối tượng JSON như trên. "
                       "Không viết phân tích trước hoặc sau JSON.")


def pick_window(texts: List[str], limit: int) -> str:
    """Texts spread evenly by start until about `limit` characters (never past it)."""
    step = 1
    while step < len(texts) and sum(len(t) + 1 for t in texts[::step]) > limit:
        step += 1
    return "\n".join(texts[::step])[:limit]


def parse_reply(raw: Optional[str]) -> Optional[dict]:
    """The reply's JSON object if it holds `new_forms` or `description`, else None."""
    if raw is None or not is_readable(raw):
        return None
    for obj in objects_in(raw):
        if "new_forms" in obj or "description" in obj:
            return obj
    return None


def validate_forms(raw_forms, texts: List[str], dictionary_forms) -> List[str]:
    """Qwen's proposals that are real, new, 1-3 whole tokens of this speaker's words."""
    if not isinstance(raw_forms, list):
        return []
    known = {tuple(fold_tokens(f)) for f in dictionary_forms}
    token_lists = [fold_tokens(t) for t in texts]
    kept = set()
    for raw in raw_forms:
        if not isinstance(raw, str):
            continue
        form = fold_tokens(raw)
        if not 1 <= len(form) <= 3 or tuple(form) in known:
            continue
        if any(count_form(tokens, form) for tokens in token_lists):
            kept.add(" ".join(form))
    return sorted(kept)[:MAX_CANDIDATE_FORMS]


def validate_description(raw, rate_n: int) -> Optional[str]:
    if not isinstance(raw, str) or rate_n < MIN_DIST_N:
        return None
    text = raw.strip()
    if not text or len(text) > DESCRIPTION_MAX_CHARS:
        return None
    return text


@dataclass
class ProfileResult:
    profile: dict
    stats: dict
    complete: bool                       # True = safe to checkpoint
    skipped: Optional[str] = None


class PersonaProfileService:
    """Compute-only persona profile of one recording; the caller checkpoints and writes."""

    def __init__(self, llm, logger=None, *, llm_enabled: bool = True,
                 thinking: bool = False, max_new_tokens: int = 1024):
        self.llm = llm
        self.logger = logger
        self.llm_enabled = bool(llm_enabled)
        self.thinking = bool(thinking)
        self.max_new_tokens = reply_budget(max_new_tokens, self.thinking)
        self.unreadable_retries = 1

    # -- contract used by the pipeline ------------------------------------
    def prepare(self, segments, timeline, noise=None) -> list:
        return prepare_segments(segments, timeline, noise)

    def source_digest(self, prepared) -> str:
        return source_digest_of(prepared)

    def checkpoint_namespace_for(self, source_digest: str) -> str:
        model = getattr(self.llm, "model_name", None) or "unknown"
        key = (f"{PROFILE_PROMPT_VERSION}-{'think' if self.thinking else 'nothink'}"
               f"-{'llm' if self.llm_enabled else 'nollm'}-{model}"
               f"-new{self.max_new_tokens}-{PARAMS_VERSION}-{source_digest}")
        return re.sub(r"[^A-Za-z0-9._-]+", "_", key)

    def profile(self, prepared, recording_id: str) -> ProfileResult:
        if not prepared:
            return ProfileResult({}, {}, False, "no_segments")
        core = compute_core(prepared, recording_id)
        complete = True
        if self.llm_enabled:
            complete = self._ask_llm(core)
        return ProfileResult(core.profile.to_dict(), core.stats.to_dict(), complete)

    # -- LLM --------------------------------------------------------------
    def _windows_per_call(self) -> int:
        batch = max(1, int(getattr(self.llm, "batch_size", 1) or 1))
        limit = int(getattr(self.llm, "max_batch_tokens", 0) or 0)
        if limit:
            batch = min(batch, max(1, limit // _WINDOW_TOKEN_ESTIMATE))
        return batch

    @staticmethod
    def _message(label: str, measure: StyleMeasure) -> str:
        rate, sent = measure.style.speech_rate_wps, measure.style.words_per_sentence
        seen = [f.form for f in measure.style.fillers or [] if f.per_min > 0]
        seen += [p.form for p in measure.style.particles or [] if p.per_100_sent > 0]
        seen += [a.form for a in measure.style.address_terms or [] if a.count > 0]
        known = ", ".join(FILLERS + PARTICLES + ADDRESS_TERMS)
        return (f"Người nói: {label}\n"
                f"Danh sách đã có (không lặp lại): {known}\n"
                f"Số liệu đã đo: nhịp {rate.p50} từ/giây, độ dài câu trung vị {sent.p50} từ, "
                f"form thấy trong lời: {', '.join(seen) or 'không có'}\n\n"
                f"Lời của người này:\n{pick_window(measure.texts, LLM_WINDOW_CHARS)}\n\n"
                "JSON:")

    def _ask(self, messages: List[str]):
        replies, unanswered = ask_in_batches(
            self.llm, PERSONA_SYSTEM_PROMPT, messages, per_call=self._windows_per_call(),
            max_new_tokens=self.max_new_tokens, label="persona window",
            logger=self.logger, thinking=self.thinking)
        pending = [i for i, r in enumerate(replies)
                   if r is not None and parse_reply(r) is None]
        for _ in range(self.unreadable_retries):
            if not pending:
                break
            retry, _ = ask_in_batches(
                self.llm, PERSONA_SYSTEM_PROMPT + _STRICT_JSON_SUFFIX,
                [messages[i] + _STRICT_JSON_SUFFIX for i in pending],
                per_call=min(self._windows_per_call(), len(pending)),
                max_new_tokens=self.max_new_tokens, label="persona JSON retry",
                logger=self.logger, thinking=False)
            still = []
            for original, reply in zip(pending, retry):
                if reply is not None:
                    replies[original] = reply
                if reply is None or parse_reply(reply) is None:
                    still.append(original)
            pending = still
        return replies

    def _ask_llm(self, core: Core) -> bool:
        """Fill llm_candidate_forms / description. True when nothing is left to retry."""
        asked = [sp for sp, m in sorted(core.measures.items())
                 if m.speech_min >= MIN_EXPOSURE_MIN]
        if not asked:
            return True
        if not self.llm.ensure_loaded():
            core.stats.warnings.append("llm_unavailable: the LLM could not be loaded")
            if self.logger:
                self.logger.warning("[persona] LLM not available; numbers only")
            return False
        replies = self._ask([self._message(sp, core.measures[sp]) for sp in asked])
        core.stats.llm_windows = len(asked)
        dictionary = FILLERS + PARTICLES + ADDRESS_TERMS
        for sp, raw in zip(asked, replies):
            parsed = parse_reply(raw)
            if parsed is None:
                continue
            core.stats.llm_ok += 1
            style = core.profile.speakers[sp].style
            style.llm_candidate_forms = validate_forms(
                parsed.get("new_forms"), core.measures[sp].texts, dictionary)
            style.description = validate_description(
                parsed.get("description"), style.speech_rate_wps.n)
        if core.stats.llm_ok == 0:
            core.stats.warnings.append("llm_unavailable: no window returned readable JSON")
        return core.stats.llm_ok == core.stats.llm_windows
