"""Word-level comparison of two transcripts, aligned by time."""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple


def clean_text(text) -> str:
    """Same normalisation as tools/tinh_wer.clean_text, plus NFC.

    Lowercase, punctuation to space, whitespace collapsed; diacritics are kept.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def tokens(text) -> List[str]:
    return clean_text(text).split()


def edit_ops(ref: List[str], hyp: List[str]) -> List[list]:
    """Levenshtein alignment as [["equal","w"],["sub","ref","hyp"],["del","w"],["ins","w"]].

    Ties prefer diagonal moves, then deletions, so the result is deterministic.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops: List[list] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            if d[i][j] == d[i - 1][j - 1] + (0 if same else 1):
                ops.append(["equal", ref[i - 1]] if same else ["sub", ref[i - 1], hyp[j - 1]])
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(["del", ref[i - 1]])
            i -= 1
        else:
            ops.append(["ins", hyp[j - 1]])
            j -= 1
    ops.reverse()
    return ops


def count_ops(ops) -> Dict[str, int]:
    c = {"sub": 0, "ins": 0, "del": 0, "equal": 0}
    for op in ops:
        c[op[0]] += 1
    return c


def wer_from_counts(c: Dict[str, int]) -> Optional[float]:
    ref_n = c["sub"] + c["del"] + c["equal"]
    if ref_n == 0:
        return None
    return (c["sub"] + c["del"] + c["ins"]) / ref_n


def timed_tokens(segments: Optional[List[dict]]) -> List[Tuple[str, float, float]]:
    """(token, start, end) for a transcript, in transcript order.

    Word timestamps are used when a segment's words match its text token for
    token. When they do not (the final text is an ensemble vote, the words come
    from Whisper alone) the tokens are spread uniformly over the span of the
    words, or over the segment span when there are none.
    """
    out: List[Tuple[str, float, float]] = []
    ordered = sorted(enumerate(segments or []), key=lambda p: (p[1]["start"], p[1]["end"], p[0]))
    for _, seg in ordered:
        toks = tokens(seg.get("text"))
        if not toks:
            continue
        words = seg.get("words")
        if words and len(words) == len(toks):
            for t, w in zip(toks, words):
                out.append((t, w["start"], w["end"]))
            continue
        lo, hi = seg["start"], seg["end"]
        if words:
            lo, hi = min(w["start"] for w in words), max(w["end"] for w in words)
        step = (hi - lo) / len(toks)
        for k, t in enumerate(toks):
            out.append((t, lo + k * step, lo + (k + 1) * step))
    return out


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_tokens(frame: List[Tuple[float, float]], toks) -> List[List[str]]:
    """Give every token to the frame segment it overlaps most (nearest if none)."""
    buckets: List[List[str]] = [[] for _ in frame]
    if not frame:
        return buckets
    for t, s, e in toks:
        best, best_ov = 0, -1.0
        for i, (fs, fe) in enumerate(frame):
            ov = _overlap(s, e, fs, fe)
            if ov > best_ov:
                best, best_ov = i, ov
        if best_ov <= 0:
            mid = (s + e) / 2
            best = min(range(len(frame)),
                       key=lambda i: (max(frame[i][0] - mid, mid - frame[i][1], 0.0), i))
        buckets[best].append(t)
    return buckets


def compare_transcripts(ref_segments: Optional[List[dict]], hyp_segments: Optional[List[dict]],
                        detail: bool = True) -> Optional[dict]:
    """Align `hyp_segments` to `ref_segments` by time and count word edits.

    Returns {"wer", "ops", "segments"} (wer None when the reference has no
    words), or None when either transcript is missing. Each reference segment
    owns the hypothesis words that fall in it, so a split or merged hypothesis
    segment still lines up.
    """
    if ref_segments is None or hyp_segments is None:
        return None
    ref_sorted = sorted(enumerate(ref_segments), key=lambda p: (p[1]["start"], p[1]["end"], p[0]))
    ref_sorted = [s for _, s in ref_sorted]
    hyp_toks = timed_tokens(hyp_segments)

    if not ref_sorted:  # nothing to anchor on: every hypothesis word is an insertion
        ref_sorted = [{"start": min((s for _, s, _ in hyp_toks), default=0.0),
                       "end": max((e for _, _, e in hyp_toks), default=0.0), "text": ""}]
    buckets = assign_tokens([(s["start"], s["end"]) for s in ref_sorted], hyp_toks)

    total = {"sub": 0, "ins": 0, "del": 0, "equal": 0}
    rows = []
    for seg, hyp in zip(ref_sorted, buckets):
        ref = tokens(seg.get("text"))
        ops = edit_ops(ref, hyp)
        c = count_ops(ops)
        for k in total:
            total[k] += c[k]
        if detail:
            errs = c["sub"] + c["ins"] + c["del"]
            rows.append({"start": seg["start"], "end": seg["end"],
                         "baseline_text": " ".join(ref), "variant_text": " ".join(hyp),
                         "ops": ops, "wer": errs / max(len(ref), 1)})
    return {"wer": wer_from_counts(total), "ops": total, "segments": rows}
