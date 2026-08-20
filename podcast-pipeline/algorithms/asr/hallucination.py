"""Detecting ASR hallucinations on short or non-speech audio.

Whisper fills silence and sub-second clips with boilerplate it saw in training
data — YouTube outros, subscribe pleas, channel names. On a conversational
dataset these land exactly on the backchannels ("ừ", "vâng") that matter most,
so they have to be filtered before ROVER votes rather than after.
"""

import re
import unicodedata

# Segments shorter than this are where Whisper hallucinates most; its output is
# treated as untrusted there. Real Vietnamese backchannels are 0.2-0.6s.
SHORT_SEGMENT_SECONDS = 1.0

# Phrases that are almost never genuine speech in a podcast transcript. Matched
# case- and accent-insensitively against the whole segment.
_BOILERPLATE_PATTERNS = [
    r"hẹn gặp lại (các bạn|quý vị)",
    r"hẹn gặp lại (trong|ở) (những )?(video|clip|tập)",
    r"(hãy |xin |mời )?(các bạn )?(đăng ký|subscribe|sub)\b",
    r"like\s*,?\s*share\s*,?\s*(và\s*)?(đăng ký|subscribe)",
    r"bấm chuông thông báo",
    r"cảm ơn (các bạn|quý vị) đã (xem|theo dõi|lắng nghe)",
    r"ghiền mì gõ",
    r"video (tiếp theo|sau)",
    r"đừng quên (like|đăng ký|subscribe)",
    r"^\s*(câu hoàn chỉnh|kết quả|transcript|output)\s*[:：]",
    r"phụ đề (được )?(thực hiện|đóng góp) bởi",
    r"^\s*(hết|the end)\s*[\.!]?\s*$",
]

def _strip_accents(text: str) -> str:
    """Fold Vietnamese diacritics so a mis-accented hallucination still matches."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


# Both the accented patterns and their folded twins, so an ASR model that drops
# or mangles diacritics does not slip its boilerplate past the filter.
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _BOILERPLATE_PATTERNS]
_COMPILED_FOLDED = [
    re.compile(_strip_accents(p), re.IGNORECASE) for p in _BOILERPLATE_PATTERNS
]


def is_boilerplate(text: str) -> bool:
    """Whether ``text`` looks like canned outro/subscribe filler."""
    if not text or not text.strip():
        return False

    if any(p.search(text) for p in _COMPILED):
        return True
    folded = _strip_accents(text)
    return any(p.search(folded) for p in _COMPILED_FOLDED)


def is_repetition(text: str, min_repeats: int = 3) -> bool:
    """Whether ``text`` is the same short phrase repeated, a decoding-loop tell."""
    words = text.split()
    if len(words) < min_repeats:
        return False

    for size in (1, 2, 3):
        if len(words) < size * min_repeats:
            continue
        phrase = words[:size]
        repeats = 1
        for i in range(size, len(words) - size + 1, size):
            if words[i:i + size] == phrase:
                repeats += 1
            else:
                break
        if repeats >= min_repeats and repeats * size >= len(words) * 0.8:
            return True
    return False


def is_hallucination(text: str, duration: float = None) -> bool:
    """Whether ``text`` should be discarded as a hallucination.

    ``duration`` is the segment length in seconds; when known, it tightens the
    check because a long transcript from a very short clip cannot be real.
    """
    if not text or not text.strip():
        return False

    if is_boilerplate(text) or is_repetition(text):
        return True

    if duration and duration > 0:
        # Vietnamese speech tops out around 7-8 syllables/second; anything past
        # double that did not come from the audio.
        words = len(text.split())
        if duration < SHORT_SEGMENT_SECONDS and words > max(4, duration * 16):
            return True

    return False


def filter_short_segment_outputs(texts, duration, trusted_index=None, logger=None, segment_id=None):
    """Blank out hallucinated candidates before voting.

    ``texts`` is the per-model transcript list. Returns a new list with
    hallucinated entries replaced by "" so ROVER neither votes for them nor
    anchors its alignment on them.
    """
    cleaned = []
    for i, text in enumerate(texts):
        if trusted_index is not None and i == trusted_index:
            cleaned.append(text)
            continue

        if is_hallucination(text, duration):
            if logger:
                where = f" in segment {segment_id}" if segment_id else ""
                logger.warning(
                    f"Discarding hallucinated ASR output{where} "
                    f"({duration:.2f}s): {text[:60]!r}"
                )
            cleaned.append("")
        else:
            cleaned.append(text)
    return cleaned
