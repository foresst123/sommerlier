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

# Vietnamese is written in Latin script. Any run of CJK, Cyrillic, Thai, Hangul
# or kana is either a decoder that drifted into another language or a homoglyph
# the tokenizer preferred over its Latin twin ("câу" with a Cyrillic у).
_FOREIGN_SCRIPT = re.compile(
    "["
    "\u4e00-\u9fff"      # CJK unified
    "\u3400-\u4dbf"      # CJK extension A
    "\u3040-\u30ff"      # hiragana + katakana
    "\uac00-\ud7af"      # hangul
    "\u0400-\u04ff"      # cyrillic
    "\u0e00-\u0e7f"      # thai
    "\u0600-\u06ff"      # arabic
    "]"
)


def foreign_script_ratio(text: str) -> float:
    """Share of the non-space characters written in a non-Latin script."""
    body = [c for c in text if not c.isspace()]
    if not body:
        return 0.0
    return sum(1 for c in body if _FOREIGN_SCRIPT.match(c)) / len(body)


def strip_foreign_runs(text: str) -> str:
    """Drop stretches of non-Latin script, keeping the Vietnamese around them.

    A whole sentence in Chinese is the model translating rather than
    transcribing; a lone Cyrillic letter inside a Vietnamese word is a
    homoglyph. Both are removed, and the surrounding text survives.
    """
    cleaned = _FOREIGN_SCRIPT.sub("", text)
    # Punctuation that belonged to the removed run would otherwise be stranded.
    cleaned = re.sub(r"[，。、；：！？]+", " ", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def diacritic_ratio(text: str) -> float:
    """Share of the words carrying a Vietnamese diacritic or the letter đ.

    English and Vietnamese are both written in Latin script, so
    ``foreign_script_ratio`` reads 0.0 for a sentence that has been translated
    wholesale into English -- the one hallucination it cannot see. Diacritics
    are what actually separates the two: running Vietnamese is densely marked
    (0.7 is typical), English is exactly 0.0.

    This is a signal, not a verdict. Vietnamese typed without diacritics scores
    0.0 as well, so a caller must compare against the text it started from
    rather than thresholding this on its own.
    """
    words = _WORD.findall(text)
    if not words:
        return 0.0
    marked = sum(
        1 for w in words
        if "đ" in w.lower()
        or any(unicodedata.category(c) == "Mn"
               for c in unicodedata.normalize("NFD", w))
    )
    return marked / len(words)


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
    """Whether ``text`` is the same short phrase repeated, a decoding-loop tell.

    Three repeats is the safe threshold for a single word, because Vietnamese
    reduplicates for emphasis: "rất rất tốt", "xa xa" are ordinary speech, not
    a stuck decoder.

    A multi-word phrase is different. Nothing in natural speech says "gì hết gì
    hết" or "đúng rồi đúng rồi" with no other words around it, so a phrase of
    two or more words repeating over essentially the whole segment is counted
    at two. That covers the doubles that reached the transcripts -- the fusion
    prompt already asks the model to collapse them, which it cannot be relied
    on to do.
    """
    words = text.split()
    if len(words) < 2:
        return False

    for size in (1, 2, 3):
        # A repeated multi-word phrase is a decoding loop even at two, but a
        # single word needs the full count to stay clear of reduplication.
        needed = min_repeats if size == 1 else min(min_repeats, 2)
        if len(words) < size * needed:
            continue
        phrase = words[:size]
        repeats = 1
        for i in range(size, len(words) - size + 1, size):
            if words[i:i + size] == phrase:
                repeats += 1
            else:
                break
        if repeats >= needed and repeats * size >= len(words) * 0.8:
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

    # A transcript that is mostly another script is not this audio. Below that
    # share the text is salvageable, and strip_foreign_runs() cleans it instead.
    if foreign_script_ratio(text) > 0.30:
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

        # Salvage first: a Vietnamese sentence with a stray Cyrillic letter or a
        # short Chinese insert is worth keeping once the foreign run is gone.
        stripped = strip_foreign_runs(text)
        if stripped != text and stripped:
            if logger:
                where = f" in segment {segment_id}" if segment_id else ""
                logger.info(f"Stripped foreign-script run{where}: {text[:50]!r}")
            text = stripped

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
