"""Flag extremely unreasonable speaker labels or ASR text, and change nothing.

One LLM pass reads the transcript the way `speaker_relabel_service` does -- the
same windows, the same lettered speakers and 1..N line numbers -- but instead of
proposing a new label it only *flags* what it cannot believe, so the conversation
finder can keep its ranges away from it:

  * `speaker_phrase`   words inside a turn that belong to the other speaker
                       (a backchannel merged into a long turn). Hard.
  * `speaker_turn`     the whole turn belongs to the other speaker. Hard.
  * `asr_hallucination` boilerplate or invented text the speaker did not say,
                       unrelated to the talk around it, that
                       `algorithms/asr/hallucination.py` did not catch. A line
                       that stays on topic (misheard, repeated, hesitant) is not
                       one. Hard.
  * `asr_garbage`      gibberish or disconnected ASR words, rather than a
                       coherent foreign-language sentence, that laughter, a
                       filler, a stutter or a repeat cannot explain. Hard.

Nothing here edits `speaker`, `text` or a time. `apply_review_flags` writes one
attribute, `review_flags`. Flagging a segment locked to overlap separation is
allowed because a flag changes no data.

The guards are the relabel ones where they carry over: a number outside 1..N or
an unknown kind is refused where it stands, a flag under the confidence floor is
dropped, and one window decides each segment. A phrase must really occur in the
segment text (compared without case or whitespace differences), otherwise the
model quoted something that is not there.

`review()` only computes, so the caller can checkpoint the decision first.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from services.speaker_relabel_service import SpeakerRelabelService, speaker_names
from utils.llm_batches import ask_in_batches, reply_budget
from utils.llm_json import clean_reply, is_cut_in_thought, is_readable, objects_in
from utils.transcript_windows import build_windows, format_line, line_number

# Part of the checkpoint namespace: a changed prompt or rule recomputes instead of
# reusing flags an older one produced.
REVIEW_PROMPT_VERSION = "review-v4"

FLAG_KINDS = ("speaker_phrase", "speaker_turn", "asr_hallucination", "asr_garbage")
# Every kind is hard: a flagged range is rejected by the selector. A reply that still
# names a retired kind (not_vietnamese) is refused as unknown_kind, with no flag.
HARD_FLAG_KINDS = ("speaker_phrase", "speaker_turn", "asr_hallucination", "asr_garbage")

_TEMPLATE_SLACK_TOKENS = 64

_STRICT_JSON_SUFFIX = (
    "\n\n### BẮT BUỘC Ở LẦN TRẢ LỜI NÀY\n"
    "Không giải thích, không phân tích thành lời, không dùng <think>, không dùng ``` . "
    "Ký tự đầu tiên phải là [ và ký tự cuối cùng phải là ]. "
    "Nếu không có gì cần gắn cờ, chỉ trả về []."
)

# The worked example names lines 901-906. No window is that long, so a model that
# copies the example's answer names lines that do not exist and is refused; the
# same answer with a real line number would be a real, false, flag.
REVIEW_SYSTEM_PROMPT = (
    "Bạn rà soát transcript của MỘT cuộc hội thoại tiếng Việt (podcast, phỏng vấn hoặc trò chuyện). "
    "Nhãn người nói do máy gán và văn bản do máy nhận dạng giọng nói (ASR) tạo ra; cả hai đôi khi sai. "
    "Bạn không nghe được audio: bạn chỉ có văn bản, mốc thời gian và nhãn máy đã gán.\n"
    "\n"
    "### NHIỆM VỤ DUY NHẤT\n"
    "Gắn cờ những chỗ CỰC KỲ bất hợp lý. Bạn KHÔNG sửa nhãn, KHÔNG sửa văn bản, KHÔNG sửa mốc thời gian, "
    "KHÔNG tách hay gộp dòng. Chỉ gắn cờ khi CỰC KỲ bất hợp lý; phân vân thì bỏ qua. "
    "Phần lớn transcript là đúng, nên thường câu trả lời là [].\n"
    "\n"
    "### ĐỊNH DẠNG MỖI DÒNG\n"
    "[số dòng] bắt_đầu-kết_thúc NGƯỜI (gap ±giây) nội dung\n"
    "- Số dòng: số trong ngoặc vuông, đếm từ 1 trong phần transcript bạn nhận. Chỉ dùng số này để chỉ một dòng.\n"
    "- NGƯỜI: một chữ cái (A, B, C...) đại diện cho một người nói. Cùng chữ là cùng một người.\n"
    "- gap: số giây từ lúc dòng trước kết thúc đến lúc dòng này bắt đầu. Gap ÂM nghĩa là hai người nói chồng lên nhau.\n"
    "- Dòng có [cố định] vẫn có thể bị gắn cờ; gắn cờ không thay đổi dữ liệu.\n"
    "\n"
    "### BỐN LOẠI CỜ (trường \"kind\")\n"
    '1. "speaker_phrase": một cụm từ NẰM TRONG lượt nói rõ ràng là của người kia, thường là lời đệm bị gộp vào '
    "lượt của người đang nói. Ví dụ: \"…điềm tĩnh đúng rồi và cẩn thận…\" thì \"đúng rồi\" là lời người nghe chen vào. "
    'BẮT BUỘC có "phrase".\n'
    '2. "speaker_turn": CẢ dòng rõ ràng là của người kia (tự hỏi rồi tự trả lời, xưng hô ngược hẳn với cả đoạn). '
    '"phrase" là null.\n'
    '3. "asr_hallucination": ASR BỊA RA nội dung mà người nói KHÔNG hề nói. Đó là một câu hoặc cụm có dạng văn mẫu '
    "(lời chào kênh, kêu gọi đăng ký/bấm chuông, cảm ơn người xem, tên kênh, đoạn phụ đề/quảng cáo) hoặc một câu trôi chảy, "
    "đủ ngữ pháp nhưng hoàn toàn lạc chủ đề so với các dòng trước và sau, chèn vào giữa câu chuyện. "
    "Dấu hiệu bắt buộc: nội dung KHÔNG liên quan mạch hội thoại xung quanh. "
    "Một dòng vẫn bám chủ đề (dù sai chính tả, nghe nhầm từ, lặp từ, ngập ngừng) thì KHÔNG phải hallucination. "
    '"phrase" là cụm bịa, hoặc null nếu cả dòng bịa.\n'
    '4. "asr_garbage": chuỗi âm tiết vô nghĩa hoặc từ rời rạc không thành câu, và KHÔNG thể giải thích được bằng tiếng cười, '
    "tiếng đệm, nói lắp hay lặp từ do ngập ngừng; đây cũng không phải là một câu tiếng nước ngoài có nghĩa. "
    "Nếu có thể hiểu đó là người nói đang cười, ngập ngừng hoặc nhắc lại cho rõ thì KHÔNG gắn cờ. "
    '"phrase" là cụm sai, hoặc null nếu cả dòng sai.\n'
    "\n"
    "### CHỈ GẮN CỜ KHI CỰC KỲ BẤT HỢP LÝ. KHÔNG GẮN CỜ:\n"
    "- Từ tiếng Anh, tên riêng, thuật ngữ hay từ mượn chen giữa câu tiếng Việt là BÌNH THƯỜNG. "
    "Một dòng chỉ có một hai từ như \"OK\", \"yeah\" cũng không gắn cờ.\n"
    "- Lỗi chính tả, dấu thanh sai, từ đồng âm, thiếu hoặc thừa dấu câu, tên riêng viết sai.\n"
    "- Câu nói lủng củng, ngập ngừng, lặp từ do người nói tự nhiên, từ địa phương, tiếng lóng.\n"
    "- Tiếng cười và âm phản ứng bằng miệng (\"ha ha\", \"hì hì\", \"hề hề\", \"ừm\", \"ờ\", \"à\", \"ồ\"), và câu bị lặp lại "
    "hoặc nói lắp vì người nói vừa nói vừa cười hay nói chưa rõ nên phải nhắc lại. Đây là lời nói THẬT, không phải ASR bịa hay rác. "
    "Lặp lại do người nói khác với bịa: hallucination là nội dung người nói không nói, còn lặp lại là họ có nói.\n"
    "- Đoạn mơ hồ, không chắc là cười, nói không rõ hay rác: bỏ qua, không gắn cờ.\n"
    "- Một người nói liền nhiều dòng (lượt nói dài) là bình thường. Một chữ chỉ xuất hiện ở vài dòng không có nghĩa là sai: "
    "đó có thể là khách mời hoặc giọng trong đoạn video chèn vào.\n"
    "- Những thứ mã đã tự lọc trước đó: danh sách câu mẫu quen thuộc (đăng ký kênh, hẹn gặp lại, bấm chuông...), "
    "chữ lặp đi lặp lại, ký tự ngoại ngữ không phải chữ Latin (Trung, Nhật, Hàn, Nga, Thái...), và quá nhiều từ so với một đoạn ngắn dưới 1 giây.\n"
    "- Đừng gắn cờ hàng loạt. Nếu muốn gắn nhiều dòng trong một đoạn, hãy xét lại: có thể bạn đang hiểu sai mạch hội thoại.\n"
    "\n"
    "### CÁCH LÀM VIỆC (giữ ngắn, càng ít suy luận càng tốt)\n"
    "- Chỉ nêu các dòng nghi ngờ. KHÔNG chép lại hay liệt kê các dòng đã ổn.\n"
    "- Mỗi dòng kết luận MỘT lần rồi thôi, KHÔNG xét lại.\n"
    "\n"
    "### ĐẦU RA\n"
    "Chỉ MỘT mảng JSON, không viết gì ngoài JSON, không dùng ```. Mỗi phần tử có đúng năm khoá:\n"
    '- "i": số dòng, là số nguyên, đúng như số trong ngoặc vuông.\n'
    '- "kind": một trong "speaker_phrase", "speaker_turn", "asr_hallucination", "asr_garbage".\n'
    '- "phrase": cụm từ chép NGUYÊN VĂN từ dòng đó (đúng từng chữ), hoặc null nếu cả dòng.\n'
    '- "conf": độ chắc chắn từ 0 đến 1. Chỉ gắn cờ khi conf từ {min_confidence} trở lên.\n'
    '- "why": lý do, tối đa 20 từ.\n'
    "Nếu không có gì bất hợp lý, xuất đúng: []\n"
    "\n"
    "### VÍ DỤ MINH HOẠ\n"
    "Số dòng và nội dung dưới đây chỉ để minh hoạ cách xét, không phải transcript của bạn.\n"
    "[901] 00:10.0-00:14.2 A (gap +1.0s) Anh làm nghề này được bao lâu rồi ạ?\n"
    "[902] 00:14.6-00:30.0 B (gap +0.4s) Một người làm nghề này phải điềm tĩnh đúng rồi và cẩn thận từng chút.\n"
    "[903] 00:30.4-00:33.1 B (gap +0.4s) Vậy điều gì giữ anh lại với nghề?\n"
    "[904] 00:33.5-00:38.0 B (gap +0.4s) Chắc là vì mình thấy mình làm được điều có ích.\n"
    "[905] 00:38.4-00:41.0 A (gap +0.4s) xờ ky pờ lanh bu đê mon tơ nhây\n"
    "[906] 00:41.4-00:47.0 A (gap +0.4s) ha ha ha thì đúng là vậy đó, đúng là vậy đó ha ha\n"
    "Dòng 902 có \"đúng rồi\" là lời đệm của người nghe bị gộp vào lượt của B. "
    "Dòng 903 là câu hỏi mang chữ B ngay trước câu trả lời của chính B, nên cả lượt là của người kia. "
    "Dòng 905 là chuỗi âm tiết không phải tiếng Việt. "
    "Dòng 906 có tiếng cười và câu nhắc lại nhưng là lời nói thật, nên KHÔNG gắn cờ. Kết quả đúng là:\n"
    '[{"i": 902, "kind": "speaker_phrase", "phrase": "đúng rồi", "conf": 0.85, "why": "lời đệm của người nghe bị gộp vào lượt nói"}, '
    '{"i": 903, "kind": "speaker_turn", "phrase": null, "conf": 0.9, "why": "câu hỏi mang chữ của người vừa trả lời"}, '
    '{"i": 905, "kind": "asr_garbage", "phrase": null, "conf": 0.9, "why": "âm tiết không phải tiếng Việt"}]'
)

_WRAPPER_KEYS = ("flags", "review", "items", "result", "issues")


def review_system_prompt(min_confidence: float = 0.7) -> str:
    """The prompt as sent, with the confidence floor the guards will apply."""
    return REVIEW_SYSTEM_PROMPT.replace("{min_confidence}", f"{min_confidence:g}")


# ---------------------------------------------------------------------------
# Parsing and phrase matching
# ---------------------------------------------------------------------------

def _clean(obj) -> Optional[dict]:
    """Keep the five keys the pass may read. Anything else is dropped."""
    if not isinstance(obj, dict) or obj.get("i") is None or obj.get("kind") is None:
        return None
    conf = obj.get("conf")
    if isinstance(conf, bool):
        conf = None
    else:
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = None
    phrase = obj.get("phrase")
    phrase = None if phrase is None else str(phrase).strip() or None
    return {
        "i": str(obj["i"]).strip().lstrip("#").strip(),
        "kind": str(obj["kind"]).strip().lower(),
        "phrase": phrase,
        "conf": conf,
        "why": str(obj.get("why") or "")[:200],
    }


def parse_flags(raw: str) -> List[dict]:
    """The flags in a model reply, each reduced to i / kind / phrase / conf / why.

    Never raises: an unreadable reply means no flags, the safe outcome.
    """
    return [c for c in (_clean(item) for item in objects_in(raw, _WRAPPER_KEYS))
            if c is not None]


def _folded(text: str) -> Tuple[str, List[int]]:
    """Case-folded text with whitespace runs collapsed, and each char's source offset."""
    out: List[str] = []
    where: List[int] = []
    in_space = True            # drops leading whitespace
    for pos, ch in enumerate(unicodedata.normalize("NFC", text)):
        if ch.isspace():
            if not in_space:
                out.append(" ")
                where.append(pos)
            in_space = True
            continue
        low = ch.lower()
        out.append(low if len(low) == 1 else ch)
        where.append(pos)
        in_space = False
    while out and out[-1] == " ":
        out.pop()
        where.pop()
    return "".join(out), where


def find_phrase(text: str, phrase: str) -> Optional[str]:
    """The stretch of `text` that `phrase` names, ignoring case and whitespace; else None.

    Returns the text as the segment wrote it, so a stored flag is always a real
    substring of `seg.text` even when the model changed the case.
    """
    norm = unicodedata.normalize("NFC", text)
    needle, _ = _folded(phrase)
    if not needle:
        return None
    hay, where = _folded(text)
    at = hay.find(needle)
    if at < 0:
        return None
    found = norm[where[at]: where[at + len(needle) - 1] + 1]
    if found in text:
        return found
    # Decomposed (NFD) text: hand back its own spelling, not the composed one.
    decomposed = unicodedata.normalize("NFD", found)
    return decomposed if decomposed in text else found


# ---------------------------------------------------------------------------
# Result and applying
# ---------------------------------------------------------------------------

@dataclass
class ReviewResult:
    flags_by_index: Dict[str, List[dict]] = field(default_factory=dict)
    windows: int = 0
    failed_windows: int = 0
    skipped: Optional[str] = None
    rejected: Dict[str, int] = field(default_factory=dict)   # reason -> count
    segments: int = 0
    thinking: bool = False
    proposed: int = 0                   # flags read, before any guard
    unreadable_windows: int = 0         # answered, but with no JSON in the reply
    cut_in_thought_windows: int = 0

    def to_report(self) -> dict:
        flagged = {i: f for i, f in self.flags_by_index.items() if f}
        by_kind: Dict[str, int] = {}
        for flags in flagged.values():
            for flag in flags:
                by_kind[flag["kind"]] = by_kind.get(flag["kind"], 0) + 1
        return {
            "prompt_version": REVIEW_PROMPT_VERSION,
            "segments": self.segments,
            "thinking": self.thinking,
            "windows": self.windows,
            "failed_windows": self.failed_windows,
            "unreadable_windows": self.unreadable_windows,
            "cut_in_thought_windows": self.cut_in_thought_windows,
            "proposed": self.proposed,
            "skipped": self.skipped,
            "reviewed": len(self.flags_by_index),
            "flagged_segments": len(flagged),
            "flags": sum(by_kind.values()),
            "by_kind": by_kind,
            "rejected": dict(self.rejected),
            "flagged": flagged,
        }


def apply_review_flags(transcripts, flags_by_index: Dict[str, List[dict]]) -> int:
    """Write `review_flags` on every reviewed segment; returns how many carry a flag.

    The only attribute touched. A reviewed segment with nothing found gets [],
    which is different from None (not reviewed). Idempotent.
    """
    flagged = 0
    for seg in transcripts:
        flags = flags_by_index.get(str(seg.index))
        if flags is None:
            continue
        seg.review_flags = [dict(f) for f in flags]
        if flags:
            flagged += 1
    return flagged


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

class TranscriptReviewService:
    """Ask the resident LLM which segments are extremely unreasonable.

    `llm` is the same surface `SpeakerRelabelService` takes (`ensure_loaded`,
    `count_tokens`, `generate_texts`, `batch_size`, `max_batch_tokens`,
    `model_name`), shared rather than reloaded.
    """

    def __init__(self, llm, logger=None, window_tokens: int = 3500,
                 overlap_segments: int = 30, max_window_segments: Optional[int] = None,
                 min_confidence: float = 0.7, thinking: bool = False,
                 max_new_tokens: Optional[int] = None):
        self.llm = llm
        self.logger = logger
        self.window_tokens = max(1, int(window_tokens))
        self.overlap_segments = max(0, int(overlap_segments))
        # Token counting in a subprocess is approximate; a segment ceiling keeps a
        # long transcript from becoming one giant prompt. Default: no ceiling
        # beyond the token budget, one segment costs at least one token.
        self.max_window_segments = (max(1, int(max_window_segments))
                                    if max_window_segments else None)
        self.min_confidence = float(min_confidence)
        self.thinking = bool(thinking)
        # Flags are sparse, but a few dozen objects with a `why` each need room.
        self.max_new_tokens = reply_budget(
            1024 if max_new_tokens is None else max_new_tokens, self.thinking)
        self.unreadable_retries = 1

    @property
    def checkpoint_namespace(self) -> str:
        """Prompt version, thinking, model and settings as a path-safe name."""
        model = getattr(self.llm, "model_name", None) or "unknown"
        mode = "think" if self.thinking else "nothink"
        seg_cap = self.max_window_segments if self.max_window_segments else "none"
        key = (
            f"{REVIEW_PROMPT_VERSION}-{mode}-{model}"
            f"-conf{self.min_confidence:g}"
            f"-win{self.window_tokens}"
            f"-seg{seg_cap}"
            f"-ov{self.overlap_segments}"
            f"-new{self.max_new_tokens}"
        )
        return re.sub(r"[^A-Za-z0-9._-]+", "_", key)

    # -- prompt ------------------------------------------------------------
    @staticmethod
    def _header(names: List[str], lines: int) -> str:
        return (f"Người nói: {', '.join(names)} (mỗi chữ là một người)\n"
                f"Số dòng: 1 đến {lines}\n\nTranscript:\n")

    _FOOTER = "\n\nDanh sách cờ (JSON):"

    def _system_prompt(self) -> str:
        return review_system_prompt(self.min_confidence)

    def _strict_system_prompt(self) -> str:
        return self._system_prompt() + _STRICT_JSON_SUFFIX

    def prompt_overhead(self, names: List[str]) -> int:
        return (self.llm.count_tokens(self._system_prompt())
                + self.llm.count_tokens(self._header(names, 999) + self._FOOTER))

    def _message(self, names: List[str], lines: List[str]) -> str:
        return self._header(names, len(lines)) + "\n".join(lines) + self._FOOTER

    def _windows_per_call(self) -> int:
        batch = max(1, int(getattr(self.llm, "batch_size", 1) or 1))
        limit = int(getattr(self.llm, "max_batch_tokens", 0) or 0)
        if limit:
            batch = min(batch, max(1, limit // (self.window_tokens + _TEMPLATE_SLACK_TOKENS)))
        return batch

    # -- the pass ----------------------------------------------------------
    def review(self, segments) -> ReviewResult:
        result = ReviewResult(segments=len(segments), thinking=self.thinking)
        if not segments:
            result.skipped = "no_segments"
            return result
        if not self.llm.ensure_loaded():
            result.skipped = "llm_unavailable"
            if self.logger:
                self.logger.warning("[review] LLM not available; nothing reviewed")
            return result

        labels = sorted({str(s.speaker) for s in segments})
        names = speaker_names(labels)
        letters = [names[label] for label in labels]
        # Locked segments are shown as such (the model can read them as context)
        # but, unlike relabel, may still be flagged.
        locked = [SpeakerRelabelService.is_locked(s) for s in segments]

        def text_of(pos: int, number: int) -> str:
            seg = segments[pos]
            return format_line(seg, number, names[str(seg.speaker)], locked[pos])

        counts = [self.llm.count_tokens(text_of(pos, pos + 1)) + 1
                  for pos in range(len(segments))]
        budget = max(1, self.window_tokens - self.prompt_overhead(letters))
        if self.max_window_segments:
            floor = max(1, (budget + self.max_window_segments - 1) // self.max_window_segments)
            counts = [max(c, floor) for c in counts]
        windows = build_windows(counts, budget, self.overlap_segments)
        result.windows = len(windows)
        if self.logger:
            largest = max((w.stop - w.start for w in windows), default=0)
            self.logger.info(f"[review] built {len(windows)} window(s); "
                             f"largest={largest} segments, token_budget={budget}")
        messages = [self._message(letters, [text_of(pos, pos - w.start + 1)
                                            for pos in range(w.start, w.stop)])
                    for w in windows]

        replies = self._ask(messages, result)

        # Every segment of a window that gave a readable answer was reviewed. Each
        # is owned by exactly one window, so a segment in an unanswered or
        # unreadable window stays absent (review_flags None, not "clean").
        flags_by_index: Dict[str, List[dict]] = {}
        for window, reply in zip(windows, replies):
            if reply is not None and is_readable(reply):
                for pos in range(window.core_start, window.core_stop):
                    flags_by_index[str(segments[pos].index)] = []

        def reject(reason: str) -> None:
            result.rejected[reason] = result.rejected.get(reason, 0) + 1

        for w_no, (window, reply) in enumerate(zip(windows, replies)):
            proposals = parse_flags(reply) if reply is not None else []
            self._record(result, segments, w_no, window, reply, len(proposals))
            for proposal in proposals:
                line = line_number(proposal["i"])
                if line is None or line > window.stop - window.start:
                    reject("unknown_index")
                    continue
                kind = proposal["kind"]
                if kind not in FLAG_KINDS:
                    reject("unknown_kind")
                    continue
                pos = window.start + line - 1
                if not window.owns(pos):
                    reject("outside_window_core")
                    continue
                conf = proposal["conf"]
                if conf is None or not 0.0 <= conf <= 1.0:
                    reject("bad_confidence")
                    continue
                if conf < self.min_confidence:
                    reject("low_confidence")
                    continue
                seg = segments[pos]
                phrase = proposal["phrase"]
                if kind == "speaker_turn":
                    phrase = None                      # the whole line by definition
                elif phrase is None:
                    if kind == "speaker_phrase":
                        reject("missing_phrase")
                        continue
                else:
                    found = find_phrase(str(seg.text or ""), phrase)
                    if found is None:
                        reject("phrase_not_found")
                        continue
                    phrase = found
                flags = flags_by_index.setdefault(str(seg.index), [])
                if any(f["kind"] == kind and f["phrase"] == phrase for f in flags):
                    continue                           # same flag twice: merged
                flags.append({"kind": kind, "phrase": phrase,
                              "conf": conf, "why": proposal["why"]})

        if self.logger:
            flagged = sum(1 for f in flags_by_index.values() if f)
            self.logger.info(
                f"[review] {flagged} of {len(segments)} segment(s) flagged, "
                f"{result.proposed} proposed, {sum(result.rejected.values())} refused "
                f"{result.rejected}, {result.failed_windows}/{result.windows} window(s) "
                f"unanswered, {result.unreadable_windows} unreadable")
        result.flags_by_index = flags_by_index
        return result

    def _record(self, result: ReviewResult, segments, w_no: int, window,
                reply: Optional[str], proposals: int) -> None:
        readable = reply is not None and is_readable(reply)
        cut = reply is not None and is_cut_in_thought(reply)
        result.proposed += proposals
        if reply is not None and not readable:
            result.unreadable_windows += 1
        if cut:
            result.cut_in_thought_windows += 1
        if self.logger:
            first, last = str(segments[window.start].index), str(segments[window.stop - 1].index)
            if reply is None:
                note = "no answer"
            elif cut:
                note = f"cut off while thinking (max_new_tokens {self.max_new_tokens}); raise it"
            else:
                shown = " ".join(clean_reply(reply).split())[:200]
                note = f"{proposals} flag(s), reply {shown!r}"
            say = self.logger.warning if reply is not None and not readable else self.logger.info
            say(f"[review] window {w_no} (#{first}-#{last}): {note}")

    def _ask(self, messages: List[str], result: ReviewResult) -> List[Optional[str]]:
        """One reply per window; prose/thinking answers are retried as JSON-only."""
        replies, unanswered = ask_in_batches(
            self.llm, self._system_prompt(), messages,
            per_call=self._windows_per_call(), max_new_tokens=self.max_new_tokens,
            label="review window", logger=self.logger, thinking=self.thinking)
        result.failed_windows += unanswered

        pending = [i for i, r in enumerate(replies) if r is not None and not is_readable(r)]
        for attempt in range(1, self.unreadable_retries + 1):
            if not pending:
                break
            if self.logger:
                self.logger.warning(f"[review] retrying {len(pending)} unreadable window(s) "
                                    f"as strict JSON (attempt {attempt}/{self.unreadable_retries})")
            retry_messages = [messages[i] + "\n\nNHẮC LẠI: chỉ trả về mảng JSON. "
                              "Không viết phân tích trước hoặc sau JSON." for i in pending]
            retry_replies, _ = ask_in_batches(
                self.llm, self._strict_system_prompt(), retry_messages,
                per_call=min(self._windows_per_call(), max(1, len(retry_messages))),
                max_new_tokens=self.max_new_tokens,
                label=f"review JSON retry {attempt}", logger=self.logger, thinking=False)
            still_bad = []
            for original, retry in zip(pending, retry_replies):
                if retry is not None:
                    replies[original] = retry
                if retry is None or not is_readable(retry):
                    still_bad.append(original)
            pending = still_bad
        return replies
