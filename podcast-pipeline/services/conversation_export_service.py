"""Export selected two-person conversation excerpts from a finished transcript.

`utils.conversation_selection` finds the candidate stretches from numbers alone: exactly
two speakers, no join, no lasting noise, one to four minutes. This service adds
the one thing numbers cannot say -- whether a stretch is a self-contained piece
of conversation -- by asking the resident refinement LLM, then writes the audio
and a JSON sidecar for each stretch it keeps.

The model is a gate, not an author. It sees the turns of one candidate, renamed
A and B and numbered 1, 2, 3..., and answers with a score and, optionally, which
lines to start and stop at. The score is for the excerpt as it was shown; the
trim is polish on something already good. So a trim that cannot be honoured --
a line that does not exist, back to front, or one that leaves too little to
keep -- is dropped and the excerpt goes on whole, marked `trim_ignored`, rather
than being thrown away for a detail the score did not depend on. A trim that can
be honoured goes back through every check the scan applies, so shortening an
excerpt cannot smuggle in what the scan would have refused.

The model refers to lines by number only: it cannot invent a timestamp. What it
is asked to copy back is deliberately small -- a line number is far easier to
copy than a zero-padded segment id -- and the prompt's examples cannot do harm if
copied (they trim nothing, and their topics are recognised and refused).

Audio is cut from the waveform the pipeline actually worked on -- music already
stripped, cuts already made -- so timestamps line up with it, and an excerpt never
holds a join (the finder does not let one in).

Each excerpt is written twice from one cut, so the two files start on the same
sample and are exactly the same length:

  audio/<id>.wav       mono, the recording of record: both voices as recorded.
  audio/<id>_2ch.wav   two channels for listening: the first speaker (A) in the
                       left ear, the second (B) in the right. Each is heard only
                       during their own turns; where they overlap, both channels
                       carry the mixture. One microphone cannot be separated by
                       gating, so this is a layout, not a separation.

Both are described by one metadata/<id>.json, and the conversation is also
written as metadata/<id>.txt, one line per turn.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from utils.conversation_selection import (
    Candidate, ConversationSelectionConfig, ConversationSelectionFinder, pick_non_overlapping, shortlist)
from utils.llm_batches import ask_in_batches
from utils.llm_json import is_readable, objects_in
from utils.transcript_windows import line_number

# Bump when the prompt or the acceptance rules change.
CONVERSATION_EXPORT_PROMPT_VERSION = "conversation-export-v2"

_TEMPLATE_SLACK_TOKENS = 64

# The two examples in the prompt. A model that copies one answers for an excerpt
# it did not read, so a verdict whose topic is one of these is refused. Neither
# trims anything, so a copied trim is harmless even before that check.
_EXAMPLE_GOOD = {"topic": "cách nấu canh chua cho người mới",
                 "reason": "mở bằng câu hỏi, kết bằng lời chốt công thức, một chủ đề",
                 "self_contained": 5, "start_line": None, "end_line": None}
_EXAMPLE_BAD = {"topic": "lời cảm ơn nhà tài trợ chương trình",
                "reason": "đọc kịch bản một chiều, không có trao đổi",
                "self_contained": 2, "start_line": None, "end_line": None}


def _topic_key(text) -> str:
    """A topic compared loosely: case, spacing and trailing punctuation ignored."""
    return " ".join(str(text or "").lower().split()).strip(" .,;:!?\"'")


_EXAMPLE_TOPICS = {_topic_key(_EXAMPLE_GOOD["topic"]), _topic_key(_EXAMPLE_BAD["topic"])}

CONVERSATION_EXPORT_SYSTEM_PROMPT = (
    "Bạn kiểm duyệt dữ liệu hội thoại tiếng Việt. Các đoạn hội thoại giữa hai người (A và B) được cắt ra từ podcast "
    "và phỏng vấn dài để làm dữ liệu huấn luyện. Máy đã chọn sẵn đoạn bạn nhận bằng số liệu (đúng hai người nói, độ dài, "
    "độ sạch âm thanh). Việc của bạn là phần máy không làm được: đọc nội dung và cho biết đoạn này có đáng giữ làm MỘT mẫu "
    "hội thoại hoàn chỉnh, có ý nghĩa hay không.\n"
    "Bạn chỉ có văn bản, không nghe được audio. Văn bản do máy nhận dạng giọng nói tạo ra nên có thể sai chính tả hoặc "
    "lệch vài từ; đừng trừ điểm vì lỗi nhỏ đó.\n"
    "\n"
    "### ĐỊNH DẠNG ĐẦU VÀO\n"
    "[số dòng] mốc_thời_gian NGƯỜI: nội dung\n"
    "- Số dòng: số trong ngoặc vuông, đếm từ 1 ở dòng đầu tiên của đoạn.\n"
    "- Mốc thời gian (phút:giây): lúc dòng đó bắt đầu, tính từ đầu đoạn.\n"
    "- NGƯỜI: A hoặc B. Nhãn chỉ để phân biệt hai người trong đoạn này.\n"
    "\n"
    "### ĐIỀU BẠN ĐÁNH GIÁ\n"
    "Một đoạn TỰ TRỌN NGHĨA khi người nghe không biết gì về phần trước và phần sau vẫn theo dõi được:\n"
    "1. Mở đầu tự nhiên: dòng đầu là lời chào, câu hỏi, lời dẫn hoặc mở ra một ý mới. Không bắt đầu giữa chừng một câu trả lời "
    "hay một câu chuyện đang dở (dấu hiệu: mở đầu bằng \"và\", \"nhưng\", \"cho nên\", \"vậy là\", hoặc \"nó\", \"họ\", "
    "\"cái đó\" mà không rõ chỉ ai).\n"
    "2. Kết thúc tự nhiên: ý được khép lại, hoặc một lượt hỏi – đáp trọn vẹn. Không dừng ngang câu, ngang một liệt kê, "
    "hay ngay trước khi người kia kịp trả lời.\n"
    "3. Một mạch: xoay quanh một chủ đề hoặc một câu chuyện. Chuyển sang chủ đề khác giữa chừng là lan man.\n"
    "4. Có nội dung: có ý, kiến thức, câu chuyện hoặc trao đổi thật. Đoạn chỉ gồm chào hỏi, cảm ơn, tạm biệt, giới thiệu "
    "chương trình, hoặc đọc lời tài trợ / quảng cáo theo kịch bản một chiều thì KHÔNG phải một cuộc hội thoại có ý nghĩa.\n"
    "Một người nói dài còn người kia chỉ hỏi hoặc đệm \"ừ\", \"dạ\" vẫn là hội thoại hợp lệ nếu ý trọn vẹn: "
    "đừng trừ điểm chỉ vì một người nói nhiều hơn. Cũng đừng chấm cao chỉ vì đoạn dài hoặc hai người thay phiên nhau nhiều.\n"
    "\n"
    "### THANG ĐIỂM self_contained (chấm cho đoạn NGUYÊN VẸN như bạn nhận)\n"
    "5 = trọn vẹn: mở và kết tự nhiên, một chủ đề rõ, có nội dung.\n"
    "4 = tốt: ý rõ, chỉ hơi thiếu hoặc thừa một chút ở đầu hoặc đuôi.\n"
    "3 = tạm: theo dõi được nhưng bị cụt ở đầu hoặc đuôi, hoặc chủ đề mờ.\n"
    "2 = kém: bị cắt ngang giữa ý, lan man nhiều chủ đề, hoặc gần như không có nội dung (chào hỏi, quảng cáo, đọc kịch bản).\n"
    "1 = không hiểu được nếu tách khỏi ngữ cảnh.\n"
    "Đoạn chỉ được giữ khi bạn chấm từ {min_semantic} trở lên. Nếu còn nghi ngờ giữa hai mức, chọn mức thấp hơn.\n"
    "\n"
    "### CẮT BỚT (tuỳ chọn, chỉ khi thật cần)\n"
    "Nếu đoạn đã tốt nhưng vài dòng ở ĐẦU hoặc ĐUÔI thừa hoặc dở dang, bạn có thể đề nghị chỉ giữ từ dòng start_line "
    "đến dòng end_line.\n"
    "- start_line và end_line là số dòng (số nguyên, như số trong ngoặc vuông), lấy từ chính đoạn bạn nhận.\n"
    "- Chỉ cắt ở đầu hoặc đuôi, không cắt giữa đoạn, và chỉ bỏ vài dòng. Sau khi cắt phải còn ít nhất {min_seconds} giây "
    "(nhìn mốc thời gian) và vẫn có cả A lẫn B.\n"
    "- Không cần cắt: để null cả hai. Chỉ cần cắt một đầu: đầu còn lại để null.\n"
    "- Không chắc thì KHÔNG cắt. Cắt sai còn tệ hơn để nguyên.\n"
    "\n"
    "### ĐẦU RA\n"
    "Chỉ MỘT đối tượng JSON, không viết gì ngoài JSON, không dùng ```. Năm khoá, theo đúng thứ tự này:\n"
    '- "topic": chủ đề thật của đoạn, tiếng Việt, tối đa 10 từ, lấy từ nội dung bạn vừa đọc.\n'
    '- "reason": vì sao bạn chấm như vậy, tối đa 20 từ, nói về phần mở đầu và phần kết thúc.\n'
    '- "self_contained": số nguyên từ 1 đến 5.\n'
    '- "start_line": số dòng đầu muốn giữ, hoặc null.\n'
    '- "end_line": số dòng cuối muốn giữ, hoặc null.\n'
    "Viết topic và reason trước, rồi mới chấm điểm.\n"
    "\n"
    "### HAI VÍ DỤ VỀ ĐỊNH DẠNG\n"
    "Nội dung dưới đây chỉ để minh hoạ định dạng và cách chấm, không phải đoạn của bạn. Đừng chép lại.\n"
    "Đoạn hỏi – đáp trọn vẹn về một việc cụ thể:\n"
    "{example_good}\n"
    "Đoạn chỉ là lời cảm ơn nhà tài trợ đọc theo kịch bản:\n"
    "{example_bad}"
)


# ---------------------------------------------------------------------------
# The model's answer
# ---------------------------------------------------------------------------

def _trim_line(value):
    """(line, garbled): the line a trim field names, and whether it named nothing usable.

    Absent, null and empty are not a trim, and not garbled either.
    """
    if value is None or str(value).strip().lower() in ("", "null", "none"):
        return None, False
    line = line_number(value)
    return line, line is None


def parse_verdict(raw: Optional[str]) -> Optional[dict]:
    """The model's verdict from one reply, reduced to the keys this pass reads.

    `self_contained` is an int in 1..5 or None when it was missing or unusable.
    `start_line` / `end_line` are 1-based line numbers or None; `trim_garbled` is
    True when either was given but is no line number at all ("đầu", 0, -2), which
    is a trim that cannot be honoured, not the absence of one. Returns None when
    the reply holds no JSON object at all. Any other key the model adds is
    dropped.
    """
    objects = objects_in(raw)
    if not objects:
        return None
    obj = objects[0]
    score = obj.get("self_contained")
    if isinstance(score, bool):
        score = None
    else:
        try:
            score = int(round(float(score)))
        except (TypeError, ValueError):
            score = None
    if score is not None and not 1 <= score <= 5:
        score = None

    start, start_garbled = _trim_line(obj.get("start_line"))
    end, end_garbled = _trim_line(obj.get("end_line"))
    return {"self_contained": score,
            "topic": str(obj.get("topic") or "").strip()[:120],
            "reason": str(obj.get("reason") or "").strip()[:240],
            "start_line": start, "end_line": end,
            "trim_garbled": start_garbled or end_garbled}


# ---------------------------------------------------------------------------
# Writing audio
# ---------------------------------------------------------------------------

def _snap_start(wave: np.ndarray, at: int, limit: int) -> int:
    """The nearest zero crossing at or after `at`, within `limit` samples.

    Inward only: the padded span was clamped so it does not reach into a
    neighbouring turn, and moving outward could undo that.
    """
    seg = wave[at:at + limit + 1]
    if len(seg) < 2:
        return at
    flips = np.flatnonzero(np.signbit(seg[:-1]) != np.signbit(seg[1:]))
    if not len(flips):
        return at
    k = int(flips[0])
    return at + (k if abs(seg[k]) <= abs(seg[k + 1]) else k + 1)


def _snap_stop(wave: np.ndarray, at: int, limit: int) -> int:
    """The nearest zero crossing at or before `at` (exclusive end), inward only."""
    lo = max(0, at - limit - 1)
    seg = wave[lo:at]
    if len(seg) < 2:
        return at
    flips = np.flatnonzero(np.signbit(seg[:-1]) != np.signbit(seg[1:]))
    if not len(flips):
        return at
    k = int(flips[-1])
    return lo + (k + 1 if abs(seg[k + 1]) <= abs(seg[k]) else k) + 1


def cut_bounds(waveform: np.ndarray, sample_rate: int, pad_start: float,
               pad_end: float, zero_cross_ms: float):
    """(lo, hi) sample indexes of the clip, after snapping to zero crossings.

    Worked out once and shared by the mono and the two-channel file, so the two
    start on the same sample and have exactly the same length.
    """
    lo = max(0, int(round(pad_start * sample_rate)))
    hi = min(len(waveform), int(round(pad_end * sample_rate)))
    if hi <= lo:
        return lo, lo
    limit = int(zero_cross_ms / 1000.0 * sample_rate)
    if limit > 0:
        lo = _snap_start(waveform, lo, limit)
        hi = max(lo + 1, _snap_stop(waveform, hi, limit))
    return lo, hi


def fade_edges(clip: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    """Fade the first and last `fade_ms` of `clip` (mono or (n, channels)) in place."""
    fade = min(int(fade_ms / 1000.0 * sample_rate), len(clip) // 2)
    if fade > 0:
        ramp_in = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        ramp_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        if clip.ndim == 2:
            ramp_in, ramp_out = ramp_in[:, None], ramp_out[:, None]
        clip[:fade] *= ramp_in
        clip[-fade:] *= ramp_out
    return clip


def cut_excerpt(waveform: np.ndarray, sample_rate: int, pad_start: float,
             pad_end: float, fade_ms: float, zero_cross_ms: float) -> np.ndarray:
    """A faded copy of `waveform` between the two times. The source is untouched."""
    lo, hi = cut_bounds(waveform, sample_rate, pad_start, pad_end, zero_cross_ms)
    if hi <= lo:
        return np.zeros(0, dtype=np.float32)
    clip = np.array(waveform[lo:hi], dtype=np.float32, copy=True)
    return fade_edges(clip, sample_rate, fade_ms)


def _room(edge: float, others, before: bool, limit: float) -> float:
    """Seconds a turn's channel may stay open past `edge` without entering another
    speaker's turn. 0 when that turn is already overlapped there."""
    room = limit
    for start, end in others:
        if before:
            if start < edge:
                if end > edge:
                    return 0.0
                room = min(room, edge - end)
        else:
            if end > edge:
                if start < edge:
                    return 0.0
                room = min(room, start - edge)
    return max(0.0, room)


def _gate(length: int, spans, others, offset: int, sample_rate: int,
          margin_s: float, fade: int) -> np.ndarray:
    """Gain over the clip: 1 during this speaker's turns, 0 elsewhere, with short ramps."""
    gain = np.zeros(length, dtype=np.float32)
    for start, end in spans:
        lo = round((start - _room(start, others, True, margin_s)) * sample_rate) - offset
        hi = round((end + _room(end, others, False, margin_s)) * sample_rate) - offset
        lo, hi = max(0, lo), min(length, hi)
        if hi <= lo:
            continue
        n = hi - lo
        idx = np.arange(n)
        shape = np.clip(np.minimum(idx + 1, n - idx) / max(1, fade), 0.0, 1.0)
        gain[lo:hi] = np.maximum(gain[lo:hi], shape.astype(np.float32))
    return gain


def gate_channels(mixture: np.ndarray, sample_rate: int, offset: int, left_spans,
                  right_spans, margin_ms: float, fade_ms: float) -> np.ndarray:
    """A two-channel copy of `mixture`: each speaker heard only in their own ear.

    `left_spans` / `right_spans` are the (start, end) seconds of each speaker's
    turns, in the same timeline as `offset` (the sample index `mixture[0]` sits at
    in the source). A channel is open during its speaker's turns, a little wider
    (`margin_ms`, never into the other speaker's turn) so a breath before a word
    is not chopped, and shut elsewhere, with `fade_ms` ramps so the gate cannot
    click. Nothing is rescaled, so where only one person speaks the channel is
    the recording sample for sample.

    This is a listening layout over one microphone, not a separation: where the
    two speak at once both channels are open and carry the same mixture.
    """
    n = len(mixture)
    margin = margin_ms / 1000.0
    fade = int(fade_ms / 1000.0 * sample_rate)
    left = _gate(n, left_spans, right_spans, offset, sample_rate, margin, fade)
    right = _gate(n, right_spans, left_spans, offset, sample_rate, margin, fade)
    mix = np.asarray(mixture, dtype=np.float32)
    return np.stack([mix * left, mix * right], axis=1)


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

@dataclass
class ConversationExportRun:
    exports: List[dict] = field(default_factory=list)  # one summary row per item
    report: dict = field(default_factory=dict)


@dataclass
class _Judged:
    original: Candidate
    candidate: Optional[Candidate] = None     # None when rejected
    semantic: Optional[int] = None
    topic: str = ""
    trimmed: bool = False
    reason: Optional[str] = None              # why it was rejected
    why: str = ""                             # the model's own reason for its score
    trim_ignored: Optional[str] = None        # a trim that could not be honoured, and why


class ConversationExportService:
    """Judge candidates with the resident LLM, then cut and describe the keepers.

    `llm` is the refinement service (or anything with the same surface); it is
    shared, so this pass adds no VRAM. Settings come from `models.conversation_selection`
    and are validated by `ConversationSelectionConfig`: an unknown key raises here.
    """

    def __init__(self, llm, logger=None, **settings):
        self.llm = llm
        self.logger = logger
        self.cfg = ConversationSelectionConfig.from_settings(settings)

    # -- judging -----------------------------------------------------------------
    @staticmethod
    def _names(cand: Candidate) -> Dict[str, str]:
        return {cand.speakers[0]: "A", cand.speakers[1]: "B"}

    def _system_prompt(self) -> str:
        return (CONVERSATION_EXPORT_SYSTEM_PROMPT
                .replace("{min_semantic}", str(self.cfg.min_semantic))
                .replace("{min_seconds}", f"{self.cfg.min_seconds:g}")
                .replace("{example_good}", json.dumps(_EXAMPLE_GOOD, ensure_ascii=False))
                .replace("{example_bad}", json.dumps(_EXAMPLE_BAD, ensure_ascii=False)))

    def _message(self, finder: ConversationSelectionFinder, cand: Candidate) -> str:
        """The candidate as the model reads it: numbered lines, A and B, time from the start."""
        names = self._names(cand)
        lines = []
        for number, pos in enumerate(range(cand.first, cand.last + 1), start=1):
            seg = finder.segs[pos]
            text = " ".join(str(seg.text or "").split())
            t = max(0.0, float(seg.start) - cand.start)
            lines.append(f"[{number}] {int(t // 60)}:{int(t % 60):02d} "
                         f"{names.get(seg.speaker, '?')}: {text}")
        return (f"Đoạn hội thoại giữa A và B (dài {cand.duration:.0f} giây, {len(lines)} dòng):\n"
                + "\n".join(lines) + "\n\nĐánh giá (JSON):")

    def _per_call(self, messages: List[str]) -> int:
        batch = max(1, int(getattr(self.llm, "batch_size", 1) or 1))
        limit = int(getattr(self.llm, "max_batch_tokens", 0) or 0)
        if limit and messages:
            longest = max(self.llm.count_tokens(m) for m in messages)
            longest += self.llm.count_tokens(self._system_prompt()) + _TEMPLATE_SLACK_TOKENS
            batch = min(batch, max(1, limit // longest))
        return batch

    def _verdict_for(self, finder, cand, verdict) -> _Judged:
        cfg = self.cfg
        judged = _Judged(original=cand)
        copied = verdict is not None and _topic_key(verdict["topic"]) in _EXAMPLE_TOPICS
        if verdict is None or verdict["self_contained"] is None or copied:
            if cfg.require_semantic:
                # A verdict whose topic is the prompt's example was written for
                # some other excerpt; it says nothing about this one.
                judged.reason = "copied_example" if copied else "no_verdict"
                return judged
            judged.candidate = cand
            return judged
        judged.semantic, judged.topic = verdict["self_contained"], verdict["topic"]
        judged.why = verdict["reason"]
        if judged.semantic < cfg.min_semantic:
            judged.reason = "not_self_contained"
            return judged

        # The score stands for the excerpt as shown. A trim is optional polish, so
        # one that cannot be honoured is dropped and the excerpt goes on whole.
        first, last = cand.first, cand.last
        lines = last - first + 1
        start, end = verdict["start_line"], verdict["end_line"]
        if verdict["trim_garbled"]:
            judged.trim_ignored = "trim_unreadable"
        elif start is not None or end is not None:
            new_first = first if start is None else first + start - 1
            new_last = last if end is None else first + end - 1
            in_range = ((start is None or start <= lines) and (end is None or end <= lines))
            if not in_range or new_first > new_last:
                judged.trim_ignored = "trim_out_of_range"
            elif (new_first, new_last) != (first, last):
                trimmed, why = finder.evaluate(new_first, new_last)
                if trimmed is None:
                    judged.trim_ignored = f"trim_{why}"
                else:
                    judged.candidate, judged.trimmed = trimmed, True
                    return judged
        judged.candidate = cand
        return judged

    def _judge(self, finder, shortlisted: List[Candidate], report: dict) -> List[_Judged]:
        verdicts: List[Optional[dict]] = [None] * len(shortlisted)
        replies: List[Optional[str]] = [None] * len(shortlisted)
        if self.llm.ensure_loaded():
            messages = [self._message(finder, c) for c in shortlisted]
            replies, unanswered = ask_in_batches(
                self.llm, self._system_prompt(), messages,
                per_call=self._per_call(messages), max_new_tokens=256,
                label="conversation export candidate", logger=self.logger)
            report["unanswered"] = unanswered
            verdicts = [parse_verdict(r) if r is not None else None for r in replies]
        else:
            report["llm_available"] = False
            if self.logger:
                self.logger.warning(
                    "[conversation-exports] LLM not available for the semantic check")
        judged = [self._verdict_for(finder, c, v) for c, v in zip(shortlisted, verdicts)]
        self._record(finder, report, shortlisted, replies, verdicts, judged)
        return judged

    def _record(self, finder, report: dict, shortlisted: List[Candidate],
                replies: List[Optional[str]], verdicts: List[Optional[dict]],
                judged: List[_Judged]) -> None:
        """Keep each reply as the model wrote it beside what was made of it.

        A verdict that is refused (`copied_example`, `not_self_contained`) only shows up
        as a count; whether the model copied the example, invented an index or
        wrote prose is only visible in the reply itself.
        """
        rows = []
        for cand, raw, verdict, item in zip(shortlisted, replies, verdicts, judged):
            first, last = str(finder.segs[cand.first].index), str(finder.segs[cand.last].index)
            row = {"first_index": first, "last_index": last,
                   "start": round(cand.start, 3), "end": round(cand.end, 3),
                   "score": cand.score, "answered": raw is not None,
                   "readable": raw is not None and is_readable(raw),
                   "raw": raw, "verdict": verdict,
                   "outcome": item.reason or "accepted", "trimmed": item.trimmed,
                   "trim_ignored": item.trim_ignored}
            rows.append(row)
            if self.logger:
                shown = " ".join((raw or "").split())[:160]
                self.logger.info(
                    f"[conversation-exports] candidate #{first}-#{last} "
                    f"({row['start']:.0f}-{row['end']:.0f}s): {row['outcome']}; "
                    + ("no answer" if raw is None else f"reply {shown!r}"))
        report["replies"] = rows
        report["unreadable"] = sum(1 for r in rows if r["answered"] and not r["readable"])
        ignored: Dict[str, int] = {}
        for item in judged:
            if item.trim_ignored:
                ignored[item.trim_ignored] = ignored.get(item.trim_ignored, 0) + 1
        report["trims_ignored"] = ignored

    # -- two-channel file ----------------------------------------------------------
    def _two_channel(self, finder, cand: Candidate, raw: np.ndarray,
                     sample_rate: int, offset: int) -> np.ndarray:
        """Speaker A in the left ear, speaker B in the right, over the same samples."""
        names = self._names(cand)
        left, right = [], []
        for pos in range(cand.first, cand.last + 1):
            seg = finder.segs[pos]
            span = (float(seg.start), float(seg.end))
            (left if names.get(seg.speaker) == "A" else right).append(span)
        return gate_channels(raw, sample_rate, offset, left, right,
                             self.cfg.gate_margin_ms, self.cfg.gate_fade_ms)

    @staticmethod
    def _transcript_text(meta: dict) -> str:
        """The conversation as plain text, one line per turn, timed from item start."""
        lines = []
        for row in meta["conversation"]:
            t = max(0.0, float(row["start"]))
            lines.append(f"[{int(t // 60):02d}:{t % 60:04.1f}] {row['speaker']}: {row['text']}")
        return "\n".join(lines) + "\n"

    # -- metadata ------------------------------------------------------------------
    def _metadata(self, finder, timeline, export_id, cand: Candidate, judged: _Judged,
                  audio_seconds: float, audio_start: Optional[float] = None,
                  audio_end: Optional[float] = None) -> dict:
        # cut_bounds may move each edge inward by a few milliseconds to the
        # nearest zero crossing. All item-relative timestamps must use the
        # sample actually written, not the requested padding edge.
        origin = cand.pad_start if audio_start is None else float(audio_start)
        source_end = cand.pad_end if audio_end is None else float(audio_end)
        names = self._names(cand)
        conversation = []
        for pos in range(cand.first, cand.last + 1):
            seg = finder.segs[pos]
            row = {"index": seg.index,
                   "speaker": names.get(seg.speaker, "?"),
                   "speaker_id": seg.speaker,
                   "start": round(float(seg.start) - origin, 3),
                   "end": round(float(seg.end) - origin, 3),
                   "text": seg.text,
                   "state": finder.state[pos]}
            if getattr(seg, "words", None):
                row["words"] = []
                for word in seg.words:
                    timed = dict(word)
                    if "start" in timed:
                        timed["start"] = round(
                            float(timed["start"]) - origin, 3)
                    if "end" in timed:
                        timed["end"] = round(
                            float(timed["end"]) - origin, 3)
                    row["words"].append(timed)
            if getattr(seg, "speaker_original", None) is not None:
                row["speaker_original"] = seg.speaker_original
            conversation.append(row)
        m = cand.metrics
        out = {
            "id": export_id,
            "audio": f"audio/{export_id}.wav",
            "start": round(cand.start - origin, 3),
            "end": round(cand.end - origin, 3),
            "duration": round(audio_seconds, 3),
            "source_start": round(origin, 3),
            "source_end": round(source_end, 3),
            "orig_spans": [{"start": round(a, 3), "end": round(b, 3)}
                           for a, b in timeline.spans_to_original(origin, source_end)],
            "speakers": ["A", "B"],
            "speaker_ids": {"A": cand.speakers[0], "B": cand.speakers[1]},
            "turn_count": m["turn_count"],
            "speaker_balance": m["speaker_balance"],
            "switch_count": m["switch_count"],
            "overlap_count": m["overlap_count"],
            "backchannel_count": m["backchannel_count"],
            "pacing_wpm": m["pacing_wpm"],
            "turn_latency_avg": m["turn_latency_avg"],
            "score": cand.score,
            "tier": cand.tier,
            "components": cand.components,
            "topic": judged.topic,
            "semantic_score": judged.semantic,
            "semantic_reason": judged.why,
            "trimmed_by_model": judged.trimmed,
            "trim_ignored": judged.trim_ignored,
            "noise": cand.noise,
            "music_patched_share": cand.music_patched_share,
            "conversation": conversation,
            "transcript": f"metadata/{export_id}.txt",
        }
        if self.cfg.stereo:
            out["audio_2ch"] = f"audio/{export_id}_2ch.wav"
            out["channels_2ch"] = {
                "left": "A", "right": "B", "method": "time_gated",
                "note": ("Same samples and length as the mono file. Each speaker is "
                         "heard only during their own turns; where the two speak at "
                         "once both channels carry the same mixture."),
            }
        return out

    # -- the pass ---------------------------------------------------------------------
    def run(self, transcripts, *, timeline, noise, music_map, waveform,
            sample_rate: int, out_dir: str, base_name: str) -> ConversationExportRun:
        """Find, judge and write selected conversation excerpts of one recording."""
        cfg = self.cfg
        finder = ConversationSelectionFinder(transcripts, timeline, noise, music_map, cfg)
        result = ConversationExportRun()
        report = {"prompt_version": CONVERSATION_EXPORT_PROMPT_VERSION, "skipped": None,
                  "candidates": 0, "shortlisted": 0, "judged_rejected": {},
                  "accepted": 0, "exported": 0}
        result.report = report

        candidates = finder.candidates()
        report["finder"] = finder.report()
        report["candidates"] = len(candidates)
        if not candidates:
            report["skipped"] = ("noise_not_measured"
                                 if cfg.require_noise and not finder.noise_measured
                                 else "no_candidates")
            self._log(f"[conversation-exports] {base_name}: no eligible item ({report['skipped']})")
            return result

        chosen = shortlist(candidates, cfg.max_candidates, cfg.shortlist_overlap)
        report["shortlisted"] = len(chosen)
        judged = self._judge(finder, chosen, report)
        if report.get("llm_available") is False and cfg.require_semantic:
            report["skipped"] = "llm_unavailable"
            return result

        rejected: Dict[str, int] = {}
        kept: Dict[int, _Judged] = {}
        for item in judged:
            if item.candidate is None:
                rejected[item.reason] = rejected.get(item.reason, 0) + 1
            else:
                kept[id(item.candidate)] = item
        report["judged_rejected"] = rejected
        report["accepted"] = len(kept)

        final = pick_non_overlapping([j.candidate for j in kept.values()])
        if waveform is None:
            report["skipped"] = "no_audio"
            return result

        import soundfile as sf
        audio_dir = os.path.join(out_dir, "audio")
        meta_dir = os.path.join(out_dir, "metadata")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)

        for number, cand in enumerate(final, start=1):
            item = kept[id(cand)]
            export_id = f"{re.sub(r'[^A-Za-z0-9._-]+', '_', base_name)}_conversation_{number:06d}"
            lo, hi = cut_bounds(waveform, sample_rate, cand.pad_start, cand.pad_end,
                                cfg.zero_cross_ms)
            if hi - lo < sample_rate:
                self._log(f"[conversation-exports] {export_id}: audio shorter than a second; skipped")
                continue
            # One cut, two files: both start on the same sample and are exactly the
            # same length, so they can be played against each other.
            raw = np.array(waveform[lo:hi], dtype=np.float32, copy=True)
            mono = fade_edges(raw.copy(), sample_rate, cfg.fade_ms)
            try:
                sf.write(os.path.join(audio_dir, f"{export_id}.wav"), mono,
                         sample_rate, subtype="PCM_16")
                if cfg.stereo:
                    stereo = fade_edges(self._two_channel(finder, cand, raw, sample_rate, lo),
                                        sample_rate, cfg.fade_ms)
                    sf.write(os.path.join(audio_dir, f"{export_id}_2ch.wav"), stereo,
                             sample_rate, subtype="PCM_16")
                meta = self._metadata(finder, finder.timeline, export_id, cand,
                                      item, len(mono) / float(sample_rate),
                                      audio_start=lo / float(sample_rate),
                                      audio_end=hi / float(sample_rate))
                with open(os.path.join(meta_dir, f"{export_id}.json"), "w",
                          encoding="utf-8") as fh:
                    json.dump(meta, fh, ensure_ascii=False, indent=2)
                with open(os.path.join(meta_dir, f"{export_id}.txt"), "w",
                          encoding="utf-8") as fh:
                    fh.write(self._transcript_text(meta))
            except Exception as exc:                       # pragma: no cover - disk problems
                if self.logger:
                    self.logger.warning(f"[conversation-exports] could not write {export_id}: {exc}")
                continue
            result.exports.append({
                "id": export_id, "tier": cand.tier, "score": cand.score,
                "audio": meta["audio"], "audio_2ch": meta.get("audio_2ch"),
                "duration": meta["duration"], "topic": item.topic,
                "semantic_score": item.semantic, "trimmed_by_model": item.trimmed,
                "source_start": meta["source_start"], "source_end": meta["source_end"],
                "speaker_ids": meta["speaker_ids"],
                "noise_kind": (cand.noise or {}).get("dominant_kind"),
            })
        report["exported"] = len(result.exports)
        self._log(
            f"[conversation-exports] {base_name}: {len(result.exports)} item(s) written to {out_dir}")
        return result

    def _log(self, message: str):
        if self.logger:
            self.logger.info(message)
