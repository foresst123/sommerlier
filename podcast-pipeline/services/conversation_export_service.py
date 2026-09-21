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

Each excerpt gets a folder of its own, filed by quality, and everything in it is
cut from the same samples so the files can be laid against one another:

  tier_1_overlap_good/ the two spoke at once and the separator pulled the voices
                       apart cleanly: each ear holds one voice.
  tier_2_S/            no overlap (or `overlap_first` off), filed by score:
  tier_3_A/            S from 85, A from 70, B from 55, C from 40.
  tier_4_B/
  tier_6_overlap_bad/  the two spoke at once and the separation is not to be
                       trusted (failed, low similarity, part left unseparated, or
                       no separated tracks at all); conversation.json says why.
                       Inside a folder the best score comes first.
    conversation_1/
      mixture.wav          mono, the recording of record: both voices as recorded.
      speaker_A.wav        one channel, one person: A alone (the separated track).
      speaker_B.wav        one channel, one person: B alone.
      stereo_2ch.wav       A in the left ear, B in the right: exactly the two files
                           above, sample for sample.
      conversation.json    the transcript with times from the start of these files,
                           who is who, scores, noise, the overlap, how each file was
                           made, and the checks that were run on them.
      transcript.txt       the same conversation, one line per turn.

The speaker files come from the separation stage's speaker tracks, so where the two
spoke at once each is heard alone. A stretch whose separation failed is silent in
both instead of carrying the other speaker's voice. Without those tracks (no
separation output at hand, or a track that comes out silent) they fall back to
gating the mixture -- each speaker heard only during their own turns, both voices
in both ears where they overlap -- and conversation.json says which.

After writing, every file is read back: same length, same rate, mono or stereo as
named, and the two stereo channels equal to the two speaker files. The result and
a SHA-256 of each file go into conversation.json, so the alignment can be checked
rather than trusted.
"""

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from utils.conversation_selection import (
    TIERS, Candidate, ConversationSelectionConfig, ConversationSelectionFinder,
    pick_non_overlapping, shortlist)
from utils.llm_batches import ask_in_batches, reply_budget
from utils.llm_json import clean_reply, is_cut_in_thought, is_readable, objects_in
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

# The files of one conversation folder.
MIXTURE_FILE = "mixture.wav"
SPEAKER_FILES = {"A": "speaker_A.wav", "B": "speaker_B.wav"}
STEREO_FILE = "stereo_2ch.wav"
JSON_FILE = "conversation.json"
TEXT_FILE = "transcript.txt"


OVERLAP_GOOD = "overlap_good"
OVERLAP_BAD = "overlap_bad"
# Best folder first: the clean overlaps, then the tiers by score, then the doubtful overlaps.
FOLDER_ORDER = (OVERLAP_GOOD,) + tuple(name for _floor, name in TIERS) + (OVERLAP_BAD,)


def tier_folder(tier: str) -> str:
    """`tier_2_S`: numbered, so the best folder sorts first in any file browser."""
    rank = FOLDER_ORDER.index(tier) + 1 if tier in FOLDER_ORDER else len(FOLDER_ORDER) + 1
    return f"tier_{rank}_{tier}"


def _merge(spans):
    merged = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _shared_seconds(a, b) -> float:
    """Seconds two lists of intervals have in common."""
    a, b = _merge(a), _merge(b)
    total, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def cross_speaker_overlaps(segments):
    """[(start, end)] wherever two different speakers' segments overlap in time, merged."""
    segs = sorted((float(s.start), float(s.end), s.speaker) for s in segments)
    spans = []
    for i, (a0, a1, who) in enumerate(segs):
        for b0, b1, other in segs[i + 1:]:
            if b0 >= a1:
                break
            if other != who and min(a1, b1) > max(a0, b0):
                spans.append((max(a0, b0), min(a1, b1)))
    return _merge(spans)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# How the two-channel file was made; recorded in each item's metadata.
STRICT_TRACKS = "strict_separation_tracks"
TIME_GATED = "time_gated"


def _fit(track, length: int) -> np.ndarray:
    """`track` as float32, trimmed or zero-padded to exactly `length` samples."""
    track = np.asarray(track, dtype=np.float32)
    if len(track) >= length:
        return track[:length]
    return np.concatenate([track, np.zeros(length - len(track), dtype=np.float32)])


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


@dataclass
class _Render:
    """One excerpt as samples: the mixture and, when asked for, one track per speaker."""
    lo: int                                   # first sample in the pipeline's waveform
    hi: int
    mixture: np.ndarray
    left: Optional[np.ndarray] = None         # speaker A alone
    right: Optional[np.ndarray] = None        # speaker B alone
    method: Optional[str] = None              # how left/right were made


@dataclass
class _Plan:
    """What is decided about one excerpt before any file is written: where it is filed."""
    cand: Candidate
    item: _Judged
    method: Optional[str]
    overlaps: List[tuple]                     # (start, end) in the pipeline's clock
    overlap_seconds: float
    tier: str                                 # the folder it goes in
    overlap_quality: Optional[str] = None     # "good" / "bad" when there is an overlap
    overlap_reasons: List[str] = field(default_factory=list)   # why "bad"
    separated_share: Optional[float] = None   # share of the overlap the separator covered
    min_similarity: Optional[float] = None    # the weakest separated span in the overlap


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
        self.max_new_tokens = reply_budget(self.cfg.max_new_tokens, self.cfg.thinking)

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
                per_call=self._per_call(messages), max_new_tokens=self.max_new_tokens,
                label="conversation export candidate", logger=self.logger,
                thinking=self.cfg.thinking)
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
                   "cut_in_thought": raw is not None and is_cut_in_thought(raw),
                   "raw": raw, "verdict": verdict,
                   "outcome": item.reason or "accepted", "trimmed": item.trimmed,
                   "trim_ignored": item.trim_ignored}
            rows.append(row)
            if self.logger:
                if raw is None:
                    note = "no answer"
                elif row["cut_in_thought"]:
                    note = (f"cut off while thinking (max_new_tokens {self.max_new_tokens}); "
                            "raise it")
                else:
                    note = f"reply {' '.join(clean_reply(raw).split())[:160]!r}"
                self.logger.info(
                    f"[conversation-exports] candidate #{first}-#{last} "
                    f"({row['start']:.0f}-{row['end']:.0f}s): {row['outcome']}; {note}")
        report["replies"] = rows
        report["unreadable"] = sum(1 for r in rows if r["answered"] and not r["readable"])
        report["cut_in_thought"] = sum(1 for r in rows if r["cut_in_thought"])
        ignored: Dict[str, int] = {}
        for item in judged:
            if item.trim_ignored:
                ignored[item.trim_ignored] = ignored.get(item.trim_ignored, 0) + 1
        report["trims_ignored"] = ignored

    # -- two-channel file ----------------------------------------------------------
    def _speaker_tracks(self, finder, cand: Candidate, raw: np.ndarray, sample_rate: int,
                        offset: int, total_samples: int, speech_segments=None,
                        separation_service=None):
        """(A, B, method): one track per speaker over the same samples as `raw`.

        The tracks come from the separation stage when it is at hand: where the two
        spoke at once each then carries one voice. Gating the mixture cannot do
        that -- one microphone recorded both, and opening a channel during a turn
        leaves the other speaker's overlapping voice in it -- so it is only the
        fallback, used when there are no tracks, when laying them out fails, or when
        one comes out silent.
        """
        if speech_segments and separation_service is not None:
            why = None
            try:
                left, right = separation_service.export_sdlm_dual_channel(
                    speech_segments, total_samples / float(sample_rate), sample_rate,
                    strict=True, speakers=(cand.speakers[0], cand.speakers[1]),
                    time_range=(offset / float(sample_rate),
                                (offset + len(raw)) / float(sample_rate)),
                    log_stats=False)
                left, right = _fit(left, len(raw)), _fit(right, len(raw))
                if np.any(np.abs(left) > 1e-7) and np.any(np.abs(right) > 1e-7):
                    return left, right, STRICT_TRACKS
                why = "a separated speaker track is silent over this excerpt"
            except Exception as exc:                   # pragma: no cover - defensive
                why = f"the separated tracks could not be laid out ({type(exc).__name__}: {exc})"
            if self.logger:
                self.logger.warning(
                    f"[conversation-exports] speaker files gated from the mixture: {why}")

        names = self._names(cand)
        left, right = [], []
        for pos in range(cand.first, cand.last + 1):
            seg = finder.segs[pos]
            span = (float(seg.start), float(seg.end))
            (left if names.get(seg.speaker) == "A" else right).append(span)
        gated = gate_channels(raw, sample_rate, offset, left, right,
                              self.cfg.gate_margin_ms, self.cfg.gate_fade_ms)
        return (np.ascontiguousarray(gated[:, 0], dtype=np.float32),
                np.ascontiguousarray(gated[:, 1], dtype=np.float32), TIME_GATED)

    def _render(self, finder, cand: Candidate, waveform: np.ndarray, sample_rate: int,
                speech_segments=None, separation_service=None) -> Optional[_Render]:
        """Cut the excerpt once; every file is made from this one cut.

        The mixture and both speaker tracks get the same start sample, the same
        length and the same fades, which is what keeps them on one clock.
        """
        cfg = self.cfg
        lo, hi = cut_bounds(waveform, sample_rate, cand.pad_start, cand.pad_end,
                            cfg.zero_cross_ms)
        if hi - lo < sample_rate:
            return None
        raw = np.array(waveform[lo:hi], dtype=np.float32, copy=True)
        out = _Render(lo, hi, fade_edges(raw.copy(), sample_rate, cfg.fade_ms))
        if cfg.stereo:
            left, right, out.method = self._speaker_tracks(
                finder, cand, raw, sample_rate, lo, len(waveform),
                speech_segments, separation_service)
            out.left = fade_edges(left, sample_rate, cfg.fade_ms)
            out.right = fade_edges(right, sample_rate, cfg.fade_ms)
        return out

    def _plan(self, finder, cand: Candidate, item: _Judged, waveform, sample_rate: int,
              speech_segments=None, separation_service=None) -> Optional[_Plan]:
        """Which tier folder the excerpt goes in, and why."""
        render = self._render(finder, cand, waveform, sample_rate,
                              speech_segments, separation_service)
        if render is None:
            return None
        overlaps = cross_speaker_overlaps(finder.segs[cand.first:cand.last + 1])
        seconds = sum(b - a for a, b in overlaps)
        plan = _Plan(cand, item, render.method, overlaps, seconds, cand.tier)
        if seconds >= self.cfg.overlap_min_seconds:
            self._judge_overlap(plan, speech_segments)
            if self.cfg.overlap_first:
                plan.tier = OVERLAP_GOOD if plan.overlap_quality == "good" else OVERLAP_BAD
        return plan

    def _judge_overlap(self, plan: _Plan, speech_segments) -> None:
        """Say whether the overlap in this excerpt was captured cleanly, and if not, why.

        Clean means each ear really holds one voice there: the separated tracks
        were used, none of the separator's failures fall inside the excerpt, every
        span it separated was a confident match to the speaker, and nearly all of
        the overlap was separated at all. Anything less and one ear may carry
        both voices, or none.
        """
        cfg, cand = self.cfg, plan.cand
        reasons = []
        if plan.method != STRICT_TRACKS or not speech_segments:
            reasons.append("not_separated_tracks")
        else:
            spans, sims = [], []
            for seg in speech_segments:
                if seg.speaker not in cand.speakers:
                    continue
                for a, b, sim in getattr(seg, "bss_spans", ()):
                    if _shared_seconds([(float(a), float(b))], plan.overlaps) > 0:
                        spans.append((float(a), float(b)))
                        sims.append(float(sim))
            total = sum(b - a for a, b in plan.overlaps)
            plan.separated_share = round(_shared_seconds(spans, plan.overlaps) / total, 4) if total else 1.0
            plan.min_similarity = round(min(sims), 4) if sims else None
            if self._zeroed_spans(speech_segments, cand):
                reasons.append("separation_failed")
            if sims and min(sims) < cfg.overlap_good_min_similarity:
                reasons.append("low_similarity")
            if plan.separated_share < cfg.overlap_good_min_coverage:
                reasons.append("overlap_not_separated")
        plan.overlap_reasons = reasons
        plan.overlap_quality = "bad" if reasons else "good"

    @staticmethod
    def _zeroed_spans(speech_segments, cand: Candidate) -> bool:
        """Whether the separator failed somewhere in the excerpt, leaving silence in both ears."""
        from services.clean_two_channel_dataset_service import (
            CleanTwoChannelDatasetService as _Tracks)
        try:
            return any(row["zeroed_in_strict_track"]
                       for row in _Tracks._failure_rows(speech_segments, cand))
        except Exception:                              # pragma: no cover - defensive
            return True                                # cannot tell, so it is not "clean"

    @staticmethod
    def _clear_previous(out_dir: str) -> None:
        """Remove the tier folders an earlier run left, so the two do not mix."""
        if not os.path.isdir(out_dir):
            return
        for name in os.listdir(out_dir):
            path = os.path.join(out_dir, name)
            if re.fullmatch(r"tier_\d+_[A-Za-z_]+", name) and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    def _verify(self, directory: str, files: Dict[str, str], sample_rate: int,
                num_samples: int) -> dict:
        """Read every file back and check that they are the same length and the same clock.

        Compared as the 16-bit samples that were written, so "equal" means the
        stereo channels are the speaker files, bit for bit.
        """
        import soundfile as sf
        info, data = {}, {}
        for key, name in files.items():
            path = os.path.join(directory, name)
            samples, rate = sf.read(path, dtype="int16", always_2d=True)
            data[key] = samples
            info[key] = {"file": name, "channels": int(samples.shape[1]),
                         "frames": int(samples.shape[0]), "sample_rate": int(rate),
                         "sha256": _sha256(path)}
        checks = {
            "same_sample_rate": {v["sample_rate"] for v in info.values()} == {sample_rate},
            "same_length": {v["frames"] for v in info.values()} == {num_samples},
            "mixture_is_mono": info["mixture"]["channels"] == 1,
        }
        if "stereo_2ch" in info:
            checks["speaker_files_are_mono"] = (info["speaker_A"]["channels"] == 1
                                                and info["speaker_B"]["channels"] == 1)
            checks["stereo_is_two_channels"] = info["stereo_2ch"]["channels"] == 2
            stereo = data["stereo_2ch"]
            checks["stereo_left_is_speaker_A"] = bool(
                stereo.shape[1] == 2 and np.array_equal(stereo[:, 0], data["speaker_A"][:, 0]))
            checks["stereo_right_is_speaker_B"] = bool(
                stereo.shape[1] == 2 and np.array_equal(stereo[:, 1], data["speaker_B"][:, 0]))
        return {"ok": all(checks.values()), "sample_rate": sample_rate,
                "num_samples": num_samples,
                "duration": round(num_samples / float(sample_rate), 6),
                "files": info, "checks": checks}

    def _write_conversation(self, finder, plan: _Plan, directory: str, folder: str,
                            export_id: str, waveform, sample_rate: int,
                            speech_segments=None, separation_service=None) -> dict:
        """Write one conversation folder and return its conversation.json content."""
        import soundfile as sf
        cand = plan.cand
        render = self._render(finder, cand, waveform, sample_rate,
                              speech_segments, separation_service)
        os.makedirs(directory, exist_ok=True)
        files = {"mixture": MIXTURE_FILE}
        sf.write(os.path.join(directory, MIXTURE_FILE), render.mixture,
                 sample_rate, subtype="PCM_16")
        if render.left is not None:
            files.update({"speaker_A": SPEAKER_FILES["A"], "speaker_B": SPEAKER_FILES["B"],
                          "stereo_2ch": STEREO_FILE})
            sf.write(os.path.join(directory, SPEAKER_FILES["A"]), render.left,
                     sample_rate, subtype="PCM_16")
            sf.write(os.path.join(directory, SPEAKER_FILES["B"]), render.right,
                     sample_rate, subtype="PCM_16")
            sf.write(os.path.join(directory, STEREO_FILE),
                     np.column_stack((render.left, render.right)),
                     sample_rate, subtype="PCM_16")
        meta = self._metadata(
            finder, finder.timeline, export_id, cand, plan.item,
            len(render.mixture) / float(sample_rate),
            audio_start=render.lo / float(sample_rate),
            audio_end=render.hi / float(sample_rate),
            two_channel=render.method, speech_segments=speech_segments,
            folder=folder, tier=plan.tier, overlaps=plan.overlaps, files=files,
            overlap_info={"quality": plan.overlap_quality, "reasons": plan.overlap_reasons,
                          "separated_share": plan.separated_share,
                          "min_similarity": plan.min_similarity},
            verification=self._verify(directory, files, sample_rate, len(render.mixture)))
        with open(os.path.join(directory, JSON_FILE), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(directory, TEXT_FILE), "w", encoding="utf-8") as fh:
            fh.write(self._transcript_text(meta))
        return meta

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
                  audio_end: Optional[float] = None, two_channel: Optional[str] = None,
                  speech_segments=None, folder: Optional[str] = None,
                  tier: Optional[str] = None, overlaps=None, files: Optional[dict] = None,
                  verification: Optional[dict] = None, overlap_info: Optional[dict] = None) -> dict:
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
            "folder": folder,
            "files": dict(files or {}),
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
            "tier": tier or cand.tier,
            "tier_by_score": cand.tier,
            "overlap_quality": (overlap_info or {}).get("quality"),
            "overlap_reasons": (overlap_info or {}).get("reasons", []),
            "overlap_separated_share": (overlap_info or {}).get("separated_share"),
            "overlap_min_similarity": (overlap_info or {}).get("min_similarity"),
            "overlap_seconds": round(sum(b - a for a, b in (overlaps or [])), 3),
            "overlap_spans": [{"start": round(a - origin, 3), "end": round(b - origin, 3)}
                              for a, b in (overlaps or [])],
            "components": cand.components,
            "topic": judged.topic,
            "semantic_score": judged.semantic,
            "semantic_reason": judged.why,
            "trimmed_by_model": judged.trimmed,
            "trim_ignored": judged.trim_ignored,
            "noise": cand.noise,
            "music_patched_share": cand.music_patched_share,
            "conversation": conversation,
            "transcript": TEXT_FILE,
            "verification": verification,
        }
        if self.cfg.stereo:
            method = two_channel or TIME_GATED
            out["channels_2ch"] = {
                "left": "A", "right": "B", "method": method,
                "left_file": SPEAKER_FILES["A"], "right_file": SPEAKER_FILES["B"],
                "note": self._TWO_CHANNEL_NOTES[method],
            }
            if method == STRICT_TRACKS and speech_segments:
                from services.clean_two_channel_dataset_service import (
                    CleanTwoChannelDatasetService as _Tracks)
                # Where the separator pulled the two voices apart, and where it
                # could not (silent in both channels), relative to the excerpt.
                try:
                    out["separated_spans"] = _Tracks._separated_rows(speech_segments, cand)
                    out["failed_separation_spans"] = _Tracks._failure_rows(speech_segments, cand)
                except Exception as exc:               # pragma: no cover - defensive
                    # A description of the tracks is not worth losing the excerpt for.
                    if self.logger:
                        self.logger.warning(
                            f"[conversation-exports] could not describe the separated "
                            f"spans of {export_id}: {exc}")
        return out

    _TWO_CHANNEL_NOTES = {
        STRICT_TRACKS: ("speaker_A.wav and speaker_B.wav are each one person's separated "
                        "track, one channel each, and stereo_2ch.wav is those two files with "
                        "A on the left and B on the right, sample for sample. Where the two "
                        "spoke at once each is heard alone. A stretch whose separation "
                        "failed is silent in both rather than carrying the other voice. "
                        "mixture.wav keeps both voices as recorded."),
        TIME_GATED: ("No separated tracks were used, so speaker_A.wav and speaker_B.wav are "
                     "the mixture gated to each person's turns, and stereo_2ch.wav is those "
                     "two files, A left and B right. Each is heard only during their own "
                     "turns; where the two speak at once both carry the same mixture. "
                     "mixture.wav keeps both voices as recorded."),
    }

    # -- the pass ---------------------------------------------------------------------
    def run(self, transcripts, *, timeline, noise, music_map, waveform,
            sample_rate: int, out_dir: str, base_name: str,
            speech_segments=None, separation_service=None) -> ConversationExportRun:
        """Find, judge and write selected conversation excerpts of one recording.

        `speech_segments` and `separation_service` are what the separation stage
        produced and the object that can lay its speaker tracks out; with both,
        the two-channel file is built from those tracks (see the module doc).
        """
        cfg = self.cfg
        finder = ConversationSelectionFinder(transcripts, timeline, noise, music_map, cfg)
        result = ConversationExportRun()
        report = {"prompt_version": CONVERSATION_EXPORT_PROMPT_VERSION,
                  "thinking": cfg.thinking, "skipped": None,
                  "candidates": 0, "shortlisted": 0, "judged_rejected": {},
                  "accepted": 0, "exported": 0, "tiers": {}, "verification_failed": 0}
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

        plans = []
        for cand in final:
            plan = self._plan(finder, cand, kept[id(cand)], waveform, sample_rate,
                              speech_segments, separation_service)
            if plan is None:
                self._log(f"[conversation-exports] {base_name}: an excerpt shorter "
                          "than a second was skipped")
                continue
            plans.append(plan)

        # The best folder first; inside one, the excerpts with overlap, then by score.
        rank = {name: k for k, name in enumerate(FOLDER_ORDER)}
        plans.sort(key=lambda p: (rank.get(p.tier, len(rank)), -p.cand.score))
        self._clear_previous(out_dir)

        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name)
        filed: Dict[str, int] = {}
        for global_number, plan in enumerate(plans, start=1):
            cand, item = plan.cand, plan.item
            tier_dir = tier_folder(plan.tier)
            filed[tier_dir] = filed.get(tier_dir, 0) + 1
            name = f"conversation_{filed[tier_dir]}"
            folder = f"{tier_dir}/{name}"
            export_id = f"{safe}_conversation_{global_number:06d}"
            try:
                meta = self._write_conversation(
                    finder, plan, os.path.join(out_dir, tier_dir, name), folder, export_id,
                    waveform, sample_rate, speech_segments, separation_service)
            except Exception as exc:                       # pragma: no cover - disk problems
                if self.logger:
                    self.logger.warning(f"[conversation-exports] could not write {export_id}: {exc}")
                continue
            if not meta["verification"]["ok"]:
                report["verification_failed"] += 1
                if self.logger:
                    bad = [k for k, ok in meta["verification"]["checks"].items() if not ok]
                    self.logger.error(
                        f"[conversation-exports] {folder}: failed {bad}; do not trust these files")
            report["tiers"][tier_dir] = report["tiers"].get(tier_dir, 0) + 1
            result.exports.append({
                "id": export_id, "folder": folder, "tier": plan.tier, "tier_by_score": cand.tier,
                "overlap_quality": plan.overlap_quality, "overlap_reasons": plan.overlap_reasons,
                "overlap_seconds": meta["overlap_seconds"], "score": cand.score,
                "files": {key: f"{folder}/{file}" for key, file in meta["files"].items()},
                "audio": f"{folder}/{MIXTURE_FILE}",
                "audio_2ch": (f"{folder}/{STEREO_FILE}" if "stereo_2ch" in meta["files"] else None),
                "duration": meta["duration"], "topic": item.topic,
                "semantic_score": item.semantic, "trimmed_by_model": item.trimmed,
                "source_start": meta["source_start"], "source_end": meta["source_end"],
                "speaker_ids": meta["speaker_ids"], "verified": meta["verification"]["ok"],
                "noise_kind": (cand.noise or {}).get("dominant_kind"),
            })
        report["exported"] = len(result.exports)
        self._log(
            f"[conversation-exports] {base_name}: {len(result.exports)} item(s) written to {out_dir}")
        return result

    def _log(self, message: str):
        if self.logger:
            self.logger.info(message)
