"""Reassign which speaker a segment belongs to, from the whole transcript.

The refinement LLM reads the transcript -- text, timing, current speaker -- and
says which segments look mislabelled and who they belong to. It cannot hear the
audio, and refinement once let it copy one speaker's words under another's
label, which is why fusion never touched labels. This pass is allowed to, on
narrow terms:

  * The model's answer is a list of {"i", "speaker", "conf", "why"} and nothing
    else. The parser reads those four keys; a reply that also carries text or a
    timestamp has them dropped before anything looks at them.
  * `apply_relabels` writes one attribute, `speaker`. Start, end, text and every
    other field are untouched, and no segment is split or merged.
  * With no audio to check against, the guards work from what the file itself
    says: only labels the file already has, a confidence floor, segments tied
    to overlap separation are locked, one window decides each segment, and a
    model that disagrees with too much of the file is treated as broken and
    discarded whole.

`relabel()` only computes: it returns a `RelabelResult` and leaves the segments
alone, so the caller can checkpoint the decision before applying it.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.llm_batches import ask_in_batches
from utils.llm_json import objects_in
from utils.transcript_windows import build_windows, format_line, norm_index

# Bump when the prompt or the acceptance rules change: it is part of the
# checkpoint namespace, so a changed prompt recomputes instead of reusing labels
# that an older one produced.
RELABEL_PROMPT_VERSION = "relabel-v1"

# Room the chat template and the reply framing take beyond the counted prompt.
_TEMPLATE_SLACK_TOKENS = 64

RELABEL_SYSTEM_PROMPT = (
    "Bạn kiểm tra nhãn người nói (speaker) trong transcript của MỘT cuộc hội thoại tiếng Việt. "
    "Bạn không nghe được audio: chỉ có văn bản, mốc thời gian và nhãn do hệ thống nhận dạng giọng nói gán, "
    "và hệ thống đó đôi khi gán sai.\n"
    "\n"
    "### NHIỆM VỤ DUY NHẤT\n"
    "Tìm các đoạn bị gán SAI nhãn và nói đoạn đó thực ra thuộc người nói nào. "
    "Bạn KHÔNG sửa văn bản, KHÔNG sửa mốc thời gian, KHÔNG tách hay gộp đoạn.\n"
    "\n"
    "### ĐỊNH DẠNG MỖI DÒNG\n"
    "#số_thứ_tự [bắt_đầu-kết_thúc] NHÃN (gap ±giây) văn bản\n"
    "- gap là số giây từ lúc đoạn trước kết thúc đến lúc đoạn này bắt đầu. Gap ÂM nghĩa là hai người nói chồng lên nhau. "
    "Gap dài thường là đổi lượt; gap rất ngắn giữa hai đoạn cùng nhãn thường là cùng một người đang nói tiếp.\n"
    "- Dòng có [cố định] đã gắn chặt với âm thanh đã tách giọng: KHÔNG đề xuất đổi nhãn cho dòng đó.\n"
    "\n"
    "### CÁCH XÉT (chỉ đề xuất khi mạch hội thoại cho bằng chứng rõ)\n"
    "1. Hỏi – đáp: một câu hỏi thường được người KHÁC trả lời. Nếu câu trả lời mang nhãn của người vừa hỏi, có thể nhãn sai.\n"
    "2. Xưng hô: người nói tự xưng và gọi người kia nhất quán ('anh/em', 'mình/bạn'). "
    "Đoạn xưng hô ngược với các đoạn xung quanh của cùng nhãn có thể là của người kia.\n"
    "3. Câu bị cắt đôi: một câu dở dang ở đoạn trước được nói tiếp ở đoạn sau thì thường là CÙNG một người.\n"
    "4. Lời đệm ngắn (ừ, dạ, vâng, à, đúng rồi) thường là của người đang NGHE, không phải người đang nói dài.\n"
    "5. Khi phân vân, KHÔNG đề xuất. Nhãn hiện tại đúng ở phần lớn các đoạn.\n"
    "\n"
    "### ĐẦU RA\n"
    "Chỉ dùng các nhãn có trong danh sách 'Nhãn hợp lệ'. Chỉ xuất MỘT mảng JSON, không giải thích ngoài JSON, "
    "mỗi phần tử có đúng bốn khoá:\n"
    '[{"i": "00012", "speaker": "SPEAKER_01", "conf": 0.85, "why": "trả lời câu hỏi ở dòng trước"}]\n'
    "- i: số thứ tự của dòng cần đổi nhãn (đúng như trong dòng).\n"
    "- speaker: nhãn ĐÚNG mà bạn cho là của đoạn đó.\n"
    "- conf: độ chắc chắn từ 0 đến 1.\n"
    "- why: tối đa 12 từ.\n"
    "Nếu không có đoạn nào sai, xuất []."
)

_WRAPPER_KEYS = ("changes", "relabels", "proposals", "items", "result")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _clean(obj) -> Optional[dict]:
    """Keep the four keys the pass is allowed to read. Anything else is dropped."""
    if not isinstance(obj, dict) or obj.get("i") is None or obj.get("speaker") is None:
        return None
    conf = obj.get("conf")
    if isinstance(conf, bool):
        conf = None
    else:
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = None
    return {
        "i": str(obj["i"]).strip().lstrip("#").strip(),
        "speaker": str(obj["speaker"]),
        "conf": conf,
        "why": str(obj.get("why") or "")[:200],
    }


def parse_proposals(raw: str) -> List[dict]:
    """The proposals in a model reply, each reduced to i / speaker / conf / why.

    Never raises: a reply that cannot be read means no changes, which is the safe
    outcome for a pass whose default is to leave labels alone.
    """
    return [c for c in (_clean(item) for item in objects_in(raw, _WRAPPER_KEYS))
            if c is not None]


# ---------------------------------------------------------------------------
# Result and applying
# ---------------------------------------------------------------------------

@dataclass
class RelabelResult:
    segments: int = 0
    labels: List[str] = field(default_factory=list)
    windows: int = 0
    failed_windows: int = 0
    skipped: Optional[str] = None       # why nothing was attempted
    discarded: Optional[str] = None     # why an attempt was thrown away whole
    applied: List[dict] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)

    @property
    def mapping(self) -> Dict[str, str]:
        """segment index -> the speaker it should carry."""
        return {a["index"]: a["to"] for a in self.applied}

    def to_report(self) -> dict:
        changed = len(self.applied)
        return {
            "prompt_version": RELABEL_PROMPT_VERSION,
            "segments": self.segments,
            "labels": self.labels,
            "windows": self.windows,
            "failed_windows": self.failed_windows,
            "skipped": self.skipped,
            "discarded": self.discarded,
            "changed": changed,
            "changed_fraction": round(changed / self.segments, 4) if self.segments else 0.0,
            "applied": self.applied,
            "rejected": self.rejected,
        }


def apply_relabels(transcripts, mapping: Dict[str, str], speech_segments=None) -> int:
    """Write the decided speakers. Returns how many transcript segments changed.

    This is the only place a segment is modified, and it assigns `speaker` and
    nothing else (plus the trace of the label it replaced). Idempotent: a segment
    already carrying its target is skipped, so re-applying a checkpointed
    decision on a later run keeps the first original label.

    `speech_segments` are matched by index and given the same speaker, because
    exported separation files and their JSON are named by speaker and would
    otherwise disagree with the transcript. Their audio is left as it is.
    """
    changed = 0
    for seg in transcripts:
        target = mapping.get(str(seg.index))
        if target is None or seg.speaker == target:
            continue
        if getattr(seg, "speaker_original", None) is None:
            seg.speaker_original = seg.speaker
        seg.speaker = target
        changed += 1
    for seg in speech_segments or ():
        target = mapping.get(str(seg.index))
        if target is not None:
            seg.speaker = target
    return changed


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

class SpeakerRelabelService:
    """Ask the resident LLM which segments carry the wrong speaker.

    `llm` is the refinement service (or anything with the same surface):
    `ensure_loaded()`, `count_tokens(text)`, `generate_texts(...)`, and the
    `batch_size` / `max_batch_tokens` / `model_name` it was configured with. It
    is shared rather than reloaded, so this pass adds no VRAM.
    """

    def __init__(self, llm, logger=None, window_tokens: int = 3500,
                 overlap_segments: int = 12, min_confidence: float = 0.7,
                 max_change_fraction: float = 0.15, min_change_allowance: int = 2,
                 max_new_tokens: int = 768):
        self.llm = llm
        self.logger = logger
        self.window_tokens = max(1, int(window_tokens))
        self.overlap_segments = max(0, int(overlap_segments))
        self.min_confidence = float(min_confidence)
        self.max_change_fraction = float(max_change_fraction)
        # A small file can still have one or two honest corrections; without a
        # floor, a 15% cap on twelve segments would forbid all of them.
        self.min_change_allowance = max(0, int(min_change_allowance))
        self.max_new_tokens = max(1, int(max_new_tokens))

    @property
    def checkpoint_namespace(self) -> str:
        """Prompt and model, as a path-safe name for CheckpointManager."""
        model = getattr(self.llm, "model_name", None) or "unknown"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{RELABEL_PROMPT_VERSION}-{model}")

    # -- prompt ------------------------------------------------------------
    @staticmethod
    def _header(labels: List[str]) -> str:
        return f"Nhãn hợp lệ: {', '.join(labels)}\n\nTranscript:\n"

    _FOOTER = "\n\nDanh sách đề xuất (JSON):"

    def prompt_overhead(self, labels: List[str]) -> int:
        """Tokens every window pays before its first transcript line."""
        return (self.llm.count_tokens(RELABEL_SYSTEM_PROMPT)
                + self.llm.count_tokens(self._header(labels) + self._FOOTER))

    def _message(self, labels: List[str], lines: List[str]) -> str:
        return self._header(labels) + "\n".join(lines) + self._FOOTER

    def _windows_per_call(self) -> int:
        batch = max(1, int(getattr(self.llm, "batch_size", 1) or 1))
        limit = int(getattr(self.llm, "max_batch_tokens", 0) or 0)
        if limit:
            batch = min(batch, max(1, limit // (self.window_tokens + _TEMPLATE_SLACK_TOKENS)))
        return batch

    @staticmethod
    def is_locked(seg) -> bool:
        """Whether the segment's audio is already tied to its current label.

        Overlap separation assigned each separated track to a speaker; moving
        the label afterwards would leave the label and the voice in the audio
        disagreeing. Such segments stay visible to the model as context.
        """
        return bool(getattr(seg, "bss", False)) or bool(getattr(seg, "unseparated", None))

    # -- the pass ----------------------------------------------------------
    def relabel(self, segments) -> RelabelResult:
        result = RelabelResult(segments=len(segments))
        if not segments:
            result.skipped = "no_segments"
            return result
        labels = sorted({str(s.speaker) for s in segments})
        result.labels = labels
        if len(labels) < 2:
            result.skipped = "single_speaker"
            return result
        if not self.llm.ensure_loaded():
            result.skipped = "llm_unavailable"
            if self.logger:
                self.logger.warning("[relabel] LLM not available; labels left as they are")
            return result

        locked = [self.is_locked(s) for s in segments]
        lines = [format_line(s, l) for s, l in zip(segments, locked)]
        counts = [self.llm.count_tokens(line) + 1 for line in lines]
        budget = max(1, self.window_tokens - self.prompt_overhead(labels))
        windows = build_windows(counts, budget, self.overlap_segments)
        result.windows = len(windows)
        messages = [self._message(labels, lines[w.start:w.stop]) for w in windows]

        replies = self._ask(messages, result)

        by_index = {norm_index(s.index): pos for pos, s in enumerate(segments)}
        by_label = {label.strip().lower(): label for label in labels}
        accepted: Dict[str, dict] = {}

        def reject(proposal, reason, window):
            result.rejected.append({
                "index": proposal["i"], "to": proposal["speaker"],
                "conf": proposal["conf"], "reason": reason, "window": window})

        for w_no, (window, reply) in enumerate(zip(windows, replies)):
            if reply is None:
                continue
            for proposal in parse_proposals(reply):
                pos = by_index.get(norm_index(proposal["i"]))
                if pos is None or not window.shows(pos):
                    reject(proposal, "unknown_index", w_no)
                    continue
                target = by_label.get(proposal["speaker"].strip().lower())
                if target is None:
                    reject(proposal, "unknown_label", w_no)
                    continue
                conf = proposal["conf"]
                if conf is None or not 0.0 <= conf <= 1.0:
                    reject(proposal, "bad_confidence", w_no)
                    continue
                if conf < self.min_confidence:
                    reject(proposal, "low_confidence", w_no)
                    continue
                if locked[pos]:
                    reject(proposal, "locked_segment", w_no)
                    continue
                if not window.owns(pos):
                    reject(proposal, "outside_window_core", w_no)
                    continue
                seg = segments[pos]
                if str(seg.index) in accepted:
                    reject(proposal, "duplicate", w_no)
                    continue
                if target == str(seg.speaker):
                    reject(proposal, "no_change", w_no)
                    continue
                accepted[str(seg.index)] = {
                    "index": str(seg.index), "from": str(seg.speaker), "to": target,
                    "conf": conf, "why": proposal["why"], "window": w_no}

        allowed = max(self.min_change_allowance,
                      int(self.max_change_fraction * len(segments)))
        if len(accepted) > allowed:
            # A model that disagrees with this much of the file is not finding
            # errors, it is not following the task. Keep no part of it.
            result.discarded = "over_cap"
            for entry in accepted.values():
                result.rejected.append({
                    "index": entry["index"], "to": entry["to"], "conf": entry["conf"],
                    "reason": "over_cap", "window": entry["window"]})
            if self.logger:
                self.logger.warning(
                    f"[relabel] proposed {len(accepted)} changes in {len(segments)} "
                    f"segments (allowed {allowed}); treating the answer as unreliable "
                    "and keeping every original label")
        else:
            result.applied = list(accepted.values())

        if self.logger:
            self.logger.info(
                f"[relabel] {len(result.applied)} of {len(segments)} segment(s) "
                f"relabelled, {len(result.rejected)} proposal(s) refused, "
                f"{result.failed_windows}/{result.windows} window(s) unanswered")
        return result

    def _ask(self, messages: List[str], result: RelabelResult) -> List[Optional[str]]:
        """One reply per window, None where the model could not answer it.

        A window that cannot be answered is recorded and skipped -- its segments
        simply keep their labels.
        """
        replies, unanswered = ask_in_batches(
            self.llm, RELABEL_SYSTEM_PROMPT, messages,
            per_call=self._windows_per_call(), max_new_tokens=self.max_new_tokens,
            label="relabel window", logger=self.logger)
        result.failed_windows += unanswered
        return replies
