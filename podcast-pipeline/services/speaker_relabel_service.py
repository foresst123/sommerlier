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
    says: only speakers the file already has, a confidence floor, segments tied
    to overlap separation are locked, one window decides each segment, and a
    model that disagrees with too much of the file is treated as broken and
    discarded whole.

What the model is asked to copy back is kept small. Each window numbers its own
lines 1, 2, 3... and the speakers are lettered A, B, C...; the model answers
"line 7 is B" and this module turns that back into a segment index and the
diarizer's label. A long zero-padded id, or a label that is a bare digit and
reads like a line number, is what a small model gets wrong; a number outside
1..N or a letter that is not in the list is a wrong answer that is refused where
it stands, and the report says so.

`relabel()` only computes: it returns a `RelabelResult` and leaves the segments
alone, so the caller can checkpoint the decision before applying it.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.llm_batches import ask_in_batches, reply_budget
from utils.llm_json import clean_reply, is_cut_in_thought, is_readable, objects_in
from utils.transcript_windows import build_windows, format_line, line_number

# Bump when the prompt or the acceptance rules change: it is part of the
# checkpoint namespace, so a changed prompt recomputes instead of reusing labels
# that an older one produced.
RELABEL_PROMPT_VERSION = "relabel-v3"

# Room the chat template and the reply framing take beyond the counted prompt.
_TEMPLATE_SLACK_TOKENS = 64

# The worked example in the prompt names lines 901-904. No window is that long, so
# a model that copies the example's answer names a line that does not exist and
# the answer is refused; the same answer with a real line number would be a
# real, false, relabel.
RELABEL_SYSTEM_PROMPT = (
    "Bạn kiểm tra nhãn người nói trong transcript của MỘT cuộc hội thoại tiếng Việt (podcast, phỏng vấn hoặc trò chuyện). "
    "Nhãn do máy nhận dạng giọng nói gán và đôi khi gán sai, nhất là ở các câu ngắn, lời đệm và chỗ hai người nói chồng lên nhau. "
    "Bạn không nghe được audio: bạn chỉ có văn bản, mốc thời gian và nhãn máy đã gán.\n"
    "\n"
    "### NHIỆM VỤ DUY NHẤT\n"
    "Tìm các dòng bị gán SAI người nói và cho biết dòng đó thực ra thuộc người nào trong số những người đã có. "
    "Bạn KHÔNG sửa văn bản, KHÔNG sửa mốc thời gian, KHÔNG tách hay gộp dòng, KHÔNG thêm người nói mới.\n"
    "\n"
    "### ĐỊNH DẠNG MỖI DÒNG\n"
    "[số dòng] bắt_đầu-kết_thúc NGƯỜI (gap ±giây) nội dung\n"
    "- Số dòng: số trong ngoặc vuông, đếm từ 1 trong phần transcript bạn nhận. Chỉ dùng số này để chỉ một dòng.\n"
    "- NGƯỜI: một chữ cái (A, B, C...) đại diện cho một người nói. Cùng chữ là cùng một người.\n"
    "- gap: số giây từ lúc dòng trước kết thúc đến lúc dòng này bắt đầu. Gap ÂM nghĩa là hai người nói chồng lên nhau. "
    "Gap dài thường là đổi lượt; gap rất ngắn giữa hai dòng cùng chữ thường là một người đang nói tiếp.\n"
    "- Dòng có [cố định] đã gắn chặt với âm thanh đã tách giọng: dùng nó làm ngữ cảnh, nhưng KHÔNG đề xuất đổi người cho nó.\n"
    "- Đoạn bạn nhận có thể bắt đầu hoặc kết thúc giữa cuộc trò chuyện. Với vài dòng sát đầu và sát cuối, bạn thiếu ngữ cảnh "
    "ở một phía, nên hãy thận trọng hơn.\n"
    "\n"
    "### CÁCH XÉT (chỉ đề xuất khi mạch hội thoại cho bằng chứng rõ)\n"
    "1. Hỏi – đáp: một câu hỏi thường được người KHÁC trả lời. Nếu câu trả lời mang chữ của chính người vừa hỏi, "
    "hoặc một câu hỏi mang chữ của người vừa nói xong, có thể nhãn sai.\n"
    "2. Xưng hô: mỗi người thường xưng hô nhất quán (anh/em, mình/bạn, tôi/bác...). "
    "Dòng xưng hô ngược với các dòng khác cùng chữ có thể là của người kia.\n"
    "3. Câu bị cắt đôi: câu dở dang ở dòng trước được nói tiếp ở dòng sau thì thường là CÙNG một người. "
    "Nếu hai dòng đó mang hai chữ khác nhau thì một trong hai có thể sai.\n"
    "4. Lời đệm ngắn (ừ, ừm, dạ, vâng, à, đúng rồi) thường của người đang NGHE. "
    "Lời đệm mang chữ của chính người đang nói dài ngay quanh nó thì có thể sai.\n"
    "5. Khi phân vân, KHÔNG đề xuất. Nhãn hiện tại đúng ở phần lớn các dòng.\n"
    "\n"
    "### KHÔNG PHẢI LÝ DO ĐỂ ĐỔI\n"
    "- Một người nói liền nhiều dòng (lượt nói dài) là bình thường.\n"
    "- Một chữ chỉ xuất hiện ở vài dòng không có nghĩa là sai: đó có thể là một người khác thật sự "
    "(khách mời, người trong đoạn video chèn vào, giọng đọc quảng cáo). "
    "Chỉ đổi khi nội dung và mạch hội thoại cho thấy rõ dòng đó là của người khác.\n"
    "- Lỗi chính tả hoặc từ lạ trong văn bản không liên quan đến người nói.\n"
    "- Đừng đề xuất hàng loạt. Nếu bạn muốn đổi nhiều dòng trong một đoạn, hãy xét lại: có thể bạn đang hiểu sai mạch hội thoại.\n"
    "\n"
    "### ĐẦU RA\n"
    "Chỉ MỘT mảng JSON, không viết gì ngoài JSON, không dùng ```. Mỗi phần tử có đúng bốn khoá:\n"
    '- "i": số dòng cần đổi, là số nguyên, đúng như số trong ngoặc vuông.\n'
    '- "speaker": chữ cái của người ĐÚNG. Chỉ dùng các chữ có trong dòng "Người nói".\n'
    '- "conf": độ chắc chắn từ 0 đến 1. Chỉ đề xuất khi conf từ {min_confidence} trở lên.\n'
    '- "why": lý do, tối đa 12 từ.\n'
    "Nếu không có dòng nào sai, xuất đúng: []\n"
    "\n"
    "### VÍ DỤ MINH HOẠ\n"
    "Số dòng và nội dung dưới đây chỉ để minh hoạ cách xét, không phải transcript của bạn.\n"
    "[901] 00:10.0-00:14.2 A (gap +1.0s) Anh làm nghề này được bao lâu rồi ạ?\n"
    "[902] 00:14.6-00:23.0 B (gap +0.4s) Cũng gần mười năm rồi em. Hồi đầu thì cực lắm.\n"
    "[903] 00:23.4-00:26.1 B (gap +0.4s) Vậy điều gì giữ anh lại với nghề?\n"
    "[904] 00:26.5-00:35.9 B (gap +0.4s) Chắc là vì mình thấy mình làm được điều có ích.\n"
    "Dòng 903 là một câu hỏi nhưng mang chữ B, cùng chữ với người vừa trả lời ở dòng 902, và dòng 904 mới là câu trả lời. "
    "Câu hỏi đó là của người hỏi ở dòng 901, nên kết quả đúng là:\n"
    '[{"i": 903, "speaker": "A", "conf": 0.9, "why": "câu hỏi, người trả lời vừa nói xong"}]'
)

_WRAPPER_KEYS = ("changes", "relabels", "proposals", "items", "result")


def relabel_system_prompt(min_confidence: float = 0.7) -> str:
    """The prompt as sent, with the confidence floor the guards will apply."""
    return RELABEL_SYSTEM_PROMPT.replace("{min_confidence}", f"{min_confidence:g}")


def speaker_names(labels: List[str]) -> Dict[str, str]:
    """The diarizer's labels as the model is told them: A, B, C...

    Labels are whatever the diarizer produced ('0'..'4', 'SPEAKER_01'); bare
    digits sit beside line numbers in the prompt and read as one. Sorted, so the
    same file always gets the same letters.
    """
    return {label: (chr(ord("A") + k) if k < 26 else f"S{k + 1}")
            for k, label in enumerate(sorted(labels))}


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
    thinking: bool = False              # whether the model was asked to reason first
    names: Dict[str, str] = field(default_factory=dict)  # the model's letter -> diarizer label
    windows: int = 0
    failed_windows: int = 0             # windows the model could not be run on
    unreadable_windows: int = 0         # answered, but with no JSON in the reply
    cut_in_thought_windows: int = 0     # of those, ended inside <think>: the budget was too small
    proposed: int = 0                   # proposals read, before any guard
    replies: List[dict] = field(default_factory=list)   # one row per window, raw reply included
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
            "thinking": self.thinking,
            "names": self.names,
            "windows": self.windows,
            "failed_windows": self.failed_windows,
            "unreadable_windows": self.unreadable_windows,
            "cut_in_thought_windows": self.cut_in_thought_windows,
            "proposed": self.proposed,
            "skipped": self.skipped,
            "discarded": self.discarded,
            "changed": changed,
            "changed_fraction": round(changed / self.segments, 4) if self.segments else 0.0,
            "applied": self.applied,
            "rejected": self.rejected,
            "replies": self.replies,
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
                 max_new_tokens: int = 768, thinking: bool = False):
        self.llm = llm
        self.logger = logger
        self.window_tokens = max(1, int(window_tokens))
        self.overlap_segments = max(0, int(overlap_segments))
        self.min_confidence = float(min_confidence)
        self.max_change_fraction = float(max_change_fraction)
        # A small file can still have one or two honest corrections; without a
        # floor, a 15% cap on twelve segments would forbid all of them.
        self.min_change_allowance = max(0, int(min_change_allowance))
        self.thinking = bool(thinking)
        self.max_new_tokens = reply_budget(max_new_tokens, self.thinking)

    @property
    def checkpoint_namespace(self) -> str:
        """Prompt, thinking and model, as a path-safe name for CheckpointManager.

        A decision made with the model reasoning first is not the one it makes
        without, so the two do not share a checkpoint.
        """
        model = getattr(self.llm, "model_name", None) or "unknown"
        mode = "-think" if self.thinking else ""
        return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{RELABEL_PROMPT_VERSION}{mode}-{model}")

    # -- prompt ------------------------------------------------------------
    @staticmethod
    def _header(names: List[str], lines: int) -> str:
        return (f"Người nói: {', '.join(names)} (mỗi chữ là một người)\n"
                f"Số dòng: 1 đến {lines}\n\nTranscript:\n")

    _FOOTER = "\n\nDanh sách đề xuất (JSON):"

    def _system_prompt(self) -> str:
        return relabel_system_prompt(self.min_confidence)

    def prompt_overhead(self, names: List[str]) -> int:
        """Tokens every window pays before its first transcript line."""
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
        result = RelabelResult(segments=len(segments), thinking=self.thinking)
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

        names = speaker_names(labels)
        result.names = {name: label for label, name in names.items()}
        letters = [names[label] for label in labels]
        by_name = {name.lower(): label for name, label in result.names.items()}

        locked = [self.is_locked(s) for s in segments]

        def text_of(pos: int, number: int) -> str:
            seg = segments[pos]
            return format_line(seg, number, names[str(seg.speaker)], locked[pos])

        # Sized with each segment's own position as its number; a window renumbers
        # from 1, which never costs more than a digit or two per line.
        counts = [self.llm.count_tokens(text_of(pos, pos + 1)) + 1
                  for pos in range(len(segments))]
        budget = max(1, self.window_tokens - self.prompt_overhead(letters))
        windows = build_windows(counts, budget, self.overlap_segments)
        result.windows = len(windows)
        messages = [self._message(letters, [text_of(pos, pos - w.start + 1)
                                            for pos in range(w.start, w.stop)])
                    for w in windows]

        replies = self._ask(messages, result)

        accepted: Dict[str, dict] = {}

        def reject(proposal, reason, window, pos=None):
            result.rejected.append({
                "line": line_number(proposal["i"]) or proposal["i"],
                "index": str(segments[pos].index) if pos is not None else None,
                "to": proposal["speaker"], "conf": proposal["conf"],
                "reason": reason, "window": window})

        for w_no, (window, reply) in enumerate(zip(windows, replies)):
            proposals = parse_proposals(reply) if reply is not None else []
            self._record(result, segments, w_no, window, reply, len(proposals))
            if reply is None:
                continue
            for proposal in proposals:
                line = line_number(proposal["i"])
                if line is None or line > window.stop - window.start:
                    reject(proposal, "unknown_index", w_no)
                    continue
                pos = window.start + line - 1
                target = by_name.get(proposal["speaker"].strip().lower())
                if target is None:
                    reject(proposal, "unknown_label", w_no, pos)
                    continue
                conf = proposal["conf"]
                if conf is None or not 0.0 <= conf <= 1.0:
                    reject(proposal, "bad_confidence", w_no, pos)
                    continue
                if conf < self.min_confidence:
                    reject(proposal, "low_confidence", w_no, pos)
                    continue
                if locked[pos]:
                    reject(proposal, "locked_segment", w_no, pos)
                    continue
                if not window.owns(pos):
                    reject(proposal, "outside_window_core", w_no, pos)
                    continue
                seg = segments[pos]
                if str(seg.index) in accepted:
                    reject(proposal, "duplicate", w_no, pos)
                    continue
                if target == str(seg.speaker):
                    reject(proposal, "no_change", w_no, pos)
                    continue
                accepted[str(seg.index)] = {
                    "index": str(seg.index), "line": line, "from": str(seg.speaker),
                    "to": target, "conf": conf, "why": proposal["why"], "window": w_no}

        allowed = max(self.min_change_allowance,
                      int(self.max_change_fraction * len(segments)))
        if len(accepted) > allowed:
            # A model that disagrees with this much of the file is not finding
            # errors, it is not following the task. Keep no part of it.
            result.discarded = "over_cap"
            for entry in accepted.values():
                result.rejected.append({
                    "line": entry["line"], "index": entry["index"], "to": entry["to"],
                    "conf": entry["conf"], "reason": "over_cap", "window": entry["window"]})
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
                f"relabelled, {result.proposed} proposed, {len(result.rejected)} refused, "
                f"{result.failed_windows}/{result.windows} window(s) unanswered, "
                f"{result.unreadable_windows} unreadable")
        return result

    def _record(self, result: RelabelResult, segments, w_no: int, window,
                reply: Optional[str], proposals: int) -> None:
        """Keep the reply as the model wrote it, and count the ones with no JSON.

        A window answered with prose looks exactly like one answered `[]` once
        parsed -- no proposals either way -- so without this a broken prompt and
        a clean transcript produce the same report.
        """
        readable = reply is not None and is_readable(reply)
        cut = reply is not None and is_cut_in_thought(reply)
        result.proposed += proposals
        if reply is not None and not readable:
            result.unreadable_windows += 1
        if cut:
            result.cut_in_thought_windows += 1
        first, last = str(segments[window.start].index), str(segments[window.stop - 1].index)
        result.replies.append({
            "window": w_no, "first_index": first, "last_index": last,
            "lines": window.stop - window.start, "answered": reply is not None,
            "readable": readable, "cut_in_thought": cut, "proposals": proposals,
            "raw": reply})
        if self.logger:
            if reply is None:
                note = "no answer"
            elif cut:
                note = (f"cut off while thinking (max_new_tokens {self.max_new_tokens}); "
                        "raise it")
            else:
                shown = " ".join(clean_reply(reply).split())[:200]
                note = f"{proposals} proposal(s), reply {shown!r}"
            say = self.logger.warning if reply is not None and not readable else self.logger.info
            say(f"[relabel] window {w_no} (#{first}-#{last}): {note}")

    def _ask(self, messages: List[str], result: RelabelResult) -> List[Optional[str]]:
        """One reply per window, None where the model could not answer it.

        A window that cannot be answered is recorded and skipped -- its segments
        simply keep their labels.
        """
        replies, unanswered = ask_in_batches(
            self.llm, self._system_prompt(), messages,
            per_call=self._windows_per_call(), max_new_tokens=self.max_new_tokens,
            label="relabel window", logger=self.logger, thinking=self.thinking)
        result.failed_windows += unanswered
        return replies
