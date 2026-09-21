"""Cut two-person conversation clips out of a finished transcript.

`utils.dialogue_clips` finds the candidate stretches from numbers alone: exactly
two speakers, no join, no lasting noise, one to four minutes. This service adds
the one thing numbers cannot say -- whether a stretch is a self-contained piece
of conversation -- by asking the resident refinement LLM, then writes the audio
and a JSON sidecar for each stretch it keeps.

The model is a gate, not an author. It sees the turns of one candidate, renamed
A and B, and answers with a score and, optionally, where to start and stop. It
refers to segments by index only: it cannot invent a timestamp, and a trim
outside the candidate is a misbehaving answer that rejects the candidate. A trim
that is inside goes back through every check the scan applies, so shortening a
clip cannot smuggle in what the scan would have refused.

Audio is cut from the waveform the pipeline actually worked on -- music already
stripped, cuts already made -- so timestamps line up with it, and a clip never
holds a join (the finder does not let one in).

Each clip is written twice from one cut, so the two files start on the same
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

from utils.dialogue_clips import (
    Candidate, ClipConfig, DialogueClipFinder, pick_non_overlapping, shortlist)
from utils.llm_batches import ask_in_batches
from utils.llm_json import objects_in
from utils.transcript_windows import norm_index

# Bump when the prompt or the acceptance rules change.
CLIP_PROMPT_VERSION = "clips-v1"

_TEMPLATE_SLACK_TOKENS = 64

CLIP_SYSTEM_PROMPT = (
    "Bạn chọn các đoạn hội thoại để cắt ra làm dữ liệu huấn luyện. Bạn nhận MỘT đoạn hội thoại tiếng Việt giữa hai người "
    "(A và B). Bạn chỉ có văn bản, không nghe được audio.\n"
    "\n"
    "### NHIỆM VỤ\n"
    "Đánh giá đoạn này có TỰ TRỌN NGHĨA không: mở đầu tự nhiên (không bắt đầu giữa chừng một ý), "
    "kết thúc tự nhiên (không bị cắt ngang câu hay chủ đề còn dang dở), và xoay quanh một chủ đề hay một ý rõ ràng.\n"
    "\n"
    "### THANG ĐIỂM self_contained\n"
    "5 = trọn vẹn, mở và kết tự nhiên, một chủ đề rõ.\n"
    "4 = tốt, có thể hơi thiếu ở đầu hoặc đuôi.\n"
    "3 = dùng tạm được.\n"
    "2 = bị cắt ngang hoặc lan man nhiều chủ đề.\n"
    "1 = không hiểu được nếu tách khỏi ngữ cảnh.\n"
    "\n"
    "### CẮT BỚT (tuỳ chọn)\n"
    "Nếu bỏ vài dòng ở đầu hoặc đuôi làm đoạn trọn nghĩa hơn, cho start_index và end_index: số thứ tự của dòng ĐẦU và dòng CUỐI "
    "muốn giữ. Cả hai PHẢI là dòng có trong đoạn. Không thêm dòng ngoài đoạn. Bỏ qua hai khoá này nếu không cần cắt.\n"
    "\n"
    "### ĐẦU RA\n"
    "Chỉ MỘT đối tượng JSON, không giải thích ngoài JSON:\n"
    '{"self_contained": 4, "topic": "tối đa 10 từ", "start_index": "00012", "end_index": "00040"}'
)


# ---------------------------------------------------------------------------
# The model's answer
# ---------------------------------------------------------------------------

def parse_verdict(raw: Optional[str]) -> Optional[dict]:
    """{'self_contained', 'topic', 'start_index', 'end_index'} from one reply.

    `self_contained` is an int in 1..5 or None when it was missing or unusable.
    Returns None when the reply holds no JSON object at all. Any other key the
    model adds is dropped.
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

    def _index(key):
        value = obj.get(key)
        return None if value is None or str(value).strip() == "" else str(value).strip()

    return {"self_contained": score,
            "topic": str(obj.get("topic") or "").strip()[:120],
            "start_index": _index("start_index"),
            "end_index": _index("end_index")}


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


def cut_clip(waveform: np.ndarray, sample_rate: int, pad_start: float,
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
class ClipRun:
    clips: List[dict] = field(default_factory=list)   # one summary row per exported clip
    report: dict = field(default_factory=dict)


@dataclass
class _Judged:
    original: Candidate
    candidate: Optional[Candidate] = None     # None when rejected
    semantic: Optional[int] = None
    topic: str = ""
    trimmed: bool = False
    reason: Optional[str] = None


class DialogueClipService:
    """Judge candidates with the resident LLM, then cut and describe the keepers.

    `llm` is the refinement service (or anything with the same surface); it is
    shared, so this pass adds no VRAM. Settings come from `models.dialogue_clips`
    and are validated by `ClipConfig`: an unknown key raises here.
    """

    def __init__(self, llm, logger=None, **settings):
        self.llm = llm
        self.logger = logger
        self.cfg = ClipConfig.from_settings(settings)

    # -- judging -----------------------------------------------------------------
    @staticmethod
    def _names(cand: Candidate) -> Dict[str, str]:
        return {cand.speakers[0]: "A", cand.speakers[1]: "B"}

    def _message(self, finder: DialogueClipFinder, cand: Candidate) -> str:
        names = self._names(cand)
        lines = []
        for pos in range(cand.first, cand.last + 1):
            seg = finder.segs[pos]
            text = " ".join(str(seg.text or "").split())
            lines.append(f"#{seg.index} {names.get(seg.speaker, '?')}: {text}")
        return "Đoạn hội thoại giữa A và B:\n" + "\n".join(lines) + "\n\nĐánh giá (JSON):"

    def _per_call(self, messages: List[str]) -> int:
        batch = max(1, int(getattr(self.llm, "batch_size", 1) or 1))
        limit = int(getattr(self.llm, "max_batch_tokens", 0) or 0)
        if limit and messages:
            longest = max(self.llm.count_tokens(m) for m in messages)
            longest += self.llm.count_tokens(CLIP_SYSTEM_PROMPT) + _TEMPLATE_SLACK_TOKENS
            batch = min(batch, max(1, limit // longest))
        return batch

    def _verdict_for(self, finder, cand, verdict) -> _Judged:
        cfg = self.cfg
        judged = _Judged(original=cand)
        if verdict is None or verdict["self_contained"] is None:
            if cfg.require_semantic:
                judged.reason = "no_verdict"
                return judged
            judged.candidate = cand
            return judged
        judged.semantic, judged.topic = verdict["self_contained"], verdict["topic"]
        if judged.semantic < cfg.min_semantic:
            judged.reason = "not_self_contained"
            return judged

        first, last = cand.first, cand.last
        start_idx, end_idx = verdict["start_index"], verdict["end_index"]
        if start_idx is not None or end_idx is not None:
            inside = {norm_index(finder.segs[p].index): p
                      for p in range(cand.first, cand.last + 1)}
            new_first = first if start_idx is None else inside.get(norm_index(start_idx))
            new_last = last if end_idx is None else inside.get(norm_index(end_idx))
            if new_first is None or new_last is None or new_first > new_last:
                # Outside the candidate, or back to front: the model is not
                # following the task, so its verdict on this one is not trusted.
                judged.reason = "bad_trim"
                return judged
            if (new_first, new_last) != (first, last):
                trimmed, why = finder.evaluate(new_first, new_last)
                if trimmed is None:
                    judged.reason = f"trim_{why}"
                    return judged
                judged.candidate, judged.trimmed = trimmed, True
                return judged
        judged.candidate = cand
        return judged

    def _judge(self, finder, shortlisted: List[Candidate], report: dict) -> List[_Judged]:
        verdicts: List[Optional[dict]] = [None] * len(shortlisted)
        if self.llm.ensure_loaded():
            messages = [self._message(finder, c) for c in shortlisted]
            replies, unanswered = ask_in_batches(
                self.llm, CLIP_SYSTEM_PROMPT, messages,
                per_call=self._per_call(messages), max_new_tokens=160,
                label="clip candidate", logger=self.logger)
            report["unanswered"] = unanswered
            verdicts = [parse_verdict(r) if r is not None else None for r in replies]
        else:
            report["llm_available"] = False
            if self.logger:
                self.logger.warning("[clips] LLM not available for the semantic check")
        return [self._verdict_for(finder, c, v) for c, v in zip(shortlisted, verdicts)]

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
        """The conversation as plain text, one line per turn, timed from the clip start."""
        lines = []
        for row in meta["conversation"]:
            t = max(0.0, float(row["start"]))
            lines.append(f"[{int(t // 60):02d}:{t % 60:04.1f}] {row['speaker']}: {row['text']}")
        return "\n".join(lines) + "\n"

    # -- metadata ------------------------------------------------------------------
    def _metadata(self, finder, timeline, clip_id, cand: Candidate, judged: _Judged,
                  audio_seconds: float) -> dict:
        names = self._names(cand)
        conversation = []
        for pos in range(cand.first, cand.last + 1):
            seg = finder.segs[pos]
            row = {"speaker": names.get(seg.speaker, "?"),
                   "speaker_id": seg.speaker,
                   "start": round(float(seg.start) - cand.pad_start, 3),
                   "end": round(float(seg.end) - cand.pad_start, 3),
                   "text": seg.text,
                   "state": finder.state[pos]}
            if getattr(seg, "speaker_original", None) is not None:
                row["speaker_original"] = seg.speaker_original
            conversation.append(row)
        m = cand.metrics
        out = {
            "id": clip_id,
            "audio": f"audio/{clip_id}.wav",
            "start": round(cand.start - cand.pad_start, 3),
            "end": round(cand.end - cand.pad_start, 3),
            "duration": round(audio_seconds, 3),
            "source_start": round(cand.pad_start, 3),
            "source_end": round(cand.pad_end, 3),
            "orig_spans": [{"start": round(a, 3), "end": round(b, 3)}
                           for a, b in timeline.spans_to_original(cand.pad_start, cand.pad_end)],
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
            "trimmed_by_model": judged.trimmed,
            "noise": cand.noise,
            "music_patched_share": cand.music_patched_share,
            "conversation": conversation,
            "transcript": f"metadata/{clip_id}.txt",
        }
        if self.cfg.stereo:
            out["audio_2ch"] = f"audio/{clip_id}_2ch.wav"
            out["channels_2ch"] = {
                "left": "A", "right": "B", "method": "time_gated",
                "note": ("Same samples and length as the mono file. Each speaker is "
                         "heard only during their own turns; where the two speak at "
                         "once both channels carry the same mixture."),
            }
        return out

    # -- the pass ---------------------------------------------------------------------
    def run(self, transcripts, *, timeline, noise, music_map, waveform,
            sample_rate: int, out_dir: str, base_name: str) -> ClipRun:
        """Find, judge and write the clips of one recording."""
        cfg = self.cfg
        finder = DialogueClipFinder(transcripts, timeline, noise, music_map, cfg)
        result = ClipRun()
        report = {"prompt_version": CLIP_PROMPT_VERSION, "skipped": None,
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
            self._log(f"[clips] {base_name}: nothing to cut ({report['skipped']})")
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
            clip_id = f"{re.sub(r'[^A-Za-z0-9._-]+', '_', base_name)}_conv_{number:06d}"
            lo, hi = cut_bounds(waveform, sample_rate, cand.pad_start, cand.pad_end,
                                cfg.zero_cross_ms)
            if hi - lo < sample_rate:
                self._log(f"[clips] {clip_id}: audio shorter than a second; skipped")
                continue
            # One cut, two files: both start on the same sample and are exactly the
            # same length, so they can be played against each other.
            raw = np.array(waveform[lo:hi], dtype=np.float32, copy=True)
            mono = fade_edges(raw.copy(), sample_rate, cfg.fade_ms)
            try:
                sf.write(os.path.join(audio_dir, f"{clip_id}.wav"), mono,
                         sample_rate, subtype="PCM_16")
                if cfg.stereo:
                    stereo = fade_edges(self._two_channel(finder, cand, raw, sample_rate, lo),
                                        sample_rate, cfg.fade_ms)
                    sf.write(os.path.join(audio_dir, f"{clip_id}_2ch.wav"), stereo,
                             sample_rate, subtype="PCM_16")
                meta = self._metadata(finder, finder.timeline, clip_id, cand,
                                      item, len(mono) / float(sample_rate))
                with open(os.path.join(meta_dir, f"{clip_id}.json"), "w",
                          encoding="utf-8") as fh:
                    json.dump(meta, fh, ensure_ascii=False, indent=2)
                with open(os.path.join(meta_dir, f"{clip_id}.txt"), "w",
                          encoding="utf-8") as fh:
                    fh.write(self._transcript_text(meta))
            except Exception as exc:                       # pragma: no cover - disk problems
                if self.logger:
                    self.logger.warning(f"[clips] could not write {clip_id}: {exc}")
                continue
            result.clips.append({
                "id": clip_id, "tier": cand.tier, "score": cand.score,
                "audio": meta["audio"], "audio_2ch": meta.get("audio_2ch"),
                "duration": meta["duration"], "topic": item.topic,
                "semantic_score": item.semantic, "trimmed_by_model": item.trimmed,
                "source_start": meta["source_start"], "source_end": meta["source_end"],
                "speaker_ids": meta["speaker_ids"],
                "noise_kind": (cand.noise or {}).get("dominant_kind"),
            })
        report["exported"] = len(result.clips)
        self._log(f"[clips] {base_name}: {len(result.clips)} clip(s) written to {out_dir}")
        return result

    def _log(self, message: str):
        if self.logger:
            self.logger.info(message)
