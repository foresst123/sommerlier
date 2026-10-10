# Transcript review → conversation selection → word alignment

Date: 2026-10-10. Branch: `solid-architecture` (work stays uncommitted until reviewed).

## Goal

Split what the conversation finder currently does in one LLM pass into three stages:

1. **Review** (new step `transcript_review`): one LLM pass over the transcript that only
   *flags* extremely unreasonable speaker labels or ASR text. It never changes `speaker`,
   `text` or times. It replaces `speaker_relabel` in the a100 profile (relabel code stays).
2. **Select** (`conversation_exports`): the finder sees the flags; the validator rejects
   ranges with hard flags and penalises soft ones.
3. **Align, then write**: word alignment runs only on the segments of the selected
   conversations, after selection; an excerpt whose alignment is incomplete is dropped.

New order in `services/pipeline_service.py`:
`asr → (refinement, off on a100) → transcript_review → select → align selected → write`.
The whole-file `word_alignment` step is no longer a precondition of selection.

## Flags (shared contract)

`schemas/transcript.py` → `TranscriptSegment.review_flags: Optional[List[dict]] = None`
(None = review did not run on this segment; [] = reviewed, nothing found).

Each flag:

```python
{"kind": "speaker_phrase" | "speaker_turn" | "asr_hallucination" | "asr_garbage",
 "phrase": Optional[str],   # verbatim substring of seg.text; None = whole segment
 "conf": float,             # 0..1
 "why": str}                # <= 20 words
```

| kind | scope | meaning | stage-2 effect |
|---|---|---|---|
| `speaker_phrase` | phrase (required) | words inside a turn that clearly belong to the other speaker, typically a merged backchannel ("…điềm tĩnh **đúng rồi** và cẩn thận…") | **soft**: score penalty, range kept |
| `speaker_turn` | whole segment | the whole turn clearly belongs to the other speaker (answers own question, reversed address) | **hard**: range rejected |
| `asr_hallucination` | phrase or whole | ASR boilerplate/hallucination not caught by `algorithms/asr/hallucination.py`, unrelated to the conversation | **hard** |
| `asr_garbage` | phrase or whole | words that do not read as Vietnamese at all | **hard** |

Not flagged: spelling, tone marks, homophones, punctuation, misspelt proper names, and anything
`algorithms/asr/hallucination.py` already removes (boilerplate list, repetition, foreign script,
too many words for <1 s). Only *extremely* unreasonable things are flagged.

Validation in code: `phrase` must be a substring of the segment text (whitespace/case-folded
compare), else the flag is dropped and counted in the report as `phrase_not_found`. A flag
below `min_confidence` (default 0.7) is dropped. Segments locked by `SpeakerRelabelService.is_locked`
(BSS overlap) **may** be flagged — flagging changes no data.

## Stage 1 API — `services/transcript_review_service.py` (agent A)

Reuse the window/ask/parse machinery of `services/speaker_relabel_service.py`
(`utils.transcript_windows.build_windows/format_line/line_number`, `utils.llm_batches.ask_in_batches`,
`utils.llm_json`), lines numbered 1..N and speakers lettered A, B, … like relabel.

```python
REVIEW_PROMPT_VERSION = "review-v1"
FLAG_KINDS = ("speaker_phrase", "speaker_turn", "asr_hallucination", "asr_garbage")
HARD_FLAG_KINDS = ("speaker_turn", "asr_hallucination", "asr_garbage")

@dataclass
class ReviewResult:
    flags_by_index: Dict[str, List[dict]]   # str(seg.index) -> flags (every reviewed index present, [] if clean)
    windows: int; failed_windows: int; skipped: Optional[str]
    rejected: Dict[str, int]                # reason -> count
    def to_report(self) -> dict: ...

def apply_review_flags(transcripts, flags_by_index) -> int   # sets seg.review_flags; returns #segments with >=1 flag

class TranscriptReviewService:
    def __init__(self, llm, logger=None, window_tokens=3500, overlap_segments=30,
                 max_window_segments=None, min_confidence=0.7, thinking=False,
                 max_new_tokens=None): ...
    checkpoint_namespace: str   # property; includes REVIEW_PROMPT_VERSION and settings
    def review(self, segments) -> ReviewResult   # computes only, does not mutate
```

Model answer: JSON array of `{"i": line, "kind": ..., "phrase": str|null, "conf": float, "why": str}`.
Overlapping windows: one window decides each segment (same rule as relabel), duplicates merged.

## Stage 2 — `utils/conversation_selection.py`, `services/conversation_export_service.py` (agent B)

* `ConversationSelectionConfig`: `review_soft_penalty: float = 5.0` (score points per soft flag in a
  range, capped), `review_hard_reject: bool = True`.
* Validator (`ConversationSelectionFinder.evaluate` path): a range containing a segment with a hard
  flag is rejected with reason `review_flag_<kind>`; soft flags lower the score and are recorded in
  `Candidate.metrics["review_soft_flags"]`. Segments with `review_flags is None` are treated as clean.
* `require_word_alignment` no longer gates selection: the finder/validator skip it. It is enforced
  after alignment (below).
* `_finder_line` shows flags, e.g. `review=speaker_turn` / `review=speaker_phrase:"đúng rồi"`
  (phrase truncated). Finder prompt: explain the field, tell the model to place boundaries so hard-flagged
  lines are outside kept regions. Remove the free-text paragraph "có thể mỗi dòng có từ không hợp lý…"
  and the step-3 clause "kiểm tra cả câu mỗi vùng để tránh label sai" — that job moved to stage 1.
  Bump `CONVERSATION_EXPORT_PROMPT_VERSION` to `conversation-export-v8-review-flags`.
* Split `ConversationExportService.run` into:

  ```python
  def select(self, transcripts, *, timeline, noise, music_map, waveform, sample_rate,
             base_name, speech_segments=None) -> ConversationSelection
      # finder + validator + pick_non_overlapping; no files written
  def write(self, selection, *, waveform, sample_rate, out_dir, base_name,
            speech_segments=None, separation_service=None) -> ConversationExportRun
      # re-checks has_complete_word_alignment on every segment of each selected excerpt when
      # cfg.require_word_alignment; drops failures (report["dropped_unaligned"]); then plans/writes
  ConversationSelection.segments_to_align() -> List[TranscriptSegment]  # unique, in order
  ```

  `run(...)` stays as `write(select(...))` so existing callers/tests keep working.
* `conversation.json` rows carry `review_flags` when present.

## Stage 3 — wiring (agent C)

* `services/pipeline_service.py`: new step `transcript_review` (opt-in via
  `opt_in_step_enabled`, checkpointed like `_relabel_speakers`: store `flags_by_index` + report,
  re-apply on resume; LLM loading/unload handled like relabel). Order:
  review → `select` → `word_alignment_svc.align(selection.segments_to_align(), audio, speech_segments)`
  (checkpoint namespace `checkpoint_namespace_for(subset)`) → `write`.
  When `conversation_exports` is on, the whole-file alignment before it is skipped; if
  `word_alignment` is on and `conversation_exports` is off, the old whole-file alignment still runs.
* `--stop_after transcript_review` supported; `stage_output_service` writes the review report.
* `main.py`: build `TranscriptReviewService` from `models.transcript_review` settings (fallback to
  `models.relabel` window settings) sharing the resident refinement LLM like relabel.
* `config.json` a100: `steps.transcript_review: true`, `steps.speaker_relabel: false`,
  `models.transcript_review` block.

## Testing

Fake LLMs only (no GPU), `rtk proxy python -m pytest … -q`. Baseline failures exist
(≈35 in `test_separation_window` etc., ≈65 under `-k conversation`); a change must add none.
