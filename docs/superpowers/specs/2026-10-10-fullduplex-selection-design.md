# Full-duplex conversation selection and output

Date: 2026-10-10. Builds on `2026-10-10-transcript-review-design.md` (review flags, select → align → write),
which is implemented and uncommitted in the working tree. Everything here is configurable; thresholds
marked *heuristic* have no published precedent and must stay in config.

## Goal

Every exported excerpt is one meaningful two-person conversation with a fixed **assistant** / **user**
role, clearly annotated full-duplex **events**, filed by **category** (no score tiers), and only
"OK or better" quality is exported. Output is directly usable by `kyutai-labs/moshi-finetune`.

## 1. Finder (LLM) — `services/conversation_export_service.py`

Rewrite `CONVERSATION_FINDER_SYSTEM_PROMPT` (Vietnamese) around these rules; bump
`CONVERSATION_EXPORT_PROMPT_VERSION` to `conversation-export-v9-fullduplex`.

* Partition the whole window 1..N into contiguous regions (unchanged contract), `keep`, `self_contained` 1–5,
  `topic`, `reason`.
* Worth keeping: exactly two speakers; **continuous dialogue flow** (each turn responds to the previous);
  understandable alone (no mid-idea start, closes an idea or a Q/A); real content (not greetings, ads, script
  reading, intro/outro); Vietnamese (a few English words fine); about **30–120 s** computed from timestamps.
* Topic shifts: **allowed** when a participant initiates it (interrupts, asks something else, disagrees) **and
  the other follows**; must split/drop when caused by a seam, music, unrelated pieces glued together, or nobody
  reacts. **Never put a boundary at an interruption**: keep ≥2 turns before and ≥2 after it.
* Must split at: `seam_before`/`seam_inside`, a third speaker, hard review flags (line must be outside kept
  regions), ads/intro/outro, long unanswered monologue. Soft `speaker_phrase` lines may stay inside.
* `self_contained` judges dialogue flow, not "single topic": 5 natural open/close, clear flow, content;
  4 good, slightly loose edge; 3 followable but cut/vague; 1–2 unusable or not Vietnamese.
* Role (kept regions only): **user** = leads (opens topics, asks, requests, redirects); **assistant** = follows and
  responds (answers, explains, confirms, backchannels while listening, yields when interrupted).
  **Not by speaking time** — the assistant may speak less. Low `role_conf` when both ask and answer equally.
* Interruptions (kept regions only): lines where one speaker cuts into the other, with intent
  `topic_shift | clarify | disagree | complete`. Timing/yield is measured by code, not the LLM.
* `summary`: 1–3 Vietnamese sentences, ≤ ~50 words: what the two talk about, how it develops (who cuts in,
  where it turns), how it ends. Refer to roles ("người hỏi"/"người đáp"), never guess names/gender; nothing
  not in the transcript.
* Thinking: short, 3 steps (break points → duration/speaker count per region → decide keep/role/interruptions).
* Output: one JSON array. Every region: `start_line, end_line, keep, self_contained, topic, reason`.
  Kept regions add `assistant` ("A"/"B" — the window's speaker letters), `role_conf` (0..1), `role_why`,
  `interruptions: [{"line": int, "intent": str}]`, `summary`.
* Worked examples in the prompt must be updated to the new fields; their topics stay in
  `_FINDER_EXAMPLE_TOPICS` (copied example → refused); a summary equal to an example summary → `summary: null`.

Parsing (`parse_finder_reply`): tolerate missing new fields (older cached replies): `assistant` → None,
`role_conf` → None, `interruptions` → [], `summary` → None. Interruption lines outside the region or unknown
intents are dropped (counted in the report). Summary trimmed at the last sentence end within ~60 words.

## 2. Validator gates (code) — `utils/conversation_selection.py`

Reject (with reason) unless all hold: `self_contained ≥ min_semantic (4)`; no invalid / hard-flagged / seam /
lasting-noise segment and no seam or lasting noise between segments; duration in `[min_seconds, max_seconds]`;
exactly 2 speakers; each speaker ≥ `min_turns_each` real turns; soft flags ≤ `max_soft_flags` (2);
noise/music limits as today; score ≥ `min_score`. Score stays (components unchanged, −5 per soft flag) but is
a **gate only**. Then `pick_non_overlapping` as today.

## 3. Events and role check (code, after word alignment) — new `utils/conversation_events.py`

Pure functions over the selected, word-aligned segments (times relative to the excerpt):

* IPU: a speaker's speech split where silence > `ipu_gap` (0.2 s) — dGSLM.
* Overlap: both speakers active ≥ `overlap_min` (0.2 s).
* `backchannel`: listener utterance ≤ `backchannel_max_seconds` (1.0) and ≤ `backchannel_max_words` (2),
  text (folded) in the backchannel word list (reuse `BACKCHANNELS` in conversation_selection), the other
  speaker has speech within `backchannel_context` (1.0 s) before and after and continues — VAP / Full-Duplex-Bench.
* `merged_backchannel`: a soft `speaker_phrase` review flag whose words have alignment times
  (from `review_flags[].phrase` matched to `seg.words`).
* `interruption` (*heuristic*): listener starts while the other is speaking, overlap ≥ `overlap_min`, not a
  backchannel. Response of the interrupted speaker: `yield` if they stop within `yield_within_seconds` (1.5),
  else `continue`; `adapt` when they stop and their next turn is the reply to the interrupter (LLM-marked line
  present). Intent taken from the LLM `interruptions` entry on that line, else `null`.
* `competitive_overlap`: overlap ≥ `overlap_min`, not a backchannel, nobody stops within 1.5 s.
* `turn_switch`: speaker change without overlap; record the gap.
* Each event: `type, by (assistant|user), of/during (assistant|user), start, end, overlap, text, intent,
  response, yield_after` as applicable.

Role: speaker letter from the LLM → speaker id. Behaviour check: if the assigned assistant is interrupted and does
**not** yield while the user yields when interrupted, or the assistant starts ≥2 `topic_shift` interruptions,
subtract 0.2 from `role_conf` per violation. `role_conf ≥ role_min_conf` (0.7) and no violation → `role: "locked"`;
otherwise `role: "symmetric"` (`swappable: true`). If `assistant` is missing, pick the speaker who is interrupted
and yields more, else the one with more turns that answer the other's question; `role: "symmetric"`.
`keep_symmetric: false` drops symmetric excerpts.

## 4. Category (replaces score tiers)

Priority: `full_duplex/interruption` (≥1 interruption) → `full_duplex/backchannel` (≥ `min_backchannels` (2)
backchannels, no interruption) → `half_duplex` (no overlap ≥ `half_duplex_max_overlap` (0.2 s) and no
backchannel) → otherwise not exported (reason `no_category`).

Separation gate: for the two `full_duplex/*` categories, wherever voices overlap (overlap spans and
backchannels that overlap) the tracks must be `strict_separation_tracks`, separated share ≥ 0.9, min
similarity ≥ 0.6, no zeroed failure inside the excerpt (reuse `_judge_overlap`). Fail → `quarantine/`
(reason recorded) when `keep_quarantine` (true), else not exported. `half_duplex` accepts `time_gated`.

`tags` = every event type present among {interruption, backchannel, merged_backchannel, competitive_overlap};
`event_counts` = counts of all event types.

## 5. Output layout and files — `conversation_export_service.write`

```
<out>/
  full_duplex/interruption/<name>/   full_duplex/backchannel/<name>/   half_duplex/<name>/   quarantine/<name>/
  manifest.jsonl
```
`<name>` = `<safe base name>_<int source start>s`. `_clear_previous` must clear these folders (and old
`tier_*` ones) from an earlier run of the same file; the manifest is per output dir and rewritten each run
(one line per exported excerpt of that file — per-file output dirs already exist).

Per excerpt:
* `stereo_2ch.wav`: channel 0 (left) = assistant, channel 1 (right) = user by default
  (`assistant_channel: "left"`, moshi-finetune convention).
* `assistant.wav`, `user.wav` (mono, same length/rate), `mixture.wav`.
* `stereo_2ch.json`: moshi-finetune format `{"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...]}` with
  the **assistant's** words only, times relative to the excerpt.
* `conversation.json`: existing content adapted to roles (`speakers: {"assistant": id, "user": id}`,
  `channels: {"left": ..., "right": ..., "method": ...}`, turns with `role`), plus `category`, `tags`,
  `event_counts`, `events`, `role`, `role_conf`, `role_why`, `swappable`, `topic`, `summary`,
  `quality` (self_contained, score, soft_flags, gates passed / quarantine reason), review flags, source and
  `orig_spans`, verification (read-back checks + sha256, adapted to the new file names).
* `transcript.txt`: one line per turn `[mm:ss.s] [user|assistant] text`, backchannels marked `(backchannel)`,
  interruptions `>> (interruption/<intent>)`, merged backchannel `[merged_bc: …]`.
* `manifest.jsonl` line: `{"path": "<category>/<name>/stereo_2ch.wav", "duration", "category", "tags",
  "event_counts", "role", "swappable", "topic", "summary"}`.

## 6. Config (`ConversationSelectionConfig`, a100 profile values)

New keys with defaults: `role_min_conf 0.7`, `keep_symmetric true`, `keep_quarantine true`,
`assistant_channel "left"`, `ipu_gap 0.2`, `overlap_min 0.2`, `backchannel_max_seconds 1.0`,
`backchannel_max_words 2`, `backchannel_context 1.0`, `yield_within_seconds 1.5`, `min_backchannels 2`,
`half_duplex_max_overlap 0.2`, `max_soft_flags 2`, `context_turns_around_interruption 2`.
a100 profile: `max_seconds 120`, `min_turns_each 3`, `min_score 70`. Keep `overlap_first`,
`overlap_good_*` working or remove them consistently (they are superseded by the category + separation gate).

## 7. Testing

Fake LLM / synthetic aligned segments only. Cover: parsing new fields + old replies; each event definition
(backchannel incl. context and word count, interruption yield/continue/adapt, competitive overlap, half duplex);
role check (locked, symmetric, violations, missing assistant); category priority incl. interruption+backchannel
→ interruption with both tags; quarantine on bad separation; channel order and `stereo_2ch.json` content;
manifest lines; summary trimming/null. Existing tests asserting `tier_*` folders or `speaker_A/B` names must be
updated to the new contract. Baseline: the full suite has ~99 pre-existing failures on this machine; add none.
