# 6-week plan — Vietnamese conversational dataset for reception robot (v2, goal-first)

## North-star goal

> Build a **Vietnamese conversational processing pipeline** and a **POC dataset with full-duplex labels** reliable enough for the reception-robot team to use downstream (train/evaluate dialogue model). Focus is **proving the pipeline scales**, not shipping a production dataset.

Every task in 6 weeks should answer: *"which dataset metric does this improve, and which robot use case does it serve?"* — this is a priority sanity check, not a reason to drop work.

**Current bottleneck: no Vietnamese conversational data.** No data → can't train/eval the robot model. So **collecting + filtering data is priority #1**, even when it doesn't directly touch pipeline code (source selection, crawling, manual review are core work, not side-quests).

## Dataset success criteria (measurable)

| Metric | v1 (end of Wk 5) | v2 (end of Wk 6) |
|---|---|---|
| Raw audio processed | ≥ 20h | ≥ 50h |
| Usable segments after filter | ≥ 1,000 | ≥ 3,000 |
| Backchannel labels (`ừ/dạ/vâng/...`) | ≥ 200 | ≥ 600 |
| Labeled overlap segments | ≥ 150 | ≥ 400 |
| Hallucination rate on held-out 200 | ≤ 8% | ≤ 3% |
| Manual review benchmark | — | 400 samples (100 per type) |
| Domain coverage | 2-person + podcast | + hospitality conv |

Concrete numbers to be confirmed with PM in week 1; table above is a starting default.

## Module → robot use case mapping (BRD)

| Pipeline module | Robot use case it serves |
|---|---|
| VAD captures `ừ/dạ` | §2.5 Small talk, §2.30 Complaint — robot doesn't pause when guest backchannels |
| Stable diarization | §2.6 Tiếp đón, §2.15 Đi lại hỏi trải nghiệm — distinguishes guests from staff |
| ASR with diacritics + fillers kept | §2.4 Tri thức chung, §2.27 Tư vấn đặt dịch vụ — correct intent understanding |
| Overlap/interruption labels | §2.3 Fallback, §2.31 Trò chuyện cảm xúc — robot knows when it's interrupted |
| Quality filter | All — garbage data teaches the model the wrong persona |

---

## Week 1 — POC: pick sources, extract clips, measure failure modes

- **Goal contribution:** evidence-based ranking of which modules to fix first → Weeks 2–5 don't tune blindly. Also proves the pipeline **can** scale to VN before further investment.
- **Hypothesis (H1):** the stock pipeline has ≥4 distinguishable failure modes on VN; the severity ranking may reorder Weeks 2–5 (the plan is not strictly linear).
- **Kill criterion:** if >50% of clips fail in >2 modules simultaneously and root causes can't be separated → escalate to PM in week 1, do not enter Week 2.

**Source-selection criteria (priority order):**

1. Closeness to robot use cases: hospitality > general talk > entertainment > news.
2. Spontaneous, not scripted — must contain real fillers, backchannels, interruptions.
3. ≥2 speakers (monologue can't test diarization).
4. Audio-quality diversity (clean podcast → vlog → noisy) and accent coverage Bắc/Trung/Nam.
5. Public + license recorded in CSV.

**Channel candidates (~2 channels per group, ≤10 channels total):**

| Group | Candidates | Stress on module |
|---|---|---|
| Hospitality / tourism | Vinpearl YouTube, travel vlogs (Khoai Lang Thang), hotel-review channels | Domain vocab, ambient noise |
| 2-speaker interview (clean) | Vietcetera *Have A Sip*, *The Quoc Khanh Show* | Backchannel, turn-taking baseline |
| Multi-host talk show | VTV *Cà Phê Sáng*, *Quán Thanh Xuân* | Overlap, 3+ speakers, music intros |
| Lifestyle / everyday talk | *Tâm Sự Bí Mật*, couple podcasts | Dense fillers, real backchannels |
| Customer service / reception | VN hospitality training videos, role-play TikTok | Directly maps to robot use case (prioritize if findable) |

**Failure-mode stress matrix — pick 3 clips × ~30–60s per row, ~24 clips ≈ 18 min total (runs in <1h on a free T4):**

| Stress condition | Predicted failure |
|---|---|
| Clean 1-on-1 interview, no music | Baseline; if it fails → fundamental issue |
| Interview with heavy `dạ`/`vâng` | VAD/diarization drops short segments |
| 3+ speakers, fast turns | Diarization fragments, overlap mislabeled |
| Background music (show intro / vlog BGM) | Music removal artifacts, speaker confusion |
| Outdoor / noisy | ASR hallucination, VAD over-triggers |
| VN–EN code-switching | ASR language detect flips, drops diacritics |
| Regional accent (2 clips per region: Bắc/Trung/Nam) | WER spike, missing diacritics |
| Audio fade / end mid-sentence | End-of-audio hallucination (e.g. `Em giỡn`, `Cảm ơn các bạn đã xem`) |

**Deliverable:** `data/clip_selection.csv` — one row per clip:
`clip_id, source_channel, source_url, license, stress_condition, predicted_failure, observed_failure, failure_severity (1-5), notes`

End of Week 1: severity ranking determines the order of Weeks 2–5 (e.g. if hallucination is severity 5 across 80% of clips → Week 4 jumps before Week 2). POC passes ⇔ ≥6/8 stress conditions show a distinguishable failure mode.

## Week 2 — VAD/chunking for short backchannels

- **Goal contribution:** ensure backchannels (`ừ/dạ/vâng/à`) survive the first stage — if they're dropped here, they're lost for the whole dataset.
- **Hypothesis (H2):** default Silero VAD drops ≥15% of backchannels <300ms; tuning `threshold` + `min_speech_duration` + `padding` lifts recall to ≥95% while raising false positives by ≤10%.
- **Experiment:** 100 hand-annotated backchannels from Week 1. Sweep a 3-axis grid. Metrics: backchannel recall, FP rate, boundary IoU.
- **Kill criterion:** if the best config still misses >10% of backchannels → VAD isn't the bottleneck; propose model replacement or a backchannel-specific detector. **Don't keep tuning.**
- **Deliverable:** `vad_chunking_vi_report.md` + best config committed to the pipeline.

## Week 3 — Music handling + diarization

- **Goal contribution:** stable speaker labels so overlap/turn-taking in Week 5 has meaning.
- **Hypothesis (H3a — music):** music removal *before* diarization improves DER by ≥10% on audio with background music, but drops ≥5% of quiet backchannels → hybrid (diar on original, ASR on processed) is optimal.
- **Hypothesis (H3b — diar):** post-processing (merge same-speaker, drop too-short except backchannels, link speakers across chunks) reduces fragmentation by ≥30%.
- **Experiment:** ablate 3 configs (A/B/C from the original plan). Annotate 30–60 min as reference. Metrics: DER, backchannel loss rate, speaker fragmentation count.
- **Kill criterion:** if all 3 configs have DER >25% → diarization is the dataset bottleneck; reduce Week 5 scope to "single-speaker monologue + backchannel detection", not full multi-speaker.
- **Deliverable:** `music_ablation_report.md`, `diarization_vi_report.md`, final config.

## Week 4 — Vietnamese ASR + hallucination filter

- **Goal contribution:** transcripts accurate enough that overlap/interruption labels aren't noisy, and that the robot team can use them as training data later.
- **Hypothesis (H4):** Whisper-large-v3 with `language=vi` is enough for text content; the 3-model ROVER MoE (Whisper + PhoWhisper + ChunkFormer) adds value primarily as a **disagreement signal → `needs_review`**, not as a text vote.
- **Experiment:** compare 3 models individually vs ROVER vote on 200 segments with manual transcripts. Metrics: WER with diacritics, hallucination rate (rule-based), filler retention rate.
- **Kill criterion:** if PhoWhisper or ChunkFormer can't beat Whisper by >2% WER → drop from the production pipeline, keep only as a `needs_review` signal.
- **Deliverable:** `asr_vi_report.md` + hallucination filter + final ASR selection strategy.
- *Note:* the `--lang` flag and Whisper VN routing are already in code (see `main_original_ASR_MoE.py` around line 3082, `asr_MoE` branch for `lang=vi`).

## Week 5 — Full-duplex labels + quality filter (dataset v1)

- **Goal contribution:** this is the week that **directly** produces the labels the robot needs — backchannel, overlap, interruption, turn_gap.
- **Hypothesis (H5):** rule-based detection (per the original plan's definitions) reaches precision ≥85% on 200 manually reviewed samples for each of the 3 labels (backchannel / overlap / interruption); the rest flows into `needs_review`.
- **Experiment:** run the rules across all Week 4 output. Sample 200 per label type, review manually, compute P/R.
- **Kill criterion:** if precision <70% for any label → that label isn't production-ready; remove it from v1 or default-tag every sample with `confidence=low`.
- **Deliverable:** `dataset_vi_fullduplex_v1.jsonl` meeting the v1 row of the criteria table + `full_duplex_labeling_guideline_vi.md`.

## Week 6 — End-to-end v2 + benchmark + handoff

- **Goal contribution:** package so the robot team can use it without coming back to ask the intern.
- **Hypothesis (H6):** with the best configs from Weeks 2–5, a 1-week compute run hits the v2 row of the criteria table.
- **Experiment:** rerun end-to-end. Measure deltas against the Week 1 baseline (Δ P/R, Δ hallucination, Δ DER). Build a 400-sample manually-reviewed benchmark (100 clean turn-taking, 100 overlap, 100 backchannel, 100 noisy-reject).
- **Kill criterion:** if v2 misses 80% of the bar → ship v1 with a clear gap doc instead of forcing v2.
- **Deliverable:** `dataset_vi_fullduplex_v2.jsonl` (or Parquet), data card, `final_report.md` (baseline → improvement → remaining gaps), demo on 1 audio (speaker timeline + transcript + labels + JSON).

---

## Top risks

1. **Free-tier GPU bottleneck** — Colab/Kaggle T4 + 30h/week quota caps throughput. Realistic ~10–20h of audio processed/week; v2 target may need to drop.
2. **POC dataset ≠ production dataset** — tens of hours is too little to actually train full-duplex models. v2 is mainly for **evaluation + proving the pipeline**; real training will need more data collected later.
3. **PhoWhisper/ChunkFormer not yet benchmarked on VN podcast/hospitality conversation** — actual performance may be worse than the paper. H4's kill criterion covers this.
4. **Manual annotation bottleneck** — 400 benchmark samples + ≥600 backchannel labels is ~2 person-weeks if the intern works solo. Either add helpers, use pre-labeling, or reduce benchmark scope.
