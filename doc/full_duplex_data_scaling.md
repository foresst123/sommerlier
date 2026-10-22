# Full-duplex data scaling — plan + findings

Date: 2026-08-12
Author: Tuan Dinh (context: pivoting vi-Sommelier R&D from half-duplex Qwen2.5-Omni SFT to full-duplex VN modeling).

## TL;DR

- **We have ~5.3 h of loose full-duplex-ready audio from 650 pulled clips.** Every filter tightening below that loses another order of magnitude (STRICT config = 0.35 h). **Skip the STRICT setting** — the yield is not worth the compute.
- **The pipeline is not the bottleneck.** GPU throughput is ~1 min wall / clip on 2×A100, ROVER + Sortformer + Demucs + FA + PANNS all work. We processed 650 clips end-to-end.
- **The bottleneck is source pool + per-clip yield.** Two independent problems:
  1. **Source availability**: we've drained Have A Sip + Liêu Hà Trinh (the two channels that give clean 2-speaker interviews). Remaining channels are shorter, monologue, or multi-speaker.
  2. **Per-clip yield for full-duplex**: only ~0.5 min of FD-clean audio per 4-min raw clip. To hit 100 h clean FD we'd need ~12,000 raw clips — an order of magnitude beyond what our source pool supports.
- **Realistic target: 20 h loose FD in 4-6 weeks.** Enough for CPT of a small full-duplex adapter, not for training a model from scratch.

## What "full-duplex" needs from the data (unlike SFT)

| | Half-duplex SFT (Qwen2.5-Omni) | Full-duplex PT/CPT (Moshi-family) |
|---|---|---|
| Sample unit | Discrete turn record | Continuous stereo audio |
| Turn count | ≥ 6 useful | Not the right metric — duration is |
| Time gap tolerance | ≤ 2.5s between turns | ≤ 1.0s (ideally ≤ 0.5s) |
| Duration per sample | ~30s aggregate turns | ≥ 30s continuous (Moshi uses 300s) |
| Speaker count | 2 required | 2 required, one per channel |
| BGM tolerance | Some (music-heavy clips excluded) | Very low — polluted channel breaks separation |
| **Reference scale (SoTA)** | ~500 h fine-tune (Qwen2.5-Omni-VN pilots) | ~20,000 h PT (Moshi/Fisher), ~5-20 h CPT (J-Moshi) |

## Current state (650 clips processed, on HF `tuanamz/vi-sommelier-v0`)

Filter profile table (from `scripts/build_training_conversations.py` on the full 650):

| Profile | min-turns | max-gap | min-dur | dyadic-only | exclude-BGM | Convs | Turns | Hours |
|---|---|---|---|---|---|---|---|---|
| Half-duplex SFT | 6 | 2.5s | — | ✅ | ✅ | 590 | 5,831 | **8.17 h** |
| Full-duplex loose | 2 | 1.0s | 30s | ✅ | ✅ | 337 | 2,099 | **5.29 h** |
| Full-duplex STRICT | 2 | 0.5s | 60s | ✅ | ✅ | 16 | 64 | **0.35 h** |

Note: previous "19 h" figure was the loose (min-turns=6, max-gap=2.5s) config WITHOUT dyadic-only or BGM-exclude. Dyadic + BGM cuts it by ~2×.

### Why STRICT is a trap
Even 500 ms of gap between turns is common in podcast conversation (Whisper's segment boundaries naturally add 200-500ms padding). 1-min continuous is beyond the 4-min raw slice length once you count gaps. Dropping from loose → strict loses **15× the data** for training gains that don't exist (Moshi tolerates gaps; its stereo alignment isn't gap-free either).

**Rule for our pipeline**: full-duplex-ready = `min-dur 30s, max-gap 1.0s, dyadic-only, exclude-bgm`. Stop below that.

## What stops us from scaling further

### Blocker 1: Source pool exhaustion
Channels tapped so far (top 5):
- **Vietcetera (Have A Sip)**: ~200 episodes tapped out of 1,358 videos, but non-podcast content (shorts, teasers, non-conversational) dominates the untapped rest
- **Liêu Hà Trinh**: ~30 episodes tapped from 436 videos; other series (music, monologue) don't fit
- **unlock fm**: 22 tapped from 48
- **VIETSUCCESS**: ~40 tapped from 1,024 (heavy tail of untapped, but only ~200 are true interview format)
- **KLT (travel vlogs)**: 80 tapped, but yield is low — vlogs are mostly single-speaker narration

**Untapped channels we identified but haven't enumerated cleanly**:
- Ta Đi Tây (diaspora interviews) — 81 clips planned, handle broken, need to find right one
- Được/Mất, GNKNN, Nhật Ký Ban Công, Have A Chat GPT — handles broken, need re-discovery
- Quán Thanh Xuân, Bơ Đi Mà Sống, Tâm Sự Bí Mật — enumerated but small pools (44, 1, 58)

**Realistic ceiling**: ~2,000 more raw clips accessible from public VN podcast YouTube. That's ~130 h raw → ~16 h loose FD extra (adds to current 5.3 h → ~21 h loose FD total).

**Beyond that**: need paid corpora (Fisher-equivalent for VN doesn't exist), commercial partnerships (Vinpearl, VTV back-catalog), or synthetic dialogue generation (LLM + VN TTS in duplex mode).

### Blocker 2: Per-clip full-duplex yield
From the 650 already-processed clips:
- **43 h raw → 5.3 h loose FD = ~12% yield**
- Why so low? A 240s podcast clip contains ~15-30 turns. Only ~2-3 contiguous runs of ≥6 turns fit both the gap and dyadic constraints. Interruptions, host preambles, ad reads, and multi-guest segments all break contiguity.

Yield levers we haven't fully pushed:
- **Longer source clips**: current windows are 4 min ([300s, 540s] slice). Pulling 10-15 min windows would give more shot at long contiguous stretches. Costs ~3× storage + pipeline time.
- **Merge cross-clip runs from same episode**: currently each clip processed independently. If we pull sequential windows from the same episode, we can stitch. Non-trivial (need epsiode-level bookkeeping).
- **Loosen dyadic filter**: allow 3-speaker moments to be dropped mid-run instead of dropping the whole run. Requires per-frame masking (feeds into task #68 assembler).

### Blocker 3: My chain-watcher fragility
Every SSH session drop kills my background watchers. Recovery required manual poke. This isn't a data-availability problem but a workflow-quality problem — see also [[feedback_verify_launched_jobs]] memory.

Fix: run watchers as `nohup ... &` on the cluster itself (not through SSH), have them touch a status file, and I check that file rather than holding a long SSH tunnel.

### Blocker 4 (moved from earlier): yt-dlp brittleness
Fixed 2026-08-12: upgraded to 2026.07.04 via `uv tool install yt-dlp`, added explicit `--js-runtimes deno:/path`. VIETSUCCESS URLs now pull cleanly.

## What we DO have that will help

1. **Sortformer per-frame speaker probs (.npy)** — already computed on all 650 clips. This is exactly what a 2-channel assembler needs to know which speaker is active at each 80ms frame.
2. **Demucs vocals-only track** — the per-turn MP3s ARE Demucs-cleaned already. No music leakage to worry about.
3. **Gated MossFormer2 outputs** — 99 co-speech windows separated. Feeds the overlap regions of the stereo audio.
4. **Force-aligned word timing** — 90%+ segments have word-level timestamps. Useful for the assembler to snap boundaries.
5. **`build_training_conversations.py`** — produces the run manifests the assembler consumes.

## Proposed roadmap (4-6 weeks)

### Week 1 (now)
- **Build `build_full_duplex_2channel.py`** (task #68, in progress). Input: FD-loose conversation JSONs. Output: stereo WAV (L=SPEAKER_00, R=SPEAKER_01) + JSON with turn timing on the stereo timeline.
- **Batch 3 pull recovery + push v3 to HF** with the yt-dlp fix. Adds ~150 clips, +1-2 h loose FD.

### Week 2
- **Channel re-enumeration**: find correct handles for Ta Đi Tây, Được/Mất, GNKNN, Have A Chat GPT, and 3 new candidates (search YouTube for `#vnpodcast`, `#tienghihoi`, VN Spotify Vietnam creators).
- **Draft batch 4 (~300 clips)** aimed at yield: prioritize channels with proven dyadic content (skip vlogs).

### Weeks 3-4
- **Extend source clip windows to 10 min** for a subset — measure whether that lifts per-clip FD yield >2×.
- **Process batches 4-6**. Target: 1,000 total clips, ~10-13 h loose FD.

### Weeks 5-6
- **First full-duplex model probe**: LoRA-adapt Moshi (English pre-trained) to VN using the ~15 h stereo output. This tests whether our data is CPT-viable at all — kill criterion if perplexity doesn't move.

## Kill criteria (don't burn time on the following)
- **PT from scratch**: not with any sommelier extension. 5-8 h → 15-20 h max is fine-tune territory only.
- **STRICT (60s / 0.5s gap) filter**: skip. Yield <1% is worse than random-cropping loose FD runs.
- **Chasing >100 h clean FD via YouTube alone**: doesn't exist. Talk to VTV/Vietcetera about corpus licensing if we hit the ceiling.

## Open questions for user

1. Is Moshi-VN CPT the target, or Qwen2.5-Omni-Talker adaptation (which is half-duplex)?
2. Do we want to spend batches 4-6 on **breadth** (many channels for style diversity) or **depth** (few channels, long episodes for stitchable stretches)?
3. Should the 2-channel assembler try to fill the "silent speaker" channel with gated-MF2 output during overlap, or leave silence (simpler, cleaner)?
