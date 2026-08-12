# vi-Sommelier pipeline — operator handoff

> Purpose: teach a new teammate to add source clips, run the full processing pipeline, and land results on HuggingFace. Written 2026-08-12 by Tuan.

**Repo**: https://github.com/tuanad121/sommelier (branch: `main`)
**HF dataset**: https://huggingface.co/datasets/tuanamz/vi-sommelier-v0 (private — request access from Tuan)

Not a design doc — for the "why" see `doc/full_duplex_data_scaling.md`. This is the "how to run it".

## 1. What this pipeline does

Takes a TSV of Vietnamese podcast YouTube URLs → produces per-clip diarized ASR JSON + per-turn audio + a stack of quality sidecars (music detection, force-alignment, speaker attribution, overlap separation). Output lands as one directory per source clip in a HuggingFace private dataset (the canonical one is Tuan's `tuanamz/vi-sommelier-v0`; you'll create and push to your own — see §2). Downstream extractors turn that into training-ready conversations (half-duplex JSONL or full-duplex 2-channel stereo).

Current state: 650 source clips on HF (~28 h raw, ~8 h clean dyadic 6-turn conversations, ~5 h stereo full-duplex). Target: extend to 15-20 h stereo FD via batches 4+.

## 2. Prerequisites

**Repos**
- `github.com/tuanad121/sommelier` — this repo (fork of NAVER Sommelier for Vietnamese)
- Submodule / vendored: `podcast-pipeline/` (the main ASR + diarization pipeline)
- External: `external/clearvoice/` (for gated MossFormer2, has its own venv)

**Access needed**
- HuggingFace: create your own private dataset repo — e.g. `<your-hf-username>/vi-sommelier-batchN` or a shared org repo you own. Then generate a personal WRITE token at https://huggingface.co/settings/tokens.
  - **Do not push to `tuanamz/vi-sommelier-v0` directly.** That's the canonical read-only-to-you dataset. Concurrent writes race; the last push wins and can clobber the other's batches.
  - Sharing your batch back: two options. (a) Push to your own repo, then Tuan does a periodic merge into `tuanamz/vi-sommelier-v0`. (b) You get READ access to `tuanamz/vi-sommelier-v0` so your downstream training can consume both your repo + Tuan's without duplication. Ask Tuan for READ access at minimum.
- Compute: **any Linux box with a GPU works** (single GPU sufficient for slow-mode, 2×A100 recommended for parallel sidecars). The Vinbdi cluster is what Tuan uses (SLURM job 9417 on `dgx-a100-5`). Teammate should get their own cluster account or bring their own compute.

**System dependencies**
- Python 3.10+ (the pipeline is 3.12 in Tuan's setup)
- `ffmpeg` on PATH
- `uv` (Astral) — package + tool manager. `curl -LsSf https://astral.sh/uv/install.sh | sh`
- `yt-dlp` via uv (see below — critical version pin)
- `deno` — JavaScript runtime yt-dlp needs for YouTube's obfuscated player. `curl -fsSL https://deno.land/install.sh | DENO_INSTALL=$HOME/.local sh -s -- -y`
- CUDA 12.x + cuDNN 8 (the pipeline expects `LD_LIBRARY_PATH=$HOME/.local/cudnn8/nvidia/cudnn/lib` — install cuDNN 8 wheel if the system's cuDNN version differs)

## 3. One-time environment setup

```bash
# Clone
git clone git@github.com:tuanad121/sommelier.git
cd sommelier

# Main pipeline venv (uv-managed)
cd podcast-pipeline
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
cd ..

# yt-dlp — MUST be recent, older versions fail on YouTube's obfuscated player
uv tool install --force yt-dlp
# Confirm >= 2026.07.04
~/.local/bin/yt-dlp --version

# deno (Java Script runtime for yt-dlp)
curl -fsSL https://deno.land/install.sh | DENO_INSTALL=$HOME/.local sh -s -- -y
# Confirm binary lands here (pick_clips.py hard-references this path):
ls ~/.local/bin/deno

# HuggingFace token
export HF_TOKEN=hf_...    # WRITE-scope token from https://huggingface.co/settings/tokens
```

Add to `~/.bashrc` / `~/.zshrc` so subsequent shells have them:
```bash
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN="hf_..."
export LD_LIBRARY_PATH="$HOME/.local/cudnn8/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

## 4. The TSV format (source of truth for what gets pulled)

`data/clips_to_pull.tsv` — tab-separated, no header, one row per clip:

```
label   url                                             start  end   channel                              genre
havesip_hieuthuhai_148   https://youtube.com/watch?v=YGBJQsNzcCs   900   1140   Have A Sip #148 (HIEUTHUHAI)   podcast_interview
havesipFULL_kienthuc_x   https://youtube.com/watch?v=abc12345678   0     0      Have A Sip full episode        podcast_interview
```

Columns:
1. **`label`** — must be unique across the whole TSV. Convention: `<channel_prefix>_<slug>_<episode_or_x>`. Filesystem-safe (letters/digits/underscore only). Used as directory/file basename throughout the pipeline.
2. **`url`** — YouTube watch URL. Must include `v=<11-char-id>`.
3. **`start`** — seconds. `0` = beginning.
4. **`end`** — seconds. If `end <= start`, pulls the **whole episode** (added 2026-08-12 for FD scaling — see full_duplex_data_scaling.md).
5. **`channel`** — human-readable channel/episode label (goes into meta.json, useful for downstream filtering by source).
6. **`genre`** — free-text tag. Common values: `podcast_interview`, `travel_vlog`, `talkshow`, `lifestyle_podcast`.

**Adding new clips**: append rows to the TSV, commit, run the pipeline (steps below). `pick_clips.py` is idempotent — it skips any `label` where `<label>.wav` or `<label>.meta.json` already exists.

**Deduplication cache**: `data/processed_urls.tsv` (regenerated after each HF push) lists everything already processed. Use it to filter your candidate list before appending.

## 5. Adding a batch — recommended workflow

### Step 5.1 — Find candidate episodes
Enumerate a YouTube channel:
```bash
yt-dlp --flat-playlist --skip-download \
  --print "%(id)s|%(title)s|%(duration)s" \
  "https://www.youtube.com/@ChannelHandle/videos" \
  > /tmp/candidates_<channel>.txt
```

Filter by title pattern (e.g. "Have A Sip") + duration (want ≥15 min for meaningful content), dedup against `data/processed_urls.tsv`. See `scripts/build_batch1.py` on the cluster (`/tmp/build_batch1.py` or in the repo) for a working template.

### Step 5.2 — Draft the batch TSV
Emit as `<batch>.tsv`, review manually (spot-check 5 rows), then append:
```bash
cp data/clips_to_pull.tsv data/clips_to_pull.tsv.pre_batchN.bak
cat /tmp/batchN.tsv >> data/clips_to_pull.tsv
```

### Step 5.3 — Pull audio
```bash
export PATH=$HOME/.local/bin:$PATH
cd $REPO
nohup uv run python scripts/pick_clips.py \
  --tsv data/clips_to_pull.tsv \
  --out data/raw \
  --workers 6 \
  > data/pick_batchN.log 2>&1 &
disown
echo "pid=$!"
```

**Verify within 60s** (per feedback memory — don't blind-sleep and assume progress):
```bash
sleep 60
tail -10 data/pick_batchN.log     # should see [pull] lines and no repeating [error]
pgrep -f yt-dlp | wc -l           # should be ~= --workers
ls data/raw/*.wav | wc -l          # baseline count before your batch
```

**ETA**: 4-min-slice clip ≈ 60-120s wall each; whole-episode ≈ 3-5 min. 100 slice clips + 6 workers ≈ 30-45 min.

### Step 5.4 — Run the main pipeline (ASR + diarization)
On a machine with GPU. If SLURM:
```bash
# see scripts/launch_pipeline_batch1_fixed.sh for the exact srun template
bash scripts/launch_pipeline_batch1_fixed.sh
```

Non-SLURM (direct GPU access):
```bash
export LD_LIBRARY_PATH=$HOME/.local/cudnn8/nvidia/cudnn/lib:$LD_LIBRARY_PATH
cd podcast-pipeline
nohup uv run python main_original_ASR_MoE.py \
  --input_folder_path $REPO/data/raw \
  --lang vi --vad --dia3 --ASRMoE --whisperx_word_timestamps --initprompt \
  --demucs --sepreformer --no-qwen3omni --merge_gap 2.0 \
  > multi_clip_pipeline.log 2>&1 &
disown
```

**CRITICAL flags** — must match the production config directory naming:
- `--demucs` + `--sepreformer` — music separation + overlap separation
- `--merge_gap 2.0` — MUST be `2.0` not `2` (Python formats int `2` as `"2"` in the dir name, breaks re-runs) — see project memory
- `--no-qwen3omni` — 4th-voter is parked (see project_5way_rover_reverted.md)

Output lands at:
```
data/raw/_final/-sepreformer-True-demucs-True-vad-True-diaModel-dia3-initPrompt-True-merge_gap-2.0-seg_th-0.15-cl_min-10-cl-th-0.5-LLM-case_2/<clip>/
├── <clip>.json                     # main pipeline output (segments + ASR)
└── <clip>/NNNNN_SPEAKER_XX.mp3     # per-turn audio (Demucs-cleaned vocals)
```

**ETA on 2×A100**: ~1 min wall per 4-min clip (~15 min per full-length episode). Idempotent — re-launching skips completed clips.

### Step 5.5 — Sidecars (5 stages, chainable)
Each of these adds a sidecar JSON next to `<clip>.json`. All idempotent (skip clips where their output already exists).

```bash
# 1. Sortformer per-frame speaker probs (fast, ~2 min for 100 clips)
bash scripts/launch_sortformer_all.sh

# 2. BGM (PANNs) + speaker attribution (ECAPA) + speaker merge (sharded across 2 GPUs)
bash scripts/launch_sidecars_new_clips.sh

# 3. Force-alignment (wav2vec2-vi-250h word timing)
bash scripts/launch_force_align_new.sh

# 4. Gated MossFormer2 overlap separation (scans co-speech windows from Sortformer)
bash scripts/launch_gated_mf2_new.sh
```

After each, verify:
```bash
FINAL=data/raw/_final/-sepreformer-True-demucs-True-vad-True-diaModel-dia3-initPrompt-True-merge_gap-2.0-seg_th-0.15-cl_min-10-cl-th-0.5-LLM-case_2
find $FINAL -name sortformer_probs.json | wc -l   # should match # of clip dirs
```

### Step 5.6 — Normalize (assemble JSONL + apply drop filters)
```bash
bash scripts/launch_normalize_all.sh
# writes <clip>.jsonl next to each <clip>.json (one line per kept turn)
```

Drop filters applied here: `phantom` (no Sortformer speaker), `misattributed` (ECAPA disagrees), `has_bgm` (PANNs music > threshold), `multi_speaker` (>2 concurrent). See `scripts/normalize_to_hf_jsonl.py` for exact thresholds.

### Step 5.7 — Push to HuggingFace

**Heads-up**: `scripts/upload_to_hf.py` has `REPO_ID = "tuanamz/vi-sommelier-v0"` hardcoded near the top of the file. **Change it to your own dataset repo before running** — do not push to Tuan's canonical repo (see §2 Access). Also update `DATASET_README` if the README wording no longer fits your batch.

```bash
export HF_TOKEN=hf_...   # WRITE token for YOUR HF dataset repo
cd $REPO
# after editing REPO_ID in scripts/upload_to_hf.py:
uv run --with huggingface_hub python scripts/upload_to_hf.py
# pushes to <your-hf-username>/<your-dataset-name> via symlink-based staging (fast, no disk copy)
```

Confirm your push in browser: https://huggingface.co/datasets/<your-hf-username>/<your-dataset-name>/tree/main/clips

### Step 5.8 — Regenerate deduplication cache
```bash
python3 - <<'PY'
import json, glob
rows = []
for p in sorted(glob.glob("data/raw/*.meta.json")):
    m = json.load(open(p))
    rows.append([m["label"], m["source_url"], m["source_channel"], m["source_genre"],
                 f"{m.get('youtube_start_s',0):.1f}", f"{m.get('youtube_end_s',0):.1f}"])
with open("data/processed_urls.tsv","w") as f:
    f.write("label\turl\tchannel\tgenre\tstart_s\tend_s\n")
    for r in rows: f.write("\t".join(r)+"\n")
print(f"wrote {len(rows)} rows")
PY
git add data/processed_urls.tsv && git commit -m "update processed_urls after batch N"
```

## 6. Building training data from what's already on HF

Two extractors, independent of the pull/pipeline chain:

### Half-duplex SFT conversations (for Qwen-style LoRA)
```bash
python scripts/build_training_conversations.py \
  --root $FINAL \
  --out data/training_conversations_sft \
  --min-turns 6 --max-gap-s 2.5 \
  --exclude-bgm --only-two-speaker \
  --preview-html data/training_conversations_sft/preview.html
```
Emits one JSON per conversation + `summary.tsv` + `preview.html`. Yields ~8 h from current 650 clips.

### Full-duplex 2-channel stereo (for Moshi-style adaptation)
```bash
# Step 1: get FD-loose conversation manifests
python scripts/build_training_conversations.py \
  --root $FINAL \
  --out data/training_conversations_fullduplex \
  --min-turns 2 --max-gap-s 1.0 --min-duration-s 30 \
  --exclude-bgm --only-two-speaker

# Step 2: assemble stereo
python scripts/build_full_duplex_2channel.py \
  --convs-dir data/training_conversations_fullduplex/conversations \
  --clips-root $FINAL \
  --out data/full_duplex_stereo
# writes <conversation_id>.wav (16 kHz stereo, L=first speaker, R=other) + .json meta
```

**Do NOT use** `min-dur 60s + max-gap 0.5s + min-turns anything` (the "STRICT" profile) — yield is <1 %, not worth it. See `doc/full_duplex_data_scaling.md`.

## 7. Storage discipline (important on shared cluster fs)

After each successful HF push, reclaim ~20 GB per batch by removing files that live in the HF snapshot:
```bash
# Raw WAVs + intermediate mkv/webm (already pushed to HF as per-turn MP3s)
find data/raw -maxdepth 1 -name "*.wav" -delete
find data/raw -maxdepth 1 \( -name "*.mkv" -o -name "*.webm" \) -delete

# Wrong-config _final dirs (any dir with `merge_gap-2-` — missing `.0` — is stale)
rm -rf data/raw/_final/*merge_gap-2-*

# HF staging dirs
rm -rf hf_stage_*
```

The `pick_clips.py` idempotence check works even after WAVs are deleted (falls back to `.meta.json` presence). So pruning WAVs doesn't cause re-pulls.

The shared filesystem on Vinbdi runs at 100% usage — leave headroom.

## 8. Known gotchas

| Symptom | Cause / fix |
|---|---|
| `yt-dlp` returns "No title found in player responses" | yt-dlp version too old. Reinstall via `uv tool install --force yt-dlp` (need ≥ 2026.07.04). |
| `yt-dlp` warns "No supported JavaScript runtime" | deno not on PATH. Check `~/.local/bin/deno` exists; pick_clips.py explicitly passes `--js-runtimes deno:/home/tuanda67/.local/bin/deno` — update path if your `$HOME` differs. |
| Output dir name is `merge_gap-2` not `merge_gap-2.0` | Passed `--merge_gap 2` (int) instead of `--merge_gap 2.0` (float). Kill the run, remove the wrong dir, relaunch with `2.0`. |
| Pipeline stuck at "Transcribing" for hours | Usually SLURM job cancelled / GPU OOM. Check log for `CANCELLED` or `Killed`. Restart the pipeline; it's idempotent. |
| SSH transport drops mid-command (exit 255) | VPN / network flakiness. The nohup'd background process on the cluster is unaffected. Reconnect and check state; don't relaunch blindly. |
| `chuyện-trò` / `Tự Tình Lúc 0h` etc. regex missing hits | Vietnamese titles have flexible spacing/punctuation. Use `re.IGNORECASE` + tolerant patterns like `chuy[ệe]n\s*-?\s*tr[òo]`. |
| `SEBT` / age-restricted videos fail | Need YouTube cookies. Skip for now, or set up `--cookies-from-browser`. |

## 9. Where to look when stuck

| Question | Where |
|---|---|
| "Why is turn X dropped?" | `<clip>.json` (raw pipeline output) + sidecars (`bgm_check.json`, `attribution_check.json`, etc.) — normalize logs which filter dropped it |
| "Was this URL already processed?" | `data/processed_urls.tsv` |
| "What filters were applied to build the SFT/FD dataset?" | `doc/full_duplex_data_scaling.md` |
| "Why is Moshi-VN CPT risky?" | `memory/project_backbone_choice_vn.md` |
| "What happened during LoRA training experiments?" | `memory/project_lora_collapse_full_sweep.md` + `doc/lora_training_qwen_omni.md` |
| "How does the Qwen-Omni architecture work / why can't we LoRA the Talker?" | `doc/qwen_omni_architecture.md` + `memory/project_qwen_codec_encoder_blocked.md` |

## 10. Handoff etiquette (from Tuan)

- **Commit style**: no `Co-Authored-By` trailers on commits — attribute to yourself.
- **Cluster file transfers**: always `rsync -a`, never `scp`. Delta + resumable + preserves mtime.
- **Launching cluster jobs**: tail the log within 30-60s to confirm real progress before quoting an ETA. Don't blind-sleep.
- **HF token**: WRITE-scope tokens should not be committed anywhere. Regenerate if leaked.
- **Chuyện với Tuan trước khi**: creating new HF repos, deleting `_final` dirs, rotating tokens, pushing anything that affects downstream users.

Good luck. When you hit something not covered here, add it to this doc.
