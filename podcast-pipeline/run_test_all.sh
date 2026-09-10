# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT

#!/bin/bash
set -euo pipefail

# ============================================================
# Audio Processing Pipeline Test Script
# ============================================================
# SepReformer usage:
#   - Default (disabled): sepreformer_flags=(--no-sepreformer)
#   - Enabled: sepreformer_flags=(--sepreformer)
#   - Test both: sepreformer_flags=(--sepreformer --no-sepreformer)
# ============================================================

# List of input folders to process
# Specify the path to the folder containing your audio files below.
folders=(
  "/your/audio/folder"
)

# LLM case list
# case_0 : No LLM post diarization.
# case_2 : LLM post diarization enabled.
llm_cases=(case_0)
korean=(--no-korean)
# VAD & DIA3 flag combinations
vad_flags=(--vad)
#dia3_flags=(--dia3 --no-dia3)
dia3_flags=(--dia3)
# INITPROMPT flag combinations
# initprompt: Injects filler-word prompt.
initprompt_flags=(--no-initprompt)
#initprompt_flags=(--initprompt --no-initprompt)

# Additional parameter combinations
seg_ths=(0.11)
min_cluster_sizes=(11)
clust_ths=(0.5)
#ASRMoE=(--ASRMoE --no-ASRMoE)
ASRMoE=(--ASRMoE)
# DEMUCS flag combinations (background music removal)
# --bs_roformer: Detect background music with PANNs, then extract vocals with BS-RoFormer
# --no-bs_roformer: No background music removal (default)
#bs_roformer_flags=(--bs_roformer)
bs_roformer_flags=(--bs_roformer)
# WhisperX word-level timestamp flags
# --whisperx_word_timestamps: Enable word-level timestamps via WhisperX alignment
# --no-whisperx_word_timestamps: Disable word-level timestamps (default)
whisperx_flags=(--whisperx_word_timestamps)
#whisperx_flags=(--whisperx_word_timestamps --no-whisperx_word_timestamps)
# Qwen3-Omni audio captioning flags
# --qwen3omni: Enable audio caption generation via Qwen3-Omni API
# --no-qwen3omni: Disable audio caption generation (default)
qwen3omni_flags=(--no-qwen3omni)
#qwen3omni_flags=(--qwen3omni --no-qwen3omni)
# Context-aware captioning flags
# --context_caption: Enable captioning using the previous 2 segments as context
# --no-context_caption: Caption each segment individually without context (default)
#context_caption_flags=(--no-context_caption)
#context_caption_flags=(--context_caption --no-context_caption)
# SepReformer overlapping speech separation flags
# --sepreformer: Enable overlapping speech separation using SepReformer
# --no-sepreformer: Disable overlapping speech separation (default)
#sepreformer_flags=(--sepreformer)
sepreformer_flags=(--sepreformer)
# SepReformer overlap threshold (minimum duration to consider as overlap, in seconds)
overlap_thresholds=(0.2)
# Cross-chunk speaker linking similarity threshold
speaker_link_thresholds=(0.6)
# MERGE_GAP combinations
# merge_gaps=(0.5 1 1.5 2)
# 0.5 and 2 produce the same results.
merge_gaps=(2)

# Sortformer segment boundary adjustment (end time -0.08s)
sortformer_param_flags=(--sortformer-param)
sortformer_pad_offset_values=(-0.24)

for folder in "${folders[@]}"; do
  for vad in "${vad_flags[@]}"; do
    for dia3 in "${dia3_flags[@]}"; do
      for initprompt in "${initprompt_flags[@]}"; do
        for llm in "${llm_cases[@]}"; do
          for seg in "${seg_ths[@]}"; do
            for min_cluster in "${min_cluster_sizes[@]}"; do
              for clust in "${clust_ths[@]}"; do
                for merge_gap in "${merge_gaps[@]}"; do
                  for asrmoe in "${ASRMoE[@]}"; do
                    for bs_roformer in "${bs_roformer_flags[@]}"; do
                      for whisperx in "${whisperx_flags[@]}"; do
                        for qwen3omni in "${qwen3omni_flags[@]}"; do
                          for sepreformer in "${sepreformer_flags[@]}"; do
                            for overlap_th in "${overlap_thresholds[@]}"; do
                              for speaker_link_th in "${speaker_link_thresholds[@]}"; do
                                for sortformer_pad_offset in "${sortformer_pad_offset_values[@]}"; do
                                  echo "▶ Folder: ${folder}, ${vad}, ${dia3}, ${initprompt}, LLM=${llm}, seg_th=${seg}, min_cluster_size=${min_cluster}, clust_th=${clust}, merge_gap=${merge_gap}, ${asrmoe}, ${bs_roformer}, ${whisperx}, ${qwen3omni}, ${sepreformer}, ${sortformer_param_flags[*]}, sortformer_pad_offset=${sortformer_pad_offset}, overlap_th=${overlap_th}, speaker_link_th=${speaker_link_th}, korean=${korean}"
                                  /mnt/fr20tb/kyudan/miniforge3/envs/dataset/bin/python main_original_ASR_MoE.py \
                                    --input_folder_path "${folder}" \
                                    ${vad} ${dia3} ${initprompt} ${asrmoe} ${bs_roformer} ${whisperx} ${qwen3omni} ${sepreformer} \
                                    ${sortformer_param_flags[@]} \
                                    --sortformer-pad-offset "${sortformer_pad_offset}" \
                                    --LLM "${llm}" \
                                    --seg_th "${seg}" \
                                    --min_cluster_size "${min_cluster}" \
                                    --clust_th "${clust}" \
                                    --merge_gap "${merge_gap}" \
                                    --overlap_threshold "${overlap_th}" \
                                    --speaker-link-threshold "${speaker_link_th}"\
                                    --opus_decode_workers 16\
                                    --ffmpeg_threads_per_decode 1
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
