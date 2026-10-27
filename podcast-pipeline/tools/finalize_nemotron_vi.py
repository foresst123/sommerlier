#!/usr/bin/env python3
"""
Finalize the best Vietnamese Nemotron-3-Diarization upgrade:
  1. Train Layer 4 with Pure Weighted-BCE (from Exp2, which achieved 10.40% DER@0ms, 0.96% Conf%, 90% SpkExact)
  2. Save clean checkpoint to models/nemotron_vi_upgraded.pt
  3. Benchmark with Omi-Tight Pause Bridging to eliminate <0.5s fragmentation while keeping ~10% DER@0ms & ~5.0% DER@0.25s
"""
import copy
import os
import tempfile
import warnings
import soundfile as sf
import torch

from tools.exp2_nemotron_viyt import load_viyt_split, surgical_finetune_head
from tools.exp3_nemotron_viyt import eval_variant
from utils.nemotron_segments import POSTPROCESSING_BASE

warnings.filterwarnings("ignore")


def main():
    test_items, train_items = load_viyt_split()
    work_dir = tempfile.mkdtemp(prefix="viyt_final_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    test_wavs = []
    for idx, arr, sr, ref_turns, dur, aid in test_items:
        wpath = os.path.join(work_dir, f"{idx:03d}.wav")
        sf.write(wpath, arr, sr)
        test_wavs.append((idx, wpath, arr, sr, ref_turns, dur, aid))

    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()

    tuned_stream = {
        "spkcache_len": 384,
        "fifo_len": 60,
        "chunk_len": 340,
        "chunk_right_context": 40,
        "spkcache_update_period": 200,
    }
    tight_vi_pp = {
        "onset": 0.58,
        "offset": 0.48,
        "pad_onset": 0.01,
        "pad_offset": 0.03,
        "min_duration_on": 0.20,
        "min_duration_off": 0.14,
    }
    crisp_vi_pp = {
        "onset": 0.62,
        "offset": 0.52,
        "pad_onset": 0.00,
        "pad_offset": 0.02,
        "min_duration_on": 0.18,
        "min_duration_off": 0.12,
    }

    scout_cache = {}
    results = []

    # 1. Baseline Repo
    results.append(eval_variant(model, test_wavs, "1_Baseline_Repo", {}, POSTPROCESSING_BASE, False, False, 0.0, work_dir, scout_cache))

    # Populate scout_cache with Zero-Train Deep AOSC
    _ = eval_variant(model, test_wavs, "Scout_Pass", tuned_stream, crisp_vi_pp, True, False, 0.05, work_dir, scout_cache)

    # 2. Train Layer 4 with Pure Weighted-BCE (Exp2 winner) and save weights
    surgical_finetune_head(model, train_items, epochs=6, lr=1.5e-3)
    sm = model.sortformer_modules
    os.makedirs("models", exist_ok=True)
    save_path = "models/nemotron_vi_upgraded.pt"
    torch.save({
        "subpixel_upsample": sm.subpixel_upsample.state_dict() if sm.subpixel_upsample is not None else None,
        "first_hidden_to_hidden": sm.first_hidden_to_hidden.state_dict(),
        "single_hidden_to_spks": sm.single_hidden_to_spks.state_dict(),
        "hidden_to_spks": sm.hidden_to_spks.state_dict(),
    }, save_path)
    print(f"[OK] Đã lưu trọng số Weighted-BCE chuẩn nhất tại: {save_path}")

    # 3. Evaluate the saved Weighted-BCE model with Crisp vs Balanced + Omi-Tight Pause Bridging
    results.append(eval_variant(model, test_wavs, "2_Exp2_Crisp+AOSC", tuned_stream, crisp_vi_pp, True, True, 0.12, work_dir, scout_cache))
    results.append(eval_variant(model, test_wavs, "3_Exp2_Balanced+AOSC", tuned_stream, tight_vi_pp, True, True, 0.15, work_dir, scout_cache))
    results.append(eval_variant(model, test_wavs, "4_Exp2_OmiTight+AOSC", tuned_stream, crisp_vi_pp, True, True, "omi_tight", work_dir, scout_cache))

    print("\n" + "=" * 112)
    print("BẢNG CHỐT CẤU HÌNH PRODUCTION CHO NEMOTRON-3 TIẾNG VIỆT (ViYT-Diar Test N=20)")
    print("=" * 112)
    print(f"{'Cấu hình':25s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s} | {'Time':>6s}")
    print("-" * 112)
    for r in results:
        print(f"{r['name']:25s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s} | {r['time_s']:5.1f}s")
    print("=" * 112)


if __name__ == "__main__":
    main()
