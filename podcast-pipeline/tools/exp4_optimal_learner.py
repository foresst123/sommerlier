#!/usr/bin/env python3
"""
Experiment 4: Upgraded Learning Algorithm for Nemotron-3 Vietnamese Adaptation
  1. Hungarian-Aligned Target Matching (prevents arrival-order channel inversion on mid-file windows)
  2. L2-SP Regularization: anchors weights to NVIDIA's pretrained theta_0 (||theta - theta_0||^2)
  3. Differential Learning Rates:
       - subpixel_upsample (80ms -> 10ms boundary conv): lr = 4e-4
       - first_hidden_to_hidden: lr = 1.5e-4
       - single_hidden_to_spks / hidden_to_spks: lr = 8e-5
  4. Mini-batch Gradient Accumulation (batch_size = 16 windows) + Linear Warmup + Cosine Decay + Grad Clipping
"""
import copy
import math
import os
import random
import tempfile
import time
import warnings
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tools.bench_nemotron_viyt import (
    compute_der_metrics,
    install_deep_aosc_hooks,
    build_preprimed_audio,
    omi_pause_bridging_segments,
)
from tools.exp2_nemotron_viyt import load_viyt_split, build_frame_targets
from tools.exp3_nemotron_viyt import eval_variant
from utils.nemotron_segments import POSTPROCESSING_BASE

warnings.filterwarnings("ignore")


def align_target_to_preds(probs_1d: torch.Tensor, target_1d: torch.Tensor) -> torch.Tensor:
    """
    Hungarian-align columns of target_1d (T, 8) to match the speaker channel ordering of probs_1d (T, 8).
    Prevents label-inversion gradients when a window boundary has ambiguous first-arrival order.
    """
    p_np = probs_1d.detach().cpu().numpy()  # (T, 8)
    t_np = target_1d.detach().cpu().numpy()  # (T, 8)
    # Only permute active ground-truth speaker columns
    active_cols = [c for c in range(t_np.shape[1]) if t_np[:, c].sum() > 0]
    if not active_cols:
        return target_1d
    cost = np.zeros((len(active_cols), 8), dtype=np.float64)
    for idx_r, c_t in enumerate(active_cols):
        for c_p in range(8):
            # Maximize dot product (overlap agreement) between target column c_t and predicted channel c_p
            cost[idx_r, c_p] = -float(np.dot(t_np[:, c_t], p_np[:, c_p]))
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned = torch.zeros_like(target_1d)
    for r, c_p in zip(row_ind, col_ind):
        aligned[:, c_p] = target_1d[:, active_cols[r]]
    return aligned


def train_with_optimal_learner(model, train_items, epochs=12, batch_size=16, l2sp_alpha=0.05, neg_weight=1.22, win_s=24.0):
    sm = model.sortformer_modules
    for p in model.parameters():
        p.requires_grad_(False)

    # Save exact pretrained weights theta_0 for L2-SP regularization
    trainable_modules = [
        ("upsample", sm.subpixel_upsample, 4e-4),
        ("hidden", sm.first_hidden_to_hidden, 1.5e-4),
        ("spk1", sm.single_hidden_to_spks, 8e-5),
        ("spk2", sm.hidden_to_spks, 8e-5),
    ]
    param_groups = []
    theta_0 = {}
    for name, mod, lr in trainable_modules:
        if mod is None:
            continue
        mod_params = []
        for p_name, p in mod.named_parameters():
            p.requires_grad_(True)
            mod_params.append(p)
            theta_0[p] = p.detach().clone()
        param_groups.append({"params": mod_params, "lr": lr, "initial_lr": lr, "name": name})

    # Extract frozen Transformer hidden states across training windows
    captured = {}
    orig_upsample = sm.upsample_hidden

    def wrapped_upsample(hidden_states):
        captured["hidden"] = hidden_states.detach()
        return orig_upsample(hidden_states)

    sm.upsample_hidden = wrapped_upsample
    dataset_windows = []

    print("[Optimal Learner] Đang trích xuất đặc trưng trên 80 file Train...")
    with torch.no_grad():
        for _, arr, sr, ref_turns, dur, _ in train_items:
            win_samples = int(win_s * sr)
            starts = [0.0] if dur <= win_s else [0.0, min(dur - win_s, dur * 0.35), min(dur - win_s, dur * 0.70)]
            for st_s in sorted(set(starts)):
                i0 = int(st_s * sr)
                chunk = arr[i0:i0 + win_samples]
                if len(chunk) < int(4.0 * sr):
                    continue
                sig = torch.from_numpy(chunk).unsqueeze(0).to("cuda")
                sig_len = torch.tensor([len(chunk)], dtype=torch.long, device="cuda")
                captured.clear()
                _ = model.forward(audio_signal=sig, audio_signal_length=sig_len)
                if "hidden" not in captured:
                    continue
                hid = captured["hidden"].cpu()  # (1, T_80ms, 192)
                # Initial prediction from frozen head to align target columns once
                init_up = orig_upsample(hid.to("cuda"))
                init_probs = sm.forward_speaker_sigmoids(init_up).cpu()[0]
                n_10ms = init_probs.shape[0]
                shifted_turns = [(s - st_s, e - st_s, spk) for s, e, spk in ref_turns if e > st_s and s < st_s + win_s]
                raw_target = build_frame_targets(shifted_turns, n_10ms, step=0.01)
                aligned_target = align_target_to_preds(init_probs, raw_target).unsqueeze(0)
                dataset_windows.append((hid, aligned_target))

    sm.upsample_hidden = orig_upsample
    print(f"[Optimal Learner] Đã căn chỉnh Hungarian cho {len(dataset_windows)} windows. Bắt đầu tối ưu (Batch={batch_size}, L2-SP={l2sp_alpha})...")

    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)  # L2-SP replaces naive weight_decay to 0
    total_steps = epochs * math.ceil(len(dataset_windows) / batch_size)
    warmup_steps = max(2, int(0.15 * total_steps))
    step_idx = 0

    all_params = [p for g in param_groups for p in g["params"]]
    rng = random.Random(42)

    for ep in range(1, epochs + 1):
        indices = list(range(len(dataset_windows)))
        rng.shuffle(indices)
        ep_loss = 0.0
        n_batches = 0

        for b_start in range(0, len(indices), batch_size):
            batch_ids = indices[b_start:b_start + batch_size]
            step_idx += 1
            # Linear warmup + Cosine decay schedule
            if step_idx <= warmup_steps:
                scale = step_idx / float(warmup_steps)
            else:
                prog = (step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))
                scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale

            opt.zero_grad()
            batch_bce = 0.0
            for idx_w in batch_ids:
                hid, target = dataset_windows[idx_w]
                hid = hid.to("cuda")
                target = target.to("cuda")
                up = sm.upsample_hidden(hid)
                probs = sm.forward_speaker_sigmoids(up)
                n_t = min(probs.shape[1], target.shape[1])
                p_slice = torch.clamp(probs[:, :n_t, :], 1e-6, 1.0 - 1e-6)
                t_slice = target[:, :n_t, :]
                weight = torch.where(t_slice > 0.5, 1.0, neg_weight)
                bce = F.binary_cross_entropy(p_slice, t_slice, weight=weight)
                batch_bce = batch_bce + bce / len(batch_ids)

            # L2-SP Regularization: anchor to NVIDIA pretrained weights theta_0
            l2sp_loss = sum(torch.sum((p - theta_0[p]) ** 2) for p in all_params)
            loss = batch_bce + l2sp_alpha * l2sp_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            opt.step()

            ep_loss += float(batch_bce.item())
            n_batches += 1

        if ep in (1, 4, 8, epochs):
            print(f"  Epoch {ep:2d}/{epochs} | Aligned BCE: {ep_loss / max(1, n_batches):.4f} | LR scale: {scale:.2f}")

    model.eval()
    os.makedirs("models", exist_ok=True)
    save_path = "models/nemotron_vi_upgraded.pt"
    torch.save({
        "subpixel_upsample": sm.subpixel_upsample.state_dict() if sm.subpixel_upsample is not None else None,
        "first_hidden_to_hidden": sm.first_hidden_to_hidden.state_dict(),
        "single_hidden_to_spks": sm.single_hidden_to_spks.state_dict(),
        "hidden_to_spks": sm.hidden_to_spks.state_dict(),
    }, save_path)
    print(f"[OK] Đã lưu trọng số Optimal-Learner tại: {save_path}")


def main():
    test_items, train_items = load_viyt_split()
    work_dir = tempfile.mkdtemp(prefix="viyt_exp4_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

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
    pp_balanced = {"onset": 0.54, "offset": 0.44, "pad_onset": 0.01, "pad_offset": 0.03, "min_duration_on": 0.20, "min_duration_off": 0.14}
    pp_crisp    = {"onset": 0.58, "offset": 0.48, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.20, "min_duration_off": 0.12}
    pp_tight    = {"onset": 0.62, "offset": 0.52, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.18, "min_duration_off": 0.10}

    scout_cache = {}
    results = []

    # 1. Baseline Repo
    results.append(eval_variant(model, test_wavs, "1_Baseline_Repo", {}, POSTPROCESSING_BASE, False, False, 0.0, work_dir, scout_cache))

    # Populate scout_cache with Zero-Train Deep AOSC
    _ = eval_variant(model, test_wavs, "Scout_Pass", tuned_stream, pp_crisp, True, False, 0.08, work_dir, scout_cache)

    # 2. Train with Optimal Learner (Hungarian-Aligned + L2-SP + Differential LR + Mini-batch 16 + Warmup)
    train_with_optimal_learner(model, train_items, epochs=12, batch_size=16, l2sp_alpha=0.05, neg_weight=1.22)

    # 3. Evaluate across 3 Hysteresis presets
    results.append(eval_variant(model, test_wavs, "2_OptLearn_Balanced+AOSC", tuned_stream, pp_balanced, True, True, "omi_tight", work_dir, scout_cache))
    results.append(eval_variant(model, test_wavs, "3_OptLearn_Crisp+AOSC", tuned_stream, pp_crisp, True, True, "omi_tight", work_dir, scout_cache))
    results.append(eval_variant(model, test_wavs, "4_OptLearn_Tight+AOSC", tuned_stream, pp_tight, True, True, 0.10, work_dir, scout_cache))

    print("\n" + "=" * 112)
    print("KẾT QUẢ THỬ NGHIỆM 4: THUẬT TOÁN HỌC HUNGARIAN + L2-SP + DIFFERENTIAL LR + WARMUP (ViYT-Diar Test N=20)")
    print("=" * 112)
    print(f"{'Cấu hình':25s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s} | {'Time':>6s}")
    print("-" * 112)
    for r in results:
        print(f"{r['name']:25s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s} | {r['time_s']:5.1f}s")
    print("=" * 112)


if __name__ == "__main__":
    main()
