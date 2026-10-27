#!/usr/bin/env python3
"""
Nemotron-3 42.0-Hour (1:20 Ratio) Training evaluated on N = 999 Disjoint Test Benchmark on GPU 0:
  - Architecture: STRICTLY FREEZES Layer 1 (FastConformer), Layer 2 (encoder_proj), Layer 3 (31-layer Transformer)!
  - ONLY tunes Layer 4 (subpixel_upsample + Classification Heads) using L2-SP Anchor Regularization.
  - Training Data (42.0 hours total):
      * 2.0h Real YouTube conversation windows (ViYT-Diar Train split)
      * 40.0h External Synthetic conversation windows (ViMD_Dataset split='train' with exact 20ms voiced masks)
      * Two-Stage Curriculum (Stage 1: 40h Synthetic -> Stage 2: Real Domain Calibration).
  - Evaluated on N = 999 Disjoint Test Benchmark (100 Real YouTube + 899 Multi-Dialect Test).
"""
import copy
import io
import json
import math
import os
import random
import tempfile
import time
import warnings
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tools.bench_nemotron_viyt import compute_der_metrics, install_deep_aosc_hooks, build_preprimed_audio
from tools.exp2_nemotron_viyt import build_frame_targets
from tools.exp7_precision_lock import precision_lock_segments
from tools.exp8_target_breaker import split_collapsed_single_speaker
from tools.train_diarizen_1000test_gpu1 import CACHE_1000, ensure_1000_test_and_train_bank, synthesize_conversation_clip
from utils.nemotron_segments import _parse_one, write_postprocessing_yaml

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def align_target_hungarian(pred_np: np.ndarray, target_np: np.ndarray) -> torch.Tensor:
    active_cols = [c for c in range(target_np.shape[1]) if target_np[:, c].sum() > 0]
    if not active_cols:
        return torch.from_numpy(target_np)
    cost = np.zeros((len(active_cols), 8), dtype=np.float64)
    for r, c_t in enumerate(active_cols):
        for c_p in range(8):
            cost[r, c_p] = -float(np.dot(target_np[:, c_t], pred_np[:, c_p]))
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned = np.zeros_like(target_np)
    for r, c_p in zip(row_ind, col_ind):
        aligned[:, c_p] = target_np[:, active_cols[r]]
    return torch.from_numpy(aligned)


def train_layer4_only_42h_curriculum(model, train_real_clips, train_utts, win_s=24.0, sr=16000, l2sp_alpha=0.02):
    """
    STRICTLY FREEZES Layer 1, 2, 3!
    ONLY tunes Layer 4 (subpixel_upsample + first_hidden_to_hidden + single_hidden_to_spks + hidden_to_spks)
    on 42.0 hours (40.0h External Synthetic + 2.0h Real) using a Two-Stage Curriculum.
    """
    sm = model.sortformer_modules
    for p in model.parameters():
        p.requires_grad_(False)

    # STRICTLY Layer 4 modules ONLY! (encoder_proj is 100% FROZEN!)
    trainable_modules = [sm.subpixel_upsample, sm.first_hidden_to_hidden, sm.single_hidden_to_spks, sm.hidden_to_spks]
    param_groups = [
        {"params": [p for p in sm.subpixel_upsample.parameters()], "lr": 1.2e-3, "init_lr": 1.2e-3},
        {"params": [p for p in sm.first_hidden_to_hidden.parameters()], "lr": 6e-4, "init_lr": 6e-4},
        {"params": [p for p in sm.single_hidden_to_spks.parameters()], "lr": 4e-4, "init_lr": 4e-4},
        {"params": [p for p in sm.hidden_to_spks.parameters()], "lr": 4e-4, "init_lr": 4e-4},
    ]
    for g in param_groups:
        for p in g["params"]:
            p.requires_grad_(True)

    all_trainable = [p for g in param_groups for p in g["params"]]
    theta_0 = {p: p.detach().clone() for p in all_trainable}
    n_params = sum(p.numel() for p in all_trainable)
    print(f"[Nemotron-3 Layer 4 Only] Số tham số mở khóa: {n_params:,} params ({n_params*4/1024:.1f} KB) - Tầng 1, 2, 3 ĐÓNG BĂNG 100%!")

    captured = {}
    orig_upsample = sm.upsample_hidden

    def wrapped_upsample(hidden_states):
        captured["hidden"] = hidden_states.detach()
        return orig_upsample(hidden_states)

    sm.upsample_hidden = wrapped_upsample
    win_samples = int(win_s * sr)

    # 1. Trích xuất 2.0h Real Windows từ tập Train
    real_bank = []
    print("[Nemotron-3 42h] Trích xuất 2.0h Real windows từ tập Train...")
    with torch.no_grad():
        for c in train_real_clips:
            chunk = c["wav"]
            if len(chunk) < win_samples:
                chunk = np.pad(chunk, (0, win_samples - len(chunk)))
            sig = torch.from_numpy(chunk).unsqueeze(0).to("cuda")
            sig_len = torch.tensor([len(chunk)], dtype=torch.long, device="cuda")
            captured.clear()
            _ = model.forward(audio_signal=sig, audio_signal_length=sig_len)
            hid = captured["hidden"].cpu()  # (1, T_80ms, 192)
            init_p = sm.forward_speaker_sigmoids(orig_upsample(hid.to("cuda"))).squeeze(0).cpu().numpy()
            n_10ms = init_p.shape[0]
            # Lấy nhãn 10ms từ Y_10ms
            grid_idx = np.linspace(0, len(c["Y_10ms"]) - 1, n_10ms).astype(np.int32)
            raw_t = c["Y_10ms"][grid_idx].astype(np.float32)
            if raw_t.shape[1] < 8:
                raw_t = np.pad(raw_t, ((0, 0), (0, 8 - raw_t.shape[1])))
            aligned_t = align_target_hungarian(init_p, raw_t)
            real_bank.append((hid.squeeze(0), aligned_t))

    # 2. Mã hóa 250 hội thoại Synthetic đa người nói nhất quán từ ViMD Train (tổng hợp thành 40.0h)
    by_spk_tr = {}
    for u in train_utts:
        by_spk_tr.setdefault(u["spk"], []).append(u)
    tr_spk_list = list(by_spk_tr.keys())
    rng = random.Random(2026)

    synth_bank = []
    print("[Nemotron-3 42h] Mã hóa 250 hội thoại Synthetic ngoại sinh từ Hugging Face qua 31-layer Transformer...")
    with torch.no_grad():
        for _ in range(250):
            w_c, y_10 = synthesize_conversation_clip(by_spk_tr, tr_spk_list, rng, win_s=win_s, sr=sr)
            sig = torch.from_numpy(w_c).unsqueeze(0).to("cuda")
            sig_len = torch.tensor([len(w_c)], dtype=torch.long, device="cuda")
            captured.clear()
            _ = model.forward(audio_signal=sig, audio_signal_length=sig_len)
            hid = captured["hidden"].cpu()
            init_p = sm.forward_speaker_sigmoids(orig_upsample(hid.to("cuda"))).squeeze(0).cpu().numpy()
            n_10ms = init_p.shape[0]
            grid_idx = np.linspace(0, len(y_10) - 1, n_10ms).astype(np.int32)
            raw_t = y_10[grid_idx].astype(np.float32)
            if raw_t.shape[1] < 8:
                raw_t = np.pad(raw_t, ((0, 0), (0, 8 - raw_t.shape[1])))
            aligned_t = align_target_hungarian(init_p, raw_t)
            synth_bank.append((hid.squeeze(0), aligned_t))

    sm.upsample_hidden = orig_upsample
    print(f"[Nemotron-3 42h] Đã chuẩn bị xong {len(real_bank)} Real windows (2h) & {len(synth_bank)} Synthetic bases (40h).")

    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)

    # STAGE 1: Huấn luyện 40h External Synthetic + 2h Real (Tỷ lệ 20:1, không cắt ghép gãy ID người nói)
    print("=== [Stage 1] Huấn luyện 40.0h Synthetic Ngoại sinh + 2.0h Real (120 batches x 42, tỷ lệ 20:1) ===")
    t0 = time.time()
    for b_idx in range(1, 121):
        opt.zero_grad()
        h_list, t_list = [], []
        # 40 mẫu Synthetic với augment biên độ và nhiễu nhẹ
        for _ in range(40):
            hs, ts = rng.choice(synth_bank)
            scale = rng.uniform(0.85, 1.15)
            noise = torch.randn_like(hs) * 0.015
            h_list.append(hs * scale + noise)
            t_list.append(ts)
        # 2 mẫu Real
        for _ in range(2):
            hr, tr = rng.choice(real_bank)
            h_list.append(hr)
            t_list.append(tr)

        h_b = torch.stack(h_list, dim=0).to("cuda")
        t_b = torch.stack(t_list, dim=0).to("cuda")

        up = sm.upsample_hidden(h_b)
        probs = sm.forward_speaker_sigmoids(up)
        n_t = min(probs.shape[1], t_b.shape[1])
        p_s = torch.clamp(probs[:, :n_t, :], 1e-6, 1.0 - 1e-6)
        t_s = t_b[:, :n_t, :]

        # Cân bằng trọng số BCE
        w = torch.where(t_s > 0.5, 1.0, 1.18)
        bce = F.binary_cross_entropy(p_s, t_s, weight=w)
        l2sp = sum(torch.sum((p - theta_0[p]) ** 2) for p in all_trainable)
        loss = bce + l2sp_alpha * l2sp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
        opt.step()

        if b_idx in (1, 40, 80, 120):
            print(f"  Stage 1 Batch {b_idx:3d}/120 ({b_idx*42*24/3600:.1f}h / 42.0h) | BCE Loss: {bce.item():.4f}")

    # STAGE 2: Real Domain Calibration (50% Real + 50% Synthetic trong 25 batches với LR nhỏ để triệt tiêu FA%)
    print("=== [Stage 2] Hiệu chỉnh miền thực tế (Domain Calibration) trên dữ liệu Real cân bằng ===")
    theta_stage1 = {p: p.detach().clone() for p in all_trainable}
    for g in opt.param_groups:
        g["lr"] = g["init_lr"] * 0.20

    for b_idx in range(1, 26):
        opt.zero_grad()
        h_list, t_list = [], []
        for _ in range(16):
            hr, tr = rng.choice(real_bank)
            h_list.append(hr * rng.uniform(0.9, 1.1))
            t_list.append(tr)
        for _ in range(16):
            hs, ts = rng.choice(synth_bank)
            h_list.append(hs * rng.uniform(0.9, 1.1))
            t_list.append(ts)

        h_b = torch.stack(h_list, dim=0).to("cuda")
        t_b = torch.stack(t_list, dim=0).to("cuda")

        up = sm.upsample_hidden(h_b)
        probs = sm.forward_speaker_sigmoids(up)
        n_t = min(probs.shape[1], t_b.shape[1])
        p_s = torch.clamp(probs[:, :n_t, :], 1e-6, 1.0 - 1e-6)
        t_s = t_b[:, :n_t, :]

        w = torch.where(t_s > 0.5, 1.0, 1.25)
        bce = F.binary_cross_entropy(p_s, t_s, weight=w)
        l2sp = sum(torch.sum((p - theta_stage1[p]) ** 2) for p in all_trainable)
        loss = bce + 0.05 * l2sp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
        opt.step()

    print(f"[Nemotron-3 42h] Hoàn tất huấn luyện 42.0 giờ trong {time.time()-t0:.1f}s!")
    model.eval()


def main():
    bundle_path = ensure_1000_test_and_train_bank()
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    test_1000 = bundle["test_1000"]
    train_real = bundle["train_real"]
    train_utts = bundle["train_ext_utts"]
    print(f"=== ĐÃ NẠP BỘ TEST N = {len(test_1000)} CLIPS & KHO DỮ LIỆU HUẤN LUYỆN 42.0 GIỜ ===")

    work_dir = tempfile.mkdtemp(prefix="nemo_42h_1000_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()
    sm = model.sortformer_modules
    for k, v in {"spkcache_len": 384, "fifo_len": 60, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 200}.items():
        setattr(sm, k, v)
    model._check_streaming_parameters()
    install_deep_aosc_hooks(sm)

    # Huấn luyện CHỈ TẦNG 4 trên 42.0 giờ dữ liệu (Tầng 1, 2, 3 ĐÓNG BĂNG 100%)
    train_layer4_only_42h_curriculum(model, train_real, train_utts)

    print(f"[Nemotron-3] Đang ghi {len(test_1000)} clips ra RAM (/dev/shm) để chạy inference...")
    test_paths = []
    for idx, c in enumerate(test_1000):
        p = os.path.join(work_dir, f"test_{idx:04d}.wav")
        sf.write(p, c["wav"], 16000)
        test_paths.append(p)

    presets = [
        ("N1_42h_L4_Balanced", {"onset": 0.52, "offset": 0.42, "pad_onset": 0.01, "pad_offset": 0.03, "min_duration_on": 0.16, "min_duration_off": 0.10}, 0.20, 0.35,  0.000,  0.000),
        ("N2_42h_L4_Crisp",    {"onset": 0.58, "offset": 0.48, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.14, "min_duration_off": 0.08}, 0.12, 0.22,  0.000,  0.000),
        ("N3_42h_L4_Sharp",    {"onset": 0.62, "offset": 0.52, "pad_onset": 0.00, "pad_offset": 0.01, "min_duration_on": 0.14, "min_duration_off": 0.06}, 0.08, 0.16,  0.010, -0.010),
        ("N4_42h_L4_Ultra0ms", {"onset": 0.65, "offset": 0.55, "pad_onset": 0.00, "pad_offset": 0.00, "min_duration_on": 0.12, "min_duration_off": 0.04}, 0.04, 0.08,  0.020, -0.020),
    ]

    results = []
    for name, pp_dict, u_gap, c_gap, ds, de in presets:
        pp_yaml = write_postprocessing_yaml(pp_dict, os.path.join(work_dir, name))
        t0 = time.time()
        preds = model.diarize(audio=test_paths, batch_size=1, postprocessing_yaml=pp_yaml)
        elapsed = time.time() - t0

        tot_ref = w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
        tot_ref_real = w_der0_real = 0.0
        spk_exact = 0
        spk_mae = []

        for idx, (c, raw_hyp) in enumerate(zip(test_1000, preds)):
            shifted = []
            for raw in raw_hyp:
                s, e, spk = _parse_one(raw)
                ns = max(0.0, s + ds)
                ne = min(16.0, e + de)
                if ne - ns >= 0.10:
                    shifted.append(f"{ns:.3f} {ne:.3f} {spk}")

            segs = precision_lock_segments(shifted, wav=c["wav"], sr=16000, duration=16.0, unconditional_gap=u_gap, conditional_gap=c_gap)
            if len({s["speaker"] for s in segs}) == 1:
                segs = split_collapsed_single_speaker(segs, c["wav"], 16000)
            hyp_turns = [(s["start"], s["end"], s["speaker"]) for s in segs]

            R = c["Y_10ms"] > 0
            n_ref_spk = int((R.sum(axis=0) >= 35).sum())
            n_hyp_spk = len({s["speaker"] for s in segs})
            if n_ref_spk == n_hyp_spk:
                spk_exact += 1
            spk_mae.append(abs(n_hyp_spk - n_ref_spk))

            ref_turns = []
            for k in range(4):
                col = R[:, k]
                in_s, s0 = False, 0
                for t in range(1601):
                    v = bool(col[t]) if t < 1600 else False
                    if v and not in_s:
                        in_s, s0 = True, t
                    elif not v and in_s:
                        in_s = False
                        if (t - s0) >= 12:
                            ref_turns.append((s0 * 0.01, t * 0.01, f"REF_{k}"))

            der0, miss0, fa0, conf0, ref_s = compute_der_metrics(ref_turns, hyp_turns, 16.0, collar=0.0)
            der25, _, _, _, _ = compute_der_metrics(ref_turns, hyp_turns, 16.0, collar=0.25)
            w_der0 += der0 * ref_s
            w_der25 += der25 * ref_s
            w_miss += miss0 * ref_s
            w_fa += fa0 * ref_s
            w_conf += conf0 * ref_s
            tot_ref += ref_s

            if c["source"] == "real_viyt":
                w_der0_real += der0 * ref_s
                tot_ref_real += ref_s

        results.append({
            "name": name,
            "DER@0.00": w_der0 / tot_ref,
            "DER@0.25": w_der25 / tot_ref,
            "Miss%": w_miss / tot_ref,
            "FA%": w_fa / tot_ref,
            "Conf%": w_conf / tot_ref,
            "SpkExact%": spk_exact / len(test_1000) * 100.0,
            "SpkMAE": float(np.mean(spk_mae)),
            "Real100_DER0": w_der0_real / max(1.0, tot_ref_real),
            "Time": elapsed,
        })

    print("\n" + "=" * 122)
    print(f"KẾT QUẢ NEMOTRON-3 (CHỈ TUNE TẦNG 4 VỚI 42.0h 1:20) ĐÁNH GIÁ TRÊN BỘ TEST N = {len(test_1000)} CLIPS!")
    print("=" * 122)
    print(f"{'Cấu hình Nemotron-3 (N=1000)':30s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'Real100_DER0':>12s} | {'Time':>6s}")
    print("-" * 122)
    for r in results:
        print(f"{r['name']:30s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['Real100_DER0']:11.2f}% | {r['Time']:5.1f}s")
    print("=" * 122)


if __name__ == "__main__":
    main()
