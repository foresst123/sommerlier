#!/usr/bin/env python3
"""
Evaluate Nemotron-3 Multi-Layer Fine-Tuning directly on N = 1,000 Disjoint Test Benchmark on GPU 0.
"""
import copy
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
from tools.train_diarizen_1000test_gpu1 import CACHE_1000, ensure_1000_test_and_train_bank
from utils.nemotron_segments import _parse_one, write_postprocessing_yaml

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def finetune_layer2_and_layer4(model, train_items, epochs=6, lr_proj=2e-4, lr_head=8e-4, win_s=24.0):
    sm = model.sortformer_modules
    for p in model.parameters():
        p.requires_grad_(False)

    trainable_groups = [
        {"params": [p for p in sm.encoder_proj.parameters()], "lr": lr_proj},
        {"params": [p for p in sm.subpixel_upsample.parameters()], "lr": lr_head},
        {"params": [p for p in sm.first_hidden_to_hidden.parameters()], "lr": lr_head * 0.5},
        {"params": [p for p in sm.single_hidden_to_spks.parameters()], "lr": lr_head * 0.5},
        {"params": [p for p in sm.hidden_to_spks.parameters()], "lr": lr_head * 0.5},
    ]
    for g in trainable_groups:
        for p in g["params"]:
            p.requires_grad_(True)

    all_trainable = [p for g in trainable_groups for p in g["params"]]
    theta_0 = {p: p.detach().clone() for p in all_trainable}
    n_params = sum(p.numel() for p in all_trainable)
    print(f"[Multi-Layer FT] Đã mở khóa Tầng 2 + Tầng 4: {n_params:,} params ({n_params*4/1024:.1f} KB)")

    sr = 16000
    win_samples = int(win_s * sr)
    train_windows = []

    for item in train_items:
        arr, dur, ref_turns = item[1], item[4], item[3]
        starts = [0.0] if dur <= win_s else [0.0, min(dur - win_s, dur * 0.45)]
        for st_s in starts:
            i0 = int(st_s * sr)
            chunk = arr[i0 : i0 + win_samples]
            if len(chunk) < int(4.0 * sr):
                continue
            shifted = [(s - st_s, e - st_s, spk) for s, e, spk in ref_turns if e > st_s and s < st_s + win_s]
            train_windows.append((chunk, shifted, len(chunk)))

    opt = torch.optim.AdamW(trainable_groups, weight_decay=0.0)
    print(f"[Multi-Layer FT] Bắt đầu huấn luyện {len(train_windows)} windows trên GPU 0...")

    for ep in range(1, epochs + 1):
        random.shuffle(train_windows)
        ep_loss = 0.0
        for chunk, shifted_turns, c_len in train_windows:
            sig = torch.from_numpy(chunk).unsqueeze(0).to("cuda")
            sig_len = torch.tensor([c_len], dtype=torch.long, device="cuda")
            opt.zero_grad()
            logits = model.forward(audio_signal=sig, audio_signal_length=sig_len)
            probs = torch.sigmoid(logits)
            n_t = probs.shape[1]
            raw_target = build_frame_targets(shifted_turns, n_t, step=0.01).to("cuda").unsqueeze(0)
            p_slice = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
            weight = torch.where(raw_target > 0.5, 1.0, 1.25)
            bce = F.binary_cross_entropy(p_slice, raw_target, weight=weight)
            l2sp = sum(torch.sum((p - theta_0[p]) ** 2) for p in all_trainable)
            loss = bce + 0.02 * l2sp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            opt.step()
            ep_loss += float(bce.item())
        print(f"  Epoch {ep}/{epochs} - BCE Loss: {ep_loss / len(train_windows):.4f}")

    model.eval()


def main():
    bundle_path = ensure_1000_test_and_train_bank()
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    test_1000 = bundle["test_1000"]
    print(f"=== ĐÃ TẢI THÀNH CÔNG BỘ TEST N = {len(test_1000)} CLIPS VÀO BÀI ĐÁNH GIÁ NEMOTRON-3 ===")

    work_dir = tempfile.mkdtemp(prefix="nemo_1000test_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    # Lấy 80 file Train từ ViYT-Diar để huấn luyện đa tầng
    from tools.exp2_nemotron_viyt import load_viyt_split
    _, train_items = load_viyt_split()

    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()
    sm = model.sortformer_modules
    for k, v in {"spkcache_len": 384, "fifo_len": 60, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 200}.items():
        setattr(sm, k, v)
    model._check_streaming_parameters()
    install_deep_aosc_hooks(sm)

    # 1. Huấn luyện Tầng 2 (encoder_proj) + Tầng 4
    finetune_layer2_and_layer4(model, train_items, epochs=6, lr_proj=2e-4, lr_head=8e-4)

    # Lưu 1,000 clips ra file tạm để chạy diarize
    print(f"[Nemotron-3] Đang ghi {len(test_1000)} clips ra RAM (/dev/shm) để chạy inference...")
    test_paths = []
    for idx, c in enumerate(test_1000):
        p = os.path.join(work_dir, f"test_{idx:04d}.wav")
        sf.write(p, c["wav"], 16000)
        test_paths.append(p)

    presets = [
        ("N1_Layer2+4_Balanced", {"onset": 0.52, "offset": 0.42, "pad_onset": 0.01, "pad_offset": 0.03, "min_duration_on": 0.16, "min_duration_off": 0.10}, 0.22, 0.38),
        ("N2_Layer2+4_Crisp",    {"onset": 0.58, "offset": 0.48, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.14, "min_duration_off": 0.08}, 0.14, 0.24),
        ("N3_Layer2+4_Sharp",    {"onset": 0.62, "offset": 0.52, "pad_onset": 0.00, "pad_offset": 0.01, "min_duration_on": 0.14, "min_duration_off": 0.06}, 0.08, 0.16),
    ]

    results = []
    for name, pp_dict, u_gap, c_gap in presets:
        pp_yaml = write_postprocessing_yaml(pp_dict, os.path.join(work_dir, name))
        t0 = time.time()
        preds = model.diarize(audio=test_paths, batch_size=1, postprocessing_yaml=pp_yaml)
        elapsed = time.time() - t0

        tot_ref = w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
        tot_ref_real = w_der0_real = 0.0
        spk_exact = 0
        spk_mae = []

        for idx, (c, raw_hyp) in enumerate(zip(test_1000, preds)):
            segs = precision_lock_segments(raw_hyp, wav=c["wav"], sr=16000, duration=16.0, unconditional_gap=u_gap, conditional_gap=c_gap)
            hyp_turns = [(s["start"], s["end"], s["speaker"]) for s in segs]

            # Chuyển nhãn tham chiếu Y_10ms sang ref_turns
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
    print(f"KẾT QUẢ NEMOTRON-3 MULTI-LAYER FINE-TUNING ĐÁNH GIÁ TRÊN BỘ TEST N = {len(test_1000)} CLIPS!")
    print("=" * 122)
    print(f"{'Cấu hình Nemotron-3 (N=1000)':30s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'Real100_DER0':>12s} | {'Time':>6s}")
    print("-" * 122)
    for r in results:
        print(f"{r['name']:30s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['Real100_DER0']:11.2f}% | {r['Time']:5.1f}s")
    print("=" * 122)


if __name__ == "__main__":
    main()
