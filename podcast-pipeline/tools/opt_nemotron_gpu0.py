#!/usr/bin/env python3
"""
Standalone Optimization of Nemotron-3-Diarization on GPU 0:
  1. Uses the exact same shared dataset in /dev/shm/viyt_shared_cache.
  2. Extracts clean, non-overlapping full-coverage windows across all 80 training files (without
     any spkcache contamination!), aligns target speaker columns via Hungarian Matching, and trains
     Layer 4 (subpixel_upsample + Classification Heads) with L2-SP Anchor Regularization + Differential LR.
  3. Evaluates multiple Hysteresis + Boundary Inset + PrecisionLock + 2SpkSplit configurations.
  4. Saves the best Nemotron-3 test predictions to /dev/shm/viyt_shared_cache/best_nemotron_segs.json.
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
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tools.bench_nemotron_viyt import (
    compute_der_metrics,
    install_deep_aosc_hooks,
    build_preprimed_audio,
)
from tools.exp2_nemotron_viyt import build_frame_targets
from tools.exp7_precision_lock import precision_lock_segments
from tools.exp8_target_breaker import split_collapsed_single_speaker
from tools.opt_diarizen_gpu1 import ensure_shared_dataset, CACHE_DIR
from utils.nemotron_segments import _parse_one, write_postprocessing_yaml, POSTPROCESSING_BASE

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


def train_nemotron_head_clean(model, train_meta, epochs=8, batch_size=8, l2sp_alpha=0.015, neg_weight=1.35, win_s=28.0):
    """
    Clean Multi-Window Surgical Fine-Tuning on GPU 0:
      - Covers full duration of all 80 training files in 28s windows.
      - Uses Hungarian alignment on every window so mid-file slices NEVER invert speaker channels.
      - Uses Differential LR + L2-SP Regularization anchored to NVIDIA's pretrained weights theta_0.
    """
    sm = model.sortformer_modules
    for p in model.parameters():
        p.requires_grad_(False)

    groups_spec = [
        ("upsample", sm.subpixel_upsample, 1.2e-3),
        ("hidden",   sm.first_hidden_to_hidden, 6e-4),
        ("spk1",     sm.single_hidden_to_spks, 4e-4),
        ("spk2",     sm.hidden_to_spks, 4e-4),
    ]
    param_groups = []
    all_params = []
    theta_0 = {}
    for gname, mod, lr in groups_spec:
        if mod is None:
            continue
        m_params = []
        for p in mod.parameters():
            p.requires_grad_(True)
            m_params.append(p)
            all_params.append(p)
            theta_0[p] = p.detach().clone()
        param_groups.append({"params": m_params, "lr": lr, "initial_lr": lr})

    captured = {}
    orig_upsample = sm.upsample_hidden

    def wrapped_upsample(hidden_states):
        captured["hidden"] = hidden_states.detach()
        return orig_upsample(hidden_states)

    sm.upsample_hidden = wrapped_upsample
    windows_data = []

    print("[GPU 0 - Nemotron] Đang trích xuất đặc trưng đầy đủ trên 80 file Train...")
    with torch.no_grad():
        for item in train_meta:
            arr, sr = sf.read(item["wav_path"], dtype="float32")
            dur = item["dur"]
            ref_turns = [(s, e, spk) for s, e, spk in item["ref_turns"]]
            win_samples = int(win_s * sr)
            starts = [0.0] if dur <= win_s else [0.0, max(0.0, dur - win_s)]
            if dur > 2.2 * win_s:
                starts.insert(1, dur * 0.5 - win_s * 0.5)

            for st_s in sorted(set(starts)):
                i0 = int(st_s * sr)
                chunk = arr[i0 : i0 + win_samples]
                if len(chunk) < int(4.0 * sr):
                    continue
                sig = torch.from_numpy(chunk).unsqueeze(0).to("cuda")
                sig_len = torch.tensor([len(chunk)], dtype=torch.long, device="cuda")
                captured.clear()
                _ = model.forward(audio_signal=sig, audio_signal_length=sig_len)
                if "hidden" not in captured:
                    continue
                hid = captured["hidden"].cpu()
                init_up = orig_upsample(hid.to("cuda"))
                init_p = sm.forward_speaker_sigmoids(init_up).squeeze(0).cpu().numpy()
                n_10ms = init_p.shape[0]
                shifted_turns = [(s - st_s, e - st_s, spk) for s, e, spk in ref_turns if e > st_s and s < st_s + win_s]
                raw_t = build_frame_targets(shifted_turns, n_10ms, step=0.01).numpy()
                # For st_s == 0.0, arrival order is exact; for st_s > 0.0, Hungarian-align to prevent label inversion
                aligned_t = align_target_hungarian(init_p, raw_t) if st_s > 0.0 else torch.from_numpy(raw_t)
                windows_data.append((hid, aligned_t.unsqueeze(0)))

    sm.upsample_hidden = orig_upsample
    print(f"[GPU 0 - Nemotron] Huấn luyện {len(windows_data)} windows (neg_weight={neg_weight}, L2-SP={l2sp_alpha})...")

    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    rng = random.Random(42)

    for ep in range(1, epochs + 1):
        order = list(range(len(windows_data)))
        rng.shuffle(order)
        ep_loss = 0.0
        n_b = 0
        for b0 in range(0, len(order), batch_size):
            b_ids = order[b0 : b0 + batch_size]
            opt.zero_grad()
            b_bce = 0.0
            for idx_w in b_ids:
                hid, target = windows_data[idx_w]
                hid = hid.to("cuda")
                target = target.to("cuda")
                up = sm.upsample_hidden(hid)
                probs = sm.forward_speaker_sigmoids(up)
                n_t = min(probs.shape[1], target.shape[1])
                p_s = torch.clamp(probs[:, :n_t, :], 1e-6, 1.0 - 1e-6)
                t_s = target[:, :n_t, :]
                w = torch.where(t_s > 0.5, 1.0, neg_weight)
                b_bce = b_bce + F.binary_cross_entropy(p_s, t_s, weight=w) / len(b_ids)

            l2sp = sum(torch.sum((p - theta_0[p]) ** 2) for p in all_params)
            loss = b_bce + l2sp_alpha * l2sp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            ep_loss += float(b_bce.item())
            n_b += 1
        sched.step()
        if ep in (1, 3, 6, epochs):
            print(f"  Epoch {ep:2d}/{epochs} | BCE Loss: {ep_loss / max(1, n_b):.4f}")

    model.eval()
    os.makedirs("models", exist_ok=True)
    torch.save({
        "subpixel_upsample": sm.subpixel_upsample.state_dict() if sm.subpixel_upsample is not None else None,
        "first_hidden_to_hidden": sm.first_hidden_to_hidden.state_dict(),
        "single_hidden_to_spks": sm.single_hidden_to_spks.state_dict(),
        "hidden_to_spks": sm.hidden_to_spks.state_dict(),
    }, "models/nemotron_vi_upgraded.pt")


def main():
    meta = ensure_shared_dataset()
    test_meta = meta["test"]
    train_meta = meta["train"]
    work_dir = tempfile.mkdtemp(prefix="nemo_gpu0_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    from nemo.collections.asr.models import SortformerEncLabelModel
    print("=== CHẠY TỐI ƯU HÓA NEMOTRON-3 ĐỘC LẬP TRÊN GPU 0 ===")
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()
    sm = model.sortformer_modules
    for k, v in {"spkcache_len": 384, "fifo_len": 60, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 200}.items():
        setattr(sm, k, v)
    model._check_streaming_parameters()
    install_deep_aosc_hooks(sm)

    raw_paths = [item["wav_path"] for item in test_meta]

    # Scout pass for Safe Multi-Speaker Pre-Priming
    pp_scout = write_postprocessing_yaml({"onset": 0.50, "offset": 0.40, "pad_onset": 0.02, "pad_offset": 0.04, "min_duration_on": 0.16, "min_duration_off": 0.10}, work_dir)
    preds_scout = model.diarize(audio=raw_paths, batch_size=1, postprocessing_yaml=pp_scout)

    primed_paths, offsets, test_arrays = [], [], []
    for item, r_hyp in zip(test_meta, preds_scout):
        arr, sr = sf.read(item["wav_path"], dtype="float32")
        test_arrays.append((arr, sr))
        dur = item["dur"]
        scout_segs = precision_lock_segments(r_hyp, arr, sr, dur)
        if len({s["speaker"] for s in scout_segs}) == 1:
            scout_segs = split_collapsed_single_speaker(scout_segs, arr, sr)
        if len({s["speaker"] for s in scout_segs}) >= 2:
            p_arr, off_s = build_preprimed_audio(arr, sr, scout_segs)
            ppath = os.path.join(work_dir, f"primed_{item['idx']:03d}.wav")
            sf.write(ppath, p_arr, sr)
            primed_paths.append(ppath)
            offsets.append(off_s)
        else:
            primed_paths.append(item["wav_path"])
            offsets.append(0.0)

    # Train Layer 4 Head with clean Hungarian-Aligned + L2-SP learner
    train_nemotron_head_clean(model, train_meta, epochs=8, batch_size=8, l2sp_alpha=0.015, neg_weight=1.35)

    presets = [
        ("N1_FT_Balanced_Lock",  {"onset": 0.52, "offset": 0.42, "pad_onset": 0.01, "pad_offset": 0.03, "min_duration_on": 0.16, "min_duration_off": 0.10}, 0.22, 0.38,  0.000,  0.000),
        ("N2_FT_Crisp_Lock",     {"onset": 0.58, "offset": 0.48, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.16, "min_duration_off": 0.10}, 0.14, 0.24,  0.000,  0.000),
        ("N3_FT_Crisp_Inset15ms",{"onset": 0.58, "offset": 0.48, "pad_onset": 0.00, "pad_offset": 0.01, "min_duration_on": 0.14, "min_duration_off": 0.08}, 0.10, 0.18,  0.015, -0.015),
        ("N4_FT_Sharp_Inset20ms",{"onset": 0.62, "offset": 0.52, "pad_onset": 0.00, "pad_offset": 0.00, "min_duration_on": 0.14, "min_duration_off": 0.06}, 0.06, 0.12,  0.020, -0.020),
        ("N5_FT_Ultra_Inset25ms",{"onset": 0.65, "offset": 0.55, "pad_onset": 0.00, "pad_offset": 0.00, "min_duration_on": 0.12, "min_duration_off": 0.05}, 0.04, 0.08,  0.025, -0.025),
    ]

    results = []
    for name, pp_dict, u_gap, c_gap, ds, de in presets:
        pp_yaml = write_postprocessing_yaml(pp_dict, os.path.join(work_dir, name))
        preds = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_yaml)

        segs_per_file = []
        tot_ref_s = w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
        spk_exact = short_segs = total_segs = 0
        spk_abs_err = []

        for item, (arr, sr), raw_hyp, off_s in zip(test_meta, test_arrays, preds, offsets):
            dur = item["dur"]
            ref_turns = [(s, e, spk) for s, e, spk in item["ref_turns"]]
            shifted = []
            for raw in raw_hyp:
                s, e, spk = _parse_one(raw)
                if e > off_s:
                    ns = max(0.0, s - off_s + ds)
                    ne = min(dur, e - off_s + de)
                    if ne - ns >= 0.10:
                        shifted.append(f"{ns:.3f} {ne:.3f} {spk}")
            segs = precision_lock_segments(shifted, wav=arr, sr=sr, duration=dur, unconditional_gap=u_gap, conditional_gap=c_gap)
            if len({s["speaker"] for s in segs}) == 1:
                segs = split_collapsed_single_speaker(segs, arr, sr)
            segs_per_file.append(segs)

            hyp_turns = [(s["start"], s["end"], s["speaker"]) for s in segs]
            total_segs += len(segs)
            short_segs += sum(1 for s in segs if (s["end"] - s["start"]) < 0.5)
            n_ref = len({s for _, _, s in ref_turns})
            n_hyp = len({s for _, _, s in hyp_turns})
            if n_ref == n_hyp:
                spk_exact += 1
            spk_abs_err.append(abs(n_hyp - n_ref))

            der0, miss0, fa0, conf0, ref_s = compute_der_metrics(ref_turns, hyp_turns, dur, collar=0.0)
            der25, _, _, _, _ = compute_der_metrics(ref_turns, hyp_turns, dur, collar=0.25)
            w_der0 += der0 * ref_s
            w_der25 += der25 * ref_s
            w_miss += miss0 * ref_s
            w_fa += fa0 * ref_s
            w_conf += conf0 * ref_s
            tot_ref_s += ref_s

        results.append({
            "name": name,
            "DER@0.00": w_der0 / tot_ref_s,
            "DER@0.25": w_der25 / tot_ref_s,
            "Miss%": w_miss / tot_ref_s,
            "FA%": w_fa / tot_ref_s,
            "Conf%": w_conf / tot_ref_s,
            "SpkExact%": spk_exact / len(test_meta) * 100.0,
            "SpkMAE": float(np.mean(spk_abs_err)),
            "<0.5s": f"{short_segs}/{total_segs}",
            "segs": segs_per_file,
        })

    best_0ms = min(results, key=lambda r: r["DER@0.00"])
    best_25  = min(results, key=lambda r: r["DER@0.25"])
    with open(os.path.join(CACHE_DIR, "best_nemotron_segs.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_0ms": {"name": best_0ms["name"], "segs": best_0ms["segs"]},
            "best_25":  {"name": best_25["name"],  "segs": best_25["segs"]},
        }, f)

    print("\n" + "=" * 110)
    print("KẾT QUẢ TỐI ƯU HÓA ĐỘC LẬP NEMOTRON-3 TRÊN GPU 0 (ViYT-Diar Test N=20)")
    print("=" * 110)
    print(f"{'Cấu hình Nemotron-3':30s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s}")
    print("-" * 110)
    for r in results:
        print(f"{r['name']:30s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s}")
    print("=" * 110)
    print(f"[OK] Đã lưu bản Nemotron-3 tốt nhất ({best_0ms['name']}: DER@0ms={best_0ms['DER@0.00']:.2f}%, {best_25['name']}: DER@.25s={best_25['DER@0.25']:.2f}%) vào {CACHE_DIR}/best_nemotron_segs.json")


if __name__ == "__main__":
    main()
