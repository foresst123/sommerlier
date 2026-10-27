#!/usr/bin/env python3
"""
Final Nemotron-3 Diarization Model for Vietnamese Podcasts/Movies.
Trains STRICTLY on Real ViYT-Diar data (80 files Train / 20 files Test)
using Dense Sliding Windows (3s hop) to maximize the 1.45 hours of data.
"""
import copy
import io
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

from tools.bench_nemotron_viyt import compute_der_metrics, install_deep_aosc_hooks, build_preprimed_audio
from tools.exp2_nemotron_viyt import build_frame_targets
from tools.exp7_precision_lock import precision_lock_segments
from tools.exp8_target_breaker import split_collapsed_single_speaker
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


def main():
    work_dir = tempfile.mkdtemp(prefix="nemo_real_only_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    from datasets import load_dataset, Audio
    print("=== 1. Tải bộ chuẩn ViYT-Diar (Real Data Only) ===")
    ds = load_dataset("tuanduy1612/ViYT-Diar", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))

    test_items = []
    train_items = []
    for idx, row in enumerate(ds):
        aud = row["audio"]
        arr, sr = sf.read(io.BytesIO(aud["bytes"]) if aud.get("bytes") else aud["path"], dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        dur = len(arr) / float(sr)
        ref_turns = [
            (float(s), float(e), str(spk))
            for s, e, spk in zip(row["timestamps_start"], row["timestamps_end"], row["speaker"])
            if float(e) > float(s)
        ]
        item = (idx, arr, sr, ref_turns, dur, row["audio_id"])
        if idx < 20:
            test_items.append(item)
        else:
            train_items.append(item)

    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()
    sm = model.sortformer_modules
    for k, v in {"spkcache_len": 384, "fifo_len": 60, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 200}.items():
        setattr(sm, k, v)
    model._check_streaming_parameters()
    install_deep_aosc_hooks(sm)

    # Trích xuất Dense Windows (trượt mỗi 3 giây) để nhân bản 1.45h lên 1.200+ mẫu!
    win_s = 24.0
    win_samples = int(win_s * 16000)
    train_windows = []
    
    captured = {}
    orig_upsample = sm.upsample_hidden
    def wrapped_upsample(hidden_states):
        captured["hidden"] = hidden_states.detach()
        return orig_upsample(hidden_states)
    sm.upsample_hidden = wrapped_upsample

    print("=== 2. Trích xuất Dense Sliding Windows trên 80 file Train ===")
    with torch.no_grad():
        for idx, arr, sr, ref_turns, dur, _ in train_items:
            # Trượt cửa sổ mỗi 3 giây để có số lượng mẫu huấn luyện lớn
            starts = np.arange(0.0, max(0.1, dur - win_s + 0.1), 3.0)
            for st_s in starts:
                i0 = int(st_s * sr)
                chunk = arr[i0 : i0 + win_samples]
                if len(chunk) < int(4.0 * sr):
                    continue
                if len(chunk) < win_samples:
                    chunk = np.pad(chunk, (0, win_samples - len(chunk)))
                sig = torch.from_numpy(chunk).unsqueeze(0).to("cuda")
                sig_len = torch.tensor([len(chunk)], dtype=torch.long, device="cuda")
                captured.clear()
                _ = model.forward(audio_signal=sig, audio_signal_length=sig_len)
                if "hidden" not in captured:
                    continue
                hid = captured["hidden"].cpu()
                init_p = sm.forward_speaker_sigmoids(orig_upsample(hid.to("cuda"))).squeeze(0).cpu().numpy()
                n_10ms = init_p.shape[0]
                shifted = [(s - st_s, e - st_s, spk) for s, e, spk in ref_turns if e > st_s and s < st_s + win_s]
                raw_t = build_frame_targets(shifted, n_10ms, step=0.01).numpy()
                aligned_t = align_target_hungarian(init_p, raw_t)
                train_windows.append((hid.squeeze(0), aligned_t))

    sm.upsample_hidden = orig_upsample
    print(f"[OK] Đã trích xuất {len(train_windows)} windows từ 80 file Real.")

    # Mở khóa Tầng 4
    for p in model.parameters():
        p.requires_grad_(False)
    param_groups = [
        {"params": [p for p in sm.subpixel_upsample.parameters()], "lr": 1.2e-3, "init_lr": 1.2e-3},
        {"params": [p for p in sm.first_hidden_to_hidden.parameters()], "lr": 6e-4, "init_lr": 6e-4},
        {"params": [p for p in sm.single_hidden_to_spks.parameters()], "lr": 4e-4, "init_lr": 4e-4},
        {"params": [p for p in sm.hidden_to_spks.parameters()], "lr": 4e-4, "init_lr": 4e-4},
    ]
    all_params = []
    theta_0 = {}
    for g in param_groups:
        for p in g["params"]:
            p.requires_grad_(True)
            all_params.append(p)
            theta_0[p] = p.detach().clone()

    batch_size = 16
    epochs = 6
    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    rng = random.Random(42)

    print(f"=== 3. Huấn luyện Tầng 4 (Dense Real Only, L2-SP) trên GPU 0 ===")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        order = list(range(len(train_windows)))
        rng.shuffle(order)
        ep_loss = 0.0
        n_b = 0
        for b0 in range(0, len(order), batch_size):
            b_ids = order[b0 : b0 + batch_size]
            opt.zero_grad()
            b_bce = 0.0
            for idx_w in b_ids:
                hid, target = train_windows[idx_w]
                hid = hid.unsqueeze(0).to("cuda")
                target = target.unsqueeze(0).to("cuda")
                up = sm.upsample_hidden(hid)
                probs = sm.forward_speaker_sigmoids(up)
                n_t = min(probs.shape[1], target.shape[1])
                p_s = torch.clamp(probs[:, :n_t, :], 1e-6, 1.0 - 1e-6)
                t_s = target[:, :n_t, :]
                w = torch.where(t_s > 0.5, 1.0, 1.30)  # Phạt nhẹ FA
                bce = F.binary_cross_entropy(p_s, t_s, weight=w)
                b_bce = b_bce + bce / len(b_ids)

            l2sp = sum(torch.sum((p - theta_0[p]) ** 2) for p in all_params)
            loss = b_bce + 0.02 * l2sp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            ep_loss += float(b_bce.item())
            n_b += 1
        sched.step()
        print(f"  Epoch {ep}/{epochs} - BCE Loss: {ep_loss / max(1, n_b):.4f}")

    print(f"[OK] Huấn luyện xong trong {time.time()-t0:.1f}s.")
    model.eval()
    os.makedirs("models", exist_ok=True)
    save_path = "models/nemotron_vi_real_master.pt"
    torch.save({
        "subpixel_upsample": sm.subpixel_upsample.state_dict(),
        "first_hidden_to_hidden": sm.first_hidden_to_hidden.state_dict(),
        "single_hidden_to_spks": sm.single_hidden_to_spks.state_dict(),
        "hidden_to_spks": sm.hidden_to_spks.state_dict(),
    }, save_path)
    print(f"[OK] Đã lưu trọng số tại: {save_path}")

    # Đánh giá
    raw_paths = []
    for idx, arr, sr, _, dur, _ in test_items:
        p = os.path.join(work_dir, f"test_{idx:03d}.wav")
        sf.write(p, arr, sr)
        raw_paths.append(p)

    pp_scout = write_postprocessing_yaml({"onset": 0.50, "offset": 0.40, "pad_onset": 0.02, "pad_offset": 0.04, "min_duration_on": 0.16, "min_duration_off": 0.10}, work_dir)
    preds_scout = model.diarize(audio=raw_paths, batch_size=1, postprocessing_yaml=pp_scout)

    primed_paths, offsets, test_arrays = [], [], []
    for item, r_hyp in zip(test_items, preds_scout):
        arr, sr, dur = item[1], item[2], item[4]
        test_arrays.append((arr, sr))
        scout_segs = precision_lock_segments(r_hyp, arr, sr, dur)
        if len({s["speaker"] for s in scout_segs}) == 1:
            scout_segs = split_collapsed_single_speaker(scout_segs, arr, sr)
        if len({s["speaker"] for s in scout_segs}) >= 2:
            p_arr, off_s = build_preprimed_audio(arr, sr, scout_segs)
            ppath = os.path.join(work_dir, f"primed_{item[0]:03d}.wav")
            sf.write(ppath, p_arr, sr)
            primed_paths.append(ppath)
            offsets.append(off_s)
        else:
            primed_paths.append(raw_paths[item[0]])
            offsets.append(0.0)

    pp_bal = write_postprocessing_yaml({"onset": 0.52, "offset": 0.42, "pad_onset": 0.01, "pad_offset": 0.03, "min_duration_on": 0.16, "min_duration_off": 0.10}, work_dir)
    pp_crisp = write_postprocessing_yaml({"onset": 0.58, "offset": 0.48, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.14, "min_duration_off": 0.08}, work_dir)

    preds_bal = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_bal)
    preds_crisp = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_crisp)

    def eval_cfg(preds_list, label, u_gap, c_gap):
        tot_ref = w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
        spk_exact = short_segs = total_segs = 0
        spk_abs_err = []
        for i, (idx, arr, sr, ref_turns, dur, _) in enumerate(test_items):
            off_s = offsets[i]
            shifted = []
            for raw in preds_list[i]:
                s, e, spk = _parse_one(raw)
                if e > off_s:
                    shifted.append(f"{max(0.0, s - off_s):.3f} {e - off_s:.3f} {spk}")
            segs = precision_lock_segments(shifted, wav=arr, sr=sr, duration=dur, unconditional_gap=u_gap, conditional_gap=c_gap)
            if len({s["speaker"] for s in segs}) == 1:
                segs = split_collapsed_single_speaker(segs, arr, sr)
            
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
            tot_ref += ref_s

        return {
            "name": label,
            "DER@0.00": w_der0 / tot_ref,
            "DER@0.25": w_der25 / tot_ref,
            "Miss%": w_miss / tot_ref,
            "FA%": w_fa / tot_ref,
            "Conf%": w_conf / tot_ref,
            "SpkExact%": spk_exact / len(test_items) * 100.0,
            "SpkMAE": float(np.mean(spk_abs_err)),
            "<0.5s": f"{short_segs}/{total_segs}",
        }

    results = [
        eval_cfg(preds_bal,   "1_Real_Balanced_Lock", 0.22, 0.38),
        eval_cfg(preds_crisp, "2_Real_Crisp_Lock",    0.14, 0.24),
    ]

    print("\n" + "=" * 108)
    print("KẾT QUẢ ĐÁNH GIÁ CHUẨN XÁC: NEMOTRON-3 (HỌC 100% REAL DATA DENSE WINDOWS) TRÊN 20 BÀI TEST!")
    print("=" * 108)
    print(f"{'Cấu hình Nemotron-3 Real':28s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s}")
    print("-" * 108)
    for r in results:
        print(f"{r['name']:28s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s}")
    print("=" * 108)


if __name__ == "__main__":
    main()
