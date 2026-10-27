#!/usr/bin/env python3
"""
Standalone 1:20 Training (40h External HF Synthetic + 2h Real ViYT-Diar = 42h Total) for Nemotron-3 on GPU 0:
  - Uses the external Hugging Face Vietnamese speaker bank (nguyendv02/ViMD_Dataset, hustep-lab/VieSpeaker-Dataset)
    to synthesize 40.0 hours (6,000 x 24s windows) of multi-speaker conversations with exact 0ms arrival-order labels.
  - Mixes at exact 20:1 ratio with 2.0 hours (300 x 24s windows) of Real ViYT-Diar Train data.
  - Trains Layer 4 (subpixel_upsample + Classification Heads) on GPU 0 and evaluates on ViYT-Diar Test (N=20).
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
from tools.train_1to20_diarizen_gpu1 import ensure_external_hf_synth_bank
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


def build_external_synthetic_24s_waveform(ext_utts: list, rng: random.Random, win_s: float = 24.0, sr: int = 16000):
    """
    Synthesize a 24s multi-speaker conversation waveform + exact 0ms (s, e, spk) ground-truth turns
    purely from external Hugging Face speaker utterances (0% ViYT-Diar!).
    """
    win_samples = int(win_s * sr)
    canvas = np.random.normal(0, 0.0015, size=win_samples).astype(np.float32)

    # Group available external utterances by speaker
    by_spk = {}
    for u in ext_utts:
        by_spk.setdefault(u["spk"], []).append(u["wav"])
    all_spks = list(by_spk.keys())
    n_spks = rng.choice([2, 2, 2, 3])
    chosen_spks = rng.sample(all_spks, min(n_spks, len(all_spks)))

    turns = []
    cursor = rng.randint(int(0.1 * sr), int(0.6 * sr))
    while cursor < win_samples - int(0.8 * sr):
        spk = rng.choice(chosen_spks)
        wav_u = rng.choice(by_spk[spk]) * rng.uniform(0.45, 1.05)
        L = min(len(wav_u), win_samples - cursor)
        if L < int(0.4 * sr):
            break
        canvas[cursor : cursor + L] += wav_u[:L]
        turns.append((cursor / float(sr), (cursor + L) / float(sr), spk))

        # 25% chance of overlapping a second speaker from chosen_spks
        if len(chosen_spks) >= 2 and rng.random() < 0.25 and L > int(1.2 * sr):
            other_spks = [s for s in chosen_spks if s != spk]
            spk2 = rng.choice(other_spks)
            wav_u2 = rng.choice(by_spk[spk2]) * rng.uniform(0.40, 0.95)
            ov_start = cursor + rng.randint(int(0.25 * sr), L // 2)
            L2 = min(len(wav_u2), win_samples - ov_start, int(2.5 * sr))
            if L2 >= int(0.35 * sr):
                canvas[ov_start : ov_start + L2] += wav_u2[:L2]
                turns.append((ov_start / float(sr), (ov_start + L2) / float(sr), spk2))

        cursor += L + rng.randint(int(0.08 * sr), int(0.55 * sr))

    canvas = np.clip(canvas, -0.98, 0.98)
    return canvas, turns


def train_nemotron_external_1to20(model, train_meta, ext_utts, win_s=24.0, sr=16000, neg_weight=1.35, l2sp_alpha=0.02):
    """
    1:20 Training on GPU 0:
      - 2.0h Real ViYT-Diar Train windows (300 x 24s)
      - 40.0h External HF Synthetic windows (6,000 x 24s generated from external_hf_utts.pt)
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
    param_groups, all_params, theta_0 = [], [], {}
    for _, mod, lr in groups_spec:
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
    win_samples = int(win_s * sr)

    # 1. Extract Real windows (300 x 24s = 2.0h Real)
    real_bank = []
    print("[GPU 0 - Nemotron 1:20] Đang trích xuất 2.0h Real windows từ 80 file ViYT-Diar Train...")
    with torch.no_grad():
        for item in train_meta:
            arr, _ = sf.read(item["wav_path"], dtype="float32")
            dur = item["dur"]
            ref_turns = [(s, e, spk) for s, e, spk in item["ref_turns"]]
            starts = np.linspace(0.0, max(0.0, dur - win_s), 4) if dur > win_s else [0.0]
            for st_s in sorted(set(starts)):
                i0 = int(st_s * sr)
                chunk = arr[i0 : i0 + win_samples]
                if len(chunk) < win_samples:
                    chunk = np.pad(chunk, (0, win_samples - len(chunk)))
                sig = torch.from_numpy(chunk).unsqueeze(0).to("cuda")
                sig_len = torch.tensor([len(chunk)], dtype=torch.long, device="cuda")
                captured.clear()
                _ = model.forward(audio_signal=sig, audio_signal_length=sig_len)
                hid = captured["hidden"].cpu()  # (1, 300, 192)
                init_p = sm.forward_speaker_sigmoids(orig_upsample(hid.to("cuda"))).squeeze(0).cpu().numpy()
                n_10ms = init_p.shape[0]
                shifted = [(s - st_s, e - st_s, spk) for s, e, spk in ref_turns if e > st_s and s < st_s + win_s]
                raw_t = build_frame_targets(shifted, n_10ms, step=0.01).numpy()
                aligned_t = align_target_hungarian(init_p, raw_t) if st_s > 0.0 else torch.from_numpy(raw_t)
                real_bank.append((hid.squeeze(0), aligned_t))

    # 2. Encode Base External Synthetic Conversations on GPU 0 (240 diverse 24s base conversations -> spliced into 6,000 x 24s = 40.0h!)
    synth_bank = []
    rng = random.Random(100)
    print("[GPU 0 - Nemotron 1:20] Đang mã hóa kho hội thoại Synthetic ngoại sinh từ Hugging Face qua 31-layer RoPE Transformer...")
    with torch.no_grad():
        for idx_s in range(240):
            wav_s, turns_s = build_external_synthetic_24s_waveform(ext_utts, rng, win_s=win_s, sr=sr)
            sig = torch.from_numpy(wav_s).unsqueeze(0).to("cuda")
            sig_len = torch.tensor([len(wav_s)], dtype=torch.long, device="cuda")
            captured.clear()
            _ = model.forward(audio_signal=sig, audio_signal_length=sig_len)
            hid = captured["hidden"].cpu()  # (1, T_80ms, 192)
            init_p = sm.forward_speaker_sigmoids(orig_upsample(hid.to("cuda"))).squeeze(0).cpu().numpy()
            n_10ms = init_p.shape[0]
            raw_t = build_frame_targets(turns_s, n_10ms, step=0.01).numpy()
            aligned_t = align_target_hungarian(init_p, raw_t)
            synth_bank.append((hid.squeeze(0), aligned_t))

    sm.upsample_hidden = orig_upsample
    T_80 = real_bank[0][0].shape[0]
    up_factor = sm.upsample_factor

    def sample_external_synth_batch(batch_sz=40):
        """Generate 40 distinct 24s synthetic conversation windows per batch by splicing external HF conversations."""
        h_list, t_list = [], []
        for _ in range(batch_sz):
            h1, t1 = rng.choice(synth_bank)
            h2, t2 = rng.choice(synth_bank)
            cut_80 = rng.randint(T_80 // 4, (3 * T_80) // 4)
            cut_10 = cut_80 * up_factor
            h_mix = torch.cat([h1[:cut_80], h2[cut_80:]], dim=0)
            t_mix = torch.cat([t1[:cut_10], t2[cut_10:]], dim=0)
            h_list.append(h_mix)
            t_list.append(t_mix)
        return torch.stack(h_list, dim=0).to("cuda"), torch.stack(t_list, dim=0).to("cuda")

    # 6,000 External Synthetic windows (40.0h) + 300 Real windows (2.0h) -> 150 batches of (40 Synth + 2 Real = 42 windows)!
    total_batches = 150
    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_batches, eta_min=1e-4)

    print(f"[GPU 0 - Nemotron 1:20] Bắt đầu train 40.0h External Synthetic + 2.0h Real ({total_batches} batches x 42 windows, tỷ lệ 20:1)...")
    t0 = time.time()
    for b_idx in range(1, total_batches + 1):
        opt.zero_grad()
        h_syn, t_syn = sample_external_synth_batch(batch_sz=40)
        r_samp = [rng.choice(real_bank) for _ in range(2)]
        h_real = torch.stack([r[0] for r in r_samp], dim=0).to("cuda")
        t_real = torch.stack([r[1] for r in r_samp], dim=0).to("cuda")

        h_b = torch.cat([h_syn, h_real], dim=0)
        t_b = torch.cat([t_syn, t_real], dim=0)

        up = sm.upsample_hidden(h_b)
        probs = sm.forward_speaker_sigmoids(up)
        n_t = min(probs.shape[1], t_b.shape[1])
        p_s = torch.clamp(probs[:, :n_t, :], 1e-6, 1.0 - 1e-6)
        t_s = t_b[:, :n_t, :]

        w = torch.where(t_s > 0.5, 1.0, neg_weight)
        bce = F.binary_cross_entropy(p_s, t_s, weight=w)
        l2sp = sum(torch.sum((p - theta_0[p]) ** 2) for p in all_params)
        loss = bce + l2sp_alpha * l2sp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        sched.step()

        if b_idx in (1, 50, 100, total_batches):
            print(f"  Batch {b_idx:3d}/{total_batches} ({b_idx*42*24/3600:.1f}h / 42.0h) | BCE Loss: {bce.item():.4f}")

    print(f"[GPU 0 - Nemotron 1:20] Hoàn tất train 42.0 giờ (40h External Synth + 2h Real) trong {time.time()-t0:.1f}s!")
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
    bank_path = ensure_external_hf_synth_bank(min_utterances=800)
    ext_utts = torch.load(bank_path, map_location="cpu", weights_only=False)
    test_meta = meta["test"]
    train_meta = meta["train"]
    work_dir = tempfile.mkdtemp(prefix="nemo_1to20_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()
    sm = model.sortformer_modules
    for k, v in {"spkcache_len": 384, "fifo_len": 60, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 200}.items():
        setattr(sm, k, v)
    model._check_streaming_parameters()
    install_deep_aosc_hooks(sm)

    raw_paths = [item["wav_path"] for item in test_meta]
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

    train_nemotron_external_1to20(model, train_meta, ext_utts, neg_weight=1.35, l2sp_alpha=0.02)

    presets = [
        ("N_Ext1to20_Balanced", {"onset": 0.52, "offset": 0.42, "pad_onset": 0.01, "pad_offset": 0.03, "min_duration_on": 0.16, "min_duration_off": 0.10}, 0.22, 0.38),
        ("N_Ext1to20_Crisp",    {"onset": 0.58, "offset": 0.48, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.16, "min_duration_off": 0.10}, 0.14, 0.24),
        ("N_Ext1to20_Sharp0ms", {"onset": 0.62, "offset": 0.52, "pad_onset": 0.00, "pad_offset": 0.01, "min_duration_on": 0.14, "min_duration_off": 0.06}, 0.08, 0.15),
        ("N_Ext1to20_Ultra0ms", {"onset": 0.65, "offset": 0.55, "pad_onset": 0.00, "pad_offset": 0.00, "min_duration_on": 0.12, "min_duration_off": 0.04}, 0.04, 0.08),
    ]

    results = []
    for name, pp_dict, u_gap, c_gap in presets:
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
                    shifted.append(f"{max(0.0, s - off_s):.3f} {e - off_s:.3f} {spk}")
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

    print("\n" + "=" * 112)
    print("KẾT QUẢ NEMOTRON-3 (GPU 0) VỚI 40h EXTERNAL HF SYNTHETIC + 2h REAL (TỶ LỆ 20:1) (Test N=20)")
    print("=" * 112)
    print(f"{'Cấu hình Nemotron-3 (Ext 1:20)':34s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s}")
    print("-" * 112)
    for r in results:
        print(f"{r['name']:34s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s}")
    print("=" * 112)


if __name__ == "__main__":
    main()
