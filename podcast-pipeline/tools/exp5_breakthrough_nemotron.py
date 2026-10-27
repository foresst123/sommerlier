#!/usr/bin/env python3
"""
Experiment 5: Breaking the 80ms Bottleneck via 10ms Acoustic-Skip Neural Refiner + Deep AOSC Quorum
Target on ViYT-Diar (Test N=20, trained strictly on disjoint Train N=80):
  - DER@0ms < 6.0%
  - Miss%   < 1.0%
  - FA%     < 2.0%
  - SpkExact >= 95.0%
"""
import copy
import inspect
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

from tools.bench_nemotron_viyt import (
    compute_der_metrics,
    install_deep_aosc_hooks,
)
from tools.exp2_nemotron_viyt import load_viyt_split, build_frame_targets
from utils.nemotron_segments import POSTPROCESSING_BASE, to_pipeline_segments, write_postprocessing_yaml

warnings.filterwarnings("ignore")


def extract_10ms_acoustic_features(wav: np.ndarray, sr: int = 16000, n_frames_target: int = None) -> torch.Tensor:
    """
    Extract 10ms-resolution acoustic features (T, 40) directly from 16kHz audio:
      - 32 log-frequency spectral bands (80 Hz to 7600 Hz) with utterance-level noise-floor normalization
      - 4 band energy ratios (fundamental F0 80-400Hz, vowel formants 400-2500Hz, sibilants 2500-7500Hz, total)
      - 4 temporal derivatives (first-order delta of energy & formants for sharp 10ms onset/offset detection)
    """
    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    if torch.cuda.is_available():
        x = x.cuda()
    win = int(0.025 * sr)  # 400 samples = 25ms
    hop = int(0.010 * sr)  # 160 samples = 10ms
    window = torch.hann_window(win, device=x.device)
    stft = torch.stft(x, n_fft=512, hop_length=hop, win_length=win, window=window, center=True, return_complex=True)
    power = (stft.abs() ** 2).T  # (T, 257) -> each bin is 16000/512 = 31.25 Hz

    if n_frames_target is not None:
        if power.shape[0] > n_frames_target:
            power = power[:n_frames_target]
        elif power.shape[0] < n_frames_target:
            pad = power[-1:].expand(n_frames_target - power.shape[0], -1)
            power = torch.cat([power, pad], dim=0)

    # Group 256 bins into 32 perceptual sub-bands (focused on 60Hz - 7500Hz)
    bins_32 = F.adaptive_avg_pool1d(power[:, 2:242].unsqueeze(0), 32).squeeze(0)  # (T, 32)
    log_spec = torch.log10(torch.clamp(bins_32, min=1e-10))

    # Utterance-level noise floor (15th percentile) & dynamic range normalization
    floor = torch.quantile(log_spec, 0.15, dim=0, keepdim=True)
    ceil = torch.quantile(log_spec, 0.95, dim=0, keepdim=True)
    norm_spec = (log_spec - floor) / torch.clamp(ceil - floor, min=0.5)  # (T, 32)

    # Explicit voice/silence cues at 10ms
    e_f0 = norm_spec[:, 0:4].mean(dim=-1, keepdim=True)        # 60 - 500 Hz (pitch / voicing)
    e_formant = norm_spec[:, 4:16].mean(dim=-1, keepdim=True)  # 500 - 3000 Hz (vowels)
    e_high = norm_spec[:, 16:32].mean(dim=-1, keepdim=True)    # 3000 - 7500 Hz (consonants)
    e_tot = norm_spec.mean(dim=-1, keepdim=True)

    cues = torch.cat([e_f0, e_formant, e_high, e_tot], dim=-1)  # (T, 4)
    delta = torch.zeros_like(cues)
    if cues.shape[0] > 2:
        delta[1:-1] = 0.5 * (cues[2:] - cues[:-2])

    feats = torch.cat([norm_spec, cues, delta], dim=-1)  # (T, 40)
    return feats.float().cpu()


def get_streaming_10ms_probs(model, audio_paths: list) -> list:
    """Run Nemotron-3 streaming diarization and return exact 10ms (T, 8) probability tensors."""
    sig = inspect.signature(model.diarize)
    if "include_tensor_outputs" in sig.parameters:
        out = model.diarize(audio=audio_paths, batch_size=1, include_tensor_outputs=True)
        if isinstance(out, tuple) and len(out) == 2:
            _, probs_list = out
            return [p.squeeze(0).detach().float().cpu() if p.ndim == 3 else p.detach().float().cpu() for p in probs_list]

    # Fallback hook if include_tensor_outputs is unavailable
    sm = model.sortformer_modules
    captured_chunks = []
    orig_sig = sm.forward_speaker_sigmoids

    def hooked_sig(hidden_out):
        p = orig_sig(hidden_out)
        captured_chunks.append(p.detach().float().cpu())
        return p

    sm.forward_speaker_sigmoids = hooked_sig
    probs_out = []
    try:
        for p in audio_paths:
            captured_chunks.clear()
            _ = model.diarize(audio=[p], batch_size=1)
            if captured_chunks:
                probs_out.append(torch.cat([c.squeeze(0) for c in captured_chunks], dim=0))
    finally:
        sm.forward_speaker_sigmoids = orig_sig
    return probs_out


class AcousticSkipDiarRefiner(nn.Module):
    """
    U-Net style 10ms Acoustic-Skip Neural Refiner:
      Fuses 10ms raw acoustic Mel/voicing features (T, 40) with Nemotron-3 10ms speaker probabilities (T, 8).
      1. AcousticVADNet (Dilated ResConv1D + BiGRU): predicts ultra-sharp 10ms speech activity & boundary embeddings.
      2. Equivariant Speaker-Slot Refiner: refines each speaker channel k using 10ms acoustic embeddings + cross-speaker competition.
    """
    def __init__(self, acoustic_dim=40, hidden_dim=96):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(acoustic_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Multi-scale dilated Conv1D stack (receptive field ~ +/- 350ms at 10ms frame rate)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, dilation=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=8, dilation=4)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.vad_head = nn.Linear(hidden_dim, 1)  # Sharp 10ms speech vs silence logit

        # Per-speaker equivariant refiner (shared across all 8 speaker slots)
        # Slot input: [p_k, logit_k, rel_dominance_k, max_other_k, spk_global_ratio, vad_prob, vad_logit] + acoustic_ctx (32)
        self.ctx_proj = nn.Linear(hidden_dim, 32)
        self.slot_gru = nn.GRU(32 + 7, 32, num_layers=1, batch_first=True, bidirectional=True)
        self.slot_out = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # Initialize residual delta near zero so initial behavior starts from Nemotron-3
        nn.init.zeros_(self.slot_out[-1].weight)
        nn.init.zeros_(self.slot_out[-1].bias)

    def forward(self, acoustic: torch.Tensor, probs: torch.Tensor):
        """
        acoustic: (B, T, 40)
        probs:    (B, T, 8) in [0, 1]
        Returns:  refined_probs (B, T, 8), vad_probs (B, T, 1)
        """
        B, T, K = probs.shape
        p_clamped = torch.clamp(probs, 1e-5, 1.0 - 1e-5)
        logits = torch.logit(p_clamped)

        p_max, _ = probs.max(dim=-1, keepdim=True)
        p_sum = probs.sum(dim=-1, keepdim=True)
        p_2nd = torch.topk(probs, k=2, dim=-1).values[:, :, 1:2]

        # 1. Shared 10ms Acoustic + Global Activity Branch
        x_in = torch.cat([acoustic, p_max, p_sum, p_2nd], dim=-1)  # (B, T, 43)
        h = self.in_proj(x_in)  # (B, T, H)

        h_c = h.transpose(1, 2)
        h = self.norm1(h + F.gelu(self.conv1(h_c)).transpose(1, 2))
        h_c = h.transpose(1, 2)
        h = self.norm2(h + F.gelu(self.conv2(h_c)).transpose(1, 2))
        h_c = h.transpose(1, 2)
        h = self.norm3(h + F.gelu(self.conv3(h_c)).transpose(1, 2))

        h_gru, _ = self.gru(h)
        h = h + h_gru  # (B, T, H)

        vad_logit = self.vad_head(h)          # (B, T, 1)
        vad_prob = torch.sigmoid(vad_logit)   # (B, T, 1)
        a_ctx = F.gelu(self.ctx_proj(h))      # (B, T, 32)

        # 2. Equivariant Per-Slot Refinement
        # Global speaker activity share across the recording
        spk_share = (probs > 0.35).float().mean(dim=1, keepdim=True)  # (B, 1, 8)
        rel_dom = probs / (p_sum + 1e-5)                              # (B, T, 8)

        slot_inputs = []
        for k in range(K):
            pk = probs[:, :, k:k+1]
            lk = logits[:, :, k:k+1]
            rk = rel_dom[:, :, k:k+1]
            others = torch.cat([probs[:, :, :k], probs[:, :, k+1:]], dim=-1)
            max_other, _ = others.max(dim=-1, keepdim=True)
            sk = spk_share[:, :, k:k+1].expand(B, T, 1)
            s_feat = torch.cat([pk, lk, rk, max_other, sk, vad_prob, vad_logit, a_ctx], dim=-1)  # (B, T, 39)
            slot_inputs.append(s_feat)

        # Stack all B*K slots for parallel BiGRU execution
        slot_batch = torch.cat(slot_inputs, dim=0)  # (B*K, T, 39)
        slot_h, _ = self.slot_gru(slot_batch)       # (B*K, T, 64)
        delta_logits = self.slot_out(slot_h).view(K, B, T).permute(1, 2, 0)  # (B, T, K)

        # Combine updated slot logits with sharp 10ms acoustic VAD gate
        refined_logits = logits + delta_logits + 0.5 * (vad_logit - 0.5)
        refined_probs = torch.sigmoid(refined_logits) * torch.clamp(vad_prob * 1.15, 0.0, 1.0)
        return refined_probs, vad_prob


def align_columns_hungarian(pred_np: np.ndarray, target_np: np.ndarray) -> torch.Tensor:
    """Align ground-truth columns (T, 8) to predicted speaker slots (T, 8) via Hungarian algorithm."""
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


def probs_to_segments(
    probs_np: np.ndarray,
    onset: float = 0.52,
    offset: float = 0.42,
    min_on_s: float = 0.14,
    min_off_s: float = 0.06,
    ghost_min_exclusive_s: float = 1.2,
    ghost_min_share: float = 0.025,
    step: float = 0.01,
) -> list:
    """
    Convert refined 10ms (T, 8) probability matrix to diarization segments:
      1. Ghost-Speaker Purity Filter: if a minor channel k has < ghost_min_exclusive_s of exclusive speech
         (or < ghost_min_share of total speech), merge its probability mass into the closest active primary speaker.
      2. Per-channel 10ms Hysteresis Thresholding (onset / offset) + micro-gap closing (min_off_s).
    """
    P = np.array(probs_np, dtype=np.float32, copy=True)  # (T, 8)
    T, K = P.shape

    # 1. Ghost-Channel Reassignment on probability matrix
    active_binary = P > offset
    total_active_frames = max(1, int(active_binary.sum()))
    exclusive_counts = np.zeros(K, dtype=np.int32)
    total_counts = active_binary.sum(axis=0)

    for k in range(K):
        others_active = np.any(np.delete(active_binary, k, axis=1), axis=1)
        exclusive_counts[k] = int(np.count_nonzero(active_binary[:, k] & (~others_active)))

    primary_spks = [
        k for k in range(K)
        if exclusive_counts[k] * step >= ghost_min_exclusive_s
        and (total_counts[k] / float(total_active_frames)) >= ghost_min_share
    ]
    if not primary_spks:
        primary_spks = [int(np.argmax(total_counts))]

    for k in range(K):
        if k not in primary_spks and total_counts[k] > 0:
            # Reassign minor/ghost channel k frame-by-frame to the most probable primary speaker
            mask_k = P[:, k] > offset
            if np.any(mask_k):
                best_prim = max(primary_spks, key=lambda p: float(np.sum(P[mask_k, p])))
                P[:, best_prim] = np.maximum(P[:, best_prim], P[:, k])
            P[:, k] = 0.0

    # 2. Hysteresis Decoding at 10ms resolution
    min_on_frames = max(1, int(round(min_on_s / step)))
    min_off_frames = max(1, int(round(min_off_s / step)))

    raw_turns = []
    for k in primary_spks:
        pk = P[:, k]
        segs = []
        in_speech = False
        s_idx = 0
        for t in range(T):
            if not in_speech and pk[t] >= onset:
                in_speech = True
                s_idx = t
            elif in_speech and pk[t] < offset:
                in_speech = False
                segs.append([s_idx, t])
        if in_speech:
            segs.append([s_idx, T])

        if not segs:
            continue

        # Merge micro-gaps <= min_off_frames
        merged = [segs[0]]
        for s_i, e_i in segs[1:]:
            if s_i - merged[-1][1] <= min_off_frames:
                merged[-1][1] = e_i
            else:
                merged.append([s_i, e_i])

        for s_i, e_i in merged:
            if e_i - s_i >= min_on_frames:
                raw_turns.append((s_i * step, e_i * step, k))

    raw_turns.sort(key=lambda x: (x[0], x[1]))
    names = {}
    for _, _, k in raw_turns:
        names.setdefault(k, f"SPEAKER_{len(names):02d}")

    return [{"start": s, "end": e, "speaker": names[k]} for s, e, k in raw_turns]


def evaluate_predictions(test_meta, segs_list):
    tot_ref_s = w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
    spk_exact = short_segs = total_segs = 0
    spk_abs_err = []

    for (_, _, _, ref_turns, dur, _), segs in zip(test_meta, segs_list):
        hyp_turns = [(s["start"], s["end"], s["speaker"]) for s in segs]
        total_segs += len(segs)
        short_segs += sum(1 for s in segs if (s["end"] - s["start"]) < 0.5)

        n_ref_spk = len({s for _, _, s in ref_turns})
        n_hyp_spk = len({s for _, _, s in hyp_turns})
        if n_ref_spk == n_hyp_spk:
            spk_exact += 1
        spk_abs_err.append(abs(n_hyp_spk - n_ref_spk))

        der0, miss0, fa0, conf0, ref_s = compute_der_metrics(ref_turns, hyp_turns, dur, collar=0.0)
        der25, _, _, _, _ = compute_der_metrics(ref_turns, hyp_turns, dur, collar=0.25)
        w_der0 += der0 * ref_s
        w_der25 += der25 * ref_s
        w_miss += miss0 * ref_s
        w_fa += fa0 * ref_s
        w_conf += conf0 * ref_s
        tot_ref_s += ref_s

    n_eval = len(test_meta)
    return {
        "DER@0.00": w_der0 / tot_ref_s,
        "DER@0.25": w_der25 / tot_ref_s,
        "Miss%": w_miss / tot_ref_s,
        "FA%": w_fa / tot_ref_s,
        "Conf%": w_conf / tot_ref_s,
        "SpkExact%": spk_exact / n_eval * 100.0,
        "SpkMAE": float(np.mean(spk_abs_err)),
        "<0.5s": f"{short_segs}/{total_segs}",
    }


def main():
    test_items, train_items = load_viyt_split()
    work_dir = tempfile.mkdtemp(prefix="viyt_exp5_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    from nemo.collections.asr.models import SortformerEncLabelModel
    print("=== 1. Nạp Nemotron-3 + Deep AOSC & Trích xuất ma trận xác suất 10ms + Phổ Mel 10ms ===")
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()

    # Load Layer-4 Vietnamese weights from Exp2 if available
    ckpt_l4 = "models/nemotron_vi_upgraded.pt"
    sm = model.sortformer_modules
    if os.path.isfile(ckpt_l4):
        st = torch.load(ckpt_l4, map_location="cuda", weights_only=True)
        if st.get("subpixel_upsample") and sm.subpixel_upsample is not None:
            sm.subpixel_upsample.load_state_dict(st["subpixel_upsample"])
        if st.get("first_hidden_to_hidden"):
            sm.first_hidden_to_hidden.load_state_dict(st["first_hidden_to_hidden"])
        if st.get("single_hidden_to_spks"):
            sm.single_hidden_to_spks.load_state_dict(st["single_hidden_to_spks"])
        if st.get("hidden_to_spks"):
            sm.hidden_to_spks.load_state_dict(st["hidden_to_spks"])

    for k, v in {"spkcache_len": 384, "fifo_len": 60, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 200}.items():
        setattr(sm, k, v)
    model._check_streaming_parameters()
    install_deep_aosc_hooks(sm)

    def prepare_split(items, tag):
        paths = []
        for idx, arr, sr, _, _, _ in items:
            p = os.path.join(work_dir, f"{tag}_{idx:03d}.wav")
            sf.write(p, arr, sr)
            paths.append(p)
        probs_list = get_streaming_10ms_probs(model, paths)
        data = []
        for (idx, arr, sr, ref_turns, dur, aid), probs in zip(items, probs_list):
            n_t = probs.shape[0]
            ac_feats = extract_10ms_acoustic_features(arr, sr, n_frames_target=n_t)
            raw_target = build_frame_targets(ref_turns, n_t, step=0.01).numpy()
            aligned_target = align_columns_hungarian(probs.numpy(), raw_target)
            vad_target = (aligned_target.max(dim=-1, keepdim=True).values > 0.5).float()
            data.append({
                "idx": idx,
                "acoustic": ac_feats,
                "probs": probs,
                "target": aligned_target,
                "vad_target": vad_target,
                "ref_turns": ref_turns,
                "dur": dur,
            })
        return data

    t0 = time.time()
    test_data = prepare_split(test_items, "test")
    train_data = prepare_split(train_items, "train")
    print(f"Đã trích xuất xong 80 file Train + 20 file Test trong {time.time() - t0:.1f}s.")

    # 2. Train 10ms AcousticSkipDiarRefiner on the 80 Training files
    print("=== 2. Huấn luyện 10ms Acoustic-Skip Neural Refiner trên 80 file Train ===")
    torch.manual_seed(42)
    refiner = AcousticSkipDiarRefiner(acoustic_dim=40, hidden_dim=96).to("cuda")
    opt = torch.optim.AdamW(refiner.parameters(), lr=1.5e-3, weight_decay=1e-4)
    epochs = 25
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    rng = random.Random(42)
    batch_size = 8

    for ep in range(1, epochs + 1):
        refiner.train()
        order = list(range(len(train_data)))
        rng.shuffle(order)
        ep_loss = 0.0
        n_b = 0

        for b0 in range(0, len(order), batch_size):
            b_ids = order[b0:b0 + batch_size]
            opt.zero_grad()
            b_loss = 0.0
            for i_d in b_ids:
                d = train_data[i_d]
                ac = d["acoustic"].unsqueeze(0).to("cuda")
                pr = d["probs"].unsqueeze(0).to("cuda")
                tg = d["target"].unsqueeze(0).to("cuda")
                vt = d["vad_target"].unsqueeze(0).to("cuda")

                ref_p, vad_p = refiner(ac, pr)
                ref_p = torch.clamp(ref_p, 1e-6, 1.0 - 1e-6)
                vad_p = torch.clamp(vad_p, 1e-6, 1.0 - 1e-6)

                # Sharp 10ms Speech Activity Loss (eliminates both Miss% and FA% at 0ms collar)
                vad_loss = F.binary_cross_entropy(vad_p, vt)
                # Per-slot speaker BCE Loss
                slot_loss = F.binary_cross_entropy(ref_p, tg)
                # Active-channel Dice Loss on VAD to sharpen 10ms boundaries
                inter = (vad_p * vt).sum()
                dice_vad = 1.0 - (2.0 * inter + 1.0) / (vad_p.sum() + vt.sum() + 1.0)

                loss = slot_loss + 0.8 * vad_loss + 0.3 * dice_vad
                b_loss = b_loss + loss / len(b_ids)

            b_loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()
            ep_loss += float(b_loss.item())
            n_b += 1

        sched.step()
        if ep in (1, 5, 10, 15, 20, epochs):
            print(f"  Epoch {ep:2d}/{epochs} | Refiner Loss: {ep_loss / max(1, n_b):.4f}")

    refiner.eval()
    os.makedirs("models", exist_ok=True)
    torch.save(refiner.state_dict(), "models/nemotron_vi_acoustic_refiner.pt")
    print("[OK] Đã lưu trọng số 10ms Acoustic Refiner tại: models/nemotron_vi_acoustic_refiner.pt")

    # 3. Evaluate on the 20 unseen Test files
    with torch.no_grad():
        raw_probs_test = [d["probs"].numpy() for d in test_data]
        ref_probs_test = []
        for d in test_data:
            ac = d["acoustic"].unsqueeze(0).to("cuda")
            pr = d["probs"].unsqueeze(0).to("cuda")
            rp, _ = refiner(ac, pr)
            ref_probs_test.append(rp.squeeze(0).cpu().numpy())

    results = []

    # Config 1: Raw Exp2 + Quorum Pruning
    segs_raw = [probs_to_segments(p, onset=0.55, offset=0.45, min_on_s=0.16, min_off_s=0.10) for p in raw_probs_test]
    r1 = evaluate_predictions(test_items, segs_raw)
    r1["name"] = "1_Nemotron_Exp2+Quorum"
    results.append(r1)

    # Sweep 4 threshold/pruning presets on the 10ms Acoustic-Skip Refiner output
    presets = [
        ("2_AcousticSkip_Balanced", 0.50, 0.40, 0.12, 0.05, 1.5, 0.03),
        ("3_AcousticSkip_Crisp",    0.54, 0.44, 0.12, 0.04, 1.8, 0.035),
        ("4_AcousticSkip_Ultra",    0.52, 0.42, 0.10, 0.04, 2.0, 0.04),
        ("5_AcousticSkip_LowMiss",  0.46, 0.36, 0.10, 0.05, 1.8, 0.035),
    ]
    for name, on_th, off_th, min_on, min_off, g_sec, g_share in presets:
        segs = [
            probs_to_segments(
                p, onset=on_th, offset=off_th, min_on_s=min_on, min_off_s=min_off,
                ghost_min_exclusive_s=g_sec, ghost_min_share=g_share,
            )
            for p in ref_probs_test
        ]
        res = evaluate_predictions(test_items, segs)
        res["name"] = name
        results.append(res)

    print("\n" + "=" * 106)
    print("KẾT QUẢ THỬ NGHIỆM 5: 10ms ACOUSTIC-SKIP NEURAL REFINER + QUORUM PRUNING (ViYT-Diar Test N=20)")
    print("=" * 106)
    print(f"{'Cấu hình':25s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s}")
    print("-" * 106)
    for r in results:
        print(f"{r['name']:25s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s}")
    print("=" * 106)


if __name__ == "__main__":
    main()
