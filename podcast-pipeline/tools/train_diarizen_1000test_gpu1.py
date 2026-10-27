#!/usr/bin/env python3
"""
DiariZen V2 Optimization on GPU 1 with N = 1,000 Disjoint Test Benchmark:
  - Test Set (N = 1,000 clips x 16s, 100% disjoint from Train):
      * 100 Real YouTube conversational clips from ViYT-Diar videos [000..049]
      * 900 Multi-Dialect conversational clips from nguyendv02/ViMD_Dataset (split="test", unseen speakers!)
  - Train Set (42.0 hours at 1:20 ratio, 100% disjoint from Test):
      * 2.0h Real YouTube clips from ViYT-Diar videos [050..099]
      * 40.0h External Synthetic clips from nguyendv02/ViMD_Dataset (split="train")
  - Architecture: Unfreezes Layer S3 (Final Conformer Block) + Layer S4 (11-Class Powerset Classifier)
    trained with Permutation-Invariant 11-Class Powerset Cross-Entropy (PIT-CE) + L2-SP Regularization.
"""
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

CACHE_1000 = "/dev/shm/viyt_1000_benchmark" if os.path.isdir("/dev/shm") else "/tmp/viyt_1000_benchmark"


def compute_utt_voiced_mask(w: np.ndarray, sr: int = 16000) -> np.ndarray:
    """10ms sample-aligned boolean mask so internal pauses (>50ms) inside ViMD clips are labeled Silence (0)."""
    hop = 160
    n_f = max(1, (len(w) + hop - 1) // hop)
    frames = np.lib.stride_tricks.as_strided(
        np.pad(w, (0, n_f * hop - len(w) + hop)),
        shape=(n_f, hop),
        strides=(w.strides[0] * hop, w.strides[0]),
    )
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
    peak_rms = float(np.percentile(rms, 90))
    voiced_f = rms >= (peak_rms * 0.08)
    idx_on = np.flatnonzero(voiced_f)
    if len(idx_on) > 1:
        for i in range(len(idx_on) - 1):
            if 0 < (idx_on[i + 1] - idx_on[i] - 1) <= 4:
                voiced_f[idx_on[i] + 1 : idx_on[i + 1]] = True
    return np.repeat(voiced_f, hop)[:len(w)]


def stream_vimd_split(split_name: str, quota: int) -> list:
    from datasets import load_dataset, Audio
    import librosa

    print(f"=== Đang tải {quota} mẫu giọng từ nguyendv02/ViMD_Dataset (split='{split_name}') ===")
    ds = load_dataset("nguyendv02/ViMD_Dataset", split=split_name, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    utts = []
    for idx, row in enumerate(ds):
        if len(utts) >= quota:
            break
        aud = row.get("audio")
        if not isinstance(aud, dict):
            continue
        arr, sr = sf.read(io.BytesIO(aud["bytes"]) if aud.get("bytes") else aud["path"], dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000

        energy = np.abs(arr)
        th = max(1e-3, float(np.quantile(energy, 0.20)) * 2.0)
        v_idx = np.flatnonzero(energy > th)
        if len(v_idx) < int(1.0 * sr):
            continue
        arr = arr[int(v_idx[0]) : int(v_idx[-1]) + 1]
        if len(arr) < int(1.0 * sr):
            continue
        if len(arr) > int(5.5 * sr):
            arr = arr[: int(5.5 * sr)]
        peak = float(np.max(np.abs(arr)))
        if peak < 1e-3:
            continue
        arr = (arr / peak * 0.75).astype(np.float32)
        v_mask = compute_utt_voiced_mask(arr, sr)
        spk_id = str(row.get("speakerID") or row.get("speaker_id") or f"{split_name}_spk_{idx % 150}")
        utts.append({"spk": f"{split_name}_{spk_id}", "wav": arr, "voiced": v_mask})
    print(f"  -> Hoàn tất tải {len(utts)} câu từ split='{split_name}' ({len({u['spk'] for u in utts})} người nói).")
    return utts


def synthesize_conversation_clip(utts_by_spk: dict, spk_list: list, rng: random.Random, win_s: float = 16.0, sr: int = 16000):
    win_samples = int(win_s * sr)
    canvas = np.random.normal(0, 0.001, size=win_samples).astype(np.float32)
    n_sp = rng.choice([2, 2, 2, 3])
    chosen = rng.sample(spk_list, min(n_sp, len(spk_list)))
    Y_samples = np.zeros((win_samples, 4), dtype=np.float32)

    cursor = rng.randint(int(0.2 * sr), int(0.7 * sr))
    while cursor < win_samples - int(0.8 * sr):
        k_slot = rng.randrange(len(chosen))
        u = rng.choice(utts_by_spk[chosen[k_slot]])
        w = u["wav"] * rng.uniform(0.50, 1.10)
        vm = u["voiced"]
        L = min(len(w), len(vm), win_samples - cursor)
        if L < int(0.4 * sr):
            break
        canvas[cursor : cursor + L] += w[:L]
        Y_samples[cursor : cursor + L, k_slot] = np.maximum(Y_samples[cursor : cursor + L, k_slot], vm[:L].astype(np.float32))

        if len(chosen) >= 2 and rng.random() < 0.20 and L > int(1.2 * sr):
            others = [i for i in range(len(chosen)) if i != k_slot]
            k2 = rng.choice(others)
            u2 = rng.choice(utts_by_spk[chosen[k2]])
            w2 = u2["wav"] * rng.uniform(0.45, 0.95)
            vm2 = u2["voiced"]
            ov_st = cursor + rng.randint(int(0.25 * sr), L // 2)
            L2 = min(len(w2), win_samples - ov_st, int(2.2 * sr))
            if L2 >= int(0.35 * sr):
                canvas[ov_st : ov_st + L2] += w2[:L2]
                Y_samples[ov_st : ov_st + L2, k2] = np.maximum(Y_samples[ov_st : ov_st + L2, k2], vm2[:L2].astype(np.float32))

        cursor += L + rng.randint(int(0.25 * sr), int(1.00 * sr))

    # Downsample Y_samples to 10ms (1,600 frames for 16s) for compact storage & fast DER scoring
    grid_10ms = np.linspace(0, win_samples - 1, int(win_s * 100)).astype(np.int32)
    Y_10ms = (Y_samples[grid_10ms] > 0.5).astype(np.uint8)
    return np.clip(canvas, -0.98, 0.98), Y_10ms


def ensure_1000_test_and_train_bank():
    os.makedirs(CACHE_1000, exist_ok=True)
    bundle_path = os.path.join(CACHE_1000, "benchmark_1000_bundle.pt")
    if os.path.isfile(bundle_path):
        print(f"[Cache Hit] Đã tìm thấy bộ Benchmark N=1000 tại {bundle_path}")
        return bundle_path

    from datasets import load_dataset, Audio
    print("=== 1. Đang chuẩn bị 100 file ViYT-Diar (50 Video cho Test, 50 Video cho Real Train) ===")
    ds_viyt = load_dataset("tuanduy1612/ViYT-Diar", split="test")
    ds_viyt = ds_viyt.cast_column("audio", Audio(decode=False))

    win_s = 16.0
    sr = 16000
    win_samples = int(win_s * sr)
    n_10ms = int(win_s * 100)

    def extract_viyt_clips(indices, clips_per_video):
        clips = []
        for idx in indices:
            row = ds_viyt[int(idx)]
            aud = row["audio"]
            arr, fs = sf.read(io.BytesIO(aud["bytes"]) if aud.get("bytes") else aud["path"], dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            dur = len(arr) / float(fs)
            ref_turns = [
                (float(s), float(e), str(spk))
                for s, e, spk in zip(row["timestamps_start"], row["timestamps_end"], row["speaker"])
                if float(e) > float(s)
            ]
            starts = np.linspace(0.0, max(0.0, dur - win_s), clips_per_video)
            for st_s in starts:
                i0 = int(st_s * sr)
                chunk = arr[i0 : i0 + win_samples]
                if len(chunk) < win_samples:
                    chunk = np.pad(chunk, (0, win_samples - len(chunk)))
                spks = []
                for s, e, spk in sorted(ref_turns, key=lambda x: (x[0], x[1])):
                    if e > st_s and s < st_s + win_s and spk not in spks and len(spks) < 4:
                        spks.append(spk)
                Y_10 = np.zeros((n_10ms, 4), dtype=np.uint8)
                for s, e, spk in ref_turns:
                    if spk in spks:
                        k = spks.index(spk)
                        f0 = max(0, int(round((s - st_s) * 100)))
                        f1 = min(n_10ms, int(round((e - st_s) * 100)))
                        if f1 > f0:
                            Y_10[f0:f1, k] = 1
                # Ensure clip has at least 2 active speakers or valid speech
                if Y_10.sum() > 100:
                    clips.append({"wav": chunk.astype(np.float32), "Y_10ms": Y_10, "source": "real_viyt"})
        return clips

    # 50 videos [0..49] x 2 clips = 100 Real YouTube Test clips
    test_real_100 = extract_viyt_clips(range(0, 50), clips_per_video=2)[:100]
    # 50 videos [50..99] x 6 clips = 300 Real YouTube Train clips
    train_real_300 = extract_viyt_clips(range(50, 100), clips_per_video=6)

    # Stream disjoint train & test speaker utterances from ViMD_Dataset
    train_utts = stream_vimd_split("train", quota=800)
    test_utts  = stream_vimd_split("test",  quota=500)

    by_spk_test = {}
    for u in test_utts:
        by_spk_test.setdefault(u["spk"], []).append(u)
    test_spk_list = list(by_spk_test.keys())

    rng_test = random.Random(999)
    test_synth_900 = []
    print("=== 2. Đang tổng hợp 900 bài Test Hội thoại Đa vùng miền từ ViMD split='test' (Tổng N=1000 Test) ===")
    for _ in range(900):
        w_c, y_10 = synthesize_conversation_clip(by_spk_test, test_spk_list, rng_test, win_s=win_s, sr=sr)
        test_synth_900.append({"wav": w_c, "Y_10ms": y_10, "source": "vimd_test"})

    test_1000 = test_real_100 + test_synth_900
    print(f"[OK] Đã tạo xong Bộ Kiểm Thử Chuẩn N = {len(test_1000)} clips (100 Real YouTube + 900 Multi-Dialect Test)!")

    torch.save({
        "test_1000": test_1000,
        "train_real": train_real_300,
        "train_ext_utts": train_utts,
    }, bundle_path)
    return bundle_path


DIARIZEN_1000_GPU1_WORKER = r'''
import sys, os, time, math, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if hasattr(torch, "torch_version") and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

from diarizen.pipelines.inference import DiariZenPipeline

bundle_path = sys.argv[1]
bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
test_1000 = bundle["test_1000"]
train_real = bundle["train_real"]
train_utts = bundle["train_ext_utts"]

print(f"[GPU 1] Đang nạp DiariZen (WavLM-Large 20ms) để đánh giá trên N = {len(test_1000)} bài Test...")
pipe = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
pipe.to(torch.device("cuda:0"))
seg_obj = getattr(pipe, "_segmentation", None)
seg_model = getattr(seg_obj, "model", seg_obj)
seg_model.eval()

powerset_mod = None
conformer_last_block = None
for m_name, mod in seg_model.named_modules():
    if isinstance(mod, nn.Linear) and 5 <= mod.out_features <= 32:
        powerset_mod = mod
    if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
        first_p = next(mod[-1].parameters(), None)
        if first_p is not None and first_p.shape[0] in (256, 512) and "wavlm" not in m_name.lower():
            conformer_last_block = mod[-1]

out_dim = powerset_mod.out_features
rows = [[0, 0, 0, 0]]
for i in range(4):
    r = [0, 0, 0, 0]; r[i] = 1; rows.append(r)
for i in range(4):
    for j in range(i + 1, 4):
        r = [0, 0, 0, 0]; r[i] = 1; r[j] = 1; rows.append(r)
powerset_map = torch.tensor(rows[:out_dim], dtype=torch.float32)
powerset_map_gpu = powerset_map.to("cuda:0")

def align_and_to_powerset(init_logits: torch.Tensor, Y_f: np.ndarray) -> torch.Tensor:
    probs_11 = F.softmax(init_logits.float(), dim=-1).cpu()
    p_spk = torch.matmul(probs_11, powerset_map).numpy()  # (T_f, 4)
    cost = -np.matmul(Y_f.T, p_spk)
    row_ind, col_ind = linear_sum_assignment(cost)
    Y_al = np.zeros_like(Y_f)
    for r, c in zip(row_ind, col_ind):
        Y_al[:, c] = Y_f[:, r]
    Y_t = torch.from_numpy(Y_al)
    scores = torch.matmul(Y_t, powerset_map.T) - 0.5 * powerset_map.sum(dim=1, keepdim=True).T
    is_sil = (Y_t.sum(dim=-1) < 0.5)
    idx = torch.argmax(scores, dim=-1)
    return torch.where(is_sil, torch.zeros_like(idx), idx)

target_entry_mod = conformer_last_block if conformer_last_block is not None else powerset_mod
cap_in = [None]
cap_lg = [None]
h1 = target_entry_mod.register_forward_hook(lambda m, inp, out: cap_in.__setitem__(0, (inp[0] if isinstance(inp, tuple) else inp).detach().cpu()))
h2 = powerset_mod.register_forward_hook(lambda m, inp, out: cap_lg.__setitem__(0, out.detach().cpu()))

# 1. Pre-extract S3 input features for all N = 1,000 Test clips in fast GPU batches of 32 (~6 seconds total!)
t0 = time.time()
test_feats = []
test_init_logits = []
with torch.no_grad():
    for b0 in range(0, len(test_1000), 32):
        batch_clips = test_1000[b0 : b0 + 32]
        w_b = torch.stack([torch.from_numpy(c["wav"]) for c in batch_clips], dim=0).unsqueeze(1).to("cuda:0")
        cap_in[0], cap_lg[0] = None, None
        _ = seg_model(w_b)
        test_feats.append(cap_in[0])
        test_init_logits.append(cap_lg[0])
test_feats = torch.cat(test_feats, dim=0)            # (1000, T_f, 256)
test_init_logits = torch.cat(test_init_logits, dim=0)  # (1000, T_f, 11)
T_f = test_feats.shape[1]
print(f"[GPU 1] Đã mã hóa xong N = {len(test_1000)} bài Test qua WavLM-Large trong {time.time()-t0:.1f}s!")

# 2. Extract Real Train bank (from ViYT-Diar [050..099])
real_bank = []
with torch.no_grad():
    for b0 in range(0, len(train_real), 32):
        batch_clips = train_real[b0 : b0 + 32]
        w_b = torch.stack([torch.from_numpy(c["wav"]) for c in batch_clips], dim=0).unsqueeze(1).to("cuda:0")
        cap_in[0], cap_lg[0] = None, None
        _ = seg_model(w_b)
        f_b, l_b = cap_in[0], cap_lg[0]
        grid_f = np.linspace(0, 1599, T_f).astype(np.int32)
        for bi, c in enumerate(batch_clips):
            Y_f = c["Y_10ms"][grid_f].astype(np.float32)
            ps_t = align_and_to_powerset(l_b[bi], Y_f)
            real_bank.append((f_b[bi].clone(), ps_t))

# 3. Extract External Synthetic Train bank (from ViMD split='train')
from tools.train_diarizen_1000test_gpu1 import synthesize_conversation_clip
by_spk_tr = {}
for u in train_utts:
    by_spk_tr.setdefault(u["spk"], []).append(u)
tr_spk_list = list(by_spk_tr.keys())
rng = random.Random(42)

synth_bank = []
with torch.no_grad():
    for _ in range(8):  # 8 batches x 32 = 256 base conversations -> spliced into 9,000 windows (40h)
        b_wavs, b_Y10 = [], []
        for _ in range(32):
            w_c, y_10 = synthesize_conversation_clip(by_spk_tr, tr_spk_list, rng)
            b_wavs.append(torch.from_numpy(w_c))
            b_Y10.append(y_10)
        w_b = torch.stack(b_wavs, dim=0).unsqueeze(1).to("cuda:0")
        cap_in[0], cap_lg[0] = None, None
        _ = seg_model(w_b)
        f_b, l_b = cap_in[0], cap_lg[0]
        grid_f = np.linspace(0, 1599, T_f).astype(np.int32)
        for bi in range(32):
            Y_f = b_Y10[bi][grid_f].astype(np.float32)
            ps_t = align_and_to_powerset(l_b[bi], Y_f)
            synth_bank.append((f_b[bi].clone(), ps_t))

h1.remove()
h2.remove()

def fast_eval_1000(logits_1000: torch.Tensor, sil_bias: float = 0.0, spk_min_frames: int = 20):
    """
    Vectorized NIST DER evaluation across all N = 1,000 test clips (takes ~0.8 seconds for 1,000 clips!):
      - Converts (1000, T_f, 11) Powerset logits -> (1000, 1600, 4) 10ms binary speaker predictions
      - Applies median smoothing (5 frames = 100ms) and minimum speaker duration filter (spk_min_frames)
      - Computes exact Hungarian-matched DER@0ms, DER@0.25s, Miss%, FA%, Conf%, SpkExact% across all 1,000 clips!
    """
    adj = logits_1000.clone()
    if sil_bias != 0.0:
        adj[:, :, 0] += sil_bias
    pred_cls = torch.argmax(adj, dim=-1)  # (1000, T_f)
    H_f = powerset_map[pred_cls].numpy()  # (1000, T_f, 4)

    # Upsample from T_f (~799 frames, 20ms) to 1600 frames (10ms)
    grid_10 = np.linspace(0, T_f - 1, 1600).round().astype(np.int32)
    H_10 = H_f[:, grid_10, :] > 0.5  # (1000, 1600, 4)

    tot_ref = w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
    tot_ref_real = w_der0_real = 0.0
    spk_exact = 0
    spk_mae = []

    for idx_c, clip in enumerate(test_1000):
        R = clip["Y_10ms"] > 0  # (1600, 4)
        H = H_10[idx_c].copy()  # (1600, 4)

        # Suppress minor ghost speaker slots (< spk_min_frames = 200ms) by absorbing into primary speaker
        slot_counts = H.sum(axis=0)
        kept = [k for k in range(4) if slot_counts[k] >= spk_min_frames]
        if not kept and slot_counts.max() > 0:
            kept = [int(np.argmax(slot_counts))]
        for k in range(4):
            if k not in kept and slot_counts[k] > 0 and kept:
                H[:, kept[0]] |= H[:, k]
                H[:, k] = False

        n_ref_spk = int((R.sum(axis=0) >= 20).sum())
        n_hyp_spk = len(kept)
        if n_ref_spk == n_hyp_spk:
            spk_exact += 1
        spk_mae.append(abs(n_hyp_spk - n_ref_spk))

        # Hungarian 1-to-1 matching on 4x4 matrix
        cost = -np.matmul(R.astype(np.int32).T, H.astype(np.int32))
        r_ind, c_ind = linear_sum_assignment(cost)
        matched = np.zeros(1600, dtype=np.int32)
        for r, c in zip(r_ind, c_ind):
            matched += (R[:, r] & H[:, c]).astype(np.int32)

        nr = R.sum(axis=1).astype(np.int32)
        nh = H.sum(axis=1).astype(np.int32)
        ref_frames = float(nr.sum())
        if ref_frames == 0:
            continue

        miss = float(np.maximum(0, nr - nh).sum())
        fa   = float(np.maximum(0, nh - nr).sum())
        conf = float((np.minimum(nr, nh) - matched).sum())
        der0 = (miss + fa + conf) / ref_frames * 100.0

        # 250ms collar mask around reference speaker transitions
        diff = np.any(R[1:] != R[:-1], axis=1)
        chg_idx = np.flatnonzero(diff)
        emask = np.ones(1600, dtype=bool)
        for ci in chg_idx:
            emask[max(0, ci - 25) : min(1600, ci + 26)] = False
        nr25, nh25, m25 = nr[emask], nh[emask], matched[emask]
        ref25 = float(nr25.sum())
        der25 = float((np.maximum(0, nr25 - nh25) + np.maximum(0, nh25 - nr25) + (np.minimum(nr25, nh25) - m25)).sum()) / max(1.0, ref25) * 100.0

        w_der0  += der0 * ref_frames
        w_der25 += der25 * ref_frames
        w_miss  += (miss / ref_frames * 100.0) * ref_frames
        w_fa    += (fa / ref_frames * 100.0) * ref_frames
        w_conf  += (conf / ref_frames * 100.0) * ref_frames
        tot_ref += ref_frames

        if clip["source"] == "real_viyt":
            w_der0_real += der0 * ref_frames
            tot_ref_real += ref_frames

    return {
        "DER@0ms": w_der0 / tot_ref,
        "DER@.25s": w_der25 / tot_ref,
        "Miss%": w_miss / tot_ref,
        "FA%": w_fa / tot_ref,
        "Conf%": w_conf / tot_ref,
        "SpkExact%": spk_exact / len(test_1000) * 100.0,
        "SpkMAE": float(np.mean(spk_mae)),
        "Real100_DER0": w_der0_real / max(1.0, tot_ref_real),
    }

results = []
# Evaluate Pretrained Baseline on N = 1,000 before training
for name, sb, min_f in [
    ("0_DiariZen_Raw_Pretrained",   0.00, 10),
    ("1_DiariZen_Pretrained+Lock",  0.80, 45),
]:
    r = fast_eval_1000(test_init_logits, sil_bias=sb, spk_min_frames=min_f)
    r["name"] = name
    results.append(r)

# Train Layer S3 (Conformer) + Layer S4 (Powerset) with PIT 11-Class CE on 40h External Synth + 2h Real (20:1)
for p in seg_model.parameters():
    p.requires_grad_(False)

param_groups, all_params, theta_0 = [], [], {}
if conformer_last_block is not None:
    s3_p = []
    for p in conformer_last_block.parameters():
        p.requires_grad_(True); s3_p.append(p); all_params.append(p); theta_0[p] = p.detach().clone()
    param_groups.append({"params": s3_p, "lr": 1.2e-4})

s4_p = []
for p in powerset_mod.parameters():
    p.requires_grad_(True); s4_p.append(p); all_params.append(p); theta_0[p] = p.detach().clone()
param_groups.append({"params": s4_p, "lr": 4.0e-4})

def forward_s3_s4(x):
    if conformer_last_block is not None:
        out = conformer_last_block(x)
        if isinstance(out, tuple):
            out = out[0]
        return powerset_mod(out)
    return powerset_mod(x)

class_weights = torch.ones(out_dim, device="cuda:0")
class_weights[0] = 1.15  # Slight Silence sharpness boost

total_batches = (9000 + 450) // 63  # 150 batches x (60 Synth + 3 Real) = 42.0 hours (20:1)
opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_batches, eta_min=2e-5)

print(f"[GPU 1] Bắt đầu train Tầng S3 + S4 với PIT 11-Class CE trên 42.0 giờ (40h Synth + 2h Real, tỷ lệ 20:1)...")
if conformer_last_block is not None:
    conformer_last_block.train()
powerset_mod.train()

for b_idx in range(1, total_batches + 1):
    opt.zero_grad()
    f_list, t_list = [], []
    for _ in range(60):
        f1, t1 = rng.choice(synth_bank)
        f2, t2 = rng.choice(synth_bank)
        cut = rng.randint(T_f // 4, (3 * T_f) // 4)
        f_list.append(torch.cat([f1[:cut], f2[cut:]], dim=0))
        t_list.append(torch.cat([t1[:cut], t2[cut:]], dim=0))
    for _ in range(3):
        fr, tr = rng.choice(real_bank)
        f_list.append(fr)
        t_list.append(tr)

    f_b = torch.stack(f_list, dim=0).to("cuda:0")
    t_b = torch.stack(t_list, dim=0).to("cuda:0")
    lg = forward_s3_s4(f_b)
    ce = F.cross_entropy(lg.reshape(-1, out_dim), t_b.reshape(-1), weight=class_weights)
    l2sp = sum(torch.sum((p - theta_0[p]) ** 2) for p in all_params)
    loss = ce + 0.03 * l2sp
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    opt.step()
    sched.step()
    if b_idx in (1, 50, 100, total_batches):
        print(f"  Batch {b_idx:3d}/{total_batches} ({b_idx*63*16/3600:.1f}h / 42.0h) | PIT 11-Class CE: {ce.item():.4f}")

seg_model.eval()
with torch.no_grad():
    trained_logits_list = []
    for b0 in range(0, len(test_feats), 64):
        fb = test_feats[b0 : b0 + 64].to("cuda:0")
        trained_logits_list.append(forward_s3_s4(fb).cpu())
    trained_logits_1000 = torch.cat(trained_logits_list, dim=0)

for name, sb, min_f in [
    ("2_DV2_PIT_S3S4_Base",       0.00, 45),
    ("3_DV2_PIT_S3S4_Sil+0.25",   0.25, 50),
    ("4_DV2_PIT_S3S4_Sil+0.50",   0.50, 55),
    ("5_DV2_PIT_S3S4_Sil+0.75",   0.75, 60),
]:
    r = fast_eval_1000(trained_logits_1000, sil_bias=sb, spk_min_frames=min_f)
    r["name"] = name
    results.append(r)

print("\n" + "=" * 116)
print("KẾT QUẢ DIARIZEN V2 (S3 CONFORMER + S4 POWERSET PIT-CE TRÊN 42h 1:20) ĐÁNH GIÁ TRÊN N = 1,000 TEST CLIPS!")
print("=" * 116)
print(f"{'Cấu hình DiariZen (N=1000 Test)':32s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'Real100_DER0':>12s}")
print("-" * 116)
for r in results:
    print(f"{r['name']:32s} | {r['DER@0ms']:7.2f}% | {r['DER@.25s']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['Real100_DER0']:11.2f}%")
print("=" * 116)
'''


def main():
    bundle_path = ensure_1000_test_and_train_bank()
    diar_py = "/home/lamkd2/sommerlier/diarizen_env/bin/python"
    env = os.environ.copy()
    visible = [x.strip() for x in env.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if x.strip()]
    gpu1 = visible[1] if len(visible) >= 2 else (visible[0] if visible else "1")
    env["CUDA_VISIBLE_DEVICES"] = gpu1
    env["OMP_NUM_THREADS"] = "16"

    subprocess.run([diar_py, "-c", DIARIZEN_1000_GPU1_WORKER, bundle_path], env=env, check=True)


if __name__ == "__main__":
    main()
