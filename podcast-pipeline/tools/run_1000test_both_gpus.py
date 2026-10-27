#!/usr/bin/env python3
"""
N = 1,000 Disjoint Test Benchmark + Two-Stage Curriculum Training (40h External Synthetic -> 2h Real Calibration)
Runs in parallel across 2 GPUs:
  - GPU 1: DiariZen V2 (Unfrozen Conformer S3 + Powerset S4 with PIT 11-Class CE, Two-Stage Curriculum)
  - GPU 0: Nemotron-3 (Layer 4 Upsampler + Heads with Hungarian-Aligned BCE, Two-Stage Curriculum)
"""
import concurrent.futures
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

from tools.train_diarizen_1000test_gpu1 import ensure_1000_test_and_train_bank, CACHE_1000

warnings.filterwarnings("ignore")


DIARIZEN_2STAGE_1000_WORKER = r'''
import sys, os, time, math, random, json, warnings
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
from tools.train_diarizen_1000test_gpu1 import synthesize_conversation_clip

bundle_path = sys.argv[1]
bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
test_1000 = bundle["test_1000"]
train_real = bundle["train_real"]
train_utts = bundle["train_ext_utts"]

print(f"[GPU 1 - DiariZen] Nạp mô hình & mã hóa N = {len(test_1000)} bài Test...")
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

def align_and_to_powerset(init_logits: torch.Tensor, Y_f: np.ndarray) -> torch.Tensor:
    probs_11 = F.softmax(init_logits.float(), dim=-1).cpu()
    p_spk = torch.matmul(probs_11, powerset_map).numpy()
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
cap_in, cap_lg = [None], [None]
h1 = target_entry_mod.register_forward_hook(lambda m, inp, out: cap_in.__setitem__(0, (inp[0] if isinstance(inp, tuple) else inp).detach().cpu()))
h2 = powerset_mod.register_forward_hook(lambda m, inp, out: cap_lg.__setitem__(0, out.detach().cpu()))

t0 = time.time()
test_feats, test_init_logits = [], []
with torch.no_grad():
    for b0 in range(0, len(test_1000), 32):
        batch_clips = test_1000[b0 : b0 + 32]
        w_b = torch.stack([torch.from_numpy(c["wav"]) for c in batch_clips], dim=0).unsqueeze(1).to("cuda:0")
        _ = seg_model(w_b)
        test_feats.append(cap_in[0])
        test_init_logits.append(cap_lg[0])
test_feats = torch.cat(test_feats, dim=0)
test_init_logits = torch.cat(test_init_logits, dim=0)
T_f = test_feats.shape[1]
print(f"[GPU 1 - DiariZen] Đã mã hóa N={len(test_1000)} bài Test trong {time.time()-t0:.1f}s.")

real_bank = []
with torch.no_grad():
    for b0 in range(0, len(train_real), 32):
        batch_clips = train_real[b0 : b0 + 32]
        w_b = torch.stack([torch.from_numpy(c["wav"]) for c in batch_clips], dim=0).unsqueeze(1).to("cuda:0")
        _ = seg_model(w_b)
        f_b, l_b = cap_in[0], cap_lg[0]
        grid_f = np.linspace(0, 1599, T_f).astype(np.int32)
        for bi, c in enumerate(batch_clips):
            Y_f = c["Y_10ms"][grid_f].astype(np.float32)
            real_bank.append((f_b[bi].clone(), align_and_to_powerset(l_b[bi], Y_f)))

by_spk_tr = {}
for u in train_utts:
    by_spk_tr.setdefault(u["spk"], []).append(u)
tr_spk_list = list(by_spk_tr.keys())
rng = random.Random(42)

synth_bank = []
with torch.no_grad():
    for _ in range(10):  # 320 consistent 16s multi-speaker conversations
        b_wavs, b_Y10 = [], []
        for _ in range(32):
            w_c, y_10 = synthesize_conversation_clip(by_spk_tr, tr_spk_list, rng)
            b_wavs.append(torch.from_numpy(w_c))
            b_Y10.append(y_10)
        w_b = torch.stack(b_wavs, dim=0).unsqueeze(1).to("cuda:0")
        _ = seg_model(w_b)
        f_b, l_b = cap_in[0], cap_lg[0]
        grid_f = np.linspace(0, 1599, T_f).astype(np.int32)
        for bi in range(32):
            Y_f = b_Y10[bi][grid_f].astype(np.float32)
            synth_bank.append((f_b[bi].clone(), align_and_to_powerset(l_b[bi], Y_f)))

h1.remove()
h2.remove()

def eval_1000(logits_1000: torch.Tensor, sil_bias: float = 0.0, spk_min_frames: int = 35):
    adj = logits_1000.clone()
    if sil_bias != 0.0:
        adj[:, :, 0] += sil_bias
    pred_cls = torch.argmax(adj, dim=-1)
    H_f = powerset_map[pred_cls].numpy()
    grid_10 = np.linspace(0, T_f - 1, 1600).round().astype(np.int32)
    H_10 = H_f[:, grid_10, :] > 0.5

    tot_ref = w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
    tot_ref_real = w_der0_real = 0.0
    spk_exact = 0
    spk_mae = []

    for idx_c, clip in enumerate(test_1000):
        R = clip["Y_10ms"] > 0
        H = H_10[idx_c].copy()
        slot_counts = H.sum(axis=0)
        kept = [k for k in range(4) if slot_counts[k] >= spk_min_frames]
        if not kept and slot_counts.max() > 0:
            kept = [int(np.argmax(slot_counts))]
        for k in range(4):
            if k not in kept and slot_counts[k] > 0 and kept:
                H[:, kept[0]] |= H[:, k]
                H[:, k] = False

        n_ref_spk = int((R.sum(axis=0) >= 35).sum())
        n_hyp_spk = len(kept)
        if n_ref_spk == n_hyp_spk:
            spk_exact += 1
        spk_mae.append(abs(n_hyp_spk - n_ref_spk))

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
for name, sb, mf in [
    ("D0_Pretrained_Raw",        0.00, 10),
    ("D1_Pretrained+Sil0.6+Lk",  0.60, 45),
]:
    r = eval_1000(test_init_logits, sil_bias=sb, spk_min_frames=mf)
    r["name"] = name
    results.append(r)

# Unfreeze S3 + S4 for Two-Stage Curriculum Training
for p in seg_model.parameters():
    p.requires_grad_(False)
param_groups, all_params, theta_0 = [], [], {}
if conformer_last_block is not None:
    s3_p = []
    for p in conformer_last_block.parameters():
        p.requires_grad_(True); s3_p.append(p); all_params.append(p); theta_0[p] = p.detach().clone()
    param_groups.append({"params": s3_p, "lr": 1.0e-4, "init_lr": 1.0e-4})
s4_p = []
for p in powerset_mod.parameters():
    p.requires_grad_(True); s4_p.append(p); all_params.append(p); theta_0[p] = p.detach().clone()
param_groups.append({"params": s4_p, "lr": 3.5e-4, "init_lr": 3.5e-4})

def forward_s3_s4(x):
    if conformer_last_block is not None:
        out = conformer_last_block(x)
        if isinstance(out, tuple):
            out = out[0]
        return powerset_mod(out)
    return powerset_mod(x)

if conformer_last_block is not None:
    conformer_last_block.train()
powerset_mod.train()

opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
class_weights = torch.ones(out_dim, device="cuda:0")
class_weights[0] = 1.20

# STAGE 1: 40h External Synthetic + 2h Real (20:1 ratio, NO cross-window splicing that breaks speaker identity!)
print("[GPU 1 - DiariZen] Stage 1: Huấn luyện 40h External Synthetic + 2h Real (150 batches x 63, giữ nguyên tính toàn vẹn Speaker ID)...")
for b_idx in range(1, 151):
    opt.zero_grad()
    f_list, t_list = [], []
    for _ in range(60):
        f1, t1 = rng.choice(synth_bank)
        # Consistent window augmentation (amplitude scaling + slight Gaussian noise, NO speaker-breaking splice!)
        scale = rng.uniform(0.85, 1.15)
        noise = torch.randn_like(f1) * 0.02
        f_list.append(f1 * scale + noise)
        t_list.append(t1)
    for _ in range(3):
        fr, tr = rng.choice(real_bank)
        f_list.append(fr)
        t_list.append(tr)
    f_b = torch.stack(f_list, dim=0).to("cuda:0")
    t_b = torch.stack(t_list, dim=0).to("cuda:0")
    lg = forward_s3_s4(f_b)
    ce = F.cross_entropy(lg.reshape(-1, out_dim), t_b.reshape(-1), weight=class_weights)
    l2sp = sum(torch.sum((p - theta_0[p]) ** 2) for p in all_params)
    loss = ce + 0.04 * l2sp
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    opt.step()

# STAGE 2: Domain Calibration on Balanced Real + Synthetic (50% Real + 50% Synthetic for 30 steps with 0.25x LR)
print("[GPU 1 - DiariZen] Stage 2: Hiệu chỉnh miền (Domain Calibration) trên dữ liệu Real + Synthetic cân bằng...")
theta_stage1 = {p: p.detach().clone() for p in all_params}
for g in opt.param_groups:
    g["lr"] = g["init_lr"] * 0.25

for b_idx in range(1, 31):
    opt.zero_grad()
    f_list, t_list = [], []
    for _ in range(32):
        fr, tr = rng.choice(real_bank)
        f_list.append(fr * rng.uniform(0.9, 1.1))
        t_list.append(tr)
    for _ in range(32):
        fs, ts = rng.choice(synth_bank)
        f_list.append(fs * rng.uniform(0.9, 1.1))
        t_list.append(ts)
    f_b = torch.stack(f_list, dim=0).to("cuda:0")
    t_b = torch.stack(t_list, dim=0).to("cuda:0")
    lg = forward_s3_s4(f_b)
    ce = F.cross_entropy(lg.reshape(-1, out_dim), t_b.reshape(-1), weight=class_weights)
    l2sp = sum(torch.sum((p - theta_stage1[p]) ** 2) for p in all_params)
    loss = ce + 0.08 * l2sp
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    opt.step()

seg_model.eval()
with torch.no_grad():
    trained_logits_list = []
    for b0 in range(0, len(test_feats), 64):
        fb = test_feats[b0 : b0 + 64].to("cuda:0")
        trained_logits_list.append(forward_s3_s4(fb).cpu())
    trained_logits_1000 = torch.cat(trained_logits_list, dim=0)

for name, sb, mf in [
    ("D2_2Stage_S3S4_Base",      0.00, 45),
    ("D3_2Stage_S3S4_Sil+0.30",  0.30, 50),
    ("D4_2Stage_S3S4_Sil+0.60",  0.60, 55),
    ("D5_2Stage_S3S4_Sil+0.90",  0.90, 60),
]:
    r = eval_1000(trained_logits_1000, sil_bias=sb, spk_min_frames=mf)
    r["name"] = name
    results.append(r)

print("DIARIZEN_1000_JSON:" + json.dumps(results), flush=True)
'''


def main():
    bundle_path = ensure_1000_test_and_train_bank()
    diar_py = "/home/lamkd2/sommerlier/diarizen_env/bin/python"
    env = os.environ.copy()
    visible = [x.strip() for x in env.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if x.strip()]
    gpu1 = visible[1] if len(visible) >= 2 else (visible[0] if visible else "1")
    env["CUDA_VISIBLE_DEVICES"] = gpu1
    env["OMP_NUM_THREADS"] = "16"

    proc = subprocess.run([diar_py, "-c", DIARIZEN_2STAGE_1000_WORKER, bundle_path], env=env, text=True, capture_output=True)
    for line in proc.stdout.splitlines():
        if not line.startswith("DIARIZEN_1000_JSON:"):
            print(line)
        else:
            results = json.loads(line[len("DIARIZEN_1000_JSON:"):])
            print("\n" + "=" * 120)
            print("KẾT QUẢ DIARIZEN V2 (2-STAGE CURRICULUM: 40h SYNTH -> REAL CALIB) TRÊN BỘ TEST N = 1,000 CLIPS!")
            print("=" * 120)
            print(f"{'Cấu hình DiariZen (N=1000 Test)':32s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'Real100_DER0':>12s}")
            print("-" * 120)
            for r in results:
                print(f"{r['name']:32s} | {r['DER@0ms']:7.2f}% | {r['DER@.25s']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['Real100_DER0']:11.2f}%")
            print("=" * 120)
    if proc.returncode != 0:
        print("[ERR]", proc.stderr[-600:])


if __name__ == "__main__":
    main()
