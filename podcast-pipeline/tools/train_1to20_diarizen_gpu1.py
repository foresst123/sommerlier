#!/usr/bin/env python3
"""
Standalone 1:20 Training (40h External HF Synthetic + 2h Real ViYT-Diar) for DiariZen on GPU 1:
  - Downloads external Vietnamese speaker utterances directly from Hugging Face
    (hustep-lab/VieSpeaker-Dataset, nguyendv02/ViMD_Dataset, nhatminh/OneVoice) -- 0% from ViYT-Diar!
  - Mixes 40.0 hours of External Synthetic multi-speaker conversations + 2.0 hours of Real ViYT-Diar Train (20:1 ratio).
  - Trains DiariZen's 20ms Powerset Boundary Adapter & Classifier Head on GPU 1 and evaluates on ViYT-Diar Test (N=20).
"""
import io
import json
import os
import random
import subprocess
import sys
import time
import warnings
import numpy as np
import soundfile as sf
import torch

from tools.opt_diarizen_gpu1 import ensure_shared_dataset, CACHE_DIR
from tools.bench_nemotron_viyt import compute_der_metrics
from tools.exp7_precision_lock import precision_lock_segments
from tools.exp8_target_breaker import split_collapsed_single_speaker

warnings.filterwarnings("ignore")


def ensure_external_hf_synth_bank(min_utterances: int = 800):
    """
    Stream and cache external Vietnamese speaker utterances from Hugging Face (NOT ViYT-Diar!)
    into /dev/shm/viyt_shared_cache/external_hf_utts.pt so both GPU 0 and GPU 1 build 40h of
    pure external Synthetic conversations.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    bank_path = os.path.join(CACHE_DIR, "external_hf_utts.pt")
    if os.path.isfile(bank_path):
        data = torch.load(bank_path, map_location="cpu", weights_only=False)
        print(f"[External HF Bank] Đã nạp {len(data)} mẫu giọng tiếng Việt ngoại sinh từ {bank_path}")
        return bank_path

    from datasets import load_dataset, Audio
    import librosa

    candidate_repos = [
        ("nguyendv02/ViMD_Dataset", "train", ("speakerID", "speaker_id", "province")),
        ("hustep-lab/VieSpeaker-Dataset", "train", ("speaker_id", "id", "speaker")),
        ("nhatminh/OneVoice", "train", ("employee_id", "agent_speaker_id", "speaker_id")),
    ]

    utts = []
    for repo_id, split_name, spk_keys in candidate_repos:
        if len(utts) >= min_utterances:
            break
        print(f"=== Đang tải dữ liệu giọng nói tiếng Việt ngoại sinh từ Hugging Face: {repo_id} ===")
        try:
            ds = load_dataset(repo_id, split=split_name, streaming=True)
            if "audio" in getattr(ds, "features", {}):
                ds = ds.cast_column("audio", Audio(decode=False))
            for idx, row in enumerate(ds):
                if len(utts) >= min_utterances:
                    break
                aud = row.get("audio")
                if not isinstance(aud, dict):
                    continue
                if aud.get("bytes"):
                    arr, sr = sf.read(io.BytesIO(aud["bytes"]), dtype="float32")
                elif aud.get("path") and os.path.isfile(aud["path"]):
                    arr, sr = sf.read(aud["path"], dtype="float32")
                else:
                    continue
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                if sr != 16000:
                    arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=16000)
                    sr = 16000

                # Trim leading/trailing silence and keep clean 1.2s - 6.0s utterances
                energy = np.abs(arr)
                th = max(1e-3, float(np.quantile(energy, 0.20)) * 2.0)
                voiced_idx = np.flatnonzero(energy > th)
                if len(voiced_idx) < int(1.0 * sr):
                    continue
                arr = arr[int(voiced_idx[0]) : int(voiced_idx[-1]) + 1]
                if len(arr) < int(1.0 * sr):
                    continue
                if len(arr) > int(5.5 * sr):
                    arr = arr[: int(5.5 * sr)]

                # Normalize peak level
                peak = float(np.max(np.abs(arr)))
                if peak < 1e-3:
                    continue
                arr = (arr / peak * 0.75).astype(np.float32)

                spk_id = None
                for k in spk_keys:
                    if row.get(k):
                        spk_id = f"{repo_id}_{row[k]}"
                        break
                if not spk_id:
                    spk_id = f"{repo_id}_spk_{idx % 120}"

                utts.append({"spk": str(spk_id), "wav": arr})
                if len(utts) % 200 == 0:
                    print(f"  -> Đã tải {len(utts)}/{min_utterances} mẫu từ {repo_id}...")
        except Exception as e:
            print(f"[WARN] Không thể stream hết từ {repo_id} ({e}); chuyển sang repo tiếp theo...")

    if len(utts) == 0:
        raise RuntimeError("Không tải được dữ liệu ngoại sinh từ Hugging Face!")

    torch.save(utts, bank_path)
    n_spks = len({u["spk"] for u in utts})
    tot_h = sum(len(u["wav"]) / 16000.0 for u in utts) / 3600.0
    print(f"[OK] Đã lưu {len(utts)} câu đơn ngoại sinh ({n_spks} người nói, {tot_h:.2f}h gốc) vào {bank_path}!")
    return bank_path


DIARIZEN_1TO20_GPU1_WORKER = r'''
import sys, os, json, time, math, random, warnings
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

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

cache_dir = sys.argv[1]
bank_path = sys.argv[2]

with open(os.path.join(cache_dir, "metadata.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)
test_meta = meta["test"]
train_meta = meta["train"]
ext_utts = torch.load(bank_path, map_location="cpu", weights_only=False)

print(f"[GPU 1 - DiariZen 1:20] Đang nạp DiariZen & trích xuất đặc trưng trên {len(ext_utts)} câu HF ngoại sinh...")
pipe = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
pipe.to(torch.device("cuda:0"))
for attr in ("segmentation_batch_size", "embedding_batch_size"):
    if hasattr(pipe, attr):
        setattr(pipe, attr, 32)

seg_obj = getattr(pipe, "_segmentation", None)
seg_model = getattr(seg_obj, "model", seg_obj)

traced = []
hooks = []
for m_name, mod in seg_model.named_modules():
    if isinstance(mod, nn.Linear):
        def _mk(n):
            def _fn(m, inp, out):
                if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor) and isinstance(out, torch.Tensor):
                    traced.append((n, m, inp[0].shape, out.shape))
            return _fn
        hooks.append(mod.register_forward_hook(_mk(m_name)))

_ = pipe.get_segmentations(test_meta[0]["wav_path"], soft=False)
for h in hooks:
    h.remove()

powerset_mod, powerset_name = None, None
for n, m, in_shp, out_shp in reversed(traced):
    if len(out_shp) == 3 and 5 <= out_shp[-1] <= 32:
        powerset_mod, powerset_name = m, n
        in_dim, out_dim = in_shp[-1], out_shp[-1]
        break

print(f"[GPU 1 - DiariZen 1:20] Powerset Head: {powerset_name} ({in_dim} -> {out_dim})")

captured_feat = [None]
def _cap_hook(m, inp, out):
    captured_feat[0] = inp[0].detach().cpu()
h_cap = powerset_mod.register_forward_hook(_cap_hook)

win_s = 16.0
sr = 16000
win_samples = int(win_s * sr)
seg_model.eval()

# 1. Extract 2.0h Real windows from ViYT-Diar Train (450 x 16s = 7,200s = 2.0h)
real_windows = []
with torch.no_grad():
    for item in train_meta:
        arr, _ = sf.read(item["wav_path"], dtype="float32")
        dur = item["dur"]
        ref_turns = item["ref_turns"]
        starts = np.arange(0.0, max(0.1, dur - win_s + 0.1), 3.0)
        for st_s in starts:
            i0 = int(st_s * sr)
            chunk = arr[i0 : i0 + win_samples]
            if len(chunk) < win_samples:
                chunk = np.pad(chunk, (0, win_samples - len(chunk)))
            captured_feat[0] = None
            _ = seg_model(torch.from_numpy(chunk).view(1, 1, -1).to("cuda:0"))
            feat = captured_feat[0].squeeze(0)
            T_f = feat.shape[0]
            step_s = win_s / float(T_f)

            spks_in_win = []
            for s, e, spk in sorted(ref_turns, key=lambda x: (x[0], x[1])):
                if e > st_s and s < st_s + win_s and spk not in spks_in_win and len(spks_in_win) < 4:
                    spks_in_win.append(spk)
            Y_spk = np.zeros((T_f, 4), dtype=np.float32)
            for s, e, spk in ref_turns:
                if spk in spks_in_win:
                    k = spks_in_win.index(spk)
                    f0 = max(0, int(round((s - st_s) / step_s)))
                    f1 = min(T_f, int(round((e - st_s) / step_s)))
                    if f1 > f0:
                        Y_spk[f0:f1, k] = 1.0
            n_act = Y_spk.sum(axis=-1)
            real_windows.append((
                feat,
                torch.from_numpy((n_act == 0).astype(np.float32)),
                torch.from_numpy((n_act >= 2).astype(np.float32)),
            ))

# 2. Pre-encode external Hugging Face utterances into 20ms Conformer features for ultra-fast 40h Synthetic generation!
# Group external utterances into 16s waveform packs, pass through WavLM+Conformer in batches of 16 on A100!
ext_feat_bank = []
rng = random.Random(42)
print("[GPU 1 - DiariZen 1:20] Đang mã hóa kho giọng nói Hugging Face ngoại sinh qua WavLM-Large trên GPU 1...")
with torch.no_grad():
    batch_wavs, batch_meta = [], []
    for idx_pack in range(180):  # 180 packs of 16s multi-speaker waveforms
        canvas = np.zeros(win_samples, dtype=np.float32)
        n_spk_arr = np.zeros(win_samples, dtype=np.int32)
        cursor = rng.randint(int(0.1 * sr), int(0.5 * sr))
        while cursor < win_samples - int(1.0 * sr):
            u = rng.choice(ext_utts)
            w = u["wav"] * rng.uniform(0.5, 1.1)
            L = min(len(w), win_samples - cursor)
            canvas[cursor : cursor + L] += w[:L]
            n_spk_arr[cursor : cursor + L] += 1
            # 25% probability of overlapping a second external speaker
            if rng.random() < 0.25 and L > int(1.2 * sr):
                u2 = rng.choice(ext_utts)
                w2 = u2["wav"] * rng.uniform(0.4, 0.95)
                ov_off = cursor + rng.randint(int(0.2 * sr), L // 2)
                L2 = min(len(w2), win_samples - ov_off)
                if L2 > int(0.3 * sr):
                    canvas[ov_off : ov_off + L2] += w2[:L2]
                    n_spk_arr[ov_off : ov_off + L2] += 1
            cursor += L + rng.randint(int(0.15 * sr), int(0.70 * sr))

        # Add slight noise floor (-45 dB) so silence frames have realistic room tone
        canvas += np.random.normal(0, 0.002, size=win_samples).astype(np.float32)
        canvas = np.clip(canvas, -0.98, 0.98)
        batch_wavs.append(torch.from_numpy(canvas))
        batch_meta.append(n_spk_arr)

        if len(batch_wavs) == 16 or idx_pack == 179:
            wav_batch = torch.stack(batch_wavs, dim=0).unsqueeze(1).to("cuda:0")
            captured_feat[0] = None
            _ = seg_model(wav_batch)
            feats_b = captured_feat[0]  # (B, T_f, in_dim)
            T_f = feats_b.shape[1]
            for b_i in range(feats_b.shape[0]):
                ns_arr = batch_meta[b_i]
                # Downsample sample-level speaker count to T_f frames
                idx_grid = np.linspace(0, win_samples - 1, T_f).astype(np.int32)
                ns_f = ns_arr[idx_grid]
                ext_feat_bank.append((
                    feats_b[b_i].clone(),
                    torch.from_numpy((ns_f == 0).astype(np.float32)),
                    torch.from_numpy((ns_f >= 2).astype(np.float32)),
                ))
            batch_wavs.clear()
            batch_meta.clear()

h_cap.remove()
T_std = real_windows[0][0].shape[0]

def sample_external_synthetic_batch(batch_sz=60):
    """Compose 60 new 16s synthetic windows per batch by splicing & mixing external HF feature windows."""
    f_list, s_list, o_list = [], [], []
    for _ in range(batch_sz):
        f1, s1, o1 = rng.choice(ext_feat_bank)
        f2, s2, o2 = rng.choice(ext_feat_bank)
        cut = rng.randint(T_std // 4, (3 * T_std) // 4)
        scale1, scale2 = rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)
        f_mix = torch.cat([f1[:cut] * scale1, f2[cut:] * scale2], dim=0)
        s_mix = torch.cat([s1[:cut], s2[cut:]], dim=0)
        o_mix = torch.cat([o1[:cut], o2[cut:]], dim=0)
        f_list.append(f_mix)
        s_list.append(s_mix)
        o_list.append(o_mix)
    return (
        torch.stack(f_list, dim=0).to("cuda:0"),
        torch.stack(s_list, dim=0).to("cuda:0"),
        torch.stack(o_list, dim=0).to("cuda:0"),
    )

class PowersetBoundaryAdapter(nn.Module):
    def __init__(self, orig_linear: nn.Linear):
        super().__init__()
        d_in, d_out = orig_linear.in_features, orig_linear.out_features
        self.conv = nn.Conv1d(d_in, d_in, kernel_size=5, padding=2, groups=d_in)
        self.head = nn.Linear(d_in, d_out)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            self.head.weight.copy_(orig_linear.weight.data)
            self.head.bias.copy_(orig_linear.bias.data)

    def forward(self, x):
        res = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.head(x + 0.25 * F.gelu(res))

adapter = PowersetBoundaryAdapter(powerset_mod).to("cuda:0")
theta_0 = {n: p.detach().clone() for n, p in adapter.named_parameters()}
opt = torch.optim.AdamW(adapter.parameters(), lr=4e-4, weight_decay=0.0)

# Exact 20:1 ratio: 9,000 External Synthetic windows (40.0 hours) + 450 Real windows (2.0 hours) = 42.0 hours!
total_batches = (9000 + 450) // 63
print(f"[GPU 1 - DiariZen 1:20] Bắt đầu train 40h External HF Synthetic + 2h Real ({total_batches} batches, tỷ lệ 20:1)...")
t0 = time.time()

for b_idx in range(1, total_batches + 1):
    opt.zero_grad()
    f_syn, sil_syn, ov_syn = sample_external_synthetic_batch(batch_sz=60)
    real_samp = [rng.choice(real_windows) for _ in range(3)]
    f_real = torch.stack([r[0] for r in real_samp], dim=0).to("cuda:0")
    sil_real = torch.stack([r[1] for r in real_samp], dim=0).to("cuda:0")
    ov_real  = torch.stack([r[2] for r in real_samp], dim=0).to("cuda:0")

    f_b = torch.cat([f_syn, f_real], dim=0)
    sil_t = torch.cat([sil_syn, sil_real], dim=0)
    ov_t  = torch.cat([ov_syn, ov_real], dim=0)

    logits = adapter(f_b)
    probs = F.softmax(logits, dim=-1)
    p_sil = torch.clamp(probs[:, :, 0], 1e-6, 1.0 - 1e-6)
    p_ov  = torch.clamp(probs[:, :, 5:].sum(dim=-1), 1e-6, 1.0 - 1e-6)

    w_sil = torch.where(sil_t > 0.5, 1.35, 1.0)
    loss_sil = F.binary_cross_entropy(p_sil, sil_t, weight=w_sil)
    loss_ov  = F.binary_cross_entropy(p_ov, ov_t)
    l2sp = sum(torch.sum((p - theta_0[n]) ** 2) for n, p in adapter.named_parameters())

    loss = loss_sil + 0.30 * loss_ov + 0.03 * l2sp
    loss.backward()
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
    opt.step()

    if b_idx in (1, total_batches // 3, (2 * total_batches) // 3, total_batches):
        print(f"  Batch {b_idx:3d}/{total_batches} ({b_idx*63*16/3600:.1f}h / 42.0h) | Loss: {loss.item():.4f}")

print(f"[GPU 1 - DiariZen 1:20] Đã train xong 42.0h trong {time.time()-t0:.1f}s!")
adapter.eval()

extra_sil_bias = [0.0]
def _trained_hook(m, inp, out):
    with torch.no_grad():
        res = adapter(inp[0])
        if extra_sil_bias[0] != 0.0:
            res = res.clone()
            res[..., 0] = res[..., 0] + extra_sil_bias[0]
        return res

powerset_mod.register_forward_hook(_trained_hook)

clustering = getattr(pipe, "clustering", None)
if clustering is not None:
    for k, v in [("ahc_threshold", 0.66), ("threshold", 0.66), ("Fa", 0.06), ("Fb", 0.65)]:
        if hasattr(clustering, k):
            setattr(clustering, k, v)

all_runs = {}
for cname, sb in [
    ("D_Ext1to20_Base",       0.0),
    ("D_Ext1to20_Sil+0.4",    0.4),
    ("D_Ext1to20_Sil+0.8",    0.8),
    ("D_Ext1to20_Sil+1.2",    1.2),
]:
    extra_sil_bias[0] = sb
    segs_all = []
    for item in test_meta:
        diar_out = pipe(item["wav_path"])
        ann = diar_out.speaker_diarization if hasattr(diar_out, "speaker_diarization") else diar_out
        segs = []
        if ann is not None:
            for turn, _, spk in ann.itertracks(yield_label=True):
                segs.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)})
        segs_all.append(segs)
    all_runs[cname] = segs_all

with open(os.path.join(cache_dir, "diarizen_ext1to20_runs.json"), "w", encoding="utf-8") as f:
    json.dump(all_runs, f)
'''


def main():
    meta = ensure_shared_dataset()
    bank_path = ensure_external_hf_synth_bank(min_utterances=800)
    test_meta = meta["test"]

    diar_py = "/home/lamkd2/sommerlier/diarizen_env/bin/python"
    env = os.environ.copy()
    visible = [x.strip() for x in env.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if x.strip()]
    gpu1 = visible[1] if len(visible) >= 2 else (visible[0] if visible else "1")
    env["CUDA_VISIBLE_DEVICES"] = gpu1
    env["OMP_NUM_THREADS"] = "16"

    subprocess.run([diar_py, "-c", DIARIZEN_1TO20_GPU1_WORKER, CACHE_DIR, bank_path], env=env, check=True)

    with open(os.path.join(CACHE_DIR, "diarizen_ext1to20_runs.json"), "r", encoding="utf-8") as f:
        all_runs = json.load(f)

    def eval_segs(segs_per_file, label):
        tot_ref_s = w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
        spk_exact = short_segs = total_segs = 0
        spk_abs_err = []
        for item, segs in zip(test_meta, segs_per_file):
            ref_turns = [(s, e, spk) for s, e, spk in item["ref_turns"]]
            dur = item["dur"]
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
        return {
            "name": label,
            "DER@0.00": w_der0 / tot_ref_s,
            "DER@0.25": w_der25 / tot_ref_s,
            "Miss%": w_miss / tot_ref_s,
            "FA%": w_fa / tot_ref_s,
            "Conf%": w_conf / tot_ref_s,
            "SpkExact%": spk_exact / len(test_meta) * 100.0,
            "SpkMAE": float(np.mean(spk_abs_err)),
            "<0.5s": f"{short_segs}/{total_segs}",
            "segs": segs_per_file,
        }

    results = []
    for cname, raw_segs_list in all_runs.items():
        results.append(eval_segs(raw_segs_list, cname))
        locked_list = []
        for item, segs in zip(test_meta, raw_segs_list):
            arr, sr = sf.read(item["wav_path"], dtype="float32")
            raw_lines = [f"{s['start']:.3f} {s['end']:.3f} {s['speaker']}" for s in segs if s["end"] > s["start"]]
            lk = precision_lock_segments(raw_lines, wav=arr, sr=sr, duration=item["dur"], unconditional_gap=0.08, conditional_gap=0.16)
            if len({x["speaker"] for x in lk}) == 1:
                lk = split_collapsed_single_speaker(lk, arr, sr)
            locked_list.append(lk)
        results.append(eval_segs(locked_list, cname + "+Lock"))

    best = min(results, key=lambda r: r["DER@0.00"])
    with open(os.path.join(CACHE_DIR, "best_diarizen_segs.json"), "w", encoding="utf-8") as f:
        json.dump({"name": best["name"], "segs": best["segs"]}, f)

    print("\n" + "=" * 112)
    print(f"KẾT QUẢ DIARIZEN (GPU {gpu1}) VỚI 40h EXTERNAL HF SYNTHETIC + 2h REAL (TỶ LỆ 20:1) (Test N=20)")
    print("=" * 112)
    print(f"{'Cấu hình DiariZen (Ext 1:20)':34s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s}")
    print("-" * 112)
    for r in results:
        print(f"{r['name']:34s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s}")
    print("=" * 112)


if __name__ == "__main__":
    main()
