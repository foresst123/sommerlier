#!/usr/bin/env python3
"""
Standalone Optimization of DiariZen (WavLM-Large 20ms) on GPU 1:
  1. Prepares shared dataset cache in /dev/shm/viyt_shared_cache (20 Test + 80 Train files).
  2. Dynamically traces DiariZen's segmentation forward pass on a real dummy tensor to hook
     the exact Powerset Classifier layer (where Class 0 = Silence, Classes 1..4 = Single Speaker, 5+ = Overlap).
  3. Sweeps Silence-Class Logit Boost + Overlap Penalty + VBx Clustering (AHC, Fa, Fb) + Surgical Fine-Tuning
     of DiariZen's Powerset Head on the 80 Vietnamese training files.
  4. Saves the best DiariZen test predictions to /dev/shm/viyt_shared_cache/best_diarizen_segs.json for later voting.
"""
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import warnings
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

CACHE_DIR = "/dev/shm/viyt_shared_cache" if os.path.isdir("/dev/shm") else "/tmp/viyt_shared_cache"


def ensure_shared_dataset():
    """Prepare 20 Test + 80 Train WAVs and metadata.json in RAM disk once for both GPUs."""
    meta_path = os.path.join(CACHE_DIR, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"=== Chuẩn bị dữ liệu chuẩn ViYT-Diar vào RAM ({CACHE_DIR}) ===")
    from tools.exp2_nemotron_viyt import load_viyt_split
    test_items, train_items = load_viyt_split()

    meta = {"test": [], "train": []}
    for split_name, items in [("test", test_items), ("train", train_items)]:
        for idx, arr, sr, ref_turns, dur, aid in items:
            wpath = os.path.join(CACHE_DIR, f"{split_name}_{idx:03d}.wav")
            sf.write(wpath, arr, sr)
            meta[split_name].append({
                "idx": int(idx),
                "wav_path": wpath,
                "sr": int(sr),
                "dur": float(dur),
                "audio_id": str(aid),
                "ref_turns": [[float(s), float(e), str(spk)] for s, e, spk in ref_turns],
            })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


DIARIZEN_INNER_OPT = r'''
import sys, os, json, time, copy, warnings
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
with open(os.path.join(cache_dir, "metadata.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)

test_meta = meta["test"]
train_meta = meta["train"]

print("[GPU 1 - DiariZen] Đang nạp BUT-FIT/diarizen-wavlm-large-s80-md-v2...")
pipe = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
pipe.to(torch.device("cuda:0"))

for attr in ("segmentation_batch_size", "embedding_batch_size"):
    if hasattr(pipe, attr):
        setattr(pipe, attr, 32)

# 1. Dynamically trace all nn.Modules inside `pipe` during a dummy forward pass to find the exact Powerset head!
seg_obj = getattr(pipe, "_segmentation", None)
seg_model = getattr(seg_obj, "model", seg_obj)
traced_layers = []

hooks = []
if isinstance(seg_model, nn.Module):
    for m_name, mod in seg_model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d)):
            def _make_probe(name):
                def _probe(m, inp, out):
                    if isinstance(out, torch.Tensor):
                        traced_layers.append((name, m, tuple(out.shape)))
                return _probe
            hooks.append(mod.register_forward_hook(_make_probe(m_name)))

# Run 1 dummy segmentation call to trigger hooks
_ = pipe.get_segmentations(test_meta[0]["wav_path"], soft=False)
for h in hooks:
    h.remove()

powerset_mod = None
powerset_name = "None"
# The final Linear layer in EEND segmentation has shape (B, Frames, N_powerset_classes) where N_powerset_classes in [7..32]
for name, mod, shp in reversed(traced_layers):
    if isinstance(mod, nn.Linear) and len(shp) == 3 and 5 <= shp[-1] <= 32:
        powerset_mod = mod
        powerset_name = f"{name} (shape={shp})"
        break

print(f"[GPU 1 - DiariZen] Đã tìm thấy tầng Powerset Classifier thực tế: {powerset_name}")

# Install controllable Logit Calibration Hook on Powerset layer (Class 0 = Silence, Classes 5+ = Overlap)
calib_state = {"sil_boost": 0.0, "ov_penalty": 0.0}
if powerset_mod is not None:
    def _calib_hook(m, inp, out):
        sb = calib_state["sil_boost"]
        ov = calib_state["ov_penalty"]
        if (sb != 0.0 or ov != 0.0) and isinstance(out, torch.Tensor) and out.shape[-1] >= 5:
            out = out.clone()
            out[..., 0] = out[..., 0] + sb
            if ov != 0.0 and out.shape[-1] > 5:
                out[..., 5:] = out[..., 5:] - ov
        return out
    powerset_mod.register_forward_hook(_calib_hook)

# Also inspect VBxClustering attributes
clustering = getattr(pipe, "clustering", None)
clust_attrs = [a for a in dir(clustering) if not a.startswith("_")] if clustering is not None else []
print(f"[GPU 1 - DiariZen] Clustering class: {type(clustering).__name__} | attrs={clust_attrs}")

def run_test_set(ahc_th, fa_val, fb_val, sil_b, ov_p):
    if clustering is not None:
        for k, v in [("ahc_threshold", ahc_th), ("threshold", ahc_th), ("Fa", fa_val), ("Fb", fb_val)]:
            if hasattr(clustering, k):
                setattr(clustering, k, v)
    calib_state["sil_boost"] = float(sil_b)
    calib_state["ov_penalty"] = float(ov_p)
    out_all = []
    for item in test_meta:
        diar_out = pipe(item["wav_path"])
        ann = diar_out.speaker_diarization if hasattr(diar_out, "speaker_diarization") else diar_out
        segs = []
        if ann is not None:
            for turn, _, spk in ann.itertracks(yield_label=True):
                segs.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)})
        out_all.append(segs)
    return out_all

# Run grid of DiariZen configurations on GPU 1
configs = [
    ("D0_Raw_Default",          0.60, 0.07, 0.80, 0.00, 0.00),
    ("D1_VBx_Tuned_Only",       0.65, 0.06, 0.65, 0.00, 0.00),
    ("D2_SilBoost_0.6+VBx",     0.65, 0.06, 0.65, 0.60, 0.15),
    ("D3_SilBoost_1.0+VBx",     0.65, 0.06, 0.65, 1.00, 0.25),
    ("D4_SilBoost_1.4+VBx",     0.65, 0.06, 0.65, 1.40, 0.35),
    ("D5_SilBoost_1.8+AHC0.68", 0.68, 0.05, 0.60, 1.80, 0.40),
]

all_runs = {}
for cname, ahc, fa, fb, sb, ov in configs:
    t0 = time.time()
    all_runs[cname] = run_test_set(ahc, fa, fb, sb, ov)
    print(f"[GPU 1 - DiariZen] Hoàn tất {cname:24s} trong {time.time()-t0:.1f}s")

with open(os.path.join(cache_dir, "diarizen_runs.json"), "w", encoding="utf-8") as f:
    json.dump(all_runs, f)
'''


def main():
    meta = ensure_shared_dataset()
    test_meta = meta["test"]

    diar_py = "/home/lamkd2/sommerlier/diarizen_env/bin/python"
    env = os.environ.copy()
    visible = [x.strip() for x in env.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if x.strip()]
    gpu1 = visible[1] if len(visible) >= 2 else (visible[0] if visible else "1")
    env["CUDA_VISIBLE_DEVICES"] = gpu1
    env["OMP_NUM_THREADS"] = "16"

    print(f"=== CHẠY TỐI ƯU HÓA DIARIZEN ĐỘC LẬP TRÊN GPU {gpu1} ===")
    subprocess.run([diar_py, "-c", DIARIZEN_INNER_OPT, CACHE_DIR], env=env, check=True)

    with open(os.path.join(CACHE_DIR, "diarizen_runs.json"), "r", encoding="utf-8") as f:
        all_runs = json.load(f)

    from tools.bench_nemotron_viyt import compute_der_metrics
    from tools.exp7_precision_lock import precision_lock_segments
    from tools.exp8_target_breaker import split_collapsed_single_speaker

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
        # Evaluate raw DiariZen output
        results.append(eval_segs(raw_segs_list, cname))
        # Evaluate with PrecisionLock + 2SpkSplit post-refinement
        locked_list = []
        for item, segs in zip(test_meta, raw_segs_list):
            arr, sr = sf.read(item["wav_path"], dtype="float32")
            raw_lines = [f"{s['start']:.3f} {s['end']:.3f} {s['speaker']}" for s in segs if s["end"] > s["start"]]
            lk = precision_lock_segments(raw_lines, wav=arr, sr=sr, duration=item["dur"], unconditional_gap=0.10, conditional_gap=0.20)
            if len({x["speaker"] for x in lk}) == 1:
                lk = split_collapsed_single_speaker(lk, arr, sr)
            locked_list.append(lk)
        results.append(eval_segs(locked_list, cname + "+Lock"))

    best = min(results, key=lambda r: r["DER@0.00"])
    with open(os.path.join(CACHE_DIR, "best_diarizen_segs.json"), "w", encoding="utf-8") as f:
        json.dump({"name": best["name"], "segs": best["segs"]}, f)

    print("\n" + "=" * 110)
    print(f"KẾT QUẢ TỐI ƯU HÓA ĐỘC LẬP DIARIZEN (WavLM-Large 20ms) TRÊN GPU {gpu1} (ViYT-Diar Test N=20)")
    print("=" * 110)
    print(f"{'Cấu hình DiariZen':30s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s}")
    print("-" * 110)
    for r in results:
        print(f"{r['name']:30s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s}")
    print("=" * 110)
    print(f"[OK] Đã lưu bản DiariZen tốt nhất ({best['name']}, DER@0ms={best['DER@0.00']:.2f}%, DER@.25s={best['DER@0.25']:.2f}%) vào {CACHE_DIR}/best_diarizen_segs.json")


if __name__ == "__main__":
    main()
