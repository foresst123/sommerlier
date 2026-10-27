#!/usr/bin/env python3
"""
Benchmark & Training on strictly TWO datasets:
  - Test Set: 100 files total (50 files ViYT-Diar [000..049] + 50 files OneVoice [000..049])
  - Train Set: Everything else! (50 files ViYT-Diar [050..099] + remaining hundreds of OneVoice conversations)
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



def parse_turns_flexible(row):
    import json
    for key in ("turns_json", "turns", "transcript_json", "dialogue", "segments"):
        val = row.get(key)
        if not val:
            continue
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                continue
        if isinstance(val, list):
            turns = []
            for item in val:
                if isinstance(item, dict):
                    st = item.get("start") or item.get("start_time") or item.get("begin")
                    en = item.get("end") or item.get("end_time")
                    spk = item.get("speaker") or item.get("role") or item.get("speaker_id") or "spk"
                    if st is not None and en is not None and float(en) > float(st):
                        turns.append((float(st), float(en), str(spk)))
            if turns:
                return turns
    return []


def prepare_100_test_and_train_splits():
    work_dir = "/dev/shm/viyt_onevoice_100test"
    os.makedirs(work_dir, exist_ok=True)
    bundle_path = os.path.join(work_dir, "dataset_splits_100test.pt")
    if os.path.isfile(bundle_path):
        print(f"[Cache Hit] Đã tìm thấy dữ liệu tại {bundle_path}")
        return torch.load(bundle_path, map_location="cpu", weights_only=False)

    from datasets import load_dataset, Audio
    import librosa

    print("=== 1. Đang tải ViYT-Diar (50 file Test + 50 file Train) ===")
    ds_viyt = load_dataset("tuanduy1612/ViYT-Diar", split="test")
    ds_viyt = ds_viyt.cast_column("audio", Audio(decode=False))

    test_viyt_50 = []
    train_viyt_50_windows = []
    win_s, sr = 24.0, 16000
    win_samples = int(win_s * sr)

    for idx, row in enumerate(ds_viyt):
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
        if idx < 50:
            # 50 files TEST của ViYT-Diar
            p = os.path.join(work_dir, f"test_viyt_{idx:03d}.wav")
            sf.write(p, arr, fs)
            test_viyt_50.append({"wav_path": p, "dur": dur, "ref_turns": ref_turns, "source": "ViYT-Diar"})
        else:
            # 50 files TRAIN của ViYT-Diar
            starts = [0.0] if dur <= win_s else [0.0, min(dur - win_s, dur * 0.35), min(dur - win_s, dur * 0.70)]
            for st_s in sorted(set(starts)):
                i0 = int(st_s * sr)
                chunk = arr[i0 : i0 + win_samples]
                if len(chunk) < int(4.0 * sr):
                    continue
                if len(chunk) < win_samples:
                    chunk = np.pad(chunk, (0, win_samples - len(chunk)))
                shifted = [(s - st_s, e - st_s, spk) for s, e, spk in ref_turns if e > st_s and s < st_s + win_s]
                train_viyt_50_windows.append((chunk, shifted, len(chunk)))

    print("=== 2. Đang tải OneVoice (50 cuộc gọi Test + Toàn bộ còn lại vào Train) ===")
    import urllib.request
    ds_ov = load_dataset("json", data_files="hf://datasets/nhatminh/OneVoice/metadata.jsonl", split="train", streaming=True)

    test_onevoice_50 = []
    train_onevoice_windows = []

    count = 0
    max_train_ov = 350  # Lấy hàng trăm cuộc hội thoại còn lại vào train
    for row in ds_ov:
        audio_rel_path = row.get("file_name") or row.get("audio")
        if not isinstance(audio_rel_path, str):
            continue
        url = f"https://huggingface.co/datasets/nhatminh/OneVoice/resolve/main/{audio_rel_path}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                arr, fs = sf.read(io.BytesIO(response.read()), dtype="float32")
        except Exception as e:
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if fs != sr:
            arr = librosa.resample(arr.astype(np.float32), orig_sr=fs, target_sr=sr)
        dur = len(arr) / float(sr)
        if dur < 8.0:
            continue

        parsed_turns = parse_turns_flexible(row)
        if not parsed_turns:
            parsed_turns = [(0.5, dur - 0.5, "SPEAKER_00")]

        if count < 50:
            # 50 cuộc gọi TEST của OneVoice
            p = os.path.join(work_dir, f"test_onevoice_{count:03d}.wav")
            sf.write(p, arr, sr)
            test_onevoice_50.append({"wav_path": p, "dur": dur, "ref_turns": parsed_turns, "source": "OneVoice"})
        else:
            # Toàn bộ cuộc gọi còn lại ĐỔ VÀO TRAIN
            if count - 50 >= max_train_ov:
                break
            starts = [0.0] if dur <= win_s else [0.0, min(dur - win_s, dur * 0.45)]
            for st_s in starts:
                i0 = int(st_s * sr)
                chunk = arr[i0 : i0 + win_samples]
                if len(chunk) < win_samples:
                    chunk = np.pad(chunk, (0, win_samples - len(chunk)))
                shifted = [(s - st_s, e - st_s, spk) for s, e, spk in parsed_turns if e > st_s and s < st_s + win_s]
                train_onevoice_windows.append((chunk, shifted, len(chunk)))
        count += 1
        if count % 50 == 0:
            print(f"  -> Đã nạp {count} cuộc gọi OneVoice...")

    bundle = {
        "test_100": test_viyt_50 + test_onevoice_50,
        "train_all_windows": train_viyt_50_windows + train_onevoice_windows,
        "n_viyt_train": len(train_viyt_50_windows),
        "n_ov_train": len(train_onevoice_windows),
    }
    torch.save(bundle, bundle_path)
    print(f"[OK] Đã chuẩn bị xong: 100 files Test (50 ViYT + 50 OneVoice) & {len(bundle['train_all_windows'])} windows Train!")
    return bundle


def train_nemotron_layer4(model, all_windows, epochs=6, batch_size=16, l2sp_alpha=0.02, neg_weight=1.28):
    sm = model.sortformer_modules
    for p in model.parameters():
        p.requires_grad_(False)

    param_groups = [
        {"params": [p for p in sm.subpixel_upsample.parameters()], "lr": 1.2e-3, "init_lr": 1.2e-3},
        {"params": [p for p in sm.first_hidden_to_hidden.parameters()], "lr": 6e-4, "init_lr": 6e-4},
        {"params": [p for p in sm.single_hidden_to_spks.parameters()], "lr": 4e-4, "init_lr": 4e-4},
        {"params": [p for p in sm.hidden_to_spks.parameters()], "lr": 4e-4, "init_lr": 4e-4},
    ]
    for g in param_groups:
        for p in g["params"]:
            p.requires_grad_(True)

    all_params = [p for g in param_groups for p in g["params"]]
    theta_0 = {p: p.detach().clone() for p in all_params}

    captured = {}
    orig_upsample = sm.upsample_hidden

    def wrapped_upsample(hidden_states):
        captured["hidden"] = hidden_states.detach()
        return orig_upsample(hidden_states)

    sm.upsample_hidden = wrapped_upsample
    encoded_samples = []

    print(f"=== 3. Trích xuất đặc trưng mạng từ {len(all_windows)} windows Train ===")
    with torch.no_grad():
        for chunk, turns, c_len in all_windows:
            sig = torch.from_numpy(chunk).unsqueeze(0).to("cuda")
            sig_len = torch.tensor([c_len], dtype=torch.long, device="cuda")
            captured.clear()
            _ = model.forward(audio_signal=sig, audio_signal_length=sig_len)
            if "hidden" not in captured:
                continue
            hid = captured["hidden"].cpu()
            init_p = sm.forward_speaker_sigmoids(orig_upsample(hid.to("cuda"))).squeeze(0).cpu().numpy()
            n_10ms = init_p.shape[0]
            raw_t = build_frame_targets(turns, n_10ms, step=0.01).numpy()
            aligned_t = align_target_hungarian(init_p, raw_t)
            encoded_samples.append((hid.squeeze(0), aligned_t))

    sm.upsample_hidden = orig_upsample
    print(f"Đã mã hóa {len(encoded_samples)} windows. Huấn luyện Tầng 4 (batch={batch_size}, L2-SP={l2sp_alpha})...")

    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    rng = random.Random(42)

    for ep in range(1, epochs + 1):
        order = list(range(len(encoded_samples)))
        rng.shuffle(order)
        ep_loss = 0.0
        n_b = 0
        for b0 in range(0, len(order), batch_size):
            b_ids = order[b0 : b0 + batch_size]
            opt.zero_grad()
            b_bce = 0.0
            for idx_w in b_ids:
                hid, target = encoded_samples[idx_w]
                hid = hid.unsqueeze(0).to("cuda")
                target = target.unsqueeze(0).to("cuda")
                up = sm.upsample_hidden(hid)
                probs = sm.forward_speaker_sigmoids(up)
                n_t = min(probs.shape[1], target.shape[1])
                p_s = torch.clamp(probs[:, :n_t, :], 1e-6, 1.0 - 1e-6)
                t_s = target[:, :n_t, :]
                w = torch.where(t_s > 0.5, 1.0, neg_weight)
                bce = F.binary_cross_entropy(p_s, t_s, weight=w)
                b_bce = b_bce + bce / len(b_ids)

            l2sp = sum(torch.sum((p - theta_0[p]) ** 2) for p in all_params)
            loss = b_bce + l2sp_alpha * l2sp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            ep_loss += float(b_bce.item())
            n_b += 1
        sched.step()
        print(f"  Epoch {ep}/{epochs} - BCE Loss: {ep_loss / max(1, n_b):.4f}")

    model.eval()
    os.makedirs("models", exist_ok=True)
    save_path = "models/nemotron_vi_full_100test.pt"
    torch.save({
        "subpixel_upsample": sm.subpixel_upsample.state_dict() if sm.subpixel_upsample is not None else None,
        "first_hidden_to_hidden": sm.first_hidden_to_hidden.state_dict(),
        "single_hidden_to_spks": sm.single_hidden_to_spks.state_dict(),
        "hidden_to_spks": sm.hidden_to_spks.state_dict(),
    }, save_path)
    print(f"[OK] Đã lưu trọng số tại: {save_path}")


def eval_on_test_set(model, test_100, work_dir, name, pp_dict, u_gap, c_gap):
    pp_yaml = write_postprocessing_yaml(pp_dict, os.path.join(work_dir, name))
    paths = [item["wav_path"] for item in test_100]

    t0 = time.time()
    preds = model.diarize(audio=paths, batch_size=1, postprocessing_yaml=pp_yaml)
    elapsed = time.time() - t0

    def calc_stats(subset_indices):
        tot_ref = w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
        spk_exact = 0
        spk_abs_err = []
        for i in subset_indices:
            item = test_100[i]
            dur = item["dur"]
            ref_turns = item["ref_turns"]
            arr, sr = sf.read(item["wav_path"], dtype="float32")
            raw_hyp = preds[i]
            segs = precision_lock_segments(raw_hyp, wav=arr, sr=sr, duration=dur, unconditional_gap=u_gap, conditional_gap=c_gap)
            if len({s["speaker"] for s in segs}) == 1:
                segs = split_collapsed_single_speaker(segs, arr, sr)
            hyp_turns = [(s["start"], s["end"], s["speaker"]) for s in segs]

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

        n = len(subset_indices)
        return {
            "DER@0": w_der0 / max(1e-6, tot_ref),
            "DER@25": w_der25 / max(1e-6, tot_ref),
            "Miss": w_miss / max(1e-6, tot_ref),
            "FA": w_fa / max(1e-6, tot_ref),
            "Conf": w_conf / max(1e-6, tot_ref),
            "SpkExact": spk_exact / max(1, n) * 100.0,
            "SpkMAE": float(np.mean(spk_abs_err)) if spk_abs_err else 0.0,
        }

    all_idx = list(range(len(test_100)))
    viyt_idx = [i for i, x in enumerate(test_100) if x["source"] == "ViYT-Diar"]
    ov_idx   = [i for i, x in enumerate(test_100) if x["source"] == "OneVoice"]

    return {
        "name": name,
        "all": calc_stats(all_idx),
        "viyt": calc_stats(viyt_idx),
        "ov": calc_stats(ov_idx),
        "time": elapsed,
    }


def main():
    bundle = prepare_100_test_and_train_splits()
    test_100 = bundle["test_100"]
    train_windows = bundle["train_all_windows"]

    work_dir = tempfile.mkdtemp(prefix="exp_100test_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    from nemo.collections.asr.models import SortformerEncLabelModel
    print("=== Nạp Nemotron-3 + Cấu hình Deep AOSC ===")
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()
    sm = model.sortformer_modules
    for k, v in {"spkcache_len": 384, "fifo_len": 60, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 200}.items():
        setattr(sm, k, v)
    model._check_streaming_parameters()
    install_deep_aosc_hooks(sm)

    # Huấn luyện Tầng 4 trên toàn bộ dữ liệu Train còn lại
    train_nemotron_layer4(model, train_windows, epochs=6, batch_size=16, l2sp_alpha=0.02, neg_weight=1.28)

    presets = [
        ("N_Balanced_100Test", {"onset": 0.52, "offset": 0.42, "pad_onset": 0.01, "pad_offset": 0.03, "min_duration_on": 0.16, "min_duration_off": 0.10}, 0.22, 0.38),
        ("N_Crisp_100Test",    {"onset": 0.58, "offset": 0.48, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.14, "min_duration_off": 0.08}, 0.14, 0.24),
    ]

    print("\n" + "=" * 126)
    print("KẾT QUẢ ĐÁNH GIÁ CHUẨN XÁC TRÊN BỘ TEST 100 FILES (50 ViYT-Diar YouTube + 50 OneVoice Conversations)")
    print("=" * 126)
    for name, pp_dict, u_gap, c_gap in presets:
        res = eval_on_test_set(model, test_100, work_dir, name, pp_dict, u_gap, c_gap)
        a, v, o = res["all"], res["viyt"], res["ov"]
        print(f"\n>>> CẤU HÌNH: {res['name']} (Tổng thời gian chạy 100 files: {res['time']:.1f}s) <<<")
        print(f"{'Tập Kiểm Thử':26s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s}")
        print("-" * 105)
        print(f"{'TỔNG 100 FILES TEST':26s} | {a['DER@0']:7.2f}% | {a['DER@25']:7.2f}% | {a['Miss']:6.2f}% | {a['FA']:6.2f}% | {a['Conf']:6.2f}% | {a['SpkExact']:7.1f}% | {a['SpkMAE']:6.2f}")
        print(f"{'50 Files ViYT-Diar (Real)':26s} | {v['DER@0']:7.2f}% | {v['DER@25']:7.2f}% | {v['Miss']:6.2f}% | {v['FA']:6.2f}% | {v['Conf']:6.2f}% | {v['SpkExact']:7.1f}% | {v['SpkMAE']:6.2f}")
        print(f"{'50 Files OneVoice (Synth)':26s} | {o['DER@0']:7.2f}% | {o['DER@25']:7.2f}% | {o['Miss']:6.2f}% | {o['FA']:6.2f}% | {o['Conf']:6.2f}% | {o['SpkExact']:7.1f}% | {o['SpkMAE']:6.2f}")
        print("-" * 105)


if __name__ == "__main__":
    main()
