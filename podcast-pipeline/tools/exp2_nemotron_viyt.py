#!/usr/bin/env python3
"""
Experiment 2 on ViYT-Diar (Test = files 0..19, Train = files 20..99):
  1. Baseline Repo
  2. Zero-Train: Tight-Boundary + Deep AOSC (No AGC, calibrated onset/offset for FA% reduction)
  3. Surgical Fine-Tuning (Tầng 4: subpixel_upsample + 156KB Speaker Head trained on files 20..99) + Deep AOSC
"""
import copy
import io
import math
import os
import tempfile
import time
import warnings
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Audio

from tools.bench_nemotron_viyt import (
    compute_der_metrics,
    install_deep_aosc_hooks,
    build_preprimed_audio,
)
from utils.nemotron_segments import (
    to_pipeline_segments,
    write_postprocessing_yaml,
    POSTPROCESSING_BASE,
)

warnings.filterwarnings("ignore")


def load_viyt_split():
    ds = load_dataset("tuanduy1612/ViYT-Diar", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    items = []
    for idx in range(len(ds)):
        row = ds[idx]
        aud = row["audio"]
        arr, sr = sf.read(io.BytesIO(aud["bytes"]) if aud.get("bytes") else aud["path"], dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        arr = np.asarray(arr, dtype=np.float32)
        sr = int(sr)
        dur = len(arr) / sr
        ref_turns = [
            (float(s), float(e), str(spk))
            for s, e, spk in zip(row["timestamps_start"], row["timestamps_end"], row["speaker"])
            if float(e) > float(s)
        ]
        items.append((idx, arr, sr, ref_turns, dur, row["audio_id"]))
    return items[:20], items[20:]  # 20 test files (same as before), 80 train files


def build_frame_targets(ref_turns, n_frames, step=0.01, max_spks=8):
    """Build (n_frames, 8) binary target matrix in arrival order of speakers."""
    spk_order = []
    for s, e, spk in sorted(ref_turns, key=lambda x: (x[0], x[1])):
        if spk not in spk_order and len(spk_order) < max_spks:
            spk_order.append(spk)
    Y = np.zeros((n_frames, max_spks), dtype=np.float32)
    for s, e, spk in ref_turns:
        if spk not in spk_order:
            continue
        k = spk_order.index(spk)
        i0 = max(0, int(round(s / step)))
        i1 = min(n_frames, int(round(e / step)))
        if i1 > i0:
            Y[i0:i1, k] = 1.0
    return torch.from_numpy(Y)


def surgical_finetune_head(model, train_items, epochs=6, lr=1.5e-3, win_s=24.0):
    """
    Freeze 99M encoder/transformer parameters; only unfreeze Tầng 4:
      - subpixel_upsample (Conv1d 80ms -> 10ms)
      - first_hidden_to_hidden (Linear 192 -> 192)
      - single_hidden_to_spks (Linear 192 -> 8)
      - hidden_to_spks (Linear 384 -> 8)
    """
    sm = model.sortformer_modules
    for p in model.parameters():
        p.requires_grad_(False)

    trainable_modules = [
        sm.subpixel_upsample,
        sm.first_hidden_to_hidden,
        sm.single_hidden_to_spks,
        sm.hidden_to_spks,
    ]
    for m in trainable_modules:
        if m is not None:
            for p in m.parameters():
                p.requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"[Surgical Fine-Tune] Số tham số mở khóa (Tầng 4 Head + Upsampler): {n_params:,} params ({n_params*4/1024:.1f} KB)")

    # Extract frozen pre-upsample hidden states on 24s windows from the 80 training files
    print("[Surgical Fine-Tune] Đang trích xuất đặc trưng Transformer (frozen) trên 80 file train...")
    cached_batches = []
    captured = {}

    def hook_pre_upsample(module, args):
        captured["hidden"] = args[0].detach()

    h = sm.upsample_hidden.__func__ if hasattr(sm.upsample_hidden, "__func__") else None
    orig_upsample = sm.upsample_hidden

    def wrapped_upsample(hidden_states):
        captured["hidden"] = hidden_states.detach()
        return orig_upsample(hidden_states)

    sm.upsample_hidden = wrapped_upsample

    with torch.no_grad():
        for _, arr, sr, ref_turns, dur, _ in train_items:
            # Take up to two 24s windows per training file
            win_samples = int(win_s * sr)
            starts = [0.0] if dur <= win_s else [0.0, min(dur - win_s, dur * 0.4)]
            for st_s in starts:
                i0 = int(st_s * sr)
                chunk = arr[i0:i0 + win_samples]
                if len(chunk) < int(4.0 * sr):
                    continue
                sig = torch.from_numpy(chunk).unsqueeze(0).to("cuda")
                sig_len = torch.tensor([len(chunk)], dtype=torch.long, device="cuda")
                captured.clear()
                _ = model.forward(audio_signal=sig, audio_signal_length=sig_len)
                if "hidden" not in captured:
                    continue
                hid = captured["hidden"].cpu()  # (1, T_80ms, 192)
                n_10ms = hid.shape[1] * sm.upsample_factor
                shifted_turns = [( s - st_s, e - st_s, spk) for s, e, spk in ref_turns if e > st_s and s < st_s + win_s]
                target = build_frame_targets(shifted_turns, n_10ms, step=0.01).unsqueeze(0)
                cached_batches.append((hid, target))

    sm.upsample_hidden = orig_upsample
    print(f"[Surgical Fine-Tune] Đã tạo {len(cached_batches)} mẫu huấn luyện từ 80 file train. Bắt đầu tối ưu Tầng 4...")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    # Asymmetric BCE weight: penalize False Alarm slightly higher since FA% is 7x Miss%
    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        for hid, target in cached_batches:
            hid = hid.to("cuda")
            target = target.to("cuda")
            opt.zero_grad()
            up = sm.upsample_hidden(hid)
            probs = sm.forward_speaker_sigmoids(up)
            # Align active speaker permutation with target
            n_t = min(probs.shape[1], target.shape[1])
            p_slice = torch.clamp(probs[:, :n_t, :], 1e-6, 1.0 - 1e-6)
            t_slice = target[:, :n_t, :]
            # Weight negative frames (silence) by 1.35 to suppress False Alarm at boundaries
            weight = torch.where(t_slice > 0.5, 1.0, 1.35)
            loss = F.binary_cross_entropy(p_slice, t_slice, weight=weight)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
        print(f"  Epoch {ep}/{epochs} - BCE Loss: {tot_loss / max(1, len(cached_batches)):.4f}")

    model.eval()


def eval_config(model, test_wavs, name, stream_cfg, pp_cfg, use_deep_aosc, use_preprime, merge_gap, min_dur, work_dir, scout_cache):
    orig_sm_state = copy.deepcopy(model.sortformer_modules.__dict__)
    try:
        for k, v in stream_cfg.items():
            setattr(model.sortformer_modules, k, v)
        if stream_cfg:
            model._check_streaming_parameters()
        if use_deep_aosc:
            install_deep_aosc_hooks(model.sortformer_modules)

        extra = {}
        if pp_cfg is not None:
            extra["postprocessing_yaml"] = write_postprocessing_yaml(pp_cfg, os.path.join(work_dir, name))

        t0 = time.time()
        audio_paths, offsets = [], []
        for idx, wpath, arr, sr, _, _, _ in test_wavs:
            if use_preprime and idx in scout_cache:
                primed_arr, off_s = build_preprimed_audio(arr, sr, scout_cache[idx])
                ppath = os.path.join(work_dir, f"{idx:03d}_{name}_primed.wav")
                sf.write(ppath, primed_arr, sr)
                audio_paths.append(ppath)
                offsets.append(off_s)
            else:
                audio_paths.append(wpath)
                offsets.append(0.0)

        preds_all = model.diarize(audio=audio_paths, batch_size=1, **extra)
        elapsed = time.time() - t0

        tot_ref_s = w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
        spk_exact = short_segs = total_segs = 0
        spk_abs_err = []

        for (idx, _, _, _, ref_turns, dur, _), raw_hyp, off_s in zip(test_wavs, preds_all, offsets):
            segs = to_pipeline_segments(raw_hyp, merge_gap=merge_gap, min_duration=min_dur)
            if off_s > 0:
                segs = [
                    {"start": max(0.0, s["start"] - off_s), "end": s["end"] - off_s, "speaker": s["speaker"]}
                    for s in segs if s["end"] > off_s
                ]
            if idx not in scout_cache and use_deep_aosc:
                scout_cache[idx] = segs

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

        return {
            "name": name,
            "DER@0.00": w_der0 / tot_ref_s,
            "DER@0.25": w_der25 / tot_ref_s,
            "Miss%": w_miss / tot_ref_s,
            "FA%": w_fa / tot_ref_s,
            "Conf%": w_conf / tot_ref_s,
            "SpkExact%": spk_exact / len(test_wavs) * 100.0,
            "SpkMAE": float(np.mean(spk_abs_err)),
            "<0.5s": f"{short_segs}/{total_segs}",
            "time_s": elapsed,
        }
    finally:
        model.sortformer_modules.__dict__.update(orig_sm_state)


def main():
    test_items, train_items = load_viyt_split()
    work_dir = tempfile.mkdtemp(prefix="viyt_exp2_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    test_wavs = []
    for idx, arr, sr, ref_turns, dur, aid in test_items:
        wpath = os.path.join(work_dir, f"{idx:03d}.wav")
        sf.write(wpath, arr, sr)
        test_wavs.append((idx, wpath, arr, sr, ref_turns, dur, aid))

    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()

    tuned_stream = {
        "spkcache_len": 384,
        "fifo_len": 60,
        "chunk_len": 340,
        "chunk_right_context": 40,
        "spkcache_update_period": 200,
    }
    # Ngưỡng cắt chặt rìa câu (chống FA% lấn vào khoảng nghỉ ngắn của tiếng Việt)
    tight_vi_pp = {
        "onset": 0.62,
        "offset": 0.52,
        "pad_onset": 0.00,
        "pad_offset": 0.02,
        "min_duration_on": 0.18,
        "min_duration_off": 0.08,
    }

    scout_cache = {}
    results = []

    # 1. Baseline Repo
    results.append(eval_config(model, test_wavs, "1_Baseline_Repo", {}, POSTPROCESSING_BASE, False, False, 0.0, 0.0, work_dir, scout_cache))

    # 2. Zero-Train: Tight-VI Thresholds + Deep AOSC (No AGC)
    results.append(eval_config(model, test_wavs, "2_TightVI+DeepAOSC", tuned_stream, tight_vi_pp, True, False, 0.05, 0.15, work_dir, scout_cache))

    # 3. Zero-Train: Tight-VI + Deep AOSC + PrePrime
    results.append(eval_config(model, test_wavs, "3_TightVI+AOSC+PrePrime", tuned_stream, tight_vi_pp, True, True, 0.05, 0.15, work_dir, scout_cache))

    # 4. Surgical Fine-Tune Tầng 4 (subpixel_upsample + 156KB Head) trên 80 file Train -> Đo trên 20 file Test
    surgical_finetune_head(model, train_items, epochs=6, lr=1.5e-3)
    results.append(eval_config(model, test_wavs, "4_FineTuned_Head+AOSC", tuned_stream, tight_vi_pp, True, True, 0.05, 0.15, work_dir, scout_cache))

    print("\n" + "=" * 110)
    print("KẾT QUẢ THỬ NGHIỆM 2: TỐI ƯU FA% + DEEP AOSC + SURGICAL FINE-TUNE TẦNG 4 (ViYT-Diar Test N=20)")
    print("=" * 110)
    print(f"{'Cấu hình':25s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s} | {'Time':>6s}")
    print("-" * 110)
    for r in results:
        print(f"{r['name']:25s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s} | {r['time_s']:5.1f}s")
    print("=" * 110)


if __name__ == "__main__":
    main()
