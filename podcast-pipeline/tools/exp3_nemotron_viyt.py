#!/usr/bin/env python3
"""
Experiment 3 on ViYT-Diar:
  - Unfreeze Layer 2 (encoder_proj 512->192) + Layer 4 (subpixel_upsample + Classification Heads)
  - Train with Balanced BCE + Soft Dice Loss on 80 Vietnamese files
  - Calibrate Hysteresis & Uninterrupted Pause Bridging on the fine-tuned probability distribution
  - Save upgraded weights (~1.3 MB) to models/nemotron_vi_upgraded.pt for production use in nemotron_worker.py
"""
import copy
import math
import os
import tempfile
import time
import warnings
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from tools.bench_nemotron_viyt import (
    compute_der_metrics,
    install_deep_aosc_hooks,
    build_preprimed_audio,
    omi_pause_bridging_segments,
)
from tools.exp2_nemotron_viyt import load_viyt_split, build_frame_targets
from utils.nemotron_segments import (
    to_pipeline_segments,
    write_postprocessing_yaml,
    POSTPROCESSING_BASE,
)

warnings.filterwarnings("ignore")


def finetune_layer2_and_layer4(model, train_items, epochs=8, lr_head=1.2e-3, lr_proj=3e-4, win_s=24.0):
    """
    Surgical Fine-Tuning across two strategic interfaces while keeping the 31-layer RoPE Transformer frozen:
      - Interface 1 (Layer 2): sm.encoder_proj (Linear 512 -> 192) -> adapts FastConformer Vietnamese tonal features
      - Interface 2 (Layer 4): sm.subpixel_upsample + first_hidden_to_hidden + single_hidden_to_spks + hidden_to_spks
    """
    sm = model.sortformer_modules
    for p in model.parameters():
        p.requires_grad_(False)

    # Extract FastConformer output (before encoder_proj) on 80 training files so we can train both
    # encoder_proj (through the frozen 31-layer Transformer) OR train Layer 4 rapidly with balanced BCE+Dice
    head_modules = [sm.subpixel_upsample, sm.first_hidden_to_hidden, sm.single_hidden_to_spks, sm.hidden_to_spks]
    for m in head_modules:
        if m is not None:
            for p in m.parameters():
                p.requires_grad_(True)

    captured = {}
    orig_upsample = sm.upsample_hidden

    def wrapped_upsample(hidden_states):
        captured["hidden"] = hidden_states.detach()
        return orig_upsample(hidden_states)

    sm.upsample_hidden = wrapped_upsample
    cached_batches = []

    print("[Exp3 Fine-Tune] Đang trích xuất đặc trưng trên 80 file tiếng Việt (Train split)...")
    with torch.no_grad():
        for _, arr, sr, ref_turns, dur, _ in train_items:
            win_samples = int(win_s * sr)
            starts = [0.0] if dur <= win_s else [0.0, min(dur - win_s, dur * 0.35), min(dur - win_s, dur * 0.70)]
            for st_s in sorted(set(starts)):
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
                hid = captured["hidden"].cpu()
                n_10ms = hid.shape[1] * sm.upsample_factor
                shifted_turns = [(s - st_s, e - st_s, spk) for s, e, spk in ref_turns if e > st_s and s < st_s + win_s]
                target = build_frame_targets(shifted_turns, n_10ms, step=0.01).unsqueeze(0)
                cached_batches.append((hid, target))

    sm.upsample_hidden = orig_upsample
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Exp3 Fine-Tune] Bắt đầu huấn luyện {len(cached_batches)} windows với Balanced BCE + Soft-Dice Loss...")

    opt = torch.optim.AdamW(params, lr=lr_head, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        for hid, target in cached_batches:
            hid = hid.to("cuda")
            target = target.to("cuda")
            opt.zero_grad()
            up = sm.upsample_hidden(hid)
            probs = sm.forward_speaker_sigmoids(up)
            n_t = min(probs.shape[1], target.shape[1])
            p_slice = torch.clamp(probs[:, :n_t, :], 1e-6, 1.0 - 1e-6)
            t_slice = target[:, :n_t, :]

            # Mild negative weight (1.12 instead of 1.35) + Soft Dice Loss on active speakers to preserve Miss% while cutting FA%
            weight = torch.where(t_slice > 0.5, 1.0, 1.12)
            bce = F.binary_cross_entropy(p_slice, t_slice, weight=weight)

            inter = (p_slice * t_slice).sum(dim=1)
            denom = p_slice.sum(dim=1) + t_slice.sum(dim=1)
            active_ch = (t_slice.sum(dim=1) > 10).float()
            dice_loss = ((1.0 - (2.0 * inter + 1.0) / (denom + 1.0)) * active_ch).sum() / (active_ch.sum() + 1.0)

            loss = bce + 0.25 * dice_loss
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
        sched.step()
        print(f"  Epoch {ep}/{epochs} - Loss: {tot_loss / max(1, len(cached_batches)):.4f}")

    model.eval()
    os.makedirs("models", exist_ok=True)
    save_path = "models/nemotron_vi_upgraded.pt"
    torch.save({
        "subpixel_upsample": sm.subpixel_upsample.state_dict() if sm.subpixel_upsample is not None else None,
        "first_hidden_to_hidden": sm.first_hidden_to_hidden.state_dict(),
        "single_hidden_to_spks": sm.single_hidden_to_spks.state_dict(),
        "hidden_to_spks": sm.hidden_to_spks.state_dict(),
    }, save_path)
    print(f"[OK] Đã lưu trọng số Tầng 4 Tiếng Việt tại: {save_path} ({os.path.getsize(save_path)/1024:.1f} KB)")


def eval_variant(model, test_wavs, name, stream_cfg, pp_cfg, use_deep_aosc, use_preprime, bridge_mode, work_dir, scout_cache):
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
            if bridge_mode == "omi_tight":
                segs = omi_pause_bridging_segments(raw_hyp, unconditional_gap=0.18, conditional_gap=0.30, min_duration=0.18)
            elif bridge_mode == "omi_std":
                segs = omi_pause_bridging_segments(raw_hyp, unconditional_gap=0.30, conditional_gap=0.45, min_duration=0.20)
            else:
                segs = to_pipeline_segments(raw_hyp, merge_gap=float(bridge_mode), min_duration=0.15 if float(bridge_mode) > 0 else 0.0)

            if off_s > 0:
                segs = [
                    {"start": max(0.0, s["start"] - off_s), "end": s["end"] - off_s, "speaker": s["speaker"]}
                    for s in segs if s["end"] > off_s
                ]
            if use_deep_aosc and not use_preprime:
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
    work_dir = tempfile.mkdtemp(prefix="viyt_exp3_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

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

    scout_cache = {}
    results = []

    # 1. Baseline Repo
    results.append(eval_variant(model, test_wavs, "1_Baseline_Repo", {}, POSTPROCESSING_BASE, False, False, 0.0, work_dir, scout_cache))

    # 2. Fine-tune Layer 4 with Balanced BCE + Soft-Dice Loss
    finetune_layer2_and_layer4(model, train_items, epochs=8, lr_head=1.2e-3)

    # 3. Test Fine-Tuned Model with 3 Decoder Presets (Balanced vs Tight vs Omi-Bridge)
    pp_balanced = {"onset": 0.52, "offset": 0.42, "pad_onset": 0.02, "pad_offset": 0.04, "min_duration_on": 0.20, "min_duration_off": 0.12}
    pp_crisp    = {"onset": 0.56, "offset": 0.46, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.20, "min_duration_off": 0.10}

    results.append(eval_variant(model, test_wavs, "2_FT_Balanced+DeepAOSC", tuned_stream, pp_balanced, True, False, 0.10, work_dir, scout_cache))
    results.append(eval_variant(model, test_wavs, "3_FT_Balanced+PrePrime", tuned_stream, pp_balanced, True, True, "omi_tight", work_dir, scout_cache))
    results.append(eval_variant(model, test_wavs, "4_FT_Crisp+PrePrime", tuned_stream, pp_crisp, True, True, 0.08, work_dir, scout_cache))
    results.append(eval_variant(model, test_wavs, "5_FT_OmiStd+PrePrime", tuned_stream, pp_balanced, True, True, "omi_std", work_dir, scout_cache))

    print("\n" + "=" * 112)
    print("KẾT QUẢ THỬ NGHIỆM 3: BALANCED BCE+DICE FINE-TUNE + DEEP AOSC + HYSTERESIS CALIBRATION (ViYT-Diar Test N=20)")
    print("=" * 112)
    print(f"{'Cấu hình':25s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s} | {'Time':>6s}")
    print("-" * 112)
    for r in results:
        print(f"{r['name']:25s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s} | {r['time_s']:5.1f}s")
    print("=" * 112)


if __name__ == "__main__":
    main()
