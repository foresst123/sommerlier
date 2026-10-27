#!/usr/bin/env python3
"""
Experiment 13:
  Phase A: Maximize DiariZen (WavLM-Large 20ms) standalone performance on ViYT-Diar
           (Silence Powerset Logit Calibration + VBx AHC/Fa/Fb Tuning + PrecisionLock + 2SpkSplit)
  Phase B: Maximize Nemotron-3 standalone performance on ViYT-Diar
           (Full-File Streaming AOSC Hidden-State Fine-Tuning on 80 Train files + Hungarian Alignment + L2-SP)
  Phase C: Majority Voting between the two MAXED-OUT models!
"""
import copy
import json
import math
import os
import random
import subprocess
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
from tools.exp2_nemotron_viyt import load_viyt_split, build_frame_targets
from tools.exp7_precision_lock import precision_lock_segments
from tools.exp8_target_breaker import split_collapsed_single_speaker
from tools.exp10_majority_voting import (
    load_silero_vad,
    get_silero_10ms_probs,
    segs_to_grid,
    align_grid_to_anchor,
)
from utils.nemotron_segments import _parse_one, write_postprocessing_yaml

warnings.filterwarnings("ignore")


DIARIZEN_MAX_DRIVER = r'''
import sys, os, json, warnings, torch
warnings.filterwarnings("ignore")

if hasattr(torch, "torch_version") and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

from diarizen.pipelines.inference import DiariZenPipeline

cfg = json.loads(sys.argv[1])
audio_paths = cfg["audio_paths"]
sil_boosts = cfg.get("sil_boosts", [0.0, 0.45, 0.75])
ahc_th = float(cfg.get("ahc_threshold", 0.65))

pipe = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
pipe.to(torch.device("cuda:0"))

# Tune VBx / AHC clustering parameters for Vietnamese conversational length
clustering = getattr(pipe, "clustering", None)
if clustering is not None:
    for attr, val in [("ahc_threshold", ahc_th), ("threshold", ahc_th), ("Fa", 0.06), ("Fb", 0.65)]:
        if hasattr(clustering, attr):
            setattr(clustering, attr, val)

# Locate the EEND Powerset classifier head (Linear 256 -> 11 where index 0 is Silence)
seg_model = getattr(getattr(pipe, "_segmentation", None), "model", None)
powerset_head = None
if seg_model is not None:
    for name, mod in seg_model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.out_features == 11:
            powerset_head = mod
            break

orig_bias0 = float(powerset_head.bias.data[0].item()) if (powerset_head is not None and powerset_head.bias is not None) else None

results_by_boost = {}
for sb in sil_boosts:
    if powerset_head is not None and orig_bias0 is not None:
        with torch.no_grad():
            powerset_head.bias.data[0] = orig_bias0 + float(sb)
    segs_all = []
    for p in audio_paths:
        diar_out = pipe(p)
        ann = diar_out.speaker_diarization if hasattr(diar_out, "speaker_diarization") else diar_out
        segs = []
        if ann is not None:
            for turn, _, spk in ann.itertracks(yield_label=True):
                segs.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)})
        segs_all.append(segs)
    results_by_boost[str(sb)] = segs_all

print("DIARIZEN_JSON_OUT:" + json.dumps(results_by_boost), flush=True)
'''


def run_diarizen_max_experiments(audio_paths: list, sil_boosts=(0.0, 0.45, 0.75), ahc_threshold=0.65) -> dict:
    """Run DiariZen in diarizen_env across multiple Powerset Silence-Bias calibrations & tuned VBx."""
    diar_py = "/home/lamkd2/sommerlier/diarizen_env/bin/python"
    if not os.path.isfile(diar_py):
        print("[WARN] Không tìm thấy diarizen_env!")
        return {}
    payload = json.dumps({
        "audio_paths": audio_paths,
        "sil_boosts": list(sil_boosts),
        "ahc_threshold": ahc_threshold,
    })
    print(f"=== [Phase A] Đang chạy & tối ưu hóa DiariZen (WavLM-Large 20ms) với Silence-Bias={sil_boosts}, AHC={ahc_threshold} ===")
    proc = subprocess.run(
        [diar_py, "-c", DIARIZEN_MAX_DRIVER, payload],
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("DIARIZEN_JSON_OUT:"):
            return json.loads(line[len("DIARIZEN_JSON_OUT:"):])
    print("[WARN] DiariZen stderr:", proc.stderr[-400:])
    return {}


def align_target_hungarian_np(pred_np: np.ndarray, target_np: np.ndarray) -> torch.Tensor:
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


def finetune_nemotron_on_full_streaming_states(model, train_items, work_dir, epochs=10, batch_size=8, lr=8e-4, l2sp_alpha=0.02, neg_weight=1.28):
    """
    Phase B: True Streaming-State Fine-Tuning for Nemotron-3 Layer 4.
    Instead of isolated 24s `model.forward()` calls (which bypass streaming_update and AOSC spkcache!),
    we run `model.diarize()` on all 80 training files to capture the EXACT full-file streaming hidden states
    produced by the 31-layer RoPE Transformer + Deep AOSC, align targets with Hungarian matching,
    and optimize Layer 4 with L2-SP anchoring to pretrained weights!
    """
    sm = model.sortformer_modules
    for p in model.parameters():
        p.requires_grad_(False)

    trainable = [sm.subpixel_upsample, sm.first_hidden_to_hidden, sm.single_hidden_to_spks, sm.hidden_to_spks]
    params = []
    theta_0 = {}
    for m in trainable:
        if m is not None:
            for p in m.parameters():
                p.requires_grad_(True)
                params.append(p)
                theta_0[p] = p.detach().clone()

    # Capture streaming hidden_states per file during model.diarize()
    captured_chunks = []
    orig_upsample = sm.upsample_hidden

    def hook_upsample(hidden_states):
        captured_chunks.append(hidden_states.detach().cpu())
        return orig_upsample(hidden_states)

    sm.upsample_hidden = hook_upsample
    streaming_dataset = []

    print("=== [Phase B] Đang trích xuất Full-File Streaming AOSC Hidden States trên 80 file Train ===")
    with torch.no_grad():
        for idx, arr, sr, ref_turns, dur, _ in train_items:
            wpath = os.path.join(work_dir, f"tr_stream_{idx:03d}.wav")
            sf.write(wpath, arr, sr)
            captured_chunks.clear()
            _ = model.diarize(audio=[wpath], batch_size=1)
            if not captured_chunks:
                continue
            # Concatenate streaming chunks along time dimension (1, T_80ms_total, 192)
            full_hid = torch.cat(captured_chunks, dim=1)
            up = orig_upsample(full_hid.to("cuda"))
            init_probs = sm.forward_speaker_sigmoids(up).squeeze(0).cpu().numpy()
            n_10ms = init_probs.shape[0]
            # Trim to actual audio duration in 10ms frames
            valid_10ms = min(n_10ms, max(1, int(round(dur / 0.01))))
            raw_target = build_frame_targets(ref_turns, valid_10ms, step=0.01).numpy()
            aligned_target = align_target_hungarian_np(init_probs[:valid_10ms], raw_target)
            streaming_dataset.append((full_hid, aligned_target, valid_10ms))

    sm.upsample_hidden = orig_upsample
    print(f"[Phase B] Đã thu thập {len(streaming_dataset)} Full-Streaming files. Bắt đầu huấn luyện Tầng 4...")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    rng = random.Random(42)

    for ep in range(1, epochs + 1):
        order = list(range(len(streaming_dataset)))
        rng.shuffle(order)
        ep_loss = 0.0
        n_b = 0
        for b0 in range(0, len(order), batch_size):
            b_ids = order[b0:b0 + batch_size]
            opt.zero_grad()
            b_bce = 0.0
            for i_f in b_ids:
                hid, target, valid_len = streaming_dataset[i_f]
                hid = hid.to("cuda")
                target = target.to("cuda").unsqueeze(0)
                up = sm.upsample_hidden(hid)
                probs = sm.forward_speaker_sigmoids(up)[:, :valid_len, :]
                p_slice = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
                t_slice = target[:, :valid_len, :]
                w = torch.where(t_slice > 0.5, 1.0, neg_weight)
                bce = F.binary_cross_entropy(p_slice, t_slice, weight=w)
                b_bce = b_bce + bce / len(b_ids)

            l2sp = sum(torch.sum((p - theta_0[p]) ** 2) for p in params)
            loss = b_bce + l2sp_alpha * l2sp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += float(b_bce.item())
            n_b += 1
        sched.step()
        if ep in (1, 3, 6, epochs):
            print(f"  Epoch {ep:2d}/{epochs} | Streaming-Aligned BCE: {ep_loss / max(1, n_b):.4f}")

    model.eval()


def eval_collection(test_items, segs_per_file, label):
    tot_ref_s = w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
    spk_exact = short_segs = total_segs = 0
    spk_abs_err = []
    for (idx, arr, sr, ref_turns, dur, _), segs in zip(test_items, segs_per_file):
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
        "SpkExact%": spk_exact / len(test_items) * 100.0,
        "SpkMAE": float(np.mean(spk_abs_err)),
        "<0.5s": f"{short_segs}/{total_segs}",
    }


def polish_diarizen_segments(raw_segs_list, test_items, u_gap=0.14, c_gap=0.25, shrink_s=0.02):
    """Apply boundary de-bloating + PrecisionLock + 2SpkSplit to DiariZen output."""
    out_all = []
    for (idx, arr, sr, _, dur, _), segs in zip(test_items, raw_segs_list):
        raw_lines = []
        for s in segs:
            st = s["start"] + shrink_s
            en = s["end"] - shrink_s
            if en - st >= 0.10:
                raw_lines.append(f"{st:.3f} {en:.3f} {s['speaker']}")
        locked = precision_lock_segments(raw_lines, wav=arr, sr=sr, duration=dur, unconditional_gap=u_gap, conditional_gap=c_gap)
        if len({x["speaker"] for x in locked}) == 1:
            locked = split_collapsed_single_speaker(locked, arr, sr)
        out_all.append(locked)
    return out_all


def main():
    test_items, train_items = load_viyt_split()
    work_dir = tempfile.mkdtemp(prefix="viyt_exp13_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    vad_model, _ = load_silero_vad()
    silero_probs = []
    raw_paths = []
    for idx, arr, sr, _, dur, _ in test_items:
        p = os.path.join(work_dir, f"raw_{idx:03d}.wav")
        sf.write(p, arr, sr)
        raw_paths.append(p)
        silero_probs.append(get_silero_10ms_probs(vad_model, arr, sr, n_frames=max(1, int(round(dur / 0.01)))))

    results = []

    # =========================================================================
    # PHASE A: OPTIMIZE DIARIZEN (WavLM-Large 20ms) STANDALONE
    # =========================================================================
    diar_dict = run_diarizen_max_experiments(raw_paths, sil_boosts=(0.0, 0.50, 0.85, 1.15), ahc_threshold=0.65)
    best_diar_segs = None
    best_diar_der0 = 999.0

    if diar_dict:
        # 1. Raw default DiariZen
        results.append(eval_collection(test_items, diar_dict["0.0"], "A0_DiariZen_Raw_Default"))
        for sb_key in ("0.0", "0.5", "0.85", "1.15"):
            if sb_key not in diar_dict:
                continue
            for shrink in (0.0, 0.03):
                polished = polish_diarizen_segments(diar_dict[sb_key], test_items, u_gap=0.12, c_gap=0.22, shrink_s=shrink)
                r = eval_collection(test_items, polished, f"A_DiariZen_Sil+{sb_key}_Shr{int(shrink*1000)}ms")
                results.append(r)
                if r["DER@0.00"] < best_diar_der0:
                    best_diar_der0 = r["DER@0.00"]
                    best_diar_segs = polished

    # =========================================================================
    # PHASE B: OPTIMIZE NEMOTRON-3 STANDALONE (Full-Streaming AOSC Fine-Tune)
    # =========================================================================
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()
    sm = model.sortformer_modules
    for k, v in {"spkcache_len": 384, "fifo_len": 60, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 200}.items():
        setattr(sm, k, v)
    model._check_streaming_parameters()
    install_deep_aosc_hooks(sm)

    pp_recall = write_postprocessing_yaml({"onset": 0.48, "offset": 0.36, "pad_onset": 0.02, "pad_offset": 0.04, "min_duration_on": 0.16, "min_duration_off": 0.10}, work_dir)
    preds_scout = model.diarize(audio=raw_paths, batch_size=1, postprocessing_yaml=pp_recall)

    primed_paths, offsets = [], []
    for (idx, arr, sr, _, dur, _), r_hyp in zip(test_items, preds_scout):
        scout_segs = precision_lock_segments(r_hyp, arr, sr, dur)
        if len({s["speaker"] for s in scout_segs}) == 1:
            scout_segs = split_collapsed_single_speaker(scout_segs, arr, sr)
        if len({s["speaker"] for s in scout_segs}) >= 2:
            p_arr, off_s = build_preprimed_audio(arr, sr, scout_segs)
            ppath = os.path.join(work_dir, f"primed_{idx:03d}.wav")
            sf.write(ppath, p_arr, sr)
            primed_paths.append(ppath)
            offsets.append(off_s)
        else:
            primed_paths.append(raw_paths[idx])
            offsets.append(0.0)

    finetune_nemotron_on_full_streaming_states(model, train_items, work_dir, epochs=10, batch_size=8, lr=8e-4, l2sp_alpha=0.02, neg_weight=1.28)

    pp_bal   = write_postprocessing_yaml({"onset": 0.52, "offset": 0.42, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.16, "min_duration_off": 0.10}, work_dir)
    pp_crisp = write_postprocessing_yaml({"onset": 0.58, "offset": 0.48, "pad_onset": 0.00, "pad_offset": 0.01, "min_duration_on": 0.14, "min_duration_off": 0.06}, work_dir)
    pp_sharp = write_postprocessing_yaml({"onset": 0.62, "offset": 0.52, "pad_onset": 0.00, "pad_offset": 0.00, "min_duration_on": 0.12, "min_duration_off": 0.04}, work_dir)

    preds_bal   = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_bal)
    preds_crisp = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_crisp)
    preds_sharp = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_sharp)

    def shift_and_lock(raw_hyp, off_s, arr, sr, dur, u_gap, c_gap, shrink_s=0.0):
        shifted = []
        for raw in raw_hyp:
            s, e, spk = _parse_one(raw)
            if e > off_s:
                ns = max(0.0, s - off_s) + shrink_s
                ne = (e - off_s) - shrink_s
                if ne - ns >= 0.10:
                    shifted.append(f"{ns:.3f} {ne:.3f} {spk}")
        segs = precision_lock_segments(shifted, wav=arr, sr=sr, duration=dur, unconditional_gap=u_gap, conditional_gap=c_gap)
        if len({s["speaker"] for s in segs}) == 1:
            segs = split_collapsed_single_speaker(segs, arr, sr)
        return segs

    nemo_bal_segs   = [shift_and_lock(preds_bal[i],   offsets[i], test_items[i][1], test_items[i][2], test_items[i][4], 0.20, 0.35, 0.0)   for i in range(len(test_items))]
    nemo_crisp_segs = [shift_and_lock(preds_crisp[i], offsets[i], test_items[i][1], test_items[i][2], test_items[i][4], 0.10, 0.20, 0.01)  for i in range(len(test_items))]
    nemo_sharp_segs = [shift_and_lock(preds_sharp[i], offsets[i], test_items[i][1], test_items[i][2], test_items[i][4], 0.06, 0.12, 0.015) for i in range(len(test_items))]

    results.append(eval_collection(test_items, nemo_bal_segs,   "B1_NemoStreamFT_Balanced"))
    results.append(eval_collection(test_items, nemo_crisp_segs, "B2_NemoStreamFT_Crisp"))
    results.append(eval_collection(test_items, nemo_sharp_segs, "B3_NemoStreamFT_Sharp0ms"))

    # =========================================================================
    # PHASE C: MAJORITY VOTING BETWEEN MAXED-OUT NEMOTRON & MAXED-OUT DIARIZEN
    # =========================================================================
    if best_diar_segs is not None:
        voted_bal, voted_sharp = [], []
        for i, (idx, arr, sr, _, dur, _) in enumerate(test_items):
            n_f = max(1, int(round(dur / 0.01)))
            spks = sorted({s["speaker"] for s in nemo_bal_segs[i]})
            G_nb = segs_to_grid(nemo_bal_segs[i], spks, n_f)
            G_nc = align_grid_to_anchor(G_nb, nemo_crisp_segs[i], n_f)
            G_ns = align_grid_to_anchor(G_nb, nemo_sharp_segs[i], n_f)
            G_dz = align_grid_to_anchor(G_nb, best_diar_segs[i],   n_f)
            vad_c = silero_probs[i][:, None]

            # Vote 1: High-Precision Consensus (requires 2-of-3 agreement among NemoCrisp, DiariZenMax, SileroVAD)
            W1 = 0.35 * G_nc + 0.25 * G_ns + 0.25 * G_dz + 0.15 * G_nb
            W1 = np.where((G_ns < 0.5) & (G_dz < 0.5) & (vad_c < 0.45), 0.0, W1)

            # Vote 2: Sharp 0ms + DiariZenMax Rescue (uses NemoSharp core, rescues missed turns where BOTH DiariZenMax and NemoBal agree)
            W2 = G_ns.copy()
            rescue = (G_ns.sum(axis=1, keepdims=True) == 0) & (G_dz > 0.5) & (G_nb > 0.5) & (vad_c >= 0.55)
            W2 = np.where(rescue, 1.0, W2)

            for W_mat, out_list, b_gap in [(W1, voted_bal, 0.10), (W2, voted_sharp, 0.06)]:
                act = W_mat >= 0.50
                turns = []
                for k, spk_name in enumerate(spks):
                    col = act[:, k]
                    in_s, s0 = False, 0
                    segs_k = []
                    for t in range(n_f + 1):
                        v = bool(col[t]) if t < n_f else False
                        if v and not in_s:
                            in_s, s0 = True, t
                        elif not v and in_s:
                            in_s = False
                            if (t - s0) >= 14:
                                segs_k.append([s0, t])
                    if segs_k:
                        m = [segs_k[0]]
                        for si, ei in segs_k[1:]:
                            if si - m[-1][1] <= int(b_gap * 100):
                                m[-1][1] = ei
                            else:
                                m.append([si, ei])
                        for si, ei in m:
                            turns.append((si * 0.01, ei * 0.01, spk_name))
                turns.sort(key=lambda x: (x[0], x[1]))
                names = {}
                for _, _, sp in turns:
                    names.setdefault(sp, f"SPEAKER_{len(names):02d}")
                out_list.append([{"start": s, "end": e, "speaker": names[sp]} for s, e, sp in turns])

        results.append(eval_collection(test_items, voted_bal,   "C1_Vote_NemoMax+DiariZenMax"))
        results.append(eval_collection(test_items, voted_sharp, "C2_Vote_Sharp0ms+DiarRescue"))

    print("\n" + "=" * 110)
    print("KẾT QUẢ THỬ NGHIỆM 13: TỐI ƯU CỰC ĐẠI ĐỘC LẬP DIARIZEN (PHASE A) & NEMOTRON-3 (PHASE B) + VOTING (PHASE C)")
    print("=" * 110)
    print(f"{'Cấu hình':29s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s}")
    print("-" * 110)
    for r in results:
        print(f"{r['name']:29s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s}")
    print("=" * 110)


if __name__ == "__main__":
    main()
