#!/usr/bin/env python3
"""
Experiment 10: Hungarian-Aligned Multi-Model Neural Majority Voting (DOVER-Lap + Neural VAD Consensus)
Combines complementary neural voters at 10ms frame resolution:
  - Voter 1 (Anchor): Nemotron-3 Fine-Tuned Balanced + PrecisionLock + 2SpkSplit (SpkExact = 100%, DER@.25s = 1.61%)
  - Voter 2 (Boundary Sharpener): Nemotron-3 Fine-Tuned Crisp (low FA%)
  - Voter 3 (Quiet-Turn Detector): Nemotron-3 Pretrained High-Recall (Miss% = 0.85%)
  - Voter 4 (Neural Speech Boundary Gate): Silero Neural VAD v6.2 (cached locally in torch hub)
"""
import copy
import math
import os
import tempfile
import warnings
import numpy as np
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment

from tools.bench_nemotron_viyt import (
    compute_der_metrics,
    install_deep_aosc_hooks,
    build_preprimed_audio,
)
from tools.exp2_nemotron_viyt import load_viyt_split, surgical_finetune_head
from tools.exp7_precision_lock import precision_lock_segments
from tools.exp8_target_breaker import split_collapsed_single_speaker
from utils.nemotron_segments import _parse_one, write_postprocessing_yaml

warnings.filterwarnings("ignore")


def load_silero_vad():
    """Load cached Silero VAD v6.2 from local torch hub cache."""
    hub_dir = os.path.expanduser("~/.cache/torch/hub/snakers4_silero-vad_v6.2")
    if os.path.isdir(hub_dir):
        model, utils = torch.hub.load(repo_or_dir=hub_dir, model="silero_vad", source="local", force_reload=False, onnx=False)
    else:
        model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=False)
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval(), utils


def get_silero_10ms_probs(vad_model, wav: np.ndarray, sr: int = 16000, n_frames: int = None) -> np.ndarray:
    """
    Run Silero Neural VAD across a 16kHz waveform (512-sample = 32ms windows) and interpolate
    smoothly to the exact 10ms frame grid (n_frames).
    """
    dev = next(vad_model.parameters()).device
    vad_model.reset_states()
    win = 512  # 32ms required by Silero at 16kHz
    n_pad = (win - (len(wav) % win)) % win
    x = np.pad(wav, (0, n_pad), mode="constant").astype(np.float32)
    chunks = torch.from_numpy(x.reshape(-1, win)).to(dev)

    probs_32ms = []
    with torch.no_grad():
        for i in range(chunks.shape[0]):
            p = vad_model(chunks[i : i + 1], sr).item()
            probs_32ms.append(p)
    probs_32ms = np.array(probs_32ms, dtype=np.float32)
    t_32 = (np.arange(len(probs_32ms)) + 0.5) * (win / float(sr))
    t_10 = (np.arange(n_frames) + 0.5) * 0.01
    return np.interp(t_10, t_32, probs_32ms).astype(np.float32)


def segs_to_grid(segs: list, spk_list: list, n_frames: int, step: float = 0.01) -> np.ndarray:
    grid = np.zeros((n_frames, len(spk_list)), dtype=np.float32)
    for s in segs:
        if s["speaker"] in spk_list:
            k = spk_list.index(s["speaker"])
            i0 = max(0, int(round(s["start"] / step)))
            i1 = min(n_frames, int(round(s["end"] / step)))
            if i1 > i0:
                grid[i0:i1, k] = 1.0
    return grid


def align_grid_to_anchor(anchor_grid: np.ndarray, voter_segs: list, n_frames: int, step: float = 0.01) -> np.ndarray:
    """
    Map speaker IDs of `voter_segs` onto the exact speaker columns of `anchor_grid` (T, K_anchor)
    using Hungarian Bipartite Matching on temporal overlap.
    Solves the speaker ID permutation problem between different diarization models!
    """
    K_a = anchor_grid.shape[1]
    voter_spks = sorted({s["speaker"] for s in voter_segs})
    if not voter_spks or K_a == 0:
        return np.zeros_like(anchor_grid)

    v_grid = segs_to_grid(voter_segs, voter_spks, n_frames, step=step)
    K_v = len(voter_spks)
    cost = np.zeros((K_a, K_v), dtype=np.float64)
    for i in range(K_a):
        for j in range(K_v):
            cost[i, j] = -float(np.dot(anchor_grid[:, i], v_grid[:, j]))
    row_ind, col_ind = linear_sum_assignment(cost)

    aligned = np.zeros_like(anchor_grid)
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < 0:  # Only map if there is positive temporal intersection
            aligned[:, i] = v_grid[:, j]

    # For any unmapped voter speaker (e.g. if voter found Speech where anchor had none), assign frame-by-frame to closest active anchor speaker
    mapped_v = set(col_ind)
    for j in range(K_v):
        if j not in mapped_v and v_grid[:, j].sum() > 0:
            best_i = int(np.argmax(anchor_grid.sum(axis=0)))
            aligned[:, best_i] = np.maximum(aligned[:, best_i], v_grid[:, j])
    return aligned


def majority_vote_fusion(
    anchor_segs: list,
    crisp_segs: list,
    recall_segs: list,
    ultra_segs: list,
    silero_p: np.ndarray,
    duration: float,
    vote_thresh: float = 0.50,
    edge_vad_thresh: float = 0.35,
    recover_vad_thresh: float = 0.75,
    min_seg_s: float = 0.15,
    bridge_gap_s: float = 0.18,
    step: float = 0.01,
) -> list:
    """
    Hungarian-Aligned Weighted Majority Voting across 4 Neural Voters + Silero Neural VAD:
      - Anchor (weight 0.40): FT-Balanced + PrecisionLock + 2SpkSplit (100% SpkExact backbone)
      - Voter 2 (weight 0.25): FT-Crisp (tight boundary voter)
      - Voter 3 (weight 0.20): Pretrained High-Recall (quiet turn voter)
      - Voter 4 (weight 0.15): FT-UltraCrisp (strict onset/offset voter)
      - Voter 5 (Boundary & Recovery Gate): Silero Neural VAD v6.2 probability
    """
    if not anchor_segs:
        return []
    n_frames = max(1, int(round(duration / step)))
    anchor_spks = sorted({s["speaker"] for s in anchor_segs})
    K = len(anchor_spks)

    G_anchor = segs_to_grid(anchor_segs, anchor_spks, n_frames, step)
    G_crisp  = align_grid_to_anchor(G_anchor, crisp_segs, n_frames, step)
    G_recall = align_grid_to_anchor(G_anchor, recall_segs, n_frames, step)
    G_ultra  = align_grid_to_anchor(G_anchor, ultra_segs, n_frames, step)

    # Weighted neural diarization vote per speaker channel (T, K)
    W_vote = 0.40 * G_anchor + 0.25 * G_crisp + 0.20 * G_recall + 0.15 * G_ultra

    # Incorporate Silero Neural VAD at boundaries & uncovered quiet turns:
    # 1. Where BOTH Recall Voter AND Ultra/Crisp Voter + Silero VAD strongly agree (> recover_vad_thresh), boost vote
    vad_col = silero_p[:, None]  # (T, 1)
    strong_speech_consensus = (G_recall > 0.5) & (vad_col >= recover_vad_thresh)
    W_vote = np.where(strong_speech_consensus, np.maximum(W_vote, 0.55), W_vote)

    # 2. At outer segment edges (where Crisp & Ultra both say 0 AND Silero VAD < edge_vad_thresh), suppress boundary overhang
    boundary_overhang = (G_crisp < 0.5) & (G_ultra < 0.5) & (vad_col < edge_vad_thresh)
    W_vote = np.where(boundary_overhang, 0.0, W_vote)

    active_grid = W_vote >= vote_thresh

    # Convert voted grid back to segments with pause bridging
    min_frames = max(1, int(round(min_seg_s / step)))
    bridge_frames = max(0, int(round(bridge_gap_s / step)))
    final_turns = []

    for k in range(K):
        col = active_grid[:, k]
        segs_k = []
        in_seg, s0 = False, 0
        for t in range(n_frames + 1):
            val = bool(col[t]) if t < n_frames else False
            if val and not in_seg:
                in_seg, s0 = True, t
            elif not val and in_seg:
                in_seg = False
                if (t - s0) >= min_frames:
                    segs_k.append([s0, t])

        if not segs_k:
            continue
        if bridge_frames > 0:
            merged = [segs_k[0]]
            for s_i, e_i in segs_k[1:]:
                if s_i - merged[-1][1] <= bridge_frames:
                    merged[-1][1] = e_i
                else:
                    merged.append([s_i, e_i])
            segs_k = merged

        for s_i, e_i in segs_k:
            final_turns.append((s_i * step, e_i * step, anchor_spks[k]))

    final_turns.sort(key=lambda x: (x[0], x[1]))
    names = {}
    for _, _, spk in final_turns:
        names.setdefault(spk, f"SPEAKER_{len(names):02d}")
    return [{"start": s, "end": e, "speaker": names[spk]} for s, e, spk in final_turns]


def main():
    test_items, train_items = load_viyt_split()
    work_dir = tempfile.mkdtemp(prefix="viyt_exp10_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    print("=== 1. Nạp Silero Neural VAD v6.2 & Nemotron-3 ===")
    vad_model, _ = load_silero_vad()
    silero_probs = []
    raw_paths = []
    for idx, arr, sr, _, dur, _ in test_items:
        p = os.path.join(work_dir, f"raw_{idx:03d}.wav")
        sf.write(p, arr, sr)
        raw_paths.append(p)
        n_f = max(1, int(round(dur / 0.01)))
        silero_probs.append(get_silero_10ms_probs(vad_model, arr, sr, n_frames=n_f))

    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()
    sm = model.sortformer_modules
    for k, v in {"spkcache_len": 384, "fifo_len": 60, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 200}.items():
        setattr(sm, k, v)
    model._check_streaming_parameters()
    install_deep_aosc_hooks(sm)

    # Voter 3 (Pretrained High-Recall) & Scout for PrePriming
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

    preds_voter_recall = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_recall)

    # Fine-Tune Layer 4 Head (Exp2) for Voters 1, 2, 4
    surgical_finetune_head(model, train_items, epochs=6, lr=1.5e-3)

    pp_bal   = write_postprocessing_yaml({"onset": 0.52, "offset": 0.42, "pad_onset": 0.01, "pad_offset": 0.03, "min_duration_on": 0.16, "min_duration_off": 0.10}, work_dir)
    pp_crisp = write_postprocessing_yaml({"onset": 0.58, "offset": 0.48, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.16, "min_duration_off": 0.10}, work_dir)
    pp_ultra = write_postprocessing_yaml({"onset": 0.64, "offset": 0.54, "pad_onset": 0.00, "pad_offset": 0.00, "min_duration_on": 0.14, "min_duration_off": 0.06}, work_dir)

    preds_voter_bal   = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_bal)
    preds_voter_crisp = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_crisp)
    preds_voter_ultra = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_ultra)

    def shift_and_lock(raw_hyp, off_s, arr, sr, dur, u_gap, c_gap, do_split=True):
        if off_s > 0:
            shifted = []
            for raw in raw_hyp:
                s, e, spk = _parse_one(raw)
                if e > off_s:
                    shifted.append(f"{max(0.0, s - off_s):.3f} {e - off_s:.3f} {spk}")
            raw_hyp = shifted
        segs = precision_lock_segments(raw_hyp, wav=arr, sr=sr, duration=dur, unconditional_gap=u_gap, conditional_gap=c_gap)
        if do_split and len({s["speaker"] for s in segs}) == 1:
            segs = split_collapsed_single_speaker(segs, arr, sr)
        return segs

    # Build clean voter segment lists per file
    anchor_list, crisp_list, recall_list, ultra_list = [], [], [], []
    for i, (idx, arr, sr, _, dur, _) in enumerate(test_items):
        off_s = offsets[i]
        anchor_list.append(shift_and_lock(preds_voter_bal[i],   off_s, arr, sr, dur, 0.22, 0.38, True))
        crisp_list.append(shift_and_lock(preds_voter_crisp[i],  off_s, arr, sr, dur, 0.14, 0.24, True))
        recall_list.append(shift_and_lock(preds_voter_recall[i], off_s, arr, sr, dur, 0.20, 0.35, True))
        ultra_list.append(shift_and_lock(preds_voter_ultra[i],  off_s, arr, sr, dur, 0.06, 0.12, False))

    def eval_segs_collection(segs_per_file, label):
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

    results = [
        eval_segs_collection(anchor_list, "1_Voter1_Anchor_Balanced"),
        eval_segs_collection(crisp_list,  "2_Voter2_Crisp"),
        eval_segs_collection(recall_list, "3_Voter3_PretrainedRecall"),
    ]

    for v_name, v_th, e_vad, r_vad, br_gap in [
        ("4_MajVote_BalancedConsensus", 0.50, 0.35, 0.75, 0.18),
        ("5_MajVote_CrispBoundary",     0.55, 0.45, 0.80, 0.12),
        ("6_MajVote_StrictVADGate",     0.55, 0.55, 0.85, 0.08),
    ]:
        voted = [
            majority_vote_fusion(
                anchor_list[i], crisp_list[i], recall_list[i], ultra_list[i],
                silero_probs[i], test_items[i][4],
                vote_thresh=v_th, edge_vad_thresh=e_vad, recover_vad_thresh=r_vad, bridge_gap_s=br_gap,
            )
            for i in range(len(test_items))
        ]
        results.append(eval_segs_collection(voted, v_name))

    print("\n" + "=" * 108)
    print("KẾT QUẢ THỬ NGHIỆM 10: HUNGARIAN-ALIGNED NEURAL MAJORITY VOTING + SILERO VAD (Test N=20)")
    print("=" * 108)
    print(f"{'Cấu hình':28s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s}")
    print("-" * 108)
    for r in results:
        print(f"{r['name']:28s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s}")
    print("=" * 108)


if __name__ == "__main__":
    main()
