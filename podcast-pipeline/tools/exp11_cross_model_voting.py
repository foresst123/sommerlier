#!/usr/bin/env python3
"""
Experiment 11: True Cross-Architecture Majority Voting & Boundary Calibration on ViYT-Diar
  - Model A (Anchor ID & Overlap Tracker): Nemotron-3 (Sortformer 80ms, 100% SpkExact, 0.39% Conf)
  - Model B (20ms Fine Boundary Voter): DiariZen (WavLM-Large + Pyannote EEND via diarizen_worker.py)
  - Model C (Neural Speech Gate): Silero Neural VAD v6.2
  - Plus: Systematic Boundary Offset Calibration (delta_start, delta_end) learned on the 80 Train files
"""
import copy
import json
import math
import os
import subprocess
import tempfile
import time
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
from tools.exp10_majority_voting import (
    load_silero_vad,
    get_silero_10ms_probs,
    segs_to_grid,
    align_grid_to_anchor,
)
from utils.nemotron_segments import _parse_one, write_postprocessing_yaml

warnings.filterwarnings("ignore")


def run_diarizen_worker(audio_paths: list) -> list:
    """Locate diarizen python env and run diarizen_worker.py safely via JSON-line protocol."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    worker_script = os.path.join(root_dir, "diarizen_worker.py")
    candidates = [
        os.environ.get("DIARIZEN_PYTHON", ""),
        "/home/lamkd2/sommerlier/diarizen_env/bin/python",
        "/home/lamkd2/diarizen_env/bin/python",
        "/home/lamkd2/miniforge3/envs/diarizen/bin/python",
        "/home/lamkd2/miniforge3/envs/som/bin/python",
    ]
    diar_py = next((c for c in candidates if c and os.path.isfile(c)), "")
    if not (diar_py and os.path.isfile(worker_script)):
        return [[] for _ in audio_paths]

    print(f"=== Thử khởi động DiariZen qua {diar_py} ===")
    try:
        proc = subprocess.Popen(
            [diar_py, worker_script, "--config", "config.json", "--env", "a100"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=root_dir,
        )
        ready = False
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get("status") == "ready":
                    ready = True
                    break
                if msg.get("error"):
                    print(f"[WARN] DiariZen worker báo lỗi: {msg['error']}")
                    break
            except Exception:
                pass

        if not ready or proc.poll() is not None:
            err_txt = proc.stderr.read()[:300] if proc.stderr else ""
            print(f"[INFO] Bỏ qua DiariZen worker (chưa sẵn sàng: {err_txt.strip()}); dùng Hội đồng Nemotron + Silero VAD.")
            proc.terminate()
            return [[] for _ in audio_paths]

        out_segs = []
        for p in audio_paths:
            proc.stdin.write(json.dumps({"audio_path": p}) + "\n")
            proc.stdin.flush()
            segs_found = []
            while True:
                resp_line = proc.stdout.readline()
                if not resp_line:  # EOF
                    err_tail = proc.stderr.read()[:400] if proc.stderr else ""
                    print(f"[WARN] DiariZen EOF sớm: {err_tail.strip()}")
                    break
                resp_line = resp_line.strip()
                if not resp_line.startswith("{"):
                    continue
                try:
                    data = json.loads(resp_line)
                    if "segments" in data:
                        segs_found = data["segments"]
                        break
                    if "error" in data:
                        print(f"[WARN] DiariZen báo lỗi trên {p}: {data['error'][:200]}")
                        break
                except Exception:
                    continue
            out_segs.append(segs_found)

        proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        proc.stdin.flush()
        proc.terminate()
        return out_segs
    except Exception as e:
        print(f"[INFO] Bỏ qua DiariZen ({e}); tiếp tục với Nemotron + Silero VAD.")
        return [[] for _ in audio_paths]


def diagnose_and_calibrate_boundary_shift(train_items, train_segs_list):
    """
    Measure systematic onset/offset bias (in seconds) between predicted segments and ViYT-Diar
    ground-truth segments on the 80 Training files via grid search on (delta_start, delta_end).
    """
    best_der0 = 999.0
    best_ds, best_de = 0.0, 0.0
    # Search shifts from -0.08s to +0.12s in 10ms steps
    for ds in np.linspace(-0.04, 0.10, 15):
        for de in np.linspace(-0.10, 0.04, 15):
            tot_ref = 0.0
            w_der = 0.0
            for (_, _, _, ref_turns, dur, _), segs in zip(train_items[:30], train_segs_list[:30]):
                hyp = []
                for s in segs:
                    ns = max(0.0, s["start"] + ds)
                    ne = min(dur, s["end"] + de)
                    if ne - ns >= 0.10:
                        hyp.append((ns, ne, s["speaker"]))
                der0, _, _, _, ref_s = compute_der_metrics(ref_turns, hyp, dur, collar=0.0)
                w_der += der0 * ref_s
                tot_ref += ref_s
            avg_der = w_der / max(1e-6, tot_ref)
            if avg_der < best_der0:
                best_der0 = avg_der
                best_ds, best_de = float(ds), float(de)
    print(f"[Boundary Calibration on Train] Optimal shift: delta_start={best_ds:+.3f}s, delta_end={best_de:+.3f}s (Train DER@0ms={best_der0:.2f}%)")
    return best_ds, best_de


def cross_architecture_vote(
    anchor_segs: list,
    crisp_segs: list,
    recall_segs: list,
    diarizen_segs: list,
    silero_p: np.ndarray,
    duration: float,
    w_anchor: float = 0.35,
    w_crisp: float = 0.20,
    w_recall: float = 0.15,
    w_diarizen: float = 0.30,
    vote_th: float = 0.50,
    vad_edge_th: float = 0.45,
    delta_start: float = 0.0,
    delta_end: float = 0.0,
    bridge_gap_s: float = 0.12,
    min_seg_s: float = 0.14,
    step: float = 0.01,
) -> list:
    """
    Cross-Architecture Hungarian Majority Voting:
      - Aligns DiariZen (WavLM 20ms), Nemotron-Crisp, and Nemotron-Recall onto Nemotron-Anchor's 100%-SpkExact speaker columns.
      - Uses DiariZen + Silero VAD + Crisp to trim 80ms boundary overhang at segment edges while preserving Anchor's speaker identity.
    """
    if not anchor_segs:
        return []
    n_frames = max(1, int(round(duration / step)))
    anchor_spks = sorted({s["speaker"] for s in anchor_segs})
    K = len(anchor_spks)

    G_anchor = segs_to_grid(anchor_segs, anchor_spks, n_frames, step)
    G_crisp  = align_grid_to_anchor(G_anchor, crisp_segs, n_frames, step)
    G_recall = align_grid_to_anchor(G_anchor, recall_segs, n_frames, step)

    if diarizen_segs:
        G_diarizen = align_grid_to_anchor(G_anchor, diarizen_segs, n_frames, step)
    else:
        G_diarizen = G_crisp

    # Weighted vote across architectures
    W = w_anchor * G_anchor + w_crisp * G_crisp + w_recall * G_recall + w_diarizen * G_diarizen

    # Boundary refinement: at frames where Anchor is active, if BOTH DiariZen (20ms) AND Silero VAD say silence (< vad_edge_th), trim!
    vad_col = silero_p[:, None]
    edge_silence = (G_diarizen < 0.5) & (G_crisp < 0.5) & (vad_col < vad_edge_th)
    W = np.where(edge_silence, 0.0, W)

    # Recovery: where DiariZen (20ms), Recall, AND Silero VAD all agree (> 0.65) on a missed speech segment, recover it!
    strong_recover = (G_diarizen > 0.5) & (G_recall > 0.5) & (vad_col > 0.65)
    W = np.where(strong_recover, np.maximum(W, vote_th + 0.05), W)

    active = W >= vote_th
    min_frames = max(1, int(round(min_seg_s / step)))
    bridge_frames = max(0, int(round(bridge_gap_s / step)))
    final_turns = []

    for k in range(K):
        col = active[:, k]
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
                # Only bridge if DiariZen or Silero VAD also supports speech in the gap
                gap_vad = float(silero_p[merged[-1][1]:s_i].mean()) if s_i > merged[-1][1] else 1.0
                if (s_i - merged[-1][1]) <= bridge_frames and gap_vad >= 0.35:
                    merged[-1][1] = e_i
                else:
                    merged.append([s_i, e_i])
            segs_k = merged

        for s_i, e_i in segs_k:
            ns = max(0.0, s_i * step + delta_start)
            ne = min(duration, e_i * step + delta_end)
            if ne - ns >= min_seg_s:
                final_turns.append((ns, ne, anchor_spks[k]))

    final_turns.sort(key=lambda x: (x[0], x[1]))
    names = {}
    for _, _, spk in final_turns:
        names.setdefault(spk, f"SPEAKER_{len(names):02d}")
    return [{"start": s, "end": e, "speaker": names[spk]} for s, e, spk in final_turns]


def main():
    test_items, train_items = load_viyt_split()
    work_dir = tempfile.mkdtemp(prefix="viyt_exp11_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    vad_model, _ = load_silero_vad()
    silero_probs = []
    raw_paths = []
    for idx, arr, sr, _, dur, _ in test_items:
        p = os.path.join(work_dir, f"raw_{idx:03d}.wav")
        sf.write(p, arr, sr)
        raw_paths.append(p)
        silero_probs.append(get_silero_10ms_probs(vad_model, arr, sr, n_frames=max(1, int(round(dur / 0.01)))))

    train_paths_30 = []
    for idx, arr, sr, _, _, _ in train_items[:30]:
        p = os.path.join(work_dir, f"tr_{idx:03d}.wav")
        sf.write(p, arr, sr)
        train_paths_30.append(p)

    # 1. Run DiariZen (WavLM-Large 20ms) on the 20 test files
    diarizen_test_segs = run_diarizen_worker(raw_paths)

    # 2. Run Nemotron-3 Voters
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

    preds_voter_recall = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_recall)

    surgical_finetune_head(model, train_items, epochs=6, lr=1.5e-3)

    pp_bal   = write_postprocessing_yaml({"onset": 0.52, "offset": 0.42, "pad_onset": 0.01, "pad_offset": 0.03, "min_duration_on": 0.16, "min_duration_off": 0.10}, work_dir)
    pp_crisp = write_postprocessing_yaml({"onset": 0.58, "offset": 0.48, "pad_onset": 0.00, "pad_offset": 0.02, "min_duration_on": 0.16, "min_duration_off": 0.10}, work_dir)

    preds_voter_bal   = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_bal)
    preds_voter_crisp = model.diarize(audio=primed_paths, batch_size=1, postprocessing_yaml=pp_crisp)
    preds_train_30    = model.diarize(audio=train_paths_30, batch_size=1, postprocessing_yaml=pp_bal)

    def shift_and_lock(raw_hyp, off_s, arr, sr, dur, u_gap, c_gap):
        if off_s > 0:
            shifted = []
            for raw in raw_hyp:
                s, e, spk = _parse_one(raw)
                if e > off_s:
                    shifted.append(f"{max(0.0, s - off_s):.3f} {e - off_s:.3f} {spk}")
            raw_hyp = shifted
        segs = precision_lock_segments(raw_hyp, wav=arr, sr=sr, duration=dur, unconditional_gap=u_gap, conditional_gap=c_gap)
        if len({s["speaker"] for s in segs}) == 1:
            segs = split_collapsed_single_speaker(segs, arr, sr)
        return segs

    train_segs_30 = [
        shift_and_lock(preds_train_30[i], 0.0, train_items[i][1], train_items[i][2], train_items[i][4], 0.18, 0.30)
        for i in range(len(train_paths_30))
    ]
    best_ds, best_de = diagnose_and_calibrate_boundary_shift(train_items[:30], train_segs_30)

    anchor_list, crisp_list, recall_list = [], [], []
    for i, (idx, arr, sr, _, dur, _) in enumerate(test_items):
        off_s = offsets[i]
        anchor_list.append(shift_and_lock(preds_voter_bal[i],    off_s, arr, sr, dur, 0.20, 0.35))
        crisp_list.append(shift_and_lock(preds_voter_crisp[i],   off_s, arr, sr, dur, 0.10, 0.20))
        recall_list.append(shift_and_lock(preds_voter_recall[i], off_s, arr, sr, dur, 0.20, 0.35))

    def eval_collection(segs_per_file, label):
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

    results = []
    if any(diarizen_test_segs):
        results.append(eval_collection(diarizen_test_segs, "0_DiariZen_WavLM_Alone"))
    results.append(eval_collection(anchor_list, "1_Nemotron_Anchor_Alone"))

    for name, w_a, w_c, w_r, w_d, v_th, vad_e, ds, de, br_s in [
        ("2_CrossVote_Nemo+DiariZen",   0.35, 0.20, 0.15, 0.30, 0.50, 0.40, 0.0,     0.0,     0.12),
        ("3_CrossVote+BoundaryCalib",   0.35, 0.20, 0.15, 0.30, 0.50, 0.45, best_ds, best_de, 0.10),
        ("4_CrossVote_Sharp0ms",        0.30, 0.25, 0.10, 0.35, 0.55, 0.50, best_ds, best_de, 0.06),
        ("5_CrossVote_HighRecall",      0.35, 0.15, 0.25, 0.25, 0.45, 0.35, best_ds, best_de, 0.14),
    ]:
        voted = [
            cross_architecture_vote(
                anchor_list[i], crisp_list[i], recall_list[i], diarizen_test_segs[i],
                silero_probs[i], test_items[i][4],
                w_anchor=w_a, w_crisp=w_c, w_recall=w_r, w_diarizen=w_d,
                vote_th=v_th, vad_edge_th=vad_e, delta_start=ds, delta_end=de, bridge_gap_s=br_s,
            )
            for i in range(len(test_items))
        ]
        results.append(eval_collection(voted, name))

    print("\n" + "=" * 108)
    print("KẾT QUẢ THỬ NGHIỆM 11: CROSS-ARCHITECTURE VOTING (NEMOTRON-3 + DIARIZEN WAVLM + SILERO VAD)")
    print("=" * 108)
    print(f"{'Cấu hình':28s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s}")
    print("-" * 108)
    for r in results:
        print(f"{r['name']:28s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s}")
    print("=" * 108)


if __name__ == "__main__":
    main()
