#!/usr/bin/env python3
"""
Benchmark Nemotron-3-Diarization on Vietnamese ViYT-Diar dataset:
Testing Community & Deep Optimizations (Omi Pause-Bridging, AGC, Sortformer Tuning, AOSC Pre-Priming & Quorum).
"""
import argparse
import copy
import math
import os
import tempfile
import time
import types
import warnings
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")


def omi_pause_bridging_segments(raw_segments, unconditional_gap=0.40, conditional_gap=0.55, min_duration=0.15):
    """
    Omi / Aduomas 2-Stage Pause Bridging Decoder:
      - Stage 1: Merge same-speaker segments separated by <= unconditional_gap (0.4s).
      - Stage 2: Merge same-speaker segments separated by <= conditional_gap (0.55s)
                 ONLY IF no other speaker is active during the pause.
    """
    from utils.nemotron_segments import _parse_one, _speaker_index

    turns = []
    for raw in raw_segments or []:
        s, e, spk = _parse_one(raw)
        if e > s:
            turns.append([float(s), float(e), _speaker_index(spk)])
    turns.sort(key=lambda x: (x[0], x[1]))
    if not turns:
        return []

    names = {}
    for _, _, spk in turns:
        names.setdefault(spk, f"SPEAKER_{len(names):02d}")

    # Group by speaker while preserving global timeline for interruption checks
    by_spk = {}
    for s, e, spk in turns:
        by_spk.setdefault(spk, []).append([s, e])

    # Pass 1: Unconditional bridge <= 0.4s per speaker
    for spk, segs in by_spk.items():
        merged = [segs[0]]
        for s, e in segs[1:]:
            if s - merged[-1][1] <= unconditional_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        by_spk[spk] = merged

    # Pass 2: Conditional bridge <= 0.55s if NO other speaker speaks in [prev_end, next_start]
    for spk, segs in list(by_spk.items()):
        other_segs = [iv for other_spk, ivs in by_spk.items() if other_spk != spk for iv in ivs]
        merged = [segs[0]]
        for s, e in segs[1:]:
            gap = s - merged[-1][1]
            if gap <= conditional_gap:
                p_start, p_end = merged[-1][1], s
                interrupted = any(os_ < p_end and oe_ > p_start for os_, oe_ in other_segs)
                if not interrupted:
                    merged[-1][1] = max(merged[-1][1], e)
                    continue
            merged.append([s, e])
        by_spk[spk] = merged

    out = []
    for spk, segs in by_spk.items():
        for s, e in segs:
            if e - s >= min_duration:
                out.append({"start": s, "end": e, "speaker": names[spk]})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def apply_soft_agc(wav: np.ndarray, sr: int = 16000, win_s: float = 1.5, target_rms: float = 0.08, max_gain: float = 4.0) -> np.ndarray:
    """Windowed RMS Automatic Gain Control (AGC) to boost quiet/distant speakers without clipping."""
    x = np.asarray(wav, dtype=np.float32)
    win = max(1, int(round(win_s * sr)))
    hop = max(1, win // 4)
    if len(x) < win:
        rms = float(np.sqrt(np.mean(x * x) + 1e-8))
        g = min(max_gain, target_rms / rms) if rms > 1e-4 else 1.0
        return np.clip(x * g, -0.98, 0.98).astype(np.float32)

    sq = x * x
    csum = np.concatenate([[0.0], np.cumsum(sq, dtype=np.float64)])
    centers = np.arange(0, len(x), hop)
    gains = np.ones(len(centers), dtype=np.float32)
    noise_floor = float(np.percentile(np.abs(x), 20)) + 1e-4

    for idx, c in enumerate(centers):
        i0 = max(0, c - win // 2)
        i1 = min(len(x), c + win // 2)
        rms = math.sqrt((csum[i1] - csum[i0]) / max(1, i1 - i0) + 1e-8)
        if rms > noise_floor:
            gains[idx] = min(max_gain, max(0.5, target_rms / rms))

    env = np.interp(np.arange(len(x)), centers, gains).astype(np.float32)
    return np.clip(x * env, -0.98, 0.98).astype(np.float32)


def build_preprimed_audio(wav: np.ndarray, sr: int, scout_segs: list, slice_s: float = 3.5) -> tuple:
    """
    Point #5: Pre-priming AOSC Cache.
    Extracts a clean, non-overlapping 3.5s audio snippet for each major speaker in order of first appearance,
    prepends them with 0.4s silence padding before the main audio, and returns (primed_wav, offset_seconds).
    """
    if not scout_segs:
        return wav, 0.0
    spk_order = []
    for seg in scout_segs:
        if seg["speaker"] not in spk_order:
            spk_order.append(seg["speaker"])

    # Find non-overlapping clean segments >= 1.5s for each speaker
    clean_clips = []
    sil = np.zeros(int(0.4 * sr), dtype=np.float32)
    for spk in spk_order[:8]:
        best_seg = None
        best_dur = 0.0
        for s in scout_segs:
            if s["speaker"] != spk:
                continue
            st, en = s["start"], s["end"]
            dur = en - st
            if dur < 1.2:
                continue
            overlapped = any(
                o is not s and o["start"] < en and o["end"] > st
                for o in scout_segs
            )
            if not overlapped and dur > best_dur:
                best_dur = dur
                best_seg = (st, min(en, st + slice_s))
        if best_seg is not None:
            i0, i1 = int(best_seg[0] * sr), int(best_seg[1] * sr)
            if i1 > i0:
                clean_clips.extend([wav[i0:i1], sil])

    if not clean_clips:
        return wav, 0.0
    prefix = np.concatenate(clean_clips, axis=0)
    offset_s = len(prefix) / float(sr)
    return np.concatenate([prefix, wav], axis=0), offset_s


def compute_der_metrics(ref_turns, hyp_turns, duration, collar=0.0, step=0.01):
    n_frames = max(1, int(round(duration / step)))
    ref_spks = sorted({s for _, _, s in ref_turns})
    hyp_spks = sorted({s for _, _, s in hyp_turns})
    R = np.zeros((n_frames, len(ref_spks)), dtype=bool)
    H = np.zeros((n_frames, len(hyp_spks)), dtype=bool)
    eval_mask = np.ones(n_frames, dtype=bool)

    for start, end, spk in ref_turns:
        i0 = max(0, int(round(start / step)))
        i1 = min(n_frames, int(round(end / step)))
        if i1 > i0:
            R[i0:i1, ref_spks.index(spk)] = True
        if collar > 0:
            c = int(round(collar / step))
            eval_mask[max(0, i0 - c):min(n_frames, i0 + c)] = False
            eval_mask[max(0, i1 - c):min(n_frames, i1 + c)] = False

    for start, end, spk in hyp_turns:
        i0 = max(0, int(round(start / step)))
        i1 = min(n_frames, int(round(end / step)))
        if i1 > i0:
            H[i0:i1, hyp_spks.index(spk)] = True

    R, H = R[eval_mask], H[eval_mask]
    n_r, n_h = len(ref_spks), len(hyp_spks)
    cost = np.zeros((n_r, n_h), dtype=np.int64)
    for i in range(n_r):
        for j in range(n_h):
            cost[i, j] = -np.count_nonzero(R[:, i] & H[:, j])
    row_ind, col_ind = linear_sum_assignment(cost)

    matched = np.zeros(len(R), dtype=np.int32)
    for i, j in zip(row_ind, col_ind):
        matched += (R[:, i] & H[:, j]).astype(np.int32)

    n_ref = R.sum(axis=1).astype(np.int32)
    n_hyp = H.sum(axis=1).astype(np.int32)
    total_ref = float(n_ref.sum())
    if total_ref == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    miss = float(np.maximum(0, n_ref - n_hyp).sum())
    fa = float(np.maximum(0, n_hyp - n_ref).sum())
    conf = float((np.minimum(n_ref, n_hyp) - matched).sum())
    der = (miss + fa + conf) / total_ref * 100.0
    return der, miss / total_ref * 100.0, fa / total_ref * 100.0, conf / total_ref * 100.0, total_ref * step


def install_deep_aosc_hooks(sm, ema_alpha=0.25, diversity_weight=0.35, quorum_ratio=0.35):
    orig_streaming_update = sm.streaming_update

    def enhanced_streaming_update(self, streaming_state, chunk, preds, lc: int = 0, rc: int = 0):
        spkcache_len = streaming_state.spkcache.shape[1]
        if streaming_state.spkcache_compressed and spkcache_len > 0 and ema_alpha > 0:
            fresh_cache_preds = preds[:, :spkcache_len].detach()
            active_mask = (streaming_state.spkcache_preds.sum(dim=-1, keepdim=True) > 1e-4).float()
            streaming_state.spkcache_preds = (
                (1.0 - ema_alpha * active_mask) * streaming_state.spkcache_preds
                + (ema_alpha * active_mask) * fresh_cache_preds
            )
        return orig_streaming_update(streaming_state, chunk, preds, lc=lc, rc=rc)

    def enhanced_compress_spkcache(self, emb_seq, preds, mean_sil_emb, permute_spk: bool = False):
        batch_size, n_frames, n_spk = preds.shape
        if self.use_learnable_sil_emb:
            mean_sil_emb = self.learnable_sil_emb.to(dtype=emb_seq.dtype, device=emb_seq.device)
            mean_sil_emb = mean_sil_emb.unsqueeze(0).expand(batch_size, -1)

        base_len_per_spk = self.spkcache_len // n_spk - self.spkcache_sil_frames_per_spk
        min_pos_scores_per_spk = max(3, math.floor(base_len_per_spk * self.min_pos_scores_rate))

        scores = self._get_log_pred_scores(preds)
        if diversity_weight > 0 and n_frames > 2:
            emb_norm = F.normalize(emb_seq.float(), dim=-1)
            adj_sim = (emb_norm[:, 1:] * emb_norm[:, :-1]).sum(dim=-1, keepdim=True)
            novelty = torch.cat([torch.ones_like(adj_sim[:, :1]), 1.0 - torch.clamp(adj_sim, 0.0, 1.0)], dim=1)
            scores = scores + diversity_weight * novelty

        scores = self._disable_low_scores(preds, scores, min_pos_scores_per_spk)
        pos_counts = (scores > 0).sum(dim=1)
        quorum_frames = max(4, int(round(min_pos_scores_per_spk * quorum_ratio)))
        active_spk_mask = pos_counts >= quorum_frames

        n_active = int(active_spk_mask.sum().item())
        eff_spks = max(2, min(n_spk, n_active)) if n_active > 0 else n_spk
        dyn_len_per_spk = min(
            n_frames,
            max(base_len_per_spk, (self.spkcache_len // eff_spks) - self.spkcache_sil_frames_per_spk),
        )
        strong_boost_per_spk = math.floor(dyn_len_per_spk * self.strong_boost_rate)
        weak_boost_per_spk = math.floor(dyn_len_per_spk * self.weak_boost_rate)

        spk_perm = None
        if self.scores_boost_latest > 0:
            scores[:, self.spkcache_len :, :] += self.scores_boost_latest

        if n_active > 0:
            ghost_mask = (~active_spk_mask).unsqueeze(1).expand_as(scores)
            scores_for_boost = torch.where(ghost_mask, torch.tensor(float("-inf"), device=scores.device), scores)
            scores_for_boost = self._boost_topk_scores(scores_for_boost, strong_boost_per_spk, scale_factor=2)
            scores_for_boost = self._boost_topk_scores(scores_for_boost, weak_boost_per_spk, scale_factor=1)
            scores = torch.where(ghost_mask, scores, scores_for_boost)
        else:
            scores = self._boost_topk_scores(scores, strong_boost_per_spk, scale_factor=2)
            scores = self._boost_topk_scores(scores, weak_boost_per_spk, scale_factor=1)

        if self.spkcache_sil_frames_per_spk > 0:
            pad = torch.full((batch_size, self.spkcache_sil_frames_per_spk, n_spk), float("inf"), device=scores.device)
            scores = torch.cat([scores, pad], dim=1)

        topk_indices, is_disabled = self._get_topk_indices(scores)
        spkcache, spkcache_preds = self._gather_spkcache_and_preds(
            emb_seq, preds, topk_indices, is_disabled, mean_sil_emb
        )
        return spkcache, spkcache_preds, spk_perm

    sm.streaming_update = types.MethodType(enhanced_streaming_update, sm)
    sm._compress_spkcache = types.MethodType(enhanced_compress_spkcache, sm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=20, help="Number of ViYT-Diar test files (max 100)")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        from datasets import load_dataset

    print("=== 1. Đang tải bộ dữ liệu chuẩn tuanduy1612/ViYT-Diar ===")
    import io
    from datasets import Audio
    ds = load_dataset("tuanduy1612/ViYT-Diar", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    n_eval = min(args.max_files, len(ds))

    work_dir = tempfile.mkdtemp(prefix="viyt_bench_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    raw_wavs, agc_wavs = [], []
    for idx in range(n_eval):
        row = ds[idx]
        aud = row["audio"]
        if aud.get("bytes"):
            arr, sr = sf.read(io.BytesIO(aud["bytes"]), dtype="float32")
        else:
            arr, sr = sf.read(aud["path"], dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        arr = np.asarray(arr, dtype=np.float32)
        sr = int(sr)
        dur = len(arr) / sr
        wpath = os.path.join(work_dir, f"{idx:03d}_raw.wav")
        apath = os.path.join(work_dir, f"{idx:03d}_agc.wav")
        sf.write(wpath, arr, sr)
        sf.write(apath, apply_soft_agc(arr, sr), sr)
        ref_turns = [
            (float(s), float(e), str(spk))
            for s, e, spk in zip(row["timestamps_start"], row["timestamps_end"], row["speaker"])
            if float(e) > float(s)
        ]
        raw_wavs.append((wpath, arr, sr, ref_turns, dur, row["audio_id"]))
        agc_wavs.append((apath, apply_soft_agc(arr, sr), sr, ref_turns, dur, row["audio_id"]))

    from nemo.collections.asr.models import SortformerEncLabelModel
    from utils.nemotron_segments import to_pipeline_segments, write_postprocessing_yaml, POSTPROCESSING_BASE

    print("=== 2. Đang nạp nvidia/Nemotron-3-Diarization lên GPU ===")
    base_model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()
    orig_sm_state = copy.deepcopy(base_model.sortformer_modules.__dict__)

    tuned_stream = {
        "spkcache_len": 384,
        "fifo_len": 60,
        "chunk_len": 340,
        "chunk_right_context": 40,
        "spkcache_update_period": 200,
    }
    vi_pp = {
        "onset": 0.50,
        "offset": 0.35,
        "pad_onset": 0.05,
        "pad_offset": 0.08,
        "min_duration_on": 0.20,
        "min_duration_off": 0.25,
    }

    configs = [
        ("1_Baseline_Repo", {}, POSTPROCESSING_BASE, False, False, False),
        ("2_Omi_PauseBridge", {}, None, True, False, False),
        ("3_Omi+AGC+TunedCache", tuned_stream, vi_pp, True, True, False),
        ("4_Full_PrePrime+AOSC", tuned_stream, vi_pp, True, True, True),
    ]

    summary = []
    scout_cache = {}

    for name, stream_cfg, pp_cfg, use_omi, use_agc, use_deep_preprime in configs:
        base_model.sortformer_modules.__dict__.update(copy.deepcopy(orig_sm_state))
        for k, v in stream_cfg.items():
            setattr(base_model.sortformer_modules, k, v)
        if stream_cfg:
            base_model._check_streaming_parameters()
        if use_deep_preprime:
            install_deep_aosc_hooks(base_model.sortformer_modules)

        items = agc_wavs if use_agc else raw_wavs
        extra = {}
        if pp_cfg is not None:
            extra["postprocessing_yaml"] = write_postprocessing_yaml(pp_cfg, os.path.join(work_dir, name))

        t0 = time.time()
        audio_paths = []
        offsets = []
        for idx, (wpath, arr, sr, _, _, _) in enumerate(items):
            if use_deep_preprime and idx in scout_cache:
                primed_arr, off_s = build_preprimed_audio(arr, sr, scout_cache[idx])
                ppath = os.path.join(work_dir, f"{idx:03d}_primed.wav")
                sf.write(ppath, primed_arr, sr)
                audio_paths.append(ppath)
                offsets.append(off_s)
            else:
                audio_paths.append(wpath)
                offsets.append(0.0)

        preds_all = base_model.diarize(audio=audio_paths, batch_size=1, **extra)
        elapsed = time.time() - t0

        tot_ref_s = 0.0
        w_der0 = w_der25 = w_miss = w_fa = w_conf = 0.0
        spk_exact = 0
        spk_abs_err = []
        short_segs = total_segs = 0

        for idx, ((_, _, _, ref_turns, dur, _), raw_hyp, off_s) in enumerate(zip(items, preds_all, offsets)):
            if use_omi:
                segs = omi_pause_bridging_segments(raw_hyp)
            else:
                segs = to_pipeline_segments(raw_hyp, merge_gap=0.0, min_duration=0.0)

            if off_s > 0:
                shifted = []
                for s in segs:
                    if s["end"] > off_s:
                        shifted.append({
                            "start": max(0.0, s["start"] - off_s),
                            "end": s["end"] - off_s,
                            "speaker": s["speaker"],
                        })
                segs = shifted

            if name == "3_Omi+AGC+TunedCache":
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

        summary.append({
            "name": name,
            "DER@0.00": w_der0 / tot_ref_s,
            "DER@0.25": w_der25 / tot_ref_s,
            "Miss%": w_miss / tot_ref_s,
            "FA%": w_fa / tot_ref_s,
            "Conf%": w_conf / tot_ref_s,
            "SpkExact%": spk_exact / n_eval * 100.0,
            "SpkMAE": float(np.mean(spk_abs_err)),
            "<0.5s": f"{short_segs}/{total_segs}",
            "time_s": elapsed,
        })

    print("\n" + "=" * 108)
    print(f"KẾT QUẢ BENCHMARK TRÊN BỘ DỮ LIỆU TIẾNG VIỆT ViYT-Diar (N={n_eval} files)")
    print("=" * 108)
    print(f"{'Cấu hình':24s} | {'DER@0ms':>8s} | {'DER@.25s':>8s} | {'Miss%':>7s} | {'FA%':>7s} | {'Conf%':>7s} | {'SpkExact':>8s} | {'SpkMAE':>6s} | {'<0.5s':>9s} | {'Time':>6s}")
    print("-" * 108)
    for r in summary:
        print(f"{r['name']:24s} | {r['DER@0.00']:7.2f}% | {r['DER@0.25']:7.2f}% | {r['Miss%']:6.2f}% | {r['FA%']:6.2f}% | {r['Conf%']:6.2f}% | {r['SpkExact%']:7.1f}% | {r['SpkMAE']:6.2f} | {r['<0.5s']:>9s} | {r['time_s']:5.1f}s")
    print("=" * 108)


if __name__ == "__main__":
    main()
