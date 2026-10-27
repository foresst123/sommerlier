#!/usr/bin/env python3
import os
import re
from pathlib import Path

# 1. Bơm thuật toán Hậu xử lý (Omi Bridge + Precision Lock + TitaNet) vào utils/nemotron_segments.py
seg_path = Path("utils/nemotron_segments.py")
seg_src = seg_path.read_text(encoding="utf-8")

extra_code = '''
# === VIETNAMESE OPTIMIZED POST-PROCESSING ===
import numpy as np

def omi_pause_bridging_segments(raw_segments, unconditional_gap=0.18, conditional_gap=0.30, min_duration=0.18):
    turns = []
    for raw in raw_segments or []:
        s, e, spk = _parse_one(raw)
        if e > s:
            turns.append([float(s), float(e), _speaker_index(spk)])
    turns.sort(key=lambda x: (x[0], x[1]))
    if not turns: return []
    names = {}
    for _, _, spk in turns:
        names.setdefault(spk, f"SPEAKER_{len(names):02d}")
    by_spk = {}
    for s, e, spk in turns:
        by_spk.setdefault(spk, []).append([s, e])
    for spk, segs in by_spk.items():
        merged = [segs[0]]
        for s, e in segs[1:]:
            if s - merged[-1][1] <= unconditional_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        by_spk[spk] = merged
    for spk, segs in list(by_spk.items()):
        other_segs = [iv for other_spk, ivs in by_spk.items() if other_spk != spk for iv in ivs]
        merged = [segs[0]]
        for s, e in segs[1:]:
            gap = s - merged[-1][1]
            if gap <= conditional_gap:
                p_start, p_end = merged[-1][1], s
                if not any(os_ < p_end and oe_ > p_start for os_, oe_ in other_segs):
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

def precision_lock_segments(raw_segments, wav=None, sr=16000, duration=60.0, unconditional_gap=0.22, conditional_gap=0.38):
    bridged = omi_pause_bridging_segments(raw_segments, unconditional_gap=0.18, conditional_gap=0.30, min_duration=0.16)
    if not bridged: return []
    step = 0.01
    n_frames = max(1, int(round(duration / step)))
    spk_names = []
    for s in bridged:
        if s["speaker"] not in spk_names: spk_names.append(s["speaker"])
    K = len(spk_names)
    grid = np.zeros((n_frames, K), dtype=bool)
    for s in bridged:
        k = spk_names.index(s["speaker"])
        i0, i1 = max(0, int(round(s["start"] / step))), min(n_frames, int(round(s["end"] / step)))
        if i1 > i0: grid[i0:i1, k] = True
    tot_frames = grid.sum(axis=0)
    order_by_dur = sorted(range(K), key=lambda k: int(tot_frames[k]), reverse=True)
    kept_ks = []
    for k in order_by_dur:
        if tot_frames[k] == 0: continue
        if len(kept_ks) == 0:
            kept_ks.append(k)
            continue
        union_kept = np.any(grid[:, kept_ks], axis=1)
        ov_ratio = int(np.count_nonzero(grid[:, k] & union_kept)) / float(max(1, int(tot_frames[k])))
        if len(kept_ks) == 1:
            if ov_ratio > 0.70 and int(np.count_nonzero(grid[:, k] & (~union_kept))) * step < 0.80:
                grid[grid[:, k] & (~union_kept), kept_ks[0]] = True
                grid[:, k] = False
            else: kept_ks.append(k)
        else:
            if ov_ratio > 0.35 or int(np.count_nonzero(grid[:, k] & (~union_kept))) * step < 2.50:
                excl = grid[:, k] & (~union_kept)
                if np.any(excl):
                    best_p = max(kept_ks, key=lambda p: int(grid[:, p].sum()))
                    grid[excl, best_p] = True
                grid[:, k] = False
            else: kept_ks.append(k)
    raw_turns = []
    for k in sorted(kept_ks):
        col = grid[:, k]
        in_seg, s0 = False, 0
        for t in range(n_frames + 1):
            val = bool(col[t]) if t < n_frames else False
            if val and not in_seg:
                in_seg, s0 = True, t
            elif not val and in_seg:
                in_seg = False
                if (t - s0) >= 14:
                    raw_turns.append((s0 * step, t * step, k))
    raw_turns.sort(key=lambda x: (x[0], x[1]))
    names = {}
    for _, _, k in raw_turns:
        names.setdefault(k, f"SPEAKER_{len(names):02d}")
    return [{"start": s, "end": e, "speaker": names[k]} for s, e, k in raw_turns]

def split_collapsed_with_titanet(segs, wav, sr=16000):
    import torch
    try:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        titanet = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large").to("cuda").eval()
    except Exception: return segs
    spks = {s["speaker"] for s in segs}
    if len(spks) != 1 or len(segs) < 4: return segs
    embs, durs, valid_ids = [], [], []
    with torch.no_grad():
        for idx, s in enumerate(segs):
            i0, i1 = max(0, int(round(s["start"] * sr))), min(len(wav), int(round(s["end"] * sr)))
            chunk = wav[i0:i1]
            if len(chunk) < int(0.35 * sr): continue
            sig, sig_len = torch.from_numpy(chunk).unsqueeze(0).to("cuda"), torch.tensor([len(chunk)], dtype=torch.long, device="cuda")
            _, emb = titanet.forward(input_signal=sig, input_signal_length=sig_len)
            v = emb.squeeze(0).cpu().numpy()
            embs.append(v / (np.linalg.norm(v) + 1e-8))
            durs.append(s["end"] - s["start"])
            valid_ids.append(idx)
    if len(embs) < 4: return segs
    E = np.stack(embs, axis=0)
    sim = E @ E.T
    min_pos = np.unravel_index(np.argmin(sim), sim.shape)
    c0, c1 = E[min_pos[0]].copy(), E[min_pos[1]].copy()
    labels = np.zeros(len(E), dtype=np.int32)
    for _ in range(10):
        labels = (E @ c1 > E @ c0).astype(np.int32)
        if np.all(labels == 0) or np.all(labels == 1): return segs
        w0, w1 = np.array(durs) * (labels == 0), np.array(durs) * (labels == 1)
        c0, c1 = (E.T @ w0) / (w0.sum() + 1e-8), (E.T @ w1) / (w1.sum() + 1e-8)
        c0 /= (np.linalg.norm(c0) + 1e-8); c1 /= (np.linalg.norm(c1) + 1e-8)
    import copy
    out = copy.deepcopy(segs)
    l_map = {valid_ids[i]: int(labels[i]) for i in range(len(valid_ids))}
    last_cl = 0
    for idx, s in enumerate(out):
        cl = l_map.get(idx, last_cl)
        last_cl = cl
        s["speaker"] = f"SPEAKER_{cl:02d}"
    names = {}
    for s in out:
        names.setdefault(s["speaker"], f"SPEAKER_{len(names):02d}")
        s["speaker"] = names[s["speaker"]]
    return out
'''
if "def precision_lock_segments" not in seg_src:
    seg_path.write_text(seg_src + extra_code, encoding="utf-8")
    print("[OK] Đã bơm thuật toán Hậu xử lý vào utils/nemotron_segments.py")

# 2. Xóa các biến thể thử nghiệm rác trong nemotron_worker.py và ghi lại bản chuẩn
worker_path = Path("nemotron_worker.py")
w_src = worker_path.read_text(encoding="utf-8")

# Dọn dẹp các def _install_deep_aosc cũ nếu có
w_src = re.sub(r'def _install_deep_aosc_and_weights\(model\):.*?(?=def _load_nemotron_config)', '', w_src, flags=re.DOTALL)
w_src = re.sub(r'# Streaming parameters.*?(?=convert = _load_converter\(\))', '', w_src, flags=re.DOTALL)

golden_load_code = '''
def _install_deep_aosc_and_weights(model):
    import os, torch, types
    sm = getattr(model, "sortformer_modules", None)
    if sm is None: return
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "nemotron_vi_upgraded.pt")
    if os.path.isfile(ckpt_path):
        st = torch.load(ckpt_path, map_location=next(model.parameters()).device, weights_only=True)
        for k in ["subpixel_upsample", "first_hidden_to_hidden", "single_hidden_to_spks", "hidden_to_spks"]:
            if st.get(k) and getattr(sm, k, None) is not None:
                getattr(sm, k).load_state_dict(st[k])
    
    orig_su = sm.streaming_update
    def enhanced_su(self, streaming_state, chunk, preds, lc=0, rc=0):
        if streaming_state.spkcache_compressed and streaming_state.spkcache.shape[1] > 0:
            active_mask = (streaming_state.spkcache_preds.sum(dim=-1, keepdim=True) > 1e-4).float()
            streaming_state.spkcache_preds = (1.0 - 0.25 * active_mask) * streaming_state.spkcache_preds + (0.25 * active_mask) * preds[:, :streaming_state.spkcache.shape[1]].detach()
        return orig_su(streaming_state, chunk, preds, lc=lc, rc=rc)
    
    def enhanced_compress(self, emb_seq, preds, mean_sil_emb, permute_spk=False):
        batch_size, n_frames, n_spk = preds.shape
        if self.use_learnable_sil_emb:
            mean_sil_emb = self.learnable_sil_emb.to(dtype=emb_seq.dtype, device=emb_seq.device).unsqueeze(0).expand(batch_size, -1)
        import math, torch.nn.functional as F
        base_len_per_spk = self.spkcache_len // n_spk - self.spkcache_sil_frames_per_spk
        min_pos = max(3, math.floor(base_len_per_spk * self.min_pos_scores_rate))
        scores = self._get_log_pred_scores(preds)
        
        if n_frames > 2:
            emb_norm = F.normalize(emb_seq.float(), dim=-1)
            adj_sim = (emb_norm[:, 1:] * emb_norm[:, :-1]).sum(dim=-1, keepdim=True)
            novelty = torch.cat([torch.ones_like(adj_sim[:, :1]), 1.0 - torch.clamp(adj_sim, 0.0, 1.0)], dim=1)
            scores = scores + 0.35 * novelty
            
        scores = self._disable_low_scores(preds, scores, min_pos)
        active_spk_mask = (scores > 0).sum(dim=1) >= max(4, int(round(min_pos * 0.35)))
        n_active = int(active_spk_mask.sum().item())
        eff_spks = max(2, min(n_spk, n_active)) if n_active > 0 else n_spk
        dyn_len = min(n_frames, max(base_len_per_spk, (self.spkcache_len // eff_spks) - self.spkcache_sil_frames_per_spk))
        
        sb, wb = math.floor(dyn_len * self.strong_boost_rate), math.floor(dyn_len * self.weak_boost_rate)
        if self.scores_boost_latest > 0: scores[:, self.spkcache_len :, :] += self.scores_boost_latest
        
        if n_active > 0:
            ghost_mask = (~active_spk_mask).unsqueeze(1).expand_as(scores)
            s_boost = torch.where(ghost_mask, torch.tensor(float("-inf"), device=scores.device), scores)
            s_boost = self._boost_topk_scores(s_boost, sb, scale_factor=2)
            s_boost = self._boost_topk_scores(s_boost, wb, scale_factor=1)
            scores = torch.where(ghost_mask, scores, s_boost)
        else:
            scores = self._boost_topk_scores(scores, sb, scale_factor=2)
            scores = self._boost_topk_scores(scores, wb, scale_factor=1)
            
        if self.spkcache_sil_frames_per_spk > 0:
            pad = torch.full((batch_size, self.spkcache_sil_frames_per_spk, n_spk), float("inf"), device=scores.device)
            scores = torch.cat([scores, pad], dim=1)
        topk_indices, is_disabled = self._get_topk_indices(scores)
        spkcache, spkcache_preds = self._gather_spkcache_and_preds(emb_seq, preds, topk_indices, is_disabled, mean_sil_emb)
        return spkcache, spkcache_preds, None

    sm.streaming_update = types.MethodType(enhanced_su, sm)
    sm._compress_spkcache = types.MethodType(enhanced_compress, sm)

def _load_nemotron_config'''
w_src = w_src.replace("def _load_nemotron_config", golden_load_code, 1)

golden_stream_apply = '''    # Streaming parameters: apply optimized Vietnamese defaults
    streaming = {**{"spkcache_len": 384, "fifo_len": 60, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 200}, **{key: int(cfg[key]) for key in STREAMING_KEYS if key in cfg}}
    for key, value in streaming.items():
        setattr(model.sortformer_modules, key, value)
    model._check_streaming_parameters()
    _install_deep_aosc_and_weights(model)

    convert = _load_converter()
'''
w_src = w_src.replace("    convert = _load_converter()", golden_stream_apply, 1)

# Ép Hysteresis về bộ Balanced
w_src = w_src.replace(
    'pp_values = convert.postprocessing_params(cfg)',
    'pp_values = convert.postprocessing_params(cfg) or {"onset": 0.52, "offset": 0.42, "pad_onset": 0.01, "pad_offset": 0.03, "min_duration_on": 0.16, "min_duration_off": 0.10}'
)

# Ốp Hậu xử lý (Precision Lock + TitaNet Split) thay vì to_pipeline_segments
old_diarize = '''        predicted = model.diarize(audio=[audio_path], batch_size=1, **extra)
        return convert.to_pipeline_segments(
            predicted[0],
            merge_gap=float(cfg.get("merge_gap", 0.0)),
            min_duration=float(cfg.get("min_duration", 0.0)),
        )'''

new_diarize = '''        predicted = model.diarize(audio=[audio_path], batch_size=1, **extra)
        if hasattr(convert, "precision_lock_segments"):
            import soundfile as sf
            wav, sr = sf.read(audio_path, dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            segs = convert.precision_lock_segments(predicted[0], wav=wav, sr=sr, duration=len(wav)/float(sr), unconditional_gap=0.22, conditional_gap=0.38)
            if len({s["speaker"] for s in segs}) == 1:
                segs = convert.split_collapsed_with_titanet(segs, wav, sr)
            return segs
        return convert.to_pipeline_segments(predicted[0], merge_gap=float(cfg.get("merge_gap", 0.0)), min_duration=float(cfg.get("min_duration", 0.0)))'''
w_src = w_src.replace(old_diarize, new_diarize, 1)

worker_path.write_text(w_src, encoding="utf-8")
print("[OK] Đã ốp vĩnh viễn Bộ Tham Số Vàng vào nemotron_worker.py!")
