#!/usr/bin/env python3
"""
Nemotron-3-Diarization Worker -- runs in its own environment (NeMo) on one GPU.

Usage:
    CUDA_VISIBLE_DEVICES=1 /path/to/nemotron_env/bin/python nemotron_worker.py

Protocol (stdin/stdout, line-delimited), identical to diarizen_worker.py:
    Input:  JSON line  {"audio_path": "/tmp/seg_xxx.wav"}
    Output: JSON line  {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}, ...]}

    Send  {"cmd": "quit"}  to shut down, {"cmd": "ping"} for {"status": "ok"}.

UNVERIFIED: written without a GPU or NeMo; see doc/nemotron_diarization_experiment.md.
"""

import argparse
import json
import os
import sys
import tempfile
import traceback
import warnings

warnings.filterwarnings("ignore")

MODEL_ID = "nvidia/Nemotron-3-Diarization"

# Everything models.nemotron may hold. Frame counts are 80 ms frames (model card).
STREAMING_KEYS = ("chunk_len", "chunk_right_context", "fifo_len", "spkcache_len",
                  "spkcache_update_period")
TWO_PASS_KEYS = (
    "two_pass", "legacy_one_pass", "batch_size", "vad_threshold",
    "vad_min_speech_ms", "vad_min_silence_ms", "vad_pad_ms",
    "chunk_target_seconds", "chunk_min_seconds", "chunk_max_seconds",
    "context_seconds", "wespeaker_min_clean_speech_seconds",
    "wespeaker_fallback",
)
ALLOWED_KEYS = STREAMING_KEYS + ("postprocessing", "postprocessing_yaml", "merge_gap",
                                 "min_duration") + TWO_PASS_KEYS


DEFAULT_VI_STREAMING = {
    "spkcache_len": 384,
    "fifo_len": 60,
    "chunk_len": 340,
    "chunk_right_context": 40,
    "spkcache_update_period": 200,
}
DEFAULT_VI_POSTPROCESSING = {
    "onset": 0.62,
    "offset": 0.52,
    "pad_onset": 0.00,
    "pad_offset": 0.02,
    "min_duration_on": 0.18,
    "min_duration_off": 0.12,
}


def _install_deep_aosc_and_vi_weights(model):
    import math
    import types
    import torch
    import torch.nn.functional as F

    sm = getattr(model, "sortformer_modules", None)
    if sm is None:
        return

    # Load fine-tuned Vietnamese Layer-4 weights if present
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "nemotron_vi_upgraded.pt")
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=next(model.parameters()).device, weights_only=True)
        if state.get("subpixel_upsample") and sm.subpixel_upsample is not None:
            sm.subpixel_upsample.load_state_dict(state["subpixel_upsample"])
        if state.get("first_hidden_to_hidden"):
            sm.first_hidden_to_hidden.load_state_dict(state["first_hidden_to_hidden"])
        if state.get("single_hidden_to_spks"):
            sm.single_hidden_to_spks.load_state_dict(state["single_hidden_to_spks"])
        if state.get("hidden_to_spks"):
            sm.hidden_to_spks.load_state_dict(state["hidden_to_spks"])

    orig_streaming_update = sm.streaming_update

    def enhanced_streaming_update(self, streaming_state, chunk, preds, lc: int = 0, rc: int = 0):
        spkcache_len = streaming_state.spkcache.shape[1]
        if streaming_state.spkcache_compressed and spkcache_len > 0:
            fresh_cache_preds = preds[:, :spkcache_len].detach()
            active_mask = (streaming_state.spkcache_preds.sum(dim=-1, keepdim=True) > 1e-4).float()
            streaming_state.spkcache_preds = (
                (1.0 - 0.25 * active_mask) * streaming_state.spkcache_preds
                + (0.25 * active_mask) * fresh_cache_preds
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
        if n_frames > 2:
            emb_norm = F.normalize(emb_seq.float(), dim=-1)
            adj_sim = (emb_norm[:, 1:] * emb_norm[:, :-1]).sum(dim=-1, keepdim=True)
            novelty = torch.cat([torch.ones_like(adj_sim[:, :1]), 1.0 - torch.clamp(adj_sim, 0.0, 1.0)], dim=1)
            scores = scores + 0.35 * novelty

        scores = self._disable_low_scores(preds, scores, min_pos_scores_per_spk)
        pos_counts = (scores > 0).sum(dim=1)
        quorum_frames = max(4, int(round(min_pos_scores_per_spk * 0.35)))
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

def _load_nemotron_config(config_path, env_name):
    """Read models.nemotron (optional tuning keys) out of the environment profile."""
    if not (config_path and os.path.exists(config_path)):
        return {}
    try:
        with open(config_path, encoding="utf-8") as handle:
            profile = json.load(handle).get("environments", {}).get(env_name, {})
        return profile.get("models", {}).get("nemotron", {})
    except Exception as e:
        print(json.dumps({"error": f"Failed to read config: {e}"}), flush=True)
        return {}


def load_model(config_path=None, env_name="kaggle"):
    # Imported here so --help and the conversion helpers work without NeMo.
    from nemo.collections.asr.models import SortformerEncLabelModel

    print(json.dumps({"status": "loading", "model": MODEL_ID}), flush=True)
    # CUDA_VISIBLE_DEVICES remaps the assigned GPU to cuda:0.
    model = SortformerEncLabelModel.from_pretrained(MODEL_ID)
    import torch
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    cfg = dict(_load_nemotron_config(config_path, env_name))
    unknown = sorted(set(cfg) - set(ALLOWED_KEYS))
    if unknown:
        raise ValueError(f"unknown models.nemotron key(s): {', '.join(unknown)}; "
                         f"allowed: {', '.join(ALLOWED_KEYS)}")
    if cfg.get("postprocessing") and cfg.get("postprocessing_yaml"):
        raise ValueError("set either models.nemotron.postprocessing or postprocessing_yaml, not both")

    # Streaming parameters: apply optimized Vietnamese defaults
    streaming = {**{"spkcache_len": 384, "fifo_len": 60, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 200}, **{key: int(cfg[key]) for key in STREAMING_KEYS if key in cfg}}
    for key, value in streaming.items():
        setattr(model.sortformer_modules, key, value)
    model._check_streaming_parameters()
    _install_deep_aosc_and_weights(model)

    convert = _load_converter()

    pp_values, pp_path = None, cfg.get("postprocessing_yaml")
    if pp_path:
        if not os.path.isfile(pp_path):
            raise FileNotFoundError(f"models.nemotron.postprocessing_yaml not found: {pp_path}")
    else:
        pp_values = convert.postprocessing_params(cfg) or {"onset": 0.52, "offset": 0.42, "pad_onset": 0.01, "pad_offset": 0.03, "min_duration_on": 0.16, "min_duration_off": 0.10} or dict(DEFAULT_VI_POSTPROCESSING)
        if pp_values:
            import tempfile
            pp_path = convert.write_postprocessing_yaml(
                pp_values, tempfile.mkdtemp(prefix="nemotron_pp_"))
    cfg["_postprocessing_path"] = pp_path

    # The VAD is loaded in the Nemotron environment so the chunk boundaries
    # come from the same process for every worker.  Profiles opt into the
    # two-pass path; legacy_one_pass remains an explicit escape hatch.
    if cfg.get("two_pass", False) and not cfg.get("legacy_one_pass", False):
        cfg["_silero_vad"] = load_silero_vad()

    print(json.dumps({"status": "ready", "streaming": streaming,
                      "postprocessing": pp_values or pp_path,
                      "two_pass": bool(cfg.get("two_pass", False)
                                       and not cfg.get("legacy_one_pass", False)),
                      "batch_size": int(cfg.get("batch_size", 16)),
                      "merge_gap": cfg.get("merge_gap", 0.0),
                      "min_duration": cfg.get("min_duration", 0.0)}), flush=True)
    return model, cfg


def _load_converter():
    # utils/nemotron_segments.py is loaded by path: putting podcast-pipeline on
    # sys.path would shadow NeMo's own 'utils' package (same reason as the
    # DiariZen worker's duplicated torch shim).
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "utils", "nemotron_segments.py")
    spec = importlib.util.spec_from_file_location("nemotron_segments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_two_pass_helpers():
    """Load the pipeline helper by path without shadowing NeMo's ``utils``."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "utils", "nemotron_two_pass.py")
    spec = importlib.util.spec_from_file_location("nemotron_two_pass", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_silero_vad():
    """Load Silero once and return a small timestamp-only callable."""
    import torch

    try:
        from silero_vad import get_speech_timestamps
        from silero_vad import load_silero_vad as load_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Nemotron two-pass mode requires the installed silero-vad package "
            "(expected silero-vad==6.2.2); install requirements-nemotron.txt"
        ) from exc
    try:
        model = load_model(onnx=False, sampling_rate=16000)
    except Exception as exc:
        raise RuntimeError(f"Could not load the offline Silero VAD package: {exc}") from exc

    def _vad(waveform, sample_rate, cfg):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
        kwargs = {
            "sampling_rate": int(sample_rate),
            "threshold": float(cfg.get("vad_threshold", 0.35)),
            "min_speech_duration_ms": int(cfg.get("vad_min_speech_ms", 80)),
            "min_silence_duration_ms": int(cfg.get("vad_min_silence_ms", 180)),
            "speech_pad_ms": int(cfg.get("vad_pad_ms", 160)),
            "return_seconds": True,
        }
        timestamps = get_speech_timestamps(waveform, model, **kwargs)
        return [(float(item["start"]), float(item["end"])) for item in timestamps]

    return _vad


def _read_audio(audio_path):
    import soundfile as sf

    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if getattr(waveform, "ndim", 1) == 2:
        waveform = waveform.mean(axis=1)
    return waveform, int(sample_rate)


def _write_window(waveform, sample_rate, window, directory, index):
    import soundfile as sf

    start = max(0, int(round(window.context_start * sample_rate)))
    end = min(len(waveform), int(round(window.context_end * sample_rate)))
    path = os.path.join(directory, f"chunk_{index:05d}.wav")
    sf.write(path, waveform[start:end], sample_rate)
    return path


def _model_pass(model, paths, cfg):
    extra = {}
    if cfg.get("_postprocessing_path"):
        extra["postprocessing_yaml"] = cfg["_postprocessing_path"]
    predicted = model.diarize(
        audio=list(paths),
        batch_size=int(cfg.get("batch_size", 16)),
        **extra,
    )
    if predicted is None or len(predicted) != len(paths):
        raise RuntimeError(
            f"Nemotron returned {0 if predicted is None else len(predicted)} results "
            f"for {len(paths)} audio chunks")
    return predicted


def _merge_mapped_segments(segments, merge_gap):
    merged = []
    for segment in sorted(segments, key=lambda s: (s["start"], s["end"])):
        if (merged and segment["speaker"] == merged[-1]["speaker"]
                and segment["start"] - merged[-1]["end"] <= merge_gap):
            merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
        else:
            merged.append(dict(segment))
    return [s for s in merged if s["end"] > s["start"]]


def _absolute_core_segments(chunks):
    """Return each chunk's local output in absolute source time, core-cropped."""
    result = []
    for chunk in chunks:
        for segment in chunk.segments:
            item = dict(segment)
            item["start"] = chunk.window.context_start + float(segment["start"])
            item["end"] = chunk.window.context_start + float(segment["end"])
            item["start"] = max(item["start"], chunk.window.core_start)
            item["end"] = min(item["end"], chunk.window.core_end)
            if item["end"] > item["start"]:
                result.append(item)
    return result


class _LazyWeSpeakerSimilarity:
    """Namespace-aware, lazy WeSpeaker cosine matcher."""

    def __init__(self, waveform, sample_rate):
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.groups = {}
        self.embeddings = {}
        self.embedder = None

    def register(self, namespace, label, segments):
        self.groups[(str(namespace), str(label))] = [dict(s) for s in segments]

    def register_candidate(self, label, segments):
        self.register("candidate", label, segments)

    def register_reference(self, label, segments):
        self.register("reference", label, segments)

    def _audio_for(self, key):
        import numpy as np

        parts = []
        for segment in self.groups.get(key, []):
            start = int(round(float(segment["start"]) * self.sample_rate))
            end = int(round(float(segment["end"]) * self.sample_rate))
            if end > start:
                parts.append(self.waveform[start:end])
        return np.concatenate(parts) if parts else None

    def _key_pair(self, left, right):
        if ("pass2", str(left)) in self.groups and ("pass1", str(right)) in self.groups:
            return ("pass2", str(left)), ("pass1", str(right))
        return ("candidate", str(left)), ("reference", str(right))

    def __call__(self, left, right):
        import numpy as np

        left_key, right_key = self._key_pair(left, right)
        try:
            if self.embedder is None:
                from models.wespeaker_embedding import WeSpeakerONNXEmbedder
                self.embedder = WeSpeakerONNXEmbedder(device="cpu")
            for key in (left_key, right_key):
                if key not in self.embeddings:
                    self.embeddings[key] = self.embedder.embed(
                        self._audio_for(key), self.sample_rate)
            left_embedding = np.asarray(
                self.embeddings[left_key].detach().cpu(), dtype=np.float32)
            right_embedding = np.asarray(
                self.embeddings[right_key].detach().cpu(), dtype=np.float32)
            denominator = float(np.linalg.norm(left_embedding)
                                * np.linalg.norm(right_embedding))
            return (float(np.dot(left_embedding, right_embedding) / denominator)
                    if denominator else -1.0)
        except Exception:
            # Speaker identity is optional evidence. A missing ONNX runtime or
            # an invalid short clip must not turn the diarization request into
            # a hard failure.
            return -1.0


def _lazy_wespeaker_similarity(pass1_segments, pass2_chunks, waveform, sample_rate,
                               min_clean_seconds):
    """Build a lazy cosine matcher; the model is not loaded for short evidence."""
    helpers = _load_two_pass_helpers()
    matcher = _LazyWeSpeakerSimilarity(waveform, sample_rate)
    pass1_audio = {}
    for segment in pass1_segments:
        pass1_audio.setdefault(str(segment["speaker"]), []).append(segment)
    pass2_audio = {}
    for segment in _absolute_core_segments(pass2_chunks):
        pass2_audio.setdefault(str(segment["speaker"]), []).append(segment)

    clean = {}
    for prefix, groups in (("pass1", pass1_audio), ("pass2", pass2_audio)):
        for label, segments in groups.items():
            matcher.register(prefix, label, segments)
            all_segments = []
            for other_label, other_segments in groups.items():
                all_segments.extend(dict(segment, speaker=other_label)
                                     for segment in other_segments)
            clean[(prefix, label)] = helpers.clean_speech_duration(all_segments, label)
    return matcher, clean


def run_two_pass(model, audio_path, cfg, *, vad_intervals=None):
    """Run VAD → pass 1 → guided context → pass 2 and return source-time segments."""
    helpers = _load_two_pass_helpers()
    convert = _load_converter()
    waveform, sample_rate = _read_audio(audio_path)
    duration = len(waveform) / float(sample_rate)
    vad = vad_intervals or cfg.get("_silero_vad")
    speech = vad(waveform, sample_rate, cfg) if vad is not None else [(0.0, duration)]
    windows = helpers.build_vad_windows(
        duration, speech,
        target_seconds=float(cfg.get("chunk_target_seconds", 25.0)),
        min_seconds=float(cfg.get("chunk_min_seconds", 20.0)),
        max_seconds=float(cfg.get("chunk_max_seconds", 30.0)),
        context_seconds=float(cfg.get("context_seconds", 2.0)),
    )
    with tempfile.TemporaryDirectory(prefix="nemotron_two_pass_") as directory:
        pass1_paths = [_write_window(waveform, sample_rate, window, directory, i)
                       for i, window in enumerate(windows)]
        pass1_raw = _model_pass(model, pass1_paths, cfg)
        pass1_chunks = [helpers.ChunkResult(
            window, convert.raw_to_segments(raw), chunk_id=str(i))
                        for i, (window, raw) in enumerate(zip(windows, pass1_raw))]
        pass1_similarity = _LazyWeSpeakerSimilarity(waveform, sample_rate)
        pass1_clean = helpers.chunk_clean_speech_seconds(pass1_chunks)
        pass1_by_chunk = helpers.globalize_pass1(
            pass1_chunks,
            speaker_similarity=pass1_similarity if cfg.get("wespeaker_fallback", True)
            else None,
            clean_speech_seconds=pass1_clean,
            min_clean_speech_seconds=float(
                cfg.get("wespeaker_min_clean_speech_seconds", 1.5)),
        )
        pass1_global = [segment for chunk in pass1_by_chunk for segment in chunk]

        guided = helpers.guide_windows_with_pass1(
            windows, pass1_global,
            context_seconds=float(cfg.get("context_seconds", 2.0)),
            duration=duration,
        )
        pass2_paths = [_write_window(waveform, sample_rate, window, directory,
                                     len(windows) + i)
                       for i, window in enumerate(guided)]
        pass2_raw = _model_pass(model, pass2_paths, cfg)
        pass2_chunks = [helpers.ChunkResult(
            window, convert.raw_to_segments(raw), chunk_id=str(i))
                        for i, (window, raw) in enumerate(zip(guided, pass2_raw))]
        similarity = None
        clean_speech = None
        if cfg.get("wespeaker_fallback", True):
            similarity, clean_speech = _lazy_wespeaker_similarity(
                pass1_global, pass2_chunks, waveform, sample_rate,
                float(cfg.get("wespeaker_min_clean_speech_seconds", 1.5)))
        mapped = helpers.map_pass2_to_global(
            pass2_chunks,
            pass1_global,
            speaker_similarity=similarity,
            clean_speech_seconds=clean_speech,
            min_clean_speech_seconds=float(
                cfg.get("wespeaker_min_clean_speech_seconds", 1.5)),
        )
        segments = [segment for chunk in mapped for segment in chunk]
    return _merge_mapped_segments(segments, float(cfg.get("merge_gap", 0.0)))


def diarize(model, audio_path, cfg):
    """Run inference on one file and return pipeline segments or an [ERROR] string."""
    try:
        if cfg.get("two_pass", False) and not cfg.get("legacy_one_pass", False):
            return run_two_pass(model, audio_path, cfg)
        convert = _load_converter()
        extra = {}
        if cfg.get("_postprocessing_path"):
            extra["postprocessing_yaml"] = cfg["_postprocessing_path"]
        predicted = model.diarize(audio=[audio_path], batch_size=1, **extra)
        if hasattr(model, "sortformer_modules") and hasattr(convert, "omi_pause_bridging_segments") and "merge_gap" not in cfg:
            return convert.omi_pause_bridging_segments(predicted[0], unconditional_gap=0.18, conditional_gap=0.30, min_duration=0.18)
        return convert.to_pipeline_segments(
            predicted[0],
            merge_gap=float(cfg.get("merge_gap", 0.0)),
            min_duration=float(cfg.get("min_duration", 0.0)),
        )
    except Exception as e:
        cause = e.__cause__ or e.__context__ or e
        return (f"[ERROR] {type(cause).__name__}: {cause}\n"
                + "".join(traceback.format_exception(type(cause), cause, cause.__traceback__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--env", default="kaggle")
    args = parser.parse_args()

    try:
        model, cfg = load_model(args.config, args.env)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {e}"}), flush=True)
        sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "invalid JSON"}), flush=True)
            continue

        cmd = request.get("cmd", "diarize")
        if cmd == "quit":
            print(json.dumps({"status": "shutdown"}), flush=True)
            break
        if cmd == "ping":
            print(json.dumps({"status": "ok"}), flush=True)
            continue

        audio_path = request.get("audio_path", "")
        if not audio_path or not os.path.exists(audio_path):
            print(json.dumps({"error": f"audio file not found: {audio_path}"}), flush=True)
            continue

        # num/min/max_speakers are ignored: the model has a fixed 8-slot output.
        segments = diarize(model, audio_path, cfg)
        if isinstance(segments, str) and segments.startswith("[ERROR]"):
            print(json.dumps({"error": segments}), flush=True)
        else:
            print(json.dumps({"segments": segments}), flush=True)


if __name__ == "__main__":
    main()
