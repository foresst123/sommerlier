import json
import os
import torch
import torchaudio
import numpy as np
from diffusers import DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ID = "sarulab-speech/DialogueSidon"
MODEL_FILES = ["ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2", "metadata.json"]
SAMPLE_RATE_IN = 16_000
CHUNK_SECONDS = 20.0  # Kept similar to original worker
OVERLAP_SECONDS = 5.0 # Kept similar to original worker

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", os.environ.get("HF_TOKEN"))

# ---------------------------------------------------------------------------
# Feature extraction (inline, no sidon src dependency)
# ---------------------------------------------------------------------------
def _pad_batch(features: list[torch.Tensor], pad_to_multiple_of: int = 2, padding_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    target_length = max(f.shape[0] for f in features)
    if pad_to_multiple_of:
        target_length = ((target_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of)
    batch_size = len(features)
    feature_dim = features[0].shape[1]
    device = features[0].device
    padded = torch.full((batch_size, target_length, feature_dim), padding_value, dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, target_length), dtype=torch.int64, device=device)
    for i, feat in enumerate(features):
        padded[i, : feat.shape[0]] = feat
        mask[i, : feat.shape[0]] = 1
    return padded, mask

def extract_fbank_features(waveforms: list[torch.Tensor], device: torch.device, num_mel_bins: int = 80, stride: int = 2) -> dict[str, torch.Tensor]:
    features = []
    for wav in waveforms:
        if wav.ndim > 1:
            wav = wav[0]
        feat = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0), sample_frequency=SAMPLE_RATE_IN, num_mel_bins=num_mel_bins, frame_length=25, frame_shift=10,
            dither=0.0, preemphasis_coefficient=0.97, remove_dc_offset=True, window_type="povey", use_energy=False, energy_floor=1.192092955078125e-07
        )
        mean = feat.mean(0, keepdim=True)
        var = feat.var(0, keepdim=True)
        feat = (feat - mean) / torch.sqrt(var + 1e-5)
        features.append(feat.to(device))
    input_features, attention_mask = _pad_batch(features)
    b, t, c = input_features.shape
    t = (t // stride) * stride
    input_features = input_features[:, :t, :]
    attention_mask = attention_mask[:, :t]
    input_features = input_features.reshape(b, t // stride, c * stride)
    attention_mask = attention_mask[:, 1::stride]
    return {"input_features": input_features, "attention_mask": attention_mask}

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
_cache: dict = {}

def load_models(device: torch.device) -> dict:
    cache_key = str(device)
    if cache_key in _cache:
        return _cache[cache_key]

    print(f"[sidon_infer] Downloading model files from {REPO_ID} ...", flush=True)
    paths = {f: hf_hub_download(repo_id=REPO_ID, filename=f, token=HF_TOKEN) for f in MODEL_FILES}

    with open(paths["metadata.json"]) as fp:
        meta = json.load(fp)

    ssl_encoder = torch.export.load(paths["ssl_encoder.pt2"]).module().to(device)
    diffusion_head = torch.export.load(paths["diffusion_head.pt2"]).module().to(device)
    vae_decoder = torch.export.load(paths["vae_decoder.pt2"]).module().to(device)

    latent_norm_mean = torch.tensor(meta["latent_norm_mean"], dtype=torch.float32, device=device).view(1, 1, -1)
    latent_norm_std = torch.tensor(meta["latent_norm_std"], dtype=torch.float32, device=device).view(1, 1, -1)

    scheduler = DPMSolverMultistepScheduler.from_config(
        meta["ddpm_config"], algorithm_type="dpmsolver++", timestep_spacing="linspace"
    )

    models = {
        "ssl_encoder": ssl_encoder, "diffusion_head": diffusion_head, "vae_decoder": vae_decoder,
        "latent_norm_mean": latent_norm_mean, "latent_norm_std": latent_norm_std,
        "latent_norm_initialized": meta["latent_norm_initialized"], "scheduler": scheduler,
        "latent_dim": meta["latent_dim"], "sample_rate": meta["sample_rate"],
    }
    _cache[cache_key] = models
    return models

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _normalize(latents: torch.Tensor, models: dict) -> torch.Tensor:
    if not models["latent_norm_initialized"]:
        return latents
    return ((latents.float() - models["latent_norm_mean"]) / models["latent_norm_std"]).to(latents.dtype)

def _denormalize(latents: torch.Tensor, models: dict) -> torch.Tensor:
    if not models["latent_norm_initialized"]:
        return latents
    return (latents.float() * models["latent_norm_std"] + models["latent_norm_mean"]).to(latents.dtype)

@torch.inference_mode()
def _separate_chunk(wav: torch.Tensor, num_steps: int, models: dict, device: torch.device) -> torch.Tensor:
    latent_dim = models["latent_dim"]
    noisy_ssl = extract_fbank_features([wav.view(-1)], device)
    features, pred0, pred1 = models["ssl_encoder"](noisy_ssl["input_features"], noisy_ssl["attention_mask"])
    predicted_latents = torch.cat([pred0, pred1], dim=-1)
    conditioning = torch.cat([_normalize(predicted_latents, models), features], dim=-1)
    seq_len = conditioning.shape[1]
    scheduler = models["scheduler"]
    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn((1, seq_len, latent_dim * 2), device=device, dtype=conditioning.dtype)
    for t in scheduler.timesteps:
        t_batch = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
        latents = scheduler.step(models["diffusion_head"](latents, t_batch, conditioning), t, latents).prev_sample
    latents = _denormalize(latents, models)
    spk1 = models["vae_decoder"](latents[:, :, :latent_dim].transpose(1, 2)).squeeze(0)
    spk2 = models["vae_decoder"](latents[:, :, latent_dim:].transpose(1, 2)).squeeze(0)
    return torch.cat([spk1, spk2], dim=0)

# Frame length for the envelope the channel test runs on. Short enough to
# follow syllables, long enough that phase does not enter into it.
_ENVELOPE_FRAME = 240          # 10ms at 24kHz


def _energy_envelope(x: torch.Tensor, frame: int = _ENVELOPE_FRAME) -> torch.Tensor:
    """Per-frame RMS, which describes when a voice is loud rather than where its
    waveform happens to sit."""
    x = x.reshape(-1)
    n = x.shape[-1] // frame
    if n < 2:
        return x
    return x[: n * frame].reshape(n, frame).pow(2).mean(dim=1).clamp_min(1e-12).sqrt()


def _channel_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Correlate two channels by energy envelope, not by raw samples.

    The decoder resynthesises each chunk independently, so the same voice comes
    back with a slightly different phase either side of a seam. Raw-sample
    correlation collapses under that: a 2ms shift takes it from 1.0 to -0.64,
    which is enough to make the caller swap two channels that were already in
    the right order. The envelope is unchanged by the same shift (0.997), and
    which speaker is loud when is what actually distinguishes the two tracks.
    """
    a, b = _energy_envelope(a), _energy_envelope(b)
    n = min(a.shape[-1], b.shape[-1])
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    a, b = a - a.mean(), b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    return float(torch.dot(a, b) / denom) if float(denom) > 1e-8 else 0.0

def _maybe_swap(prev_overlap: torch.Tensor, curr_chunk: torch.Tensor, overlap_samples: int) -> tuple[torch.Tensor, bool]:
    if overlap_samples <= 0 or prev_overlap.shape[0] != 2 or curr_chunk.shape[0] != 2:
        return curr_chunk, False
    curr_ov = curr_chunk[:, :overlap_samples]
    direct = _channel_similarity(prev_overlap[0], curr_ov[0]) + _channel_similarity(prev_overlap[1], curr_ov[1])
    swapped = _channel_similarity(prev_overlap[0], curr_ov[1]) + _channel_similarity(prev_overlap[1], curr_ov[0])
    if swapped > direct:
        return curr_chunk[[1, 0], :], True
    return curr_chunk, False

def run_separation_chunked(wav: torch.Tensor, sample_rate: int, num_steps: int, device: torch.device) -> tuple[torch.Tensor, int]:
    models = load_models(device)
    out_sr = models["sample_rate"]
    if sample_rate != SAMPLE_RATE_IN:
        wav_16k = torchaudio.functional.resample(wav, sample_rate, SAMPLE_RATE_IN)
    else:
        wav_16k = wav
    wav_16k = wav_16k.to(device)

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE_IN)
    total_samples = wav_16k.shape[-1]
    if total_samples <= chunk_samples:
        max_val = wav_16k.abs().max().clamp_min(1e-6)
        wav_norm = torch.nn.functional.pad(0.9 * wav_16k / max_val, (160, 160))
        separated = _separate_chunk(wav_norm, num_steps, models, device)
        # Undo the input scaling so the result sits at the level of the audio
        # that was handed in. With one chunk this only affects absolute level,
        # but doing it here keeps both paths on the same scale.
        return separated * (max_val / 0.9), out_sr

    overlap_samples_in = int(OVERLAP_SECONDS * SAMPLE_RATE_IN)
    hop_samples = chunk_samples - overlap_samples_in
    starts = list(range(0, total_samples, hop_samples))
    stitched: torch.Tensor | None = None
    prev_end_in = 0

    for idx, start in enumerate(starts):
        end = min(start + chunk_samples, total_samples)
        chunk = wav_16k[:, start:end]
        max_val = chunk.abs().max().clamp_min(1e-6)
        chunk_norm = torch.nn.functional.pad(0.9 * chunk / max_val, (160, 160))
        pred = _separate_chunk(chunk_norm, num_steps, models, device)
        # Each chunk is normalised by its own peak, so two neighbours whose
        # loudness differs come back on different scales. Crossfading them then
        # blends two different gains and leaves a step at the seam. Restoring
        # each chunk's own scaling first puts them all back in the recording's
        # units, which is what makes the blend meaningful.
        pred = pred * (max_val / 0.9)
        target_out = max(1, round((end - start) * out_sr / SAMPLE_RATE_IN))
        if pred.shape[-1] > target_out:
            pred = pred[:, :target_out]
        elif pred.shape[-1] < target_out:
            pad = torch.zeros(2, target_out - pred.shape[-1], device=device)
            pred = torch.cat([pred, pad], dim=-1)

        if stitched is None:
            stitched = pred
            prev_end_in = end
            continue

        overlap_in = max(0, prev_end_in - start)
        overlap_out = max(0, min(round(overlap_in * out_sr / SAMPLE_RATE_IN), stitched.shape[-1], pred.shape[-1]))
        if overlap_out > 0:
            pred, _ = _maybe_swap(stitched[:, -overlap_out:], pred, overlap_out)
            fade = torch.linspace(0.0, 1.0, overlap_out, device=device).unsqueeze(0)
            blended = stitched[:, -overlap_out:] * (1 - fade) + pred[:, :overlap_out] * fade
            stitched = torch.cat([stitched[:, :-overlap_out], blended, pred[:, overlap_out:]], dim=-1)
        else:
            stitched = torch.cat([stitched, pred], dim=-1)
        prev_end_in = end

    return stitched, out_sr
