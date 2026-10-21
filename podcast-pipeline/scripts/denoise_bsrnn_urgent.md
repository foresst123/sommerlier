# BSRNN-26 backend (`bsrnn26`)

URGENT 2026 (ICASSP 2026) Track 1 discriminative baseline: band-split RNN, 37.8 M parameters (read off
the weights), complex mask + residual on a 20 ms / 10 ms STFT, 8-48 kHz. Code Apache-2.0
(urgent-challenge/urgent2026_challenge_track1), weights `lichenda/icassp_2026_urgent_baseline`
tagged `license:mit`. Trained on the "700h-TBF" curated subset (LibriVox, LibriTTS, VCTK, EARS,
MLS-HQ, Common Voice 19) with simulated noise, reverb, clipping, bandwidth limitation, codec,
packet loss and wind. Languages EN/DE/ES/FR/ZH. NO Vietnamese. Listen to Vietnamese output and
check WER before using it for training audio. It is a predictive mask model (no vocoder, no
sampling), so it does not hallucinate content, but it also trades low SDR gains (about 10-11 dB on the
challenge validation set) for that; it is a reference baseline, not a challenge winner.

## Setup

```bash
pip install -r requirements-denoise-bsrnn.txt
pip install --no-deps espnet==202412
git clone https://github.com/urgent-challenge/urgent2026_challenge_track1 ~/urgent2026_challenge_track1   # b1dc3ad
export URGENT26_DIR=~/urgent2026_challenge_track1
mkdir -p ~/urgent26_ckpt && curl -L -o ~/urgent26_ckpt/bsrnn.ckpt \
  https://huggingface.co/lichenda/icassp_2026_urgent_baseline/resolve/main/bsrnn.ckpt
export URGENT26_BSRNN_CKPT=~/urgent26_ckpt/bsrnn.ckpt
```

`bsrnn.ckpt` is 151,456,890 bytes (sha256 of the copy checked: 5d6b24eb...431c8). Do not use
`flow_bsrnn.ckpt` (generative; the backend refuses it).

## Behaviour

- Rates: 8, 16, 24, 32 and 48 kHz run natively (24 kHz is not resampled). Others go to the next
  supported rate and back (44.1 -> 48 kHz, 22.05 -> 24 kHz); above 48 kHz is downsampled.
- Output length equals input length; measured lag 0 (centred STFT).
- The model output level is arbitrary (scale-invariant training loss). The backend multiplies each
  chunk by the least-squares gain <x,y>/<y,y> (clamped at 0), so speech returns to its input level
  and silence stays silent. Upstream instead rescales to a 0.9 peak.
- Config keys: `repo_dir`, `checkpoint`, `chunk_s` (20), `overlap_s` (2), `device`, `seed`.
- Checkpoint loading uses a restricted unpickler (the file stores training objects); only torch,
  collections and numpy globals are allowed.
