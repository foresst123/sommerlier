# Denoise model weights

The noise-filter stage (`--denoise dfn3 | mf2se48 | mpsenet`, or chains such as `dfn3>mf2se48`)
needs the packages in `requirements-denoise.txt` and the weights below. Nothing is downloaded
automatically unless a backend's `auto_download` config key is set. Check what is usable with:

```bash
cd podcast-pipeline
python -c "from models.denoise import available_backends as a; [print(k, v) for k, v in a().items()]"
```

Weights live under `$SOMMELIER_DENOISE_DIR` (default `~/.cache/sommelier/denoise`) unless the
per-model variable below says otherwise.

## dfn3 (DeepFilterNet3, 48 kHz)

- Source: `models/DeepFilterNet3.zip` in https://github.com/Rikorose/DeepFilterNet (about 7.6 MB;
  holds `config.ini` and `checkpoints/`).
- Default location: df's cache folder, `<user cache dir>/DeepFilterNet/DeepFilterNet3` (on Linux
  `~/.cache/DeepFilterNet/DeepFilterNet3`). Override with `DFN_MODEL_DIR=<folder with config.ini>`.
- Get it:
  ```bash
  pip install -r requirements-denoise.txt
  python -c "from df.enhance import maybe_download_model; print(maybe_download_model('DeepFilterNet3'))"
  ```
  (`maybe_download_model` fetches from GitHub; this is also what `auto_download: true` does at load.)

## mf2se48 (ClearerVoice-Studio MossFormer2_SE_48K, 48 kHz)

- Source: HuggingFace `alibabasglab/MossFormer2_SE_48K` (a few hundred MB).
- Location: folder containing the text file `last_best_checkpoint` plus the checkpoint it names.
  Default `$SOMMELIER_DENOISE_DIR/MossFormer2_SE_48K`; override with `MF2SE48_DIR`.
- Get it:
  ```bash
  huggingface-cli download alibabasglab/MossFormer2_SE_48K --local-dir ~/.cache/sommelier/denoise/MossFormer2_SE_48K
  ```
  (`auto_download: true` makes clearvoice run the same `snapshot_download` into that folder on load.)
- Inference is in-process on arrays through `clearvoice.utils.decode.decode_one_audio`; no temp wav files.

## mpsenet (MP-SENet, 16 kHz)

- Source: https://github.com/yxlu-0102/MP-SENet. Not a pip package; the checkpoints are in the repo
  (`best_ckpt/g_best_vb`, VoiceBank+DEMAND, and `best_ckpt/g_best_dns`, about 8.7 MB each, next to the
  shared `best_ckpt/config.json`, sampling_rate 16000). Verified in the clone: both files exist. The
  upstream README documents only g_best_vb; it does not say what g_best_dns was trained on (the name
  suggests the DNS Challenge, unconfirmed).
- Get it: `git clone --depth 1 https://github.com/yxlu-0102/MP-SENet <somewhere>` then
  `export MPSENET_DIR=<somewhere>`.
- Optional: `MPSENET_CKPT=<file>` to use another checkpoint (its `config.json` is read from the same
  folder, as upstream does); config key `ckpt_name: g_best_dns` picks that model from `best_ckpt/`, or use the registry alias
  `mpsenet_dns` (same backend, ckpt_name defaults to g_best_dns).

## Per-backend config keys

All backends accept `chunk_s`, `overlap_s` (at most `chunk_s / 4`), `device`, `seed`,
`delay_compensation`, `max_delay_ms`. Defaults: dfn3 60 s / 2 s, mf2se48 15 s / 1 s (must stay under
20 s), mpsenet 10 s / 1 s. Backend-specific keys are listed in each class docstring in
`models/denoise/`. Pass them as `{"chunk_s": 20, "dfn3": {"chunk_s": 60}}`: top-level keys are shared,
the sub-dict named after a backend wins for that backend.
