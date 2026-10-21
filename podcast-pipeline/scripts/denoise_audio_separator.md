# audio_separator backend and aliases (ANALYSIS-FIRST)

Any `audio-separator` checkpoint as a denoise-chain stage, e.g. `rf_vocals_3005>dfn3`. These are
music/vocal/reverb separators, not speech enhancers: the output changes the signal and must pass the
verification gates before it is exported. None was tested on Vietnamese conversation.

| alias | checkpoint | kept stem | note |
|---|---|---|---|
| rf_vocals_3005 | model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt | Vocals | a100 profile music stage |
| rf_vocals_kim | mel_band_roformer_kim_ft2_bleedless_unwa.ckpt | vocals | kaggle profile |
| rf_denoise | denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt | dry | no model card |
| rf_denoise_aggr | denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt | dry | stronger, may eat soft consonants |
| rf_dereverb | deverb_bs_roformer_8_384dim_10depth.ckpt | noreverb | GPL-3.0 risk if anvuew's (unverified) |
| uvr_deecho | UVR-DeEcho-DeReverb.pth | No Reverb | VR arch; alignment/level UNTESTED |
| rf_crowd | mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt | other (non-crowd) | built for applause; talking bystanders UNVERIFIED |

No checkpoint has a stated license (UVR code is MIT, weights are not covered). The generic backend is
`audio_separator` with `checkpoint`, `stem_keep` (name or ordered list) and optional `invert`.

## Prepare checkpoints on the DGX (once, with network)

```bash
pip install -r requirements-denoise-audio-separator.txt
export AUDIO_SEPARATOR_MODEL_DIR=~/audio_separator_models      # or reuse BS_ROFORMER_MODEL_DIR
mkdir -p $AUDIO_SEPARATOR_MODEL_DIR
python - <<'PY'
import os
from audio_separator.separator import Separator
s = Separator(model_file_dir=os.environ["AUDIO_SEPARATOR_MODEL_DIR"])
for ckpt in ["model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
             "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"]:   # add the ones you want
    s.load_model(ckpt)          # downloads checkpoint + YAML (VR: + vr_model_data.json, mdx_model_data.json)
PY
ls $AUDIO_SEPARATOR_MODEL_DIR   # must contain download_checks.json (needed on every load, even offline)
```

Offline runs need `download_checks.json` beside the checkpoint: `load_model` reads it each time
(`Separator.list_supported_model_files`). `auto_download` defaults to false, so a missing file makes the
backend report unavailable instead of downloading.

## Notes

- Input is 24 kHz mono, upsampled to 44.1 kHz stereo; content above 12 kHz is empty.
- `normalization_threshold` defaults to 1.0 (the library's 0.9 would squash loud stems).
- `overlap` is the number of overlapping windows; `batch_size` is ignored by Roformers.
- `device="cpu"` cannot be forced through audio-separator; it uses CUDA when present.
- The second stem of these single-target models is `input - target`, so `stem_keep: other` for rf_crowd
  equals `stem_keep: crowd` with `invert: true`.
