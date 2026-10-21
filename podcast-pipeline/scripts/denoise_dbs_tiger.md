# DBS-TIGER backend (`dbs_tiger`), ANALYSIS-ONLY

Dialogue / music / sfx separator, 1.0 M parameters, 44.1 kHz mono, Apache-2.0 code and weights.
Trained only on DnR v2 (clean English audiobook speech, FMA, FSD50k). It removes music and sfx but
NOT crowd voices. Unknown until listened to: laughs, breaths, coughs, backchannels (may be cut as sfx),
Vietnamese speech, sung vocals, reverberant or overlapped speech. Do not export its dialogue stem as
training audio without a listening test and the verification gates.

## Setup (DGX)

```bash
pip install -r requirements-denoise-dbs-tiger.txt
git clone --depth 1 https://github.com/JusperLee/Dynamic-Band-Splitting ~/Dynamic-Band-Splitting   # checked at ff52646
export DBS_TIGER_DIR=~/Dynamic-Band-Splitting
# weights (4.1 MB): either download once ...
huggingface-cli download JusperLee/DBS-TIGER-DnR model.safetensors config.json --local-dir ~/dbs_tiger_ckpt
export DBS_TIGER_CKPT=~/dbs_tiger_ckpt/model.safetensors   # config.json beside it is used if present
# ... or pass auto_download=true in the backend config (saves under $SOMMELIER_DENOISE_DIR/dbs_tiger)
```

## Use

Backend name `dbs_tiger`. Config keys: `stem` (speech | music | sfx), `residual_gain` (0.0 = pure
dialogue; 0.25 keeps a quarter of music+sfx), `window_s` 6 / `hop_s` 3, `device`, `repo_dir`,
`checkpoint`, `auto_download`, `hf_repo`.

Listen to what it removes: run twice, once with `stem=speech` and once with `stem=sfx` (and `music`).
If soft non-lexical sounds disappear in the speech stem, try `residual_gain=0.25`.
