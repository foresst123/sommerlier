# RE-USE denoise backend (separate environment)

Backends `re_use` and `re_use_laughter` (`--denoise re_use`). ANALYSIS-FIRST: nothing here was run against the
real model. NVIDIA NemotronLabs RE-USE (release field 2026-10-22, version v1, OpenMDW 1.1, commercial use stated
on the model card): 9.6 M parameters, bidirectional Mamba, 8-48 kHz mono. Trained on about 2500 h of synthetic
single-speaker speech (en, de, es, fr, zh); no Vietnamese and no conversation data. With overlapping speech it
keeps the main speaker and suppresses the others, so do not use it on stretches whose overlap you want to keep.

The older `nvidia/RE-USE` release (2026-03-18) is under NVIDIA's non-commercial research licence. This backend
targets the NemotronLabs release.

It cannot share the pipeline env: it needs CUDA (the upstream script raises on CPU) and `mamba_ssm`. The backend
runs `models/denoise/workers/re_use_worker.py` with the interpreter of its own env (`models/denoise/subproc.py`).

## 1. Download (vendor code and weights are in the Hugging Face repo, 39.5 MB)

```bash
huggingface-cli download nvidia/NVIDIA-NemotronLabs-RE-USE --local-dir ~/NemotronLabs_REUSE --local-dir-use-symlinks False
ls ~/NemotronLabs_REUSE      # model.safetensors  config.json  models/  utils/  recipes/  inference.py ...
# optional laughter checkpoint (verify the repo name on the model card first)
huggingface-cli download nvidia/NVIDIA-NemotronLabs-RE-USE-laughter --local-dir ~/NemotronLabs_REUSE_laughter --local-dir-use-symlinks False
```

## 2. Environment

The model card points to the pre-built Docker environments of the SEMamba repository for the Mamba setup. Any
env that can run the card's own `sh inference.sh` will do. Check it:

```bash
~/venvs/reuse/bin/python -c "import torch, torchaudio, mamba_ssm, librosa, huggingface_hub, yaml; print(torch.__version__, torch.cuda.is_available())"
```

Run the vendor script once on a short file before involving the pipeline:

```bash
cd ~/NemotronLabs_REUSE && mkdir -p noisy_audio && cp /path/to/short.wav noisy_audio/
CUDA_VISIBLE_DEVICES=0 ~/venvs/reuse/bin/python inference.py --input_folder noisy_audio --output_folder enhanced_audio \
  --config recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml
```

Note: `inference.py` calls `hf_hub_download` at import, so the first run needs network (or a populated HF cache).

## 3. Pipeline-side settings

```bash
export REUSE_PYTHON=~/venvs/reuse/bin/python
export REUSE_DIR=~/NemotronLabs_REUSE
export REUSE_LAUGHTER_DIR=~/NemotronLabs_REUSE_laughter      # only for re_use_laughter
cd podcast-pipeline
python -c "from models.denoise import available_backends as a; r=a(); print(r['re_use'], r['re_use_laughter'])"
```

The pipeline env only needs numpy/scipy for this backend (no torch import). While these variables are unset,
`run_variants.py --only-available` skips the `re_use*` variants and says why.

Config keys: `chunk_s` (5, upstream's long-file setting; the model was trained on clips under 10 s),
`hop_portion` (0.5), `timeout_s` (1800), `device`, `recipe`, `hf_repo`, `delay_compensation` (auto).
The audio is processed at the pipeline's 24 kHz; the STFT sizes of the recipe (n_fft 320, hop 40 at 8 kHz) are
scaled to 960 / 120, as the vendor script does.

## 4. Smoke test of the worker alone

```bash
python - <<'PY'
import numpy as np, json, os
np.save("/tmp/in.npy", (0.1*np.random.randn(24000*12)).astype("float32"))
json.dump({"repo_dir": os.path.expanduser("~/NemotronLabs_REUSE")}, open("/tmp/cfg.json", "w"))
PY
PYTHONPATH=. $REUSE_PYTHON models/denoise/workers/re_use_worker.py --in /tmp/in.npy --out /tmp/out.npy --sr 24000 --cfg-json /tmp/cfg.json
```
stderr prints the measured lag; the base class removes it if there is one.

## 5. Before trusting the output

Same gates as the other analysis backends: ASR WER and insertion rate against the unfiltered audio, speaker
similarity, a listening pass on breaths, laughs and backchannels, and a check of every stretch where two people
talk at once. The model card does not discuss hallucination.
