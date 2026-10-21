# MambAttention denoise backend (separate environment)

Backend name `mambattention` (`--denoise mambattention`). ANALYSIS-FIRST: English-only training (DNS 2020,
about 3000 h), no Vietnamese or tone evidence, 16 kHz so 8-12 kHz of 24 kHz sources is lost. Run it on noisy
Vietnamese files and measure tone preservation (F0 contour correlation, ASR CER before/after) before using
its output for anything.

It cannot share the pipeline env: it needs CUDA, `torch==2.2.2` and the vendored `mamba_ssm` 1.2.2 built from
source, while the pipeline uses torch 2.8. The backend therefore runs
`models/denoise/workers/mambattention_worker.py` with the interpreter of its own env (`models/denoise/subproc.py`).
Everything below is UNTESTED (no CUDA/mamba build was available when this was written).

## 1. Code and checkpoint (both are in the git repo, MIT)

```bash
git clone https://github.com/NikolaiKyhne/MambAttention ~/MambAttention      # verified at commit b0fec7c
ls ~/MambAttention/checkpoints   # seed4321_DNS.pth (DNS 2020, used by default), seed3441.pth (VB-DemandEx), *.yaml
```

The DNS checkpoint has the same tensor shapes as the VB-DemandEx one; the yaml of the latter is used for both
(the DNS STFT config is not published, so n_fft 400 / hop 100 / win 400 / compress 0.3 is an assumption).

## 2. Environment (A100 node, CUDA 12.x with nvcc)

```bash
nvcc --version                                  # must exist; export CUDA_HOME=/usr/local/cuda-12.x if needed
python3.10 -m venv ~/venvs/mambattention && source ~/venvs/mambattention/bin/activate
pip install -U pip wheel setuptools
pip install -r /path/to/podcast-pipeline/requirements-denoise-mambattention.txt
cd ~/MambAttention/mamba_install
TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=4 pip install --no-build-isolation .   # builds selective_scan_cuda
python -c "import torch, mamba_ssm; print(torch.__version__, torch.cuda.is_available())"
```

`causal_conv1d` is optional (slower fallback). If the build fails with the current nvcc, use a CUDA 12.1 toolkit
(or the `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel` image); do not try it in the pipeline's torch 2.8 env.

## 3. Pipeline-side settings

```bash
export MAMBATTENTION_PYTHON=~/venvs/mambattention/bin/python
export MAMBATTENTION_DIR=~/MambAttention
cd podcast-pipeline
python -c "from models.denoise import available_backends as a; print(a()['mambattention'])"
```

The pipeline env only needs numpy/scipy for this backend (no torch import). Config keys: `checkpoint`
(default `seed4321_DNS.pth`), `chunk_s` (4; attention memory is quadratic, about 2.6 GB per layer-call at 4 s,
about 16 GB at 10 s, estimated), `overlap_s` (0.5), `timeout_s` (1800), `device`, `config_yaml`.

## 4. Smoke test of the worker alone

```bash
python - <<'PY'
import numpy as np; np.save("/tmp/in.npy", (0.1*np.random.randn(16000*10)).astype("float32"))
import json; json.dump({"repo_dir": "$HOME/MambAttention"}, open("/tmp/cfg.json", "w"))
PY
PYTHONPATH=. $MAMBATTENTION_PYTHON models/denoise/workers/mambattention_worker.py --in /tmp/in.npy --out /tmp/out.npy --sr 16000 --cfg-json /tmp/cfg.json
```
(Replace `$HOME` with the literal path in the JSON.) stderr prints the measured lag; the base class removes it
when `delay_compensation` is "auto" (the STFT is centred, so 0 is expected).
