"""NVIDIA NemotronLabs RE-USE (universal speech enhancement), run in a SEPARATE Python environment.

ANALYSIS-FIRST: the output changes the signal, so it goes through the verification gates before it is
exported. Facts to keep in mind (from the model card, nothing here was run against the real model):
  - 9.6 M parameters, bidirectional Mamba, 8-48 kHz mono; trained on about 2500 h of synthetic single-speaker
    speech (English, German, Spanish, French, Mandarin). No Vietnamese, no conversation data.
  - with overlapping speech the model aims to keep the main speaker and suppress the others: do NOT run it
    on a stretch where two people talk at once if the overlap is to be kept (use single-speaker audio, or
    each speaker's own channel after separation).
  - the card says nothing about hallucination: compare ASR text and speaker similarity before trusting it.
  - needs CUDA (no CPU path upstream) and mamba_ssm, which conflicts with the pipeline env: it runs through
    models/denoise/workers/re_use_worker.py in its own env (models.denoise.subproc). This module never
    imports torch.
  - license of the weights: OpenMDW 1.1 (NemotronLabs release, commercial use stated on the card). The older
    nvidia/RE-USE release is NVIDIA's one-way non-commercial licence and is NOT what this backend targets.
  - `re_use_laughter` is the companion checkpoint that is meant to keep laughter better.

Config: python (env REUSE_PYTHON, interpreter of the separate env), repo_dir (env REUSE_DIR, the downloaded
Hugging Face folder), recipe (default recipes/USEMamba_30x1_..._early_001.yaml in repo_dir), hf_repo (used only
when repo_dir has no model.safetensors), chunk_s (default 5, as upstream's long-file script), hop_portion
(default 0.5), timeout_s (default 1800), device, seed, worker (script override, used by tests),
delay_compensation (default "auto": the STFT is centred so no lag is expected).
Setup: scripts/denoise_re_use.md. UNTESTED against the real model.
"""
import os
from typing import Optional, Tuple

import numpy as np

from . import subproc
from .base import DenoiseBackend, DenoiseError

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers", "re_use_worker.py")
REQUIRED_MODULES = ("torch", "torchaudio", "mamba_ssm", "librosa", "huggingface_hub", "yaml", "numpy")
DEFAULT_RECIPE = os.path.join("recipes", "USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml")
MODEL_FILE = os.path.join("models", "generator_SEMamba_time_d4.py")


class ReUseBackend(DenoiseBackend):
    name = "re_use"
    native_sr = 24000          # the rate the checkpoint runs at (base.enhance resamples 16 kHz pipeline audio up and back); the model accepts 8-48 kHz
    default_chunk_s = 5.0      # upstream's long-file script uses 5 s chunks, hop 0.5
    default_overlap_s = 1.0    # unused: the worker overlaps by hop_portion
    default_delay = "auto"
    dir_env = "REUSE_DIR"
    hf_repo = "nvidia/NVIDIA-NemotronLabs-RE-USE"

    @classmethod
    def _python(cls, cfg: dict) -> str:
        return str(cfg.get("python") or os.environ.get("REUSE_PYTHON") or "")

    @classmethod
    def _repo_dir(cls, cfg: dict) -> str:
        return str(cfg.get("repo_dir") or os.environ.get(cls.dir_env) or "")

    @classmethod
    def check_available(cls, cfg: Optional[dict] = None) -> Tuple[bool, str]:
        cfg = cfg or {}
        py = cls._python(cfg)
        if not py:
            return False, "REUSE_PYTHON is not set (python of the env that has mamba_ssm, see scripts/denoise_re_use.md)"
        if cfg.get("worker"):  # custom worker script (tests): it brings its own model
            if not os.path.isfile(str(cfg["worker"])):
                return False, f"worker script not found: {cfg['worker']}"
            return subproc.check_python_modules(py, cfg.get("modules") or ("numpy",))
        repo = cls._repo_dir(cfg)
        if not repo:
            return False, f"{cls.dir_env} is not set (download the model with huggingface-cli and point {cls.dir_env} at the folder)"
        if not os.path.isfile(os.path.join(repo, MODEL_FILE)):
            return False, f"{cls.dir_env}={repo} does not look like a RE-USE download ({MODEL_FILE} missing)"
        if not os.path.isfile(os.path.join(repo, "model.safetensors")) and not cfg.get("hf_repo"):
            return False, f"model.safetensors not found in {repo}"
        if not os.path.isfile(str(cfg.get("recipe") or os.path.join(repo, DEFAULT_RECIPE))):
            return False, f"recipe yaml not found in {repo}/recipes"
        return subproc.check_python_modules(py, cfg.get("modules") or REQUIRED_MODULES)

    def _load(self):
        ok, why = self.check_available(self.cfg)
        if not ok:
            raise DenoiseError(f"{self.name}: {why}")

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        cfg = dict(self.cfg)
        cfg.update(chunk_s=self.chunk_s, hop_portion=float(self.cfg.get("hop_portion", 0.5)), device=self.device,
                   seed=self.seed, repo_dir=self._repo_dir(self.cfg), hf_repo=self.cfg.get("hf_repo") or self.hf_repo)
        return subproc.run_worker(self._python(self.cfg), str(self.cfg.get("worker") or WORKER), x, self.native_sr, cfg,
                                  timeout=float(self.cfg.get("timeout_s", 1800)))


class ReUseLaughterBackend(ReUseBackend):
    """The companion checkpoint that is meant to preserve laughter (separate download, REUSE_LAUGHTER_DIR)."""
    name = "re_use_laughter"
    dir_env = "REUSE_LAUGHTER_DIR"
    hf_repo = "nvidia/NVIDIA-NemotronLabs-RE-USE-laughter"
