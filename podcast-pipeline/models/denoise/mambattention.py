"""MambAttention (NikolaiKyhne/MambAttention, TASLP 2026), 16 kHz, run in a SEPARATE Python environment.

ANALYSIS-FIRST: run it on noisy Vietnamese files and measure tone preservation (F0 contour
correlation, ASR CER pre vs post) before using its output for anything. Facts to keep in mind:
  - trained on English only (DNS 2020, about 3000 h; or VB-DemandEx); no Vietnamese or tonal-language
    evidence anywhere in the paper or repo. "Keeps all tones" is unsupported.
  - native rate is 16 kHz, so for a 24 kHz source the 8-12 kHz band is lost unless restored afterwards.
  - needs CUDA, torch 2.2.2 and the vendored mamba_ssm 1.2.2 built from source, which conflicts with
    this repo's torch 2.8: it runs through models/denoise/workers/mambattention_worker.py in its own env
    (models.denoise.subproc). This module never imports torch.
  - attention is quadratic in time (need_weights=True): keep chunks at 4-6 s.
The DNS checkpoint's STFT/model config is not published; the VB-DemandEx yaml is used (identical weight shapes).

Config: python (env MAMBATTENTION_PYTHON, interpreter of the separate env), repo_dir (env
MAMBATTENTION_DIR), checkpoint (default <repo_dir>/checkpoints/seed4321_DNS.pth), config_yaml, chunk_s
(default 4), overlap_s (default 0.5), timeout_s (default 1800), device, seed, worker (script override,
used by tests), delay_compensation (default "auto"; the STFT is centred so no lag is expected).
Setup: scripts/denoise_mambattention.md. UNTESTED against the real model.
"""
import os
from typing import Optional, Tuple

import numpy as np

from . import subproc
from .base import DenoiseBackend, DenoiseError

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers", "mambattention_worker.py")
REQUIRED_MODULES = ("torch", "mamba_ssm", "einops", "yaml", "numpy")


def _python(cfg: dict) -> str:
    return str(cfg.get("python") or os.environ.get("MAMBATTENTION_PYTHON") or "")


def _repo_dir(cfg: dict) -> str:
    return str(cfg.get("repo_dir") or os.environ.get("MAMBATTENTION_DIR") or "")


def _ckpt_path(cfg: dict) -> str:
    return str(cfg.get("checkpoint") or os.path.join(_repo_dir(cfg), "checkpoints", "seed4321_DNS.pth"))


class MambAttentionBackend(DenoiseBackend):
    name = "mambattention"
    native_sr = 16000
    default_chunk_s = 4.0
    default_overlap_s = 0.5
    default_delay = "auto"

    @classmethod
    def check_available(cls, cfg: Optional[dict] = None) -> Tuple[bool, str]:
        cfg = cfg or {}
        py = _python(cfg)
        if not py:
            return False, "MAMBATTENTION_PYTHON is not set (python of the torch 2.2.2 + mamba_ssm env, see scripts/denoise_mambattention.md)"
        if cfg.get("worker"):  # custom worker script (tests): it brings its own model
            if not os.path.isfile(str(cfg["worker"])):
                return False, f"worker script not found: {cfg['worker']}"
            return subproc.check_python_modules(py, cfg.get("modules") or ("numpy",))
        repo = _repo_dir(cfg)
        if not repo:
            return False, "MAMBATTENTION_DIR is not set (clone NikolaiKyhne/MambAttention and point MAMBATTENTION_DIR at it)"
        if not os.path.isfile(os.path.join(repo, "models", "generator.py")):
            return False, f"MAMBATTENTION_DIR={repo} does not look like a MambAttention clone (models/generator.py missing)"
        ckpt = _ckpt_path(cfg)
        if not os.path.isfile(ckpt):
            return False, f"checkpoint not found: {ckpt}"
        if not os.path.isfile(str(cfg.get("config_yaml") or os.path.join(repo, "checkpoints", "MambAttention_seed3441_VB-DemandEx.yaml"))):
            return False, "model yaml not found (checkpoints/MambAttention_seed3441_VB-DemandEx.yaml)"
        return subproc.check_python_modules(py, cfg.get("modules") or REQUIRED_MODULES)

    def _load(self):
        ok, why = self.check_available(self.cfg)
        if not ok:
            raise DenoiseError(f"{self.name}: {why}")

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        cfg = dict(self.cfg)
        cfg.update(chunk_s=self.chunk_s, overlap_s=self.overlap_s, device=self.device, seed=self.seed,
                   repo_dir=_repo_dir(self.cfg), checkpoint=_ckpt_path(self.cfg))
        return subproc.run_worker(_python(self.cfg), str(self.cfg.get("worker") or WORKER), x, self.native_sr, cfg,
                                  timeout=float(self.cfg.get("timeout_s", 1800)))
