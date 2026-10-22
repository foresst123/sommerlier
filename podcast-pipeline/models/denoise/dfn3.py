"""DeepFilterNet3 (Rikorose/DeepFilterNet, pip package `deepfilternet`), 48 kHz."""
import os
import sys
from typing import Optional, Tuple

import numpy as np

from .base import DenoiseBackend, DenoiseError, has_module, missing_modules

MODEL_NAME = "DeepFilterNet3"


def _cache_dir() -> str:
    # Same location as df.utils.get_cache_dir(), without importing df.
    if has_module("appdirs"):
        from appdirs import user_cache_dir
        return user_cache_dir("DeepFilterNet")
    return os.path.expanduser("~/.cache/DeepFilterNet/")


def _model_dir(cfg: dict) -> str:
    return str(cfg.get("model_dir") or os.environ.get("DFN_MODEL_DIR") or os.path.join(_cache_dir(), MODEL_NAME))


def _has_weights(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.ini")) and os.path.isdir(os.path.join(path, "checkpoints"))


class DeepFilterNet3Backend(DenoiseBackend):
    """DeepFilterNet3 through df.enhance.init_df / df.enhance.enhance, 48 kHz.

    Delay: enhance(..., pad=True) pads n_fft zeros and crops d = n_fft - hop_size samples off the
    front, so the output is already sample aligned and the same length (checked in df/enhance.py);
    delay_compensation therefore defaults to False.

    Config: model_dir (folder with config.ini and checkpoints/, env DFN_MODEL_DIR, default the df
    cache folder for DeepFilterNet3), auto_download (let df fetch the model from GitHub when it is
    missing, default False), post_filter (default False), atten_lim_db (default None = no limit),
    chunk_s (default 60), overlap_s (default 2), device (default auto: cuda if available), seed.
    Weights: see scripts/denoise_weights.md.
    """
    name = "dfn3"
    native_sr = 48000
    default_chunk_s = 60.0
    default_overlap_s = 2.0

    @classmethod
    def check_available(cls, cfg: Optional[dict] = None) -> Tuple[bool, str]:
        cfg = cfg or {}
        miss = missing_modules("torch", "df")
        if miss:
            return False, f"missing python package(s): {', '.join(miss)} (pip install -r requirements-denoise.txt)"
        path = _model_dir(cfg)
        if not _has_weights(path) and not cfg.get("auto_download"):
            return False, f"weights not found at {path} (see scripts/denoise_weights.md, or set auto_download)"
        return True, ""

    def _load(self):
        ok, why = self.check_available(self.cfg)
        if not ok:
            raise DenoiseError(f"{self.name}: {why}")
        import torch
        from df.config import config as df_config
        from df.enhance import init_df

        path = _model_dir(self.cfg)
        if not _has_weights(path):
            path = MODEL_NAME  # auto_download: init_df fetches the pretrained model by name
        self._seed_all()
        try:
            # log_file=None keeps df from writing enhance.log into the model folder
            # 0.5.6 returns (model, df_state, suffix); the repo's main adds a fourth value (epoch)
            model, df_state = init_df(path, post_filter=bool(self.cfg.get("post_filter", False)),
                                      log_level="WARNING", log_file=None)[:2]
        except SystemExit:  # init_df calls exit(1) when the checkpoint cannot be found
            raise DenoiseError(f"{self.name}: DeepFilterNet could not load a checkpoint from {path}")
        if self.device != "auto":
            df_config.set("DEVICE", self.device, str, "train")  # read by df.utils.get_device()
            model = model.to(torch.device(self.device))
        self._model, self._df_state = model.eval(), df_state
        if df_state.sr() != self.native_sr:
            raise DenoiseError(f"{self.name}: expected a 48 kHz model, got {df_state.sr()} Hz")

    def _unload(self):
        self._model = self._df_state = None
        if has_module("torch"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        import torch
        from df.enhance import enhance

        atten = self.cfg.get("atten_lim_db")

        def run(chunk: np.ndarray) -> np.ndarray:
            audio = torch.from_numpy(np.array(chunk, dtype=np.float32, copy=True)).unsqueeze(0)  # [1, T]
            out = enhance(self._model, self._df_state, audio, pad=True, atten_lim_db=atten)
            return out.squeeze(0).detach().cpu().numpy()

        return self._chunked(run, x)
