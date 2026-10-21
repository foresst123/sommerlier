"""MossFormer2_SE_48K from ClearerVoice-Studio (pip package `clearvoice`), 48 kHz."""
import os
from typing import Optional, Tuple

import numpy as np

from .base import DenoiseBackend, DenoiseError, has_module, missing_modules, weights_root

NETWORK = "MossFormer2_SE_48K"
# decode_one_audio switches to its own sliding-window path above this length (config
# one_time_decode_length); chunk_s must stay below it so every call takes the one-pass path.
ONE_PASS_LIMIT_S = 20.0


def _ckpt_dir(cfg: dict) -> str:
    return str(cfg.get("checkpoint_dir") or os.environ.get("MF2SE48_DIR") or os.path.join(weights_root(), NETWORK))


def _has_weights(path: str) -> bool:
    marker = os.path.join(path, "last_best_checkpoint")  # text file holding the checkpoint file name
    if not os.path.isfile(marker):
        return False
    with open(marker) as f:
        name = f.readline().strip()
    return bool(name) and os.path.isfile(os.path.join(path, name))


class MossFormer2SE48Backend(DenoiseBackend):
    """MossFormer2_SE_48K, run in-process on arrays (no temp files).

    The network object is built the way clearvoice.network_wrapper does it (load_args_se ->
    CLS_MossFormer2_SE_48K), then clearvoice.utils.decode.decode_one_audio is called on a
    (1, T) float array. The model is a spectral mask applied with a length-preserving STFT/ISTFT
    (centre=False, hamming 1920/384), so it adds no delay; upstream feeds this network
    un-normalised audio in [-1, 1] and so do we. torchaudio's fbank adds dither: the torch seed
    is reset before every chunk so runs are repeatable.

    Config: checkpoint_dir (folder holding last_best_checkpoint, env MF2SE48_DIR, default
    $SOMMELIER_DENOISE_DIR/MossFormer2_SE_48K), auto_download (let clearvoice fetch it from
    HuggingFace alibabasglab/MossFormer2_SE_48K when missing, default False), chunk_s (default 15,
    must be < 20), overlap_s (default 1), device (default auto), seed.
    Weights: see scripts/denoise_weights.md.
    """
    name = "mf2se48"
    native_sr = 48000
    default_chunk_s = 15.0
    default_overlap_s = 1.0
    default_delay = "auto"  # mask model: expect 0, but verify cheaply

    def __init__(self, cfg=None, logger=None):
        super().__init__(cfg, logger)
        if self.chunk_s <= 0 or self.chunk_s >= ONE_PASS_LIMIT_S:
            raise ValueError(f"{self.name}: chunk_s must be in (0, {ONE_PASS_LIMIT_S}) seconds, got {self.chunk_s}")

    @classmethod
    def check_available(cls, cfg: Optional[dict] = None) -> Tuple[bool, str]:
        cfg = cfg or {}
        miss = missing_modules("torch", "torchaudio", "clearvoice")
        if miss:
            return False, f"missing python package(s): {', '.join(miss)} (pip install -r requirements-denoise.txt)"
        path = _ckpt_dir(cfg)
        if not _has_weights(path) and not cfg.get("auto_download"):
            return False, f"weights not found in {path} (see scripts/denoise_weights.md, or set auto_download)"
        return True, ""

    def _load(self):
        ok, why = self.check_available(self.cfg)
        if not ok:
            raise DenoiseError(f"{self.name}: {why}")
        import torch
        from clearvoice.network_wrapper import network_wrapper
        from clearvoice.networks import CLS_MossFormer2_SE_48K

        path = _ckpt_dir(self.cfg)
        wrapper = network_wrapper()
        wrapper.model_name = NETWORK
        wrapper.load_args_se()  # reads clearvoice/config/inference/MossFormer2_SE_48K.yaml
        args = wrapper.args
        args.task, args.network, args.checkpoint_dir = "speech_enhancement", NETWORK, path
        self._seed_all()
        net = CLS_MossFormer2_SE_48K(args)  # downloads from HuggingFace only if last_best_checkpoint is absent
        if not _has_weights(path):
            raise DenoiseError(f"{self.name}: no usable checkpoint in {path}")
        if self.device != "auto":
            net.device = torch.device(self.device)
            net.model.to(net.device)
        self._net = net

    def _unload(self):
        self._net = None
        if has_module("torch"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        import torch
        from clearvoice.utils.decode import decode_one_audio

        net = self._net

        min_len = 4 * int(net.args.win_len)  # the centre=False STFT/fbank need a few frames

        def run(chunk: np.ndarray) -> np.ndarray:
            n = len(chunk)
            if n < min_len:
                chunk = np.concatenate([chunk, np.zeros(min_len - n, dtype=np.float32)])
            torch.manual_seed(self.seed)
            with torch.no_grad():
                out = decode_one_audio(net.model, net.device, chunk.astype(np.float32)[None, :], net.args)
            return np.asarray(out, dtype=np.float32).reshape(-1)[:n]

        return self._chunked(run, x)
