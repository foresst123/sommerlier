"""Generic `audio-separator` (nomadkaraoke, MIT) backend: any UVR/Roformer checkpoint as a denoise-chain stage.

STATUS: ANALYSIS-FIRST: changes the signal; output goes through the verification gates before being
exported. These are music/vocal/reverb separators, not speech enhancers. None was trained on Vietnamese
conversation; what each does to soft speech, breaths, laughs and backchannels is UNVERIFIED. Listen first.

How it works: the 24 kHz mono input is upsampled to the checkpoint's 44.1 kHz (base class; the band above
12 kHz is empty, so the model sees narrowband audio it was not trained on), duplicated to stereo, written as a
32-bit float WAV, separated by `Separator.separate`, and the kept stem is read back, folded to mono, trimmed or
zero-padded to the exact input length and resampled to the input rate. These models have no delay, so no
delay compensation is applied. One library detail matters: for these single-target Roformers the second
stem is `input - target` computed inside audio-separator (mdxc_separator.py), so "keep the non-crowd stem" and
"input minus the crowd stem" (`invert`) are the same signal.

Alias table (names verified against audio-separator 0.47.1 @465b60f, scratchpad clone, and UVR's
download_checks.json fetched 2026-10-21; stems from the model YAML `training.instruments`):

  alias            checkpoint                                               arch        kept stem  license status
  rf_vocals_3005   model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt        MelRoformer Vocals     viperx, none stated (UVR code MIT, weights unstated)
  rf_vocals_kim    mel_band_roformer_kim_ft2_bleedless_unwa.ckpt            MelRoformer vocals     unwa, none stated
  rf_denoise       denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt        MelRoformer dry        aufr33, no model card, none stated
  rf_denoise_aggr  denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt   MelRoformer dry        aufr33, no model card, none stated
  rf_dereverb      deverb_bs_roformer_8_384dim_10depth.ckpt                 BS-Roformer noreverb   listed as "BS-Roformer-De-Reverb", author not
                                                                                                   stated in models.json; the deton24 doc links
                                                                                                   huggingface.co/anvuew/deverb_bs_roformer, whose
                                                                                                   license is GPL-3.0 (UNVERIFIED here): treat as GPL risk
  uvr_deecho       UVR-DeEcho-DeReverb.pth                                  VR (v5)     no reverb  FoxJoy via UVR, none stated
  rf_crowd         mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt    MelRoformer other      aufr33/viperx, none stated

  what each is for
  rf_vocals_3005   vocal stem of a music mix (the a100 profile's music-stage checkpoint); as a chain step it
                   removes music/background, e.g. `rf_vocals_3005>dfn3`.
  rf_vocals_kim    same job, Kim FT2 "bleedless" variant (kaggle profile); stems are vocals/other.
  rf_denoise       general denoiser; the model's "dry" stem is the cleaned signal ("other" is the removed noise).
  rf_denoise_aggr  stronger variant of the same; higher chance of eating soft consonants.
  rf_dereverb      keeps the dry ("noreverb") stem; reverb tails are removed.
  uvr_deecho       VR-architecture de-echo/de-reverb; keeps "No Reverb". Handled by the same Separator API
                   (model type VR, config found by file hash), but alignment and level are UNTESTED.
  rf_crowd         built to remove audience cheering/applause; the speech is in the NON-crowd stem ("other").
                   UNVERIFIED whether it removes talking bystanders (no model card, no test on speech).

Evidence (file:line in the audio-separator clone): models.json:37 / :40 / :46 / :49 / :61 list the denoise,
aggr, crowd, deverb and kim-ft2-bleedless checkpoints with their YAML; ep_3005 and UVR-DeEcho-DeReverb.pth are
not in models.json but in UVR's remote download_checks.json (roformer_download_list and vr_download_list) that
Separator.list_supported_model_files downloads (separator.py:666-670) and in models-scores.json:45918 / :11568.
Stems: models-scores.json:11568+ (no reverb/reverb), YAML instruments (dry/other, noreverb/reverb, crowd/other,
Vocals/Instrumental, vocals/other).

Config (all optional except as noted):
  checkpoint          file name inside model_file_dir (required for the generic `audio_separator`; aliases preset it)
  stem_keep           name or ordered list of stem names to keep; the first one that exists in the output wins.
                      Matching ignores case, spaces and punctuation and is EXACT on the stem name parsed from
                      "<input>_(<Stem>)_<model>.wav" ("noreverb" does not match "reverb"); if nothing matches exactly
                      a unique substring match is tried.
  invert              false; true returns input minus the matched stem (for models whose useful output is the residual)
  model_file_dir      folder holding the checkpoints (env AUDIO_SEPARATOR_MODEL_DIR, then BS_ROFORMER_MODEL_DIR).
                      The library itself lets AUDIO_SEPARATOR_MODEL_DIR override the argument; this backend sets
                      it for the duration of the load so the configured folder wins.
  auto_download       false: a missing file makes check_available return (False, reason). true lets the library
                      fetch checkpoint, YAML and its lists (needs network)
  chunk_duration      seconds; the library splits long files itself (default None: off). chunk_s/overlap_s of the
                      base class (default 0 = off) add a crossfaded wrapper-level chunking if wanted
  segment_size, overlap, batch_size   passed as mdxc_params (partial dict is fine: arch_config.get(...) with the
                      checkpoint's own inference.num_overlap as fallback, mdxc_separator.py:100-135). overlap is the
                      number of overlapping windows (step = chunk // overlap); batch_size is ignored by Roformers
  native_fp16         false; CUDA only (mutually exclusive with autocast, which this backend never enables)
  normalization_threshold  1.0 (library default 0.9 would squash loud stems). The library scales a stem down only
                      when its peak exceeds this; the input is pre-scaled here the same way and the gain undone
  device              "auto" | "cpu" | "cuda" | "cuda:1". audio-separator picks CUDA when present; "cpu" cannot be
                      forced through its API (a warning is logged); "cuda:N" selects the device via torch.cuda.device
Setup: scripts/denoise_audio_separator.md.
"""
import contextlib
import inspect
import os
import re
import shutil
import tempfile
from typing import List, Optional, Tuple

import numpy as np

from .base import DenoiseBackend, DenoiseError, has_module, missing_modules
from .chunking import fit_length

NATIVE_SR = 44100
SILENT_IN = 1e-3    # input peak above which a silent output counts as a failure
SILENT_OUT = 1e-5
_STEM = re.compile(r"^in_\((.+?)\)_.*\.[A-Za-z0-9]+$")
_LIST_FILES = ("download_checks.json",)                    # read by every load_model
_HASH_FILES = ("vr_model_data.json", "mdx_model_data.json")  # read for models without a YAML (VR/MDX)


def _import_separator():
    """Seam for tests: the audio-separator Separator class (imports torch lazily)."""
    import torch
    try:
        torch.set_num_threads(max(16, os.cpu_count() or 32))
    except Exception:
        pass
    try:
        import samplerate, torchaudio
        if not getattr(samplerate, "_gpu_patched", False):
            _orig_resample = samplerate.resample
            def _fast_resample(data, ratio, converter_type="sinc_best", *a, **kw):
                try:
                    x = np.asarray(data, dtype=np.float32)
                    orig_sr = 44100
                    new_sr = int(round(orig_sr * float(ratio)))
                    if new_sr == orig_sr:
                        return x
                    t = torch.from_numpy(x.T if x.ndim == 2 else x).contiguous()
                    if torch.cuda.is_available():
                        t = t.cuda()
                    out = torchaudio.functional.resample(t, orig_sr, new_sr).cpu().numpy()
                    return out.T if x.ndim == 2 else out
                except Exception:
                    return _orig_resample(data, ratio, converter_type, *a, **kw)
            samplerate.resample = _fast_resample
            samplerate._gpu_patched = True
    except Exception:
        pass
    from audio_separator.separator import Separator
    return Separator


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_stem(path: str) -> str:
    """Stem name of an audio-separator output file ("in_(No Reverb)_model.wav" -> "No Reverb"), "" if unparsable."""
    m = _STEM.match(os.path.basename(path))
    return m.group(1) if m else ""


def pick_stem(paths: List[str], keep: List[str]) -> Tuple[Optional[str], List[str]]:
    """(path of the kept stem or None, parsed stem names). First name in `keep` wins; exact match, then a unique substring."""
    stems = [parse_stem(p) for p in paths]
    for want in map(_norm, keep):
        exact = [p for p, s in zip(paths, stems) if _norm(s) == want]
        if exact:
            return exact[0], stems
        sub = [p for p, s in zip(paths, stems) if want and want in _norm(s)]
        if len(sub) == 1:
            return sub[0], stems
    return None, stems


def _model_dir(cfg: dict) -> str:
    raw = cfg.get("model_file_dir") or os.environ.get("AUDIO_SEPARATOR_MODEL_DIR") or os.environ.get("BS_ROFORMER_MODEL_DIR") or ""
    return os.path.expanduser(os.path.expandvars(str(raw))) if raw else ""


class AudioSeparatorBackend(DenoiseBackend):
    """Keep one stem (or the residual) of any audio-separator checkpoint. ANALYSIS-FIRST: see the module docstring."""
    name = "audio_separator"
    native_sr = NATIVE_SR
    default_chunk_s = 0.0
    default_overlap_s = 0.0
    # alias defaults, overridable per config
    checkpoint = ""
    config_file = ""            # YAML next to the checkpoint (Roformer/MDXC); "" for VR
    stem_keep: Tuple[str, ...] = ()
    invert = False

    def __init__(self, cfg: Optional[dict] = None, logger=None):
        super().__init__(cfg, logger)
        c = self.cfg
        self.checkpoint_name = str(c.get("checkpoint") or self.checkpoint)
        keep = c.get("stem_keep", self.stem_keep)
        self.keep = [str(keep)] if isinstance(keep, str) else [str(k) for k in keep]
        self.do_invert = bool(c.get("invert", self.invert))
        if not self.keep:
            raise ValueError(f"{self.name}: stem_keep is required (a stem name or a list of names)")
        self.norm_threshold = float(c.get("normalization_threshold", 1.0))
        if not 0 < self.norm_threshold <= 1:
            raise ValueError(f"{self.name}: normalization_threshold must be in (0, 1], got {self.norm_threshold}")
        self.native_fp16 = bool(c.get("native_fp16", False))
        self.chunk_duration = c.get("chunk_duration")
        self._sep = None
        self._work = None

    # ---------------------------------------------------------------- registry
    @classmethod
    def check_available(cls, cfg: Optional[dict] = None) -> Tuple[bool, str]:
        cfg = cfg or {}
        miss = missing_modules("torch", "soundfile", "audio_separator")
        if miss:
            return False, f"missing python package(s): {', '.join(miss)} (pip install -r requirements-denoise-audio-separator.txt)"
        ckpt = str(cfg.get("checkpoint") or cls.checkpoint)
        if not ckpt:
            return False, "no checkpoint configured (set `checkpoint` to a file name in model_file_dir)"
        if cfg.get("auto_download"):
            return True, ""
        folder = _model_dir(cfg)
        if not folder:
            return False, "no model folder: set AUDIO_SEPARATOR_MODEL_DIR (or BS_ROFORMER_MODEL_DIR / model_file_dir) or pass auto_download=true"
        if not os.path.isdir(folder):
            return False, f"model folder does not exist: {folder}"
        yaml = str(cfg.get("config_file") or cls.config_file)
        # .ckpt models read a YAML (named when known); VR/MDX models (.pth/.onnx) look their parameters up by file hash
        extra = [yaml] if yaml else ([] if ckpt.endswith(".ckpt") else list(_HASH_FILES))
        need = [ckpt] + extra + list(_LIST_FILES)
        absent = [f for f in need if not os.path.isfile(os.path.join(folder, f))]
        if absent:
            return False, (f"missing in {folder}: {', '.join(absent)} (see scripts/denoise_audio_separator.md; "
                           "or pass auto_download=true, which needs network)")
        return True, ""

    # ------------------------------------------------------------- lifecycle
    def _use_gpu(self) -> bool:
        if self.device.startswith("cuda"):
            return True
        if self.device == "auto" and has_module("torch"):
            import torch
            return bool(torch.cuda.is_available())
        return False

    def _device_ctx(self):
        if self.device.startswith("cuda") and has_module("torch"):
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device(self.device)
        return contextlib.nullcontext()

    def _separator_kwargs(self, Separator, work: str, folder: str) -> dict:
        gpu = self._use_gpu()
        wanted = {
            "output_dir": work,
            "output_format": "WAV",
            "use_autocast": bool(self.cfg.get("use_autocast", True)) and gpu and not (self.native_fp16 and gpu),
            "use_native_fp16": self.native_fp16 and gpu,
            "log_level": 40,
            "normalization_threshold": self.norm_threshold,
        }
        if folder:
            wanted["model_file_dir"] = folder
        if self.chunk_duration is not None:
            wanted["chunk_duration"] = float(self.chunk_duration)
        mdxc = {"overlap": int(self.cfg.get("overlap", 2)), "batch_size": int(self.cfg.get("batch_size", 4))}
        for key, cast in (("segment_size", int), ("overlap", int), ("batch_size", int)):
            if self.cfg.get(key) is not None:
                mdxc[key] = cast(self.cfg[key])
        if "segment_size" in mdxc:
            mdxc["override_model_segment_size"] = bool(self.cfg.get("override_model_segment_size", False))
        if mdxc:
            wanted["mdxc_params"] = mdxc
        accepted = inspect.signature(Separator.__init__).parameters
        kwargs = {k: v for k, v in wanted.items() if k in accepted}
        dropped = sorted(set(wanted) - set(kwargs) - {"log_level"})
        if dropped:
            self._log(f"installed audio-separator does not accept {dropped}; using its defaults", "warning")
        return kwargs

    def _load(self):
        ok, why = self.check_available(self.cfg)
        if not ok:
            raise DenoiseError(f"{self.name}: {why}")
        if self.device == "cpu" and self._use_gpu_available():
            self._log("audio-separator chooses CUDA when present; device='cpu' cannot be forced", "warning")
        Separator = _import_separator()
        folder = _model_dir(self.cfg)
        self._work = tempfile.mkdtemp(prefix="denoise_as_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        saved = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR")
        try:
            if folder:  # the library lets this env var override the argument: make the configured folder win
                os.environ["AUDIO_SEPARATOR_MODEL_DIR"] = folder
            self._seed_all()
            with self._device_ctx():
                sep = Separator(**self._separator_kwargs(Separator, self._work, folder))
                sep.load_model(model_filename=self.checkpoint_name)
        except Exception as exc:
            shutil.rmtree(self._work, ignore_errors=True)
            self._work = None
            raise DenoiseError(f"{self.name}: could not load {self.checkpoint_name} ({type(exc).__name__}: {exc})") from exc
        finally:
            if saved is None:
                os.environ.pop("AUDIO_SEPARATOR_MODEL_DIR", None)
            else:
                os.environ["AUDIO_SEPARATOR_MODEL_DIR"] = saved
        self._sep = sep

    @staticmethod
    def _use_gpu_available() -> bool:
        if not has_module("torch"):
            return False
        import torch
        return bool(torch.cuda.is_available())

    def _unload(self):
        self._sep = None
        if self._work:
            shutil.rmtree(self._work, ignore_errors=True)
            self._work = None
        if has_module("torch"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------- inference
    def _clean_work(self):
        for entry in os.listdir(self._work):
            path = os.path.join(self._work, entry)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    os.remove(path)

    def _separate_array(self, x: np.ndarray) -> np.ndarray:
        """One mono 44.1 kHz array -> kept stem (or residual), same length, mono, float32."""
        import soundfile as sf

        n = len(x)
        peak = float(np.max(np.abs(x))) if n else 0.0
        gain = self.norm_threshold / peak if peak > self.norm_threshold else 1.0  # same rule as the library
        scaled = (x * np.float32(gain)).astype(np.float32)
        in_path = os.path.join(self._work, "in.wav")
        try:
            sf.write(in_path, np.stack([scaled, scaled], axis=1), NATIVE_SR, subtype="FLOAT")
            with self._device_ctx():
                outputs = self._sep.separate(in_path)
            paths = [p if os.path.isabs(p) else os.path.join(self._work, p) for p in (outputs or [])]
            chosen, stems = pick_stem(paths, self.keep)
            if chosen is None:
                raise DenoiseError(f"{self.name}: no output stem matches {self.keep}; the model produced {stems or 'no files'}")
            stem, sr = sf.read(chosen, dtype="float32", always_2d=True)
        finally:
            self._clean_work()
        if sr != NATIVE_SR:
            raise DenoiseError(f"{self.name}: separator wrote {sr} Hz, expected {NATIVE_SR}")
        mono = stem.mean(axis=1)
        if abs(len(mono) - n) > max(1024, n // 100):
            raise DenoiseError(f"{self.name}: separator returned {len(mono)} samples for {n}")
        mono = fit_length(mono, n)
        out = (scaled - mono) if self.do_invert else mono
        return (out / np.float32(gain)).astype(np.float32)

    def _enhance_native(self, x: np.ndarray) -> np.ndarray:
        y = self._chunked(self._separate_array, x)
        if not np.all(np.isfinite(y)):
            raise DenoiseError(f"{self.name}: model produced non-finite samples")
        if float(np.max(np.abs(x))) > SILENT_IN and float(np.max(np.abs(y))) < SILENT_OUT:
            raise DenoiseError(f"{self.name}: output is silent although the input has signal "
                               f"(kept stem {self.keep}{', inverted' if self.do_invert else ''}); wrong stem or a model that removes speech")
        return y


# ------------------------------------------------------------------- aliases

class RFVocals3005Backend(AudioSeparatorBackend):
    """Mel-Roformer viperx ep_3005 (SDR 11.436), vocals stem; the a100 profile's music-stage checkpoint."""
    name = "rf_vocals_3005"
    checkpoint = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
    config_file = "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"
    stem_keep = ("vocals",)


class RFVocalsKimBackend(AudioSeparatorBackend):
    """Kim Mel-Roformer FT2 bleedless (unwa), vocals stem; the kaggle profile's checkpoint."""
    name = "rf_vocals_kim"
    checkpoint = "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt"
    config_file = "config_mel_band_roformer_kim_ft_unwa.yaml"
    stem_keep = ("vocals",)


class RFDenoiseBackend(AudioSeparatorBackend):
    """aufr33 Mel-Roformer denoise (SDR 27.9959); the cleaned signal is the 'dry' stem."""
    name = "rf_denoise"
    checkpoint = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
    config_file = "denoise_mel_band_roformer_aufr33_sdr_27.9959_config.yaml"
    stem_keep = ("dry",)


class RFDenoiseAggrBackend(AudioSeparatorBackend):
    """aufr33 Mel-Roformer denoise, aggressive variant (SDR 27.9768); 'dry' stem."""
    name = "rf_denoise_aggr"
    checkpoint = "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt"
    config_file = "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768_config.yaml"
    stem_keep = ("dry",)


class RFDereverbBackend(AudioSeparatorBackend):
    """BS-Roformer de-reverb (8 / 384 dim / 10 depth); 'noreverb' stem. Author not stated by audio-separator; GPL-3.0 risk if anvuew's (unverified)."""
    name = "rf_dereverb"
    checkpoint = "deverb_bs_roformer_8_384dim_10depth.ckpt"
    config_file = "deverb_bs_roformer_8_384dim_10depth_config.yaml"
    stem_keep = ("noreverb",)


class UVRDeEchoBackend(AudioSeparatorBackend):
    """UVR VR-architecture DeEcho-DeReverb; 'No Reverb' stem. Config comes from the file hash (UVR model data), not a YAML."""
    name = "uvr_deecho"
    checkpoint = "UVR-DeEcho-DeReverb.pth"
    config_file = ""
    stem_keep = ("noreverb",)


class RFCrowdBackend(AudioSeparatorBackend):
    """aufr33/viperx Mel-Roformer de-crowd (SDR 8.7144). Speech is in the NON-crowd ('other') stem. Removes cheering/applause; talking bystanders UNVERIFIED."""
    name = "rf_crowd"
    checkpoint = "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
    config_file = "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144_config.yaml"
    stem_keep = ("other",)


class RFVocalsResurrectionBackend(AudioSeparatorBackend):
    """unwa BS-Roformer Vocals Resurrection; 'vocals' stem."""
    name = "rf_vocals_resurrection"
    checkpoint = "bs_roformer_vocals_resurrection_unwa.ckpt"
    config_file = "config_bs_roformer_vocals_resurrection_unwa.yaml"
    stem_keep = ("vocals", "other")


class RFDereverbAnvuewBackend(AudioSeparatorBackend):
    """anvuew Mel-Band Roformer de-reverb (SDR 19.1729); 'noreverb' stem."""
    name = "rf_dereverb_anvuew"
    checkpoint = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"
    config_file = "dereverb_mel_band_roformer_anvuew.yaml"
    stem_keep = ("noreverb",)


class RFDeEchoV2Backend(AudioSeparatorBackend):
    """Sucial Mel-Band Roformer de-reverb-echo V2 (SDR 13.4843); 'noreverb' stem."""
    name = "rf_deecho_v2"
    checkpoint = "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt"
    config_file = "config_dereverb-echo_mel_band_roformer_sdr_13.4843_v2.yaml"
    stem_keep = ("noreverb", "no reverb", "dry")


ALIASES = (RFVocals3005Backend, RFVocalsResurrectionBackend, RFVocalsKimBackend, RFDenoiseBackend, RFDenoiseAggrBackend,
           RFDereverbBackend, RFDereverbAnvuewBackend, RFDeEchoV2Backend, UVRDeEchoBackend, RFCrowdBackend)
