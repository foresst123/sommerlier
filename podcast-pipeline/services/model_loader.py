import functools
import os
import gc
import threading
import torch
from typing import Dict, Any

from utils.steps import step_enabled
from utils.performance_config import resolve_music_devices, resolve_music_worker_devices

from models.whisper_wrapper import WhisperASR
from models.whisper_vllm import WhisperVLLMClient
from utils.asr_model_config import (
    asr_device, phowhisper_kwargs, whisper_backend, whisper_ct2_kwargs)
from models.phowhisper import PhoWhisperASR
from models.silero_vad import SileroVAD
from models.pyannote import PyannoteDiarizer
from models.diarizen_model import DiariZenDiarizer
# from models.sortformer import SortformerDiarizer
from models.bss_model import BssSeparator
from models.sslam import SSLAMDetector
from models.qwen3_omni import Qwen3OmniCaptioner
from models.qwen3_asr import Qwen3ASRClient
from services.qwen3_worker_service import Qwen3WorkerService


def _serialized(method):
    """Run a loader method under the loader's lock.

    Every loader is idempotent, which is not the same as safe to call twice at
    once: with two files in flight, both can see `"tagger" not in self.models`
    and both build one. The second overwrites the first in the dict, so the
    first is never unloaded and keeps its VRAM until the process exits. An
    RLock, because load_music_models calls load_tagger.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # __init__ creates the lock, but tests build a loader through __new__
        # to skip device probing. setdefault is a single atomic dict operation,
        # so two callers racing to create it still end up with the same one.
        lock = (self.__dict__.get("_load_lock")
                or self.__dict__.setdefault("_load_lock", threading.RLock()))
        with lock:
            return method(self, *args, **kwargs)
    return wrapper


class ModelLoader:
    """Orchestrates loading and unloading of models onto GPU/CPU.

    Every loader is idempotent: calling it again when its models are already
    resident is a no-op. That is what lets PipelineService call the loader for
    a stage at the point the stage runs, rather than loading everything up
    front -- under stage-major execution the same loader is reached once per
    file, and only the first call should pay for it.
    """
    
    def __init__(self, config: Dict[str, Any], args: Any, logger=None):
        self.config = config
        self.args = args
        self.logger = logger
        self.models = {}
        self._load_lock = threading.RLock()
        # {"qwen3"|"whisper"|"phowhisper": gpu id}; None keeps the legacy layout
        # (PhoWhisper on device_1, Whisper on device_2). Set by main.py.
        self.asr_placement = None
        
        self.device_1 = torch.device(f"cuda:{args.gpu_1}" if torch.cuda.is_available() else "cpu")
        self.device_2 = torch.device(f"cuda:{args.gpu_2}" if torch.cuda.is_available() else "cpu")
        
    @_serialized
    def load_base_models(self):
        """Load essential models (VAD, DNSMOS)."""
        if "vad" in self.models:
            return
        if self.logger: self.logger.info(f"Loading Base models on {self.device_1}")
        vad_cfg = self.config.get("environments", {}).get(
            self.args.env, {}).get("models", {}).get("vad", {})
        self.models["vad"] = SileroVAD(device=self.device_1, **vad_cfg)
        
    @_serialized
    def load_diarization_models(self, diarizen_service=None):
        """Load Pyannote/DiariZen based on args."""
        if "diarizer" in self.models:
            return
        if self.args.dia3:
            if self.logger: self.logger.info(f"Loading Pyannote Diarization on {self.device_1}")
            self.models["diarizer"] = PyannoteDiarizer(
                token=self.config.get("huggingface_token", ""),
                device=self.device_1,
                use_community=True
            )
        else:
            if self.logger: self.logger.info("Connecting to DiariZen worker pool")
            endpoint = None
            if diarizen_service is not None:
                endpoint = (diarizen_service
                            if hasattr(diarizen_service, "request")
                            else diarizen_service.process)
            self.models["diarizer"] = DiariZenDiarizer(process=endpoint)
            
    @_serialized
    def load_separation_models(self, sidon_service=None):
        """Load the separator and its WeSpeaker assignment, if enabled.

        `sidon_service` is only consulted by the out-of-process backend; the
        in-process one never looks at it, so passing it unconditionally keeps
        the caller from having to know which separator the profile named.
        """
        if "separator" in self.models:
            return
        if getattr(self.args, "bss", False):
            # Same resolution order the extractor uses: an explicit flag wins,
            # then the profile (published as BSS_SEPARATOR in main.py), then the
            # default. Resolved here too so the log line names what actually ran.
            separator = (getattr(self.args, "separator", None)
                         or os.environ.get("BSS_SEPARATOR") or "sidon")
            bss_cfg = self.config.get("environments", {}).get(
                self.args.env, {}).get("models", {}).get("bss", {})
            perf = getattr(self.args, "performance_config", None) or {}
            sep_perf = perf.get("stages", {}).get("separation", {})
            postprocess_device = self.device_1
            if (perf.get("enabled", False)
                    and str(separator).strip().lower() == "sidon"
                    and sep_perf.get("postprocess_device", "cpu") == "cpu"):
                postprocess_device = torch.device("cpu")
            if self.logger:
                self.logger.info(
                    f"Loading separator assignment models on {postprocess_device}")
            if self.logger: self.logger.info(f"  separator backend: {separator}")
            endpoint = None
            if sidon_service is not None:
                endpoint = (sidon_service if hasattr(sidon_service, "request")
                            else sidon_service.process)
            self.models["separator"] = BssSeparator(
                device=postprocess_device,
                process=endpoint,
                separator=separator,
                embedding_repository=bss_cfg.get("embedding_repository"),
                embedding_filename=bss_cfg.get("embedding_filename"),
                embedding_revision=bss_cfg.get("embedding_revision"),
                logger=self.logger,
            )
            
    @_serialized
    def load_tagger(self):
        """Load just the frame-level tagger.

        Separate from load_music_models because the sweep that decides what is
        playing needs the tagger and nothing else: pulling the vocal separator
        in with it would hold it in VRAM across a check that never uses it, and
        on a recording with no music bed it would never be used at all.
        """
        if "tagger" in self.models:
            return
        if step_enabled(self.args, "music_analysis"):
            if self.logger: self.logger.info("Loading SSLAM tagger")
            self.models["tagger"] = SSLAMDetector(device=str(self.device_1))

    @_serialized
    def load_music_models(self):
        """Load the tagger and BS-RoFormer if music removal is enabled."""
        if "bs_roformer" in self.models:
            return
        if step_enabled(self.args, "music_removal"):
            self.load_tagger()

            # By default loaded onto device_1, the same card as SSLAM and the
            # DiariZen worker. What keeps them from colliding is time rather
            # than space: the music stage releases both before diarization
            # starts, so they are not resident together, and within one file
            # SSLAM must finish first anyway (it produces the music map that
            # is this model's job list).
            # With music.cross_file_overlap on, the sole instance moves to
            # device_2 instead (see resolve_music_devices), so file N's removal
            # and file N+1's SSLAM classification run on different cards.
            # Copied before popping: self.config is the live profile, and
            # load_music_models runs once per file in a batch.
            bs_roformer_cfg = dict(self.config.get("environments", {}).get(self.args.env, {})
                              .get("models", {}).get("bs_roformer", {}))
            # `model` names the checkpoint, and is the one key that is not a
            # constructor argument under its own name. Forwarding it would
            # reach BSRoformerRemover as an argument it does not take.
            checkpoint = bs_roformer_cfg.pop("model", None)
            # --music_separator overrides the profile, the profile overrides
            # the module default. Set here rather than through an env var so
            # one batch run cannot leak a checkpoint into the next.
            checkpoint = getattr(self.args, "music_separator", None) or checkpoint
            if checkpoint:
                bs_roformer_cfg["model_filename"] = checkpoint

            from models.bs_roformer import BSRoformerPool
            from models.bs_roformer_process import build_bs_roformers
            perf = (self.config.get("environments", {}).get(self.args.env, {})
                    .get("performance", {}))
            music_perf = perf.get("stages", {}).get("music", {})
            devices = resolve_music_devices(
                self.device_1, self.device_2,
                perf_enabled=perf.get("enabled", False),
                max_separator_workers=int(music_perf.get("max_separator_workers", 1)),
                cross_file_overlap=bool(music_perf.get("cross_file_overlap", False)),
                logger=self.logger)
            devices = resolve_music_worker_devices(
                devices, music_perf.get("workers_per_gpu", 1)
                if perf.get("enabled", False) else 1,
                bool(bs_roformer_cfg.get("isolate_process", False)), self.logger)
            if self.logger:
                self.logger.info(
                    f"Loading {len(devices)} BS-RoFormer worker(s) on "
                    f"{', '.join(map(str, devices))}")
            models = build_bs_roformers(devices, bs_roformer_cfg, self.logger)
            self.models["bs_roformer"] = (
                BSRoformerPool(models) if len(models) > 1 else models[0])
            
    @_serialized
    def load_asr_models(self, qwen3_service: Qwen3WorkerService = None,
                        whisper_service=None):
        """Load ASR models (Whisper, PhoWhisper, Qwen3)."""
        if "phowhisper" in self.models:
            return
        # --stop_after names the last stage to run, so anything that halts
        # before ASR must not pay for a 3GB model it will never call.
        stop_after = getattr(self.args, "stop_after", None)
        if stop_after in ("diarization", "separation", "music_removal"):
            if self.logger:
                self.logger.info(
                    f"Skipping ASR models (--stop_after {stop_after} runs before ASR)")
            return

        # PhoWhisper goes on GPU 2, not alongside everything else. GPU 1 already
        # hosts the DiariZen worker, the embedder, ECAPA, and
        # Whisper; adding a second 3.1GB model there took the card to ~15.2GB
        # against a 14.56GB T4, and the first thing to ask for memory afterwards
        # was diarization. It failed on chunk 0 with DiariZen reporting
        # "batch_size (12) is probably too large" -- a misleading message, since
        # that batch needs under 1GB. The card was simply already full.
        pho_device = self._asr_device("phowhisper", self.device_1)
        if self.logger: self.logger.info(f"Loading PhoWhisper on {pho_device}")
        self.models["phowhisper"] = self.make_phowhisper(pho_device)

        
        if getattr(self.args, "ASRMoE", False) and getattr(self.args, "lang", "vi") == "vi":
            models_cfg = self._models_cfg()
            whisper_cfg = dict(models_cfg.get("whisper", {}))
            if whisper_backend(models_cfg) == "vllm":
                if whisper_service is None:
                    raise RuntimeError(
                        "models.whisper.backend=vllm requires a Whisper vLLM worker")
                if self.logger:
                    self.logger.info("Connecting to Whisper large-v3 vLLM worker")
                self.models["whisper"] = WhisperVLLMClient(
                    whisper_service,
                    batch_size=whisper_cfg.get("batch_size", 16))
            else:
                whisper_device = self._asr_device("whisper", self.device_2)
                if self.logger:
                    self.logger.info(f"Loading Whisper on {whisper_device}")
                self.models["whisper"] = self.make_whisper_ct2(whisper_device)
            if qwen3_service:
                if self.logger: self.logger.info("Connecting to Qwen3 worker")
                self.models["qwen3"] = Qwen3ASRClient(qwen3_service.process)
                
    @_serialized
    def load_caption_model(self):
        """Load Omni caption client if enabled."""
        if "captioner" in self.models:
            return
        if getattr(self.args, "qwen3omni", False):
            if self.logger: self.logger.info("Initializing Qwen3-Omni Client")
            self.models["captioner"] = Qwen3OmniCaptioner()
            
    def _models_cfg(self) -> dict:
        return self.config.get("environments", {}).get(
            self.args.env, {}).get("models", {})

    def _asr_device(self, kind, default):
        return asr_device(self.asr_placement, kind, default, torch.cuda.is_available())

    def make_phowhisper(self, device, batch_size=None):
        """A PhoWhisper on `device`; a replica passes the boost batch size."""
        return PhoWhisperASR(
            device=device, **phowhisper_kwargs(self._models_cfg(), batch_size))

    def make_whisper_ct2(self, device, batch_size=None):
        """A CTranslate2 Whisper on `device`, without the vLLM-only settings."""
        return WhisperASR(
            device=device, **whisper_ct2_kwargs(self._models_cfg(), batch_size))

    def get(self, model_name: str):
        return self.models.get(model_name)
        
    @_serialized
    def unload(self, model_name: str):
        """Unload model to free VRAM."""
        if model_name in self.models:
            model = self.models.pop(model_name)
            cleanup = getattr(model, "unload", None) or getattr(model, "close", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:
                    if self.logger:
                        self.logger.warning(
                            f"Cleanup for {model_name} failed before unload: {exc}")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.logger: self.logger.info(f"Unloaded {model_name} from VRAM")
