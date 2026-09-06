import os
import torch
from typing import Dict, Any

from utils.steps import step_enabled

from models.whisper_wrapper import WhisperASR
from models.phowhisper import PhoWhisperASR
from models.silero_vad import SileroVAD
from models.pyannote import PyannoteDiarizer
from models.diarizen_model import DiariZenDiarizer
from models.pyannote_embedding import PyannoteEmbedder
# from models.sortformer import SortformerDiarizer
from models.tse_model import TargetSpeakerExtractor
from models.panns import PANNSDetector
from models.demucs import DemucsRemover
from models.qwen3_omni import Qwen3OmniCaptioner
from models.qwen3_asr import Qwen3ASRClient
from services.qwen3_worker_service import Qwen3WorkerService

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
        
        self.device_1 = torch.device(f"cuda:{args.gpu_1}" if torch.cuda.is_available() else "cpu")
        self.device_2 = torch.device(f"cuda:{args.gpu_2}" if torch.cuda.is_available() else "cpu")
        
    def load_base_models(self):
        """Load essential models (VAD, DNSMOS)."""
        if "vad" in self.models:
            return
        if self.logger: self.logger.info(f"Loading Base models on {self.device_1}")
        vad_cfg = self.config.get("environments", {}).get(
            self.args.env, {}).get("models", {}).get("vad", {})
        self.models["vad"] = SileroVAD(device=self.device_1, **vad_cfg)
        
    def load_diarization_models(self, diarizen_service=None):
        """Load Pyannote/DiariZen based on args."""
        if "diarizer" in self.models and "embedder" in self.models:
            return
        if self.args.dia3:
            if self.logger: self.logger.info(f"Loading Pyannote Diarization on {self.device_1}")
            self.models["diarizer"] = PyannoteDiarizer(
                token=self.config.get("huggingface_token", ""),
                device=self.device_1,
                use_community=True
            )
        else:
            # main.py spawns the DiariZen worker with device_id=args.gpu_1, so
            # device_1 is where it actually lives. The old message said device_2.
            if self.logger: self.logger.info(f"Connecting to DiariZen worker on {self.device_1}")
            self.models["diarizer"] = DiariZenDiarizer(process=diarizen_service.process if diarizen_service else None)
            
        # Embedder is needed for cross-chunk diarization fusion AND SR-CorrNet speaker identification
        if self.logger: self.logger.info(f"Loading Pyannote Embedder on {self.device_1}")
        self.models["embedder"] = PyannoteEmbedder(
            token=self.config.get("huggingface_token", ""),
            device=self.device_1
        )
            
    def load_separation_models(self, sidon_service=None):
        """Load Target Speaker Extractor (TSE) if enabled."""
        if "separator" in self.models:
            return
        if getattr(self.args, "tse", False):
            if self.logger: self.logger.info(f"Loading Target Speaker Extractor on {self.device_1}")
            # Same resolution order the extractor uses: an explicit flag wins,
            # then the profile (published as TSE_SEPARATOR in main.py), then the
            # default. Resolved here too so the log line names what actually ran.
            separator = (getattr(self.args, "separator", None)
                         or os.environ.get("TSE_SEPARATOR") or "usef")
            if self.logger: self.logger.info(f"  separator backend: {separator}")
            self.models["separator"] = TargetSpeakerExtractor(
                device=self.device_1,
                process=sidon_service.process if sidon_service else None,
                separator=separator,
                logger=self.logger,
            )
            
    def load_panns(self):
        """Load just the music detector.

        Separate from load_music_models because the music sweep that runs after
        diarization needs the tagger and nothing else: pulling Demucs in with it
        would hold a source-separation model in VRAM from diarization all the
        way to music removal, for a check that never uses it.
        """
        if "panns" in self.models:
            return
        if step_enabled(self.args, "music_analysis"):
            if self.logger: self.logger.info("Loading PANNS detector")
            self.models["panns"] = PANNSDetector(device=str(self.device_1))

    def load_music_models(self):
        """Load PANNS and Demucs if background music removal is enabled."""
        # Keyed on demucs alone: panns may already be resident from the music
        # sweep, and testing it here would skip loading Demucs entirely.
        if "demucs" in self.models:
            return
        if step_enabled(self.args, "music_removal"):
            self.load_panns()

            # GPU 1 hosts the DiariZen worker plus the embedder, TSE and ASR
            # models; separating a full podcast needs ~1GB of headroom that is
            # not there, so Demucs runs on GPU 2 alongside the Qwen3 worker.
            demucs_cfg = dict(self.config.get("environments", {}).get(self.args.env, {})
                              .get("models", {}).get("demucs", {}))
            # Which model isolates vocals. Same interface either way, so
            # MusicService does not know the difference; the key stays "demucs"
            # because that is what the service and the free-list call it.
            #
            # `model` is popped rather than forwarded: the rest of the block is
            # constructor kwargs, and leaving it in would reach DemucsRemover as
            # an argument it does not take.
            # Popped unconditionally: the rest of the block is constructor
            # kwargs, and short-circuiting past the pop when --music_separator
            # was given left `model` in there to reach DemucsRemover, which
            # does not take it.
            configured = demucs_cfg.pop("model", None)
            which = (getattr(self.args, "music_separator", None) or configured
                     or os.environ.get("MUSIC_SEPARATOR") or "demucs")
            if which == "bs_roformer":
                from models.bs_roformer import BSRoformerRemover
                if self.logger: self.logger.info(f"Loading BS-RoFormer on {self.device_1}")
                self.models["demucs"] = BSRoformerRemover(
                    device=str(self.device_1), logger=self.logger, **demucs_cfg)
            else:
                if self.logger: self.logger.info(f"Loading Demucs on {self.device_1}")
                self.models["demucs"] = DemucsRemover(
                    device=str(self.device_1), logger=self.logger, **demucs_cfg)
            
    def load_asr_models(self, qwen3_service: Qwen3WorkerService = None):
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
        # hosts the DiariZen worker, the embedder, ECAPA, the Sidon worker and
        # Whisper; adding a second 3.1GB model there took the card to ~15.2GB
        # against a 14.56GB T4, and the first thing to ask for memory afterwards
        # was diarization. It failed on chunk 0 with DiariZen reporting
        # "batch_size (12) is probably too large" -- a misleading message, since
        # that batch needs under 1GB. The card was simply already full.
        if self.logger: self.logger.info(f"Loading PhoWhisper on {self.device_2}")
        pho_cfg = self.config.get("environments", {}).get(
            self.args.env, {}).get("models", {}).get("phowhisper", {})
        self.models["phowhisper"] = PhoWhisperASR(device=self.device_2, **pho_cfg)

        
        if getattr(self.args, "ASRMoE", False) and getattr(self.args, "lang", "vi") == "vi":
            if self.logger: self.logger.info(f"Loading Whisper on {self.device_1}")
            
            whisper_cfg = self.config.get("environments", {}).get(self.args.env, {}).get("models", {}).get("whisper", {})
            self.models["whisper"] = WhisperASR(
                device=self.device_1,
                **whisper_cfg
            )
            if qwen3_service:
                if self.logger: self.logger.info("Connecting to Qwen3 worker")
                self.models["qwen3"] = Qwen3ASRClient(qwen3_service.process)
                
    def load_caption_model(self):
        """Load Omni caption client if enabled."""
        if "captioner" in self.models:
            return
        if getattr(self.args, "qwen3omni", False):
            if self.logger: self.logger.info("Initializing Qwen3-Omni Client")
            self.models["captioner"] = Qwen3OmniCaptioner()
            
    def get(self, model_name: str):
        return self.models.get(model_name)
        
    def unload(self, model_name: str):
        """Unload model to free VRAM."""
        if model_name in self.models:
            del self.models[model_name]
            torch.cuda.empty_cache()
            if self.logger: self.logger.info(f"Unloaded {model_name} from VRAM")
