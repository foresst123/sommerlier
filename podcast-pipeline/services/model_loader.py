import os
import torch
from typing import Dict, Any

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
    """Orchestrates loading and unloading of models onto GPU/CPU."""
    
    def __init__(self, config: Dict[str, Any], args: Any, logger=None):
        self.config = config
        self.args = args
        self.logger = logger
        self.models = {}
        
        self.device_1 = torch.device(f"cuda:{args.gpu_1}" if torch.cuda.is_available() else "cpu")
        self.device_2 = torch.device(f"cuda:{args.gpu_2}" if torch.cuda.is_available() else "cpu")
        
    def load_base_models(self):
        """Load essential models (VAD, DNSMOS)."""
        if self.logger: self.logger.info(f"Loading Base models on {self.device_1}")
        self.models["vad"] = SileroVAD(device=self.device_1)
        
    def load_diarization_models(self, diarizen_service=None):
        """Load Pyannote/DiariZen based on args."""
        if self.args.dia3:
            if self.logger: self.logger.info(f"Loading Pyannote Diarization on {self.device_1}")
            self.models["diarizer"] = PyannoteDiarizer(
                token=self.config.get("huggingface_token", ""),
                device=self.device_1,
                use_community=True
            )
        else:
            if self.logger: self.logger.info(f"Connecting to DiariZen worker on {self.device_2}")
            self.models["diarizer"] = DiariZenDiarizer(process=diarizen_service.process if diarizen_service else None)
            
        # Embedder is needed for cross-chunk diarization fusion AND SR-CorrNet speaker identification
        if self.logger: self.logger.info(f"Loading Pyannote Embedder on {self.device_1}")
        self.models["embedder"] = PyannoteEmbedder(
            token=self.config.get("huggingface_token", ""),
            device=self.device_1
        )
            
    def load_separation_models(self):
        """Load Target Speaker Extractor (TSE) if enabled."""
        if getattr(self.args, "tse", False):
            if self.logger: self.logger.info(f"Loading Target Speaker Extractor on {self.device_1}")
            self.models["separator"] = TargetSpeakerExtractor(
                device=self.device_1
            )
            
    def load_music_models(self):
        """Load PANNS and Demucs if background music removal is enabled."""
        if getattr(self.args, "panns", False):
            if self.logger: self.logger.info("Loading PANNS detector")
            self.models["panns"] = PANNSDetector(device=str(self.device_1))
            self.models["demucs"] = DemucsRemover(device=str(self.device_1))
            
    def load_asr_models(self, qwen3_service: Qwen3WorkerService = None):
        """Load ASR models (Whisper, PhoWhisper, Qwen3)."""
        if self.logger: self.logger.info(f"Loading PhoWhisper on {self.device_1}")
        self.models["phowhisper"] = PhoWhisperASR(device=self.device_1)

        
        if getattr(self.args, "ASRMoE", False) and getattr(self.args, "lang", "vi") == "vi":
            if self.logger: self.logger.info(f"Loading Whisper on {self.device_1}")
            self.models["whisper"] = WhisperASR(
                model_size=getattr(self.args, "model_size", "large-v3-turbo"),
                device=self.device_1
            )
            if qwen3_service:
                if self.logger: self.logger.info("Connecting to Qwen3 worker")
                self.models["qwen3"] = Qwen3ASRClient(qwen3_service.process)
                
    def load_caption_model(self):
        """Load Omni caption client if enabled."""
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
