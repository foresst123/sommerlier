import argparse
import json
import os
from utils.logger import Logger
def parse_args():
    parser = argparse.ArgumentParser(description="Sommelier ASR Pipeline")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--job_id", default="default", help="Job ID for checkpointing")
    parser.add_argument("--cache_dir", default="cache", help="Cache directory")
    parser.add_argument("--save_path", default="./output", help="Output directory")
    parser.add_argument("--gpu_1", default=0, type=int, help="GPU for VAD/Diarization/Separation/Whisper")
    parser.add_argument("--gpu_2", default=1, type=int, help="GPU for PhoWhisper/Sortformer/Qwen3")
    parser.add_argument("--lang", default="vi", help="Language code")
    parser.add_argument("--ASRMoE", action="store_true", help="Enable MoE ASR")
    parser.add_argument("--dia3", action="store_true", help="Use Pyannote community model (default is Sortformer if false)")
    parser.add_argument("--tse", action="store_true", help="Enable Target Speaker Extraction (TSE) for overlapping speech")
    parser.add_argument("--panns", action="store_true", help="Enable background music removal")
    parser.add_argument("--qwen3omni", action="store_true", help="Enable Qwen3-Omni audio captioning")
    parser.add_argument("--llm_refinement", action="store_true", help="Enable LLM label refinement")
    parser.add_argument("--sortformer_pad_onset", default=0.0, type=float, help="Sortformer start padding")
    parser.add_argument("--sortformer_pad_offset", default=0.0, type=float, help="Sortformer end padding")
    parser.add_argument("--vad", action="store_true", help="Enable VAD")
    parser.add_argument("--merge_gap", default=2.0, type=float, help="Gap threshold for merging segments")
    parser.add_argument("--seg_th", default=0.11, type=float, help="Segmentation threshold")
    parser.add_argument("--min_cluster_size", default=11, type=int, help="Minimum cluster size")
    parser.add_argument("--clust_th", default=0.5, type=float, help="Clustering threshold")
    parser.add_argument("--LLM", default="case_0", type=str, help="LLM refinement case")
    parser.add_argument("--initprompt", action="store_true", help="Use initial prompt for LLM")
    parser.add_argument("--env", default="kaggle", type=str, help="Environment profile name in config.json")
    return parser.parse_args()

# ==========================================
# 1. EARLY ENVIRONMENT SETUP (PRE-IMPORT)
# ==========================================
args = parse_args()

with open(args.config, 'r', encoding='utf-8') as f:
    config = json.load(f)
    
env_profile = config.get("environments", {}).get(args.env, {})

if env_profile.get("offline_mode", False):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    custom_offline_dir = env_profile.get("offline_weights_dir", "./offline_weights")
    if custom_offline_dir.startswith("./"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        offline_dir = os.path.join(base_dir, custom_offline_dir.replace("./", "", 1))
    else:
        offline_dir = custom_offline_dir
        
    os.environ["HF_HOME"] = os.path.join(offline_dir, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(offline_dir, "torch")
    os.environ["XDG_CACHE_HOME"] = offline_dir
    os.environ["HOME"] = offline_dir  # For PANNS
    os.environ["SRCORRNET_PATH"] = os.path.join(offline_dir, "srcorrnet")
    print(f"[*] Running in Offline Mode (env: {args.env}). Using weights from: {offline_dir}")
    
if env_profile.get("use_bf16", False):
    os.environ["SOMMELIER_USE_BF16"] = "1"
    print(f"[*] bfloat16 enabled via config for env: {args.env}")

# ==========================================
# 2. DELAYED IMPORTS
# ==========================================
from services.model_loader import ModelLoader
from services.audio_service import AudioService
from services.diarization_service import DiarizationService
from services.separation_service import TargetExtractionService
from services.music_service import MusicService
from services.asr_service import ASRService
from services.caption_service import CaptionService
from services.diarization_refinement_service import DiarizationRefinementService
from services.export_service import ExportService
from services.pipeline_service import PipelineService
from services.qwen3_worker_service import Qwen3WorkerService

def main():
        
    logger = Logger.get_logger()
    logger.info(f"Starting Sommelier Pipeline for Job: {args.job_id}")

    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        logger.info(f"Only 1 GPU detected. Overriding gpu_2 ({args.gpu_2}) to use gpu_1 ({args.gpu_1}).")
        args.gpu_2 = args.gpu_1



    # 1. Start Qwen3 Worker (if MoE enabled)
    qwen3_service = None
    if args.ASRMoE:
        qwen3_env_bin = os.environ.get("QWEN3_PYTHON")
        if not qwen3_env_bin:
            # Fallback checks: try ../ and ../../
            if os.path.exists("../../qwen3_env/bin/python"):
                qwen3_env_bin = "../../qwen3_env/bin/python"
            else:
                qwen3_env_bin = "../qwen3_env/bin/python"
        qwen3_worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_worker.py")
        qwen3_service = Qwen3WorkerService(qwen3_env_bin, qwen3_worker_script, device_id=args.gpu_2, logger=logger)
        qwen3_service.start()

    try:
        # 2. Load Models
        model_loader = ModelLoader(config, args, logger=logger)
        model_loader.load_base_models()
        model_loader.load_diarization_models()
        model_loader.load_separation_models()
        model_loader.load_music_models()
        model_loader.load_asr_models(qwen3_service)
        model_loader.load_caption_model()

        # 3. Initialize Services
        audio_svc = AudioService(logger=logger)
        diarization_svc = DiarizationService(
            diarizer=model_loader.get("diarizer"),
            vad_model=model_loader.get("vad"),
            embedder=model_loader.get("embedder"),
            logger=logger
        )
        separation_svc = TargetExtractionService(
            tse_model=model_loader.get("separator"),
            logger=logger
        )
        music_svc = MusicService(
            panns_model=model_loader.get("panns"),
            demucs_model=model_loader.get("demucs"),
            logger=logger
        )
        asr_svc = ASRService(
            whisper=model_loader.get("whisper"),
            phowhisper=model_loader.get("phowhisper"),
            qwen3=model_loader.get("qwen3"),
            logger=logger,
            model_loader=model_loader,
            qwen3_service=qwen3_service
        )
        caption_svc = CaptionService(
            captioner=model_loader.get("captioner"),
            logger=logger
        )
        refinement_svc = DiarizationRefinementService(logger=logger)
        export_svc = ExportService(logger=logger)

        # 4. Orchestrate via PipelineService
        pipeline = PipelineService(
            audio_svc, diarization_svc, separation_svc, music_svc, 
            asr_svc, caption_svc, refinement_svc, export_svc, logger=logger,
            model_loader=model_loader
        )
        
        logger.info(f"Running pipeline on audio: {args.audio}")
        pipeline.run(args, config, args.audio)

    finally:
        if qwen3_service:
            qwen3_service.stop()
        logger.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()
