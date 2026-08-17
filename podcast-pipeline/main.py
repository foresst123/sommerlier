import argparse
import json
import os
from utils.logger import Logger
from services.model_loader import ModelLoader
from services.audio_service import AudioService
from services.diarization_service import DiarizationService
from services.separation_service import SeparationService
from services.music_service import MusicService
from services.asr_service import ASRService
from services.caption_service import CaptionService
from services.diarization_refinement_service import DiarizationRefinementService
from services.export_service import ExportService
from services.pipeline_service import PipelineService
from services.qwen3_worker_service import Qwen3WorkerService

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
    parser.add_argument("--srcorrnet", action="store_true", help="Enable SR-CorrNet overlapping speech separation")
    parser.add_argument("--panns", action="store_true", help="Enable background music removal")
    parser.add_argument("--qwen3omni", action="store_true", help="Enable Qwen3-Omni audio captioning")
    parser.add_argument("--llm_refinement", action="store_true", help="Enable LLM label refinement")
    parser.add_argument("--sortformer_pad_onset", default=0.0, type=float, help="Sortformer start padding")
    parser.add_argument("--sortformer_pad_offset", default=0.0, type=float, help="Sortformer end padding")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = Logger(logging_level="DEBUG").get_logger()
    logger.info(f"Starting Sommelier Pipeline for Job: {args.job_id}")

    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        logger.info(f"Only 1 GPU detected. Overriding gpu_2 ({args.gpu_2}) to use gpu_1 ({args.gpu_1}).")
        args.gpu_2 = args.gpu_1

    with open(args.config, 'r') as f:
        config = json.load(f)

    # 1. Start Qwen3 Worker (if MoE enabled)
    qwen3_service = None
    if args.ASRMoE:
        qwen3_env_bin = os.environ.get("QWEN3_PYTHON", "../qwen3_env/bin/python")
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
        separation_svc = SeparationService(
            separator=model_loader.get("separator"),
            embedder=model_loader.get("embedder"),
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
            logger=logger
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
            asr_svc, caption_svc, refinement_svc, export_svc, logger=logger
        )
        
        logger.info(f"Running pipeline on audio: {args.audio}")
        pipeline.run(args, config, args.audio)

    finally:
        if qwen3_service:
            qwen3_service.stop()
        logger.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()
