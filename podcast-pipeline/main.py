import argparse
import json
import os
from utils.logger import Logger
def parse_args():
    parser = argparse.ArgumentParser(description="Sommelier ASR Pipeline")
    parser.add_argument("--audio", help="Path to a single input audio file")
    parser.add_argument("--audio_dir",
                        help="Directory of audio files to process in one run. Files are "
                             "grouped into batches under batch.max_hours_per_run from "
                             "config.json; each still writes its own output folder.")
    parser.add_argument("--max_hours", type=float,
                        help="Override batch.max_hours_per_run for this run.")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--job_id", default="default", help="Job ID for checkpointing")
    parser.add_argument("--cache_dir", default="cache", help="Cache directory")
    parser.add_argument("--save_path", default="./output", help="Output directory")
    parser.add_argument("--gpu_1", default=0, type=int, help="GPU for VAD/Diarization/Separation/Whisper")
    parser.add_argument("--gpu_2", default=1, type=int, help="GPU for PhoWhisper/Sortformer/Qwen3")
    parser.add_argument("--lang", default="vi", help="Language code")
    parser.add_argument("--ASRMoE", action="store_true", help="Enable MoE ASR")
    parser.add_argument("--dia3", action="store_true", help="Use Pyannote community model (default is DiariZen if false)")
    parser.add_argument("--tse", action="store_true", help="Enable Target Speaker Extraction (TSE) for overlapping speech")
    parser.add_argument("--panns", action="store_true", help="Enable background music removal")
    parser.add_argument("--qwen3omni", action="store_true", help="Enable Qwen3-Omni audio captioning")
    parser.add_argument("--llm_refinement", action="store_true", help="Enable LLM label refinement")
    parser.add_argument("--sortformer_pad_onset", default=0.0, type=float, help="Sortformer start padding")
    parser.add_argument("--sortformer_pad_offset", default=0.0, type=float, help="Sortformer end padding")
    parser.add_argument("--vad", action="store_true", help="Enable VAD")
    parser.add_argument("--LLM", default="case_0", type=str, help="LLM refinement case")
    parser.add_argument("--initprompt", action="store_true", help="Use initial prompt for LLM")
    parser.add_argument("--env", default="kaggle", type=str, help="Environment profile name in config.json")
    parser.add_argument("--by_stage", action="store_true",
                        help="Run each stage across the whole batch before the next "
                             "stage, instead of the whole pipeline per file. Loads "
                             "each model once per batch rather than once per file.")
    parser.add_argument("--only_batch", type=int, default=None,
                        help="Run only this batch number (1-based) and exit. Lets a "
                             "directory larger than max_hours be finished across "
                             "several sessions without exceeding a runtime limit.")
    parser.add_argument("--keep_models", action="store_true",
                        help="Keep models in VRAM between stages instead of unloading them. "
                             "Saves reload time when processing many files, at the cost of a "
                             "higher peak: only use it when the GPU has room for every model at once.")
    parser.add_argument("--no_stage_output", action="store_true",
                        help="Skip the per-stage artifact directories (01_diarization/, "
                             "02_separation/, ...). They are written by default so a run "
                             "stopped or crashed part-way still leaves its finished work "
                             "on disk.")
    parser.add_argument("--stop_after", type=str, choices=["diarization", "separation", "music_removal", "asr", "captioning"], help="Stop pipeline gracefully after this stage")
    return parser.parse_args()

# ==========================================
# 1. EARLY ENVIRONMENT SETUP (PRE-IMPORT)
# ==========================================
args = parse_args()

if not args.audio and not args.audio_dir:
    parser_error = "one of --audio or --audio_dir is required"
    raise SystemExit(f"main.py: error: {parser_error}")

with open(args.config, 'r', encoding='utf-8') as f:
    config = json.load(f)

# --- Fetch HuggingFace Token from Environment or Kaggle Secrets ---
hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
        if not hf_token:
            hf_token = user_secrets.get_secret("HF_TOKEN")
    except Exception:
        pass

if hf_token:
    config["huggingface_token"] = hf_token
# ------------------------------------------------------------------

env_profile = config.get("environments", {}).get(args.env, {})

# Inject config parameters into args dynamically
pipeline_cfg = env_profile.get("pipeline", {})
for k, v in pipeline_cfg.items():
    setattr(args, k, v)
    
if "gpu_1" in env_profile:
    args.gpu_1 = env_profile["gpu_1"]
if "gpu_2" in env_profile:
    args.gpu_2 = env_profile["gpu_2"]

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
    os.environ["TSE_PATH"] = os.path.join(offline_dir, "tse_model")
    print(f"[*] Running in Offline Mode (env: {args.env}). Using weights from: {offline_dir}")
    
if env_profile.get("use_bf16", False):
    os.environ["SOMMELIER_USE_BF16"] = "1"
    print(f"[*] bfloat16 enabled via config for env: {args.env}")

# ==========================================
# 2. DELAYED IMPORTS
# ==========================================
from utils.torch_compat import install_torch_load_shim

# Must run before any model module imports torch and loads a checkpoint.
install_torch_load_shim()

from utils.batch import audio_duration, find_audio_files, find_name_collisions, plan_batches, run_batch_by_stage
from utils.worker_env import resolve_worker_python
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
from services.diarizen_worker_service import DiarizenWorkerService
from services.sidon_worker_service import SidonWorkerService

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
        qwen3_env_bin = resolve_worker_python("qwen3", config=config, env_profile=env_profile, logger=logger)
        qwen3_worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_worker.py")
        qwen3_service = Qwen3WorkerService(qwen3_env_bin, qwen3_worker_script, device_id=args.gpu_2, logger=logger, env_name=args.env, config_path=args.config)
        qwen3_service.spawn()

    # 1b. Start DiariZen Worker (if dia3 is not used)
    diarizen_service = None
    if not args.dia3:
        diarizen_env_bin = resolve_worker_python("diarizen", config=config, env_profile=env_profile, logger=logger)
        diarizen_worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diarizen_worker.py")
        diarizen_service = DiarizenWorkerService(diarizen_env_bin, diarizen_worker_script, device_id=args.gpu_1, logger=logger, env_name=args.env, config_path=args.config)
        diarizen_service.spawn()
        
    # 1c. Start Sidon Worker (if TSE enabled)
    sidon_service = None
    if args.tse:
        sidon_service = SidonWorkerService(config, args, logger)
        sidon_service.spawn()

    # 1d. Join. All three were spawned above without blocking, so total startup
    # is now bounded by the slowest worker rather than the sum of all three.
    for _svc in (qwen3_service, diarizen_service, sidon_service):
        if _svc is not None:
            _svc.wait_ready()

    try:
        # 2. Load Models
        model_loader = ModelLoader(config, args, logger=logger)
        model_loader.load_base_models()
        model_loader.load_diarization_models(diarizen_service)
        model_loader.load_separation_models(sidon_service)
        model_loader.load_music_models()
        model_loader.load_asr_models(qwen3_service)
        model_loader.load_caption_model()

        # 3. Initialize Services
        audio_svc = AudioService(logger=logger)
        diarization_svc = DiarizationService(
            diarizer=model_loader.get("diarizer"),
            vad_model=model_loader.get("vad"),
            embedder=model_loader.get("embedder"),
            logger=logger,
            diarizer_config=env_profile.get("models", {}).get("diarizen", {})
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
            qwen3_service=qwen3_service,
            language=args.lang,
            batch_size=env_profile.get("models", {}).get("qwen3", {}).get("batch_size", 4),
            keep_models=args.keep_models
        )
        caption_svc = CaptionService(
            captioner=model_loader.get("captioner"),
            logger=logger
        )
        refinement_cfg = env_profile.get("models", {}).get("refinement", {})
        refinement_svc = DiarizationRefinementService(logger=logger, **refinement_cfg)
        export_svc = ExportService(logger=logger)

        # 4. Orchestrate via PipelineService
        pipeline = PipelineService(
            audio_svc, diarization_svc, separation_svc, music_svc,
            asr_svc, caption_svc, refinement_svc, export_svc, logger=logger,
            model_loader=model_loader,
            worker_services={
                "diarizen": diarizen_service,
                "sidon": sidon_service,
                "qwen3": qwen3_service,
            }
        )
        
        # One worker set serves the whole batch: loading models per file cost
        # ~64s each, which dominates once a run holds more than a couple of
        # files. Each file still gets its own output folder and its own
        # checkpoint scope.
        batch_cfg = config.get("batch", {})
        max_hours = args.max_hours if args.max_hours is not None else \
            float(batch_cfg.get("max_hours_per_run", 5.0))

        if args.audio_dir:
            exts = batch_cfg.get("audio_extensions",
                                 [".mp3", ".wav", ".m4a", ".flac", ".opus", ".ogg", ".aac"])
            paths = find_audio_files(args.audio_dir, exts)
            if not paths:
                raise RuntimeError(
                    f"No audio files in {args.audio_dir} (looked for {', '.join(exts)})")
        else:
            paths = [args.audio]

        collisions = find_name_collisions(paths)
        if collisions:
            logger.error(
                "These files share a basename and would write to the same output "
                "directory, silently overwriting each other:")
            for stem, group in collisions.items():
                logger.error(f"  {stem}: " + ", ".join(os.path.basename(g) for g in group))
            logger.error("Rename them or move them apart, then re-run.")
            return

        batches = plan_batches(paths, max_hours, logger=logger)
        total_hours = sum(audio_duration(p) for p in paths) / 3600.0
        logger.info(
            f"Processing {len(paths)} file(s), {total_hours:.2f}h total, "
            f"in {len(batches)} run(s) of up to {max_hours}h")

        if args.only_batch is not None:
            if not (1 <= args.only_batch <= len(batches)):
                logger.error(f"--only_batch {args.only_batch} is outside 1..{len(batches)}")
                return
            selected = [(args.only_batch, batches[args.only_batch - 1])]
            logger.info(f"Running batch {args.only_batch}/{len(batches)} only")
        else:
            selected = list(enumerate(batches, start=1))

        failures = []
        done = 0
        for bi, batch in selected:
            batch_hours = sum(audio_duration(p) for p in batch) / 3600.0
            logger.info(f"--- Batch {bi}/{len(batches)}: "
                        f"{len(batch)} file(s), {batch_hours:.2f}h ---")

            if args.by_stage:
                failures.extend(run_batch_by_stage(pipeline, args, config, batch, logger=logger))
                done += len(batch)
                continue

            for path in batch:
                done += 1
                logger.info(f"[{done}/{len(paths)}] Running pipeline on audio: {path}")
                try:
                    pipeline.run(args, config, path)
                except Exception as e:
                    # One bad file must not cost the rest of a five-hour run.
                    logger.error(f"Failed on {path}: {type(e).__name__}: {e}")
                    failures.append((path, f"{type(e).__name__}: {e}"))

        if failures:
            logger.warning(f"{len(failures)}/{len(paths)} file(s) failed:")
            for path, err in failures:
                logger.warning(f"  {os.path.basename(path)}: {err}")
        else:
            logger.info(f"All {len(paths)} file(s) completed")

    finally:
        if qwen3_service:
            qwen3_service.stop()
        if diarizen_service:
            diarizen_service.stop()
        if sidon_service:
            sidon_service.stop()
        logger.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()
