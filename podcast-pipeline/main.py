# Thread budget first: torch reads OMP_NUM_THREADS at import and caches it, so
# this has to run before anything pulls torch in transitively.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from utils.cpu_plan import configure_process as _configure_cpu
_CPU_THREADS = _configure_cpu(n_workers=3)

import argparse
import json
import os
import time
from utils.logger import Logger
def _build_parser():
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
                        help="Stop after one pass instead of working through the "
                             "whole directory. Lets a corpus larger than a session "
                             "limit be finished across several sessions -- progress "
                             "is kept in the input directory's ledger, so the next "
                             "run picks up where this one stopped.")
    parser.add_argument("--keep_models", action="store_true",
                        help="Keep models in VRAM between stages instead of unloading them. "
                             "Saves reload time when processing many files, at the cost of a "
                             "higher peak: only use it when the GPU has room for every model at once.")
    parser.add_argument("--no_stage_output", action="store_true",
                        help="Skip the per-stage artifact directories (01_diarization/, "
                             "02_separation/, ...). They are written by default so a run "
                             "stopped or crashed part-way still leaves its finished work "
                             "on disk.")
    # Tuning knobs that normally live in the profile. Declared here so a run can
    # override one without editing config.json -- handy for a sweep. Anything
    # not passed falls back to the profile.
    parser.add_argument("--merge_gap", type=float,
                        help="Override pipeline.merge_gap from the profile")
    parser.add_argument("--max_segment_length", type=float,
                        help="Override pipeline.max_segment_length from the profile")
    parser.add_argument("--no_review_page", dest="review_page", action="store_false",
                        default=None,
                        help="Skip the HTML review page. It is built by default at "
                             "the end of each file and embeds every clip, so it is "
                             "worth turning off when only the transcripts matter.")
    parser.add_argument("--review_max_mb", type=int, default=None,
                        help="Cap on audio embedded in the review page (default 400).")
    parser.add_argument("--stop_after", type=str, choices=["diarization", "separation", "music_removal", "asr", "captioning"], help="Stop pipeline gracefully after this stage")
    return parser


def parse_args():
    return _build_parser().parse_args()

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

# --- Config -> args, with anything typed on the command line winning -------
# argparse cannot tell a default from a value the user passed, so re-parse with
# every default suppressed: what survives is what was actually typed. Without
# this the profile overwrites deliberate flags, and `--env a100 --tse` would
# quietly run with the profile's tse rather than the requested one.
_probe = _build_parser()
for _action in _probe._actions:
    if _action.dest != "help":
        _action.default = argparse.SUPPRESS
_explicit = set(vars(_probe.parse_args()).keys())


def _from_config(key, value):
    """Apply a config value unless the command line already set this key."""
    if key in _explicit:
        return False
    setattr(args, key, value)
    return True


# Everything under "pipeline" lands on args, so which stages run and their
# thresholds can live in the profile instead of the command line.
for k, v in env_profile.get("pipeline", {}).items():
    _from_config(k, v)

for k in ("gpu_1", "gpu_2"):
    if k in env_profile:
        _from_config(k, env_profile[k])

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
from utils.progress import ProgressLedger
from utils.worker_env import resolve_worker_python
# TSE thresholds live in the profile, but separation_service and tse_model read
# them at import time. Publish them as environment variables here -- before those
# imports run -- or the modules capture the defaults instead. An env var set by
# hand still wins, which keeps a quick sweep possible without editing config.
for _cfg_key, _env_key in (("qc_sim_threshold", "TSE_QC_SIM_THRESHOLD"),
                           ("min_voiced_sec", "TSE_MIN_VOICED_SEC"),
                           ("stitch_solo", "TSE_STITCH_SOLO"),
                           ("stitch_edge_pad", "TSE_STITCH_EDGE_PAD")):
    _value = env_profile.get("models", {}).get("tse", {}).get(_cfg_key)
    if _value is not None and _env_key not in os.environ:
        os.environ[_env_key] = str(_value)

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

def _discard_partial(ledger, args, audio_path, logger, pipeline=None):
    """Remove a failed file's checkpoint and output so the retry starts clean.

    The checkpoint matters most: leaving it means the next attempt reloads the
    stages that did finish and jumps straight to the one that broke, with the
    same state that broke it. The output directory goes too, so a half-written
    transcript is not mistaken for a finished one.
    """
    base_job = getattr(args, "job_id", "default_job")
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    cache_dir = getattr(args, "cache_dir", "cache")
    targets = [os.path.join(cache_dir, f"{base_job}_{stem}")]
    if pipeline is not None:
        try:
            targets.append(pipeline._resolve_output_dir(args, audio_path))
        except Exception:
            pass
    ledger.discard_partial_output(*targets, logger=logger)


def main():
        
    logger = Logger.get_logger()
    logger.info(f"Starting Sommelier Pipeline for Job: {args.job_id}")
    from utils.cpu_plan import usable_cores
    # Workers inherit this through os.environ.copy() in base_worker_service.
    logger.info(f"CPU: {usable_cores()} core(s) usable, "
                f"{_CPU_THREADS} thread(s) per process")

    import torch

    # TF32 on the fp32 paths: DiariZen, Demucs, PANNS and ECAPA all run in
    # fp32, and on Ampere and later their matmuls and convolutions can use
    # TF32 tensor cores instead. Same code, same memory, roughly an order of
    # magnitude more throughput on those ops, at a precision that is ample for
    # inference. Turing has no TF32 units, so this is a no-op there rather
    # than a regression -- which is why it can default to on.
    tf32 = env_profile.get("allow_tf32", True)
    if torch.cuda.is_available() and tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        caps = {torch.cuda.get_device_capability(i)[0]
                for i in range(torch.cuda.device_count())}
        if any(c >= 8 for c in caps):
            logger.info("TF32 enabled for fp32 matmul/conv (Ampere or newer detected)")
        else:
            logger.info("TF32 requested but this GPU predates Ampere; fp32 ops are unchanged")
    elif torch.cuda.is_available():
        logger.info("TF32 disabled by config (allow_tf32=false)")

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
        # The profile's batch settings win over the shared defaults: how many
        # hours fit in one run depends on the machine, not on the corpus.
        batch_cfg = dict(config.get("batch", {}))
        batch_cfg.update(env_profile.get("batch") or {})
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

        # A corpus is processed in passes. Each pass re-scans the directory,
        # takes the next max_hours of unfinished audio, and runs every stage
        # across that whole group before moving on. Re-scanning is what lets
        # files be added while the run is going: they are simply there on the
        # next pass.
        ledger = ProgressLedger(args.audio_dir, logger=logger)
        if ledger.done or ledger.failed:
            logger.info(f"Resuming: {ledger.summary(len(paths))} "
                        f"(ledger: {os.path.basename(ledger.path)})")

        failures = []
        pass_no = 0
        while True:
            pass_no += 1
            # Re-scan every pass, not once at the top: this is the only reason
            # newly-copied files get picked up without restarting the run.
            current = find_audio_files(args.audio_dir, exts)
            todo = ledger.pending(current)
            if not todo:
                break

            group = plan_batches(todo, max_hours, logger=logger)[0]
            group_hours = sum(audio_duration(p) for p in group) / 3600.0
            logger.info(
                f"--- Pass {pass_no}: {len(group)} file(s), {group_hours:.2f}h "
                f"({len(todo)} of {len(current)} still to do) ---")

            started = time.time()
            if args.by_stage:
                # Stage-major: one model loaded, every file in the group pushed
                # through it, then the next stage.
                pass_failures = run_batch_by_stage(
                    pipeline, args, config, group, logger=logger)
                failed_paths = {p for p, _ in pass_failures}
                for path, err in pass_failures:
                    logger.error(f"FAILED {os.path.basename(path)}: {err}")
                    _discard_partial(ledger, args, path, logger, pipeline)
                    ledger.mark_failed(path, err)
                    failures.append((path, err))
                for path in group:
                    if path not in failed_paths:
                        ledger.mark_done(path)
            else:
                for path in group:
                    logger.info(f"Running pipeline on audio: {path}")
                    file_started = time.time()
                    try:
                        pipeline.run(args, config, path)
                        ledger.mark_done(path, time.time() - file_started)
                    except Exception as e:
                        # One bad file must not cost the rest of the pass.
                        err = f"{type(e).__name__}: {e}"
                        logger.error(f"FAILED {os.path.basename(path)}: {err}")
                        _discard_partial(ledger, args, path, logger, pipeline)
                        ledger.mark_failed(path, err)
                        failures.append((path, err))

            # Written after every pass, so a run killed between passes resumes
            # from the last completed group rather than the beginning.
            ledger.save()
            logger.info(f"Pass {pass_no} finished in {(time.time() - started) / 60:.1f} min "
                        f"({ledger.summary()})")

            if args.only_batch is not None:
                logger.info("--only_batch was given; stopping after this pass")
                break

        logger.info(f"Corpus complete: {ledger.summary()}")

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
