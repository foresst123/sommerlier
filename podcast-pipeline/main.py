# Two things have to happen before anything pulls torch or matplotlib in
# transitively, and both are settings a caller's environment can poison.
import os as _os

# 1. A batch pipeline must never get an interactive matplotlib backend.
#
# Nothing here plots. matplotlib arrives anyway, five imports deep --
# whisperx -> pyannote.audio -> lightning -> torchmetrics -> matplotlib -- and
# on import it does `rcParams['backend'] = os.environ.get('MPLBACKEND')`. A
# notebook kernel exports `module://matplotlib_inline.backend_inline`, which
# only exists inside that kernel's own interpreter, so a run launched from
# Kaggle or Jupyter dies with ValueError before reaching a single stage.
#
# Fixing this from the caller does not hold: IPython rewrites MPLBACKEND when
# matplotlib is first configured, and an `!env VAR=... cmd` line is rewritten
# again by the shell. The process that needs the value is this one, so it sets
# it here. An explicit non-notebook choice is left alone.
if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"
else:
    _os.environ.setdefault("MPLBACKEND", "Agg")

# 2. Thread budget: torch reads OMP_NUM_THREADS at import and caches it, so
# this has to run before anything pulls torch in transitively.
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
    parser.add_argument("--separator", choices=["usef"], default=None,
                        help="Which model produces the two tracks. usef (default) "
                             "is target-conditioned TF-GridNet at 8kHz: it needs no "
                             "channel assignment, but discards everything above "
                             "4kHz.")
    parser.add_argument("--panns", action="store_true", help="Enable background music removal")
    parser.add_argument("--music_separator", default=None, metavar="CKPT",
                        help="Which BS-RoFormer checkpoint isolates vocals once "
                             "PANNs finds music, as an audio-separator model "
                             "filename (e.g. model_bs_roformer_ep_368_sdr_12.9628.ckpt). "
                             "Defaults to models.bs_roformer.model in the profile.")
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
    parser.add_argument("--prefetch_workers", action="store_true",
                        help="Start the DiariZen and Qwen3 workers at launch "
                             "instead of when their stage runs. Hides their "
                             "load time behind earlier stages, at the cost of "
                             "holding their VRAM for the whole run.")
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
    parser.add_argument("--steps", type=str, default=None,
                        help="Turn stages on or off for this run, overriding the "
                             "profile's `steps` block: --steps diarization=off, or "
                             "several as name=on,name=off. Exists so a run does not "
                             "have to edit config.json -- which a re-clone reverts "
                             "without saying so.")
    parser.add_argument("--stop_after", type=str, choices=["music", "diarization", "separation", "music_removal", "asr", "captioning"], help="Stop pipeline gracefully after this stage")
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

# "steps" is the on/off list, kept apart from the tuning values above: each key
# is one stage, and false skips it. Landed as step_<name> so a stage's switch
# cannot collide with a threshold that happens to share its name.
_steps = dict(env_profile.get("steps", {}))

# --steps wins over the profile. The command line is the only place a run can
# say what it wants without editing a tracked file, and editing config.json is
# a trap: a re-clone restores it and the run silently does something else.
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
for _pair in (args.steps or "").replace(";", ",").split(","):
    _pair = _pair.strip()
    if not _pair:
        continue
    _name, _, _value = _pair.partition("=")
    _name, _value = _name.strip(), _value.strip().lower()
    # A bare name means on, so `--steps music_analysis` reads the way it looks.
    if _value and _value not in _TRUTHY and _value not in _FALSY:
        raise SystemExit(f"main.py: error: --steps {_pair!r}: expected on/off")
    _steps[_name] = _value not in _FALSY

for k, v in _steps.items():
    setattr(args, f"step_{k}", bool(v))

# Printed, not left to be deduced from what did not happen. A stage silently
# off is the hardest kind of run to read: the log shows work that happened and
# nothing about the work that was never asked for.
_off = sorted(k for k, v in _steps.items() if not v)
_STEPS_NOTE = ("all stages on" if not _off
               else "stages OFF: " + ", ".join(_off))

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
from utils.steps import will_run
from utils.worker_env import resolve_worker_python
# TSE thresholds live in the profile, but separation_service and tse_model read
# them at import time. Publish them as environment variables here -- before those
# imports run -- or the modules capture the defaults instead. An env var set by
# hand still wins, which keeps a quick sweep possible without editing config.
for _cfg_key, _env_key in (("qc_sim_threshold", "TSE_QC_SIM_THRESHOLD"),
                           ("min_voiced_sec", "TSE_MIN_VOICED_SEC")):
    _value = env_profile.get("models", {}).get("tse", {}).get(_cfg_key)
    if _value is not None and _env_key not in os.environ:
        os.environ[_env_key] = str(_value)



# Enrollment memory is read at import by separation_service, so it is published
# here with the TSE thresholds rather than at call time.
_memory = env_profile.get("models", {}).get("tse", {}).get("enrollment_memory")
if _memory is not None and "TSE_MEMORY" not in os.environ:
    os.environ["TSE_MEMORY"] = "1" if _memory else "0"

# The separator is read at TargetSpeakerExtractor construction, not at import,
# but it is published here with the other TSE settings so one profile switch
# controls it like everything else.
_sep = env_profile.get("models", {}).get("tse", {}).get("separator")
if _sep and "TSE_SEPARATOR" not in os.environ:
    os.environ["TSE_SEPARATOR"] = str(_sep)

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
    logger.info(f"Steps ({args.env} profile{', +--steps' if args.steps else ''}): {_STEPS_NOTE}")
    from utils.cpu_plan import usable_cores
    # Workers inherit this through os.environ.copy() in base_worker_service.
    logger.info(f"CPU: {usable_cores()} core(s) usable, "
                f"{_CPU_THREADS} thread(s) per process")

    import torch

    # TF32 on the fp32 paths: DiariZen, BS-RoFormer, PANNS and ECAPA all run in
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



    # 1. Workers, built here and started here only as a head start.
    #
    # The stage that needs a worker is what actually starts it, in
    # PipelineService._load. This block exists because starting them one at a
    # time serialised ~100s of model loading, so the ones this run will reach
    # are launched together and joined once below. That makes it a prefetch:
    # it may fail, and a failure here is not the run's problem -- the stage
    # will try again and raise properly if the worker genuinely cannot start.
    #
    # The interpreter is resolved inside spawn() rather than now, so a missing
    # venv for a stage this run never reaches costs nothing.
    def _prefetch(service):
        """Start a worker now, or leave it for the stage that needs it.

        Off by default. Starting both workers up front hides their model-load
        time behind the music stage, but it also parks them in VRAM for the
        whole run: Qwen3-ASR sits on its GPU from the first second until the
        ASR stage, which on a two-file run is a quarter of an hour of a 1.7B
        model holding memory it is not using. `_ensure_worker` already starts
        each one immediately before its stage loads models, so the lazy path
        costs a wait, not a failure.

        Turn it back on with `pipeline.prefetch_workers` when the GPUs have
        room to spare and the wall clock matters more.
        """
        if service is None:
            return None
        if not getattr(args, "prefetch_workers", False):
            logger.info(f"{service.name} worker will start when its stage does")
            return service
        try:
            service.spawn()
        except Exception as e:
            logger.warning(f"Could not pre-start the {service.name} worker ({e}); "
                           "its stage will start it when it gets there")
        return service

    qwen3_service = None
    if args.ASRMoE and will_run(args, "asr"):
        qwen3_worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_worker.py")
        qwen3_service = _prefetch(Qwen3WorkerService(
            lambda: resolve_worker_python("qwen3", config=config,
                                          env_profile=env_profile, logger=logger),
            qwen3_worker_script, device_id=args.gpu_2, logger=logger,
            env_name=args.env, config_path=args.config))

    # 1b. Start DiariZen Worker (if dia3 is not used)
    #
    # Gated on the step as well as the flag: these workers are separate
    # processes with their own interpreter and weights, spawned before the
    # first file is opened. A run that stops after the music stage was
    # starting a diarizer it would never speak to -- and on an install without
    # DiariZen, dying there instead of producing the music output it was asked
    # for.
    diarizen_service = None
    if not args.dia3 and will_run(args, "diarization"):
        diarizen_worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diarizen_worker.py")
        diarizen_service = _prefetch(DiarizenWorkerService(
            lambda: resolve_worker_python("diarizen", config=config,
                                          env_profile=env_profile, logger=logger),
            diarizen_worker_script, device_id=args.gpu_1, logger=logger,
            env_name=args.env, config_path=args.config))
        


    # 1d. Join whichever actually started. They were launched without blocking,
    # so startup is bounded by the slowest rather than the sum -- that is the
    # whole point of doing it here. One that did not start is simply skipped;
    # its stage will start and join it.
    for _svc in (qwen3_service, diarizen_service):
        if _svc is not None and getattr(_svc, "process", None) is not None:
            try:
                _svc.wait_ready()
            except Exception as e:
                # Torn down, not left half-alive: _ensure_worker treats a
                # non-None process as a working one, so a spawned-but-never-
                # ready worker would be handed to the stage as if it were fine.
                logger.warning(f"{_svc.name} worker did not come up ({e}); "
                               "its stage will start it again")
                try:
                    _svc.stop()
                except Exception:
                    pass

    try:
        # 2. Build the loader, but load nothing yet.
        #
        # PipelineService calls the loader for each stage at the point that
        # stage runs, so peak VRAM is the largest pair of stages rather than
        # the sum of every model. Loading here instead put DiariZen (5.15GB),
        # Sidon (2.62GB), PhoWhisper, Whisper, BS-RoFormer and the captioner on the
        # card before the first stage had produced anything -- and a run that
        # resumes from a checkpoint paid for models it never called.
        model_loader = ModelLoader(config, args, logger=logger)

        # 3. Initialize Services
        audio_svc = AudioService(logger=logger)
        diarization_svc = DiarizationService(
            model_loader=model_loader,
            logger=logger,
            diarizer_config=env_profile.get("models", {}).get("diarizen", {})
        )
        separation_svc = TargetExtractionService(
            model_loader=model_loader,
            logger=logger
        )
        music_svc = MusicService(
            model_loader=model_loader,
            logger=logger
        )
        asr_svc = ASRService(
            logger=logger,
            model_loader=model_loader,
            qwen3_service=qwen3_service,
            language=args.lang,
            batch_size=env_profile.get("models", {}).get("qwen3", {}).get("batch_size", 4),
            keep_models=args.keep_models
        )
        caption_svc = CaptionService(
            model_loader=model_loader,
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

        logger.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()
