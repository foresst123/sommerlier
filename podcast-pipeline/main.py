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
# Worker subprocess types that can be alive at once: qwen3, qwen3_replica,
# sidon, and two replicated DiariZen workers. Count both diarizers so the CPU
# thread budget matches the pool that can actually be resident.
_CPU_THREADS = _configure_cpu(n_workers=5)
# _CPU_THREADS=3
# Sidon worker là subprocess riêng, nên publish rõ thread budget cho nó kế thừa.
# WorkerProcessService/base worker thường dùng os.environ.copy(), vì vậy biến này
# phải có trước khi SidonWorkerService spawn subprocess.
_os.environ.setdefault("SIDON_CPU_THREADS", str(_CPU_THREADS))

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
                             "split into duration-bounded groups; with --by_stage (the "
                             "default profile setting), every stage runs across one "
                             "group before the next stage. Each file writes its own "
                             "output folder.")
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
    parser.add_argument("--bss", "--tse", dest="bss", action="store_true",
                        help="Enable blind source separation of overlapped speech")
    parser.add_argument("--separator", choices=["sidon"], default=None,
                        help="Which model produces the two tracks")
    parser.add_argument("--music", "--panns", dest="music", action="store_true",
                        help="Enable background music analysis and removal")
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
                        help="Run each stage across one duration-bounded group before the next "
                             "stage, instead of the whole pipeline per file. Loads "
                             "each model once per group rather than once per file.")
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
    parser.add_argument("--kill_orphans", action="store_true",
                        help="At start-up, kill this user's leftover vLLM engines and "
                             "pipeline workers from an earlier run (parent gone, still "
                             "holding GPU memory). Without it they are only reported.")
    parser.add_argument("--keep_models", action="store_true",
                        help="Keep models in VRAM between stages instead of unloading them. "
                             "Saves reload time when processing many files, at the cost of a "
                             "higher peak: only use it when the GPU has room for every model at once.")
    parser.add_argument("--performance", action="store_true",
                        help="Turn on the performance scheduling path (true ASR batching, "
                             "pipelined refinement, split diarization placement, telemetry). "
                             "Forces it on for a profile whose performance.enabled "
                             "is false; a profile that already enables it ignores the flag.")
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
    parser.add_argument("--bridge_gap", type=float,
                        help="Override pipeline.bridge_gap for interrupted speaker turns")
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
    parser.add_argument(
        "--stop_after", type=str,
        choices=["music", "diarization", "separation", "music_removal", "asr",
                 "captioning", "refinement", "speaker_relabel", "word_alignment",
                 "conversation_exports"],
        help="Stop pipeline gracefully after this stage")
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
# this the profile overwrites deliberate flags, and `--env a100 --bss` would
# quietly run with the profile's bss rather than the requested one.
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
    os.environ["HOME"] = offline_dir  # where cached checkpoints are looked up
    os.environ["BSS_PATH"] = os.path.join(offline_dir, "bss_model")
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
# Separation thresholds live in the profile, but separation_service and bss_model read
# them at import time. Publish them as environment variables here -- before those
# imports run -- or the modules capture the defaults instead. An env var set by
# hand still wins, which keeps a quick sweep possible without editing config.
for _cfg_key, _env_key in (("qc_sim_threshold", "BSS_QC_SIM_THRESHOLD"),
                           ("min_voiced_sec", "BSS_MIN_VOICED_SEC"),
                           ("context_per_side_seconds", "BSS_STITCH_EDGE_PAD"),
                           ("context_max_per_side_seconds", "BSS_STITCH_EDGE_MAX"),
                           ("padding_min_per_speaker_seconds", "BSS_PADDING_MIN_PER_SPEAKER"),
                           ("boundary_search_seconds", "BSS_STITCH_SEARCH"),
                           # 0 = tắt hẳn pool build cửa sổ song song; số dương
                           # ép cứng số worker; để trống trong config thì
                           # separation_service tự tính theo usable_cores().
                           # Mặc định trong config.json là 0 (tắt) -- lợi ích
                           # đo được phụ thuộc số job overlap và độ trễ Sidon
                           # thật, chưa được xác nhận trên dữ liệu sản xuất.
                           ("window_workers", "BSS_WINDOW_WORKERS")):
    _value = env_profile.get("models", {}).get("bss", {}).get(_cfg_key)
    if _value is not None and _env_key not in os.environ:
        os.environ[_env_key] = str(_value)



# Enrollment memory is read at import by separation_service, so it is published
# here with the other separation thresholds rather than at call time.
_memory = env_profile.get("models", {}).get("bss", {}).get("enrollment_memory")
if _memory is not None and "BSS_MEMORY" not in os.environ:
    os.environ["BSS_MEMORY"] = "1" if _memory else "0"

_max_pending = env_profile.get("performance", {}).get("max_pending_jobs")
if _max_pending is not None and "BSS_WINDOW_MAX_PENDING" not in os.environ:
    os.environ["BSS_WINDOW_MAX_PENDING"] = str(_max_pending)

# The separator is read at BssSeparator construction, not at import,
# but it is published here with the other separation settings so one profile switch
# controls it like everything else.
_sep = env_profile.get("models", {}).get("bss", {}).get("separator")
if _sep and "BSS_SEPARATOR" not in os.environ:
    os.environ["BSS_SEPARATOR"] = str(_sep)

from services.model_loader import ModelLoader
from services.audio_service import AudioService
from services.diarization_service import DiarizationService
from services.separation_service import SeparationService
from services.music_service import MusicService
from services.asr_service import ASRService
from services.caption_service import CaptionService
from services.diarization_refinement_service import DiarizationRefinementService
from services.speaker_relabel_service import SpeakerRelabelService
from services.word_alignment_service import WordAlignmentService
from services.conversation_export_service import ConversationExportService
from services.clean_two_channel_dataset_service import CleanTwoChannelDatasetService
from services.export_service import ExportService
from services.pipeline_service import PipelineService
from services.qwen3_worker_service import Qwen3WorkerService
from services.whisper_vllm_worker_service import WhisperVLLMWorkerService
from services.phowhisper_worker_service import PhoWhisperWorkerService
from models.phowhisper_client import PhoWhisperClient
from models.qwen3_asr import Qwen3ASRClient
from models.whisper_vllm import WhisperVLLMClient
from services.diarizen_worker_service import DiarizenWorkerService
from services.sidon_worker_service import SidonWorkerService
from services.assignment_worker_service import AssignmentWorkerService
from services.worker_pool_service import WorkerPoolService


def _discard_partial(ledger, args, audio_path, logger, pipeline=None):
    """Remove a failed file's checkpoint and output so the retry starts clean.

    The checkpoint matters most: leaving it means the next attempt reloads the
    stages that did finish and jumps straight to the one that broke, with the
    same state that broke it. The output directory goes too, so a half-written
    transcript is not mistaken for a finished one.

    Worker subprocesses are stopped here too. A worker that crashed (OOM, bad
    input) may be in an undefined state and still holding VRAM. The next pass
    will restart it cleanly via _rebind_worker.
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

    # This file's diarization checkpoint is gone, so its next attempt
    # recomputes diarization from scratch. Drop any window plan already
    # prefetched for the attempt that just failed, so a retry cannot be
    # handed a plan built for a checkpoint that no longer exists.
    if pipeline is not None:
        separation_svc = getattr(pipeline, "separation_svc", None)
        drop = getattr(separation_svc, "drop_prefetched_plan", None)
        if callable(drop):
            drop(audio_path)

    # Release any worker subprocess that may be holding GPU memory after the
    # failure. _release_worker is a no-op when the worker is already gone.
    if pipeline is not None:
        for worker_name in ("diarizen", "sidon", "qwen3"):
            try:
                pipeline._release_worker(args, worker_name)
            except Exception as exc:
                if logger:
                    logger.warning(f"Could not release {worker_name} worker after failure: {exc}")


def main():
        
    logger = Logger.get_logger()
    logger.info(f"Starting Sommelier Pipeline for Job: {args.job_id}")
    from utils import preflight
    preflight.run(logger, kill_orphans=args.kill_orphans)
    logger.info(f"Steps ({args.env} profile{', +--steps' if args.steps else ''}): {_STEPS_NOTE}")
    from utils.cpu_plan import usable_cores
    # Workers inherit this through os.environ.copy() in base_worker_service.
    logger.info(f"CPU: {usable_cores()} core(s) usable, "
                f"{_CPU_THREADS} thread(s) per process")

    import torch
    from utils.performance_monitor import PerformanceMonitor
    from utils import performance_config
    # Resolved and logged once, here, so every later read is a plain lookup and
    # a mistyped key is a warning at startup rather than a silent baseline run.
    perf_cfg = performance_config.resolve(
        env_profile, logger=logger,
        enabled_override=True if getattr(args, "performance", False) else None)
    args.performance_config = perf_cfg
    performance_monitor = PerformanceMonitor(
        os.path.join(args.cache_dir, args.job_id, "performance"),
        interval_seconds=perf_cfg["telemetry_interval_seconds"],
        enabled=perf_cfg["enabled"], logger=logger)
    performance_monitor.start()
    if perf_cfg["enabled"]:
        from utils import profiling
        profiling.install(performance_monitor)     # step timings -> performance file

    # TF32 on the fp32 paths: DiariZen, BS-RoFormer, SSLAM and ECAPA all run in
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

    # Which GPU each ASR model sits on. Without the performance block this is
    # the layout from before placement was configurable.
    asr_perf = perf_cfg["stages"]["asr"]
    asr_cross_file = bool(perf_cfg["enabled"] and asr_perf["cross_file"])
    if perf_cfg["enabled"]:
        asr_placement = performance_config.resolve_asr_placement(
            asr_perf, args.gpu_1, args.gpu_2)
    else:
        asr_placement = {"qwen3": args.gpu_2, "whisper": args.gpu_2,
                         "phowhisper": args.gpu_1}

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
    qwen3_replica_service = None
    whisper_service = None
    if args.ASRMoE and will_run(args, "asr"):
        qwen_cfg = env_profile.get("models", {}).get("qwen3", {})
        qwen_env = ("vllm" if str(qwen_cfg.get("backend", "transformers")).lower()
                    == "vllm" else "qwen3")
        qwen3_worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_worker.py")
        qwen3_service = _prefetch(Qwen3WorkerService(
            lambda: resolve_worker_python(qwen_env, config=config,
                                          env_profile=env_profile, logger=logger),
            qwen3_worker_script, device_id=asr_placement["qwen3"], logger=logger,
            env_name=args.env, config_path=args.config,
            isolate_library_path=(qwen_env == "vllm")))
        if (perf_cfg["enabled"]
                and asr_perf["dynamic_replicas"]
                and not asr_cross_file      # the scheduler makes its own replicas
                and args.gpu_1 != args.gpu_2):
            replica_gpu = (args.gpu_1 if asr_placement["qwen3"] == args.gpu_2
                           else args.gpu_2)
            qwen3_replica_service = Qwen3WorkerService(
                lambda: resolve_worker_python(
                    qwen_env, config=config, env_profile=env_profile,
                    logger=logger),
                qwen3_worker_script, device_id=replica_gpu, logger=logger,
                env_name=args.env, config_path=args.config,
                isolate_library_path=(qwen_env == "vllm"))

        whisper_cfg = env_profile.get("models", {}).get("whisper", {})
        if str(whisper_cfg.get("backend", "ctranslate2")).lower() == "vllm":
            whisper_worker_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "whisper_vllm_worker.py")
            whisper_service = _prefetch(WhisperVLLMWorkerService(
                lambda: resolve_worker_python(
                    "vllm", config=config, env_profile=env_profile,
                    logger=logger),
                whisper_worker_script, device_id=asr_placement["whisper"],
                logger=logger, env_name=args.env, config_path=args.config))

    # 1a'. PhoWhisper as worker process(es), when the profile asks for them
    # (stages.asr.phowhisper_workers > 0). Otherwise it stays in this process. More
    # than one copy pulls from a single queue, spread across the cards.
    pho_worker_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "phowhisper_worker.py")
    phowhisper_service = None
    if (perf_cfg["enabled"] and asr_perf["phowhisper_workers"] > 0
            and will_run(args, "asr")):
        _pho_gpus = [g for g in dict.fromkeys((args.gpu_1, args.gpu_2))
                     if not torch.cuda.is_available()
                     or 0 <= int(g) < torch.cuda.device_count()]
        _pho_devices = performance_config.pho_worker_devices(
            asr_placement["phowhisper"], _pho_gpus, asr_perf["phowhisper_workers"])
        _pho_workers = [PhoWhisperWorkerService(
            _sys.executable, pho_worker_script, device_id=device, logger=logger,
            env_name=args.env, config_path=args.config) for device in _pho_devices]
        if _pho_workers:
            phowhisper_service = _prefetch(
                WorkerPoolService(_pho_workers, name="PhoWhisper")
                if len(_pho_workers) > 1 else _pho_workers[0])
            logger.info(
                f"[performance] PhoWhisper: {len(_pho_workers)} worker(s) on "
                + ", ".join(f"GPU {d}" for d in _pho_devices))

    # 1b. Start DiariZen workers (if dia3 is not used)
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
        diar_perf = perf_cfg["stages"]["diarization"]
        configured_devices = list(dict.fromkeys((args.gpu_1, args.gpu_2)))
        if torch.cuda.is_available():
            available_devices = [
                device for device in configured_devices
                if 0 <= int(device) < torch.cuda.device_count()
            ]
        else:
            # Preserve the existing startup error from the GPU model instead of
            # turning a missing CUDA runtime into an empty worker-pool error.
            available_devices = configured_devices[:1]
        if not available_devices:
            raise RuntimeError(
                "No configured diarization GPU is visible: requested "
                f"{configured_devices}, CUDA exposes {torch.cuda.device_count()}")
        requested_workers = (
            int(diar_perf["workers"]) if perf_cfg["enabled"] else 1)
        worker_count = min(
            requested_workers, int(perf_cfg["max_gpus"]),
            len(available_devices))

        if requested_workers > worker_count:
            logger.warning(
                f"[performance] diarization requested {requested_workers} workers "
                f"but only {len(available_devices)} distinct GPU(s) are configured; "
                f"using {worker_count}")

        if worker_count > 1:
            diarizen_workers = [DiarizenWorkerService(
                lambda: resolve_worker_python(
                    "diarizen", config=config, env_profile=env_profile,
                    logger=logger),
                diarizen_worker_script, device_id=device_id, logger=logger,
                env_name=args.env, config_path=args.config)
                for device_id in available_devices[:worker_count]]
            diarizen_service = WorkerPoolService(
                diarizen_workers, name="DiariZen")
            logger.info(
                f"[performance] DiariZen pool: {worker_count} workers on "
                + ", ".join(f"GPU {d}" for d in available_devices[:worker_count]))
        else:
            diar_devices = args.gpu_1
            if (perf_cfg["enabled"]
                    and diar_perf["placement"] == "split_components"
                    and args.gpu_1 != args.gpu_2):
                diar_devices = [args.gpu_1, args.gpu_2]
            diarizen_service = DiarizenWorkerService(
                lambda: resolve_worker_python(
                    "diarizen", config=config, env_profile=env_profile,
                    logger=logger),
                diarizen_worker_script, device_id=diar_devices, logger=logger,
                env_name=args.env, config_path=args.config)
        diarizen_service = _prefetch(diarizen_service)
        


    # 1c. Start the Sidon worker, when the profile asks for that separator.
    #
    # Gated on the separator name as well as on --bss: the in-process backend
    # needs no worker at all, and spawning one for it would download the
    # DialogueSidon weights and hold a GPU for a process nothing talks to.
    sidon_service = None
    sidon_worker_count = 1
    _separator = (getattr(args, "separator", None)
                  or env_profile.get("models", {}).get("bss", {}).get("separator")
                  or "sidon")
    if (str(_separator).strip().lower() == "sidon" and getattr(args, "bss", False)
            and will_run(args, "separation")):
        sidon_worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sidon_worker.py")
        sep_perf = perf_cfg["stages"]["separation"]
        max_sidon_workers = int(sep_perf["max_workers"])
        configured_devices = list(dict.fromkeys((args.gpu_1, args.gpu_2)))
        if torch.cuda.is_available():
            sidon_devices = [
                device for device in configured_devices
                if 0 <= int(device) < torch.cuda.device_count()
            ]
        else:
            sidon_devices = configured_devices[:1]
        requested_sidon_workers = (
            max_sidon_workers if perf_cfg["enabled"] else 1)
        if not sidon_devices:
            raise RuntimeError(
                "No configured separation GPU is visible: requested "
                f"{configured_devices}, CUDA exposes {torch.cuda.device_count()}")
        if requested_sidon_workers > len(sidon_devices):
            logger.warning(
                f"[performance] separation requested {requested_sidon_workers} "
                f"GPU(s) but only {len(sidon_devices)} are available")
        # One entry per worker process; a GPU repeats when workers_per_gpu > 1.
        sidon_devices = performance_config.sidon_worker_devices(
            sidon_devices, requested_sidon_workers, sep_perf["workers_per_gpu"],
            int(perf_cfg["max_gpus"]), perf_cfg["enabled"])
        sidon_worker_count = len(sidon_devices)
        sidon_workers = [SidonWorkerService(
            resolve_worker_python("sidon", config=config,
                                  env_profile=env_profile, logger=logger),
            sidon_worker_script, device_id=device_id, logger=logger,
            env_name=args.env, config_path=args.config)
            for device_id in sidon_devices]
        sidon_service = _prefetch(
            WorkerPoolService(sidon_workers, name="Sidon")
            if len(sidon_workers) > 1 else sidon_workers[0])
        logger.info(
            f"[performance] Sidon pool: {len(sidon_workers)} worker(s) on "
            + ", ".join(f"GPU {d}" for d in sidon_devices))

    # 1c'. Speaker-assignment workers (Silero VAD + WeSpeaker), one or more
    # per GPU. Started by the separation stage; see assignment_worker.py.
    assignment_service = None
    _assign_per_gpu = int(perf_cfg["stages"]["separation"].get(
        "assignment_workers_per_gpu", 0))
    if (perf_cfg["enabled"] and _assign_per_gpu > 0 and getattr(args, "bss", False)
            and will_run(args, "separation") and torch.cuda.is_available()):
        _assign_gpus = [g for g in dict.fromkeys((args.gpu_1, args.gpu_2))
                        if 0 <= int(g) < torch.cuda.device_count()]
        _assign_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assignment_worker.py")
        _assign_workers = [AssignmentWorkerService(
            _sys.executable, _assign_script, device_id=g,
            threads=int(perf_cfg["stages"]["separation"]["assignment_worker_threads"]),
            logger=logger)
            for _ in range(_assign_per_gpu) for g in _assign_gpus]
        if _assign_workers:
            assignment_service = (WorkerPoolService(_assign_workers, name="Assignment")
                                  if len(_assign_workers) > 1 else _assign_workers[0])
            logger.info(
                f"[performance] Speaker-assignment pool: {len(_assign_workers)} "
                "worker(s) on " + ", ".join(f"GPU {g}" for g in _assign_gpus)
                + f" x{_assign_per_gpu}")

    # 1d. Join whichever actually started. They were launched without blocking,
    # so startup is bounded by the slowest rather than the sum -- that is the
    # whole point of doing it here. One that did not start is simply skipped;
    # its stage will start and join it.
    for _svc in (qwen3_service, diarizen_service, sidon_service):
        if _svc is not None and getattr(_svc, "process", None) is not None:
            try:
                _svc.wait_ready()
            except Exception as e:
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
        model_loader.asr_placement = asr_placement

        # Replicas the ASR scheduler may start on a GPU that has just been freed.
        # Each factory returns (model, release); the scheduler chooses the GPU
        # and the batch size.
        def _free_cuda():
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        asr_replica_factories = {}
        if asr_cross_file and args.ASRMoE and will_run(args, "asr"):
            if qwen3_service is not None:
                def _qwen3_replica(gpu, batch):
                    service = Qwen3WorkerService(
                        lambda: resolve_worker_python(
                            qwen_env, config=config, env_profile=env_profile,
                            logger=logger),
                        qwen3_worker_script, device_id=gpu, logger=logger,
                        env_name=args.env, config_path=args.config,
                        batch_size=batch,
                        isolate_library_path=(qwen_env == "vllm"))
                    service.spawn()
                    service.wait_ready()
                    return Qwen3ASRClient(service.process), service.stop
                asr_replica_factories["qwen3"] = _qwen3_replica
            if whisper_service is not None:
                def _whisper_replica(gpu, batch):
                    service = WhisperVLLMWorkerService(
                        lambda: resolve_worker_python(
                            "vllm", config=config, env_profile=env_profile,
                            logger=logger),
                        whisper_worker_script, device_id=gpu, logger=logger,
                        env_name=args.env, config_path=args.config)
                    service.spawn()
                    service.wait_ready()
                    return WhisperVLLMClient(service, batch_size=batch), service.stop
                asr_replica_factories["whisper"] = _whisper_replica
            else:
                asr_replica_factories["whisper"] = lambda gpu, batch: (
                    model_loader.make_whisper_ct2(torch.device(f"cuda:{gpu}"), batch),
                    _free_cuda)
            if phowhisper_service is not None:
                def _pho_replica(gpu, batch):
                    service = PhoWhisperWorkerService(
                        _sys.executable, pho_worker_script, device_id=gpu,
                        logger=logger, env_name=args.env, config_path=args.config)
                    service.spawn()
                    service.wait_ready()
                    return PhoWhisperClient(service, batch_size=batch), service.stop
                asr_replica_factories["phowhisper"] = _pho_replica
            else:
                asr_replica_factories["phowhisper"] = lambda gpu, batch: (
                    model_loader.make_phowhisper(torch.device(f"cuda:{gpu}"), batch),
                    _free_cuda)

        # 3. Initialize Services
        audio_svc = AudioService(logger=logger)
        diarization_svc = DiarizationService(
            model_loader=model_loader,
            logger=logger,
            diarizer_config=env_profile.get("models", {}).get("diarizen", {}),
            performance_config=perf_cfg["stages"]["diarization"],
        )
        separation_svc = SeparationService(
            model_loader=model_loader,
            logger=logger,
            performance_config={
                **perf_cfg["stages"]["separation"],
                "enabled": perf_cfg["enabled"],
                # How many Sidon processes can serve a window at once.
                "gpu_workers": max(sidon_worker_count,
                                   perf_cfg["stages"]["separation"]["max_workers"]),
                # Lets the admission byte budget derive its ceiling from the RAM limit.
                "ram_soft_fraction": perf_cfg["ram_soft_fraction"],
            },
        )
        music_svc = MusicService(
            model_loader=model_loader,
            logger=logger,
            performance_config={
                **perf_cfg["stages"]["music"],
                "enabled": perf_cfg["enabled"],
            },
        )
        asr_svc = ASRService(
            logger=logger,
            model_loader=model_loader,
            qwen3_service=qwen3_service,
            qwen3_replica_service=qwen3_replica_service,
            performance_config=perf_cfg["stages"]["asr"],
            performance_monitor=performance_monitor,
            language=args.lang,
            batch_size=env_profile.get("models", {}).get("qwen3", {}).get("batch_size", 4),
            keep_models=args.keep_models,
            replica_factories=asr_replica_factories,
            asr_workers={name: service for name, service in
                         (("qwen3", qwen3_service), ("whisper", whisper_service),
                          ("phowhisper", phowhisper_service))
                         if service is not None},
            asr_placement=asr_placement,
        )
        caption_svc = CaptionService(
            model_loader=model_loader,
            logger=logger
        )
        refinement_cfg = dict(
            env_profile.get("models", {}).get("refinement", {}))
        # Only when the feature is on: otherwise the refinement service keeps
        # the device_map and batch rules the baseline was measured with.
        if perf_cfg["enabled"]:
            refinement_perf = perf_cfg["stages"]["refinement"]
            for key in ("placement", "gpu_memory_utilization", "max_batch_tokens",
                        "micro_batch_size", "pipeline_split_ratio", "cpu_threads",
                        "workers", "shared_queue", "chunk_size", "parallel_windows",
                        "progress_interval_seconds"):
                target = ("progress_interval"
                          if key == "progress_interval_seconds" else key)
                refinement_cfg[target] = refinement_perf[key]
            refinement_cfg["pipeline_devices"] = [args.gpu_1, args.gpu_2]
            refinement_cfg["device"] = f"cuda:{args.gpu_1}"
        refinement_svc = DiarizationRefinementService(
            logger=logger, config=config, env_profile=env_profile,
            **refinement_cfg)
        # Passes over the same resident LLM. Building them loads nothing and
        # switches nothing on: `steps.speaker_relabel` decides whether it runs.
        # An unknown key in `models.relabel` raises here, so a typo is an error
        # rather than a setting that quietly did not apply.
        relabel_svc = SpeakerRelabelService(
            refinement_svc, logger=logger,
            **dict(env_profile.get("models", {}).get("relabel", {})))
        conversation_export_svc = ConversationExportService(
            refinement_svc, logger=logger,
            **dict(env_profile.get("models", {}).get("conversation_selection", {})))
        alignment_cfg = dict(
            env_profile.get("models", {}).get("word_alignment", {}))
        if alignment_cfg.get("device", "auto") == "auto":
            alignment_cfg["device"] = (
                f"cuda:{args.gpu_1}" if torch.cuda.is_available() else "cpu")
        alignment_cfg["model_cache_only"] = bool(
            env_profile.get("offline_mode", False))
        if alignment_cfg.get("workers_per_gpu", 0) and torch.cuda.is_available():
            # One pool spanning both cards; a single-GPU box collapses to one.
            alignment_cfg.setdefault("worker_gpus", sorted({args.gpu_1, args.gpu_2}))
        word_alignment_svc = WordAlignmentService(
            language=args.lang, logger=logger, **alignment_cfg)
        clean_dataset_svc = CleanTwoChannelDatasetService(
            logger=logger,
            # Same bounded pool size as the conversation-export renderer.
            workers=int(env_profile.get("models", {}).get(
                "conversation_selection", {}).get("render_workers", 1)))
        export_svc = ExportService(logger=logger)

        # 4. Orchestrate via PipelineService
        pipeline = PipelineService(
            audio_svc, diarization_svc, separation_svc, music_svc,
            asr_svc, caption_svc, refinement_svc, export_svc, logger=logger,
            model_loader=model_loader,
            worker_services={
                "diarizen": diarizen_service,
                "qwen3": qwen3_service,
                "whisper": whisper_service,
                "phowhisper": phowhisper_service,
                "sidon": sidon_service,
                "assignment": assignment_service,
            },
            performance_monitor=performance_monitor,
            relabel_svc=relabel_svc,
            conversation_export_svc=conversation_export_svc,
            word_alignment_svc=word_alignment_svc,
            clean_dataset_svc=clean_dataset_svc,
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

        # Freeze the files visible at startup. Every full-pipeline pass below is
        # duration-bounded, including stage-major passes. Files copied in during
        # the run wait for the next invocation instead of extending this run.
        ledger = ProgressLedger(args.audio_dir, logger=logger)
        if ledger.done or ledger.failed:
            logger.info(f"Resuming: {ledger.summary(len(paths))} "
                        f"(ledger: {os.path.basename(ledger.path)})")

        failures = []
        pass_no = 0
        run_snapshot = list(paths)
        while True:
            pass_no += 1
            todo = ledger.pending(run_snapshot)
            if not todo:
                break

            # The max-hours contract applies to every full-pipeline pass. In
            # stage-major mode this group completes ASR before its refinement
            # model is loaded, then finishes every remaining stage before the
            # next duration-bounded group starts.
            group = plan_batches(todo, max_hours, logger=logger)[0]
            group_hours = sum(audio_duration(p) for p in group) / 3600.0
            logger.info(
                f"--- Pass {pass_no}: {len(group)} file(s), {group_hours:.2f}h "
                f"({len(todo)} of {len(run_snapshot)} still to do) ---")

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

        from utils.batch import split_final_failures
        still_failed, recovered = split_final_failures(failures, ledger.is_done)
        for path, err in recovered:
            logger.info(f"Recovered on retry: {os.path.basename(path)} "
                        f"(earlier failure: {err})")
        if still_failed:
            logger.warning(f"{len(still_failed)}/{len(paths)} file(s) failed:")
            for path, err in still_failed:
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
        if assignment_service:
            assignment_service.stop()
        performance_monitor.stop()
        if perf_cfg["enabled"]:
            from utils import profiling
            profiling.uninstall()
            logger.info("Performance report: "
                        + os.path.join(performance_monitor.output_dir,
                                       "performance_report.txt"))

        logger.info("Pipeline execution finished.")


if __name__ == "__main__":
    main()
