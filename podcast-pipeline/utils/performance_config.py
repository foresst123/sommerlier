"""Validate, resolve and fingerprint the `performance` block of a profile.

Every unknown or malformed key is reported rather than dropped. A typo that
quietly becomes {} reads exactly like "the user asked for the baseline", so the
run would be measured as if the feature were on while none of it executed --
the one failure mode that makes a benchmark lie.

`resolve()` returns a fully populated dict: every key below is present
afterwards, so callers read `cfg["stages"]["asr"]["true_batching"]` without
chasing defaults at each call site.
"""

import hashlib
import json

BOOL, INT, FLOAT, STR = "bool", "int", "float", "str"

# name -> (type, default, low, high). low/high are None when unbounded.
_TOP = {
    "enabled": (BOOL, False, None, None),
    "max_gpus": (INT, 2, 1, 8),
    "gpu_reserve_gib": (FLOAT, 2.0, 0.0, 128.0),
    "cpu_reserve_for_coordinator": (INT, 1, 0, 64),
    "ram_soft_fraction": (FLOAT, 0.75, 0.1, 1.0),
    "ram_hard_fraction": (FLOAT, 0.85, 0.1, 1.0),
    "max_pending_jobs": (INT, 32, 1, 4096),
    "telemetry_interval_seconds": (FLOAT, 1.0, 0.25, 3600.0),
}

_STAGES = {
    "asr": {
        "true_batching": (BOOL, True, None, None),
        "dynamic_replicas": (BOOL, True, None, None),
        "max_replicas_per_model": (INT, 2, 1, 4),
        "replica_min_pending_jobs": (INT, 12, 1, 100000),
        # A replica opens only when it is forecast to end the ASR stage at
        # least this much sooner. The next two are only a first guess: after a
        # replica has run once, its measured load time and added throughput
        # replace them for the rest of the process.
        "replica_min_gain_seconds": (FLOAT, 15.0, 0.0, 3600.0),
        "replica_load_seconds": (FLOAT, 30.0, 1.0, 1800.0),
        "replica_speed_ratio": (FLOAT, 0.5, 0.0, 1.0),
        # Cross-file scheduling (services/asr_scheduler.py): models consume a
        # queue spanning `files_in_flight` files instead of meeting at a
        # per-file barrier. Off by default, which keeps the per-file path.
        "cross_file": (BOOL, False, None, None),
        "files_in_flight": (INT, 3, 1, 8),
        # Batch size while two models share a GPU, and the larger one a model
        # gets when its GPU peer has finished or when a replica starts.
        "shared_batch_size": (INT, 16, 1, 256),
        "boost_batch_size": (INT, 48, 1, 512),
        # Which of the two configured GPUs each model sits on. The defaults are
        # the layout before these keys existed.
        "qwen3_gpu": (STR, "gpu_2", None, None),
        "whisper_gpu": (STR, "gpu_2", None, None),
        "phowhisper_gpu": (STR, "gpu_1", None, None),
        # PhoWhisper as worker processes (phowhisper_worker.py) instead of inside the
        # main process. 0 keeps it in-process; more than 1 runs copies that pull from
        # one queue, spread over the cards starting with `phowhisper_gpu`.
        "phowhisper_workers": (INT, 0, 0, 4),
    },
    "refinement": {
        "workers": (INT, 1, 1, 2),
        "placement": (STR, "auto", None, None),
        "gpu_memory_utilization": (FLOAT, 0.82, 0.50, 0.95),
        "max_batch_tokens": (INT, 0, 0, 1000000),
        "micro_batch_size": (INT, 1, 1, 1024),
        "pipeline_split_ratio": (FLOAT, 0.5, 0.20, 0.80),
        "cpu_threads": (INT, 0, 0, 256),
    },
    "diarization": {
        "workers": (INT, 1, 1, 2),
        "placement": (STR, "single", None, None),
    },
    "separation": {
        "max_workers": (INT, 1, 1, 2),
        # Sidon worker processes on each GPU (max_workers still picks how many
        # GPUs). Sidon runs one small diffusion job at a time, so a card often
        # waits on kernel launches; a second process can use those gaps.
        "workers_per_gpu": (INT, 1, 1, 4),
        "gpu_prefetch_per_worker": (INT, 1, 1, 8),
        "postprocess_workers": (INT, 1, 1, 16),
        "postprocess_device": (STR, "cpu", None, None),
        "ordered_postprocess": (BOOL, True, None, None),
        # Speaker assignment (WeSpeaker on the CPU) runs after every Sidon
        # window, in order, and can be slower than the Sidon workers -- which then
        # sit idle. assignment_threads is the ONNX Runtime thread count for it
        # (0 = the process's CPU budget); assignment_parallel scores the four
        # probe embeddings of a window at the same time.
        "assignment_threads": (INT, 0, 0, 64),
        "assignment_parallel": (INT, 1, 1, 4),
        # Worker processes per GPU running Silero VAD + WeSpeaker for speaker
        # assignment (assignment_worker.py). 0 keeps both models in the main
        # process; with workers, assignment_parallel is how many probes of one
        # window are in flight at once.
        "assignment_workers_per_gpu": (INT, 0, 0, 4),
    },
    "music": {
        "tagger_workers": (INT, 1, 1, 2),
        "max_separator_workers": (INT, 1, 1, 2),
        # BS-RoFormer worker processes on each GPU (max_separator_workers still
        # picks how many GPUs). Only honoured with bs_roformer.isolate_process,
        # since each extra instance needs its own process.
        "workers_per_gpu": (INT, 1, 1, 4),
        # SSLAM (tagger) always stays on device_1. When this is on, the sole
        # BS-RoFormer instance moves to device_2 instead, so file N's removal
        # and file N+1's classification land on different cards when the
        # music-stage batch pass overlaps them. Mutually exclusive with
        # max_separator_workers>1 -- both ask for the same second GPU in
        # different ways; ModelLoader picks this one and warns if both are set.
        "cross_file_overlap": (BOOL, False, None, None),
        # Same meaning as separation's own keys of the same name: whether
        # strip_music_spans() runs its raw GPU call and CPU postprocessing
        # through persistent background pools (MusicService._async_runtime)
        # instead of doing both inline on the same thread.
        "postprocess_workers": (INT, 1, 1, 16),
        "ordered_postprocess": (BOOL, True, None, None),
    },
}

_GPU_NAMES = ("gpu_1", "gpu_2")

_ENUMS = {
    ("asr", "qwen3_gpu"): _GPU_NAMES,
    ("asr", "whisper_gpu"): _GPU_NAMES,
    ("asr", "phowhisper_gpu"): _GPU_NAMES,
    ("refinement", "placement"): ("auto", "balanced", "sharded", "pipelined"),
    ("diarization", "placement"): ("single", "split_components", "replicated"),
    ("separation", "postprocess_device"): ("cpu", "cuda"),
}

# Keys that change what the pipeline computes, as opposed to how it is
# scheduled. Only these belong in the cache fingerprint: per the plan, GPU ids
# and pure-scheduling worker counts must not invalidate a usable checkpoint.
_SEMANTIC_KEYS = (
    ("asr", "true_batching"),
    ("refinement", "placement"),
    ("refinement", "max_batch_tokens"),
    ("diarization", "placement"),
)


def _coerce(value, kind):
    if kind == BOOL:
        if isinstance(value, bool):
            return value
        raise ValueError("expected true or false")
    if kind == STR:
        if isinstance(value, str):
            return value
        raise ValueError("expected a string")
    if isinstance(value, bool):
        # bool is an int subclass; accepting it here turns `true` into 1.
        raise ValueError("expected a number")
    if kind == INT:
        return int(value)
    return float(value)


def _read(raw, spec, path, problems):
    """Resolve one table of keys, reporting anything unknown or unusable."""
    kind, default, low, high = spec
    if raw is None:
        return default
    try:
        value = _coerce(raw, kind)
    except (TypeError, ValueError) as exc:
        problems.append(f"{path}={raw!r} is not valid ({exc}); using {default!r}")
        return default
    if low is not None and value < low:
        problems.append(f"{path}={value} is below {low}; using {low}")
        return low
    if high is not None and value > high:
        problems.append(f"{path}={value} is above {high}; using {high}")
        return high
    return value


def resolve(env_profile, logger=None, enabled_override=None):
    """Return the effective performance config and log it.

    `enabled_override` is the CLI switch: True forces the feature on for a
    trial run even though the profile ships it off, None leaves the profile
    in charge.
    """
    raw = (env_profile or {}).get("performance")
    if raw is None:
        raw = {}
    problems = []
    if not isinstance(raw, dict):
        problems.append(f"performance must be a table, not {type(raw).__name__}; ignoring it")
        raw = {}

    resolved = {}
    for name, spec in _TOP.items():
        resolved[name] = _read(raw.get(name), spec, f"performance.{name}", problems)

    for name in raw:
        if name not in _TOP and name != "stages":
            problems.append(f"performance.{name} is not a known setting; ignoring it")

    raw_stages = raw.get("stages") or {}
    if not isinstance(raw_stages, dict):
        problems.append("performance.stages must be a table; ignoring it")
        raw_stages = {}
    for name in raw_stages:
        if name not in _STAGES:
            problems.append(f"performance.stages.{name} is not a known stage; ignoring it")

    resolved["stages"] = {}
    for stage, keys in _STAGES.items():
        raw_stage = raw_stages.get(stage) or {}
        if not isinstance(raw_stage, dict):
            problems.append(f"performance.stages.{stage} must be a table; ignoring it")
            raw_stage = {}
        for name in raw_stage:
            if name not in keys:
                problems.append(
                    f"performance.stages.{stage}.{name} is not a known setting; ignoring it")
        table = {}
        for name, spec in keys.items():
            path = f"performance.stages.{stage}.{name}"
            value = _read(raw_stage.get(name), spec, path, problems)
            allowed = _ENUMS.get((stage, name))
            if allowed and value not in allowed:
                problems.append(
                    f"{path}={value!r} must be one of {', '.join(allowed)}; "
                    f"using {keys[name][1]!r}")
                value = keys[name][1]
            table[name] = value
        resolved["stages"][stage] = table

    if resolved["ram_hard_fraction"] < resolved["ram_soft_fraction"]:
        problems.append(
            f"ram_hard_fraction ({resolved['ram_hard_fraction']}) is below "
            f"ram_soft_fraction ({resolved['ram_soft_fraction']}); raising it to match")
        resolved["ram_hard_fraction"] = resolved["ram_soft_fraction"]

    if enabled_override is not None and bool(enabled_override) != resolved["enabled"]:
        resolved["enabled"] = bool(enabled_override)
        problems.append(
            f"--performance overrode the profile; enabled={resolved['enabled']}")

    if logger:
        for problem in problems:
            logger.warning(f"[performance config] {problem}")
        logger.info(
            "[performance config] effective: "
            + json.dumps(resolved, ensure_ascii=False, sort_keys=True))
    resolved["_problems"] = problems
    return resolved


def fingerprint(resolved):
    """Short token for the output path, covering only semantic settings.

    Returned for the disabled case too, so a run with the feature off keeps
    landing in the directory the baseline has always used.
    """
    if not resolved.get("enabled"):
        return "off"
    stages = resolved.get("stages", {})
    payload = {f"{stage}.{name}": stages.get(stage, {}).get(name)
               for stage, name in _SEMANTIC_KEYS}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"on-{digest[:8]}"


def resolve_music_devices(device_1, device_2, *, perf_enabled: bool,
                          max_separator_workers: int = 1,
                          cross_file_overlap: bool = False,
                          logger=None) -> list:
    """Which device(s) BS-RoFormer loads onto.

    SSLAM (the tagger) is not a parameter here -- it always stays on
    device_1, since it is the fast, small model and the one every file's
    music stage starts with.

    Two different reasons ask for a second GPU, and they conflict: splitting
    ONE file's own music spans across two BS-RoFormer instances
    (`max_separator_workers>1`), versus moving the SOLE instance to device_2
    so file N's removal and file N+1's classification land on different
    cards when the batch overlaps them (`cross_file_overlap`). Both want the
    same card for a different job; `cross_file_overlap` wins, because
    without it nothing overlaps files at all, while a lone file with several
    music spans still gets removal done -- just serialized on one card.
    """
    if not perf_enabled or device_2 == device_1:
        return [device_1]
    if cross_file_overlap:
        if max_separator_workers > 1 and logger:
            logger.warning(
                "[performance] music.cross_file_overlap and "
                "music.max_separator_workers>1 both ask for the second GPU "
                "in different ways; using cross_file_overlap (one BS-RoFormer "
                "instance on device_2, off SSLAM's card) and ignoring "
                "max_separator_workers")
        return [device_2]
    if max_separator_workers > 1:
        return [device_1, device_2]
    return [device_1]


def resolve_asr_placement(asr_cfg, gpu_1: int, gpu_2: int) -> dict:
    """Physical GPU id for each ASR model, from the resolved `stages.asr` table."""
    ids = {"gpu_1": int(gpu_1), "gpu_2": int(gpu_2)}
    return {
        "qwen3": ids[asr_cfg["qwen3_gpu"]],
        "whisper": ids[asr_cfg["whisper_gpu"]],
        "phowhisper": ids[asr_cfg["phowhisper_gpu"]],
    }


def pho_worker_devices(placement_gpu, available, count) -> list:
    """GPU id for each PhoWhisper worker: its own card first, then alternating.

    Alternating keeps the pool's first idle leases on different cards so two copies
    do not start on the same one.
    """
    cards = list(dict.fromkeys(available))
    if not cards:
        return []
    if placement_gpu in cards:
        cards.remove(placement_gpu)
        cards.insert(0, placement_gpu)
    return [cards[i % len(cards)] for i in range(max(0, int(count)))]


def sidon_worker_devices(available, requested_gpus, workers_per_gpu, max_gpus,
                         enabled) -> list:
    """GPU id for every Sidon worker process, cards interleaved.

    Interleaving ([0, 1, 0, 1]) keeps the pool's first idle leases alternating
    between cards instead of filling one card before the next.
    """
    if not enabled:
        return list(available)[:1]
    gpus = list(available)[:max(1, min(int(requested_gpus), int(max_gpus)))]
    return [gpu for _ in range(max(1, int(workers_per_gpu))) for gpu in gpus]


def resolve_music_worker_devices(devices, workers_per_gpu, isolate_process,
                                 logger=None) -> list:
    """One entry per BS-RoFormer instance, the cards interleaved ([0, 1, 0, 1]).

    Without process isolation extra instances on a card would share one process
    (and its global SDPA flags), so they are not created.
    """
    per_gpu = max(1, int(workers_per_gpu))
    if per_gpu > 1 and not isolate_process:
        if logger:
            logger.warning(
                f"[performance] music.workers_per_gpu={per_gpu} needs "
                "models.bs_roformer.isolate_process; using one worker per GPU")
        per_gpu = 1
    return [device for _ in range(per_gpu) for device in devices]
