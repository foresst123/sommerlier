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
        "ordered_postprocess": (BOOL, True, None, None),
    },
    "music": {
        "tagger_workers": (INT, 1, 1, 2),
        "max_separator_workers": (INT, 1, 1, 2),
    },
}

_ENUMS = {
    ("refinement", "placement"): ("auto", "balanced", "sharded", "pipelined"),
    ("diarization", "placement"): ("single", "split_components", "replicated"),
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
