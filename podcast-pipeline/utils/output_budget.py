"""Total hours of exported conversation audio, recomputed from manifests.

Each file's conversation_exports/manifest.jsonl is rewritten wholesale on every
run of that file, so totals are summed fresh, never incremented. The quarantine
manifest is a separate file and is not counted.
"""
import json
import math
import os
from typing import Callable, Iterable, Optional

MANIFEST_DIR = "conversation_exports"
MANIFEST_FILE = "manifest.jsonl"


def conversation_seconds(manifest_path: str) -> float:
    """Seconds of exported audio listed in one manifest; 0.0 if it cannot be read.

    A manifest cut off mid-write can end in a half line or bad bytes. Counting
    what is readable beats failing a run over the accounting.
    """
    total = 0.0
    try:
        with open(manifest_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                d = row.get("duration")
                if isinstance(d, bool) or not isinstance(d, (int, float)):
                    continue
                try:
                    d = float(d)
                except OverflowError:
                    continue
                if not math.isfinite(d) or d < 0:
                    continue
                total += d
    except OSError:
        return total
    return total


def total_conversation_hours(paths: Iterable[str], out_dir_of: Callable[[str], str]) -> float:
    seconds = 0.0
    for path in paths:
        try:
            out_dir = out_dir_of(path)
        except Exception:
            continue
        seconds += conversation_seconds(os.path.join(out_dir, MANIFEST_DIR, MANIFEST_FILE))
    return seconds / 3600.0


def target_reached(total_hours: float, target_hours: Optional[float]) -> bool:
    if target_hours is None or target_hours <= 0:
        return False
    return total_hours >= target_hours


def resolve_target(cli_value, config_value) -> Optional[float]:
    """The target in hours for this run, or None when it is off.

    The command line wins over the profile. Unset, null, zero, negative, NaN and
    infinity all mean off; text that is not a number is a configuration mistake
    and raises, rather than silently running without the limit.
    """
    value = cli_value if cli_value is not None else config_value
    if value is None or isinstance(value, bool):
        return None
    try:
        hours = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"target_output_hours must be a number of hours, got {value!r}")
    return hours if math.isfinite(hours) and hours > 0 else None


def enforced(target_hours: Optional[float], partial_run: bool) -> bool:
    """Whether the target stops this run. Runs that stop early (stop_after) only
    stage the work, so they never count towards or against it."""
    return target_hours is not None and not partial_run
