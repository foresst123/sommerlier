"""Checks made before a run starts: what an earlier run left behind, and how busy the box is.

Two things made earlier measurements meaningless without anyone noticing: a vLLM engine
process that outlived its worker and held 12 GiB of a card, and a load average of 167 on
24 usable cores. Both are cheap to see at start-up and expensive to discover afterwards.

Nothing here can stop a run: a check that fails is reported and skipped. Processes are only
killed when asked (`--kill_orphans`), and only ones this user owns that were re-parented to
init, i.e. whose parent is gone.
"""

import os
import signal

# What an orphan is recognised by: vLLM's engine process title, or one of our worker scripts.
_ORPHAN_MARKERS = (
    "EngineCore", "sidon_worker.py", "assignment_worker.py", "phowhisper_worker.py",
    "qwen3_worker.py", "whisper_vllm_worker.py", "refinement_worker.py",
    "diarizen_worker.py", "word_align_worker.py",
)

# Above this many runnable tasks per usable core the run is competing for the CPU.
LOAD_PER_CORE_WARN = 2.0

# A process younger than this may be a worker that is starting right now, not a leftover.
MIN_ORPHAN_AGE_SECONDS = 60


def _process_table():
    """Every visible process as a dict; skips the ones that vanish or refuse to be read."""
    import psutil
    import time
    now = time.time()
    table = []
    for proc in psutil.process_iter(["pid", "ppid", "cmdline", "username", "create_time"]):
        info = proc.info
        table.append({
            "pid": info["pid"], "ppid": info["ppid"],
            "cmdline": info.get("cmdline") or [],
            "username": info.get("username") or "",
            "age": now - (info.get("create_time") or now),
        })
    return table


def find_orphans(table, user, min_age=MIN_ORPHAN_AGE_SECONDS):
    """This user's leftover engines and workers: parent gone (ppid 1), old enough."""
    found = []
    for proc in table:
        if proc["username"] != user or proc["ppid"] != 1 or proc["age"] < min_age:
            continue
        text = " ".join(str(part) for part in proc["cmdline"])
        if any(marker in text for marker in _ORPHAN_MARKERS):
            found.append(proc)
    return sorted(found, key=lambda p: p["pid"])


def load_warning(load, cores):
    """A sentence when the 1-minute load average says the CPU is saturated, else None."""
    if cores > 0 and load > LOAD_PER_CORE_WARN * cores:
        return (f"the load average is {load:.0f} on {cores} usable core(s): other processes "
                "are using the CPU, so every CPU-bound step (window building, VAD, ROVER, "
                "resampling, ffmpeg) will be slower and any timing from this run is noisy")
    return None


def _kill(pid):
    os.kill(pid, signal.SIGKILL)


def run(logger, table=None, user=None, load=None, cores=None, kill=_kill,
        kill_orphans=False):
    """Report (and optionally remove) leftovers, and warn about a saturated machine."""
    try:
        if user is None:
            import getpass
            user = getpass.getuser()
        processes = table() if callable(table) else (table if table is not None
                                                     else _process_table())
        orphans = find_orphans(processes, user)
        if orphans:
            names = ", ".join(f"{p['pid']} ({os.path.basename(str(p['cmdline'][0]))})"
                              if p["cmdline"] else str(p["pid"]) for p in orphans)
            logger.warning(
                f"[preflight] {len(orphans)} leftover process(es) from an earlier run are "
                f"still alive and may hold GPU memory: {names}. "
                f"Remove them with: kill -9 {' '.join(str(p['pid']) for p in orphans)}")
            if kill_orphans:
                removed = 0
                for proc in orphans:
                    try:
                        kill(proc["pid"])
                        removed += 1
                    except OSError as exc:
                        logger.warning(f"[preflight] could not kill {proc['pid']}: {exc}")
                logger.warning(f"[preflight] Killed {removed} of {len(orphans)} leftover "
                               "process(es) (--kill_orphans)")
        if load is None:
            load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        if cores is None:
            from utils.cpu_plan import usable_cores
            cores = usable_cores()
        warning = load_warning(load, cores)
        if warning:
            logger.warning(f"[preflight] {warning}")
        if not orphans and not warning:
            logger.info("[preflight] no leftover processes; machine load is normal")
    except Exception as exc:                      # never let a diagnostic stop the run
        logger.warning(f"[preflight] could not check the machine ({type(exc).__name__}: {exc})")
