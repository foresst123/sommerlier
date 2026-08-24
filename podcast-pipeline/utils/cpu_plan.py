"""Deciding how many CPU threads each process may use.

The pipeline runs four processes at once -- the main one plus the DiariZen,
Qwen3 and Sidon workers -- and every one of them links against OpenMP/MKL
through torch. Left alone, each grabs a thread per visible core, so four
processes on sixteen cores spawn sixty-four threads that fight over sixteen.
On the two-core allocation this pipeline was last run on, that contention made
things slower than single-threaded.

Nothing here speeds up a process in isolation. It stops them slowing each other
down, which on an oversubscribed box is the larger effect.
"""
import os


def usable_cores() -> int:
    """Cores this process may actually run on.

    os.cpu_count() reports the machine, not the allocation: a SLURM job pinned
    to 2 cores on a 128-core node still sees 128. Prefer the scheduler's own
    answer, then the CPU affinity mask, and only fall back to the machine count.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = os.environ.get(var)
        if raw and raw.isdigit() and int(raw) > 0:
            return int(raw)
    try:
        return len(os.sched_getaffinity(0))          # respects taskset/cgroup
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def thread_plan(n_workers: int = 3, reserve_for_main: bool = True) -> dict:
    """Threads per process, as environment variables to hand to subprocesses.

    `n_workers` is how many worker subprocesses will run alongside the main
    process. The split is deliberately conservative: a worker that is idle
    waiting on the GPU costs nothing by holding fewer threads, whereas one that
    oversubscribes costs every other process.
    """
    cores = usable_cores()
    processes = n_workers + (1 if reserve_for_main else 0)
    per_process = max(1, cores // max(1, processes))

    return {
        "cores_detected": cores,
        "processes": processes,
        "per_process": per_process,
        "env": {
            "OMP_NUM_THREADS": str(per_process),
            "MKL_NUM_THREADS": str(per_process),
            "OPENBLAS_NUM_THREADS": str(per_process),
            "NUMEXPR_NUM_THREADS": str(per_process),
            # Tokenizers spawn their own pool and warn loudly when forked;
            # it is parallel per call, so it does not need the whole box.
            "RAYON_NUM_THREADS": str(per_process),
        },
    }


def apply_to_env(env: dict, per_process: int) -> dict:
    """Write the thread limits into `env`, leaving anything already set alone."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        env.setdefault(key, str(per_process))
    return env


def configure_process(n_workers: int = 3, logger=None) -> int:
    """Apply the plan to this process and return its thread budget.

    Must run before torch is imported: torch reads OMP_NUM_THREADS at import
    and caches it, so setting it afterwards has no effect on the OpenMP pool.
    """
    plan = thread_plan(n_workers)
    apply_to_env(os.environ, plan["per_process"])

    try:
        import torch
        torch.set_num_threads(plan["per_process"])
        # Inter-op is for running independent ops concurrently; with this many
        # processes already competing, one is the right number.
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        # set_num_interop_threads raises if any parallel work already started.
        pass

    if logger:
        logger.info(
            f"CPU: {plan['cores_detected']} core(s) usable, "
            f"{plan['per_process']} thread(s) per process "
            f"across {plan['processes']} process(es)"
        )
    return plan["per_process"]
