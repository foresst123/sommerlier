"""Waiting for GPU memory that another process is still giving back."""

import time


def _torch_probe(device):
    import torch
    with torch.cuda.device(int(device)):
        return torch.cuda.mem_get_info()


def wait_for_free_vram(devices, fraction, timeout=120.0, poll=2.0, logger=None,
                       probe=None, sleep=time.sleep, clock=time.monotonic) -> bool:
    """Block until every device has `fraction` of its memory free, or `timeout` passes.

    A vLLM engine asks for a share of the card (gpu_memory_utilization) and refuses to
    start when less than that is free -- before loading anything. A worker that has just
    been stopped, or a model still loading onto the card, leaves it short for a while,
    and the engine fails on a condition that clears by itself. Returns True when the
    cards are free, or when they cannot be measured; False on timeout (the caller goes
    on and lets the engine report its own error, with what was in the way in the log).
    """
    probe = probe or _torch_probe
    deadline = clock() + float(timeout)
    last_report = None
    while True:
        short = {}
        for device in devices:
            try:
                free, total = probe(device)
            except Exception:
                continue                      # not measurable: do not hold the run up
            need = fraction * total
            if free < need:
                short[device] = (free, need)
        if not short:
            return True
        now = clock()
        if now >= deadline:
            if logger:
                logger.warning("Starting anyway after waiting %.0fs for VRAM: %s" % (
                    timeout, _describe(short)))
            return False
        if logger and (last_report is None or now - last_report >= 10.0):
            last_report = now
            logger.info("Waiting for VRAM to be freed before starting: %s" % _describe(short))
        sleep(poll)


def _describe(short):
    return "; ".join(
        "GPU %s has %.1f GiB free, needs %.1f GiB" % (device, free / 1024 ** 3,
                                                     need / 1024 ** 3)
        for device, (free, need) in sorted(short.items()))
