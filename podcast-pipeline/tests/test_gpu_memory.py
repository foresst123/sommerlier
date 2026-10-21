"""A vLLM engine that asks for a share of a card fails at once when less is free, so the
refinement stage waits for the cards to empty first. Fake probe and clock, no GPU."""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu_memory import wait_for_free_vram

GIB = 1024 ** 3
TOTAL = 40 * GIB


class World:
    """Time and per-device free memory that changes as time passes."""

    def __init__(self, schedule):
        self.now = 0.0
        self.schedule = schedule            # {device: [(from_time, free_bytes), ...]}
        self.probes = 0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def probe(self, device):
        self.probes += 1
        free = [f for t, f in self.schedule[device] if t <= self.now][-1]
        return free, TOTAL


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def _logger():
    logger = logging.getLogger(f"gpu-memory-{id(object())}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = Capture()
    logger.addHandler(handler)
    return logger, handler


def _wait(world, devices, **kwargs):
    logger, handler = _logger()
    ok = wait_for_free_vram(
        devices, 0.90, timeout=kwargs.pop("timeout", 60), poll=2.0, logger=logger,
        probe=world.probe, sleep=world.sleep, clock=world.clock, **kwargs)
    return ok, handler.records


def test_free_cards_do_not_wait_at_all():
    world = World({0: [(0, 39 * GIB)], 1: [(0, 38 * GIB)]})
    ok, records = _wait(world, [0, 1])
    assert ok and world.now == 0.0 and not records


def test_it_waits_until_a_card_that_is_still_busy_empties():
    world = World({0: [(0, 39 * GIB)], 1: [(0, 15 * GIB), (10, 39 * GIB)]})
    ok, records = _wait(world, [0, 1])

    assert ok and 10 <= world.now < 14
    assert any("GPU 1" in message and "15.0" in message for _lvl, message in records)


def test_it_gives_up_after_the_timeout_and_says_what_was_missing():
    world = World({0: [(0, 12 * GIB)]})
    ok, records = _wait(world, [0], timeout=20)

    assert ok is False and world.now >= 20
    level, message = records[-1]
    assert level == "WARNING" and "GPU 0" in message and "36.0" in message


def test_the_requirement_is_the_share_of_the_card_the_engine_will_ask_for():
    # 0.90 of 40 GiB is 36 GiB: 35.9 is not enough, 36.0 is.
    assert _wait(World({0: [(0, int(35.9 * GIB))]}), [0], timeout=4)[0] is False
    assert _wait(World({0: [(0, 36 * GIB)]}), [0])[0] is True


def test_a_card_that_cannot_be_probed_does_not_block_the_run():
    def broken(_device):
        raise RuntimeError("no cuda")

    logger, _ = _logger()
    assert wait_for_free_vram([0], 0.9, probe=broken, logger=logger) is True


def test_refinement_waits_for_the_cards_before_it_starts_its_workers():
    source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "services", "diarization_refinement_service.py"),
        encoding="utf-8").read()
    body = source[source.index("def _ensure_worker_started"):]
    assert "wait_for_free_vram(" in body
    assert body.index("wait_for_free_vram(") < body.index("self._worker.start()")
