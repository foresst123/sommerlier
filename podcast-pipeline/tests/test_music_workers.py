"""Several BS-RoFormer worker processes per GPU."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import performance_config as pc


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, **_):
        self.warnings.append(message)


def test_workers_per_gpu_interleave_the_cards():
    assert pc.resolve_music_worker_devices([0, 1], 3, True) == [0, 1, 0, 1, 0, 1]
    assert pc.resolve_music_worker_devices([0, 1], 1, True) == [0, 1]
    assert pc.resolve_music_worker_devices([1], 2, True) == [1, 1]     # cross_file_overlap


def test_more_than_one_per_gpu_needs_separate_processes():
    logger = _Logger()
    assert pc.resolve_music_worker_devices([0, 1], 3, False, logger) == [0, 1]
    assert logger.warnings and "isolate_process" in logger.warnings[0]


def test_one_per_gpu_never_warns():
    logger = _Logger()
    assert pc.resolve_music_worker_devices([0, 1], 1, False, logger) == [0, 1]
    assert logger.warnings == []


def test_the_schema_key_has_a_default_and_bounds():
    def resolved(value=None):
        music = {} if value is None else {"workers_per_gpu": value}
        return pc.resolve({"performance": {"enabled": True, "stages": {"music": music}}})

    assert resolved()["stages"]["music"]["workers_per_gpu"] == 1
    result = resolved(9)
    assert result["stages"]["music"]["workers_per_gpu"] == 4
    assert any("workers_per_gpu" in problem for problem in result["_problems"])


def test_the_new_key_does_not_change_the_output_fingerprint():
    def fingerprint(per_gpu):
        return pc.fingerprint(pc.resolve({"performance": {"enabled": True, "stages": {
            "music": {"workers_per_gpu": per_gpu}}}}))
    assert fingerprint(1) == fingerprint(3)


def test_the_loader_expands_the_device_list_when_processes_are_isolated():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "services", "model_loader.py"), encoding="utf-8").read()
    assert "resolve_music_worker_devices(" in source
    assert 'music_perf.get("workers_per_gpu"' in source
