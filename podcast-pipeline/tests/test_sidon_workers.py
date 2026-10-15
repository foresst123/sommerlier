"""Several Sidon workers per GPU: the device list, the schema key, and how many
windows the separation stage keeps in flight."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import performance_config as pc
from services.separation_service import SeparationService


def test_one_worker_per_gpu_is_unchanged():
    assert pc.sidon_worker_devices([0, 1], 2, 1, 2, True) == [0, 1]
    assert pc.sidon_worker_devices([0, 1], 1, 1, 2, True) == [0]


def test_workers_per_gpu_interleave_the_cards_so_idle_leases_alternate():
    assert pc.sidon_worker_devices([0, 1], 2, 2, 2, True) == [0, 1, 0, 1]
    assert pc.sidon_worker_devices([0, 1], 2, 3, 2, True) == [0, 1, 0, 1, 0, 1]


def test_the_gpu_limits_still_apply_before_the_per_gpu_count():
    assert pc.sidon_worker_devices([0, 1], 2, 2, 1, True) == [0, 0]     # max_gpus=1
    assert pc.sidon_worker_devices([0], 2, 2, 2, True) == [0, 0]        # one card


def test_with_the_performance_block_off_there_is_a_single_worker():
    assert pc.sidon_worker_devices([0, 1], 2, 2, 2, False) == [0]


def test_the_schema_key_has_a_default_and_bounds():
    profile = {"performance": {"enabled": True, "stages": {"separation": {}}}}
    assert pc.resolve(profile)["stages"]["separation"]["workers_per_gpu"] == 1
    profile["performance"]["stages"]["separation"]["workers_per_gpu"] = 9
    result = pc.resolve(profile)
    assert result["stages"]["separation"]["workers_per_gpu"] == 4
    assert any("workers_per_gpu" in problem for problem in result["_problems"])


def test_the_new_key_does_not_change_the_output_fingerprint():
    def fingerprint(per_gpu):
        return pc.fingerprint(pc.resolve({"performance": {"enabled": True, "stages": {
            "separation": {"workers_per_gpu": per_gpu}}}}))
    assert fingerprint(1) == fingerprint(3)


def test_the_gpu_executor_is_sized_by_the_number_of_sidon_workers():
    service = SeparationService(
        logger=None, performance_config={"enabled": True, "max_workers": 2,
                                         "gpu_workers": 4, "postprocess_workers": 2})

    class _Model:
        def separate_raw(self, audio, sr): ...
        def postprocess_separated(self, raw, audio, sr): ...

    service.bss_model = _Model()
    gpu_executor, _post, _model = service._async_runtime()
    try:
        assert gpu_executor._max_workers == 4
    finally:
        service.close_async_pools()


def test_without_gpu_workers_the_old_max_workers_still_sizes_it():
    service = SeparationService(
        logger=None, performance_config={"enabled": True, "max_workers": 2,
                                         "postprocess_workers": 2})

    class _Model:
        def separate_raw(self, audio, sr): ...
        def postprocess_separated(self, raw, audio, sr): ...

    service.bss_model = _Model()
    gpu_executor, _post, _model = service._async_runtime()
    try:
        assert gpu_executor._max_workers == 2
    finally:
        service.close_async_pools()
