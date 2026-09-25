"""A Qwen3 replica starts with the boost batch size instead of the profile's."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.qwen3_worker_service import Qwen3WorkerService

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _service(**kwargs):
    return Qwen3WorkerService("/usr/bin/python3", "qwen3_worker.py", device_id=0, **kwargs)


def test_a_worker_without_an_override_gets_no_batch_argument():
    assert "--batch-size" not in _service().extra_args


def test_a_replica_passes_its_batch_size_to_the_worker():
    args = _service(batch_size=48).extra_args
    assert args[args.index("--batch-size") + 1] == "48"


def test_the_worker_script_applies_the_override_before_building_the_engine():
    source = open(os.path.join(ROOT, "qwen3_worker.py"), encoding="utf-8").read()
    assert '"--batch-size"' in source
    assert 'qwen_cfg["batch_size"] = int(batch_size)' in source
    assert "load_model(args.config, args.env, args.batch_size)" in source
