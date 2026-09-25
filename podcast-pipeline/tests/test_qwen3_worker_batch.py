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
    import re
    assert re.search(r"load_model\(\s*args\.config, args\.env, args\.batch_size\)", source)


class _QwenAsrLike:
    """Accepts what qwen_asr accepts: a path or a (waveform, sample_rate) pair."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, language):
        import numpy as np
        self.calls += 1
        items = audio if isinstance(audio, list) else [audio]
        for item in items:
            if not (isinstance(item, tuple) and isinstance(item[0], np.ndarray)):
                raise TypeError(f"Unsupported audio input type: {type(item)}")
        return [type("Out", (), {"text": " xin chao "})() for _ in items]


def _npy(tmp_path, name):
    import numpy as np
    path = tmp_path / name
    np.save(path, np.zeros(1600, dtype=np.float32))
    return str(path)


def test_the_vllm_batch_hands_qwen_asr_waveform_and_rate_pairs(tmp_path):
    import qwen3_worker
    model = _QwenAsrLike()
    jobs = [{"id": "a", "audio_path": _npy(tmp_path, "a.npy")},
            {"id": "b", "audio_path": _npy(tmp_path, "b.npy")}]

    results = qwen3_worker.transcribe_batch(model, None, "cpu", jobs, "vi", "vllm")

    assert results == [{"id": "a", "text": "xin chao"}, {"id": "b", "text": "xin chao"}]
    assert model.calls == 1   # one batch, not a per-clip fallback


def test_the_vllm_single_clip_path_does_the_same(tmp_path):
    import qwen3_worker
    text = qwen3_worker.transcribe(
        _QwenAsrLike(), None, "cpu", _npy(tmp_path, "a.npy"), "vi", "vllm")

    assert text == "xin chao"
