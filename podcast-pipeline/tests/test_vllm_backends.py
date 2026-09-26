import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_a100_uses_one_shared_vllm_environment_for_all_three_models():
    config = json.loads((ROOT / "config.json").read_text())
    models = config["environments"]["a100"]["models"]

    assert models["qwen3"]["backend"] == "vllm"
    assert models["qwen3"]["model_name"] == "Qwen/Qwen3-ASR-1.7B"
    assert models["whisper"]["backend"] == "vllm"
    assert models["whisper"]["model_name"] == "openai/whisper-large-v3"
    assert models["refinement"]["backend"] == "vllm"
    assert models["refinement"]["model_name"] == "Qwen/Qwen3.5-9B"

    main_source = (ROOT / "main.py").read_text()
    assert '"vllm", config=config' in main_source
    assert "qwen_env" in main_source


def test_whisper_vllm_client_clips_vad_and_maps_results_by_id():
    from models.whisper_vllm import WhisperVLLMClient

    class Endpoint:
        def request(self, payload):
            assert payload["language"] == "vi"
            lengths = {job["id"]: len(np.load(job["audio_path"]))
                       for job in payload["jobs"]}
            assert lengths == {"0": 16000, "1": 8000}
            return {"results": [
                {"id": "1", "text": "hai", "language": "vi"},
                {"id": "0", "text": "một", "language": "vi"},
            ]}

    client = WhisperVLLMClient(Endpoint(), batch_size=2)
    results = client.transcribe_batch(
        [np.zeros(32000, np.float32), np.zeros(16000, np.float32)],
        [[{"start": 0.5, "end": 1.5}], [{"start": 0.0, "end": 0.5}]],
        language="vi")

    assert [item["text"] for item in results] == ["một", "hai"]
    assert all(item["words"] == [] for item in results)


def test_refinement_vllm_pool_uses_both_replicas_and_preserves_order():
    import threading

    from services.refinement_worker_service import RefinementWorkerPoolService

    class Replica:
        def __init__(self, name):
            self.name = name
            self.model_name = "Qwen/Qwen3.5-9B"
            self.process = object()
            self.seen = []

        def generate_texts(self, system_prompt, messages, **kwargs):
            self.seen.extend(messages)
            # Both replicas must be busy at the same time. A real generate takes
            # seconds; without this the first thread would take every chunk off the
            # queue before the second one woke up, and the test would prove nothing.
            barrier.wait(timeout=5)
            return True, [f"{self.name}:{message}" for message in messages]

        def ping(self):
            return True

        def stop(self):
            self.process = None

    barrier = threading.Barrier(2)
    left, right = Replica("gpu0"), Replica("gpu1")
    pool = RefinementWorkerPoolService([left, right])
    ok, texts = pool.generate_texts("system", ["a", "b", "c", "d"])

    assert ok
    # Replicas pull work from one queue, so which of them answers a given message is
    # not fixed any more; what must hold is that the replies come back in the order
    # they were asked, every message is answered once, and both replicas are used.
    assert [text.split(":", 1)[1] for text in texts] == ["a", "b", "c", "d"]
    assert sorted(left.seen + right.seen) == ["a", "b", "c", "d"]
    assert left.seen and right.seen
