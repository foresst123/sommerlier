"""PhoWhisper runs in its own worker process(es): a batch travels as one .npy of
concatenated clips plus their lengths, and comes back as one text per clip.
The model is a fake; the real one needs a GPU."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phowhisper_worker
from models.phowhisper_client import PhoWhisperClient
from services.phowhisper_worker_service import PhoWhisperWorkerService


class FakeModel:
    """Echoes each clip's length, so a mixed-up batch is visible in the text."""

    def __init__(self):
        self.calls = []

    def transcribe_batch(self, arrays, batch_size=None, logger=None, callback=None):
        self.calls.append((len(arrays), batch_size))
        return [f"len{len(a)}" for a in arrays]


class DirectEndpoint:
    """Stands in for the worker process: hands the request to the real handler."""

    def __init__(self, model):
        self.model, self.requests, self.stopped = model, [], False

    def request(self, payload, *, response_id=None):
        self.requests.append(payload)
        return phowhisper_worker.handle_request(self.model, payload)

    def stop(self):
        self.stopped = True


def _clips(*lengths):
    return [np.full(n, 0.1, dtype=np.float32) for n in lengths]


# --- the worker side -------------------------------------------------------------

def test_the_worker_splits_the_carrier_back_into_the_original_clips(tmp_path):
    model = FakeModel()
    carrier = np.concatenate(_clips(100, 250, 40))
    path = tmp_path / "batch.npy"
    np.save(path, carrier)

    reply = phowhisper_worker.handle_request(model, {
        "cmd": "transcribe_batch", "id": "r1", "audio_path": str(path),
        "lengths": [100, 250, 40], "ids": ["a", "b", "c"], "batch_size": 8})

    assert reply == {"id": "r1", "results": [
        {"id": "a", "text": "len100"}, {"id": "b", "text": "len250"},
        {"id": "c", "text": "len40"}]}
    assert model.calls == [(3, 8)]


def test_a_batch_that_does_not_add_up_is_an_error_not_a_guess(tmp_path):
    path = tmp_path / "batch.npy"
    np.save(path, np.zeros(10, dtype=np.float32))
    reply = phowhisper_worker.handle_request(FakeModel(), {
        "cmd": "transcribe_batch", "id": "r", "audio_path": str(path),
        "lengths": [4, 4], "ids": ["a", "b"]})
    assert "error" in reply and reply["id"] == "r"


def test_a_missing_file_and_an_unknown_command_are_reported():
    assert "error" in phowhisper_worker.handle_request(FakeModel(), {
        "cmd": "transcribe_batch", "id": "x", "audio_path": "/nope.npy",
        "lengths": [1], "ids": ["a"]})
    assert "error" in phowhisper_worker.handle_request(FakeModel(), {"cmd": "dance", "id": "y"})


def test_a_model_failure_becomes_an_error_reply(tmp_path):
    class Broken(FakeModel):
        def transcribe_batch(self, *args, **kwargs):
            raise RuntimeError("cuda out of memory")

    path = tmp_path / "b.npy"
    np.save(path, np.zeros(4, dtype=np.float32))
    reply = phowhisper_worker.handle_request(Broken(), {
        "cmd": "transcribe_batch", "id": "z", "audio_path": str(path),
        "lengths": [4], "ids": ["a"]})
    assert "cuda out of memory" in reply["error"]


# --- the client side ----------------------------------------------------------------

def test_the_client_sends_chunks_of_batch_size_and_keeps_the_order():
    model = FakeModel()
    client = PhoWhisperClient(DirectEndpoint(model), batch_size=2)

    texts = client.transcribe_batch(_clips(10, 20, 30, 40, 50))

    assert texts == ["len10", "len20", "len30", "len40", "len50"]
    assert [count for count, _size in model.calls] == [2, 2, 1]


def test_an_explicit_batch_size_overrides_the_configured_one():
    model = FakeModel()
    client = PhoWhisperClient(DirectEndpoint(model), batch_size=2)
    client.transcribe_batch(_clips(1, 2, 3, 4), batch_size=4)
    assert [count for count, _size in model.calls] == [4]


def test_the_callback_fires_once_per_clip():
    seen = []
    PhoWhisperClient(DirectEndpoint(FakeModel()), batch_size=2).transcribe_batch(
        _clips(1, 2, 3), callback=lambda: seen.append(1))
    assert len(seen) == 3


def test_a_worker_error_reaches_the_caller():
    class Endpoint(DirectEndpoint):
        def request(self, payload, *, response_id=None):
            return {"id": payload["id"], "error": "boom"}

    with pytest.raises(RuntimeError, match="boom"):
        PhoWhisperClient(Endpoint(FakeModel())).transcribe_batch(_clips(5))


def test_temporary_files_are_removed_even_when_the_worker_fails():
    import glob
    import tempfile
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "sommelier_pho_*.npy")))

    class Endpoint(DirectEndpoint):
        def request(self, payload, *, response_id=None):
            raise RuntimeError("worker died")

    with pytest.raises(RuntimeError):
        PhoWhisperClient(Endpoint(FakeModel())).transcribe_batch(_clips(5, 6))
    PhoWhisperClient(DirectEndpoint(FakeModel())).transcribe_batch(_clips(5, 6))

    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "sommelier_pho_*.npy")))
    assert after == before


def test_an_empty_batch_sends_nothing():
    endpoint = DirectEndpoint(FakeModel())
    assert PhoWhisperClient(endpoint).transcribe_batch([]) == []
    assert endpoint.requests == []


def test_a_single_clip_and_closing():
    endpoint = DirectEndpoint(FakeModel())
    client = PhoWhisperClient(endpoint)
    assert client.transcribe(_clips(7)[0]) == "len7"
    client.unload()
    assert endpoint.stopped


# --- the service ---------------------------------------------------------------------

def test_the_service_pins_one_gpu_and_only_trusts_the_json_ready_line():
    service = PhoWhisperWorkerService("/usr/bin/python3", "phowhisper_worker.py",
                                      device_id=1, env_name="a100", config_path="c.json")
    assert service.cuda_visible_devices == "1"
    assert service.ready_requires_json is True
    assert service.extra_args == ["--config", "c.json", "--env", "a100"]
    assert not service.is_ready_line("model ready soon")
    assert service.is_ready_line('{"status": "ready"}')


# --- wiring: loader, pipeline, workers ------------------------------------------------

def _loader(config=None):
    import types
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_steps import _stub_model_modules     # the loader imports every model wrapper
    ModelLoader = _stub_model_modules()
    loader = ModelLoader.__new__(ModelLoader)
    loader.models = {}
    loader.logger = None
    loader.args = types.SimpleNamespace(ASRMoE=False, env="a100", lang="vi")
    loader.config = config or {"environments": {"a100": {"models": {
        "phowhisper": {"batch_size": 12}}}}}
    loader.asr_placement = None
    loader.device_1 = loader.device_2 = "cpu"
    return loader


def test_a_pho_worker_replaces_the_in_process_model_and_builds_no_weights():
    endpoint = DirectEndpoint(FakeModel())
    loader = _loader()
    loader.load_asr_models(phowhisper_service=endpoint)

    client = loader.get("phowhisper")
    assert isinstance(client, PhoWhisperClient) and client.batch_size == 12
    assert client.endpoint is endpoint


def test_the_asr_stage_starts_the_pho_worker_with_the_others():
    from services.pipeline_service import PipelineService
    assert PipelineService.WORKERS_FOR_STAGE["asr"] == ("qwen3", "whisper", "phowhisper")


def _pipeline_with(services, cross_file):
    import types
    from services.pipeline_service import PipelineService
    pipe = PipelineService.__new__(PipelineService)
    pipe.logger = None
    pipe.model_loader = None
    pipe.worker_services = services
    pipe.asr_svc = types.SimpleNamespace(cross_file_enabled=cross_file)
    pipe._register_worker_pids = lambda name, service: []
    return pipe


class _SlowWorker:
    def __init__(self):
        import threading
        self.process = None
        self.gate = threading.Event()
        self.readied = threading.Event()

    def spawn(self):
        self.process = object()

    def wait_ready(self):
        self.gate.wait(5)
        self.readied.set()


def test_without_waiting_the_workers_are_spawned_and_come_up_in_the_background():
    slow = _SlowWorker()
    pipe = _pipeline_with({"pho": slow}, cross_file=True)

    processes = pipe._ensure_workers(("pho",), wait=False)

    assert processes["pho"] is slow.process and not slow.readied.is_set()
    slow.gate.set()
    assert slow.readied.wait(2)


# --- the real worker protocol, over real processes (only the model is fake) -------------

_WRAPPER = '''
import sys
sys.path.insert(0, {root!r})
import phowhisper_worker as w

class Model:
    def transcribe_batch(self, arrays, batch_size=None, logger=None, callback=None):
        print("noise on stdout that must not break the protocol")
        return ["len%d" % len(a) for a in arrays]

w.build_model = lambda config, env: Model()
w.serve()
'''


def _real_services(tmp_path, count):
    script = tmp_path / "fake_pho_worker.py"
    script.write_text(_WRAPPER.format(root=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    return [PhoWhisperWorkerService(sys.executable, str(script), device_id=None)
            for _ in range(count)]


def test_a_real_worker_process_serves_batches_and_ignores_library_prints(tmp_path):
    (service,) = _real_services(tmp_path, 1)
    service.spawn()
    try:
        service.wait_ready()
        client = PhoWhisperClient(service, batch_size=2)
        assert client.transcribe_batch(_clips(11, 22, 33)) == ["len11", "len22", "len33"]
    finally:
        service.stop()


def test_a_pool_of_real_workers_serves_concurrent_batches(tmp_path):
    import threading
    from services.worker_pool_service import WorkerPoolService
    pool = WorkerPoolService(_real_services(tmp_path, 2), name="PhoWhisper")
    pool.spawn()
    try:
        pool.wait_ready()
        client = PhoWhisperClient(pool, batch_size=3)
        results = {}

        def run(tag):
            results[tag] = client.transcribe_batch(_clips(*[tag + i for i in range(6)]))

        threads = [threading.Thread(target=run, args=(tag,)) for tag in (100, 200, 300)]
        [t.start() for t in threads]
        [t.join(30) for t in threads]
    finally:
        pool.stop()

    for tag in (100, 200, 300):
        assert results[tag] == [f"len{tag + i}" for i in range(6)]
    assert sum(item["calls"] for item in pool.profile()["workers"]) == 6
