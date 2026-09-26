"""A pool of SSLAM taggers: sweeps of different files run at once, each on a tagger
of its own, and wait in the order they arrived.

Before, one tagger sat behind one lock and every file queued for it, so the
BS-RoFormer instances downstream waited for their next job. With `tagger_workers`
taggers spread over the two cards, and every file in flight, a job reaches the
separators as soon as its file has been swept.

Run:  python -m pytest tests/test_sslam_pool.py -q     (from podcast-pipeline/)
"""
import contextlib
import importlib
import os
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sslam import SSLAMPool
from services.pipeline_service import PipelineService
from utils.batch import UNLIMITED_FILES, _stage_parallelism, run_batch_by_stage


class Fake:
    """A tagger that notices when two sweeps are on it at once."""

    def __init__(self, name="t", delay=0.0, barrier=None, order=None):
        self.name, self.delay, self.barrier, self.order = name, delay, barrier, order
        self.busy = self.max_busy = self.calls = 0
        self.unloaded = False
        self._lock = threading.Lock()

    def tag_framewise(self, audio, sample_rate=16000):
        with self._lock:
            self.busy += 1
            self.max_busy = max(self.max_busy, self.busy)
        if self.order is not None:
            self.order.append(audio)
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        time.sleep(self.delay)
        with self._lock:
            self.busy -= 1
            self.calls += 1
        return {"speech": audio}, 2.0

    def unload(self):
        self.unloaded = True


def _run(pool, n, fn=None):
    results, errors = [None] * n, []

    def sweep(k):
        try:
            results[k] = (fn or (lambda k: pool.tag_framewise(k)))(k)
        except Exception as exc:               # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=sweep, args=(k,)) for k in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors, errors
    return results


# --- the pool ---------------------------------------------------------------------------

def test_as_many_sweeps_run_at_once_as_there_are_taggers():
    """A barrier only opens when all four are inside a tagger together, so a pool
    that ran them one at a time would time the barrier out."""
    barrier = threading.Barrier(4, timeout=5)
    pool = SSLAMPool([Fake(str(k), barrier=barrier) for k in range(4)])
    assert len(_run(pool, 4)) == 4


def test_a_tagger_is_never_on_two_sweeps_at_once():
    taggers = [Fake(str(k), delay=0.01) for k in range(2)]
    _run(SSLAMPool(taggers), 12)
    assert all(t.max_busy == 1 for t in taggers)
    assert sum(t.calls for t in taggers) == 12


def test_more_sweeps_than_taggers_wait_their_turn_and_all_finish():
    results = _run(SSLAMPool([Fake(delay=0.01) for _ in range(3)]), 20)
    assert [r[0]["speech"] for r in results] == list(range(20))


def test_a_pool_of_one_is_a_lone_tagger_that_files_take_turns_on():
    tagger = Fake(delay=0.01)
    _run(SSLAMPool([tagger]), 6)
    assert tagger.max_busy == 1 and tagger.calls == 6


def test_waiting_sweeps_are_served_in_the_order_they_arrived():
    order = []
    tagger = Fake(delay=0.05, order=order)
    pool = SSLAMPool([tagger])
    threads = []
    for k in range(6):
        t = threading.Thread(target=pool.tag_framewise, args=(k,))
        t.start()
        threads.append(t)
        time.sleep(0.02)                       # each arrives after the one before
    for t in threads:
        t.join(10)
    assert order == list(range(6))


def test_a_tagger_is_handed_back_even_when_its_sweep_fails():
    class Breaks(Fake):
        def tag_framewise(self, audio, sample_rate=16000):
            if audio == "bad":
                raise RuntimeError("cuda out of memory")
            return super().tag_framewise(audio, sample_rate)

    pool = SSLAMPool([Breaks()])
    with pytest.raises(RuntimeError):
        pool.tag_framewise("bad")
    assert pool.tag_framewise("good")[0] == {"speech": "good"}


def test_the_pool_answers_tag_framewise_like_one_tagger():
    tagger = Fake()
    assert SSLAMPool([tagger]).tag_framewise([1, 2, 3], 24000) == ({"speech": [1, 2, 3]}, 2.0)


def test_the_pool_says_it_is_safe_to_share_and_a_lone_tagger_does_not():
    from models.sslam import SSLAMDetector
    assert SSLAMPool([Fake()]).shared_safe is True
    assert not getattr(SSLAMDetector, "shared_safe", False)


def test_unloading_the_pool_unloads_every_tagger():
    taggers = [Fake(str(k)) for k in range(4)]
    SSLAMPool(taggers).unload()
    assert all(t.unloaded for t in taggers)


def test_a_pool_needs_a_tagger():
    with pytest.raises(ValueError):
        SSLAMPool([])


# --- the loader builds the pool -----------------------------------------------------------

def _loader(monkeypatch, workers=4, enabled=True, devices=("cuda:0", "cuda:1")):
    """A ModelLoader with every model class stubbed, and a recording SSLAM."""
    built = []

    class Recorded:
        def __init__(self, device=None):
            self.device, self.loaded = device, 0
            built.append(self)

        def _load(self):
            self.loaded += 1
            built_order.append(self)

        def unload(self):
            self.unloaded = True

    built_order = []
    stubs = {
        "models.whisper_wrapper": {"WhisperASR": object},
        "models.phowhisper": {"PhoWhisperASR": object},
        "models.silero_vad": {"SileroVAD": object},
        "models.pyannote": {"PyannoteDiarizer": object},
        "models.diarizen_model": {"DiariZenDiarizer": object},
        "models.bss_model": {"BssSeparator": object},
        "models.sslam": {"SSLAMDetector": Recorded, "SSLAMPool": SSLAMPool},
        "models.qwen3_omni": {"Qwen3OmniCaptioner": object},
        "models.qwen3_asr": {"Qwen3ASRClient": object},
        "services.qwen3_worker_service": {"Qwen3WorkerService": object},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "services.model_loader", raising=False)
    loader_module = importlib.import_module("services.model_loader")
    config = {"environments": {"a100": {"performance": {
        "enabled": enabled, "stages": {"music": {"tagger_workers": workers}}}}}}
    loader = loader_module.ModelLoader(config, SimpleNamespace(gpu_1=0, gpu_2=1, env="a100"))
    loader.device_1, loader.device_2 = devices
    return loader, built, built_order


@pytest.fixture(autouse=True)
def _forget_the_stubbed_loader():
    yield
    sys.modules.pop("services.model_loader", None)


def test_four_taggers_are_loaded_alternating_between_the_two_cards(monkeypatch):
    loader, built, _ = _loader(monkeypatch, workers=4)
    loader.load_tagger()
    assert [d.device for d in built] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert isinstance(loader.models["tagger"], SSLAMPool)
    assert loader.models["tagger"].detectors == built


def test_every_tagger_is_loaded_once_and_one_after_another_before_any_sweep(monkeypatch):
    loader, built, order = _loader(monkeypatch, workers=4)
    loader.load_tagger()
    assert order == built and all(d.loaded == 1 for d in built)


def test_the_number_of_taggers_is_not_capped(monkeypatch):
    loader, built, _ = _loader(monkeypatch, workers=9)
    loader.load_tagger()
    assert len(built) == 9 and len(loader.models["tagger"].detectors) == 9


def test_one_tagger_is_a_lone_detector_not_a_pool(monkeypatch):
    loader, built, _ = _loader(monkeypatch, workers=1)
    loader.load_tagger()
    assert len(built) == 1 and loader.models["tagger"] is built[0]
    assert built[0].device == "cuda:0" and built[0].loaded == 0


def test_with_performance_off_there_is_one_tagger_whatever_the_setting(monkeypatch):
    loader, built, _ = _loader(monkeypatch, workers=4, enabled=False)
    loader.load_tagger()
    assert len(built) == 1 and loader.models["tagger"] is built[0]


def test_on_one_card_every_tagger_goes_on_it(monkeypatch):
    loader, built, _ = _loader(monkeypatch, workers=3, devices=("cuda:0", "cuda:0"))
    loader.load_tagger()
    assert [d.device for d in built] == ["cuda:0"] * 3


def test_two_files_asking_for_the_tagger_at_once_build_the_pool_once(monkeypatch):
    loader, built, _ = _loader(monkeypatch, workers=4)
    threads = [threading.Thread(target=loader.load_tagger) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(built) == 4 and list(loader.models) == ["tagger"]


def test_unloading_the_tagger_unloads_the_whole_pool(monkeypatch):
    loader, built, _ = _loader(monkeypatch, workers=4)
    loader.load_tagger()
    loader.unload("tagger")
    assert "tagger" not in loader.models and all(d.unloaded for d in built)


# --- the pipeline does not add a lock of its own to a pool ------------------------------------

def _pipeline():
    pipe = PipelineService.__new__(PipelineService)
    pipe._tagger_lock = threading.Lock()
    return pipe


def test_a_lone_tagger_is_swept_under_the_lock():
    pipe = _pipeline()
    assert pipe._tagger_guard(Fake()) is pipe._tagger_lock


def test_a_pool_is_swept_without_the_lock_so_files_are_not_put_back_in_one_line():
    pipe = _pipeline()
    guard = pipe._tagger_guard(SSLAMPool([Fake()]))
    assert guard is not pipe._tagger_lock
    with guard:
        assert not pipe._tagger_lock.locked()


def test_with_no_lock_at_all_the_guard_is_still_a_context_manager():
    pipe = PipelineService.__new__(PipelineService)
    with pipe._tagger_guard(Fake()):
        pass


def test_two_sweeps_through_the_pipeline_guard_overlap_on_a_pool():
    barrier = threading.Barrier(2, timeout=5)
    pipe = _pipeline()
    pool = SSLAMPool([Fake(barrier=barrier), Fake(barrier=barrier)])

    def sweep(k):
        with pipe._tagger_guard(pool):
            return pool.tag_framewise(k)

    assert len(_run(pool, 2, sweep)) == 2


# --- every file is in flight at once when there is a pool -----------------------------------------

def _args(**music):
    return SimpleNamespace(performance_config={"enabled": True, "stages": {"music": music}},
                           stop_after=None, dia3=False)


def test_with_several_taggers_the_music_stage_takes_every_file():
    assert _stage_parallelism(_args(tagger_workers=4), "music") == UNLIMITED_FILES
    assert min(_stage_parallelism(_args(tagger_workers=4), "music"), 13) == 13


def test_the_old_two_file_limit_stays_for_a_single_tagger():
    assert _stage_parallelism(_args(tagger_workers=1, max_separator_workers=2), "music") == 2
    assert _stage_parallelism(_args(tagger_workers=1, cross_file_overlap=True), "music") == 2
    assert _stage_parallelism(_args(tagger_workers=1), "music") == 1


def test_with_performance_off_the_music_stage_is_one_file_at_a_time():
    args = SimpleNamespace(performance_config={"enabled": False}, stop_after=None)
    assert _stage_parallelism(args, "music") == 1


def test_thirteen_files_are_all_in_the_music_stage_together_when_there_are_four_taggers():
    """A barrier of thirteen only opens when every file is inside the stage at once."""
    barrier = threading.Barrier(13, timeout=10)
    seen = []

    class Pipe:
        def parallel_stage_view(self, stage):
            return self

        def run(self, args, config, path):
            barrier.wait()
            seen.append(path)

    args = SimpleNamespace(performance_config={"enabled": True, "stages": {
        "music": {"tagger_workers": 4}}}, stop_after=None, dia3=False, env="a100")
    files = [f"/data/{k}.wav" for k in range(13)]
    failures = run_batch_by_stage(Pipe(), args, {}, files, stages=("music",))
    assert failures == [] and sorted(seen) == sorted(files)
