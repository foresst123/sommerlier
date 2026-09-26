"""ASRService in cross-file mode: same transcripts as the per-file path, several
files through shared model lanes, and VRAM released once no more files will come.
Fake models only -- no GPU."""

import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.segment import SpeechSegment
from services.asr_service import ASRService

SR = 16000


class FakeWhisper:
    batch_size = 16

    def transcribe_batch(self, audios, vads, language="vi", batch_size=None, callback=None):
        rows = []
        for audio in audios:
            rows.append({"text": f"w{len(audio)} xin chào bạn",
                         "language": language, "words": []})
            if callback:
                callback()
        return rows

    def transcribe(self, audio, vad, language="vi"):
        return {"text": f"w{len(audio)} xin chào bạn", "language": language,
                "words": []}


class FakePho:
    batch_size = 16

    def transcribe_batch(self, audios, batch_size=None, logger=None, callback=None):
        out = []
        for audio in audios:
            out.append(f"p{len(audio)} xin chào bạn")
            if callback:
                callback()
        return out


class FakeQwenClient:
    def transcribe_batch(self, jobs, language="vi"):
        return [f"q{len(np.load(path))} xin chào bạn" for _id, path in jobs]


class FakeLoader:
    def __init__(self):
        self.unloaded = []

    def get(self, name):
        return None

    def unload(self, name):
        self.unloaded.append(name)


class FakeWorker:
    process = object()

    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def _segments(count=3, length=2.0):
    """Segments `length` seconds long: the fakes echo the clip length, so each
    file (its own length) is recognisable in the transcripts."""
    return [SpeechSegment(index=f"{i:05d}", start=i * 5.0, end=i * 5.0 + length,
                          speaker="SPEAKER_00") for i in range(count)]


def _audio(level=0.2, seconds=30):
    return SimpleNamespace(waveform=np.full(seconds * SR, level, dtype=np.float32),
                           sample_rate=SR)


def _service(cross_file, loader=None, keep_models=False, workers=None, **cfg):
    performance = {"cross_file": cross_file, "shared_batch_size": 16,
                   "boost_batch_size": 48, **cfg}
    svc = ASRService(
        whisper=FakeWhisper(), phowhisper=FakePho(), qwen3=FakeQwenClient(),
        model_loader=loader, performance_config=performance,
        batch_size=8, keep_models=keep_models, edge_pad=0.0,
        asr_workers=workers, asr_placement={"qwen3": 0, "whisper": 1, "phowhisper": 1})
    return svc


def _summary(transcripts):
    return [(t.index, t.text, t.text_whisper, t.text_phowhisper, t.text_qwen3, t.language)
            for t in transcripts]


def test_cross_file_transcripts_match_the_per_file_path():
    legacy = _service(cross_file=False).process(_segments(), _audio(0.2))

    svc = _service(cross_file=True)
    svc.begin_cross_file_stage(1)
    try:
        cross = svc.process(_segments(), _audio(0.2))
        svc.settle_file()
    finally:
        svc.end_cross_file_stage()

    assert len(cross) == 3
    assert _summary(cross) == _summary(legacy)


def test_cross_file_mode_is_off_until_a_stage_begins():
    svc = _service(cross_file=True)
    assert svc.cross_file_enabled is False
    svc.begin_cross_file_stage(2)
    assert svc.cross_file_enabled is True
    svc.end_cross_file_stage()
    assert svc.cross_file_enabled is False


def test_files_in_flight_keep_their_own_results_when_they_share_the_lanes():
    svc = _service(cross_file=True)
    svc.begin_cross_file_stage(3)
    outputs = {}

    def run(length):
        outputs[length] = svc.process(_segments(2, length), _audio())
        svc.settle_file()

    threads = [threading.Thread(target=run, args=(length,)) for length in (2.0, 3.0, 4.0)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
    finally:
        svc.end_cross_file_stage()

    assert sorted(outputs) == [2.0, 3.0, 4.0]
    for length, transcripts in outputs.items():
        tag = int(length * SR)
        assert len(transcripts) == 2
        assert all(t.text_whisper.startswith(f"w{tag} ") for t in transcripts)
        assert all(t.text_phowhisper.startswith(f"p{tag} ") for t in transcripts)
        assert all(t.text_qwen3.startswith(f"q{tag} ") for t in transcripts)


def test_models_are_released_once_every_file_has_submitted_or_been_skipped():
    loader, qwen_worker = FakeLoader(), FakeWorker()
    svc = _service(cross_file=True, loader=loader, workers={"qwen3": qwen_worker})
    svc.begin_cross_file_stage(2)
    try:
        svc.process(_segments(), _audio(0.2))
        svc.settle_file()
        time.sleep(0.2)
        assert loader.unloaded == []            # a second file may still come
        svc.settle_file()                       # ...it was skipped (checkpointed)
        deadline = time.monotonic() + 3
        while len(loader.unloaded) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        svc.end_cross_file_stage()
    assert sorted(loader.unloaded) == ["phowhisper", "qwen3", "whisper"]
    assert qwen_worker.stopped is True


def test_the_users_keep_models_choice_prevents_any_release():
    loader = FakeLoader()
    svc = _service(cross_file=True, loader=loader, keep_models=True)
    svc.begin_cross_file_stage(1)
    try:
        svc.process(_segments(), _audio(0.2))
        svc.settle_file()
        time.sleep(0.2)
    finally:
        svc.end_cross_file_stage()
    assert loader.unloaded == []


def test_ending_the_stage_stops_every_scheduler_thread():
    svc = _service(cross_file=True)
    svc.begin_cross_file_stage(1)
    svc.process(_segments(), _audio(0.2))
    svc.settle_file()
    svc.end_cross_file_stage()
    assert not [t for t in threading.enumerate() if t.name.startswith("asr-") and t.is_alive()]


def test_a_new_replica_uses_the_configured_factory_and_the_boost_batch():
    made = []

    def factory(gpu, batch):
        made.append((gpu, batch))
        return FakePho(), lambda: None

    loader = FakeLoader()
    svc = ASRService(
        whisper=FakeWhisper(), phowhisper=FakePho(), qwen3=FakeQwenClient(),
        model_loader=loader,
        performance_config={"cross_file": True, "shared_batch_size": 16,
                            "boost_batch_size": 48},
        batch_size=8, edge_pad=0.0,
        asr_placement={"qwen3": 0, "whisper": 1, "phowhisper": 1},
        replica_factories={"phowhisper": factory})
    lane = svc._build_scheduler("tmp").lanes["phowhisper"]
    worker = lane.replica_factory(0, 48)
    assert made == [(0, 48)] and worker.batch_size == 48 and worker.gpu == 0
    # whisper and phowhisper share GPU 1, so both start at the shared batch size
    scheduler = svc._build_scheduler("tmp")
    assert scheduler.lanes["whisper"].primary.batch_size == 16
    assert scheduler.lanes["phowhisper"].primary.batch_size == 16
    assert scheduler.lanes["qwen3"].primary.batch_size == 8      # alone on GPU 0


# --- PipelineService hooks --------------------------------------------------

def test_the_asr_view_is_per_file_only_in_cross_file_mode():
    from services.pipeline_service import PipelineService

    pipeline = PipelineService(*(None,) * 8)
    asr = _service(cross_file=True)
    pipeline.asr_svc = asr
    pipeline.noise_track = "the noise of file A"

    assert pipeline.parallel_stage_view("asr") is pipeline      # no stage open yet

    asr.begin_cross_file_stage(2)
    try:
        view = pipeline.parallel_stage_view("asr")
        assert view is not pipeline
        assert view.asr_svc is asr, "every file must feed the one shared scheduler"
        assert view.noise_track is None
    finally:
        asr.end_cross_file_stage()


def test_closing_the_asr_stage_scope_ends_the_cross_file_stage_and_restores_keep_models():
    from services.pipeline_service import PipelineService

    pipeline = PipelineService.__new__(PipelineService)
    pipeline.logger = None
    pipeline.model_loader = None
    pipeline.worker_services = {}
    pipeline.performance_monitor = None
    asr = _service(cross_file=True)
    pipeline.asr_svc = asr

    pipeline.begin_stage_scope()
    pipeline.prepare_stage_scope("asr")
    assert asr.keep_models is True
    asr.begin_cross_file_stage(1)
    assert asr.cross_file_enabled is True

    pipeline.end_stage_scope()

    assert asr.cross_file_enabled is False
    assert asr.keep_models is False


def test_a_released_lane_drops_its_model_and_refuses_further_work():
    loader = FakeLoader()
    svc = _service(cross_file=True, loader=loader)
    lane = svc._build_scheduler("tmp").lanes["phowhisper"]

    lane.primary.release()

    assert loader.unloaded == ["phowhisper"]
    with pytest.raises(RuntimeError, match="released"):
        lane.primary.run_batch([np.zeros(10, np.float32)], 1)


def test_a_replica_release_drops_the_replica_model_too():
    released = []

    def factory(gpu, batch):
        return FakePho(), lambda: released.append(gpu)

    svc = ASRService(
        whisper=FakeWhisper(), phowhisper=FakePho(), qwen3=FakeQwenClient(),
        model_loader=FakeLoader(), edge_pad=0.0,
        performance_config={"cross_file": True},
        asr_placement={"qwen3": 0, "whisper": 1, "phowhisper": 1},
        replica_factories={"phowhisper": factory})
    worker = svc._build_scheduler("tmp").lanes["phowhisper"].replica_factory(0, 48)
    assert worker.run_batch([np.zeros(10, np.float32)], 48) == ["p10 xin chào bạn"]

    worker.release()

    assert released == [0]
    with pytest.raises(RuntimeError, match="released"):
        worker.run_batch([np.zeros(10, np.float32)], 48)


class FakePool:
    """A worker pool of two processes, as main.py builds for PhoWhisper."""

    def __init__(self, workers=2, ready_error=None):
        self.services = [object() for _ in range(workers)]
        self.process = object()
        self.ready_calls = 0
        self.ready_error = ready_error
        self.stopped = 0

    def wait_ready(self):
        self.ready_calls += 1
        if self.ready_error:
            raise RuntimeError(self.ready_error)

    def stop(self):
        self.stopped += 1


def _lanes(svc):
    import tempfile
    scheduler = svc._build_scheduler(tempfile.mkdtemp())
    return scheduler, scheduler.lanes


def test_a_pooled_model_gets_one_lane_worker_per_process_and_waits_for_its_own_readiness():
    pool = FakePool(workers=2)
    svc = _service(cross_file=True, workers={"phowhisper": pool})
    scheduler, lanes = _lanes(svc)
    try:
        assert len(lanes["phowhisper"].workers) == 2
        assert len(lanes["whisper"].workers) == 1 and lanes["whisper"].primary.ready is None
        assert all(w.ready is not None for w in lanes["phowhisper"].workers)
        lanes["phowhisper"].workers[0].ready()
        assert pool.ready_calls == 1
    finally:
        scheduler.shutdown()


def test_releasing_a_pooled_lane_stops_the_pool_and_is_safe_to_repeat():
    pool = FakePool(workers=2)
    svc = _service(cross_file=True, workers={"phowhisper": pool})
    scheduler, lanes = _lanes(svc)
    try:
        for worker in lanes["phowhisper"].workers:
            worker.release()
    finally:
        scheduler.shutdown()
    assert pool.stopped >= 1


def test_a_pool_that_never_comes_up_fails_the_file_instead_of_voting_on_nothing():
    pool = FakePool(workers=1, ready_error="no GPU")
    svc = _service(cross_file=True, workers={"phowhisper": pool})
    svc.begin_cross_file_stage(1)
    try:
        with pytest.raises(RuntimeError, match="phowhisper.*no GPU"):
            svc.process(_segments(3), _audio())
    finally:
        svc.end_cross_file_stage()
