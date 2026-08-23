"""Worker lifecycle across a multi-file run.

The three worker subprocesses are spawned once in main(), before the file loop.
A stage that stops one without putting it back leaves every later file in the
run with nothing to talk to -- which never shows up on a single-file test.

Run:  python -m pytest tests/test_worker_lifecycle.py -q   (from podcast-pipeline/)
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.pipeline_service import PipelineService


class _FakeProcess:
    def __init__(self, generation):
        self.generation = generation


class _FakeWorkerService:
    """Counts spawns so a restart is distinguishable from a survivor."""

    def __init__(self):
        self.process = _FakeProcess(1)
        self.spawns = 1
        self.stops = 0

    def spawn(self):
        self.spawns += 1
        self.process = _FakeProcess(self.spawns)

    def wait_ready(self):
        pass

    def stop(self):
        self.stops += 1
        self.process = None


class _Client:
    def __init__(self, process):
        self.process = process


def _pipeline(worker):
    p = PipelineService.__new__(PipelineService)
    p.logger = None
    p.model_loader = object()
    p.worker_services = {"diarizen": worker}
    # No stage scope open: releases fire immediately, as they do in file-major
    # runs. The deferred path has its own tests below.
    p.defer_free = None
    p.defer_workers = None
    return p


def _args(keep_models):
    return types.SimpleNamespace(keep_models=keep_models)


def test_keep_models_leaves_the_worker_running():
    worker = _FakeWorkerService()
    _pipeline(worker)._release_worker(_args(True), "diarizen")

    assert worker.stops == 0
    assert worker.process is not None


def test_without_keep_models_the_worker_is_released():
    worker = _FakeWorkerService()
    _pipeline(worker)._release_worker(_args(False), "diarizen")

    assert worker.stops == 1
    assert worker.process is None


def test_a_released_worker_is_restarted_for_the_next_file():
    """The regression: nothing respawned, so file two had no worker."""
    worker = _FakeWorkerService()
    pipe = _pipeline(worker)

    pipe._release_worker(_args(False), "diarizen")     # end of file 1
    process = pipe._ensure_worker(_args(False), "diarizen")   # start of file 2

    assert process is not None
    assert worker.spawns == 2


def test_a_live_worker_is_not_restarted():
    worker = _FakeWorkerService()
    process = _pipeline(worker)._ensure_worker(_args(False), "diarizen")

    assert process is worker.process
    assert worker.spawns == 1


def test_the_client_is_repointed_at_the_new_process():
    """A restart makes a new Popen; a client still holding the old one would
    write into a closed pipe."""
    worker = _FakeWorkerService()
    pipe = _pipeline(worker)
    service = types.SimpleNamespace(diarizer=_Client(worker.process))
    stale = worker.process

    pipe._release_worker(_args(False), "diarizen")
    pipe._rebind_worker(_args(False), "diarizen", service, "diarizer")

    assert service.diarizer.process is not stale
    assert service.diarizer.process is worker.process


def test_rebinding_is_a_no_op_while_the_worker_lives():
    worker = _FakeWorkerService()
    pipe = _pipeline(worker)
    service = types.SimpleNamespace(diarizer=_Client(worker.process))

    pipe._rebind_worker(_args(True), "diarizen", service, "diarizer")

    assert worker.spawns == 1
    assert service.diarizer.process is worker.process


def test_an_absent_worker_is_tolerated():
    """--dia3 replaces DiariZen with pyannote, so the service is not there."""
    pipe = _pipeline(_FakeWorkerService())
    pipe.worker_services = {}

    assert pipe._ensure_worker(_args(False), "diarizen") is None
    pipe._rebind_worker(_args(False), "diarizen", None, "diarizer")


# --- stage-major deferral ---------------------------------------------------

class _Loader:
    def __init__(self):
        self.unloaded = []

    def unload(self, name):
        self.unloaded.append(name)


def test_a_stage_scope_holds_releases_until_the_stage_ends():
    """Stage-major runs one stage across every file. Freeing at the end of each
    file would drop the model the next file is about to use."""
    worker = _FakeWorkerService()
    pipe = _pipeline(worker)
    pipe.model_loader = _Loader()
    args = _args(False)

    pipe.begin_stage_scope()
    for _ in range(3):                      # three files through one stage
        pipe._free(args, "diarizer", "vad")
        pipe._release_worker(args, "diarizen")

    assert pipe.model_loader.unloaded == [], "nothing may be freed mid-stage"
    assert worker.stops == 0

    pipe.end_stage_scope()

    assert sorted(pipe.model_loader.unloaded) == ["diarizer", "vad"]
    assert worker.stops == 1, "released once, not once per file"


def test_keep_models_still_frees_nothing_inside_a_scope():
    worker = _FakeWorkerService()
    pipe = _pipeline(worker)
    pipe.model_loader = _Loader()

    pipe.begin_stage_scope()
    pipe._free(_args(True), "diarizer")
    pipe._release_worker(_args(True), "diarizen")
    pipe.end_stage_scope()

    assert pipe.model_loader.unloaded == []
    assert worker.stops == 0


def test_closing_a_scope_that_was_never_opened_is_safe():
    pipe = _pipeline(_FakeWorkerService())
    pipe.model_loader = _Loader()
    pipe.end_stage_scope()          # must not raise
