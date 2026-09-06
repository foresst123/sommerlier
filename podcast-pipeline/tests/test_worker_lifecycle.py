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
    process = pipe._ensure_worker("diarizen")   # start of file 2

    assert process is not None
    assert worker.spawns == 2


def test_a_live_worker_is_not_restarted():
    worker = _FakeWorkerService()
    process = _pipeline(worker)._ensure_worker("diarizen")

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

    assert pipe._ensure_worker("diarizen") is None
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


# --- workers are not revived for stages that will not use them --------------

def test_a_checkpointed_stage_does_not_revive_its_worker():
    """The OOM: reviving all three workers at the top of run() meant that by
    the refinement stage the diarizer (5.15GB) and Sidon (2.62GB) were resident
    again, leaving 0.03GB on a card that had just been emptied for the LLM."""
    import re

    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "services/pipeline_service.py"), encoding="utf-8").read()

    for worker, stage in (("diarizen", "diarization"),
                          ("qwen3", "asr")):
        call = re.search(rf'.*_rebind_worker\(args, "{worker}".*', src).group(0)
        line_no = src[:src.index(call)].count("\n")
        guard = src.splitlines()[line_no - 1]
        assert f'checkpoint.exists("{stage}")' in guard, (
            f"{worker} is revived unconditionally; it should only start when "
            f"the {stage} stage is actually going to run")


def test_refinement_raises_when_most_batches_fail():
    """Silently returning un-refined text made a run that produced nothing look
    identical to one that worked, and the ledger recorded the file as done."""
    import ast

    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "services/diarization_refinement_service.py"), encoding="utf-8").read()

    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    refine = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == "refine")

    raises = [n for n in ast.walk(refine) if isinstance(n, ast.Raise)]
    assert raises, "refine() must fail loudly when the stage did not work"
    assert "REFINE_FAILURE_LIMIT" in src


def test_a_stage_scope_closes_even_when_the_stage_blows_up():
    """An open scope swallows every later release, so a stage that dies outside
    the per-file try would leave its models resident for the rest of the run."""
    import utils.batch as batch

    class _Exploding:
        def __init__(self):
            self.opened = 0
            self.closed = 0

        def begin_stage_scope(self):
            self.opened += 1

        def end_stage_scope(self):
            self.closed += 1

        def run(self, args, config, path):
            raise KeyboardInterrupt("stage interrupted")

    pipe = _Exploding()
    args = types.SimpleNamespace(stop_after=None, keep_models=False)
    try:
        batch.run_batch_by_stage(pipe, args, {}, ["f1"], stages=("diarization",))
    except KeyboardInterrupt:
        pass

    assert pipe.opened == 1
    assert pipe.closed == 1, "the scope was left open after the stage failed"


# --- the stage is what starts its worker -------------------------------------

class _NeverStarted(_FakeWorkerService):
    """A worker main() never managed to launch: constructed, not running."""

    def __init__(self, fails=False):
        self.process = None
        self.spawns = 0
        self.stops = 0
        self.fails = fails

    def spawn(self):
        if self.fails:
            raise FileNotFoundError("no interpreter for this worker")
        super().spawn()


class _RecordingLoader:
    """Records the order of calls, so 'worker first' is checkable."""

    def __init__(self, worker):
        self.worker = worker
        self.order = []

    def _note(self, what):
        self.order.append((what, self.worker.process is not None))

    def load_diarization_models(self, service=None):
        self._note("diarization")

    def load_base_models(self):
        self._note("base")

    def load_panns(self):
        self._note("panns")


def _staged(worker):
    p = _pipeline(worker)
    p.model_loader = _RecordingLoader(worker)
    return p


def test_a_stage_starts_the_worker_it_needs():
    """The point of the change: main() is a head start, not the authority.

    A run whose worker never launched -- prefetch failed, or the stage was not
    reachable when main() decided -- still has to work when the stage arrives.
    """
    worker = _NeverStarted()
    pipe = _staged(worker)

    pipe._load("diarization")

    assert worker.spawns == 1
    assert worker.process is not None


def test_the_worker_is_up_before_the_models_load():
    """The loader reads service.process; starting after it would pass None."""
    worker = _NeverStarted()
    pipe = _staged(worker)

    pipe._load("diarization")

    assert pipe.model_loader.order == [("diarization", True)]


def test_a_stage_with_no_worker_loads_normally():
    worker = _NeverStarted()
    pipe = _staged(worker)

    pipe._load("panns")

    assert worker.spawns == 0
    assert pipe.model_loader.order == [("panns", False)]


def test_a_live_worker_is_not_started_twice():
    worker = _FakeWorkerService()
    pipe = _staged(worker)

    pipe._load("diarization")

    assert worker.spawns == 1, "already running; nothing to do"


def test_a_worker_that_cannot_start_stops_its_stage():
    """Loudly, and here. Returning None built a client wired to no process,
    and the failure surfaced several steps later as an empty result."""
    worker = _NeverStarted(fails=True)
    pipe = _staged(worker)

    try:
        pipe._load("diarization")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a stage with no worker must not proceed")

    assert pipe.model_loader.order == [], "models must not load without it"


# --- when a worker starts ----------------------------------------------------

def test_workers_wait_for_their_stage_by_default():
    """Starting both up front hides their load time behind the music stage,
    but parks them in VRAM for the whole run: Qwen3-ASR is a 1.7B model that
    sits on its GPU from the first second until the ASR stage. On a two-file
    run that measured a quarter of an hour of memory held and unused."""
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for env, profile in config["environments"].items():
        assert profile["pipeline"]["prefetch_workers"] is False, env


def test_the_lazy_path_is_the_one_that_already_existed():
    """_ensure_worker runs immediately before each stage loads its models, so
    not pre-starting costs a wait rather than a failure. This is what makes the
    default safe."""
    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "services", "pipeline_service.py"),
        encoding="utf-8").read()
    load = source.index("def _load(self, group: str):")
    body = source[load:source.index("def ", load + 10)]
    assert "self._ensure_worker(worker)" in body, (
        "_load must start the stage's worker before loading its models")


def test_prefetch_is_reachable_for_a_box_with_room():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main = open(os.path.join(root, "main.py"), encoding="utf-8").read()
    assert '"--prefetch_workers"' in main
    assert 'getattr(args, "prefetch_workers", False)' in main


def test_the_tagger_is_released_when_the_music_stage_ends():
    """PANNs was only released at the end of step 5, the per-segment music
    fallback. That stage is skipped whenever a music map exists, and under
    stage-major execution the run leaves for diarization long before reaching
    it -- so the tagger held ~600MB of VRAM for the whole run, on the card that
    then has to fit DiariZen, the embedder, TSE and ASR.

    Observed in a real run: "Unloaded bs_roformer from VRAM" appeared and no
    matching line for panns ever did."""
    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "services", "pipeline_service.py"),
        encoding="utf-8").read()

    release = source.index('self._free(args, "panns")')
    diarization = source.index("# 3. Diarization")
    assert release < diarization, (
        "panns must be released inside the music stage, not after it")

    fallback = source.index("# 5. Background Music Removal")
    assert source.index('self._free(args, "panns", "bs_roformer")') > fallback, (
        "the fallback keeps its own release for runs that do reach it")
