# BS-RoFormer Process Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run each GPU's BS-RoFormer separator in its own subprocess instead of its own thread, so PyTorch's process-global SDPA backend flags can no longer be corrupted by two concurrent `sdpa_kernel()` contexts.

**Architecture:** A new worker script (`bs_roformer_worker.py`) hosts one `BSRoformerRemover` and speaks the same line-JSON-over-stdio protocol as the Sidon worker, moving audio as `.npy` files. `BSRoformerProcessRemover` subclasses `BSRoformerRemover`, overrides only the GPU half (`separate_raw`) to call the worker, and inherits every CPU-side method unchanged, so `music_service.py` and its `gpu_executor` threads are untouched (those threads now only wait on a pipe). A profile flag `bs_roformer.isolate_process` selects it; default stays in-process.

**Tech Stack:** Python 3.12, `subprocess` via the existing `WorkerProcessService`, NumPy `.npy` files for IPC, pytest.

**Spec:** Conversation of 2026-09-24/25 (root cause: `audio_separator/.../roformer/attend.py` wraps attention in `sdpa_kernel(...)`, which mutates process-global flags; two BS-RoFormer threads interleave enter/exit and leave `math`/`mem_efficient` disabled, so the later fp32 Wav2Vec2 word-alignment fails with "No available kernel"). `utils/sdpa_guard.py` already restores the flags at stage boundaries as a stopgap; this plan removes the cause. Expected speed change: neutral to slightly slower (IPC); the goal is reliability.

## Global Constraints

- Worker uses the same interpreter as the main process (`sys.executable`); no new virtualenv.
- Default behaviour unchanged: without `isolate_process: true` the in-process `BSRoformerRemover` is used exactly as today.
- `music_service.py` is not modified.
- Failure contract preserved: `separate_raw` returns `None` on any failure (span stays mixture, never silence).
- A worker that died is restarted transparently on the next call.
- Tests follow the repo style: `pytest`, small fakes instead of GPU/model-heavy runs. Run from `podcast-pipeline/`: `python -m pytest tests/<file> -q`.
- Commits end with the line `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `podcast-pipeline/bs_roformer_worker.py` (create) | Worker process: load one `BSRoformerRemover`, answer `separate_raw` requests over stdio |
| `podcast-pipeline/services/bs_roformer_worker_service.py` (create) | Client-side lifecycle for that worker (`WorkerProcessService` subclass) |
| `podcast-pipeline/models/bs_roformer_process.py` (create) | `BSRoformerProcessRemover` proxy, `physical_gpu_index`, `build_bs_roformers` factory |
| `podcast-pipeline/services/model_loader.py` (modify `load_music_models`) | Use the factory |
| `podcast-pipeline/config.json` (modify `a100` profile) | `"isolate_process": true` |
| `podcast-pipeline/tests/fake_bs_roformer_worker.py` (create) | Model-free stand-in worker for tests |
| `podcast-pipeline/tests/test_bs_roformer_worker.py` (create) | Worker request handler |
| `podcast-pipeline/tests/test_bs_roformer_process.py` (create) | Proxy + service + factory |

---

### Task 1: Worker script and its request handler

**Files:**
- Create: `podcast-pipeline/bs_roformer_worker.py`
- Test: `podcast-pipeline/tests/test_bs_roformer_worker.py`

**Interfaces:**
- Produces: `handle_request(remover, req: dict) -> dict` returning one of
  `{"id", "out_path", "out_sr": int, "stereo_in": bool}` | `{"id", "result": None}` | `{"id", "error": str}`;
  `serve()`; CLI `--device <str, "" = none>` `--kwargs <json>`. Startup prints `{"status": "ready"}` on the real stdout, or `{"status": "error", "message": ...}` then exits 1.
- Consumes: `models.bs_roformer.BSRoformerRemover(device, logger, **kwargs)`, `._get_model()`, `.separate_raw(audio, sample_rate) -> (out, out_sr, stereo_in) | None`.

- [ ] **Step 1: Write the failing test**

Create `podcast-pipeline/tests/test_bs_roformer_worker.py`:

```python
"""The BS-RoFormer worker answers separate_raw requests over .npy files."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bs_roformer_worker import handle_request


class _Remover:
    def __init__(self, raw=None, error=None):
        self.raw, self.error, self.seen = raw, error, None

    def separate_raw(self, audio, sample_rate):
        self.seen = (audio.shape, sample_rate)
        if self.error:
            raise self.error
        return self.raw


def _request(tmp_path, audio, **extra):
    path = tmp_path / "in.npy"
    np.save(path, audio)
    return {"id": "r1", "audio_path": str(path), "sample_rate": 44100, **extra}


def test_the_stem_is_written_next_to_the_input_and_its_rate_reported(tmp_path):
    remover = _Remover(raw=(np.full((800, 2), 0.25, np.float32), 44100, True))
    response = handle_request(remover, _request(tmp_path, np.ones((800, 2), np.float32)))
    assert response["id"] == "r1"
    assert response["out_sr"] == 44100 and response["stereo_in"] is True
    assert np.allclose(np.load(response["out_path"]), 0.25)
    assert remover.seen == ((800, 2), 44100)


def test_a_separator_that_returns_nothing_reports_a_null_result(tmp_path):
    response = handle_request(_Remover(raw=None), _request(tmp_path, np.ones(800, np.float32)))
    assert response == {"id": "r1", "result": None}


def test_a_separator_exception_is_reported_not_raised(tmp_path):
    response = handle_request(
        _Remover(error=ValueError("bad shape")), _request(tmp_path, np.ones(800, np.float32)))
    assert response["id"] == "r1"
    assert "ValueError" in response["error"] and "bad shape" in response["error"]


def test_a_request_without_audio_is_an_error():
    assert handle_request(_Remover(), {"id": "r2"}) == {"id": "r2", "error": "Missing audio_path"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd podcast-pipeline && python -m pytest tests/test_bs_roformer_worker.py -q`
Expected: FAIL / collection error `ModuleNotFoundError: No module named 'bs_roformer_worker'`

- [ ] **Step 3: Write minimal implementation**

Create `podcast-pipeline/bs_roformer_worker.py`:

```python
"""BS-RoFormer worker: one separator instance in its own process.

Speaks line-JSON on stdio (see services/base_worker_service.py) and moves audio
as .npy files, like the Sidon worker. Each GPU's separator gets its own
process, so its copy of PyTorch's process-global SDPA backend flags cannot be
corrupted by another separator: audio-separator wraps attention in
sdpa_kernel(), and two threads of one process can leave it stuck (see
utils/sdpa_guard.py).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class _StderrLogger:
    """Diagnostics go to stderr, which the parent drains; stdout is protocol only."""

    def _emit(self, level, message):
        print(f"[BSRoformerWorker:{level}] {message}", file=sys.stderr, flush=True)

    def debug(self, message, **_):
        pass

    def info(self, message, **_):
        self._emit("info", message)

    def warning(self, message, **_):
        self._emit("warning", message)

    def error(self, message, **_):
        self._emit("error", message)


def handle_request(remover, req: dict) -> dict:
    req_id = req.get("id", "unknown")
    audio_path = req.get("audio_path")
    if not audio_path:
        return {"id": req_id, "error": "Missing audio_path"}
    try:
        audio = np.load(audio_path)
        raw = remover.separate_raw(audio, int(req.get("sample_rate", 44100)))
        if raw is None:
            return {"id": req_id, "result": None}
        out, out_sr, stereo_in = raw
        out_path = os.path.splitext(audio_path)[0] + "_out.npy"
        np.save(out_path, np.asarray(out, dtype=np.float32))
        return {"id": req_id, "out_path": out_path,
                "out_sr": int(out_sr), "stereo_in": bool(stereo_in)}
    except Exception as exc:
        return {"id": req_id, "error": f"{type(exc).__name__}: {exc}"}


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="")
    parser.add_argument("--kwargs", default="{}")
    args = parser.parse_args()

    # Library code (audio-separator prints "A100 GPU detected...") writes to
    # stdout; keep the real stdout for protocol lines only.
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def emit(message):
        print(json.dumps(message), file=protocol, flush=True)

    try:
        from models.bs_roformer import BSRoformerRemover
        remover = BSRoformerRemover(
            device=args.device or None, logger=_StderrLogger(), **json.loads(args.kwargs))
        remover._get_model()
    except Exception as exc:
        emit({"status": "error", "message": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)
    emit({"status": "ready", "device": args.device})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        emit(handle_request(remover, req))


if __name__ == "__main__":
    serve()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd podcast-pipeline && python -m pytest tests/test_bs_roformer_worker.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add podcast-pipeline/bs_roformer_worker.py podcast-pipeline/tests/test_bs_roformer_worker.py
git commit -m "feat(music): add a BS-RoFormer worker process speaking the line-JSON protocol

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Worker service client and `BSRoformerProcessRemover`

**Files:**
- Create: `podcast-pipeline/services/bs_roformer_worker_service.py`
- Create: `podcast-pipeline/models/bs_roformer_process.py`
- Create: `podcast-pipeline/tests/fake_bs_roformer_worker.py`
- Test: `podcast-pipeline/tests/test_bs_roformer_process.py`

**Interfaces:**
- Consumes (Task 1): CLI `--device`, `--kwargs`; response shapes of `handle_request`.
- Consumes (existing): `WorkerProcessService(name, python_bin, worker_script, extra_args, device_id, ready_timeout, logger)` with `.process`, `.start()`, `.stop()`, `.request(payload, response_id=...)`; `BSRoformerRemover` methods `separate_raw`, `postprocess_separated`, `separate_span_raw`, `separate_span_postprocess`, `separate_segment`, `unload`.
- Produces: `BSRoformerWorkerService(python_env_path, worker_script_path, device_id, remover_kwargs, logger=None)`; `BSRoformerProcessRemover(device=None, logger=None, python_bin=None, worker_script=None, **kwargs)` with attributes `_service` (the worker service or `None`) and `_io_dir`; `physical_gpu_index(device) -> Optional[int]`.

- [ ] **Step 1: Write the fake worker and the failing tests**

Create `podcast-pipeline/tests/fake_bs_roformer_worker.py`:

```python
"""Stand-in for bs_roformer_worker.py: same protocol, no models.

Behaviour comes from the --kwargs JSON: fake_mode is echo (default; halves the
audio), none, error, or die_once (exit on the first request, once, tracked by
the die_marker file, so a restarted worker then behaves like echo).
"""
import argparse
import json
import os
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="")
parser.add_argument("--kwargs", default="{}")
args = parser.parse_args()
cfg = json.loads(args.kwargs)
mode = cfg.get("fake_mode", "echo")
marker = cfg.get("die_marker")

print(json.dumps({"status": "ready"}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if mode == "die_once" and marker and not os.path.exists(marker):
        open(marker, "w").close()
        os._exit(1)
    if mode == "error":
        print(json.dumps({"id": req["id"], "error": "boom"}), flush=True)
        continue
    if mode == "none":
        print(json.dumps({"id": req["id"], "result": None}), flush=True)
        continue
    audio = np.load(req["audio_path"])
    out_path = os.path.splitext(req["audio_path"])[0] + "_out.npy"
    np.save(out_path, (audio * 0.5).astype(np.float32))
    print(json.dumps({
        "id": req["id"], "out_path": out_path, "out_sr": req["sample_rate"],
        "stereo_in": bool(audio.ndim == 2 and audio.shape[1] > 1),
    }), flush=True)
```

Create `podcast-pipeline/tests/test_bs_roformer_process.py`:

```python
"""Each GPU's BS-RoFormer separator runs in its own process."""

import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bs_roformer_process import BSRoformerProcessRemover, physical_gpu_index

FAKE_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fake_bs_roformer_worker.py")


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, message, **_):
        self.errors.append(message)

    def info(self, *_, **__):
        pass

    warning = debug = info


def _remover(logger=None, **kwargs):
    return BSRoformerProcessRemover(
        device=None, logger=logger, python_bin=sys.executable,
        worker_script=FAKE_WORKER, **kwargs)


def test_a_span_round_trips_through_the_worker_process():
    remover = _remover(fake_mode="echo")
    try:
        out, out_sr, stereo_in = remover.separate_raw(np.ones(1600, np.float32), 16000)
        assert out_sr == 16000 and stereo_in is False
        assert np.allclose(out, 0.5)
        # the inherited CPU half works on the worker's raw result
        vocals = remover.separate_segment(np.ones(1600, np.float32), 16000)
        assert len(vocals) == 1600 and np.allclose(vocals, 0.5)
    finally:
        remover.unload()


def test_stereo_input_keeps_its_shape():
    remover = _remover(fake_mode="echo")
    try:
        out, _, stereo_in = remover.separate_raw(np.ones((1600, 2), np.float32), 44100)
        assert stereo_in is True and out.shape == (1600, 2)
    finally:
        remover.unload()


def test_a_null_result_leaves_the_segment_as_mixture():
    remover = _remover(fake_mode="none")
    try:
        assert remover.separate_raw(np.ones(1600, np.float32), 16000) is None
        segment = np.ones(1600, np.float32)
        assert remover.separate_segment(segment, 16000) is segment
    finally:
        remover.unload()


def test_a_worker_error_is_logged_and_returns_none():
    logger = _Logger()
    remover = _remover(logger=logger, fake_mode="error")
    try:
        assert remover.separate_raw(np.ones(1600, np.float32), 16000) is None
        assert any("boom" in message for message in logger.errors)
    finally:
        remover.unload()


def test_a_worker_that_died_is_restarted_on_the_next_call(tmp_path):
    remover = _remover(fake_mode="die_once", die_marker=str(tmp_path / "died"))
    try:
        assert remover.separate_raw(np.ones(1600, np.float32), 16000) is None
        out, _, _ = remover.separate_raw(np.ones(1600, np.float32), 16000)
        assert np.allclose(out, 0.5)
    finally:
        remover.unload()


def test_unload_stops_the_worker_and_removes_the_scratch_dir():
    remover = _remover(fake_mode="echo")
    remover.separate_raw(np.ones(1600, np.float32), 16000)
    process, io_dir = remover._service.process, remover._io_dir
    assert process.poll() is None and os.path.isdir(io_dir)
    remover.unload()
    assert process.poll() is not None
    assert remover._service is None and not os.path.exists(io_dir)


def test_two_removers_run_in_different_processes_concurrently():
    a, b = _remover(fake_mode="echo"), _remover(fake_mode="echo")
    try:
        results = []
        threads = [
            threading.Thread(
                target=lambda r=r: results.append(r.separate_raw(np.ones(800, np.float32), 16000)))
            for r in (a, b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 2 and all(r is not None for r in results)
        assert a._service.process.pid != b._service.process.pid
        assert os.getpid() not in (a._service.process.pid, b._service.process.pid)
    finally:
        a.unload()
        b.unload()


def test_the_separator_is_never_loaded_in_the_parent_process():
    with pytest.raises(RuntimeError):
        _remover()._get_model()


@pytest.mark.parametrize("device, expected", [
    ("cuda:1", 1), ("cuda:0", 0), ("cuda", 0), ("cpu", None), (None, None), ("", None)])
def test_physical_gpu_index(device, expected):
    assert physical_gpu_index(device) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd podcast-pipeline && python -m pytest tests/test_bs_roformer_process.py -q`
Expected: FAIL / collection error `ModuleNotFoundError: No module named 'models.bs_roformer_process'`

- [ ] **Step 3: Write the worker service**

Create `podcast-pipeline/services/bs_roformer_worker_service.py`:

```python
import json
from typing import Optional

from services.base_worker_service import WorkerProcessService


class BSRoformerWorkerService(WorkerProcessService):
    """Lifecycle of one BS-RoFormer worker process, one per GPU.

    Same interpreter as the main process (audio-separator is installed there),
    so unlike Sidon this needs no separate virtualenv. CUDA_VISIBLE_DEVICES maps
    the physical card to the worker's local cuda:0.
    """

    def __init__(self, python_env_path: str, worker_script_path: str,
                 device_id: Optional[int], remover_kwargs: dict, logger=None):
        super().__init__(
            name="BS-RoFormer",
            python_bin=python_env_path,
            worker_script=worker_script_path,
            extra_args=["--device", "cuda:0" if device_id is not None else "",
                        "--kwargs", json.dumps(remover_kwargs)],
            device_id=device_id,
            logger=logger,
            ready_timeout=900.0,
        )
```

- [ ] **Step 4: Write the proxy**

Create `podcast-pipeline/models/bs_roformer_process.py`:

```python
"""BS-RoFormer whose GPU half runs in a worker process.

Subclasses BSRoformerRemover and overrides only separate_raw(); the CPU-side
methods (postprocess_separated, separate_span_raw, ...) are inherited, and
music_service.py sees the same interface. See bs_roformer_worker.py for why the
separator lives in its own process.
"""

import os
import shutil
import sys
import tempfile
import threading
import uuid
from typing import Optional

import numpy as np

from models.bs_roformer import BSRoformerRemover
from services.bs_roformer_worker_service import BSRoformerWorkerService

_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bs_roformer_worker.py")


def physical_gpu_index(device) -> Optional[int]:
    """Host GPU index behind a torch device string, or None for CPU."""
    text = str(device or "")
    if not text.startswith("cuda"):
        return None
    _, _, index = text.partition(":")
    return int(index) if index.isdigit() else 0


class BSRoformerProcessRemover(BSRoformerRemover):
    def __init__(self, device=None, logger=None, python_bin=None,
                 worker_script=None, **kwargs):
        super().__init__(device=device, logger=logger, **kwargs)
        self._worker_kwargs = dict(kwargs)
        self._python_bin = python_bin or sys.executable
        self._worker_script = worker_script or _WORKER_SCRIPT
        self._service = None
        self._service_lock = threading.Lock()
        self._io_dir = None

    def _get_model(self):
        raise RuntimeError(
            "BSRoformerProcessRemover keeps the separator in a worker process")

    def _ensure_worker(self):
        with self._service_lock:
            if self._service is None:
                self._service = BSRoformerWorkerService(
                    self._python_bin, self._worker_script,
                    physical_gpu_index(self.device), self._worker_kwargs,
                    logger=self.logger)
            process = self._service.process
            if process is not None and process.poll() is not None:
                self._service.stop()          # reap a worker that died
            if self._service.process is None:
                self._service.start()
            if self._io_dir is None:
                self._io_dir = tempfile.mkdtemp(prefix="bsroformer_io_")
            return self._service

    def separate_raw(self, audio_array: np.ndarray, sample_rate: int):
        """GPU half of _run(), executed by the worker. None on any failure."""
        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.ndim > 2:
            audio = audio.reshape(len(audio), -1)
        if audio.size == 0:
            return None
        request_id = uuid.uuid4().hex
        in_path = out_path = None
        try:
            service = self._ensure_worker()
            in_path = os.path.join(self._io_dir, f"{request_id}.npy")
            np.save(in_path, audio)
            response = service.request(
                {"id": request_id, "audio_path": in_path,
                 "sample_rate": int(sample_rate)},
                response_id=request_id)
            if response.get("error"):
                raise RuntimeError(response["error"])
            out_path = response.get("out_path")
            if not out_path:
                return None
            out = np.load(out_path)
            return out, int(response["out_sr"]), bool(response["stereo_in"])
        except Exception as exc:
            if self.logger:
                self.logger.error(
                    f"BS-RoFormer worker failed ({type(exc).__name__}: {exc}); "
                    "keeping the mixture for this span")
            return None
        finally:
            for path in (in_path, out_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def unload(self):
        with self._service_lock:
            service, self._service = self._service, None
            io_dir, self._io_dir = self._io_dir, None
        if service is not None:
            service.stop()
        if io_dir:
            shutil.rmtree(io_dir, ignore_errors=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd podcast-pipeline && python -m pytest tests/test_bs_roformer_process.py tests/test_bs_roformer_worker.py -q`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add podcast-pipeline/services/bs_roformer_worker_service.py podcast-pipeline/models/bs_roformer_process.py podcast-pipeline/tests/fake_bs_roformer_worker.py podcast-pipeline/tests/test_bs_roformer_process.py
git commit -m "feat(music): BSRoformerProcessRemover runs the separator in a worker process

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Select it from the profile and enable it for A100

**Files:**
- Modify: `podcast-pipeline/models/bs_roformer_process.py` (add `build_bs_roformers`)
- Modify: `podcast-pipeline/services/model_loader.py` (`load_music_models`, the `BSRoformerRemover(...)` list comprehension near line 200)
- Modify: `podcast-pipeline/config.json` (`a100` profile, `bs_roformer` block near line 309)
- Test: `podcast-pipeline/tests/test_bs_roformer_process.py`

**Interfaces:**
- Consumes (Task 2): `BSRoformerProcessRemover`.
- Produces: `build_bs_roformers(devices, cfg: dict, logger=None) -> list` returning `BSRoformerProcessRemover` instances when `cfg["isolate_process"]` is truthy, else plain `BSRoformerRemover` instances; `isolate_process` is removed from the kwargs before construction and `cfg` is not mutated.

- [ ] **Step 1: Write the failing test**

Append to `podcast-pipeline/tests/test_bs_roformer_process.py` (add `build_bs_roformers` to the existing import line from `models.bs_roformer_process`):

```python
def test_isolate_process_selects_the_process_remover():
    cfg = {"isolate_process": True, "chunk_duration": 600}
    isolated = build_bs_roformers(["cuda:0", "cuda:1"], cfg)
    assert all(isinstance(r, BSRoformerProcessRemover) for r in isolated)
    assert [r.device for r in isolated] == ["cuda:0", "cuda:1"]
    assert isolated[0].chunk_duration == 600
    assert cfg == {"isolate_process": True, "chunk_duration": 600}   # not mutated


def test_without_the_flag_the_separator_stays_in_process():
    plain = build_bs_roformers(["cuda:0"], {"chunk_duration": 600})
    assert type(plain[0]).__name__ == "BSRoformerRemover"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd podcast-pipeline && python -m pytest tests/test_bs_roformer_process.py -q`
Expected: FAIL with `ImportError: cannot import name 'build_bs_roformers'`

- [ ] **Step 3: Implement the factory**

Append to `podcast-pipeline/models/bs_roformer_process.py`:

```python
def build_bs_roformers(devices, cfg: dict, logger=None) -> list:
    """One separator per device; each in its own process when isolate_process is set."""
    cfg = dict(cfg)
    if cfg.pop("isolate_process", False):
        cls = BSRoformerProcessRemover
    else:
        from models.bs_roformer import BSRoformerRemover as cls
    return [cls(device=str(device), logger=logger, **cfg) for device in devices]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd podcast-pipeline && python -m pytest tests/test_bs_roformer_process.py -q`
Expected: PASS

- [ ] **Step 5: Use the factory in the loader**

In `podcast-pipeline/services/model_loader.py`, `load_music_models`, replace

```python
            from models.bs_roformer import BSRoformerPool, BSRoformerRemover
```
with
```python
            from models.bs_roformer import BSRoformerPool
            from models.bs_roformer_process import build_bs_roformers
```
and replace
```python
            models = [BSRoformerRemover(
                device=str(device), logger=self.logger, **bs_roformer_cfg)
                for device in devices]
```
with
```python
            models = build_bs_roformers(devices, bs_roformer_cfg, self.logger)
```

- [ ] **Step 6: Enable it for the A100 profile**

In `podcast-pipeline/config.json`, in the **`a100`** profile's `bs_roformer` block (the one with `"chunk_duration": 600`, near line 309), add the flag after `"hi_res": true`:

```json
          "hi_res": true,
          "isolate_process": true
```

(The `kaggle` profile is left unchanged.)

- [ ] **Step 7: Run the related tests**

Run: `cd podcast-pipeline && python -m pytest tests/test_bs_roformer_process.py tests/test_bs_roformer_worker.py tests/test_music_separator.py tests/test_lazy_model_loading.py tests/test_performance_gating.py -q`
Expected: no new failures compared with `git stash` baseline (many unrelated tests in the repo already fail on the baseline; compare counts before and after).

- [ ] **Step 8: Commit**

```bash
git add podcast-pipeline/models/bs_roformer_process.py podcast-pipeline/services/model_loader.py podcast-pipeline/config.json podcast-pipeline/tests/test_bs_roformer_process.py
git commit -m "feat(music): isolate BS-RoFormer per GPU in worker processes on the a100 profile

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Verify on the real machine (manual, no code)

Cannot be run here (needs the A100 host and audio-separator).

- [ ] **Step 1:** `git pull` on `lamkd2`, delete the `music_map`/`timeline`/`music_patches` checkpoints of at least two files with many music spans (e.g. `lm8-vongtaynang-reaction-131131` and `thu_that_thach_10m`), and rerun to `word_alignment`.
- [ ] **Step 2:** Expected in the log: two `Starting BS-RoFormer worker subprocess on CUDA_VISIBLE_DEVICES=0` / `=1` lines, two `BS-RoFormer worker is ready.`, `BS-RoFormer worker terminated.` at the end of the music stage, **no** `[sdpa] backend flags drifted` warning, and `word_alignment` succeeding.
- [ ] **Step 3:** Compare against the previous run: the music-stage wall time (previous: 45.3 min for 46 files) and the `Stripped music from ...s` lines should be similar. A markedly slower music stage means IPC or worker start-up dominates and the flag should be turned off (`isolate_process: false`) while it is investigated.

---

## Self-Review

- **Spec coverage:** worker process per GPU (Task 1-2), same interface so `music_service` untouched (Task 2), profile switch with default unchanged (Task 3), restart on death and `None`-on-failure contract (Task 2 tests), same interpreter (Task 2 `sys.executable`), real-machine check (Task 4). No gaps found.
- **Placeholders:** none; every code step has full code.
- **Type consistency:** `handle_request` response keys (`out_path`, `out_sr`, `stereo_in`, `result`, `error`) match `separate_raw`'s reads; `_service`/`_io_dir` attribute names used by tests match the implementation; `physical_gpu_index` and `build_bs_roformers` names consistent across tasks.
- **Known risk:** first request on each worker pays model load at spawn (≈ seconds, from `wait_ready`); `unload` at stage end stops both workers, so a batch pays this once per music stage, as today's `Loading BS-RoFormer` does.
