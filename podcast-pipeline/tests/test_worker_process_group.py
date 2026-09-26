"""Stopping a worker must stop everything it started. vLLM launches its engine as a
child process that holds the GPU memory and does not exit when its parent is killed
(no parent-death monitor), so a terminated worker left the memory taken -- including a
worker that was stopped while its model was still loading. Real processes, no GPU."""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.base_worker_service import WorkerProcessService

# A worker that starts a child (the "engine"), records both pids, optionally takes a
# while to say it is ready (loading), then serves until stdin closes.
WORKER = '''
import json, os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
with open(sys.argv[1], "w") as handle:
    handle.write("%d %d" % (os.getpid(), child.pid))
time.sleep(float(sys.argv[2]))
print(json.dumps({"status": "ready"}), flush=True)
for line in sys.stdin:
    pass
time.sleep(600)   # ignores a closed stdin, like a busy engine
'''


def _service(tmp_path, loading_seconds):
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    pids = tmp_path / "pids.txt"
    service = WorkerProcessService("engine", sys.executable, str(script),
                                   extra_args=[str(pids), str(loading_seconds)])
    return service, pids


def _pids(path, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if path.exists() and path.read_text().strip().count(" ") == 1:
            leader, child = (int(x) for x in path.read_text().split())
            return leader, child
        time.sleep(0.05)
    raise AssertionError("worker never wrote its pids")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie that nobody reaped still answers signal 0
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().split()[2] != "Z"
    except OSError:
        return True


def _gone(pid, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def cleanup():
    started = []
    yield started
    for pid in started:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def test_stopping_a_loaded_worker_also_stops_the_process_it_started(tmp_path, cleanup):
    service, pids = _service(tmp_path, loading_seconds=0)
    service.spawn()
    service.wait_ready()
    leader, child = _pids(pids)
    cleanup.extend([leader, child])

    service.stop()

    assert _gone(leader) and _gone(child), "the engine process outlived its worker"


def test_stopping_a_worker_that_is_still_loading_releases_it_and_the_waiter(tmp_path, cleanup):
    service, pids = _service(tmp_path, loading_seconds=60)   # never says ready in time
    service.spawn()
    leader, child = _pids(pids)
    cleanup.extend([leader, child])
    outcome = []

    def wait():
        try:
            service.wait_ready()
            outcome.append("ready")
        except Exception as exc:
            outcome.append(type(exc).__name__)

    waiter = threading.Thread(target=wait)
    waiter.start()
    time.sleep(0.5)

    service.stop()
    waiter.join(20)

    assert not waiter.is_alive(), "wait_ready hung on a worker that was stopped"
    assert outcome and outcome[0] != "ready"
    assert _gone(leader) and _gone(child), "a worker stopped while loading kept its engine"


def test_a_worker_no_longer_shares_the_parents_process_group(tmp_path, cleanup):
    service, pids = _service(tmp_path, loading_seconds=0)
    service.spawn()
    service.wait_ready()
    leader, child = _pids(pids)
    cleanup.extend([leader, child])
    try:
        assert os.getpgid(leader) == leader != os.getpgid(0)      # its own group
        assert os.getpgid(child) == leader                        # the engine joins it
    finally:
        service.stop()


def test_workers_still_alive_at_exit_are_stopped_by_the_exit_hook(tmp_path, cleanup):
    import subprocess
    script = tmp_path / "parent.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})\n"
        "from services.base_worker_service import WorkerProcessService\n"
        f"s = WorkerProcessService('engine', sys.executable, {str(tmp_path / 'worker.py')!r},"
        f" extra_args=[{str(tmp_path / 'pids.txt')!r}, '0'])\n"
        "s.spawn(); s.wait_ready()\n"          # then exits without calling stop()
    )
    (tmp_path / "worker.py").write_text(WORKER)
    subprocess.run([sys.executable, str(script)], check=True, timeout=60)

    leader, child = _pids(tmp_path / "pids.txt")
    cleanup.extend([leader, child])
    assert _gone(leader) and _gone(child)
