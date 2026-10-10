"""A partially-started worker pool must not retain processes or GPU memory."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.refinement_worker_service import RefinementWorkerPoolService
from services.worker_pool_service import WorkerPoolService


class _StartupReplica:
    def __init__(self, name, *, spawn_error=None, ready_error=None, stop_error=None):
        self.name = name
        self.model_name = "fake-model"
        self.spawn_error = spawn_error
        self.ready_error = ready_error
        self.stop_error = stop_error
        self.process = None
        self.stops = 0

    def spawn(self):
        if self.spawn_error:
            raise RuntimeError(self.spawn_error)
        self.process = object()

    def wait_ready(self):
        if self.ready_error:
            raise RuntimeError(self.ready_error)

    def stop(self):
        self.stops += 1
        self.process = None
        if self.stop_error:
            raise RuntimeError(self.stop_error)


def _assert_failed_pool_is_empty(pool, replicas, action):
    try:
        action()
    except RuntimeError as exc:
        assert "out of memory" in str(exc)
    else:
        raise AssertionError("pool startup should have failed")
    assert all(replica.process is None for replica in replicas)
    assert all(replica.stops >= 1 for replica in replicas)


def test_generic_pool_cleans_earlier_replicas_when_spawn_fails():
    replicas = [
        _StartupReplica("gpu0"),
        _StartupReplica("gpu1", spawn_error="CUDA out of memory"),
    ]
    pool = WorkerPoolService(replicas)
    _assert_failed_pool_is_empty(pool, replicas, pool.spawn)


def test_generic_pool_cleans_ready_peer_when_one_replica_fails_readiness():
    replicas = [
        _StartupReplica("gpu0"),
        _StartupReplica("gpu1", ready_error="CUDA out of memory"),
    ]
    pool = WorkerPoolService(replicas)
    pool.spawn()
    _assert_failed_pool_is_empty(pool, replicas, pool.wait_ready)


def test_refinement_pool_cleans_ready_peer_when_one_vllm_replica_ooms():
    replicas = [
        _StartupReplica("gpu0"),
        _StartupReplica("gpu1", ready_error="CUDA out of memory"),
    ]
    pool = RefinementWorkerPoolService(replicas)
    _assert_failed_pool_is_empty(pool, replicas, pool.start)


def test_pool_attempts_every_cleanup_when_one_replica_stop_raises():
    replicas = [
        _StartupReplica("gpu0", stop_error="broken shutdown"),
        _StartupReplica("gpu1"),
    ]
    pool = WorkerPoolService(replicas)
    pool.spawn()
    try:
        pool.stop()
    except RuntimeError as exc:
        assert "broken shutdown" in str(exc)
    else:
        raise AssertionError("the cleanup error should still be reported")
    assert replicas[1].stops == 1
    assert replicas[1].process is None
