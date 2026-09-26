"""Each GPU's BS-RoFormer separator runs in its own process."""

import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bs_roformer_process import (
    BSRoformerProcessRemover, build_bs_roformers, physical_gpu_index)

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
