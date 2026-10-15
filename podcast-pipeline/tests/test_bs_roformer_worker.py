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
