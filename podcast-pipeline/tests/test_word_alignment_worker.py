"""The alignment worker rebuilds clips from one packed .npy and returns word rows."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import word_alignment_worker
from word_alignment_worker import handle_request


def _request(tmp_path, clips, **extra):
    path = tmp_path / "batch.npy"
    np.save(path, np.concatenate(clips))
    items = [{"index": f"{i:05d}", "start": float(i), "end": float(i) + 1.0,
              "text": "xin chào", "n": len(clip)} for i, clip in enumerate(clips)]
    return {"id": "r1", "audio_path": str(path), "items": items, **extra}


def test_clips_are_sliced_back_out_of_the_packed_audio(tmp_path, monkeypatch):
    seen = {}

    def fake_align_prepared(prepared, model, metadata, device, interpolate_method):
        seen.update(prepared=prepared, model=model, metadata=metadata,
                    device=device, interpolate=interpolate_method)
        return {item["index"]: [{"word": "x", "start": 0.0, "end": 1.0}] for item in prepared}

    monkeypatch.setattr(word_alignment_worker, "align_prepared", fake_align_prepared)
    clips = [np.full(100, 1.0, np.float32), np.full(250, 2.0, np.float32)]
    response = handle_request("M", "META", "cuda:0", "nearest", _request(tmp_path, clips))

    assert response["id"] == "r1" and set(response["words_by_index"]) == {"00000", "00001"}
    first, second = seen["prepared"]
    assert len(first["audio"]) == 100 and np.all(first["audio"] == 1.0)
    assert len(second["audio"]) == 250 and np.all(second["audio"] == 2.0)
    assert (seen["model"], seen["metadata"], seen["device"], seen["interpolate"]) == (
        "M", "META", "cuda:0", "nearest")


def test_an_alignment_exception_is_reported_not_raised(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise ValueError("bad clip")

    monkeypatch.setattr(word_alignment_worker, "align_prepared", boom)
    response = handle_request(None, None, "cpu", "nearest",
                              _request(tmp_path, [np.zeros(10, np.float32)]))
    assert response["id"] == "r1" and "ValueError: bad clip" in response["error"]
