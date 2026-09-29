import pytest

from models import wespeaker_embedding
from models.wespeaker_embedding import WeSpeakerONNXEmbedder


class _Session:
    def __init__(self, path, sess_options=None, providers=None):
        _Session.providers = providers


@pytest.fixture(autouse=True)
def fake_session(monkeypatch):
    _Session.providers = None
    monkeypatch.setattr(wespeaker_embedding.ort, "InferenceSession", _Session)
    monkeypatch.setattr(WeSpeakerONNXEmbedder, "_model_path", lambda self: "model.onnx")


def test_the_cuda_provider_does_not_search_convolution_algorithms_per_input_length():
    WeSpeakerONNXEmbedder("cuda:1")._get_session()
    assert _Session.providers == [
        ("CUDAExecutionProvider",
         {"device_id": 1, "cudnn_conv_algo_search": "DEFAULT"}),
        "CPUExecutionProvider"]


def test_the_search_mode_can_be_chosen():
    WeSpeakerONNXEmbedder("cuda:0", cudnn_conv_algo_search="HEURISTIC")._get_session()
    assert _Session.providers[0][1]["cudnn_conv_algo_search"] == "HEURISTIC"


def test_the_cpu_gets_no_cuda_provider():
    WeSpeakerONNXEmbedder("cpu")._get_session()
    assert _Session.providers == ["CPUExecutionProvider"]
