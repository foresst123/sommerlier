import types

import numpy as np

import assignment_worker


class _Embedder:
    def __init__(self, providers):
        self._providers = providers
        self.calls = 0

    def active_providers(self):
        return list(self._providers)

    def embed(self, audio, sample_rate):
        self.calls += 1
        assert audio.dtype == np.float32 and sample_rate == 16000


def _sep(providers):
    embedder = _Embedder(providers)
    return types.SimpleNamespace(speaker_embedder=embedder), embedder


def test_reports_gpu_when_the_cuda_provider_is_active():
    sep, embedder = _sep(["CUDAExecutionProvider", "CPUExecutionProvider"])
    line = assignment_worker.describe_embedder(sep, "cuda:1", 4)
    assert "running_on=GPU" in line and "requested=cuda:1" in line
    assert "threads=4" in line and "usable_cpus=" in line
    assert "CUDA was requested" not in line
    assert embedder.calls == 1


def test_warns_when_cuda_was_requested_but_only_the_cpu_provider_loaded():
    sep, _ = _sep(["CPUExecutionProvider"])
    line = assignment_worker.describe_embedder(sep, "cuda:0", 2)
    assert "running_on=CPU" in line
    assert "CUDA was requested but WeSpeaker runs on the CPU" in line


def test_no_warning_when_the_cpu_was_asked_for():
    sep, _ = _sep(["CPUExecutionProvider"])
    line = assignment_worker.describe_embedder(sep, "cpu", 2)
    assert "running_on=CPU" in line and "CUDA was requested" not in line


def test_the_ready_message_carries_the_report_and_flags_a_cpu_fallback(monkeypatch, capsys):
    import io
    import json
    import sys

    class Sep:
        def __init__(self, providers):
            self.speaker_embedder = _Embedder(providers)

        def _get_embedding(self, audio, sr):
            return None

    def ready_message(providers, device):
        monkeypatch.setattr(assignment_worker, "build_separator",
                            lambda dev, threads=None: Sep(providers))
        monkeypatch.setattr(sys, "argv", ["assignment_worker.py", "--device", device])
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        real = sys.stdout
        buffer = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buffer)
        try:
            assignment_worker.serve()
        finally:
            monkeypatch.setattr(sys, "stdout", real)
        return json.loads(buffer.getvalue().splitlines()[0])

    gpu = ready_message(["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda:0")
    assert gpu["status"] == "ready" and "running_on=GPU" in gpu["info"]
    assert gpu["warning"] is False
    cpu = ready_message(["CPUExecutionProvider"], "cuda:0")
    assert cpu["warning"] is True and "runs on the CPU" in cpu["info"]
