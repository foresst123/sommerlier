"""The seam between "produce two tracks" and "decide which is whose".

Only the first is model-specific. `BssSeparator` runs a backend and then works
out, with ECAPA, which returned track belongs to which speaker -- and these pin
that seam, including the part that caused a real regression: what the backend
needs in the window handed to it.

Run:  python -m pytest tests/test_separation_backends.py -q   (from podcast-pipeline/)
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.separation_backends import BACKENDS, SidonBackend, make_backend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _profiles():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)["environments"]


# --- choosing a backend -----------------------------------------------------

def test_sidon_is_the_default():
    assert make_backend(None, process=None, temp_dir="/tmp").name == "sidon"


def test_names_are_case_and_space_insensitive():
    assert make_backend("  SIDON ", process=None, temp_dir="/tmp").name == "sidon"


def test_an_unknown_name_fails_rather_than_falling_back():
    """A typo that quietly kept a different model would make two runs look
    comparable when they are not."""
    with pytest.raises(ValueError, match="unknown separator"):
        make_backend("usef", process=None, temp_dir="/tmp")


def test_only_sidon_ships():
    """USEF-TFGridNet was removed rather than left switched off.

    While both were present the constants they share drifted toward whichever
    ran last, and the mixture window ended up pinned at USEF's 2s -- a hard
    constraint of its ONNX graph and the wrong length for a blind separator.
    Keeping an unused second backend is what let that happen quietly.
    """
    assert set(BACKENDS) == {"sidon"}


def test_both_profiles_declare_a_separator():
    """Named, not defaulted: which separator ran is the difference between two
    runs being comparable and not, so it is written down."""
    for name, profile in _profiles().items():
        assert profile["models"]["bss"]["separator"] in BACKENDS, name


# --- blind means the assignment has to run ----------------------------------

def test_sidon_declares_itself_blind():
    """`ordered` is what the caller branches on, and getting it wrong is silent
    in both directions: True on a blind backend hands speaker B's audio to
    speaker A, False on a conditioned one runs repair paths with nothing to
    repair."""
    assert SidonBackend.ordered is False


def test_the_assignment_path_is_reachable_for_a_blind_backend():
    """The swap repair, the not-A test and the qc_sim gate exist for exactly
    this case. With a conditioned backend they were dead code; with this one
    they are the only thing deciding who is who."""
    src = open(os.path.join(ROOT, "models", "bss_model.py"), encoding="utf-8").read()
    assert 'if not getattr(self.backend, "ordered", False):' in src
    assert "_repair_chunk_swaps" in src


def test_the_generative_backend_says_so_where_it_is_defined():
    """Sidon resynthesises through a diffusion head and a VAE decoder: its
    output is audio the model produced, not audio the microphone recorded.
    Whatever it fills in becomes training data. That has to be legible at the
    definition, not only in a commit message."""
    src = open(os.path.join(ROOT, "models", "separation_backends.py"),
               encoding="utf-8").read()
    block = src[:src.index("class SidonBackend")]
    assert "generative" in block
    assert "not audio the" in block


# --- the worker exchange ----------------------------------------------------

class _FakeProcess:
    """Stands in for the worker: answers one request, records what it was sent."""

    def __init__(self, tmp_path, target_sr=24000, error=None):
        self.tmp_path = tmp_path
        self.target_sr = target_sr
        self.error = error
        self.sent = []
        self.stdin = self
        self.stdout = self
        self._replies = []

    # stdin
    def write(self, line):
        req = json.loads(line)
        self.sent.append(req)
        if self.error:
            self._replies.append(json.dumps({"id": req["id"], "error": self.error}))
            return
        t1 = os.path.join(self.tmp_path, f"t1_{req['id']}.npy")
        t2 = os.path.join(self.tmp_path, f"t2_{req['id']}.npy")
        np.save(t1, np.ones(100, dtype=np.float32))
        np.save(t2, np.full(100, 2.0, dtype=np.float32))
        # A line of chatter first: the worker prints progress on stdout too,
        # and anything that is not the answer has to be skipped rather than
        # treated as a protocol error.
        self._replies.append("loading something")
        self._replies.append(json.dumps(
            {"id": req["id"], "track_1_path": t1, "track_2_path": t2,
             "target_sr": self.target_sr}))

    def flush(self):
        pass

    # stdout
    def readline(self):
        return (self._replies.pop(0) + "\n") if self._replies else ""


def test_a_request_carries_the_mixture_as_a_file(tmp_path):
    """Arrays cross as .npy rather than inline: a full podcast issues thousands
    of these and base64 in JSON would dominate the exchange."""
    proc = _FakeProcess(str(tmp_path))
    backend = SidonBackend(process=proc, temp_dir=str(tmp_path))
    backend.separate(np.zeros(1600, dtype=np.float32), 16000)

    req = proc.sent[0]
    assert req["sample_rate"] == 16000
    assert req["audio_path"].endswith(".npy")


def test_the_rate_the_worker_reports_is_the_one_returned(tmp_path):
    """Sidon's VAE decoder emits 24kHz whatever it was fed. Echoing the input
    rate instead would resample against the wrong number, silently."""
    proc = _FakeProcess(str(tmp_path), target_sr=24000)
    backend = SidonBackend(process=proc, temp_dir=str(tmp_path))
    _t1, _t2, sr = backend.separate(np.zeros(1600, dtype=np.float32), 16000)
    assert sr == 24000


def test_chatter_on_stdout_is_skipped_not_parsed_as_the_answer(tmp_path):
    proc = _FakeProcess(str(tmp_path))
    backend = SidonBackend(process=proc, temp_dir=str(tmp_path))
    t1, t2, _sr = backend.separate(np.zeros(1600, dtype=np.float32), 16000)
    assert t1[0] == pytest.approx(1.0)
    assert t2[0] == pytest.approx(2.0)


def test_a_worker_error_is_raised_not_returned_as_audio(tmp_path):
    proc = _FakeProcess(str(tmp_path), error="out of memory")
    backend = SidonBackend(process=proc, temp_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="out of memory"):
        backend.separate(np.zeros(1600, dtype=np.float32), 16000)


def test_the_exchange_files_are_cleaned_up(tmp_path):
    """Thousands of jobs per recording; leaving them behind fills the disk."""
    proc = _FakeProcess(str(tmp_path))
    backend = SidonBackend(process=proc, temp_dir=str(tmp_path))
    backend.separate(np.zeros(1600, dtype=np.float32), 16000)
    assert [f for f in os.listdir(tmp_path) if f.endswith(".npy")] == []


def test_no_worker_is_a_clear_error_rather_than_a_crash():
    backend = SidonBackend(process=None, temp_dir="/tmp")
    with pytest.raises(RuntimeError, match="not connected"):
        backend.separate(np.zeros(1600, dtype=np.float32), 16000)


def test_a_restarted_worker_can_be_handed_over(tmp_path):
    """The worker is released at the end of the stage and a fresh one starts
    for the next file. A backend still holding the dead process fails on every
    job from the second file onwards."""
    backend = SidonBackend(process=None, temp_dir=str(tmp_path))
    proc = _FakeProcess(str(tmp_path))
    backend.set_process(proc)
    backend.separate(np.zeros(1600, dtype=np.float32), 16000)
    assert proc.sent, "the new process was never used"
