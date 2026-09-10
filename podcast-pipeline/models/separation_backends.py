"""Pluggable separators behind the ECAPA assignment and QC layer.

`BssSeparator` does two jobs: it produces two tracks from a mixture,
and it decides which track belongs to whom. Only the first is model-specific,
and swapping it is how a different separator gets evaluated without touching
the enrollment mining, the QC thresholds, or the splice logic that were tuned
against this corpus.

One backend ships: DialogueSidon, which is blind and generative.

Blind means it is never told who is in the mixture -- it returns two tracks in
whatever order it chose, and ECAPA decides which is which. That decision is
what `_repair_chunk_swaps`, `not_a_fail` and `qc_sim` exist for, and it is why
the window handed to this backend has to contain solo speech from at least one
speaker: with nothing either track can be scored against, the assignment has no
evidence to work from.

USEF-TFGridNet used to ship alongside it and is gone. It was target-conditioned
and masking -- told whose voice to extract, so track 1 *was* speaker A and no
assignment ran. That difference is not a tuning knob, and while both were
present the constants shared between them drifted toward whichever ran last:
the mixture window ended up pinned at USEF's 2s, which is a hard constraint of
its ONNX graph and ten times shorter than the 20s chunk Sidon is built around.
Feeding Sidon 2s windows with no solo audio in them took ECAPA similarity from
p50 0.58 down to p50 0.15 -- the score two unrelated speakers get.

The generative half deserves stating plainly, because it is a property of the
corpus and not of the code: Sidon resynthesises through a diffusion head and a
VAE decoder, so its output is audio the model produced, not audio the
microphone recorded. Whatever it fills in becomes training data for whatever is
trained on this corpus. That is a deliberate trade being made here, not an
oversight -- see doc/audio-cleanliness.md for the argument that ruled it out
before, and treat any corpus built with it as carrying that caveat.
"""

import os

import numpy as np


class SeparationBackend:
    """Produce two tracks from one mixture.

    `ordered` is True when track 1 is already speaker A. Backends that separate
    blindly leave it False and let the caller assign.
    """

    name = "base"
    ordered = False

    def separate(self, mixture, sample_rate, enroll_A=None, enroll_B=None):
        """Return (track_1, track_2, output_sample_rate) as float32 arrays.

        The rate is returned rather than assumed: Sidon's VAE decoder emits
        24 kHz whatever it was fed, independently of the rate the pipeline
        carries, and the caller resamples back.
        """
        raise NotImplementedError

    def set_process(self, proc):
        """Re-point at a restarted worker. In-process backends ignore it."""

    def close(self):
        pass


# --------------------------------------------------------------------------
# DialogueSidon -- blind, generative, out-of-process
# --------------------------------------------------------------------------

class SidonBackend(SeparationBackend):
    """DialogueSidon through its worker subprocess.

    Out of process because it needs `diffusers` and a torch build the rest of
    the pipeline does not agree with; the worker owns its own interpreter and
    speaks one JSON object per line over stdin/stdout.

    Arrays cross the boundary as .npy files rather than inline, which is why a
    temp dir is part of the constructor: a full podcast issues thousands of
    these and base64 in JSON would dominate the exchange.
    """

    name = "sidon"
    ordered = False                 # blind: ECAPA decides which track is whom

    def __init__(self, *, process=None, temp_dir=None, device=None, logger=None):
        import tempfile

        self._process = process
        self._temp_dir = temp_dir or tempfile.mkdtemp(prefix="sidon_exchange_")
        self._logger = logger
        self._counter = 0

    def set_process(self, proc):
        self._process = proc

    def separate(self, mixture, sample_rate, enroll_A=None, enroll_B=None):
        """Blind separation: the enrollments are not used, and cannot be.

        Sidon is not target-conditioned -- it splits a dialogue into its two
        voices without being told who they are. The enrollments still reach
        this method because the interface is shared with a conditioned backend;
        they are consumed by the ECAPA assignment that runs afterwards.
        """
        import json

        if self._process is None:
            raise RuntimeError(
                "the Sidon worker is not connected; separation cannot run")
        mixture = np.ascontiguousarray(mixture, dtype=np.float32).reshape(-1)
        if not len(mixture):
            raise ValueError("empty mixture")

        self._counter += 1
        req_id = str(self._counter)
        mix_path = os.path.join(self._temp_dir, f"mix_{req_id}.npy")
        np.save(mix_path, mixture)

        produced = [mix_path]
        try:
            self._process.stdin.write(json.dumps(
                {"id": req_id, "audio_path": mix_path,
                 "sample_rate": int(sample_rate)}) + "\n")
            self._process.stdin.flush()

            # Read until this request's id comes back. The worker also prints
            # progress and warnings on stdout, so anything that is not the
            # answer is skipped rather than treated as a protocol error.
            resp = None
            while resp is None:
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError("Sidon worker closed stdout")
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                if parsed.get("id") == req_id:
                    resp = parsed

            for key in ("track_1_path", "track_2_path"):
                if resp.get(key):
                    produced.append(resp[key])
            if resp.get("error"):
                raise RuntimeError(f"Sidon worker error: {resp['error']}")

            # The worker reports the rate it decoded at rather than echoing the
            # input rate: Sidon's VAE decoder emits 24kHz whatever it was fed,
            # and resampling back against the wrong number is silent.
            target_sr = int(resp.get("target_sr") or sample_rate)
            return (np.load(resp["track_1_path"]).astype(np.float32),
                    np.load(resp["track_2_path"]).astype(np.float32),
                    target_sr)
        finally:
            for path in produced:
                try:
                    os.path.exists(path) and os.unlink(path)
                except OSError:
                    pass


# --------------------------------------------------------------------------

BACKENDS = {"sidon": SidonBackend}


def make_backend(name, *, process=None, temp_dir=None, device=None, logger=None,
                 **_retired):
    """Build the separator named in the profile.

    Unknown names fail here rather than silently falling back: a typo in a
    config that quietly kept the old model would make two runs look comparable
    when they are not.

    `process` and `temp_dir` are only meaningful to an out-of-process backend;
    an in-process one ignores them rather than refusing the keyword, so the
    caller does not have to know which kind it asked for.
    """
    key = (name or "sidon").strip().lower()
    if key not in BACKENDS:
        raise ValueError(f"unknown separator {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[key](process=process, temp_dir=temp_dir,
                         device=device, logger=logger)
