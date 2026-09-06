"""Pluggable separators behind the ECAPA assignment and QC layer.

`TargetSpeakerExtractor` does two jobs: it produces two tracks from a mixture,
and it decides which track belongs to whom. Only the first is model-specific,
and swapping it is how a different separator gets evaluated without touching
the enrollment mining, the QC thresholds, or the splice logic that were tuned
against this corpus.

Only one backend ships now: USEF-TFGridNet, which is target-conditioned. It
runs once per enrollment, so track 1 *is* speaker A and the whole "which track
is whom" question disappears -- `ordered = True` says so.

The blind case is still described by the interface because it is what the
assignment and QC layer around it was built for. A blind separator returns two
tracks in unknown order, ECAPA has to decide which is which, and on this corpus
that decision was measurably shaky: similarity sat at p50 0.58 where natural
speech scores 0.70-0.90. `_maybe_swap`, `not_a_fail` and `qc_sim` exist for
that case; with an ordered backend they are cheap no-ops rather than dead code,
and they are what a future blind backend would plug into.
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

        The rate is returned rather than assumed: USEF runs at 8 kHz while
        the pipeline carries 16 kHz, and the caller resamples back.
        """
        raise NotImplementedError

    def close(self):
        pass


# --------------------------------------------------------------------------
# USEF-TFGridNet -- masking, target-conditioned, in-process ONNX
# --------------------------------------------------------------------------

# The exported graph bakes TF-GridNet's unfold constants in, so these three
# shapes are fixed and cannot be negotiated at run time.
USEF_SR = 8000
USEF_MIX_SAMPLES = 16000        # 2.0s
USEF_ENROLL_SAMPLES = 64000     # 8.0s

USEF_REPO = "bitsydarel/usef-tse-onnx"
USEF_DEFAULT_FILE = "usef_tse_tfgridnet_wsj0-2mix.onnx"


class UsefOnnxBackend(SeparationBackend):
    """USEF-TFGridNet through its ONNX export.

    Chosen over the PyTorch checkpoint because that one needs the upstream
    repository's module layout (`models.local.TFgridnet`, `utils.feature.STFT`)
    and a hyperpyyaml config; the ONNX graph is self-contained.

    Two things about this model shape the code below:

      * it runs at 8 kHz, so everything above 4 kHz is discarded on the way in.
        On this corpus that is 0.0075% of total energy but 18.2% of the energy
        in fricative frames -- the band that carries /s/, /x/, final /c/ and
        /t/. That was measured against DialogueSidon's 46% frame gating and
        4.8 dB run-to-run spread before that backend was removed; the number
        stands on its own as the price of running at 8 kHz.

      * the mixture window is fixed at 2s. Longer audio is chunked and the
        outputs concatenated, which is what the export's own README prescribes.

    Licence: CC BY-NC 4.0, inherited from the upstream USEF-TSE weights.
    Non-commercial use only.
    """

    name = "usef"
    ordered = True           # one pass per enrollment: the order is known

    def __init__(self, device=None, model_file=None, logger=None):
        self.logger = logger
        self.model_file = model_file or os.environ.get("USEF_ONNX_FILE", USEF_DEFAULT_FILE)
        self._session = None
        self._device = device
        self._input_names = None

    def _load(self):
        if self._session is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=USEF_REPO, filename=self.model_file)

        # CUDA first when it is there; the graph is small enough that CPU is a
        # workable fallback rather than a failure.
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in available] or ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(path, providers=providers)
        self._input_names = [i.name for i in self._session.get_inputs()]
        if self.logger:
            self.logger.info(
                f"[USEF] loaded {self.model_file} on {providers[0]} "
                f"(8 kHz, 2s windows, inputs={self._input_names})")

    @staticmethod
    def _resample(x, src_sr, dst_sr):
        if src_sr == dst_sr:
            return np.asarray(x, dtype=np.float32)
        import torch
        import torchaudio.functional as AF
        t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        return AF.resample(t, src_sr, dst_sr).numpy().astype(np.float32)

    @classmethod
    def _prepare_enrollment(cls, enroll, sample_rate):
        """One 8s clip at 8 kHz, from however many clips the caller mined.

        The clips are concatenated before trimming so that a speaker whose
        enrollment is spread over several short spans still fills the window.
        """
        if enroll is None:
            raise ValueError("USEF is a target-conditioned model: it needs an enrollment.")
        clips = enroll if isinstance(enroll, (list, tuple)) else [enroll]
        clips = [np.asarray(c, dtype=np.float32).reshape(-1) for c in clips
                 if c is not None and len(c)]
        if not clips:
            raise ValueError("Enrollment is empty.")

        joined = cls._resample(np.concatenate(clips), sample_rate, USEF_SR)
        if len(joined) >= USEF_ENROLL_SAMPLES:
            return joined[:USEF_ENROLL_SAMPLES]
        return np.pad(joined, (0, USEF_ENROLL_SAMPLES - len(joined)))

    def _extract_one(self, mixture_8k, enrollment_8k):
        """Run the fixed-size graph across a mixture of any length."""
        enroll_in = enrollment_8k.reshape(1, -1)
        out = np.zeros(len(mixture_8k), dtype=np.float32)

        for start in range(0, len(mixture_8k), USEF_MIX_SAMPLES):
            chunk = mixture_8k[start:start + USEF_MIX_SAMPLES]
            take = len(chunk)
            if take < USEF_MIX_SAMPLES:
                chunk = np.pad(chunk, (0, USEF_MIX_SAMPLES - take))
            fed = self._session.run(
                None,
                {self._input_names[0]: chunk.reshape(1, -1).astype(np.float32),
                 self._input_names[1]: enroll_in.astype(np.float32)},
            )[0].reshape(-1)
            out[start:start + take] = fed[:take]
        return out

    def separate(self, mixture, sample_rate, enroll_A=None, enroll_B=None):
        self._load()

        mix_8k = self._resample(np.asarray(mixture, dtype=np.float32).reshape(-1),
                                sample_rate, USEF_SR)
        track_a = self._extract_one(mix_8k, self._prepare_enrollment(enroll_A, sample_rate))
        track_b = self._extract_one(mix_8k, self._prepare_enrollment(enroll_B, sample_rate))
        return track_a, track_b, USEF_SR

    def close(self):
        self._session = None


# --------------------------------------------------------------------------

BACKENDS = {"usef": UsefOnnxBackend}


def make_backend(name, *, device=None, logger=None, **_retired):
    """Build the separator named in the profile.

    Unknown names fail here rather than silently falling back: a typo in a
    config that quietly kept the old model would make two runs look comparable
    when they are not.

    `**_retired` swallows `process` and `temp_dir`, which only an
    out-of-process backend ever needed. Callers that still pass them keep
    working instead of failing on a keyword.
    """
    key = (name or "usef").strip().lower()
    if key not in BACKENDS:
        raise ValueError(f"unknown separator {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[key](device=device, logger=logger)
