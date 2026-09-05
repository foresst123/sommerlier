"""Pluggable separators behind the ECAPA assignment and QC layer.

`TargetSpeakerExtractor` does two jobs: it produces two tracks from a mixture,
and it decides which track belongs to whom. Only the first is model-specific,
and swapping it is how a different separator gets evaluated without touching
the enrollment mining, the QC thresholds, or the splice logic that were tuned
against this corpus.

The two families behave differently in a way the caller has to know about:

    BSS   (Sidon, MossFormer2)  two outputs in unknown order.
                                ECAPA has to decide which is which, and that
                                decision is measurably shaky here -- similarity
                                sits at p50 0.58 where natural speech scores
                                0.70-0.90, and `_maybe_swap` / `not_a_fail` /
                                `qc_sim` all exist to cope with it.

    TSE   (USEF-TFGridNet)      one output per enrollment, run twice. The order
                                is known by construction, so the whole
                                assignment question disappears.

`ordered` is what tells the caller which case it is.
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

        The rate is returned rather than assumed: DialogueSidon decodes at
        24 kHz and USEF runs at 8 kHz, and the caller resamples back.
        """
        raise NotImplementedError

    def close(self):
        pass


# --------------------------------------------------------------------------
# Sidon -- diffusion, blind, out-of-process
# --------------------------------------------------------------------------

class SidonBackend(SeparationBackend):
    """DialogueSidon over the worker's stdin/stdout protocol.

    Kept out-of-process because the model is exported with `torch.export` and
    pins its own torch state; the exchange is via .npy files in a scratch dir
    rather than the pipe, since a 48s window of float32 is megabytes.
    """

    name = "sidon"
    ordered = False          # blind: ECAPA decides which track is whom

    def __init__(self, process, temp_dir):
        self.process = process
        self._temp_dir = temp_dir
        self._counter = 0

    def separate(self, mixture, sample_rate, enroll_A=None, enroll_B=None):
        import json

        if not self.process:
            raise RuntimeError("Sidon worker process is not connected.")

        self._counter += 1
        req_id = str(self._counter)
        temp_path = os.path.join(self._temp_dir, f"mix_{req_id}.npy")
        np.save(temp_path, np.ascontiguousarray(mixture, dtype=np.float32))

        produced = [temp_path]
        try:
            self.process.stdin.write(json.dumps({
                "id": req_id, "audio_path": temp_path, "sample_rate": sample_rate,
            }) + "\n")
            self.process.stdin.flush()

            resp = None
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("Sidon worker crashed or closed stdout")
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                if parsed.get("id") == req_id:
                    resp = parsed
                    break

            for key in ("track_1_path", "track_2_path"):
                if resp.get(key):
                    produced.append(resp[key])
            if resp.get("error"):
                raise RuntimeError(f"Sidon worker error: {resp['error']}")

            out_sr = int(resp.get("target_sr") or sample_rate)
            return np.load(resp["track_1_path"]), np.load(resp["track_2_path"]), out_sr
        finally:
            # Always, including on worker errors: a long run would otherwise
            # leak one .npy per failed overlap.
            for path in produced:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
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
        /t/. Whether that costs more than Sidon's 46% frame gating and 4.8 dB
        run-to-run spread is exactly what having both backends lets us measure.

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

BACKENDS = {"sidon": SidonBackend, "usef": UsefOnnxBackend}


def make_backend(name, *, process=None, temp_dir=None, device=None, logger=None):
    """Build the separator named in the profile.

    Unknown names fail here rather than silently falling back: a typo in a
    config that quietly kept the old model would make two runs look comparable
    when they are not.
    """
    key = (name or "sidon").strip().lower()
    if key not in BACKENDS:
        raise ValueError(f"unknown separator {name!r}; choose from {sorted(BACKENDS)}")
    if key == "sidon":
        return SidonBackend(process=process, temp_dir=temp_dir)
    return UsefOnnxBackend(device=device, logger=logger)
