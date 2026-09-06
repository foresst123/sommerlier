"""BS-RoFormer for music removal.

BS-RoFormer splits the spectrum into bands and runs a rotary-attention
transformer over them; it is what the SiSEC / Sound Demixing entries have been
built on since 2024 -- roughly 2 dB better SDR on vocal isolation than the
htdemucs time-domain model this replaced.

Whether that matters here is a separate question, and worth stating plainly:
on this corpus music removal ran on **1 of 932 segments** in one file and
**0 of 200** in the other, because there is only 30s of music in 50 minutes.
The swap is worth having for a corpus with more music in it, not for this one.

The interface is the one `MusicService` calls:

    separate_full(audio, sr)                    -> vocals for the whole recording
    separate_segment(audio, sr)                 -> vocals for one segment
    separate_span(path, start, end, sr, ref)    -> one span, decoded at 44.1kHz
    unload()

`audio-separator` (MIT) is used rather than calling the model directly: it
handles weight download, band configuration and chunking, all of which differ
per checkpoint. Its API takes file paths, so arrays are written to a scratch
file and read back -- acceptable because separation runs once per recording or
once per music span, not once per segment.

Two things this module is careful about, both learned the hard way:

* **Peak VRAM must not scale with file length.** It does not, because
  `audio-separator` windows internally; what bounds the peak is
  `segment_size` x `batch_size`, so both are profile settings rather than
  library defaults. A wrapper-level chunking pass on top would duplicate the
  windowing and add seams the model's own overlap already handles.

* **The checkpoint runs at 44.1kHz stereo.** The pipeline carries 16kHz mono,
  which throws away everything the band-split model uses above 8kHz -- most of
  what separates it from Demucs. `separate_span` therefore decodes the span
  from the *original* file at the model's native rate and downsamples only
  after separating. See `_native_span` for how the level is matched back.
"""

import inspect
import os
import tempfile

import numpy as np

# The best-scoring vocal checkpoint in audio-separator's bundled
# models-scores.json: ep_368 averages 11.63 dB vocal SDR against ep_317's
# 11.43 over the same 40 tracks, at identical cost. Named rather than
# resolved dynamically so a run is reproducible: `audio-separator`'s default
# model changes between releases. The profile setting wins over the env var,
# which wins over this.
DEFAULT_MODEL = os.environ.get(
    "BS_ROFORMER_MODEL", "model_bs_roformer_ep_368_sdr_12.9628.ckpt")

# The rate the UVR checkpoints were trained at. Not a tuning knob -- changing it
# does not resample the model, it just lies to it about what it is hearing.
NATIVE_SAMPLE_RATE = 44100


class BSRoformerRemover:
    """Vocal isolation with BS-RoFormer."""

    def __init__(self, device: str = None, model_filename: str = None,
                 model_file_dir: str = None, segment_size: int = None,
                 override_model_segment_size: bool = False, overlap: int = None,
                 batch_size: int = None, chunk_duration: float = None,
                 normalization_threshold: float = None, native_fp16: bool = False,
                 torch_compile: bool = False, hi_res: bool = True,
                 logger=None, **_ignored):
        self.device = device
        self.model_filename = model_filename or DEFAULT_MODEL
        # Where audio-separator keeps checkpoints. It defaults to a temp dir
        # and downloads on first use, which is a failed run on an offline box;
        # the env var is what download_offline_weights.py tells you to export.
        model_file_dir = model_file_dir or os.environ.get("BS_ROFORMER_MODEL_DIR")
        self.model_file_dir = (os.path.expanduser(os.path.expandvars(model_file_dir))
                               if model_file_dir else None)
        self.segment_size = segment_size
        self.override_model_segment_size = bool(override_model_segment_size)
        # `overlap` is the number of overlapping prediction windows:
        # audio-separator computes step = chunk_size // overlap, so 8 costs 8x
        # the forward passes of 1. Left None on purpose -- the library then
        # falls back to the checkpoint's own `inference.num_overlap`, which is
        # the value UVR shipped with these weights. Setting it here would
        # override a tuned number with a guessed one.
        self.overlap = overlap
        # Accepted, and a no-op for this architecture: audio-separator's MDXC
        # separator says outright that "for Roformer models, batch_size is not
        # utilized due to negligible performance improvements" and runs one
        # window per iteration. Kept so a profile carrying it does not crash.
        self.batch_size = batch_size
        # Splits long inputs before separating. Without it the overlap-add
        # result and counter buffers are allocated for the whole track --
        # ~1.4MB per second of audio, so ~4GB for a 50-minute recording. On
        # CUDA those live in host RAM rather than VRAM, which makes this a
        # Kaggle OOM rather than a CUDA one, and no less fatal.
        self.chunk_duration = chunk_duration
        # The library normalizes each output stem down to this peak. At its
        # default of 0.9 a loud vocal stem is quietly attenuated, which would
        # land *after* separate_span measured its level and so escape the
        # correction there. 1.0 leaves anything below full scale untouched.
        self.normalization_threshold = normalization_threshold
        # float16 weights, a path audio-separator lists as verified for
        # (cuda, bs_roformer) specifically. Mutually exclusive with autocast --
        # passing both raises -- so it wins where it applies.
        self.native_fp16 = bool(native_fp16)
        # Verified for this model family too, but it pays a compilation cold
        # start that a handful of short music spans never earns back. Off
        # unless a profile is separating whole recordings.
        self.torch_compile = bool(torch_compile)
        # Decode music spans from the source file at 44.1kHz instead of reusing
        # the pipeline's 16kHz waveform. Off makes the stage cheaper and is the
        # right setting when the sources are themselves narrowband.
        self.hi_res = bool(hi_res)
        self.logger = logger
        self._separator = None
        self._work_dir = None

    # ------------------------------------------------------------------
    def _separator_kwargs(self):
        """Constructor arguments, filtered to what this `audio-separator` takes.

        The library renames and moves these between releases, and this runs
        pinned only to `>=0.47`. An unknown key raises TypeError at load time --
        deep inside a Kaggle run, after the weights have already downloaded --
        so keys it does not accept are dropped with a log line instead.
        """
        use_gpu = bool(self.device) and "cuda" in str(self.device)
        # Both are CUDA-only fast paths; on CPU they are refused by the
        # library's capability table and cost a warning per run.
        native_fp16 = self.native_fp16 and use_gpu
        wanted = {
            "output_dir": self._work_dir,
            "output_format": "WAV",
            # Exclusive by construction: audio-separator raises if both are set.
            "use_autocast": use_gpu and not native_fp16,
            "use_native_fp16": native_fp16,
            "use_torch_compile": self.torch_compile and use_gpu,
            "log_level": 40,       # ERROR: the library logs every chunk at INFO
        }
        if self.model_file_dir:
            wanted["model_file_dir"] = self.model_file_dir
        if self.chunk_duration is not None:
            wanted["chunk_duration"] = float(self.chunk_duration)
        if self.normalization_threshold is not None:
            wanted["normalization_threshold"] = float(self.normalization_threshold)

        # BS-RoFormer is an MDXC-architecture model in audio-separator's
        # taxonomy, so its windowing knobs live in this block. Every one of
        # them is left unset by default: the checkpoint's own config carries
        # values UVR tuned, and the library falls back to those.
        mdxc = {}
        if self.segment_size is not None:
            mdxc["segment_size"] = int(self.segment_size)
            mdxc["override_model_segment_size"] = self.override_model_segment_size
        if self.overlap is not None:
            mdxc["overlap"] = int(self.overlap)
        if self.batch_size is not None:
            mdxc["batch_size"] = int(self.batch_size)
        if mdxc:
            wanted["mdxc_params"] = mdxc

        from audio_separator.separator import Separator
        accepted = inspect.signature(Separator.__init__).parameters
        kwargs = {k: v for k, v in wanted.items() if k in accepted}
        dropped = sorted(set(wanted) - set(kwargs))
        if dropped and self.logger:
            self.logger.warning(
                f"audio-separator does not accept {dropped}; running its "
                "defaults for those. Check the installed version against "
                "requirements.txt if separation is slower than expected.")
        return kwargs

    def _get_model(self):
        if self._separator is not None:
            return self._separator

        from audio_separator.separator import Separator

        self._work_dir = tempfile.mkdtemp(prefix="bsroformer_")
        self._separator = Separator(**self._separator_kwargs())
        if self.logger:
            self.logger.info(f"Loading BS-RoFormer {self.model_filename}")
        self._separator.load_model(model_filename=self.model_filename)
        return self._separator

    # ------------------------------------------------------------------
    def _run(self, audio_array: np.ndarray, sample_rate: int):
        """Vocals for one array, or None when separation could not run.

        Accepts mono (n,) or interleaved stereo (n, 2); returns the same shape.
        """
        import soundfile as sf

        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.ndim > 2:
            audio = audio.reshape(len(audio), -1)
        if audio.size == 0:
            return None
        stereo_in = audio.ndim == 2 and audio.shape[1] > 1

        separator = self._get_model()
        in_path = os.path.join(self._work_dir, "in.wav")
        sf.write(in_path, audio, sample_rate)

        produced = [in_path]
        try:
            outputs = separator.separate(in_path)
            # The library returns names relative to output_dir.
            paths = [p if os.path.isabs(p) else os.path.join(self._work_dir, p)
                     for p in outputs]
            produced.extend(paths)

            # Stems are named "<input>_(Vocals)_<model>.wav"; picking by name
            # rather than by position because the order is not documented and
            # taking the instrumental stem would be a silent, total failure.
            vocals = next((p for p in paths if "vocal" in os.path.basename(p).lower()), None)
            if vocals is None:
                if self.logger:
                    self.logger.warning(
                        f"BS-RoFormer produced no vocal stem "
                        f"({[os.path.basename(p) for p in paths]}); keeping the mixture")
                return None

            out, out_sr = sf.read(vocals, dtype="float32", always_2d=stereo_in)
            if not stereo_in and out.ndim > 1:
                out = out.mean(axis=1)
            if out_sr != sample_rate:
                out = _resample(out, out_sr, sample_rate)
            return _match_length(out, len(audio)).astype(np.float32)
        except Exception as exc:
            if self.logger:
                self.logger.error(f"BS-RoFormer failed ({type(exc).__name__}: {exc}); "
                                  "keeping the mixture for this span")
            return None
        finally:
            for path in produced:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    def separate_full(self, audio_array: np.ndarray, sample_rate: int):
        """Vocals for the whole recording, or None so the caller falls back."""
        return self._run(audio_array, sample_rate)

    def separate_segment(self, segment_audio: np.ndarray, sample_rate: int):
        """Vocals for one segment; returns the input unchanged on failure.

        A failed segment stays as mixture rather than becoming silence, because
        silence would enter the dataset labelled as speech.
        """
        result = self._run(segment_audio, sample_rate)
        return segment_audio if result is None else result

    def separate_span(self, source_path: str, start: float, end: float,
                      out_sr: int, reference: np.ndarray):
        """One music span, separated at 44.1kHz, returned at `out_sr` mono.

        `reference` is the pipeline's own 16kHz slice for the same span. It is
        needed for two things: the exact output length, and the level. The
        pipeline normalizes each recording to -20 dBFS at decode and does not
        keep the gain it applied, so a span decoded fresh from the source is at
        the *original* level -- writing it back unscaled would leave an audible
        step at both seams and hand ASR a passage at the wrong loudness.
        Recovering the gain as the RMS ratio of the two mixtures needs no
        plumbing and stays correct if the normalization ever changes.

        Returns None when the hi-res path is unavailable, so the caller can
        fall back to separating the 16kHz slice it already has.
        """
        if not self.hi_res or not source_path or not os.path.exists(source_path):
            return None
        duration = end - start
        if duration <= 0 or reference is None or len(reference) == 0:
            return None
        try:
            import librosa

            mix, sr = librosa.load(source_path, sr=NATIVE_SAMPLE_RATE, mono=False,
                                   offset=float(start), duration=float(duration))
            # librosa gives (channels, n) for stereo; soundfile wants (n, channels).
            mix = np.asarray(mix, dtype=np.float32)
            if mix.ndim == 2:
                mix = mix.T
            if mix.size == 0:
                return None

            vocals = self._run(mix, sr)
            if vocals is None:
                return None

            vocals = _to_mono(vocals)
            mix_mono = _to_mono(mix)
            if sr != out_sr:
                vocals = _resample(vocals, sr, out_sr)
                mix_mono = _resample(mix_mono, sr, out_sr)

            scale = _level_ratio(reference, mix_mono)
            out = _match_length(vocals * scale, len(reference))
            if self.logger:
                self.logger.debug(
                    f"Hi-res span {start:.1f}-{end:.1f}s separated at {sr}Hz "
                    f"(level x{scale:.3f})")
            return np.clip(out, -1.0, 1.0).astype(np.float32)
        except Exception as exc:
            if self.logger:
                self.logger.warning(
                    f"Hi-res separation of {start:.1f}-{end:.1f}s failed "
                    f"({type(exc).__name__}: {exc}); falling back to the "
                    f"{out_sr}Hz slice")
            return None

    def unload(self):
        import shutil
        self._separator = None
        if self._work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._work_dir = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# --- array helpers, kept out of the class so the tests can reach them --------

def _to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa
    # librosa wants (channels, n); this module carries (n, channels).
    if audio.ndim == 2:
        return librosa.resample(audio.T, orig_sr=orig_sr, target_sr=target_sr).T
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def _match_length(audio: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad to exactly `length` frames.

    Callers write this straight into a waveform slice, so a resampler that is
    one frame out must not become an exception or a shifted span.
    """
    if len(audio) == length:
        return audio
    if len(audio) > length:
        return audio[:length]
    pad = [(0, length - len(audio))] + [(0, 0)] * (audio.ndim - 1)
    return np.pad(audio, pad)


def _level_ratio(reference: np.ndarray, mixture: np.ndarray) -> float:
    """The gain the pipeline applied to this recording, read off one span.

    The pipeline's waveform is the source times a constant, so the ratio of
    RMS over several seconds recovers that constant to well within the error
    of the two different decoders involved. Silence on either side means there
    is nothing to measure and nothing to get wrong: leave the level alone.
    """
    n = min(len(reference), len(mixture))
    if n == 0:
        return 1.0
    ref_rms = float(np.sqrt(np.mean(np.square(reference[:n], dtype=np.float64))))
    mix_rms = float(np.sqrt(np.mean(np.square(mixture[:n], dtype=np.float64))))
    if ref_rms <= 1e-9 or mix_rms <= 1e-9:
        return 1.0
    return ref_rms / mix_rms
