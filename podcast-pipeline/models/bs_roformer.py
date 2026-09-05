"""BS-RoFormer as a drop-in replacement for Demucs in music removal.

Demucs (htdemucs) is a 2022 time-domain model. BS-RoFormer splits the spectrum
into bands and runs a rotary-attention transformer over them, and it is what
the SiSEC / Sound Demixing entries have been built on since 2024 -- roughly
2 dB better SDR on vocal isolation.

Whether that matters here is a separate question, and worth stating plainly:
on this corpus music removal ran on **1 of 932 segments** in one file and
**0 of 200** in the other, because there is only 30s of music in 50 minutes.
The swap is worth having for a corpus with more music in it, not for this one.

The interface is `DemucsRemover`'s, so `MusicService` does not change:

    separate_full(audio, sr)     -> vocals for the whole recording
    separate_segment(audio, sr)  -> vocals for one segment
    unload()

`audio-separator` (MIT) is used rather than calling the model directly: it
handles weight download, band configuration and chunking, all of which differ
per checkpoint. Its API takes file paths, so arrays are written to a scratch
file and read back -- acceptable because `separate_full` runs once per
recording, not once per segment.
"""

import os
import tempfile

import numpy as np

# UVR's BS-RoFormer vocal checkpoints, best SDR first. Named rather than
# resolved dynamically so a run is reproducible: `audio-separator`'s default
# model changes between releases.
DEFAULT_MODEL = os.environ.get(
    "BS_ROFORMER_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")


class BSRoformerRemover:
    """Vocal isolation with BS-RoFormer, interface-compatible with Demucs."""

    def __init__(self, device: str = None, model_filename: str = None,
                 logger=None, **_ignored):
        # **_ignored swallows Demucs' segment/overlap knobs so the same profile
        # block can drive either model without the loader having to branch.
        self.device = device
        self.model_filename = model_filename or DEFAULT_MODEL
        self.logger = logger
        self._separator = None
        self._work_dir = None

    def _get_model(self):
        if self._separator is not None:
            return self._separator

        from audio_separator.separator import Separator

        self._work_dir = tempfile.mkdtemp(prefix="bsroformer_")
        use_gpu = bool(self.device) and "cuda" in str(self.device)
        self._separator = Separator(
            output_dir=self._work_dir,
            output_format="WAV",
            use_autocast=use_gpu,
            log_level=40,          # ERROR: the library logs every chunk at INFO
        )
        if self.logger:
            self.logger.info(f"Loading BS-RoFormer {self.model_filename}")
        self._separator.load_model(model_filename=self.model_filename)
        return self._separator

    def _run(self, audio_array: np.ndarray, sample_rate: int):
        """Vocals for one array, or None when separation could not run."""
        import soundfile as sf

        audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return None

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

            out, out_sr = sf.read(vocals, dtype="float32")
            if out.ndim > 1:
                out = out.mean(axis=1)
            if out_sr != sample_rate:
                import torch
                import torchaudio.functional as AF
                out = AF.resample(torch.from_numpy(out), out_sr, sample_rate).numpy()

            # Match the caller's length exactly: MusicService writes this
            # straight into a segment slice.
            if len(out) > len(audio):
                out = out[:len(audio)]
            elif len(out) < len(audio):
                out = np.pad(out, (0, len(audio) - len(out)))
            return out.astype(np.float32)
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

    def separate_full(self, audio_array: np.ndarray, sample_rate: int):
        """Vocals for the whole recording, or None so the caller falls back."""
        return self._run(audio_array, sample_rate)

    def separate_segment(self, segment_audio: np.ndarray, sample_rate: int):
        """Vocals for one segment; returns the input unchanged on failure.

        Matches Demucs' contract: a failed segment stays as mixture rather than
        becoming silence, because silence would enter the dataset labelled as
        speech.
        """
        result = self._run(segment_audio, sample_rate)
        return segment_audio if result is None else result

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
