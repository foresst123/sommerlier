import numpy as np
import torch


class DemucsRemover:
    """Wrapper for Demucs htdemucs to remove background music.

    The model is loaded once and applied in-process. Shelling out to
    `demucs.separate` per call would reload the ~300MB checkpoint every time,
    which dominates runtime when the per-segment fallback path is taken.
    """

    def __init__(self, device: str = None, segment: int = 10, overlap: float = 0.1,
                 gpu_chunk_sec: float = 60.0, gpu_chunk_overlap_sec: float = 1.0,
                 logger=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.segment = segment
        self.overlap = overlap
        # Window fed to the GPU at a time. Peak VRAM tracks this, not the
        # length of the file.
        self.gpu_chunk_sec = gpu_chunk_sec
        self.gpu_chunk_overlap_sec = gpu_chunk_overlap_sec
        self.logger = logger
        self._model = None
        self._vocals_index = None

    def _get_model(self):
        if self._model is not None:
            return self._model

        from demucs.pretrained import get_model

        model = get_model("htdemucs")
        model.to(self.device)
        model.eval()

        # htdemucs is a bag of sub-models; its transformer cannot exceed the
        # segment length it was trained on, so only shrink, never grow.
        if self.segment:
            for sub in getattr(model, "models", [model]):
                current = getattr(sub, "segment", None)
                if current and self.segment < current:
                    sub.segment = self.segment

        self._model = model
        self._vocals_index = model.sources.index("vocals")
        if self.logger:
            self.logger.info(f"Loaded Demucs htdemucs on {self.device}")
        return self._model

    def _separate(self, audio_array: np.ndarray, sample_rate: int):
        """Return the vocal stem for ``audio_array`` at ``sample_rate``.

        Chunked on this side, not handed whole to apply_model. Setting
        `sub.segment` bounds what the transformer sees, but the input tensor is
        still moved to the GPU in full first: a 2978s podcast is ~1GB resampled
        to 44.1kHz stereo, and apply_model then allocates four source stems on
        top of it. That is the 3.92 GiB allocation that failed on both a 14.5GB
        T4 and a 39GB A100 -- VRAM was scaling with file length, which it should
        not. Feeding fixed windows keeps the peak flat however long the file is.
        """
        import torchaudio.functional as AF
        from demucs.apply import apply_model

        model = self._get_model()

        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size == 0:
            return None

        chunk = max(1, int(self.gpu_chunk_sec * sample_rate))
        # Overlap is trimmed off after separation, so the model never sees a
        # window boundary in the middle of the audio it is judged on.
        pad = max(1, int(self.gpu_chunk_overlap_sec * sample_rate))

        pieces = []
        with torch.no_grad():
            for start in range(0, audio.size, chunk):
                end = min(audio.size, start + chunk)
                lo, hi = max(0, start - pad), min(audio.size, end + pad)
                wav = torch.from_numpy(np.ascontiguousarray(audio[lo:hi])).to(self.device)
                if sample_rate != model.samplerate:
                    wav = AF.resample(wav, sample_rate, model.samplerate)

                # htdemucs expects (batch, channels, time) at its channel count.
                wav = wav.unsqueeze(0).repeat(model.audio_channels, 1)

                # Demucs normalises against the mixture's own statistics. Done
                # per window: a window of near-silence would otherwise be
                # normalised against the whole file's loud passages.
                ref = wav.mean(0)
                mean, std = ref.mean(), ref.std().clamp_min(1e-8)
                wav = (wav - mean) / std

                sources = apply_model(
                    model,
                    wav.unsqueeze(0),
                    device=self.device,
                    overlap=self.overlap,
                    progress=False,
                )[0]

                piece = sources[self._vocals_index] * std + mean
                piece = piece.mean(0)                      # back to mono

                if sample_rate != model.samplerate:
                    piece = AF.resample(piece, model.samplerate, sample_rate)

                # Trim the padding back off in source-rate samples.
                head = start - lo
                pieces.append(piece[head:head + (end - start)].cpu())

                del wav, sources, piece
                if str(self.device).startswith("cuda"):
                    torch.cuda.empty_cache()

        vocals = torch.cat(pieces).numpy().astype(np.float32)
        if vocals.size < audio.size:
            vocals = np.pad(vocals, (0, audio.size - vocals.size))
        return vocals[:audio.size]

    def separate_full(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """Run Demucs on the entire audio to obtain the vocal stem."""
        try:
            return self._separate(audio_array, sample_rate)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Demucs separation failed: {e}")
            return None

    def separate_segment(self, segment_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Run Demucs on a single segment, returning the input unchanged on failure."""
        result = self.separate_full(segment_audio, sample_rate)
        if result is None:
            return segment_audio
        return result

    def unload(self):
        """Release the model and its VRAM."""
        if self._model is not None:
            self._model = None
            self._vocals_index = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
