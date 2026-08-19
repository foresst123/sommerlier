import numpy as np
import torch


class DemucsRemover:
    """Wrapper for Demucs htdemucs to remove background music.

    The model is loaded once and applied in-process. Shelling out to
    `demucs.separate` per call would reload the ~300MB checkpoint every time,
    which dominates runtime when the per-segment fallback path is taken.
    """

    def __init__(self, device: str = None, segment: int = 10, overlap: float = 0.1, logger=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.segment = segment
        self.overlap = overlap
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
        """Return the vocal stem for ``audio_array`` at ``sample_rate``."""
        import torchaudio.functional as AF
        from demucs.apply import apply_model

        model = self._get_model()

        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size == 0:
            return None

        wav = torch.from_numpy(audio).to(self.device)
        if sample_rate != model.samplerate:
            wav = AF.resample(wav, sample_rate, model.samplerate)
        resampled_len = wav.shape[-1]

        # htdemucs expects (batch, channels, time) at its own channel count.
        wav = wav.unsqueeze(0).repeat(model.audio_channels, 1)

        # Demucs normalises against the mixture's own statistics.
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std().clamp_min(1e-8)
        wav = (wav - mean) / std

        with torch.no_grad():
            # `segment` is a property of the model (htdemucs bakes its training
            # segment into the transformer), so set it there rather than passing
            # it to apply_model, which would fight the model's own chunking.
            sources = apply_model(
                model,
                wav.unsqueeze(0),
                device=self.device,
                overlap=self.overlap,
                progress=False,
            )[0]

        vocals = sources[self._vocals_index] * std + mean
        vocals = vocals.mean(0)

        # apply_model can return a slightly different length than it was given.
        if vocals.shape[-1] != resampled_len:
            vocals = vocals[..., :resampled_len]
            if vocals.shape[-1] < resampled_len:
                vocals = torch.nn.functional.pad(
                    vocals, (0, resampled_len - vocals.shape[-1])
                )

        if sample_rate != model.samplerate:
            vocals = AF.resample(vocals, model.samplerate, sample_rate)

        vocals = vocals.detach().cpu().numpy().astype(np.float32)

        # Resampling can shift the length by a sample or two.
        if len(vocals) > len(audio):
            vocals = vocals[:len(audio)]
        elif len(vocals) < len(audio):
            vocals = np.pad(vocals, (0, len(audio) - len(vocals)))
        return vocals

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
