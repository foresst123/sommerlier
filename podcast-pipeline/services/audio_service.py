import hashlib
import os

import numpy as np

from schemas.audio import AudioData

# Widest correction allowed when normalizing loudness. Enough to lift a
# quiet recording to target without amplifying a near-silent one into noise.
MAX_GAIN_DB = 12.0

# Cache the decoded waveform beside the source. --by_stage calls load_audio()
# once per stage on the same file: on a 50-minute podcast that was four decodes
# of ~13s each, for a result that never changes.
AUDIO_CACHE = os.environ.get("SOMMELIER_AUDIO_CACHE", "1") not in ("0", "false", "False")


class AudioService:
    """Service to handle audio loading, normalization and basic manipulations."""

    def __init__(self, logger=None):
        self.logger = logger

    # ------------------------------------------------------------------
    def _cache_path(self, file_path: str, target_sr: int) -> str:
        """Cache file for one (source, sample rate) pair.

        Keyed by absolute path so two inputs sharing a basename in different
        folders cannot share a cache entry.
        """
        key = hashlib.sha1(
            f"{os.path.abspath(file_path)}|{target_sr}".encode()).hexdigest()[:16]
        return os.path.join(os.path.dirname(os.path.abspath(file_path)),
                            f".{os.path.basename(file_path)}.{key}.npy")

    def _load_cached(self, file_path: str, target_sr: int):
        if not AUDIO_CACHE:
            return None
        cache = self._cache_path(file_path, target_sr)
        try:
            if not os.path.exists(cache):
                return None
            # A cache older than its source is stale; re-decoding costs less
            # than silently transcribing the previous version of the file.
            if os.path.getmtime(cache) < os.path.getmtime(file_path):
                return None
            # mmap: the OS pages it in on demand rather than reading ~286MB up
            # front, and pages are shared between stages opening the same file.
            return np.load(cache, mmap_mode="r")
        except Exception as e:
            if self.logger:
                self.logger.debug(
                    f"Audio cache miss for {file_path}: {type(e).__name__}")
            return None

    def _store_cached(self, file_path: str, target_sr: int, waveform: np.ndarray):
        if not AUDIO_CACHE:
            return
        cache = self._cache_path(file_path, target_sr)
        try:
            # Written through a file object: np.save(path, ...) appends ".npy"
            # when the name does not already end in it, so passing a ".tmp"
            # path produced ".tmp.npy" and the rename below found nothing.
            tmp = cache + ".tmp.npy"
            with open(tmp, "wb") as fh:
                np.save(fh, np.asarray(waveform))
            os.replace(tmp, cache)      # atomic: an interrupted run leaves no half file
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Could not cache audio: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    def _decode(self, file_path: str, target_sr: int):
        """Decode to mono float32 at `target_sr`, and measure its dBFS.

        pydub first, measured rather than assumed: on a 300s 44.1kHz stereo
        source resampled to 24kHz mono it took 0.29s and peaked at 111MB, where
        soundfile + librosa.resample took 0.64s and 163MB. librosa's resampler
        is the better one, but this pipeline feeds ASR models that were trained
        on ordinary decoded audio, and 2.2x the time is not worth it.

        soundfile is the fallback for anything pydub cannot open without
        ffmpeg, which on a box with no ffmpeg is every mp3.
        """
        try:
            waveform = self._decode_pydub(file_path, target_sr)
        except Exception:
            waveform = self._decode_soundfile(file_path, target_sr)

        rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))) if waveform.size else 0.0
        dbfs = 20.0 * np.log10(rms) if rms > 0 else float("-inf")
        return waveform, dbfs

    def _decode_soundfile(self, file_path: str, target_sr: int) -> np.ndarray:
        """Fallback when pydub cannot decode (no ffmpeg on this box)."""
        import soundfile as sf
        waveform, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != target_sr:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        return np.ascontiguousarray(waveform, dtype=np.float32)

    def _decode_pydub(self, file_path: str, target_sr: int) -> np.ndarray:
        """Fallback for formats libsndfile will not open (mp3, m4a)."""
        from pydub import AudioSegment

        seg = AudioSegment.from_file(file_path)
        seg = seg.set_frame_rate(target_sr).set_sample_width(2).set_channels(1)
        waveform = np.array(seg.get_array_of_samples(), dtype=np.float32)
        waveform /= float(1 << (8 * seg.sample_width - 1))
        return waveform

    # ------------------------------------------------------------------
    def load_audio(self, file_path: str, target_sr: int = 16000) -> AudioData:
        """Load audio file into memory, normalize and return AudioData schema."""
        cached = self._load_cached(file_path, target_sr)
        if cached is not None:
            if self.logger:
                self.logger.info(
                    f"Loading audio from {file_path} at {target_sr}Hz (cached)")
            return AudioData(
                waveform=cached,
                sample_rate=target_sr,
                name=os.path.basename(file_path),
                # Rebuilt on demand by export_service, which already has that
                # branch: holding a second full copy of the audio for the whole
                # run costs ~143MB per file and nothing else reads it.
                audio_segment=None,
                duration=len(cached) / target_sr,
            )

        if self.logger:
            self.logger.info(f"Loading audio from {file_path} at {target_sr}Hz")

        waveform, measured_dbfs = self._decode(file_path, target_sr)

        # Loudness normalization to -20 dBFS. The ASR models were trained on
        # normalized speech, so leaving a quiet recording quiet costs accuracy
        # exactly on the soft passages (backchannels) this pipeline cares about.
        target_dBFS = -20.0
        gain = target_dBFS - measured_dbfs if np.isfinite(measured_dbfs) else 0.0
        applied_gain = float(min(max(gain, -MAX_GAIN_DB), MAX_GAIN_DB))

        if self.logger:
            if abs(applied_gain - gain) > 0.01:
                self.logger.info(
                    f"Applying gain: {applied_gain:.2f} dB (capped from {gain:.2f} dB)")
            else:
                self.logger.info(f"Applying gain: {applied_gain:.2f} dB")

        # In place: pydub's apply_gain returned a whole new AudioSegment to do
        # one multiplication.
        if applied_gain:
            waveform *= 10.0 ** (applied_gain / 20.0)

        # Scaling by the peak would undo the loudness normalization just
        # applied, so only rescale when something actually clips.
        max_amplitude = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if max_amplitude > 1.0:
            waveform /= max_amplitude

        # Cached after gain, so a stage that hits the cache sees exactly the
        # samples a stage that decoded would have seen.
        self._store_cached(file_path, target_sr, waveform)

        return AudioData(
            waveform=waveform,
            sample_rate=target_sr,
            name=os.path.basename(file_path),
            audio_segment=None,
            duration=len(waveform) / target_sr,
        )
