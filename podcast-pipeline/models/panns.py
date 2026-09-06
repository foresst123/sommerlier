import os

import numpy as np
import torch
from panns_inference import AudioTagging, SoundEventDetection

# AudioSet label groups this pipeline routes on. Cnn14 predicts all 527 on
# every call; reading only "Music" is what left singing indistinguishable from
# speech, so song lyrics reached ASR as dialogue.
#
# Matched by name rather than index: panns_inference exposes `labels` and the
# ordering is not guaranteed stable across releases.
SPEECH_LABELS = ("Speech",)
SINGING_LABELS = ("Singing", "Choir", "Male singing", "Female singing",
                  "Rapping", "Yodeling", "Chant", "Humming", "Song")
MUSIC_LABELS = ("Music", "Musical instrument", "Background music")


class PANNSDetector:
    """Wrapper for PANNS Audio Tagging (background music detection)."""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # panns_inference compares `device == "cuda"` exactly, so an indexed
        # string like "cuda:0" silently falls back to CPU ("Using CPU." in its
        # own log). It then wraps the model in DataParallel across every visible
        # GPU, so the index cannot be honoured anyway -- pin the device with
        # CUDA_VISIBLE_DEVICES if placement matters.
        panns_device = "cuda" if str(self.device).startswith("cuda") else "cpu"

        # checkpoint_path=None makes panns_inference re-download Cnn14 (312MB from
        # Zenodo, ~5 min on Kaggle) on every run. PANNS_CHECKPOINT lets the caller
        # point at a pre-staged copy; None keeps the old download behaviour.
        checkpoint_path = os.environ.get("PANNS_CHECKPOINT") or None
        self.model = AudioTagging(checkpoint_path=checkpoint_path, device=panns_device)

        # AudioTagging wraps the model in DataParallel across every visible GPU
        # (it prints "GPU number: 2"). Inference here always runs with batch size
        # 1, so the second replica never receives a sample -- it only burns VRAM
        # on the GPU hosting the Qwen3/Demucs workers and adds scatter/gather
        # overhead. Unwrap it and pin the module to the requested device.
        inner = getattr(self.model, "model", None)
        if isinstance(inner, torch.nn.DataParallel):
            self.model.model = inner.module.to(self.device)
            self.model.device = self.device

        # Frame-level tagger, loaded on first use: it is a second 312MB
        # checkpoint and only the timeline sweep needs it.
        self._sed = None
        
    # Frame rate of Cnn14_DecisionLevelMax: 32000 Hz / hop 320 = 100 fps.
    SED_FPS = 100.0
    SED_HOP = 320                  # samples per output frame, at 32kHz

    # How much audio goes through the frame tagger at once. panns_inference
    # runs one forward pass over whatever it is handed, and the framewise stack
    # keeps an activation per frame: a 50-minute podcast is 96M samples, which
    # asks for tens of gigabytes and dies on any GPU here. Chunking is not an
    # optimisation, it is what makes a full recording possible at all.
    SED_CHUNK_SECONDS = float(os.environ.get("PANNS_SED_CHUNK", "60"))

    # Context carried either side of each chunk and then thrown away. The model
    # has a receptive field; frames at a chunk edge would otherwise be judged
    # with silence on one side, which reads as a boundary that is not there.
    SED_CONTEXT_SECONDS = float(os.environ.get("PANNS_SED_CONTEXT", "1.0"))

    def _get_sed(self):
        """Load the frame-level tagger, once and only if asked for.

        AudioTagging answers "is there music in this clip" and nothing about
        *where*, so locating music meant sliding a 2s window -- 200x coarser
        than this model, which labels every 10ms.

        The trade is real and worth stating: the SED checkpoint scores
        mAP 0.385 against AudioTagging's 0.43. Time resolution is bought with
        about ten percent of label accuracy.
        """
        if getattr(self, "_sed", None) is not None:
            return self._sed
        panns_device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        checkpoint = os.environ.get("PANNS_SED_CHECKPOINT") or None
        self._sed = SoundEventDetection(checkpoint_path=checkpoint, device=panns_device)

        # Same DataParallel unwrap as AudioTagging: inference runs at batch 1,
        # so a second replica only burns VRAM on the GPU hosting other workers.
        inner = getattr(self._sed, "model", None)
        if isinstance(inner, torch.nn.DataParallel):
            self._sed.model = inner.module.to(self.device)
            self._sed.device = self.device
        return self._sed

    def _label_columns(self, names):
        """Column indices of `names` in this build's label list."""
        wanted = {n.lower() for n in names}
        return [i for i, label in enumerate(self.model.labels)
                if label.lower() in wanted]

    def _sed_framewise(self, audio):
        """Frame-level scores for a whole recording, a chunk at a time.

        Each chunk is fed with context on both sides and then cropped back to
        its own span, so the result is the same array a single pass would have
        produced -- minus the edge effects of judging a frame with silence
        beside it, and minus the memory that pass would need.
        """
        sed = self._get_sed()
        total = len(audio)
        hop = self.SED_HOP

        # Chunk and context in whole frames, so cropping is exact rather than
        # rounded: a half-frame slip per chunk would accumulate into a
        # timestamp error over a fifty-minute file.
        chunk = max(int(self.SED_CHUNK_SECONDS * 32000) // hop, 1) * hop
        context = max(int(self.SED_CONTEXT_SECONDS * 32000) // hop, 0) * hop

        if total <= chunk:
            return sed.inference(audio[None, :])[0]

        pieces = []
        for start in range(0, total, chunk):
            end = min(start + chunk, total)

            fed_start = max(0, start - context)
            fed_end = min(total, end + context)
            # The model needs a second of audio to survive its pooling stack;
            # a short final chunk borrows the length from behind it.
            if fed_end - fed_start < 32000:
                fed_start = max(0, fed_end - 32000)

            frames = sed.inference(audio[None, fed_start:fed_end])[0]

            # Keep only the frames this chunk is responsible for. The model
            # pads its output to the input length, so frame i covers sample
            # i * hop of what it was fed.
            lo = (start - fed_start) // hop
            hi = min(lo + (end - start) // hop, len(frames))
            pieces.append(frames[lo:hi])

        return np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 527), np.float32)

    def tag_framewise(self, audio_array, sample_rate: int = 32000):
        """Speech / singing / music strength every 10ms.

        Returns (scores, fps) where scores is a dict of 1-D arrays over frames.
        Each group takes the strongest of its labels rather than their sum:
        "Singing" and "Male singing" fire together on the same voice, and
        adding them would double-count it.
        """
        import librosa

        audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
        if sample_rate != 32000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=32000)
        if len(audio) < 32000:
            # Below one second the pooling stack has nothing to reduce.
            empty = np.zeros(0, dtype=np.float32)
            return {"speech": empty, "singing": empty, "music": empty}, self.SED_FPS

        # Normalised once over the whole recording rather than per chunk: a
        # quiet chunk scaled up on its own would be judged against a different
        # loudness from its neighbours, and the labels would step at every
        # chunk boundary.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = (audio / peak) * 0.9

        framewise = self._sed_framewise(audio)                     # (frames, 527)

        out = {}
        for key, names in (("speech", SPEECH_LABELS),
                           ("singing", SINGING_LABELS),
                           ("music", MUSIC_LABELS)):
            cols = self._label_columns(names)
            out[key] = (framewise[:, cols].max(axis=1) if cols
                        else np.zeros(len(framewise), dtype=np.float32))
        return out, self.SED_FPS

    def detect_music(self, audio_array, sample_rate: int = 32000, threshold: float = 0.5) -> tuple:
        """Detect if music is present in audio."""
        import librosa
        if sample_rate != 32000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=32000)
            
        # PANNs (Cnn14) requires a minimum length to pass through all pooling layers (approx 1 second)
        if len(audio_array) < 32000:
            return False, 0.0
            
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1) if audio_array.shape[1] == 2 else audio_array

        # Unlike the ASR and separation models, PANNs applies no input
        # normalization of its own, so its tagging confidence tracks absolute
        # level. Scale to a consistent peak here rather than depending on how
        # loud the source happened to be.
        peak = float(np.max(np.abs(audio_array))) if audio_array.size else 0.0
        if peak > 0:
            audio_array = (audio_array / peak) * 0.9

        clipwise_output, _ = self.model.inference(audio_array[None, :])
        
        music_idx = None
        for i, label in enumerate(self.model.labels):
            if label.lower() == "music":
                music_idx = i
                break
                
        if music_idx is not None:
            music_prob = float(clipwise_output[0, music_idx])
            has_music = music_prob > threshold
            return has_music, music_prob
        return False, 0.0
