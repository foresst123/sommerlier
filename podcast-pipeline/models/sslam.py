"""SSLAM as a drop-in frame-level tagger beside PANNs.

SSLAM (ICLR 2025) scores the same 527 AudioSet labels as PANNs, so the group
definitions in models/panns.py -- which labels count as speech, as singing, as
music, as each noise kind -- carry over unchanged. Only the way frames are
produced differs, and that is the whole reason to try it: SSLAM reports
mAP 50.2 on AudioSet-2M against PANNs' 38.5 for the frame-level checkpoint.

Resolution. The released checkpoint has a clip-level head only, and there is
no frame-level SSLAM to download. Two ways to get a curve out of it were tried
against a recording whose first ten seconds are a music intro:

  - applying the head to each of the 64 patch tokens (160ms each) instead of
    the pooled vector. It fails. Every label sits at 0.45-0.53 for the whole
    file -- the head was fitted on pooled features, so on a bare token its
    logit is near zero and sigmoid returns one half. The curve is flat and
    carries no information. Kept here only as the reason not to try it again.

  - sliding a short window and reading the calibrated clip head. This works,
    and works better than expected: at a one-second window Speech swings
    0.02 to 0.87 and Music 0.22 to 0.87 across that intro, against a PANNs
    range on the same audio that barely leaves 0.04-0.11 for anything but
    Speech and Music. The window is zero-padded to the 10.24s the model was
    trained on, which is out of distribution and evidently fine.

So the resolution is WINDOW_SECONDS, not 160ms, and it is bought at one forward
pass per hop. That is the honest trade: better labels, coarser edges than the
320ms PANNs actually resolves.
"""
import os
import numpy as np

MODEL_ID = os.environ.get("SSLAM_MODEL", "ta012/SSLAM_AS2M_Finetuned")

# The preprocessing the model card documents. These are not free parameters --
# the checkpoint was trained against exactly these numbers.
SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
FRAME_SHIFT_MS = 10
TARGET_FRAMES = 1024                  # 10.24s per forward pass
NORM_MEAN, NORM_STD = -4.268, 4.569

# How much audio each decision covers, and how far apart decisions sit. The
# window is what the model sees; the hop is the frame rate of the result.
WINDOW_SECONDS = float(os.environ.get("SSLAM_WINDOW", "1.0"))
HOP_SECONDS = float(os.environ.get("SSLAM_HOP", "0.5"))
FPS = 1.0 / HOP_SECONDS

# Windows per forward batch. Each one is padded to the full 10.24s buffer
# whatever WINDOW_SECONDS is, so this trades memory for throughput only.
BATCH = int(os.environ.get("SSLAM_BATCH", "16"))


def _patch_transformers():
    """transformers 5.x renamed what this checkpoint's remote code expects.

    The model ships `transformers_version: 4.51.3`. On 5.x, loading dies in
    `PreTrainedModel` looking for `all_tied_weights_keys`. SSLAM ties no
    weights, so an empty mapping is the correct answer rather than a guess.
    """
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(lambda self: {})


class SSLAMDetector:
    """Same interface as PANNSDetector for the parts the music map uses."""

    SED_FPS = FPS
    SED_MIN_SAMPLES = SAMPLE_RATE               # one second, as for PANNs

    def __init__(self, device: str = None):
        import torch
        from models.panns import PANNSDetector

        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self._model = None
        self._reported_missing = set()

        # Borrowed rather than re-declared: if the two detectors disagreed on
        # which labels are "music", the comparison between them would be
        # measuring the label lists, not the models.
        self._label_columns = PANNSDetector._label_columns.__get__(self)
        self.group_scores = PANNSDetector.group_scores.__get__(self)

        from panns_inference.config import labels as audioset_labels
        self.labels = list(audioset_labels)

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        _patch_transformers()
        from transformers import AutoModel
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        self._model = model.eval().to(self.device)
        self._head = self._find_head()
        return self._model

    def _find_head(self):
        import torch.nn as nn
        for name, module in self._model.named_modules():
            if name.endswith("head") and isinstance(module, nn.Linear):
                return module
        raise RuntimeError("SSLAM: no Linear classification head found")

    def _mel(self, audio):
        """Kaldi fbank exactly as the model card specifies, then its own norm."""
        import torch
        import torchaudio
        wave = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
        wave = wave - wave.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            wave, htk_compat=True, sample_frequency=SAMPLE_RATE,
            use_energy=False, window_type="hanning", num_mel_bins=NUM_MEL_BINS,
            dither=0.0, frame_shift=FRAME_SHIFT_MS)
        return (mel - NORM_MEAN) / (NORM_STD * 2)

    def framewise_raw(self, audio_array, sample_rate: int = SAMPLE_RATE):
        """All 527 scores per hop, plus how the audio was scaled.

        Mirrors PANNSDetector.framewise_raw down to the peak normalisation, so
        a threshold that means one thing on one detector means the same on the
        other. Returns (framewise, fps, scale).
        """
        import librosa
        import torch

        audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate,
                                     target_sr=SAMPLE_RATE)
        if len(audio) < self.SED_MIN_SAMPLES:
            return np.zeros((0, 527), dtype=np.float32), FPS, 1.0

        # Normalised once over the whole recording rather than per window, for
        # the same reason PANNs does it: a quiet window scaled up on its own
        # would be judged against a different loudness from its neighbours.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        scale = (0.9 / peak) if peak > 0 else 1.0
        if scale != 1.0:
            audio = audio * scale

        self._load()
        win = int(round(WINDOW_SECONDS * SAMPLE_RATE))
        hop = int(round(HOP_SECONDS * SAMPLE_RATE))
        n_frames = max(1, int(np.ceil(len(audio) / hop)))

        # Each window is centred on its hop so a span's edges land either side
        # of it, rather than the window covering only what follows.
        starts = (np.arange(n_frames) * hop) - (win - hop) // 2

        # Mel is computed per batch, not for the whole recording up front: a
        # half-hour file is thousands of windows, and materialising every one
        # of them costs more memory than the model does.
        out = []
        with torch.no_grad():
            for i in range(0, n_frames, BATCH):
                chunk = starts[i:i + BATCH]
                buf = torch.zeros(len(chunk), TARGET_FRAMES, NUM_MEL_BINS)
                for j, start in enumerate(chunk):
                    a, b = max(0, start), min(len(audio), start + win)
                    seg = audio[a:b]
                    if len(seg) < win:              # edges of the recording
                        head = max(0, -start)
                        seg = np.pad(seg, (head, max(0, win - len(seg) - head)))
                    mel = self._mel(seg[:win])
                    take = min(mel.shape[0], TARGET_FRAMES)
                    buf[j, :take] = mel[:take]
                pred = self._model(buf.unsqueeze(1).to(self.device))
                pred = pred[0] if isinstance(pred, (tuple, list)) else pred
                pred = getattr(pred, "logits", pred)
                out.append(torch.sigmoid(pred).reshape(len(chunk), -1)
                           .float().cpu().numpy())

        framewise = np.concatenate(out, axis=0)[:n_frames]
        return framewise.astype(np.float32), FPS, scale

    def tag_framewise(self, audio_array, sample_rate: int = SAMPLE_RATE):
        framewise, fps, _scale = self.framewise_raw(audio_array, sample_rate)
        return self.group_scores(framewise), fps

    def clip_scores(self, audio_array, sample_rate: int = SAMPLE_RATE):
        """The calibrated clip head, for checking a frame curve against."""
        import librosa
        import torch
        audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate,
                                     target_sr=SAMPLE_RATE)
        self._load()
        mel = self._mel(audio)
        if mel.shape[0] < TARGET_FRAMES:
            mel = torch.cat(
                [mel, torch.zeros(TARGET_FRAMES - mel.shape[0], NUM_MEL_BINS)])
        mel = mel[:TARGET_FRAMES].unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self._model(mel)
        pred = pred[0] if isinstance(pred, (tuple, list)) else pred
        pred = getattr(pred, "logits", pred)
        return torch.sigmoid(pred).reshape(-1).float().cpu().numpy()

    def unload(self):
        self._model = None
