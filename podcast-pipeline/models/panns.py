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

# --- non-speech, non-music contamination -------------------------------------
# The pipeline could not see any of this before: it read three groups out of
# 527 and routed on them, so a segment recorded beside a motorbike and one
# recorded in a treated room were indistinguishable to it.
#
# These are NOT used to modify audio. Enhancement was ruled out for this corpus
# -- a denoiser alters the recording and what it invents becomes training data
# that never happened. They exist so a dirty segment can be found and left out
# instead. Fewer, honest segments beat more, repaired ones.
#
# Split three ways because the three mean different things downstream, not for
# reporting neatness. Background voices break diarization and put words in the
# transcript that nobody in the conversation said; the other two mostly cost
# ASR accuracy.
NOISE_SPEECH_LABELS = ("Chatter", "Crowd", "Hubbub, speech noise, speech babble",
                       "Babbling", "Children playing", "Applause", "Clapping",
                       "Cheering", "Television", "Radio")
NOISE_ENV_LABELS = ("Vehicle", "Motor vehicle (road)", "Traffic noise, roadway noise",
                    "Motorcycle", "Vehicle horn, car horn, honking", "Car",
                    "Truck", "Bus", "Siren", "Emergency vehicle", "Train",
                    "Aircraft", "Wind", "Wind noise (microphone)", "Rain",
                    "Rain on surface", "Thunder", "Bird",
                    "Bird vocalization, bird call, bird song", "Dog",
                    "Stream", "Water")
NOISE_ROOM_LABELS = ("Typing", "Computer keyboard", "Typewriter",
                     "Mechanical fan", "Air conditioning", "Door", "Sliding door",
                     "Dishes, pots, and pans", "Cutlery, silverware", "Clatter",
                     "Rustle", "Rustling leaves", "Thump, thud", "Walk, footsteps",
                     "Hum", "Mains hum", "Noise", "Static",
                     "Tap", "Squeak", "Scratch")

# Deliberately NOT noise: these come out of the speakers themselves and a
# full-duplex conversation corpus wants them. Breathing before a turn and a
# laugh over someone else's sentence are the phenomena being collected, not
# contamination to be filtered.
#
#   Breathing, Cough, Throat clearing, Sneeze, Sniff, Laughter, Giggle,
#   Humming, Sigh, Gasp, Whispering
#
# "Humming" is already claimed by SINGING_LABELS above, where it belongs.

NOISE_GROUPS = {"noise_speech": NOISE_SPEECH_LABELS,
                "noise_env": NOISE_ENV_LABELS,
                "noise_room": NOISE_ROOM_LABELS}


class PANNSDetector:
    """Wrapper for PANNS Audio Tagging (background music detection)."""
    
    def __init__(self, device: str = None):
        # Label groups whose misses have already been reported, so a warning
        # about a stale name appears once per run rather than once per chunk.
        self._reported_missing = set()

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

        self._panns_device = panns_device

        # Both taggers are 312MB checkpoints and each is loaded on first use.
        # The clip-level one (AudioTagging) answers only `detect_music`, which
        # is the per-segment fallback path -- a run with a music map never
        # calls it. Loading it eagerly cost that 312MB of download and VRAM on
        # every run, for a model whose label list is all the main path wanted.
        self._tagger = None
        self._sed = None
        self._labels = None

    @property
    def labels(self):
        """The 527 AudioSet label names, in column order.

        Read from panns_inference's own config, which loads
        class_labels_indices.csv at import -- no checkpoint involved. That is
        what lets both 312MB taggers stay lazy: the columns a group maps to are
        the same names whether or not a model is in memory. Falls back to the
        clip tagger's own list if the package is laid out differently, which
        costs a checkpoint load but never a wrong answer.
        """
        if self._labels is None:
            try:
                from panns_inference.config import labels
                self._labels = labels
            except Exception:
                self._labels = self.model.labels
        return self._labels

    def _unwrap(self, wrapper):
        """Pin the module to one device instead of every visible GPU.

        Both wrappers spread the model with DataParallel (they print "GPU
        number: 2"). Inference here always runs at batch size 1, so the second
        replica never receives a sample -- it only burns VRAM on the GPU
        hosting the Qwen3 and separator workers, and adds scatter/gather.
        """
        inner = getattr(wrapper, "model", None)
        if isinstance(inner, torch.nn.DataParallel):
            wrapper.model = inner.module.to(self.device)
            wrapper.device = self.device
        return wrapper

    @property
    def model(self):
        """The clip-level tagger, loaded on first use.

        `PANNS_CHECKPOINT` points at a pre-staged copy; None lets
        panns_inference fetch it, which on a box without wget fails -- see
        download_offline_weights.py.
        """
        if self._tagger is None:
            checkpoint_path = os.environ.get("PANNS_CHECKPOINT") or None
            self._tagger = self._unwrap(AudioTagging(
                checkpoint_path=checkpoint_path, device=self._panns_device))
        return self._tagger
        
    # Frame rate of Cnn14_DecisionLevelMax: 32000 Hz / hop 320 = 100 fps.
    SED_FPS = 100.0
    SED_HOP = 320                  # samples per output frame, at 32kHz

    # Cnn14_DecisionLevelMax pools time by 2 five times over, so it makes one
    # decision per 32 output frames and then repeats it -- `interpolate` in
    # panns_inference is a repeat, not a smoothing. The 100 fps the model
    # advertises is therefore a presentation: the real resolution is
    # 32 * 10ms = 320ms, and every frame inside one of those blocks carries
    # exactly the same number.
    SED_DECISION_FRAMES = 32

    # Below one second the pooling stack has nothing left to reduce.
    SED_MIN_SAMPLES = 32000

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
        checkpoint = os.environ.get("PANNS_SED_CHECKPOINT") or None
        self._sed = self._unwrap(SoundEventDetection(
            checkpoint_path=checkpoint, device=self._panns_device))
        return self._sed

    def _label_columns(self, names, group=None):
        """Column indices of `names` in this build's label list.

        A name that matches nothing contributes nothing, silently -- which is
        how a typo in a group turns into a detector that always answers zero
        and looks like clean audio. Report the misses once per group instead.
        """
        available = {label.lower(): i for i, label in enumerate(self.labels)}
        cols, missing = [], []
        for name in names:
            index = available.get(name.lower())
            if index is None:
                missing.append(name)
            else:
                cols.append(index)
        if missing and group and group not in self._reported_missing:
            self._reported_missing.add(group)
            print(f"[PANNs] {group}: {len(missing)} label(s) not in this "
                  f"build's AudioSet list and will never fire: {missing}")
        return cols

    def _chunking(self):
        """(chunk, context) in samples, both whole decision blocks.

        The model's 320ms grid is aligned to the start of whatever it is
        handed, so a chunk that is not a whole number of blocks starts the next
        one on a different phase: the grid would shift every 60 seconds and a
        boundary's error would depend on where in the recording it fell.
        Rounding both to the block keeps one grid across the whole file.

        The chunk rounds down and the context rounds up, and the asymmetry is
        deliberate -- context is fed and then discarded, so too much of it
        costs a little compute, while too little (rounding 0.25s down to zero)
        silently removes the thing it exists for.
        """
        block = self.SED_HOP * self.SED_DECISION_FRAMES
        chunk = max(int(self.SED_CHUNK_SECONDS * 32000) // block, 1) * block
        wanted = self.SED_CONTEXT_SECONDS * 32000
        context = 0 if wanted <= 0 else max(1, round(wanted / block)) * block
        return chunk, context

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
        chunk, context = self._chunking()

        if total <= chunk:
            return sed.inference(audio[None, :])[0]

        pieces = []
        for start in range(0, total, chunk):
            end = min(start + chunk, total)

            fed_start = max(0, start - context)
            fed_end = min(total, end + context)
            # The model needs a second of audio to survive its pooling stack.
            # Borrow it from behind first -- that keeps the block grid, since
            # everything here is a whole number of blocks -- and only reach
            # forward when there is nothing behind, which is the first chunk of
            # a recording barely longer than one chunk.
            if fed_end - fed_start < self.SED_MIN_SAMPLES:
                fed_start = max(0, fed_end - self.SED_MIN_SAMPLES)
                if fed_end - fed_start < self.SED_MIN_SAMPLES:
                    fed_end = min(total, fed_start + self.SED_MIN_SAMPLES)

            frames = sed.inference(audio[None, fed_start:fed_end])[0]

            # Keep only the frames this chunk is responsible for. The model
            # pads its output to the input length, so frame i covers sample
            # i * hop of what it was fed.
            lo = (start - fed_start) // hop
            hi = min(lo + (end - start) // hop, len(frames))
            pieces.append(frames[lo:hi])

        return np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 527), np.float32)

    def framewise_raw(self, audio_array, sample_rate: int = 32000):
        """All 527 AudioSet scores per frame, plus how the audio was scaled.

        The single preprocessing path. `tag_framewise` groups what this
        returns, and tools/dump_panns.py writes it out unchanged -- so an
        analysis of the dump is an analysis of exactly the numbers the pipeline
        routed on, not of a second signal that resembles them.

        Returns (framewise, fps, scale) where `framewise` is (frames, 527) and
        `scale` is the factor the waveform was multiplied by. The scale is
        carried out because it is not a detail: normalising to a fixed peak
        over the whole recording means one loud transient -- a door, a clipped
        laugh -- quietly lowers every score in the file. Anyone setting a
        threshold on these numbers needs to know what they are relative to.
        """
        import librosa

        audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
        if sample_rate != 32000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=32000)
        if len(audio) < self.SED_MIN_SAMPLES:
            return np.zeros((0, 527), dtype=np.float32), self.SED_FPS, 1.0

        # Normalised once over the whole recording rather than per chunk: a
        # quiet chunk scaled up on its own would be judged against a different
        # loudness from its neighbours, and the labels would step at every
        # chunk boundary.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        scale = (0.9 / peak) if peak > 0 else 1.0
        if scale != 1.0:
            audio = audio * scale

        return self._sed_framewise(audio), self.SED_FPS, scale

    def group_scores(self, framewise):
        """The six routing curves, from a raw framewise matrix.

        Each group takes the strongest of its labels rather than their sum:
        "Singing" and "Male singing" fire together on the same voice, and
        adding them would double-count it.
        """
        groups = [("speech", SPEECH_LABELS), ("singing", SINGING_LABELS),
                  ("music", MUSIC_LABELS)] + list(NOISE_GROUPS.items())
        out = {}
        for key, names in groups:
            cols = self._label_columns(names, group=key)
            out[key] = (framewise[:, cols].max(axis=1) if len(framewise) and cols
                        else np.zeros(len(framewise), dtype=np.float32))
        return out

    def tag_framewise(self, audio_array, sample_rate: int = 32000):
        """Speech / singing / music / noise strength every 10ms.

        Returns (scores, fps) where scores is a dict of 1-D arrays over frames.
        """
        framewise, fps, _scale = self.framewise_raw(audio_array, sample_rate)
        return self.group_scores(framewise), fps

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
