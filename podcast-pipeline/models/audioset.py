"""The AudioSet label groups this pipeline routes on, and how to read them.

Kept apart from any one tagger on purpose. PANNs and SSLAM both score all 527
AudioSet labels, so the only thing that made them comparable was routing both
through identical group definitions -- had the groups lived inside a tagger,
swapping the tagger would have swapped the label lists too and no measurement
across the two would have meant anything.

Matched by name rather than index: a checkpoint exposes its own `labels` and
the ordering is not guaranteed stable across releases.
"""

import csv
import os

import numpy as np

# The official AudioSet ordering, shipped rather than imported. Both taggers
# emit 527 scores in this order, and the alternative -- reading the list out of
# panns_inference -- would keep a dependency on the tagger being removed just to
# know what column 137 means.
LABELS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "audioset_labels.csv")


def audioset_labels():
    """The 527 display names, in index order."""
    with open(LABELS_CSV, encoding="utf-8") as f:
        return [row[2] for row in list(csv.reader(f))[1:]]


SPEECH_LABELS = ("Speech",)
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

NOISE_GROUPS = {"noise_speech": NOISE_SPEECH_LABELS,
                "noise_env": NOISE_ENV_LABELS,
                "noise_room": NOISE_ROOM_LABELS}

# Retired from routing, kept for the record. There used to be a SINGING kind
# that read these; utils/music_map.py explains why it is gone.
#
# Folding them into MUSIC_LABELS instead was the obvious repair, and it was
# measured rather than assumed. On three recordings it flags:
#
#     thu_that_thach_10m   +1.5s
#     lm8                  +3.0s
#     vimeanhphanchiatay  +47.0s   <- of which 39.5s carries speech >= 0.2
#
# Those 47 seconds are "Male singing", "Female singing" and "Humming" firing on
# ordinary Vietnamese speech -- the same false positive the old singing
# threshold was calibrated around. Folding would send 39.5s of plain speech
# through a separator that has nothing to remove, and permanently cut the other
# 7.5s. So they stay out.
#
# The cost of leaving them out, stated plainly: a-cappella singing -- a voice
# with no instruments under it -- scores low on MUSIC_LABELS and is therefore
# not detected at all. None was observed in this corpus. If it appears, this is
# the decision to revisit.
SINGING_LABELS = ("Singing", "Choir", "Male singing", "Female singing",
                  "Rapping", "Yodeling", "Chant", "Humming", "Song")


def label_columns(labels, names, group=None, reported=None):
    """Column indices of `names` in a checkpoint's label list.

    A name that matches nothing contributes nothing, silently -- which is how a
    typo in a group turns into a detector that always answers zero and looks
    like clean audio. Report the misses once per group instead.
    """
    available = {label.lower(): i for i, label in enumerate(labels)}
    cols, missing = [], []
    for name in names:
        index = available.get(name.lower())
        if index is None:
            missing.append(name)
        else:
            cols.append(index)
    if missing and group and reported is not None and group not in reported:
        reported.add(group)
        print(f"[audioset] {group}: {len(missing)} label(s) not in this "
              f"build's AudioSet list and will never fire: {missing}")
    return cols


def group_scores(framewise, labels, reported=None):
    """The routing curves, from a raw (frames, 527) matrix.

    Each group takes the strongest of its labels rather than their sum: two
    labels for the same event fire together, and adding them would double-count
    it.
    """
    groups = [("speech", SPEECH_LABELS), ("music", MUSIC_LABELS)]
    groups += list(NOISE_GROUPS.items())
    out = {}
    for key, names in groups:
        cols = label_columns(labels, names, group=key, reported=reported)
        out[key] = (framewise[:, cols].max(axis=1) if len(framewise) and cols
                    else np.zeros(len(framewise), dtype=np.float32))
    return out
