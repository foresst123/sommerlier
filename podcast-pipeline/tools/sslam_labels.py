"""What is actually in these recordings, according to all 527 labels.

Ranks every label by how much of the recording it claims, then sorts them into
what the corpus should do about them. The split that matters is not
speech/not-speech: laughter, breathing and backchannels are not noise in a
conversation corpus, they are the conversation. They get their own bucket so
the decision about them is made deliberately rather than by a threshold.
"""
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "stage1_matrix", "sslam_full")

# Labels that ARE the conversation, whatever a classifier calls them.
SPEECH = {"Speech", "Male speech, man speaking", "Female speech, woman speaking",
          "Child speech, kid speaking", "Conversation", "Narration, monologue",
          "Speech synthesizer", "Babbling"}

# Human, non-lexical, and worth keeping in a full-duplex dialogue corpus --
# removing them would leave a transcript of words with the turn-taking cut out.
HUMAN = {"Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle",
         "Baby laughter", "Breathing", "Sigh", "Gasp", "Cough", "Sneeze",
         "Throat clearing", "Sniff", "Hiccup", "Whispering", "Crying, sobbing",
         "Whimper", "Wail, moan", "Groan", "Grunt", "Yawn", "Hubbub, speech noise, speech babble"}

MUSICAL = {"Music", "Musical instrument", "Background music", "Theme music",
           "Singing", "Choir", "Male singing", "Female singing", "Child singing",
           "Rapping", "Humming", "Song", "Jingle (music)", "Soundtrack music"}

ROOM = {"Inside, small room", "Inside, large room or hall", "Inside, public space",
        "Outside, urban or manmade", "Outside, rural or natural", "Echo",
        "Reverberation", "Silence", "Environmental noise", "Room"}


# AudioSet puts genres in their own subtree, so "Pop music", "Afrobeat" and
# "Music of Asia" are music without the word appearing in most of them. Missing
# these was not cosmetic: they made the "other" bucket look full of noise when
# it was full of music.
GENRE = {"Pop music", "Rock music", "Hip hop music", "Classical music", "Jazz",
         "Blues", "Country", "Reggae", "Funk", "Soul music", "Disco", "Techno",
         "House music", "Electronic music", "Dance music", "Folk music",
         "Afrobeat", "Salsa music", "Music of Asia", "Music of Africa",
         "Music of Latin America", "Middle Eastern music", "Christian music",
         "Gospel music", "Music for children", "Exciting music", "Happy music",
         "Sad music", "Tender music", "Scary music", "Angry music",
         "Background music", "Ambient music", "New-age music", "Opera",
         "Chant", "Mantra", "Wedding music", "Christmas music", "Video game music"}


def bucket(name):
    if name in GENRE or ("music" in name.lower() and name not in SPEECH):
        return "music"
    if name in SPEECH:
        return "speech"
    if name in HUMAN:
        return "human"
    if name in MUSICAL:
        return "music"
    if name in ROOM:
        return "room"
    return "other"


def main():
    level = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
    for rec in ("thu_that_thach_10m", "lm8", "vimeanh"):
        path = os.path.join(OUT, f"{rec}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path, allow_pickle=True)
        S, fps = z["scores"], float(z["fps"])
        labels = list(z["labels"])
        share = (S >= level).mean(axis=0) * 100
        secs = (S >= level).sum(axis=0) / fps
        order = np.argsort(share)[::-1]
        hits = [i for i in order if share[i] > 0.05]

        print(f"\n=== {rec}   {len(S)/fps/60:.1f} phut, {fps} fps, "
              f"{len(hits)}/527 nhan vuot {level} tren >0.05% khung")
        print(f"    {'nhan':40}{'nhom':8}{'% khung':>9}{'giay':>9}{'p99':>7}{'max':>7}")
        for i in hits[:32]:
            print(f"    {labels[i][:39]:40}{bucket(labels[i]):8}"
                  f"{share[i]:>8.2f}%{secs[i]:>9.1f}{np.percentile(S[:, i], 99):>7.3f}"
                  f"{S[:, i].max():>7.3f}")

        # What a "keep only conversation" rule would remove.
        by = {}
        for i in hits:
            by.setdefault(bucket(labels[i]), []).append(i)
        keep = np.zeros(len(S), bool)
        for b in ("speech", "human"):
            for i in by.get(b, []):
                keep |= S[:, i] >= level
        drop = np.zeros(len(S), bool)
        for b in ("music", "other"):
            for i in by.get(b, []):
                drop |= S[:, i] >= level
        print(f"    --> khung co speech/human: {keep.mean()*100:.1f}%  |  "
              f"co music/other: {drop.mean()*100:.1f}%  |  "
              f"chi co music/other (khong speech): {(drop & ~keep).mean()*100:.1f}% "
              f"= {(drop & ~keep).sum()/fps:.0f}s")


if __name__ == "__main__":
    main()
