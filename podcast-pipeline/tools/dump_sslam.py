"""Every SSLAM score for every label on every frame, written per recording.

The grouped npz the matrix run produced answers "how much music" but not "what
else is in here" -- the six groups are a hand-written selection out of 527
labels, so anything nobody thought to list is invisible in them. Deciding what
counts as noise needs the other 521.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 24000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "stage1_matrix", "sslam_full")
SOURCES = {
    "thu_that_thach_10m": "/Users/lam/Downloads/music_stage_audio (6)/thu_that_thach_10m/before_thu_that_thach_10m.mp3",
    "lm8": "/Users/lam/Downloads/music_stage_audio (6)/lm8-vongtaynang-reaction-131131/before_lm8-vongtaynang-reaction-131131.mp3",
    "vimeanh": "/Users/lam/Downloads/music_stage_audio (6)/vimeanhphanchiatay-145413/before_vimeanhphanchiatay-145413.mp3",
}


def main():
    import gc

    import librosa
    from models.sslam import SSLAMDetector

    os.makedirs(OUT, exist_ok=True)
    det = SSLAMDetector()
    for rec, path in SOURCES.items():
        dest = os.path.join(OUT, f"{rec}.npz")
        if os.path.exists(dest):
            print(f"[{rec}] da co, bo qua", flush=True)
            continue
        t0 = time.time()
        audio, _ = librosa.load(path, sr=SR, mono=True)
        framewise, fps, scale = det.framewise_raw(audio, SR)
        np.savez_compressed(dest, scores=framewise, fps=fps, scale=scale,
                            labels=np.array(det.labels, dtype=object))
        print(f"[{rec}] {framewise.shape} @ {fps} fps, scale {scale:.3f}, "
              f"{time.time()-t0:.0f}s -> {os.path.getsize(dest)/1e6:.1f} MB",
              flush=True)
        del audio, framewise
        gc.collect()
    print("\nDUMP_DONE")


if __name__ == "__main__":
    main()
