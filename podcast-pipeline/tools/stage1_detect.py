"""Step one of the stage-1 matrix: run each detector over each recording.

Writes, per (recording, detector): the framewise 527 scores, the music map it
produced, and one wav per span that needs separating. Separation runs in a
different interpreter -- audio-separator and the pipeline disagree about numpy
-- so the spans have to cross the boundary as files.
"""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import music_map as MM          # noqa: E402

SR = 24000                                  # what the pipeline itself loads at
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "stage1_matrix")
SOURCES = {
    "thu_that_thach_10m": "/Users/lam/Downloads/music_stage_audio (6)/thu_that_thach_10m/before_thu_that_thach_10m.mp3",
    "lm8": "/Users/lam/Downloads/music_stage_audio (6)/lm8-vongtaynang-reaction-131131/before_lm8-vongtaynang-reaction-131131.mp3",
    "vimeanh": "/Users/lam/Downloads/music_stage_audio (6)/vimeanhphanchiatay-145413/before_vimeanhphanchiatay-145413.mp3",
}


class _Precomputed:
    """Hands `build_maps` the sweep that already ran.

    `build_maps` calls the detector itself, so passing the real detector after
    `tag_framewise` would run the model a second time over the same audio --
    twice the wall clock and, on a half-hour file, twice the peak memory for
    numbers we already have.
    """

    def __init__(self, scores, fps):
        self._scores, self._fps = scores, fps

    def tag_framewise(self, waveform, sample_rate):
        return self._scores, self._fps


def detector(name):
    if name == "panns":
        from models.panns import PANNSDetector
        return PANNSDetector()
    from models.sslam import SSLAMDetector
    return SSLAMDetector()


def main():
    import gc

    import librosa
    os.makedirs(OUT, exist_ok=True)
    index_path = os.path.join(OUT, "index.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)

    for det_name in ("panns", "sslam"):
        det = None
        for rec, path in SOURCES.items():
            key = f"{rec}__{det_name}"
            if key in index:
                print(f"[{key}] da co, bo qua", flush=True)
                continue
            if det is None:
                det = detector(det_name)
            t0 = time.time()
            audio, _ = librosa.load(path, sr=SR, mono=True)
            duration = len(audio) / SR

            scores, fps = det.tag_framewise(audio, SR)
            elapsed = time.time() - t0

            found, _noise = MM.build_maps(audio, SR, _Precomputed(scores, fps))
            spans = [(float(a), float(b), k) for a, b, k in found.spans]

            span_dir = os.path.join(OUT, "spans", key)
            os.makedirs(span_dir, exist_ok=True)

            # Only MUSIC spans go to a separator. SINGING and SONG leave the
            # recording; handing them to a vocal model would ask it to invent
            # the voice it is supposed to be isolating.
            cut = []
            for i, (a, b, kind) in enumerate(spans):
                if kind != MM.MUSIC:
                    continue
                clip = audio[int(a * SR):int(b * SR)]
                if not len(clip):
                    continue
                name = f"{i:04d}_{a:.2f}_{b:.2f}.wav"
                sf.write(os.path.join(span_dir, name), clip, SR, subtype="PCM_16")
                cut.append({"i": i, "start": a, "end": b, "file": name})

            per_kind = {k: float(found.total_of(k))
                        for k in (MM.MUSIC, MM.SINGING, MM.SONG)}
            index[key] = {
                "recording": rec, "detector": det_name, "source": path,
                "duration": duration, "fps": float(fps),
                "detect_seconds": elapsed,
                "spans": spans, "separate": cut,
                "totals": per_kind,
                "excised": per_kind[MM.SINGING] + per_kind[MM.SONG],
                "curves": {k: [float(np.percentile(v, 50)),
                               float(np.percentile(v, 90)), float(v.max())]
                           for k, v in scores.items() if len(v)},
            }
            np.savez_compressed(os.path.join(OUT, f"{key}_scores.npz"),
                                **{k: v for k, v in scores.items()}, fps=fps)
            with open(index_path, "w") as f:
                json.dump(index, f, indent=1)
            del audio, scores
            gc.collect()
            print(f"[{key}] {duration/60:.1f}min in {elapsed:.0f}s | "
                  f"music {per_kind[MM.MUSIC]:.0f}s, sing {per_kind[MM.SINGING]:.0f}s, "
                  f"song {per_kind[MM.SONG]:.0f}s | {len(cut)} span(s) to separate",
                  flush=True)
        if det is not None and hasattr(det, "unload"):
            det.unload()
        det = None
        gc.collect()

    print("\nDETECT_DONE")


if __name__ == "__main__":
    main()
