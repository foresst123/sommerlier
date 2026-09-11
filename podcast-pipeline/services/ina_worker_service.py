"""Second-pass music/noise proposals from an isolated INA segmenter."""

from pathlib import Path
import tempfile

import numpy as np

from services.json_worker_service import JsonWorkerService
from utils.music_map import MusicMap, MUSIC
from utils.noise_map import NoiseTrack
from utils.worker_env import resolve_worker_python


def merge_scan(music_map, noise_track, segments, duration):
    additions = []
    for kind, lo, hi in segments:
        lo, hi = max(0.0, lo), min(duration, hi)
        if kind == "music":
            # INA cannot distinguish speech over a music bed. New proposals
            # request bed separation, never removal of possible dialogue.
            additions.extend((a, b, MUSIC) for a, b in music_map.clean_parts(lo, hi))
    fps = noise_track.fps if noise_track else 2.0
    count = int(np.ceil(duration * fps))
    curves = {k: np.pad(v[:count], (0, max(0, count - len(v))))
              for k, v in (noise_track or NoiseTrack()).curves.items()}
    before = np.maximum.reduce(list(curves.values()))
    for kind, lo, hi in segments:
        if kind == "noise":
            a, b = max(0, int(lo * fps)), min(count, int(np.ceil(hi * fps)))
            curves["noise_env"][a:b] = 1.0
    merged_noise = NoiseTrack(curves, fps)
    merged_music = MusicMap(music_map.spans + additions, music_map.fps)
    # Compare coverage, not summed spans (padding can overlap).
    from utils.excise import _merge
    added_music = sum(b - a for a, b in _merge([(a, b) for a, b, _ in additions]))
    diagnostic = {"added_music_seconds": added_music,
                  "added_noise_seconds": float(np.sum((before < 0.1) & (merged_noise.combined >= 0.1)) / fps),
                  "segments": segments}
    return merged_music, merged_noise, diagnostic


def scan_and_merge(audio, music_map, noise_track, config, args, logger=None):
    import soundfile as sf
    profile = config.get("environments", {}).get(args.env, {})
    worker = JsonWorkerService("INA", resolve_worker_python("ina", config, profile),
        str(Path(__file__).resolve().parents[1] / "ina_worker.py"),
        device_id=getattr(args, "gpu_1", 0), logger=logger)
    try:
        with tempfile.TemporaryDirectory(prefix="sommelier-ina-") as directory:
            path = str(Path(directory) / "input.wav")
            sf.write(path, audio.waveform, audio.sample_rate, subtype="FLOAT")
            response = worker.request({"audio_path": path}, timeout=3600)
        result = merge_scan(music_map, noise_track or NoiseTrack(), response["segments"], audio.duration)
        if logger:
            logger.info(f"[ina:added] music={result[2]['added_music_seconds']:.2f}s "
                        f"noise={result[2]['added_noise_seconds']:.2f}s")
        return result
    finally:
        worker.stop()
