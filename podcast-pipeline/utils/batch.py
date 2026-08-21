"""Grouping audio files into runs bounded by total duration."""

import os


def audio_duration(path: str) -> float:
    """Length of `path` in seconds, or 0.0 if it cannot be read.

    Reads the container header rather than decoding, so a five-hour batch is
    planned in milliseconds.
    """
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.frames and info.samplerate:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        # soundfile cannot read mp3 on every build; mutagen reads the header.
        from mutagen import File as MutagenFile
        m = MutagenFile(path)
        if m is not None and getattr(m, "info", None) is not None:
            return float(m.info.length)
    except Exception:
        pass
    try:
        from pydub.utils import mediainfo
        return float(mediainfo(path).get("duration") or 0.0)
    except Exception:
        return 0.0


def find_audio_files(directory: str, extensions) -> list:
    """Audio files directly under `directory`, sorted by name."""
    exts = tuple(e.lower() for e in extensions)
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(
        os.path.join(directory, n) for n in names
        if n.lower().endswith(exts) and os.path.isfile(os.path.join(directory, n))
    )


def plan_batches(paths, max_hours: float, logger=None):
    """Split `paths` into runs whose total duration stays under `max_hours`.

    Files keep their given order, so a run is a contiguous slice and the output
    layout does not depend on how the batches fall. A file longer than the limit
    becomes a batch of its own rather than being skipped -- refusing to process
    it would be a worse answer than exceeding the bound once, deliberately.

    Files whose duration cannot be read are placed in their own batch too: an
    unknown length cannot be added to a budget honestly.
    """
    limit = max(0.0, float(max_hours)) * 3600.0
    batches, current, current_total = [], [], 0.0

    for path in paths:
        dur = audio_duration(path)

        if dur <= 0.0:
            if logger:
                logger.warning(
                    f"Could not read duration of {os.path.basename(path)}; "
                    "running it on its own")
            if current:
                batches.append(current)
                current, current_total = [], 0.0
            batches.append([path])
            continue

        if limit > 0 and dur > limit:
            if logger:
                logger.warning(
                    f"{os.path.basename(path)} is {dur / 3600:.2f}h, longer than the "
                    f"{max_hours}h limit; running it on its own")
            if current:
                batches.append(current)
                current, current_total = [], 0.0
            batches.append([path])
            continue

        if limit > 0 and current and current_total + dur > limit:
            batches.append(current)
            current, current_total = [], 0.0

        current.append(path)
        current_total += dur

    if current:
        batches.append(current)
    return batches
