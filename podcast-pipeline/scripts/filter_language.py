"""Drop non-Vietnamese recordings from an input folder before the pipeline spends time on them.

Run (from podcast-pipeline/), dry run first:
  python scripts/filter_language.py /home/lamkd2/data/data_input/data_run_2
  python scripts/filter_language.py /home/lamkd2/data/data_input/data_run_2 --apply

Only the audio files directly under the folder are looked at. A file the progress ledger
marks done or rejected is left alone, so nothing finished is ever redone. For every other
file a small Whisper model guesses the language from a few 30 s windows; when the mean
probability of Vietnamese is under --min_vi the file is dropped:
  * the audio file is deleted (or moved with --trash DIR),
  * <folder>/_final/<params>/<stem>/ (the per-file outputs) is deleted,
  * <cache_dir>/<job_id>_<stem>/ (its checkpoints) is deleted.
The pipeline code, the progress ledger and the final dataset root are not touched.
"""
import argparse
import json
import os
import shutil
import sys
import time

AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".opus", ".ogg", ".aac")
LEDGER_NAME = "_sommelier_progress.json"
WINDOW_SECONDS = 30.0
FRACTIONS = (0.10, 0.35, 0.60, 0.85)
SAMPLE_RATE = 16000


def list_audio(folder):
    """Audio files directly under `folder` (no recursion), sorted."""
    return sorted(
        os.path.join(folder, name) for name in os.listdir(folder)
        if name.lower().endswith(AUDIO_EXT) and os.path.isfile(os.path.join(folder, name)))


def read_ledger(folder):
    """(done, rejected) as sets of file names; both empty when the ledger is missing or unreadable."""
    try:
        with open(os.path.join(folder, LEDGER_NAME), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return set(), set()
    return set(data.get("done", {}) or {}), set(data.get("rejected", {}) or {})


def plan(folder):
    """(todo, skipped_done, skipped_rejected) as lists of paths."""
    done, rejected = read_ledger(folder)
    todo, skipped_done, skipped_rejected = [], [], []
    for path in list_audio(folder):
        name = os.path.basename(path)
        if name in done:
            skipped_done.append(path)
        elif name in rejected:
            skipped_rejected.append(path)
        else:
            todo.append(path)
    return todo, skipped_done, skipped_rejected


def stem_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def outputs_of(folder, stem, cache_dir, job_id):
    """Directories holding this recording's pipeline outputs and checkpoints (exact names, no globbing)."""
    found = []
    final = os.path.join(folder, "_final")
    if os.path.isdir(final):
        for entry in sorted(os.scandir(final), key=lambda e: e.name):
            candidate = os.path.join(entry.path, stem)
            if entry.is_dir() and os.path.isdir(candidate):
                found.append(candidate)
    cache = os.path.join(cache_dir, f"{job_id}_{stem}")
    if os.path.isdir(cache):
        found.append(cache)
    return found


def dir_size(path):
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def verdict(window_probs, min_vi):
    """(mean P(vi), most common top language, flagged). `window_probs` is a list of {lang: prob}."""
    if not window_probs:
        return None, None, False                      # nothing readable: never drop on no evidence
    mean_vi = sum(p.get("vi", 0.0) for p in window_probs) / len(window_probs)
    tops = [max(p, key=p.get) for p in window_probs]
    top = max(set(tops), key=tops.count)
    return mean_vi, top, mean_vi < min_vi


def sample_windows(path, fractions=FRACTIONS, window=WINDOW_SECONDS, sr=SAMPLE_RATE):
    """Mono float32 windows at `sr`, spread across the file. Unreadable windows are skipped."""
    import numpy as np
    import librosa
    try:
        import soundfile as sf
        info = sf.info(path)
        duration, file_sr, use_sf = float(info.duration), int(info.samplerate), True
    except Exception:
        duration, file_sr, use_sf = float(librosa.get_duration(path=path)), sr, False
    out = []
    for fraction in fractions:
        length = min(window, duration)
        start = max(0.0, min(duration - length, duration * fraction - length / 2))
        try:
            if use_sf:
                data, _ = sf.read(path, start=int(start * file_sr), frames=int(length * file_sr),
                                  dtype="float32", always_2d=True)
                audio = data.mean(axis=1)
                if file_sr != sr:
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
            else:
                audio, _ = librosa.load(path, sr=sr, offset=start, duration=length, mono=True)
        except Exception:
            continue
        if audio.size and float(np.sqrt(np.mean(audio ** 2))) > 1e-4:   # skip silent windows
            out.append(audio.astype("float32"))
    return out


class WhisperLanguageId:
    """Language probabilities from a small multilingual Whisper (faster-whisper)."""

    def __init__(self, model="base", device="cpu", gpu_index=0, model_dir=None, offline=False):
        from faster_whisper import WhisperModel
        kwargs = {"device": device, "compute_type": "int8" if device == "cpu" else "float16"}
        if device == "cuda":
            kwargs["device_index"] = gpu_index
        if model_dir:
            kwargs["download_root"] = model_dir
        if offline:
            kwargs["local_files_only"] = True
        self.model = WhisperModel(model, **kwargs)

    def __call__(self, path):
        probs = []
        for audio in sample_windows(path):
            _, _, all_probs = self.model.detect_language(audio)
            probs.append(dict(all_probs))
        return probs


def run(args, detector=None):
    folder = os.path.abspath(args.folder)
    todo, skipped_done, skipped_rejected = plan(folder)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{folder}: {len(todo)} file cần kiểm tra | bỏ qua {len(skipped_done)} đã done, "
          f"{len(skipped_rejected)} đã rejected")
    if not todo:
        return []
    if detector is None:
        detector = WhisperLanguageId(args.model, args.device, args.gpu_index, args.model_dir, args.offline)

    log_path = args.log or f"language_filter_{time.strftime('%Y%m%d_%H%M%S')}.tsv"
    flagged_rows, freed = [], 0
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("file\tmean_p_vi\ttop_language\tflagged\taction\n")
        for i, path in enumerate(todo, start=1):
            try:
                mean_vi, top, flagged = verdict(detector(path), args.min_vi)
            except Exception as exc:                       # never drop a file on an error
                print(f"[{i}/{len(todo)}] LỖI nhận diện, giữ lại: {os.path.basename(path)}: {exc}")
                log.write(f"{os.path.basename(path)}\t\t\t0\terror\n")
                continue
            action = "keep"
            if flagged:
                targets = outputs_of(folder, stem_of(path), args.cache_dir, args.job_id)
                size = os.path.getsize(path) + sum(dir_size(t) for t in targets)
                action = "dry-run" if not args.apply else "deleted"
                if args.apply:
                    for target in targets:
                        shutil.rmtree(target, ignore_errors=True)
                    if args.trash:
                        os.makedirs(args.trash, exist_ok=True)
                        shutil.move(path, os.path.join(args.trash, os.path.basename(path)))
                        action = "moved"
                    else:
                        os.remove(path)
                    freed += size
                flagged_rows.append((path, mean_vi, top, targets))
            shown = "n/a" if mean_vi is None else f"{mean_vi:.2f}"
            print(f"[{i}/{len(todo)}] P(vi)={shown} top={top} -> {action}: {os.path.basename(path)}")
            log.write(f"{os.path.basename(path)}\t{shown}\t{top}\t{int(flagged)}\t{action}\n")

    print(f"\n{len(flagged_rows)} file không phải tiếng Việt "
          f"({'đã xử lý, giải phóng ' + format(freed / 1e9, '.1f') + ' GB' if args.apply else 'CHƯA xóa, thêm --apply để xóa'})")
    print("log:", log_path)
    return flagged_rows


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="input folder; only the audio files directly under it are checked")
    ap.add_argument("--apply", action="store_true", help="delete for real (default is a dry run)")
    ap.add_argument("--trash", default=None, help="move the audio here instead of deleting it")
    ap.add_argument("--min_vi", type=float, default=0.5, help="drop when mean P(vi) is below this")
    ap.add_argument("--cache_dir", default="/home/lamkd2/sommerlier/podcast-pipeline/cache")
    ap.add_argument("--job_id", default="default", help="checkpoint folders are <job_id>_<file stem>")
    ap.add_argument("--model", default="base", help="multilingual Whisper size: tiny, base, small")
    ap.add_argument("--model_dir", default=None, help="folder with (or to download) the model files")
    ap.add_argument("--offline", action="store_true", help="never download the model")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu_index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="check only the first N files (0 = all)")
    ap.add_argument("--log", default=None, help="TSV log path")
    return ap


def main(argv=None):
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
