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
The pipeline code and the progress ledger are not touched.

--include_done also checks files the ledger marks done (their audio, outputs and cache are
removed the same way). Those files usually have conversations in the final dataset: give
--dataset_root to remove them too (their item folders, their rows in manifest.jsonl, their
lhotse/<stem>.jsonl.gz and the swapped WAVs). Stop the pipeline first.
"""
import argparse
import hashlib
import json
import os
import re
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


def plan(folder, include_done=False):
    """(todo, skipped_done, skipped_rejected) as lists of paths."""
    done, rejected = read_ledger(folder)
    todo, skipped_done, skipped_rejected = [], [], []
    for path in list_audio(folder):
        name = os.path.basename(path)
        if name in done and not include_done:
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


DATASET_FOLDERS = ("interrupt", "backchannel", "turn_by_turn", "_quarantine",
                   "full_duplex", "half_duplex", "quarantine")
DATASET_MANIFESTS = ("manifest.jsonl", "manifest_quarantine.jsonl")


def safe_name(stem):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def _item_belongs(item_dir, name, stem):
    """Was this export item written from recording `stem`? (conversation.json metadata, else its name)"""
    try:
        with open(os.path.join(item_dir, "conversation.json"), encoding="utf-8") as handle:
            meta_stem = json.load(handle).get("source_stem")
    except (OSError, ValueError, AttributeError):
        meta_stem = None
    if meta_stem is not None:
        return meta_stem == stem
    return re.fullmatch(re.escape(safe_name(stem)) + r"_\d+s(_b)*", name) is not None


def dataset_targets(root, stem):
    """(item folders, extra files) the final dataset holds for recording `stem`."""
    items, files = [], []
    for folder in DATASET_FOLDERS:
        base = os.path.join(root, folder)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            item_dir = os.path.join(base, name)
            if os.path.isdir(item_dir) and _item_belongs(item_dir, name, stem):
                items.append(item_dir)
                tag = hashlib.sha1(os.path.abspath(item_dir).encode("utf-8")).hexdigest()[:8]
                swapped = os.path.join(root, "lhotse", "audio", f"{name}_{tag}.wav")
                if os.path.isfile(swapped):
                    files.append(swapped)
    manifest = os.path.join(root, "lhotse", f"{safe_name(stem)}.jsonl.gz")
    if os.path.isfile(manifest):
        files.append(manifest)
    return items, files


def _drop_manifest_rows(root, stem):
    """Remove this recording's rows from manifest.jsonl / manifest_quarantine.jsonl; returns how many."""
    dropped = 0
    lock_path = os.path.join(root, ".manifest.lock")
    try:
        import fcntl
        lock = open(lock_path, "a+", encoding="utf-8")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    except (OSError, ImportError):
        lock = None
    try:
        for name in DATASET_MANIFESTS:
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            kept = []
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        row_stem = json.loads(line).get("source_stem")
                    except (ValueError, AttributeError):
                        row_stem = None
                    if row_stem == stem:
                        dropped += 1
                    else:
                        kept.append(line)
            tmp = f"{path}.tmp-{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as handle:
                for line in kept:
                    handle.write(line + "\n")
            os.replace(tmp, path)
    finally:
        if lock is not None:
            lock.close()
    return dropped


def purge_dataset(root, stem):
    """Delete recording `stem`'s items, swapped WAVs, lhotse manifest and manifest rows."""
    items, files = dataset_targets(root, stem)
    for item in items:
        shutil.rmtree(item, ignore_errors=True)
    for path in files:
        try:
            os.remove(path)
        except OSError:
            pass
    return len(items), _drop_manifest_rows(root, stem)


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
    todo, skipped_done, skipped_rejected = plan(folder, include_done=args.include_done)
    done_names = read_ledger(folder)[0]
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
                stem = stem_of(path)
                targets = outputs_of(folder, stem, args.cache_dir, args.job_id)
                items, ds_files = dataset_targets(args.dataset_root, stem) if args.dataset_root else ([], [])
                size = (os.path.getsize(path) + sum(dir_size(t) for t in targets)
                        + sum(dir_size(i) for i in items) + sum(os.path.getsize(f) for f in ds_files))
                action = "dry-run" if not args.apply else "deleted"
                if args.apply:
                    for target in targets:
                        shutil.rmtree(target, ignore_errors=True)
                    if args.dataset_root:
                        purge_dataset(args.dataset_root, stem)
                    if args.trash:
                        os.makedirs(args.trash, exist_ok=True)
                        shutil.move(path, os.path.join(args.trash, os.path.basename(path)))
                        action = "moved"
                    else:
                        os.remove(path)
                    freed += size
                flagged_rows.append((path, mean_vi, top, targets))
            shown = "n/a" if mean_vi is None else f"{mean_vi:.2f}"
            note = ""
            if flagged and os.path.basename(path) in done_names:
                note = f" [done; {len(items)} conversation trong dataset"
                note += "]" if args.dataset_root else ", thêm --dataset_root để xóa]"
            elif flagged and args.dataset_root and items:
                note = f" [{len(items)} conversation trong dataset]"
            print(f"[{i}/{len(todo)}] P(vi)={shown} top={top} -> {action}: {os.path.basename(path)}{note}")
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
    ap.add_argument("--include_done", action="store_true",
                    help="also check files the ledger marks done (default: skip them)")
    ap.add_argument("--dataset_root", default=None,
                    help="final dataset root (holds interrupt/, backchannel/, turn_by_turn/, lhotse/): "
                         "also delete the dropped recording's conversations there")
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
