"""Forced-align the final transcript text to audio with Vietnamese Wav2Vec2.

ASR word times are tied to the text that Whisper originally decoded.  The
refinement pass can change that text, so carrying those old times forward would
silently attach words to the wrong spans.  This service deliberately runs after
refinement and speaker relabel and aligns ``TranscriptSegment.text`` again.

WhisperX supplies the CTC trellis implementation and selects a language-specific
Wav2Vec2 model.  For Vietnamese its default is
``nguyenvulebinh/wav2vec2-base-vi-vlsp2020``.  Models are loaded lazily so a
checkpointed file does not pay any model or VRAM cost.
"""

import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


ALIGNMENT_VERSION = "word-align-v1"
TARGET_SAMPLE_RATE = 16000
MAX_WORKER_CPU_THREADS = 8
_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "word_alignment_worker.py")


@dataclass
class WordAlignmentResult:
    words_by_index: Dict[str, List[dict]] = field(default_factory=dict)
    report: dict = field(default_factory=dict)


def apply_word_alignments(transcripts, words_by_index: Dict[str, List[dict]]) -> int:
    """Apply checkpointed rows and erase any stale pre-refinement ASR words."""
    changed = 0
    for segment in transcripts or ():
        segment.words = None
        words = words_by_index.get(str(segment.index))
        if words is None:
            continue
        segment.words = [dict(word) for word in words]
        changed += 1
    return changed


def align_prepared(prepared, model, metadata, device, interpolate_method) -> Dict[str, List[dict]]:
    """Align prepared independent clips packed into one carrier waveform.

    Shared by the in-process service and the worker processes so both produce
    identical rows.
    """
    import whisperx

    silence = np.zeros(int(0.05 * TARGET_SAMPLE_RATE), dtype=np.float32)
    pieces, requests, spans = [], [], []
    cursor = 0
    for item in prepared:
        start = cursor / TARGET_SAMPLE_RATE
        pieces.append(item["audio"])
        cursor += len(item["audio"])
        end = cursor / TARGET_SAMPLE_RATE
        requests.append({"start": start, "end": end, "text": item["text"]})
        spans.append((item, start, end))
        pieces.append(silence)
        cursor += len(silence)

    carrier = np.concatenate(pieces) if pieces else np.empty(0, np.float32)
    aligned = whisperx.align(
        requests, model, metadata, carrier, device,
        interpolate_method=interpolate_method,
        return_char_alignments=False, print_progress=False,
    )
    all_words = list((aligned or {}).get("word_segments", []))
    result: Dict[str, List[dict]] = {}

    for item, packed_start, packed_end in spans:
        words = []
        for raw in all_words:
            if "start" not in raw or "end" not in raw:
                continue
            word_start, word_end = float(raw["start"]), float(raw["end"])
            # WhisperX aligns every request inside its supplied bounds.  A
            # tiny epsilon keeps a word exactly on a packed boundary.
            if word_start < packed_start - 1e-4 or word_start > packed_end + 1e-4:
                continue
            absolute_start = float(item["start"]) + word_start - packed_start
            absolute_end = float(item["start"]) + word_end - packed_start
            absolute_start = min(max(absolute_start, float(item["start"])),
                                 float(item["end"]))
            absolute_end = min(max(absolute_end, absolute_start), float(item["end"]))
            row = {
                "word": str(raw.get("word", "")),
                "start": round(absolute_start, 3),
                "end": round(absolute_end, 3),
            }
            if raw.get("score") is not None:
                row["score"] = round(float(raw["score"]), 3)
            words.append(row)
        result[item["index"]] = words
    return result


class WordAlignmentService:
    """Align final text with WhisperX's language-specific Wav2Vec2 model."""

    def __init__(self, *, language: str = "vi", device: str = "cpu",
                 model_name: Optional[str] = None, model_dir: Optional[str] = None,
                 model_cache_only: bool = False, interpolate_method: str = "nearest",
                 batch_seconds: float = 180.0, workers_per_gpu: int = 0,
                 worker_gpus=None, worker_python: Optional[str] = None,
                 worker_script: Optional[str] = None, logger=None):
        if interpolate_method not in {"nearest", "linear", "ignore"}:
            raise ValueError("interpolate_method must be nearest, linear, or ignore")
        if float(batch_seconds) <= 0:
            raise ValueError("batch_seconds must be positive")
        if int(workers_per_gpu) < 0:
            raise ValueError("workers_per_gpu must be zero or positive")
        self.language = str(language)
        self.device = str(device)
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_cache_only = bool(model_cache_only)
        self.interpolate_method = interpolate_method
        self.batch_seconds = float(batch_seconds)
        # 0 aligns in this process; N > 0 starts N worker processes per GPU and
        # spreads a file's batches over them.
        self.workers_per_gpu = int(workers_per_gpu)
        self.worker_gpus = list(worker_gpus) if worker_gpus else None
        self._worker_python = worker_python or sys.executable
        self._worker_script = worker_script or _WORKER_SCRIPT
        self.logger = logger
        self._model = None
        self._metadata = None
        self._pool = None
        self._io_dir = None

    @property
    def checkpoint_namespace(self) -> str:
        settings = {
            "version": ALIGNMENT_VERSION,
            "language": self.language,
            "model": self.model_name or "whisperx-default",
            "interpolate": self.interpolate_method,
        }
        digest = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return f"{ALIGNMENT_VERSION}-{digest}"

    def checkpoint_namespace_for(self, transcripts) -> str:
        """Include final text and bounds so edited text never reuses stale words."""
        rows = [
            [str(segment.index), round(float(segment.start), 3),
             round(float(segment.end), 3), str(segment.text or "")]
            for segment in (transcripts or ())
        ]
        digest = hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
            .encode("utf-8")).hexdigest()[:12]
        return f"{self.checkpoint_namespace}-{digest}"

    def _worker_gpu_ids(self) -> list:
        """Physical GPUs to place workers on; None means the CPU."""
        if self.worker_gpus:
            return list(self.worker_gpus)
        device = self.device
        if device.startswith("cuda"):
            _, _, index = device.partition(":")
            return [int(index) if index.isdigit() else 0]
        return [None]

    def _ensure_pool(self):
        if self._pool is not None:
            return
        from utils.cpu_plan import usable_cores
        from services.word_alignment_worker_service import WordAlignmentWorkerService
        from services.worker_pool_service import WorkerPoolService

        gpus = self._worker_gpu_ids()
        total = len(gpus) * self.workers_per_gpu
        threads = max(1, min(MAX_WORKER_CPU_THREADS, usable_cores() // total))
        services = [
            WordAlignmentWorkerService(
                self._worker_python, self._worker_script, gpu,
                language=self.language, interpolate_method=self.interpolate_method,
                threads=threads, model_name=self.model_name, model_dir=self.model_dir,
                model_cache_only=self.model_cache_only, logger=self.logger)
            for gpu in gpus for _ in range(self.workers_per_gpu)]
        pool = WorkerPoolService(services, name="word-align")
        try:
            pool.spawn()
            pool.wait_ready()
        except Exception as exc:
            pool.stop()
            raise RuntimeError(
                f"Could not start the {self.language} alignment workers: {exc}") from exc
        self._pool = pool
        self._io_dir = tempfile.mkdtemp(prefix="wordalign_io_")
        if self.logger:
            self.logger.info(
                f"[word-align] {total} worker process(es) on GPU(s) {gpus}, "
                f"{threads} CPU thread(s) each")

    def _align_batch_remote(self, prepared) -> Dict[str, List[dict]]:
        """Send one batch to whichever worker is idle."""
        request_id = uuid.uuid4().hex
        path = os.path.join(self._io_dir, f"{request_id}.npy")
        np.save(path, np.concatenate([item["audio"] for item in prepared]))
        items = [{"index": item["index"], "start": item["start"], "end": item["end"],
                  "text": item["text"], "n": len(item["audio"])} for item in prepared]
        try:
            response = self._pool.request(
                {"id": request_id, "audio_path": path, "items": items},
                response_id=request_id)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response["words_by_index"]

    def _ensure_loaded(self):
        if self.workers_per_gpu > 0:
            self._ensure_pool()
            return
        if self._model is not None:
            return
        try:
            import whisperx
            self._model, self._metadata = whisperx.load_align_model(
                language_code=self.language,
                device=self.device,
                model_name=self.model_name,
                model_dir=self.model_dir,
                model_cache_only=self.model_cache_only,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load the {self.language} Wav2Vec2 alignment model: {exc}"
            ) from exc
        if self.logger:
            chosen = self.model_name or "WhisperX language default"
            self.logger.info(
                f"[word-align] loaded {chosen} for {self.language} on {self.device} "
                f"({self._environment_summary()})")

    def _environment_summary(self) -> str:
        """torch/GPU/dtype facts needed to diagnose kernel-availability errors."""
        try:
            import torch
            parts = [f"torch {torch.__version__}"]
            param = next(self._model.parameters(), None)
            if param is not None:
                parts.append(f"dtype {param.dtype}")
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                parts.append(torch.cuda.get_device_name(torch.device(self.device)))
            return ", ".join(parts)
        except Exception as exc:
            return f"environment unavailable: {exc}"

    @staticmethod
    def _source_audio(segment, audio, speech_by_index) -> np.ndarray:
        speech = speech_by_index.get(str(segment.index))
        separated = getattr(speech, "audio", None) if speech is not None else None
        if separated is not None and len(separated):
            return np.asarray(separated, dtype=np.float32)
        sr = int(audio.sample_rate)
        lo = max(0, int(round(float(segment.start) * sr)))
        hi = min(len(audio.waveform), int(round(float(segment.end) * sr)))
        return np.asarray(audio.waveform[lo:hi], dtype=np.float32)

    @staticmethod
    def _resample(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if int(sample_rate) == TARGET_SAMPLE_RATE:
            return np.ascontiguousarray(waveform, dtype=np.float32)
        import librosa
        return np.ascontiguousarray(
            librosa.resample(waveform, orig_sr=int(sample_rate),
                             target_sr=TARGET_SAMPLE_RATE),
            dtype=np.float32,
        )

    def _align_batch(self, prepared) -> Dict[str, List[dict]]:
        if self._pool is not None:
            return self._align_batch_remote(prepared)
        return align_prepared(prepared, self._model, self._metadata,
                              self.device, self.interpolate_method)

    def align(self, transcripts, audio, speech_segments=None) -> WordAlignmentResult:
        """Return final-text word timestamps in the pipeline's cut timeline."""
        transcripts = list(transcripts or ())
        speech_by_index = {
            str(segment.index): segment for segment in (speech_segments or ())
        }
        prepared, skipped = [], []
        source_rate = int(audio.sample_rate)
        for segment in transcripts:
            text = str(getattr(segment, "text", "") or "").strip()
            if not text:
                skipped.append(str(segment.index))
                continue
            source = self._source_audio(segment, audio, speech_by_index)
            if not len(source):
                skipped.append(str(segment.index))
                continue
            prepared.append({
                "index": str(segment.index),
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "audio": self._resample(source, source_rate),
            })

        if not prepared:
            raise RuntimeError("Word alignment has no non-empty transcript/audio segments")
        self._ensure_loaded()

        words_by_index: Dict[str, List[dict]] = {}
        failed = []
        batches, current, current_seconds = [], [], 0.0
        for item in prepared:
            seconds = len(item["audio"]) / TARGET_SAMPLE_RATE
            if current and current_seconds + seconds > self.batch_seconds:
                batches.append(current)
                current, current_seconds = [], 0.0
            current.append(item)
            current_seconds += seconds
        if current:
            batches.append(current)

        def align_isolated(batch):
            words, failures = {}, []
            try:
                words.update(self._align_batch(batch))
            except Exception as exc:
                # Isolate a bad segment so one malformed transcript does not
                # discard every other alignment in the carrier.
                if self.logger:
                    self.logger.warning(
                        f"[word-align] batch failed ({type(exc).__name__}: {exc}); "
                        "retrying segment by segment", exc_info=True)
                for item in batch:
                    try:
                        words.update(self._align_batch([item]))
                    except Exception as item_exc:
                        error = f"{type(item_exc).__name__}: {item_exc}"
                        failures.append({"index": item["index"], "error": error})
                        if self.logger:
                            self.logger.warning(
                                f"[word-align] segment {item['index']} failed ({error})",
                                exc_info=True)
            return words, failures

        if self._pool is not None and len(batches) > 1:
            # Threads only wait on worker pipes; the alignment runs in the workers.
            with ThreadPoolExecutor(max_workers=len(self._pool.services),
                                    thread_name_prefix="word-align") as executor:
                results = list(executor.map(align_isolated, batches))
        else:
            results = [align_isolated(batch) for batch in batches]
        for words, failures in results:
            words_by_index.update(words)
            failed.extend(failures)

        expected_words = sum(len(item["text"].split()) for item in prepared)
        aligned_words = sum(len(words) for words in words_by_index.values())
        complete = sum(
            len(words_by_index.get(item["index"], ())) == len(item["text"].split())
            and len(words_by_index.get(item["index"], ())) > 0
            for item in prepared
        )
        report = {
            "version": ALIGNMENT_VERSION,
            "language": self.language,
            "model": self.model_name or "whisperx-default",
            "device": self.device,
            "segments": len(transcripts),
            "segments_aligned": len(words_by_index),
            "segments_complete": complete,
            "segments_skipped": skipped,
            "segments_failed": failed,
            "expected_words": expected_words,
            "aligned_words": aligned_words,
            "word_coverage": round(aligned_words / expected_words, 4)
            if expected_words else 0.0,
        }
        if aligned_words == 0:
            first_error = f" (first error: {failed[0]['error']})" if failed else ""
            raise RuntimeError(
                "Wav2Vec2 returned no timed words for the final transcript"
                f"{first_error}")
        apply_word_alignments(transcripts, words_by_index)
        if self.logger:
            self.logger.info(
                f"[word-align] {aligned_words}/{expected_words} words timed; "
                f"{complete}/{len(prepared)} segments complete")
        return WordAlignmentResult(words_by_index=words_by_index, report=report)

    def unload(self):
        """Release the alignment model (and any worker processes) at the stage boundary."""
        pool, self._pool = self._pool, None
        io_dir, self._io_dir = self._io_dir, None
        if pool is not None:
            pool.stop()
        if io_dir:
            shutil.rmtree(io_dir, ignore_errors=True)
        self._model = None
        self._metadata = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
