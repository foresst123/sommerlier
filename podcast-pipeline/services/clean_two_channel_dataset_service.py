"""Optional export of clean, aligned two-speaker training items.

This output is deliberately separate from the pipeline's normal artifacts.
The dialogue finder contributes only boundaries and quality tiers; audio comes
from SeparationService's strict speaker tracks, never from a time-gated copy of
the original mixture.
"""

import json
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from utils.conversation_selection import (
    ConversationSelectionConfig, ConversationSelectionFinder, pick_non_overlapping)


SCHEMA_VERSION = 1
_REGISTRY_LOCK = threading.Lock()


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2,
                  default=_json_default)


class CleanTwoChannelDatasetService:
    """Write a separate numeric, tiered corpus when an output root is set."""

    def __init__(self, logger=None, workers: int = 1):
        self.logger = logger
        # Threads that cut and write items (numpy/soundfile release the GIL);
        # 1 is the sequential path. Item ids and output are the same at any width.
        self.workers = max(1, int(workers))

    @staticmethod
    def resolve_root(value: Any) -> str:
        """Expand a configured path; blank means completely disabled."""
        if value is None or not str(value).strip():
            return ""
        return os.path.abspath(os.path.expanduser(os.path.expandvars(str(value).strip())))

    @staticmethod
    def _source_facts(source_path: str) -> dict:
        path = os.path.abspath(os.path.expanduser(source_path))
        try:
            stat = os.stat(path)
            size, mtime_ns = stat.st_size, stat.st_mtime_ns
        except OSError:
            size, mtime_ns = None, None
        return {
            "original_path": path,
            "original_name": os.path.basename(path),
            "size_bytes": size,
            "mtime_ns": mtime_ns,
        }

    @staticmethod
    def _locked_registry(root: str, update):
        """Read/modify/write index.json under a process and OS file lock."""
        import fcntl

        index_path = os.path.join(root, "index.json")
        lock_path = os.path.join(root, ".index.lock")
        with _REGISTRY_LOCK, open(lock_path, "a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    with open(index_path, "r", encoding="utf-8") as handle:
                        registry = json.load(handle)
                except (FileNotFoundError, json.JSONDecodeError, TypeError):
                    registry = {"schema_version": SCHEMA_VERSION, "sources": {}}
                registry.setdefault("schema_version", SCHEMA_VERSION)
                registry.setdefault("sources", {})
                result = update(registry)
                temp = f"{index_path}.tmp-{os.getpid()}-{threading.get_ident()}"
                _write_json(temp, registry)
                os.replace(temp, index_path)
                return result
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _reserve_source(self, root: str, source_path: str) -> Tuple[str, dict]:
        facts = self._source_facts(source_path)

        def reserve(registry):
            sources = registry["sources"]
            for source_id, item in sources.items():
                if item.get("original_path") == facts["original_path"]:
                    item.update(facts)
                    item["status"] = "building"
                    return source_id
            used = [int(key) for key in sources if str(key).isdigit()]
            source_id = f"{max(used, default=0) + 1:06d}"
            sources[source_id] = {**facts, "folder": source_id,
                                  "status": "building"}
            return source_id

        return self._locked_registry(root, reserve), facts

    def _finish_source(self, root: str, source_id: str, summary: dict) -> None:
        def finish(registry):
            registry["sources"].setdefault(source_id, {}).update({
                "status": "complete",
                "item_count": summary["item_count"],
                "tiers": summary["tiers"],
            })

        self._locked_registry(root, finish)

    @staticmethod
    def _conversation(finder, candidate) -> List[dict]:
        rows = []
        names = {candidate.speakers[0]: "SP1", candidate.speakers[1]: "SP2"}
        for pos in range(candidate.first, candidate.last + 1):
            seg = finder.segs[pos]
            row = {
                "index": seg.index,
                "speaker": names.get(seg.speaker),
                "speaker_id": seg.speaker,
                "start": round(float(seg.start) - candidate.pad_start, 3),
                "end": round(float(seg.end) - candidate.pad_start, 3),
                "text": getattr(seg, "text", ""),
                "state": finder.state[pos],
            }
            if getattr(seg, "words", None):
                words = []
                for word in seg.words:
                    timed = dict(word)
                    if timed.get("start") is not None:
                        timed["start"] = round(
                            float(timed["start"]) - candidate.pad_start, 3)
                    if timed.get("end") is not None:
                        timed["end"] = round(
                            float(timed["end"]) - candidate.pad_start, 3)
                    words.append(timed)
                row["words"] = words
            rows.append(row)
        return rows

    @staticmethod
    def _failure_rows(speech_segments: Iterable, candidate) -> List[dict]:
        rows, seen = [], set()
        for seg in speech_segments:
            if seg.speaker not in candidate.speakers:
                continue
            for start, end, reason, detail in getattr(seg, "bss_failed_spans", ()):
                lo, hi = max(float(start), candidate.pad_start), min(float(end), candidate.pad_end)
                key = (seg.speaker, round(lo, 6), round(hi, 6), reason, detail)
                if hi <= lo or key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "speaker_id": seg.speaker,
                    "start": round(lo - candidate.pad_start, 3),
                    "end": round(hi - candidate.pad_start, 3),
                    "reason": reason,
                    "detail": detail,
                    "zeroed_in_strict_track": reason != "same_speaker",
                })
        return rows

    @staticmethod
    def _separated_rows(speech_segments: Iterable, candidate) -> List[dict]:
        rows, seen = [], set()
        for seg in speech_segments:
            if seg.speaker not in candidate.speakers:
                continue
            for start, end, similarity in getattr(seg, "bss_spans", ()):
                lo = max(float(start), candidate.pad_start)
                hi = min(float(end), candidate.pad_end)
                key = (seg.speaker, round(lo, 6), round(hi, 6),
                       round(float(similarity), 6))
                if hi <= lo or key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "speaker_id": seg.speaker,
                    "start": round(lo - candidate.pad_start, 3),
                    "end": round(hi - candidate.pad_start, 3),
                    "similarity": round(float(similarity), 6),
                })
        return rows

    @staticmethod
    def _replace_directory(staging: str, destination: str) -> None:
        backup = f"{destination}.old-{uuid.uuid4().hex}"
        had_old = os.path.exists(destination)
        if had_old:
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except Exception:
            if had_old and os.path.exists(backup):
                os.replace(backup, destination)
            raise
        if had_old:
            shutil.rmtree(backup, ignore_errors=True)

    def _export_item(self, finder, candidate, item_number, *, source_id, staging,
                     total_samples, audio_duration, sample_rate, speech_segments,
                     separation_service):
        """Write one item; returns (summary row, False) or (None, True) when a channel is empty."""
        import soundfile as sf

        pair = tuple(candidate.speakers)
        lo = max(0, round(candidate.pad_start * sample_rate))
        hi = min(total_samples, round(candidate.pad_end * sample_rate))
        if hi <= lo:
            return None, True
        left, right = separation_service.export_sdlm_dual_channel(
            speech_segments, audio_duration, sample_rate,
            strict=True, speakers=pair,
            time_range=(lo / sample_rate, hi / sample_rate),
            log_stats=False)
        sp1 = np.asarray(left, dtype=np.float32)
        sp2 = np.asarray(right, dtype=np.float32)
        if (not np.any(np.abs(sp1) > 1e-7)
                or not np.any(np.abs(sp2) > 1e-7)):
            return None, True

        item_id = f"{item_number:06d}"
        rel_dir = os.path.join(f"tier_{candidate.tier}", item_id)
        item_dir = os.path.join(staging, rel_dir)
        os.makedirs(item_dir, exist_ok=True)
        sf.write(os.path.join(item_dir, "sp1_clean.wav"), sp1,
                 sample_rate, subtype="PCM_16")
        sf.write(os.path.join(item_dir, "sp2_clean.wav"), sp2,
                 sample_rate, subtype="PCM_16")
        sf.write(os.path.join(item_dir, "audio_2ch.wav"),
                 np.column_stack((sp1, sp2)), sample_rate,
                 subtype="PCM_16")

        failures = self._failure_rows(speech_segments, candidate)
        separated = self._separated_rows(speech_segments, candidate)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "source_id": source_id,
            "item_id": item_id,
            "tier": candidate.tier,
            "score": candidate.score,
            "source_start": round(lo / sample_rate, 6),
            "source_end": round(hi / sample_rate, 6),
            "duration": round((hi - lo) / sample_rate, 6),
            "orig_spans": [
                {"start": round(a, 6), "end": round(b, 6)}
                for a, b in finder.timeline.spans_to_original(
                    lo / sample_rate, hi / sample_rate)
            ],
            "sample_rate": sample_rate,
            "num_samples": hi - lo,
            "speakers": {
                "SP1": pair[0], "SP2": pair[1],
                "left": "SP1", "right": "SP2",
            },
            "audio": {
                "sp1": "sp1_clean.wav",
                "sp2": "sp2_clean.wav",
                "stereo": "audio_2ch.wav",
                "method": "strict_separation_tracks",
            },
            "metrics": candidate.metrics,
            "components": candidate.components,
            "noise": candidate.noise,
            "music_patched_share": candidate.music_patched_share,
            "separated_spans": separated,
            "failed_separation_spans": failures,
            "conversation": self._conversation(finder, candidate),
        }
        _write_json(os.path.join(item_dir, "metadata.json"), metadata)
        return {
            "id": item_id, "tier": candidate.tier,
            "score": candidate.score,
            "folder": rel_dir,
            "speaker_ids": {"SP1": pair[0], "SP2": pair[1]},
            "duration": metadata["duration"],
        }, False

    def export(self, *, root: str, source_path: str, transcripts: list,
               speech_segments: list, separation_service, timeline, noise,
               music_map, sample_rate: int, audio_duration: float,
               selection_settings: Dict[str, Any]) -> dict:
        """Build one source folder. A blank root returns before any filesystem IO."""
        root = self.resolve_root(root)
        if not root:
            return {"enabled": False, "item_count": 0}

        os.makedirs(root, exist_ok=True)
        source_id, source_facts = self._reserve_source(root, source_path)
        staging = os.path.join(root, f".{source_id}.tmp-{uuid.uuid4().hex}")
        destination = os.path.join(root, source_id)
        os.makedirs(staging)

        try:
            cfg = ConversationSelectionConfig.from_settings(selection_settings)
            finder = ConversationSelectionFinder(transcripts, timeline, noise, music_map, cfg)
            candidates = pick_non_overlapping(finder.candidates())
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        items, skipped_silent = [], 0
        try:
            total_samples = round(audio_duration * sample_rate)

            def export_one(numbered):
                item_number, candidate = numbered
                return self._export_item(
                    finder, candidate, item_number, source_id=source_id, staging=staging,
                    total_samples=total_samples, audio_duration=audio_duration,
                    sample_rate=sample_rate, speech_segments=speech_segments,
                    separation_service=separation_service)

            numbered = list(enumerate(candidates, start=1))
            if self.workers > 1 and len(numbered) > 1:
                with ThreadPoolExecutor(max_workers=self.workers,
                                        thread_name_prefix="clean2ch") as pool:
                    outcomes = list(pool.map(export_one, numbered))
            else:
                outcomes = [export_one(entry) for entry in numbered]
            for row, silent in outcomes:
                if silent:
                    skipped_silent += 1
                else:
                    items.append(row)

            items.sort(key=lambda item: item["id"])
            tiers: Dict[str, int] = {}
            for item in items:
                tiers[item["tier"]] = tiers.get(item["tier"], 0) + 1
            source_summary = {
                "schema_version": SCHEMA_VERSION,
                "source_id": source_id,
                **source_facts,
                "audio_format": {
                    "sample_rate": sample_rate,
                    "subtype": "PCM_16",
                    "channels": {"left": "SP1", "right": "SP2"},
                },
                "selection": {
                    "candidate_count": len(candidates),
                    "item_count": len(items),
                    "skipped_empty_channel": skipped_silent,
                    "finder": finder.report(),
                },
                "tiers": tiers,
                "items": items,
            }
            _write_json(os.path.join(staging, "source.json"), source_summary)
            self._replace_directory(staging, destination)
            result = {
                "enabled": True,
                "root": root,
                "source_id": source_id,
                "source_dir": destination,
                "item_count": len(items),
                "tiers": tiers,
                "skipped_empty_channel": skipped_silent,
            }
            self._finish_source(root, source_id, result)
            if self.logger:
                self.logger.info(
                    f"[clean-2ch] {source_facts['original_name']}: "
                    f"{len(items)} item(s) -> {destination}")
            return result
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
