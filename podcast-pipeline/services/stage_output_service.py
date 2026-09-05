"""Per-stage artifacts, written as each stage finishes rather than at the end.

The pipeline used to write everything in its final export block, so stopping
early -- deliberately with --stop_after, or because a later stage crashed --
left nothing on disk for the work that had already succeeded. Forty minutes of
diarization could end with no file to show for it.

Each stage now closes with its own directory holding three kinds of artifact:

  segments.json / transcripts.json   the data the stage produced
  stats.json                         measurements, plus warnings when a
                                     measurement looks wrong
  audio/                             clips, where the stage produces any

stats.json is the part that catches mistakes. A run of this pipeline had
diarization emitting 0.98% overlapped speech -- five to fifteen times below
what two people in conversation produce -- and nothing said so; it took reading
the JSON by hand, much later, to notice. Anything cheap enough to measure and
specific enough to be wrong is measured here and flagged at the time.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


# Two people talking over each other in a podcast normally lands here. Well
# outside it means the diarizer is missing overlap or inventing it.
OVERLAP_PCT_MIN = 3.0
OVERLAP_PCT_MAX = 25.0
# Below this, one speaker is being absorbed into the other's turns.
SPEAKER_BALANCE_MIN = 10.0
# Coverage is measured as unlabelled *long* gaps, not as total silence. A
# conversation is mostly short pauses between phrases, and a small merge_gap
# leaves every one of them unlabelled: a real run at merge_gap=0.3 reported
# 79.1% coverage, of which 184 of 196 gaps were under a second -- ordinary
# breathing room, not lost speech. Only a gap long enough to hold a sentence
# suggests the diarizer actually dropped something.
LONG_GAP_SEC = 2.0
LONG_GAP_PCT_MAX = 10.0

# Refinement legitimately shortens text -- it deletes ASR stutters and
# hallucinated boilerplate. Losing this much of what it touched is past that.
WORD_DROP_PCT_MAX = 15.0


def _gap_profile(merged, span) -> dict:
    """Unlabelled stretches, split by whether they could hold speech.

    Short gaps are what a conversation is made of -- the pause between phrases,
    a breath -- and they multiply as merge_gap shrinks, so their total says
    nothing about whether the diarizer missed anything. Gaps long enough to
    hold a sentence do.
    """
    if not merged or not span:
        return {"long_count": 0, "long_seconds": 0.0, "long_pct": 0.0,
                "short_count": 0, "longest": 0.0}

    gaps = []
    if merged[0][0] > 0.1:
        gaps.append(merged[0][0])
    for a, b in zip(merged, merged[1:]):
        if b[0] - a[1] > 0.1:
            gaps.append(b[0] - a[1])
    if span - merged[-1][1] > 0.1:
        gaps.append(span - merged[-1][1])

    long_gaps = [g for g in gaps if g >= LONG_GAP_SEC]
    return {
        "long_count": len(long_gaps),
        "long_seconds": round(sum(long_gaps), 1),
        "long_pct": round(100.0 * sum(long_gaps) / span, 1),
        "short_count": len(gaps) - len(long_gaps),
        "longest": round(max(gaps), 1) if gaps else 0.0,
    }


def _as_dict(obj) -> dict:
    """Plain dict for one segment/transcript, without any waveform payload."""
    d = dict(obj.__dict__) if hasattr(obj, "__dict__") else dict(obj)
    d.pop("enhanced_audio", None)
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray):
            d[k] = {"_ndarray": True, "shape": list(v.shape), "dtype": str(v.dtype)}
        elif isinstance(v, (np.floating, np.integer)):
            d[k] = v.item()
    return d


class StageOutputService:
    """Writes one directory per pipeline stage."""

    # Numbered by the order they run, which changed when music analysis moved
    # ahead of diarization: the diarizer now segments audio whose music bed has
    # already been stripped, rather than seeing it and being told about it
    # afterwards.
    STAGES = {
        "music": "01_music",
        "diarization": "02_diarization",
        "separation": "03_separation",
        "music_removal": "04_music_removal",
        "asr": "05_asr",
        "refinement": "06_refinement",
    }

    def __init__(self, output_dir: str, logger=None, enabled: bool = True):
        self.output_dir = output_dir
        self.logger = logger
        self.enabled = enabled
        self.manifest: Dict[str, Any] = {"stages": {}}
        self._warned_sf = False

    # -- paths ---------------------------------------------------------
    def stage_dir(self, stage: str, *sub) -> str:
        d = os.path.join(self.output_dir, self.STAGES.get(stage, stage), *sub)
        os.makedirs(d, exist_ok=True)
        return d

    def _write_json(self, path: str, payload) -> bool:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"[stage-out] failed writing {path}: {e}")
            return False

    # -- measurements --------------------------------------------------
    @staticmethod
    def segment_stats(segs: List[Any], total_dur: Optional[float] = None) -> dict:
        """Counts, duration spread, speaker balance, coverage and overlap.

        Overlap is measured between different speakers only: two segments of the
        same speaker touching is a merge artifact, not simultaneous speech.
        """
        items = [(_as_dict(s)) for s in segs]
        if not items:
            return {"n_segments": 0, "warnings": ["stage produced no segments"]}

        durs = sorted(float(d["end"]) - float(d["start"]) for d in items)
        n = len(durs)
        spans = sorted((float(d["start"]), float(d["end"])) for d in items)

        # Union of all labelled time, for coverage.
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        labelled = sum(b - a for a, b in merged)

        # Cross-speaker overlap.
        ordered = sorted(items, key=lambda d: float(d["start"]))
        overlap = 0.0
        for i, s in enumerate(ordered):
            for o in ordered[i + 1:]:
                if float(o["start"]) >= float(s["end"]):
                    break
                if o.get("speaker") != s.get("speaker"):
                    overlap += min(float(s["end"]), float(o["end"])) - float(o["start"])

        by_spk: Dict[str, float] = {}
        for d in items:
            by_spk[str(d.get("speaker"))] = by_spk.get(str(d.get("speaker")), 0.0) + \
                (float(d["end"]) - float(d["start"]))
        spoken = sum(by_spk.values()) or 1.0

        span = total_dur or (merged[-1][1] if merged else 0.0)
        stats = {
            "n_segments": n,
            "n_speakers": len(by_spk),
            "duration": {
                "min": round(durs[0], 3), "p50": round(durs[n // 2], 3),
                "max": round(durs[-1], 3), "total": round(sum(durs), 2),
            },
            "short_segments": {
                "under_0.5s": sum(d < 0.5 for d in durs),
                "under_1.0s": sum(d < 1.0 for d in durs),
            },
            "speaker_share_pct": {k: round(100.0 * v / spoken, 1)
                                  for k, v in sorted(by_spk.items())},
            "overlap": {"seconds": round(overlap, 2),
                        "pct_of_audio": round(100.0 * overlap / span, 2) if span else 0.0},
            "coverage_pct": round(100.0 * labelled / span, 1) if span else None,
            "unlabelled_gaps": _gap_profile(merged, span),
        }

        w = []
        ov = stats["overlap"]["pct_of_audio"]
        if ov < OVERLAP_PCT_MIN:
            w.append(
                f"overlap {ov:.2f}% is below {OVERLAP_PCT_MIN}%: the diarizer is "
                "probably missing backchannels, which get buried inside the other "
                "speaker's turn and then break separation")
        elif ov > OVERLAP_PCT_MAX:
            w.append(f"overlap {ov:.2f}% is above {OVERLAP_PCT_MAX}%: likely spurious")
        if len(by_spk) < 2:
            w.append(f"only {len(by_spk)} speaker labelled")
        else:
            lo = min(stats["speaker_share_pct"].values())
            if lo < SPEAKER_BALANCE_MIN:
                w.append(f"speaker share is {lo}%: one speaker may be absorbed into the other")
        gaps = stats["unlabelled_gaps"]
        if gaps["long_pct"] > LONG_GAP_PCT_MAX:
            w.append(
                f"{gaps['long_pct']}% of the audio sits in unlabelled gaps longer "
                f"than {LONG_GAP_SEC}s ({gaps['long_count']} of them, worst "
                f"{gaps['longest']}s): speech the diarizer did not label")
        if w:
            stats["warnings"] = w
        return stats

    @staticmethod
    def separation_stats(segs: List[Any]) -> dict:
        items = [_as_dict(s) for s in segs]
        sep = [d for d in items if d.get("tse")]
        spliced = sum(len(d.get("tse_spans") or []) for d in items)
        failed = sum(len(d.get("tse_failed_spans") or []) for d in items)
        sims = [sp[2] for d in items for sp in (d.get("tse_spans") or [])
                if len(sp) > 2 and sp[2] is not None and sp[2] >= 0]

        reasons: Dict[str, int] = {}
        for d in items:
            for fs in (d.get("tse_failed_spans") or []):
                if len(fs) > 2:
                    reasons[str(fs[2])] = reasons.get(str(fs[2]), 0) + 1

        total = spliced + failed
        stats = {
            "segments_total": len(items),
            "segments_separated": len(sep),
            "spans_spliced": spliced,
            "spans_failed": failed,
            "failure_pct": round(100.0 * failed / total, 1) if total else 0.0,
            "failure_reasons": reasons,
            "similarity": {
                "p10": round(float(np.percentile(sims, 10)), 3),
                "p50": round(float(np.percentile(sims, 50)), 3),
                "p90": round(float(np.percentile(sims, 90)), 3),
            } if sims else None,
        }
        w = []
        if total and stats["failure_pct"] > 20.0:
            w.append(f"{stats['failure_pct']}% of overlapping spans could not be separated")
        if reasons.get("empty_track"):
            w.append(f"{reasons['empty_track']} span(s) came back silent where the "
                     "mixture has speech -- the separator emitted one source and silence")
        if not sep and items:
            w.append("no segment was separated at all")
        if w:
            stats["warnings"] = w
        return stats

    @staticmethod
    def transcript_stats(items: List[Any]) -> dict:
        rows = [_as_dict(t) for t in items]
        texts = [(r.get("text") or "").strip() for r in rows]
        empty = sum(1 for t in texts if not t)
        words = sum(len(t.split()) for t in texts)
        stats = {
            "n_transcripts": len(rows),
            "empty_text": empty,
            "total_words": words,
            "avg_words_per_segment": round(words / len(rows), 2) if rows else 0.0,
        }
        w = []
        if rows and empty / len(rows) > 0.15:
            w.append(f"{empty}/{len(rows)} segments have no text")
        if w:
            stats["warnings"] = w
        return stats

    # -- writers -------------------------------------------------------
    def _finish(self, stage: str, payload_name: str, rows: List[Any], stats: dict,
                extra: Optional[dict] = None) -> Optional[str]:
        if not self.enabled:
            return None
        d = self.stage_dir(stage)
        self._write_json(os.path.join(d, payload_name), [_as_dict(r) for r in rows])
        self._write_json(os.path.join(d, "stats.json"), stats)
        if extra:
            for name, obj in extra.items():
                self._write_json(os.path.join(d, name), obj)

        entry = {"dir": self.STAGES.get(stage, stage), "stats": stats}
        self.manifest["stages"][stage] = entry
        if self.logger:
            head = ", ".join(f"{k}={v}" for k, v in list(stats.items())[:3]
                             if not isinstance(v, (dict, list)))
            self.logger.info(f"[stage-out] {stage}: {head} -> {d}")
            for msg in stats.get("warnings", []):
                self.logger.warning(f"[stage-out] {stage}: {msg}")
        return d

    def write_music(self, music_map, timeline=None, audio=None, sample_rate=None):
        """What the tagger found, and what was done about it.

        Written before diarization runs, so a `--stop_after music` run leaves
        the whole verdict on disk: which stretches were called singing, which
        were beds, how much left the recording, and the audio that remains.
        """
        if not self.enabled:
            return
        stage = self.stage_dir("music")

        payload = dict(music_map.summary())
        payload["spans"] = [{"start": round(a, 3), "end": round(b, 3), "kind": k}
                            for a, b, k in music_map.spans]
        if timeline is not None:
            payload["removed_seconds"] = round(timeline.removed, 2)
            payload["kept_stretches"] = len(timeline.kept)
        self._write_json(os.path.join(stage, "music_map.json"), payload)

        # The trimmed recording itself, so the cuts and their joins can be
        # judged by ear rather than from counters.
        if audio is not None and sample_rate:
            try:
                import soundfile as sf
                sf.write(os.path.join(stage, "after_music.wav"), audio, sample_rate)
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"Could not write after_music.wav: {exc}")

    def write_diarization(self, segments, total_dur=None):
        return self._finish("diarization", "segments.json", segments,
                            self.segment_stats(segments, total_dur))

    def write_separation(self, segments, total_dur=None, report=None):
        stats = self.separation_stats(segments)
        stats["segments"] = self.segment_stats(segments, total_dur)
        return self._finish("separation", "segments.json", segments, stats,
                            extra={"report.json": report} if report else None)

    def write_music_removal(self, segments, total_dur=None):
        items = [_as_dict(s) for s in segments]
        stats = {"segments_total": len(items),
                 "segments_demucs": sum(1 for d in items if d.get("demucs"))}
        stats["segments"] = self.segment_stats(segments, total_dur)
        return self._finish("music_removal", "segments.json", segments, stats)

    def write_asr(self, transcripts):
        return self._finish("asr", "transcripts.json", transcripts,
                            self.transcript_stats(transcripts))

    def write_refinement(self, transcripts, before=None):
        stats = self.transcript_stats(transcripts)
        extra = None
        if before is not None:
            b = {str(_as_dict(t).get("index")): (_as_dict(t).get("text") or "") for t in before}
            changes = []
            for t in transcripts:
                d = _as_dict(t)
                old = b.get(str(d.get("index")))
                new = d.get("text") or ""
                if old is not None and old != new:
                    changes.append({"index": d.get("index"), "before": old, "after": new})
            stats["segments_changed"] = len(changes)

            # Counting rewritten segments does not separate repair from damage.
            # Vietnamese ASR output arrives with no punctuation and frequent
            # stutters, so a good pass touches most of it: a real run rewrote
            # 171 of 200, and inspection showed 42% punctuation only, 18%
            # removing ASR repetitions, and the rest genuine corrections
            # ("miệt danh" -> "biệt danh"). What would signal damage is words
            # disappearing in bulk, so that is what is measured.
            words_before = sum(len(c["before"].split()) for c in changes)
            words_after = sum(len(c["after"].split()) for c in changes)
            drop_pct = (100.0 * (words_before - words_after) / words_before
                        if words_before else 0.0)
            stats["word_drop_pct"] = round(drop_pct, 1)
            if drop_pct > WORD_DROP_PCT_MAX:
                stats.setdefault("warnings", []).append(
                    f"refinement removed {drop_pct:.1f}% of the words it touched "
                    f"({words_before - words_after} of {words_before}): check "
                    "06_refinement/changes.json for deleted content")
            extra = {"changes.json": changes}
        return self._finish("refinement", "transcripts.json", transcripts, stats, extra)

    # -- separated audio ------------------------------------------------
    def write_separated_audio(self, segments, sample_rate: int) -> dict:
        """Clips for spans TSE actually touched, split by outcome.

        Only segments carrying separated audio are written. Exporting every
        segment put 106 untouched mixture clips in a directory named
        "separation" next to 57 real ones, with nothing in the filename to tell
        them apart -- so the directory could not answer the one question it
        exists for. Failed spans are written too, under failed/, because a
        rejected extraction has to be listened to before a QC threshold can be
        judged.
        """
        counts = {"separated": 0, "failed": 0, "skipped": 0}
        if not self.enabled:
            return counts
        try:
            import soundfile as sf
        except Exception as e:
            if self.logger and not self._warned_sf:
                self.logger.warning(f"[stage-out] soundfile unavailable, no clips written: {e}")
                self._warned_sf = True
            return counts

        ok_dir = self.stage_dir("separation", "audio", "separated")
        bad_dir = self.stage_dir("separation", "audio", "failed")
        for seg in segments:
            audio = getattr(seg, "enhanced_audio", None)
            spans = getattr(seg, "tse_spans", None) or []
            fails = getattr(seg, "tse_failed_spans", None) or []
            if audio is None or not len(audio):
                counts["skipped"] += 1
                continue
            if not spans and not fails:
                counts["skipped"] += 1        # never overlapped; nothing to audit
                continue

            name = f"{seg.index}_{seg.speaker}"
            try:
                if spans:
                    sim = max((s[2] for s in spans if len(s) > 2), default=-1.0)
                    sf.write(os.path.join(ok_dir, f"{name}_sim{sim:.2f}.wav"),
                             audio, sample_rate)
                    counts["separated"] += 1
                if fails:
                    reason = str(fails[0][2]) if len(fails[0]) > 2 else "unknown"
                    sf.write(os.path.join(bad_dir, f"{name}_{reason}.wav"),
                             audio, sample_rate)
                    counts["failed"] += 1
            except Exception as e:
                if self.logger and not self._warned_sf:
                    self.logger.warning(f"[stage-out] clip write failed for {name}: {e}")
                    self._warned_sf = True

        if self.logger:
            self.logger.info(
                f"[stage-out] separation audio: {counts['separated']} separated, "
                f"{counts['failed']} failed, {counts['skipped']} untouched (not written)")
        return counts

    # -- manifest -------------------------------------------------------
    def write_manifest(self, metadata: dict, extra: Optional[dict] = None):
        """One file tying the stages together, so losses can be traced.

        Segment counts per stage sit side by side here: a stage that silently
        drops rows shows up as a step down the list rather than as a surprise
        at the end.
        """
        if not self.enabled:
            return None

        path = os.path.join(self.output_dir, "manifest.json")

        # Merge with whatever is already on disk. Under stage-major execution
        # this service is constructed fresh inside every run(), so self.manifest
        # only ever holds the one stage this call computed -- writing it plain
        # left the file describing the last stage alone, and the flow it exists
        # to show could not be reconstructed. Stages from this run win, so a
        # recomputed stage replaces its earlier entry rather than being ignored.
        stages = {}
        try:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    stages = (json.load(f) or {}).get("stages", {}) or {}
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"[stage-out] could not read the existing manifest ({e}); "
                    "starting a fresh one")
        stages.update(self.manifest["stages"])
        self.manifest["stages"] = stages

        flow = []
        for name in ("diarization", "separation", "music_removal", "asr", "refinement"):
            st = self.manifest["stages"].get(name, {}).get("stats")
            if not st:
                continue
            n = st.get("n_segments") or st.get("segments_total") or st.get("n_transcripts")
            flow.append({"stage": name, "n": n, "warnings": len(st.get("warnings", []))})

        payload = dict(self.manifest)
        payload["metadata"] = metadata
        payload["flow"] = flow
        if extra:
            payload.update(extra)

        drops = [(a, b) for a, b in zip(flow, flow[1:])
                 if a["n"] and b["n"] and b["n"] < a["n"]]
        if drops:
            payload["warnings"] = [
                f"segment count fell from {a['n']} to {b['n']} between "
                f"{a['stage']} and {b['stage']}" for a, b in drops]
            if self.logger:
                for msg in payload["warnings"]:
                    self.logger.warning(f"[stage-out] {msg}")

        self._write_json(path, payload)
        if self.logger:
            self.logger.info(f"[stage-out] manifest -> {path}")
        return path
