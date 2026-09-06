import os
import tempfile
import pandas as pd
from typing import List, Tuple, Any
from pyannote.core import Annotation
from schemas.audio import AudioData, DiarizationChunk
from schemas.segment import Segment
from schemas.diarization import DiarizationResult, OverlapPair
from utils.segment_utils import (
    build_silence_intervals, 
    build_chunk_ranges, 
    apply_sortformer_segment_padding,
    df_to_list,
    deduplicate_segments_by_index,
    split_long_segments,
    split_at_seams,
    cut_by_speaker_label,
    merge_ghost_speakers,
)
from algorithms.diarization.fusion import align_speakers_across_chunks
from algorithms.diarization.overlap import detect_overlapping_segments

class DiarizationService:
    """Handles audio chunking, model inference (Pyannote/Sortformer), and cross-chunk fusion."""
    
    def __init__(self, diarizer=None, vad_model=None, embedder=None, logger=None,
                 diarizer_config=None, model_loader=None):
        self._diarizer = diarizer
        self._vad_model = vad_model
        self._embedder = embedder
        self.model_loader = model_loader
        self.logger = logger
        self.diarizer_config = diarizer_config or {}

    # Models are fetched from the loader on use, not captured at construction.
    # PipelineService loads each stage's models when that stage runs, so a
    # reference taken here would be None for every stage that had not loaded
    # yet -- and would stay None after it did.
    def _model(self, held, name):
        if held is not None:
            return held
        return self.model_loader.get(name) if self.model_loader else None

    @property
    def diarizer(self):
        return self._model(self._diarizer, "diarizer")

    @diarizer.setter
    def diarizer(self, model):
        self._diarizer = model

    @property
    def vad_model(self):
        return self._model(self._vad_model, "vad")

    @vad_model.setter
    def vad_model(self, model):
        self._vad_model = model

    @property
    def embedder(self):
        return self._model(self._embedder, "embedder")
        
    def _log_segment_stats(self, stage: str, seg_list: list):
        """One line per pipeline stage so segment loss can be attributed."""
        if not self.logger:
            return
        if not seg_list:
            self.logger.info(f"[DIAR:{stage}] 0 segments")
            return
        durs = sorted(s["end"] - s["start"] for s in seg_list)
        n = len(durs)
        speakers = len({s.get("speaker") for s in seg_list})
        self.logger.info(
            f"[DIAR:{stage}] n={n} speakers={speakers} | "
            f"dur min={durs[0]:.2f} p50={durs[n // 2]:.2f} max={durs[-1]:.2f} | "
            f"<0.5s={sum(d < 0.5 for d in durs)} <0.2s={sum(d < 0.2 for d in durs)}"
        )

    @embedder.setter
    def embedder(self, model):
        self._embedder = model

    def prepare_chunks(self, audio: AudioData, max_duration: float = 120.0, min_silence: float = 0.5) -> Tuple[List[DiarizationChunk], str]:
        """Deprecated. Returns a single dummy chunk since diarization now processes the full audio natively."""
        if self.logger:
            self.logger.info("Chunking bypassed. Diarization will process the full audio natively.")
        return [DiarizationChunk(path="memory", offset=0.0, duration=audio.duration)], ""
        
    def run_diarization(self, chunks: List[DiarizationChunk], audio: AudioData, args: Any) -> DiarizationResult:
        """Run diarization model on the full audio natively."""
        is_diarizen = not getattr(args, "dia3", False)
        import pandas as pd
        import torch
        
        # Prepare memory tensor for pipeline
        audio_input = {
            "waveform": torch.from_numpy(audio.waveform).unsqueeze(0),
            "sample_rate": audio.sample_rate
        }
        
        if self.logger:
            self.logger.info(f"Running diarization on the full audio file natively (Duration: {audio.duration:.2f}s)...")
            
        # Both backends read the same bounds from config, but they take them at
        # different points: DiariZen applies them to its clustering config when
        # the worker loads the model, so passing them again per call makes the
        # pipeline raise. Only pyannote accepts them as call arguments.
        num_speakers = self.diarizer_config.get("num_speakers")
        min_speakers = self.diarizer_config.get("min_speakers")
        max_speakers = self.diarizer_config.get("max_speakers")

        if is_diarizen:
            if self.logger:
                self.logger.info("Diarizing with DiariZen (speaker bounds applied via worker config)")
            diar_out = self.diarizer.diarize(audio_input)
        else:
            if self.logger:
                self.logger.info(
                    f"Diarizing with pyannote, speaker bounds num={num_speakers} "
                    f"min={min_speakers} max={max_speakers}"
                )
            diar_out = self.diarizer.diarize(
                audio_input,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            
        data = []
        annotation = (
            diar_out.speaker_diarization
            if hasattr(diar_out, "speaker_diarization")
            else diar_out
        )
        
        if annotation is not None:
            try:
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    data.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })
            except Exception as e:
                if self.logger: self.logger.error(f"Error iterating diarization result: {e}")
                
        combined_df = pd.DataFrame(data) if data else pd.DataFrame(columns=["start", "end", "speaker"])
        combined_df = combined_df.sort_values("start").reset_index(drop=True)

        # The exported *_intermediate_diarization.json is written after VAD, merge
        # and split, so it cannot show whether a low segment count came from the
        # diarizer or from cut_by_speaker_label. Log both ends.
        self._log_segment_stats("raw", df_to_list(combined_df))
        
        # Apply VAD to split long continuous segments if VAD is available
        if getattr(args, "vad", False) and self.vad_model:
            vad_audio = {"waveform": audio.waveform, "sample_rate": audio.sample_rate}
            raw_list = self.vad_model.vad(combined_df, vad_audio)
        else:
            raw_list = df_to_list(combined_df)
            
        # Apply merge and smooth logic
        merge_gap = getattr(args, "merge_gap", 2.0)
        # The merge loop keeps absorbing same-speaker turns until it hits this
        # ceiling, so the ceiling is what actually decides turn length -- a run
        # of it produced 47 segments piled against a 30s limit, none of them a
        # real turn. It also sets how lopsided a TSE window can get: a long turn
        # swallowing a 0.3s backchannel is what makes the separator emit one
        # source and silence. Keep merge and split reading the same number.
        max_seg = getattr(args, "max_segment_length", None) or 30.0
        self._log_segment_stats("post-vad", raw_list)
        smoothed_list = cut_by_speaker_label(
            raw_list, merge_gap=merge_gap, max_segment_length=max_seg, logger=self.logger)
        self._log_segment_stats(f"post-merge(gap={merge_gap} max={max_seg})", smoothed_list)

        # Dissolve speakers too small to be participants before anything
        # downstream has to reason about them. Target extraction is the caller
        # that cares: it refuses an overlap whose window holds three speakers
        # and cannot enrol anyone with under 1.5s of clean audio, so a
        # three-second cluster costs far more than its length.
        smoothed_list = merge_ghost_speakers(smoothed_list, logger=self.logger)
        self._log_segment_stats("post-ghost-merge", smoothed_list)

        # Split segments that are too long. Passing the waveform lets the cut
        # land on a pause instead of on the stopwatch, so a forced split stops
        # clipping words in half.
        final_list = split_long_segments(
            smoothed_list, max_duration=max_seg,
            waveform=audio.waveform, sample_rate=audio.sample_rate)
        self._log_segment_stats("post-split", final_list)

        # Last, and last on purpose: no segment may straddle a join left by
        # excising. Merging is what creates them -- two turns of one speaker
        # either side of a join sit 0.05s apart in the cut timeline, well
        # inside merge_gap -- so this has to come after merge, not before.
        #
        # Read through getattr for the same reason music_map is: the pipeline
        # sets it, a service built for a test has none, and no timeline must
        # read as "nothing was cut".
        timeline = getattr(self, "timeline", None)
        seams = timeline.seams() if timeline else []
        if seams:
            before = len(final_list)
            final_list = split_at_seams(final_list, seams)
            self._log_segment_stats(f"post-seam-split({len(seams)} join(s))", final_list)
            if self.logger and len(final_list) != before:
                self.logger.info(
                    f"Split {len(final_list) - before} segment(s) that spanned a "
                    "cut; each piece is now one contiguous stretch of the source")

        # Build schemas
        final_segments = []
        for d in final_list:
            final_segments.append(Segment(index=str(d.get("index", "00000")).zfill(5), start=d["start"], end=d["end"], speaker=d["speaker"]))
            
        num_spk = len(combined_df["speaker"].unique())
        
        return DiarizationResult(
            segments=final_segments,
            num_speakers=num_spk,
            method="diarizen" if is_diarizen else "pyannote"
        )
