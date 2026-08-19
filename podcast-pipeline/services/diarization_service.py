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
    cut_by_speaker_label
)
from algorithms.diarization.fusion import align_speakers_across_chunks
from algorithms.diarization.overlap import detect_overlapping_segments

class DiarizationService:
    """Handles audio chunking, model inference (Pyannote/Sortformer), and cross-chunk fusion."""
    
    def __init__(self, diarizer, vad_model=None, embedder=None, logger=None, diarizer_config=None):
        self.diarizer = diarizer
        self.vad_model = vad_model
        self.embedder = embedder
        self.logger = logger
        self.diarizer_config = diarizer_config or {}
        
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
            
        # Both backends get the same speaker bounds so switching --dia3 does not
        # silently change how many speakers the run may find.
        num_speakers = self.diarizer_config.get("num_speakers")
        min_speakers = self.diarizer_config.get("min_speakers")
        max_speakers = self.diarizer_config.get("max_speakers")

        if self.logger:
            self.logger.info(
                f"Diarizing with speaker bounds num={num_speakers} min={min_speakers} max={max_speakers}"
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
        
        # Apply VAD to split long continuous segments if VAD is available
        if getattr(args, "vad", False) and self.vad_model:
            vad_audio = {"waveform": audio.waveform, "sample_rate": audio.sample_rate}
            raw_list = self.vad_model.vad(combined_df, vad_audio)
        else:
            raw_list = df_to_list(combined_df)
            
        # Apply merge and smooth logic
        merge_gap = getattr(args, "merge_gap", 2.0)
        smoothed_list = cut_by_speaker_label(raw_list, merge_gap=merge_gap, logger=self.logger)
        
        # Split segments that are too long
        final_list = split_long_segments(smoothed_list, max_duration=30.0)
        
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
