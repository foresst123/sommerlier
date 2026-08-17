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
    
    def __init__(self, diarizer, vad_model=None, embedder=None, logger=None):
        self.diarizer = diarizer
        self.vad_model = vad_model
        self.embedder = embedder
        self.logger = logger
        
    def prepare_chunks(self, audio: AudioData, max_duration: float = 120.0, min_silence: float = 0.5) -> Tuple[List[DiarizationChunk], str]:
        """Split long audio using silence from VAD."""
        waveform = audio.waveform
        sr = audio.sample_rate
        
        if self.vad_model:
            def vad_func(wf, srate):
                return self.vad_model.get_speech_timestamps(wf, sampling_rate=srate)
            total_duration, silence_intervals = build_silence_intervals(waveform, sr, vad_func, min_silence)
        else:
            total_duration = len(waveform) / sr
            silence_intervals = [(0.0, total_duration)]
            
        chunk_ranges = build_chunk_ranges(total_duration, silence_intervals, max_duration)
        
        epsilon = 1e-3
        normalized_audio = audio.audio_segment
        temp_dir = tempfile.mkdtemp(prefix="pre_diar_")
        
        if len(chunk_ranges) == 1 and chunk_ranges[0][0] <= epsilon and abs(chunk_ranges[0][1] - total_duration) <= epsilon:
            temp_path = os.path.join(temp_dir, "full_audio.wav")
            normalized_audio.set_channels(1).export(temp_path, format="wav", parameters=["-ac", "1"])
            return [DiarizationChunk(path=temp_path, offset=0.0, duration=total_duration)], temp_dir

        normalized_audio = normalized_audio.set_channels(1)
        chunk_entries = []
        for idx, (start_sec, end_sec) in enumerate(chunk_ranges):
            start_ms = max(0, int(round(start_sec * 1000)))
            end_ms = max(start_ms, int(round(end_sec * 1000)))
            chunk_audio = normalized_audio[start_ms:end_ms]
            chunk_path = os.path.join(temp_dir, f"chunk_{idx:03d}.wav")
            chunk_audio.export(chunk_path, format="wav", parameters=["-ac", "1"])
            chunk_entries.append(DiarizationChunk(path=chunk_path, offset=start_sec, duration=end_sec - start_sec))
            
        if self.logger:
            self.logger.info(f"Chunked audio into {len(chunk_entries)} pieces for Diarization")
            
        return chunk_entries, temp_dir
        
    def run_diarization(self, chunks: List[DiarizationChunk], audio: AudioData, args: Any) -> DiarizationResult:
        """Run diarization model on all chunks, apply padding/fusion, and return unified segments."""
        is_diarizen = not getattr(args, "dia3", False)
        chunk_frames = []
        import pandas as pd
        
        if is_diarizen:
            # DiariZen
            from tqdm import tqdm
            for chunk in tqdm(chunks, desc="[DiariZen] Phân rã", leave=True):
                diar_out = self.diarizer.diarize(chunk.path)
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
                                "start": turn.start + chunk.offset,
                                "end": turn.end + chunk.offset,
                                "speaker": speaker
                            })
                    except Exception as e:
                        if self.logger: self.logger.error(f"Error iterating diarization result: {e}")
                df = pd.DataFrame(data) if data else pd.DataFrame(columns=["start", "end", "speaker"])
                chunk_frames.append(df)

        else:
            # Pyannote
            from tqdm import tqdm
            for chunk in tqdm(chunks, desc="[Pyannote] Phân rã", leave=True):
                # Dùng num_speakers=2 thay vì max_speakers=2 để khóa chính xác 2 người (đạt accuracy tốt nhất)
                diar_out = self.diarizer.diarize(chunk.path, num_speakers=2)
                data = []
                # Mirror original code: community-1 model wraps result in object with
                # .speaker_diarization attribute. Must check this first before treating
                # result as bare Annotation.
                annotation = (
                    diar_out.speaker_diarization
                    if hasattr(diar_out, "speaker_diarization")
                    else diar_out
                )
                if annotation is not None:
                    try:
                        for turn, _, speaker in annotation.itertracks(yield_label=True):
                            data.append({
                                "start": turn.start + chunk.offset,
                                "end": turn.end + chunk.offset,
                                "speaker": speaker
                            })
                    except Exception as e:
                        if self.logger: self.logger.error(f"Error iterating diarization result: {e}")
                df = pd.DataFrame(data) if data else pd.DataFrame(columns=["start", "end", "speaker"])
                chunk_frames.append(df)

                
        # Cross-chunk fusion
        if len(chunks) > 1 and self.embedder:
            speaker_link_threshold = getattr(args, "speaker_link_threshold", 0.49)
            aligned_frames = align_speakers_across_chunks(
                chunk_frames, 
                {"waveform": audio.waveform, "sample_rate": audio.sample_rate},
                self.embedder.inference,
                similarity_threshold=speaker_link_threshold,
                logger=self.logger
            )
        else:
            aligned_frames = chunk_frames
            
        # Combine
        valid_dfs = [df for df in aligned_frames if not df.empty]
        combined_df = pd.concat(valid_dfs, ignore_index=True) if valid_dfs else pd.DataFrame(columns=["start", "end", "speaker"])
        
        if combined_df.empty:
            return DiarizationResult(segments=[], num_speakers=0, method="sortformer" if is_sortformer else "pyannote")
            
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
