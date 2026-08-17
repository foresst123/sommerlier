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
    split_long_segments
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
        
    def prepare_chunks(self, audio: AudioData, max_duration: float = 600.0, min_silence: float = 0.5) -> Tuple[List[DiarizationChunk], str]:
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
        is_sortformer = not args.dia3
        chunk_frames = []
        
        if is_sortformer:
            # Sortformer can batch diarize
            paths = [c.path for c in chunks]
            batch_result = self.diarizer.diarize(paths)
            for idx, res in enumerate(batch_result):
                # Sortformer returns a list of strings per chunk, e.g. ["0.0 1.5 speaker_0", ...]
                data = []
                if res:
                    # NeMo can sometimes nest the result depending on batch config
                    lists = [x for x in res if isinstance(x, (list, tuple))]
                    if not lists:
                        lists = [res] if isinstance(res, list) else [[res]]
                    segs = [s for sub in lists for s in sub if isinstance(s, str)]
                    
                    for seg in segs:
                        parts = seg.split()
                        if len(parts) >= 3:
                            start = float(parts[0])
                            end = float(parts[1])
                            # sp format is usually "speaker_X"
                            sp = parts[2]
                            num = int(sp.split('_')[1]) if '_' in sp else 0
                            speaker = f"SPEAKER_{num:02d}"
                            data.append({"start": start, "end": end, "speaker": speaker})
                            
                df = pd.DataFrame(data) if data else pd.DataFrame(columns=["start", "end", "speaker"])
                df = apply_sortformer_segment_padding(
                    df, 
                    pad_onset=getattr(args, "sortformer_pad_onset", 0.0),
                    pad_offset=getattr(args, "sortformer_pad_offset", 0.0),
                    audio_duration=chunks[idx].duration
                )
                # Apply global offset
                if not df.empty:
                    df["start"] += chunks[idx].offset
                    df["end"] += chunks[idx].offset
                chunk_frames.append(df)
        else:
            # Pyannote
            for chunk in chunks:
                diar_out = self.diarizer.diarize(chunk.path)
                data = []
                if isinstance(diar_out, Annotation):
                    for turn, _, speaker in diar_out.itertracks(yield_label=True):
                        data.append({
                            "start": turn.start + chunk.offset,
                            "end": turn.end + chunk.offset,
                            "speaker": speaker
                        })
                df = pd.DataFrame(data)
                chunk_frames.append(df)
                
        # Cross-chunk fusion
        if len(chunks) > 1 and self.embedder:
            aligned_frames = align_speakers_across_chunks(
                chunk_frames, 
                {"waveform": audio.waveform, "sample_rate": audio.sample_rate},
                self.embedder.inference,
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
        
        # Build schemas
        final_segments = []
        for d in raw_list:
            final_segments.append(Segment(index=str(d.get("index", "00000")).zfill(5), start=d["start"], end=d["end"], speaker=d["speaker"]))
            
        num_spk = len(combined_df["speaker"].unique())
        
        return DiarizationResult(
            segments=final_segments,
            num_speakers=num_spk,
            method="sortformer" if is_sortformer else "pyannote"
        )
