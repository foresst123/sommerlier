import bisect
import numpy as np
import librosa
from typing import List, Dict, Tuple
from schemas.audio import AudioData
from schemas.segment import Segment, EnhancedSegment
from algorithms.diarization.overlap import detect_overlapping_segments

class TargetExtractionService:
    """
    Replaces SeparationService. Uses Target Speaker Extraction (TSE) to isolate overlapping speech.
    Preserves backchannels natively using enrollment profiles and context windows.
    """
    
    def __init__(self, tse_model, logger=None):
        self.tse_model = tse_model
        self.logger = logger
        
    def mine_enrollments(self, segments: List[Segment], audio: AudioData, min_dur: float = 2.0, top_k: int = 5) -> Dict[str, List[np.ndarray]]:
        """
        Component 1: Extract clean, non-overlapping segments for each speaker to serve as TSE enrollments.
        """
        if self.logger: self.logger.info("Mining enrollments for Target Speaker Extraction...")
        
        # Convert to dicts for overlap detection
        seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "index": s.index} for s in segments]
        
        # We need to find segments that DO NOT overlap with anything else.
        # A simple way is to use detect_overlapping_segments and exclude them.
        overlaps = detect_overlapping_segments(seg_dicts, overlap_threshold=0.1)
        overlap_ranges = [(o["overlap_start"], o["overlap_end"]) for o in overlaps]
        
        enrollments = {}
        sr = audio.sample_rate
        waveform = audio.waveform

        # Sorting once lets a single sweep find each segment's intersecting
        # overlaps, instead of rescanning the full overlap list per segment.
        overlap_ranges.sort(key=lambda x: x[0])
        overlap_starts = [o[0] for o in overlap_ranges]

        # The clean sub-segments of a segment do not depend on the duration
        # threshold, only the filter over them does. Compute them once.
        clean_by_speaker: Dict[str, List[Tuple[float, float]]] = {}
        for s in segments:
            curr_start = s.start
            sub_segments = []

            # Overlaps starting at or after s.end cannot intersect it.
            end_idx = bisect.bisect_left(overlap_starts, s.end)
            for o_start, o_end in overlap_ranges[:end_idx]:
                if o_end <= s.start:
                    continue
                if curr_start < o_start:
                    sub_segments.append((curr_start, o_start))
                curr_start = max(curr_start, o_end)

            if curr_start < s.end:
                sub_segments.append((curr_start, s.end))

            clean_by_speaker.setdefault(s.speaker, []).extend(sub_segments)

        for spk in set(s.speaker for s in segments):
            candidates = clean_by_speaker.get(spk, [])
            clean_clips = []
            for dur_thresh in [min_dur, min_dur / 2, 0.5, 0.1]:
                clean_clips = [c for c in candidates if (c[1] - c[0]) >= dur_thresh]
                if clean_clips:
                    break  # Found clips, stop lowering threshold

            if not clean_clips and self.logger:
                self.logger.warning(
                    f"No clean (non-overlapping) audio found for speaker {spk}; "
                    "TSE will skip every overlap involving them."
                )

            # Sort by duration descending
            clean_clips.sort(key=lambda c: c[1] - c[0], reverse=True)

            enrollments[spk] = []
            for start, end in clean_clips[:top_k]:
                start_f = int(start * sr)
                end_f = int(end * sr)
                enrollments[spk].append(waveform[start_f:end_f].copy())

        return enrollments

    # Removed mock _compute_itc and _compute_itd as they are replaced by real sim_A, sim_B from tse_model
        
    def _cross_fade(self, orig_audio: np.ndarray, new_audio: np.ndarray, fade_samples: int) -> np.ndarray:
        """Smoothly replace a portion of orig_audio with new_audio using Hanning crossfade."""
        result = orig_audio.copy()
        limit = min(len(orig_audio), len(new_audio))
        fade_samples = min(fade_samples, limit // 2)
        if fade_samples <= 0 or limit == 0:
            result[:limit] = new_audio[:limit]
            return result
            
        # Create Hanning window for fading
        fade_in = np.hanning(2 * fade_samples)[:fade_samples]
        fade_out = 1.0 - fade_in
        
        # Crossfade start
        result[:fade_samples] = orig_audio[:fade_samples] * fade_out + new_audio[:fade_samples] * fade_in
        
        # Replace middle
        mid_limit = limit - fade_samples
        result[fade_samples:mid_limit] = new_audio[fade_samples:mid_limit]
        
        # Crossfade end
        result[mid_limit:limit] = orig_audio[mid_limit:limit] * fade_in + new_audio[mid_limit:limit] * fade_out
        return result
        
    def process_overlaps(self, segments: List[Segment], audio: AudioData, overlap_threshold: float = 0.1) -> List[EnhancedSegment]:
        """
        Process overlapping regions using TSE, Context Windows, and Confidence Gating.
        """
        if not self.tse_model:
            return [EnhancedSegment(**s.__dict__) for s in segments]
            
        if self.logger: self.logger.info("Processing overlaps with Target Speaker Extraction (TSE)")
        
        seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "index": s.index} for s in segments]
        overlapping_pairs = detect_overlapping_segments(seg_dicts, overlap_threshold=overlap_threshold, logger=self.logger)
        
        if not overlapping_pairs:
            return [EnhancedSegment(**s.__dict__) for s in segments]
            
        enhanced_segments = [EnhancedSegment(**s.__dict__) for s in segments]
        sr = audio.sample_rate
        waveform = audio.waveform
        total_len = len(waveform)
        
        # Initialize enhanced_audio
        for enh_seg in enhanced_segments:
            start_f = int(enh_seg.start * sr)
            end_f = int(enh_seg.end * sr)
            enh_seg.enhanced_audio = waveform[start_f:end_f].copy()
            
        # Component 1: Mine Enrollments
        enrollments = self.mine_enrollments(segments, audio, min_dur=2.0, top_k=5)

        # Indices are unique after split_long_segments, so one lookup table beats
        # rescanning every segment per overlap pair.
        seg_by_index = {s.index: s for s in enhanced_segments}

        from tqdm import tqdm
        for pair in tqdm(overlapping_pairs, desc="[TSE Extractor]", leave=True):
            seg1, seg2 = pair["seg1"], pair["seg2"]
            overlap_start = pair["overlap_start"]
            overlap_end = pair["overlap_end"]
            
            # Component 2: Context Window (±1.5s)
            ctx_pad = 1.5
            ctx_start = max(0.0, overlap_start - ctx_pad)
            ctx_end = min(total_len / sr, overlap_end + ctx_pad)
            
            ctx_start_f = int(ctx_start * sr)
            ctx_end_f = int(ctx_end * sr)
            context_audio = waveform[ctx_start_f:ctx_end_f].copy()
            
            target_A = seg1["speaker"]
            target_B = seg2["speaker"]
            
            # Ensure we have enrollments
            if target_A not in enrollments or target_B not in enrollments or not enrollments[target_A] or not enrollments[target_B]:
                continue
                
            # Component 3: TSE Engine (1-Pass Separation)
            track_A_ctx, track_B_ctx, sim_A, sim_B = self.tse_model.separate_two_speakers(
                context_audio, 
                enroll_A=enrollments[target_A], 
                enroll_B=enrollments[target_B], 
                sample_rate=sr, 
                id_A=target_A, 
                id_B=target_B
            )
            
            # Crop the ±1.5s padding to extract just the overlap core
            pad_start_frames = int((overlap_start - ctx_start) * sr)
            core_len = int((overlap_end - overlap_start) * sr)

            # A track shorter than the nominal core would silently yield a short
            # slice, leaving part of the overlap unprocessed while still flagged
            # as separated. Detect that instead of hiding it.
            available = min(len(track_A_ctx), len(track_B_ctx)) - pad_start_frames
            if available < core_len:
                if available <= 0:
                    if self.logger:
                        self.logger.warning(
                            f"TSE returned too little audio for overlap at {overlap_start:.2f}s; skipping."
                        )
                    continue
                if self.logger:
                    self.logger.warning(
                        f"TSE track shorter than overlap core at {overlap_start:.2f}s "
                        f"({available} < {core_len} samples); splicing what is available."
                    )
                core_len = available

            track_A_core = track_A_ctx[pad_start_frames : pad_start_frames + core_len]
            track_B_core = track_B_ctx[pad_start_frames : pad_start_frames + core_len]

            # Component 4: Quality Control

            # 4.1 Check real ECAPA matching scores (Intra-Track Consistency equivalent)
            if sim_A < 0.25 or sim_B < 0.25:
                if self.logger: self.logger.warning(f"TSE QC failed (Low ECAPA similarity A:{sim_A:.2f} B:{sim_B:.2f}) at {overlap_start}s.")
                if self.logger: self.logger.warning(f"Discarding failed extraction for overlap at {overlap_start}s. Retaining original mixture.")
                continue

            # Component 5: Splice with Cross-fade (20ms fade)
            fade_samples = int(0.02 * sr)
            for seg_dict, track_core in ((seg1, track_A_core), (seg2, track_B_core)):
                enh_seg = seg_by_index.get(seg_dict["index"])
                if enh_seg is None:
                    continue

                rel_start = int((overlap_start - enh_seg.start) * sr)
                if rel_start < 0:
                    continue

                limit = min(len(track_core), len(enh_seg.enhanced_audio) - rel_start)
                if limit <= 0:
                    continue

                enh_seg.enhanced_audio[rel_start:rel_start + limit] = self._cross_fade(
                    enh_seg.enhanced_audio[rel_start:rel_start + limit], track_core[:limit], fade_samples
                )
                enh_seg.tse = True  # Flag indicating it was processed

        return enhanced_segments

    def export_sdlm_dual_channel(self, enhanced_segments: List[EnhancedSegment], audio_duration: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Component 6: Export Dual-Channel Audio for SDLM training.
        Reconstructs the full timeline into two continuous tracks (Track 0, Track 1) from the separated segments.
        Uses additive mixing to preserve overlapping segments belonging to the same speaker.
        """
        total_samples = int(audio_duration * sr)
        track_0 = np.zeros(total_samples, dtype=np.float32)
        track_1 = np.zeros(total_samples, dtype=np.float32)
        
        # Identify the two speakers. Sort alphabetically to assign Track 0 and Track 1 consistently.
        speakers = sorted(list(set(s.speaker for s in enhanced_segments)))
        if not speakers:
            return track_0, track_1
            
        if len(speakers) > 2:
            if self.logger: self.logger.warning(f"export_sdlm_dual_channel: Expected 2 speakers, found {len(speakers)}. Extra speakers will be ignored.")
            
        spk_0 = speakers[0]
        spk_1 = speakers[1] if len(speakers) > 1 else None
        
        for seg in enhanced_segments:
            if seg.speaker not in [spk_0, spk_1]:
                continue
                
            start_idx = int(seg.start * sr)
            end_idx = start_idx + len(seg.enhanced_audio)
            
            # Ensure we don't exceed buffer bounds
            if end_idx > total_samples:
                audio_to_paste = seg.enhanced_audio[:total_samples - start_idx]
                end_idx = total_samples
            else:
                audio_to_paste = seg.enhanced_audio
                
            # Additive mixing to preserve same-speaker overlaps
            if seg.speaker == spk_0:
                track_0[start_idx:end_idx] += audio_to_paste
            elif seg.speaker == spk_1:
                track_1[start_idx:end_idx] += audio_to_paste
                
        # Clip to prevent clipping distortion if additive mixing exceeded [-1.0, 1.0]
        np.clip(track_0, -1.0, 1.0, out=track_0)
        np.clip(track_1, -1.0, 1.0, out=track_1)
        
        return track_0, track_1
