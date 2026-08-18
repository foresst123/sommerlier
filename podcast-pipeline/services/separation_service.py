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
        
        for spk in set(s.speaker for s in segments):
            clean_clips = []
            for dur_thresh in [2.0, 1.0, 0.5, 0.1]:
                for s in segments:
                    if s.speaker != spk:
                        continue
                        
                    # Find clean sub-segments within this segment
                    curr_start = s.start
                    sub_segments = []
                    
                    # Sort overlaps that intersect this segment
                    intersecting_overlaps = sorted(
                        [o for o in overlap_ranges if not (s.end <= o[0] or s.start >= o[1])],
                        key=lambda x: x[0]
                    )
                    
                    for o_start, o_end in intersecting_overlaps:
                        if curr_start < o_start:
                            sub_segments.append((curr_start, o_start))
                        curr_start = max(curr_start, o_end)
                        
                    if curr_start < s.end:
                        sub_segments.append((curr_start, s.end))
                        
                    for sub_start, sub_end in sub_segments:
                        if (sub_end - sub_start) >= dur_thresh:
                            # We store an object that has start and end attributes to mimic Segment
                            class CleanClip:
                                def __init__(self, start, end):
                                    self.start = start
                                    self.end = end
                            clean_clips.append(CleanClip(sub_start, sub_end))
                
                if clean_clips:
                    break # Found clips, stop lowering threshold
            
            # Sort by duration descending
            clean_clips.sort(key=lambda x: x.end - x.start, reverse=True)
            top_clips = clean_clips[:top_k]
            
            enrollments[spk] = []
            for c in top_clips:
                start_f = int(c.start * sr)
                end_f = int(c.end * sr)
                enrollments[spk].append(waveform[start_f:end_f].copy())
                
        return enrollments

    def _compute_itc(self, track_audio: np.ndarray) -> float:
        """Stub for Intra-Track Consistency."""
        # TODO: Implement WeSpeaker embedding cosine similarity
        return 0.5

    def _compute_itd(self, track_a: np.ndarray, track_b: np.ndarray) -> float:
        """Stub for Inter-Track Distinctiveness."""
        # TODO: Implement 1 - Cosine Similarity between A and B
        return 0.9
        
    def _compute_energy(self, track_audio: np.ndarray) -> float:
        """Compute signal energy (RMS)."""
        if len(track_audio) == 0: return 0.0
        return float(np.sqrt(np.mean(track_audio**2)))

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
                
            # Component 3: TSE Engine
            track_A_ctx = self.tse_model.extract_speaker(context_audio, enrollments[target_A], sample_rate=sr)
            track_B_ctx = self.tse_model.extract_speaker(context_audio, enrollments[target_B], sample_rate=sr)
            
            # Crop the ±1.5s padding to extract just the overlap core
            pad_start_frames = int((overlap_start - ctx_start) * sr)
            pad_end_frames = int((ctx_end - overlap_end) * sr)
            
            core_len = int((overlap_end - overlap_start) * sr)
            track_A_core = track_A_ctx[pad_start_frames : pad_start_frames + core_len]
            track_B_core = track_B_ctx[pad_start_frames : pad_start_frames + core_len]
            
            # Component 4: Quality Control
            itc_A = self._compute_itc(track_A_core)
            itd = self._compute_itd(track_A_core, track_B_core)
            
            # 4.1 Energy-based Confidence (Heuristic)
            energy_A = self._compute_energy(track_A_core)
            energy_B = self._compute_energy(track_B_core)
            total_energy = energy_A + energy_B + 1e-8
            energy_ratio_A = energy_A / total_energy
            
            # 4.2 Context Gap Flagging (Heuristic cho Backchannel)
            sorted_segs = sorted(segments, key=lambda x: x.start)
            def get_gap_to_last_turn(speaker: str, current_start: float) -> float:
                gap = float('inf')
                for s in reversed(sorted_segs):
                    if s.speaker == speaker and s.end <= current_start + 0.1:
                        gap = current_start - s.end
                        break # Found the most recent turn
                return gap
                
            gap_A = get_gap_to_last_turn(target_A, overlap_start)
            is_backchannel = (overlap_end - overlap_start) < 1.5
            
            flagged = False
            if itc_A < 0.35 or itd < 0.85:
                flagged = True
                if self.logger: self.logger.warning(f"TSE QC failed (ITC/ITD) at {overlap_start}s.")
            elif is_backchannel and gap_A > 2.0:
                flagged = True
                if self.logger: self.logger.warning(f"TSE QC failed (Context Gap > 2s for backchannel) at {overlap_start}s.")
                
            if flagged:
                # Fallback: Strict Discarding. Replace extracted track with silence to preserve SDLM quality
                if self.logger: self.logger.warning(f"Discarding failed extraction for overlap at {overlap_start}s.")
                track_A_core = np.zeros_like(track_A_core)
                track_B_core = np.zeros_like(track_B_core)
            
            # Component 5: Splice with Cross-fade (20ms fade = 320 samples at 16kHz)
            fade_samples = int(0.02 * sr)
            for enh_seg in enhanced_segments:
                if enh_seg.index == seg1["index"]:
                    rel_start = int((overlap_start - enh_seg.start) * sr)
                    limit = min(len(track_A_core), len(enh_seg.enhanced_audio[rel_start:]))
                    if limit > 0:
                        enh_seg.enhanced_audio[rel_start:rel_start+limit] = self._cross_fade(
                            enh_seg.enhanced_audio[rel_start:rel_start+limit], track_A_core[:limit], fade_samples
                        )
                        enh_seg.srcorrnet = True # Flag indicating it was processed
                elif enh_seg.index == seg2["index"]:
                    rel_start = int((overlap_start - enh_seg.start) * sr)
                    limit = min(len(track_B_core), len(enh_seg.enhanced_audio[rel_start:]))
                    if limit > 0:
                        enh_seg.enhanced_audio[rel_start:rel_start+limit] = self._cross_fade(
                            enh_seg.enhanced_audio[rel_start:rel_start+limit], track_B_core[:limit], fade_samples
                        )
                        enh_seg.srcorrnet = True
                        
        return enhanced_segments

    def export_sdlm_dual_channel(self, enhanced_segments: List[EnhancedSegment], audio_duration: float, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Component 6: Export Dual-Channel Audio for SDLM training.
        Reconstructs the full timeline into two continuous tracks (Track 0, Track 1) from the separated segments.
        """
        total_samples = int(audio_duration * sr)
        track_0 = np.zeros(total_samples, dtype=np.float32)
        track_1 = np.zeros(total_samples, dtype=np.float32)
        
        # Identify the two speakers. Sort alphabetically to assign Track 0 and Track 1 consistently.
        speakers = sorted(list(set(s.speaker for s in enhanced_segments)))
        if not speakers:
            return track_0, track_1
            
        spk_0 = speakers[0]
        spk_1 = speakers[1] if len(speakers) > 1 else None
        
        for seg in enhanced_segments:
            start_idx = int(seg.start * sr)
            end_idx = start_idx + len(seg.enhanced_audio)
            
            # Ensure we don't exceed buffer bounds
            if end_idx > total_samples:
                audio_to_paste = seg.enhanced_audio[:total_samples - start_idx]
                end_idx = total_samples
            else:
                audio_to_paste = seg.enhanced_audio
                
            if seg.speaker == spk_0:
                track_0[start_idx:end_idx] = audio_to_paste
            elif seg.speaker == spk_1:
                track_1[start_idx:end_idx] = audio_to_paste
                
        return track_0, track_1
