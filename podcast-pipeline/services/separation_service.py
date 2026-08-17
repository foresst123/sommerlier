import torch
import librosa
from typing import List, Tuple
from schemas.audio import AudioData
from schemas.segment import Segment, EnhancedSegment
from algorithms.diarization.overlap import detect_overlapping_segments
from algorithms.separation.postprocess import match_target_amplitude

class SeparationService:
    """Detects overlapping speech and separates them using SR-CorrNet."""
    
    def __init__(self, separator, embedder, logger=None):
        self.separator = separator
        self.embedder = embedder
        self.logger = logger
        self.min_embed_duration = 0.5
        
    def _identify_speaker(self, audio_segment, sample_rate, ref_embeddings, candidate_labels):
        if not self.embedder: return candidate_labels[0] if candidate_labels else None
        
        if sample_rate != 16000:
            audio_16k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=16000)
        else:
            audio_16k = audio_segment
            
        if len(audio_16k) / 16000 < self.min_embed_duration:
            return candidate_labels[0] if candidate_labels else None
            
        try:
            audio_tensor = torch.tensor(audio_16k, dtype=torch.float32).unsqueeze(0).to(self.embedder.inference.device)
            with torch.inference_mode():
                embedding = self.embedder.inference(audio_tensor)
                
            best_speaker = None
            best_sim = -1.0
            
            for spk in candidate_labels:
                if spk in ref_embeddings:
                    ref_emb = ref_embeddings[spk]
                    sim = torch.nn.functional.cosine_similarity(
                        embedding.mean(dim=1),
                        torch.tensor(ref_emb).to(self.embedder.inference.device).mean(dim=0).unsqueeze(0).mean(dim=1), # Handling shapes
                        dim=0
                    ).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_speaker = spk
            return best_speaker
        except Exception:
            return candidate_labels[0] if candidate_labels else None

    def process_overlaps(self, segments: List[Segment], audio: AudioData, overlap_threshold: float = 1.0) -> List[EnhancedSegment]:
        if not self.separator:
            return [EnhancedSegment(**s.__dict__) for s in segments]
            
        if self.logger: self.logger.info("Processing overlapping segments with SR-CorrNet")
        
        # Convert to dict for algorithm
        seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "index": s.index} for s in segments]
        overlapping_pairs = detect_overlapping_segments(seg_dicts, overlap_threshold=overlap_threshold, logger=self.logger)
        
        if not overlapping_pairs:
            return [EnhancedSegment(**s.__dict__) for s in segments]
            
        enhanced_segments = [EnhancedSegment(**s.__dict__) for s in segments]
        sr = audio.sample_rate
        waveform = audio.waveform
        
        # Build reference embeddings
        ref_embeddings = {}
        if self.embedder:
            from algorithms.diarization.fusion import _compute_chunk_speaker_centroids
            import pandas as pd
            df = pd.DataFrame(seg_dicts)
            ref_embeddings = _compute_chunk_speaker_centroids(df, {"waveform": waveform, "sample_rate": sr}, self.embedder.inference)
            
        for pair in overlapping_pairs:
            seg1, seg2 = pair["seg1"], pair["seg2"]
            start_sec = pair["overlap_start"]
            end_sec = pair["overlap_end"]
            
            start_frame = int(start_sec * sr)
            end_frame = int(end_sec * sr)
            mixed_audio = waveform[start_frame:end_frame]
            
            src1, src2 = self.separator.separate(mixed_audio, sr)
            
            # Amplitude matching
            src1 = match_target_amplitude(src1, mixed_audio)
            src2 = match_target_amplitude(src2, mixed_audio)
            
            # Identify
            candidates = [seg1["speaker"], seg2["speaker"]]
            id_1 = self._identify_speaker(src1, sr, ref_embeddings, candidates)
            id_2 = seg1["speaker"] if id_1 == seg2["speaker"] else seg2["speaker"]
            
            # Update enhanced_segments
            for enh_seg in enhanced_segments:
                if enh_seg.index == seg1["index"] and enh_seg.speaker == id_1:
                    enh_seg.enhanced_audio = src1
                    enh_seg.srcorrnet = True
                elif enh_seg.index == seg2["index"] and enh_seg.speaker == id_1:
                    enh_seg.enhanced_audio = src1
                    enh_seg.srcorrnet = True
                elif enh_seg.index == seg1["index"] and enh_seg.speaker == id_2:
                    enh_seg.enhanced_audio = src2
                    enh_seg.srcorrnet = True
                elif enh_seg.index == seg2["index"] and enh_seg.speaker == id_2:
                    enh_seg.enhanced_audio = src2
                    enh_seg.srcorrnet = True
                    
        return enhanced_segments
