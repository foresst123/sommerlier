import os
from typing import Any
from utils.checkpoint import CheckpointManager
from schemas.audio import AudioData

class PipelineService:
    """Orchestrates the entire ASR pipeline using the defined services and Checkpoint mechanism."""
    
    def __init__(self, 
                 audio_svc, 
                 diarization_svc, 
                 separation_svc, 
                 music_svc, 
                 asr_svc, 
                 caption_svc, 
                 refinement_svc, 
                 export_svc, 
                 logger=None):
        self.audio_svc = audio_svc
        self.diarization_svc = diarization_svc
        self.separation_svc = separation_svc
        self.music_svc = music_svc
        self.asr_svc = asr_svc
        self.caption_svc = caption_svc
        self.refinement_svc = refinement_svc
        self.export_svc = export_svc
        self.logger = logger
        
    def run(self, args: Any, config: dict, audio_path: str):
        job_id = getattr(args, "job_id", "default_job")
        cache_dir = getattr(args, "cache_dir", "cache")
        checkpoint = CheckpointManager(cache_dir, job_id)
        
        # 1. Audio Preprocessing
        audio_data = self.audio_svc.load_audio(audio_path, target_sr=24000)
        
        # 2. Diarization (with VAD & Chunking)
        if checkpoint.exists("diarization"):
            if self.logger: self.logger.info("Loading Diarization from checkpoint")
            diarization_result = checkpoint.load("diarization")
        else:
            chunks, _ = self.diarization_svc.prepare_chunks(audio_data)
            diarization_result = self.diarization_svc.run_diarization(chunks, audio_data, args)
            checkpoint.save("diarization", diarization_result)
            
        # 3. Speech Separation (Overlap)
        if checkpoint.exists("separation"):
            if self.logger: self.logger.info("Loading Separation from checkpoint")
            enhanced_segments = checkpoint.load("separation")
        else:
            enhanced_segments = self.separation_svc.process_overlaps(diarization_result.segments, audio_data)
            checkpoint.save("separation", enhanced_segments)
            
        # 4. Background Music Removal
        if checkpoint.exists("music_removal"):
            if self.logger: self.logger.info("Loading Music Removal from checkpoint")
            enhanced_segments = checkpoint.load("music_removal")
        else:
            enhanced_segments = self.music_svc.process_segments(enhanced_segments, audio_data)
            checkpoint.save("music_removal", enhanced_segments)
            
        # 5. ASR Ensemble (MoE)
        if checkpoint.exists("asr"):
            if self.logger: self.logger.info("Loading ASR from checkpoint")
            transcripts = checkpoint.load("asr")
        else:
            transcripts = self.asr_svc.process(enhanced_segments, audio_data)
            checkpoint.save("asr", transcripts)
            
        # 6. Qwen3-Omni Captioning
        if getattr(args, "qwen3omni", False):
            if checkpoint.exists("captioning"):
                if self.logger: self.logger.info("Loading Captioning from checkpoint")
                transcripts = checkpoint.load("captioning")
            else:
                enhanced_audio_dict = {s.index: s.enhanced_audio for s in enhanced_segments if s.enhanced_audio is not None}
                transcripts = self.caption_svc.add_captions(transcripts, audio_data, enhanced_audio_dict)
                checkpoint.save("captioning", transcripts)
                
        # 7. LLM Refinement
        if getattr(args, "llm_refinement", False):
            prompt = getattr(args, "llm_prompt", "")
            transcripts = self.refinement_svc.refine(transcripts, prompt)
            
        # 8. Export Results
        save_path = getattr(args, "save_path", "./output")
        os.makedirs(save_path, exist_ok=True)
        base_name = audio_data.name.split('.')[0]
        
        self.export_svc.export_json(transcripts, os.path.join(save_path, f"{base_name}.json"))
        self.export_svc.export_srt(transcripts, os.path.join(save_path, f"{base_name}.srt"))
        self.export_svc.export_mp3_segments(transcripts, audio_data, save_path, base_name)
        
        if self.logger: self.logger.info(f"Pipeline completed successfully. Results saved to {save_path}")
        return transcripts
