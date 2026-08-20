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
                 logger=None,
                 model_loader=None,
                 worker_services=None):
        self.audio_svc = audio_svc
        self.diarization_svc = diarization_svc
        self.separation_svc = separation_svc
        self.music_svc = music_svc
        self.asr_svc = asr_svc
        self.caption_svc = caption_svc
        self.refinement_svc = refinement_svc
        self.export_svc = export_svc
        self.logger = logger
        self.model_loader = model_loader
        # {"diarizen": svc, "sidon": svc, ...} so each worker's VRAM can be
        # released as soon as its stage is done rather than at process exit.
        self.worker_services = worker_services or {}

    def _release_worker(self, name: str):
        """Stop a worker subprocess now that its stage is finished."""
        service = self.worker_services.get(name)
        if not service or getattr(service, "process", None) is None:
            return
        if self.logger:
            self.logger.info(f"Releasing {name} worker (stage complete, freeing VRAM)")
        try:
            service.stop()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to stop {name} worker: {e}")


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
            if self.logger: self.logger.info(f"[DEBUG] Diarization returned {len(diarization_result.segments)} segments via {diarization_result.method}")

            # Everything downstream is per-segment, so an empty result silently
            # produces an empty transcript that still reports success.
            if not diarization_result.segments:
                raise RuntimeError(
                    f"Diarization ({diarization_result.method}) produced no segments for "
                    f"{audio_path}. Check the worker log above for the underlying error; "
                    "continuing would write an empty transcript."
                )

            checkpoint.save("diarization", diarization_result)
            
        if self.model_loader:
            self.model_loader.unload("diarizer")
            self.model_loader.unload("vad")
        # Unloading the client does not touch the subprocess holding the weights.
        self._release_worker("diarizen")

        if getattr(args, "stop_after", None) == "diarization":
            if self.logger: self.logger.info("Stopping pipeline after diarization as requested by --stop_after.")
            return None
            
        # 3. Speech Separation (Overlap)
        if checkpoint.exists("separation"):
            if self.logger: self.logger.info("Loading Separation from checkpoint")
            enhanced_segments = checkpoint.load("separation")
        else:
            enhanced_segments = self.separation_svc.process_overlaps(diarization_result.segments, audio_data)
            if self.logger: self.logger.info(f"[DEBUG] After Separation: {len(enhanced_segments)} segments")
            checkpoint.save("separation", enhanced_segments)
            
        if self.model_loader:
            self.model_loader.unload("separator")
            self.model_loader.unload("embedder")
        self._release_worker("sidon")

        if getattr(args, "stop_after", None) == "separation":
            if self.logger: self.logger.info("Stopping pipeline after separation as requested by --stop_after.")
            return None
            
        # 4. Background Music Removal
        if checkpoint.exists("music_removal"):
            if self.logger: self.logger.info("Loading Music Removal from checkpoint")
            enhanced_segments = checkpoint.load("music_removal")
        else:
            enhanced_segments = self.music_svc.process_segments(enhanced_segments, audio_data)
            if self.logger: self.logger.info(f"[DEBUG] After Music Removal: {len(enhanced_segments)} segments")
            checkpoint.save("music_removal", enhanced_segments)
            
        if self.model_loader:
            self.model_loader.unload("panns")
            self.model_loader.unload("demucs")
            
        if getattr(args, "stop_after", None) == "music_removal":
            if self.logger: self.logger.info("Stopping pipeline after music_removal as requested by --stop_after.")
            return None
            
        # 5. ASR Ensemble (MoE)
        if checkpoint.exists("asr"):
            if self.logger: self.logger.info("Loading ASR from checkpoint")
            transcripts = checkpoint.load("asr")
        else:
            if self.logger: self.logger.info(f"[DEBUG] Sending {len(enhanced_segments)} segments to ASR")
            transcripts = self.asr_svc.process(enhanced_segments, audio_data)
            if self.logger: self.logger.info(f"[DEBUG] ASR returned {len(transcripts)} transcripts")
            checkpoint.save("asr", transcripts)

        # The Qwen3 worker is done; the refinement LLM needs its ~4.7GB.
        self._release_worker("qwen3")

        if getattr(args, "stop_after", None) == "asr":
            if self.logger: self.logger.info("Stopping pipeline after asr as requested by --stop_after.")
            return transcripts
            
        # 6. Qwen3-Omni Captioning
        if getattr(args, "qwen3omni", False):
            if checkpoint.exists("captioning"):
                if self.logger: self.logger.info("Loading Captioning from checkpoint")
                transcripts = checkpoint.load("captioning")
            else:
                enhanced_audio_dict = {s.index: s.enhanced_audio for s in enhanced_segments if s.enhanced_audio is not None}
                transcripts = self.caption_svc.add_captions(transcripts, audio_data, enhanced_audio_dict)
                checkpoint.save("captioning", transcripts)
                
        if getattr(args, "stop_after", None) == "captioning":
            if self.logger: self.logger.info("Stopping pipeline after captioning as requested by --stop_after.")
            return transcripts
                
        # 7. LLM Refinement
        if getattr(args, "llm_refinement", False):
            # None keeps the service's built-in fusion prompt; only an explicit
            # override replaces it.
            transcripts = self.refinement_svc.refine(
                transcripts, getattr(args, "llm_prompt", None) or None
            )
            
        # 8. Export Results
        save_path = getattr(args, "save_path", "./output")
        if save_path == "./output":
            suffix = "pyannote" if getattr(args, "dia3", False) else "diarizen"
            do_vad = getattr(args, "vad", False)
            merge_gap = getattr(args, "merge_gap", 2.0)
            seg_th = getattr(args, "seg_th", 0.11)
            min_cluster = getattr(args, "min_cluster_size", 11)
            clust_th = getattr(args, "clust_th", 0.5)
            llm = getattr(args, "LLM", "case_0")
            
            base_dir = os.path.dirname(audio_path)
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            
            save_path = os.path.join(
                base_dir, "_final",
                f"-tse-{getattr(args, 'tse', False)}-demucs-{getattr(args, 'panns', False)}-vad-{do_vad}-diaModel-{suffix}-initPrompt-True-merge_gap-{merge_gap}-seg_th-{seg_th}-cl_min-{min_cluster}-cl-th-{clust_th}-LLM-{llm}",
                audio_name
            )
            
        os.makedirs(save_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        metadata = {
            "audio_file": os.path.basename(audio_path),
            # Stem without extension: what downstream readers key results on.
            "audio_name": base_name,
            "diarization_model": "pyannote" if getattr(args, "dia3", False) else "diarizen",
            # Report what actually ran, not what the flags requested.
            "asr_models": self.asr_svc.active_models(),
            "panns_enabled": getattr(args, "panns", False),
            "vad_enabled": getattr(args, "vad", False),
            "tse_enabled": getattr(args, "tse", False),
            "llm_refinement": getattr(args, "llm_refinement", False),
            "qwen3omni_caption": getattr(args, "qwen3omni", False)
        }
        
        self.export_svc.export_json(transcripts, os.path.join(save_path, f"{base_name}.json"), metadata=metadata)
        self.export_svc.export_srt(transcripts, os.path.join(save_path, f"{base_name}.srt"))
        self.export_svc.export_mp3_segments(transcripts, audio_data, save_path, base_name)
        
        # Xuất loạt audio đã tách ra thư mục `separation`
        self.export_svc.export_separated_audio(enhanced_segments, audio_data.sample_rate, save_path)
        
        # [Added Feature] Dump intermediate Diarization and Separation results to the final folder for comparison
        import json
        try:
            with open(os.path.join(save_path, f"{base_name}_intermediate_diarization.json"), "w", encoding="utf-8") as f:
                json.dump([seg.__dict__ for seg in diarization_result.segments], f, ensure_ascii=False, indent=2)
            
            with open(os.path.join(save_path, f"{base_name}_intermediate_separation.json"), "w", encoding="utf-8") as f:
                # enhanced_segments might contain large numpy arrays, so we exclude 'enhanced_audio' from the dump
                sep_data = []
                for s in enhanced_segments:
                    s_dict = s.__dict__.copy()
                    s_dict.pop('enhanced_audio', None)
                    sep_data.append(s_dict)
                json.dump(sep_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.logger: self.logger.error(f"Failed to export intermediate results: {e}")
        
        if self.logger: self.logger.info(f"Pipeline completed successfully. Results saved to {save_path}")
        return transcripts
