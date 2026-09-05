import os
from typing import Any
from utils.checkpoint import CheckpointManager
from utils.music_map import MusicMap, build as build_music_map
from utils.excise import TimelineMap, excise
from services.stage_output_service import StageOutputService
from schemas.audio import AudioData

# Above this share of a recording marked for removal, the map is not trusted:
# a podcast that is mostly singing is either mis-tagged or the wrong file, and
# either way an empty waveform downstream is the worst possible answer.
CUT_SHARE_LIMIT = float(os.environ.get("MUSIC_CUT_SHARE_LIMIT", "0.60"))


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
        # Non-None while a stage-major pass is running: model and worker
        # releases collect here instead of firing at the end of each file.
        self.defer_free = None
        self.defer_workers = None

    def _load(self, group: str):
        """Load one stage's models, at the point that stage runs.

        Loading everything up front put DiariZen (5.15GB), Sidon (2.62GB),
        PhoWhisper, Whisper, Demucs and the captioner on the card before the
        first stage had produced anything, so peak VRAM was the sum of every
        model rather than the largest pair. Each loader is idempotent, so under
        stage-major execution -- where this is reached once per file -- only the
        first call pays for it.

        A stage that loads from checkpoint never calls this, which is the point:
        a resumed run does not pay for models it will not use.
        """
        if self.model_loader is None:
            return
        w = self.worker_services
        {
            "base":        lambda: self.model_loader.load_base_models(),
            "diarization": lambda: self.model_loader.load_diarization_models(w.get("diarizen")),
            "separation":  lambda: self.model_loader.load_separation_models(w.get("sidon")),
            "music":       lambda: self.model_loader.load_music_models(),
            "panns":       lambda: self.model_loader.load_panns(),
            "asr":         lambda: self.model_loader.load_asr_models(w.get("qwen3")),
            "caption":     lambda: self.model_loader.load_caption_model(),
        }[group]()

    # A stage's switch in the profile, falling back to the flag that used to
    # control it. Named separately from those flags because several of them --
    # `panns`, `tse` -- gate more than one stage, and the profile now needs to
    # say which stages run rather than which models load.
    _LEGACY_FLAG = {
        "music_analysis": "panns",
        "music_removal": "panns",
        "separation": "tse",
        "captioning": "qwen3omni",
        "refinement": "llm_refinement",
    }

    @classmethod
    def step_enabled(cls, args, name: str) -> bool:
        """Whether `name` runs. Unlisted steps run, which is the old behaviour."""
        explicit = getattr(args, f"step_{name}", None)
        if explicit is not None:
            return bool(explicit)
        legacy = cls._LEGACY_FLAG.get(name)
        return bool(getattr(args, legacy, True)) if legacy else True

    def _free(self, args, *model_names):
        """Unload finished models unless the caller asked to keep them.

        --keep_models trades peak VRAM for reload time: worth it when several
        files run back to back on a GPU with room for everything, wrong when
        the models only fit because each stage releases the last.

        Under stage-major execution the caller runs one stage across every file
        before starting the next, so freeing here -- at the end of each file --
        would drop the model the very next file is about to use. The names are
        recorded instead and released once, when the stage finishes.
        """
        if getattr(args, "keep_models", False) or not self.model_loader:
            return
        if self.defer_free is not None:
            self.defer_free.update(model_names)
            return
        for name in model_names:
            self.model_loader.unload(name)

    def begin_stage_scope(self):
        """Hold model releases until end_stage_scope(), for stage-major runs."""
        self.defer_free = set()
        self.defer_workers = set()

    def end_stage_scope(self):
        """Release everything the stage deferred. Safe when no scope is open."""
        names, workers = self.defer_free, self.defer_workers
        self.defer_free = None
        self.defer_workers = None
        if names and self.model_loader:
            for name in sorted(names):
                self.model_loader.unload(name)
        for worker in sorted(workers or ()):
            service = self.worker_services.get(worker)
            if not service or getattr(service, "process", None) is None:
                continue
            if self.logger:
                self.logger.info(f"Releasing {worker} worker (stage complete, freeing VRAM)")
            try:
                service.stop()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to stop {worker} worker: {e}")

    def _release_worker(self, args, name: str):
        """Stop a worker subprocess now that its stage is finished.

        Honours --keep_models for the same reason _free does, and for a sharper
        one: the workers are spawned once in main(), before the file loop, and
        nothing spawns them again. Stopping one here without that guard leaves
        every later file in the run without a worker to talk to -- invisible on
        a single file, fatal from the second onwards.
        """
        if getattr(args, "keep_models", False):
            return
        if self.defer_workers is not None:
            self.defer_workers.add(name)
            return
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

    def _rebind_worker(self, args, worker_name: str, service, attr: str):
        """Restart `worker_name` if needed and hand the live process to its client.

        No-op when the worker is still running, which is the --keep_models case
        and the first file of any run.
        """
        if service is None or self.model_loader is None:
            return
        client = getattr(service, attr, None)
        if client is None:
            return
        process = self._ensure_worker(args, worker_name)
        if process is not None and getattr(client, "process", None) is not process:
            client.process = process

    def _ensure_worker(self, args, name: str):
        """Bring a worker back up if an earlier file released it.

        Pairs with _release_worker: without --keep_models a stage frees its
        worker's VRAM when it finishes, so the next file has to start it again.
        Returns the live process, or None when there is no such worker.
        """
        service = self.worker_services.get(name)
        if service is None:
            return None
        if getattr(service, "process", None) is not None:
            return service.process
        if self.logger:
            self.logger.info(f"Restarting {name} worker for this file")
        try:
            service.spawn()
            service.wait_ready()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to restart {name} worker: {e}")
            return None
        return getattr(service, "process", None)


    def _resolve_output_dir(self, args, audio_path: str) -> str:
        """The single place that decides where one file's outputs live.

        Every file gets its own directory, whether or not --save_path was given.
        The old code only built a per-file path when save_path was left at the
        literal default, so passing --save_path made a whole batch share one
        directory -- and export_separated_audio, which names its files
        "{index}_{speaker}_separated.wav" with an index that restarts at 00000
        per audio, then overwrote the previous file's audio with no warning.

        Returning a path that always ends in the audio's name is what keeps that
        collision impossible: every later export is relative to this directory.
        """
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        save_path = getattr(args, "save_path", None) or "./output"

        if save_path == "./output":
            suffix = "pyannote" if getattr(args, "dia3", False) else "diarizen"
            root = os.path.join(
                os.path.dirname(audio_path), "_final",
                f"-tse-{getattr(args, 'tse', False)}"
                f"-demucs-{getattr(args, 'panns', False)}"
                f"-vad-{getattr(args, 'vad', False)}"
                f"-diaModel-{suffix}-initPrompt-True"
                f"-merge_gap-{getattr(args, 'merge_gap', 2.0)}"
                f"-seg_th-{getattr(args, 'seg_th', 0.11)}"
                f"-cl_min-{getattr(args, 'min_cluster_size', 11)}"
                f"-cl-th-{getattr(args, 'clust_th', 0.5)}"
                f"-LLM-{getattr(args, 'LLM', 'case_0')}",
            )
        else:
            root = save_path

        return os.path.join(root, audio_name)

    def run(self, args: Any, config: dict, audio_path: str):
        # Scope the checkpoint to this audio file. Sharing one job_id across a
        # batch would make the second file load the first file's diarization and
        # emit its transcript -- the checkpoint exists, so nothing recomputes.
        # Resolved once, at the top: the separation dump needs it long before
        # the export stage does, and computing it twice is how they drifted apart.
        output_dir = self._resolve_output_dir(args, audio_path)
        if self.logger:
            self.logger.info(f"Outputs for this file: {output_dir}")

        # Stage artifacts are written as each stage completes, so --stop_after
        # and a mid-pipeline crash both leave the finished work on disk.
        stage_out = StageOutputService(
            output_dir, logger=self.logger,
            enabled=not getattr(args, "no_stage_output", False))

        base_job = getattr(args, "job_id", "default_job")
        job_id = f"{base_job}_{os.path.splitext(os.path.basename(audio_path))[0]}"
        cache_dir = getattr(args, "cache_dir", "cache")
        checkpoint = CheckpointManager(cache_dir, job_id)

        # Services are constructed once and reused for every file in a batch.
        for svc in (self.separation_svc, self.refinement_svc):
            reset = getattr(svc, "reset_stats", None)
            if reset:
                reset()

        # Without --keep_models each stage releases its worker when it finishes,
        # so a second file arrives with dead workers and they have to be
        # restarted. Only the ones this call will actually reach, though:
        # reviving all three at the top meant that by the refinement stage the
        # diarizer (5.15GB) and Sidon (2.62GB) were both resident again, and
        # the LLM hit OOM with 0.03GB free on a card that had just been emptied
        # for it. A checkpointed stage does not need its worker at all.
        if not checkpoint.exists("diarization"):
            self._rebind_worker(args, "diarizen", self.diarization_svc, "diarizer")
        if not checkpoint.exists("separation"):
            self._rebind_worker(args, "sidon", self.separation_svc, "tse_model")
        if not checkpoint.exists("asr"):
            self._rebind_worker(args, "qwen3", self.asr_svc, "qwen3")
        
        # 1. Audio Preprocessing
        audio_data = self.audio_svc.load_audio(audio_path, target_sr=24000)
        
        # Stage artifacts are written once per file, not once per run() entry:
        # under stage-major execution this method is re-entered per stage.
        computed_stages = set()

        # 2. What is playing, before anything else looks at the audio.
        #
        # PANNs' frame-level tagger labels every 10ms as speech, singing or
        # music, and the three need different handling: singing is skipped
        # entirely (song lyrics scored as dialogue are dirty data for a
        # full-duplex corpus), a music bed under speech is stripped, and clean
        # speech is left alone -- which on this corpus is almost all of it.
        #
        # Swept first so that everything downstream sees the verdict:
        # diarization segments cleaned audio, and separation's search for solo
        # speech to enrol on can avoid the beds.
        if checkpoint.exists("music_map"):
            music_map = MusicMap.from_json(checkpoint.load("music_map"))
        else:
            music_map = MusicMap()
            if self.step_enabled(args, "music_analysis"):
                # The tagger only. The vocal separator is loaded below, and
                # only when the map actually found a bed to strip.
                self._load("panns")
                detector = self.model_loader.get("panns") if self.model_loader else None
                music_map = build_music_map(
                    audio_data.waveform, audio_data.sample_rate, detector,
                    logger=self.logger)
                checkpoint.save("music_map", music_map.to_json())
        # Strip the beds now, so the diarizer -- and everything after it --
        # works on audio without music under the speech.
        #
        # The result is cached as the replaced stretches rather than as the
        # whole waveform. Under stage-major execution run() is re-entered once
        # per stage and reloads the audio from disk each time, so the strip has
        # to be re-applied on every entry or later stages would see the bed
        # again; caching a 50-minute waveform to achieve that would cost
        # ~290MB per file, while the stretches themselves are seconds.
        from utils.music_map import MUSIC
        if music_map.total_of(MUSIC) > 0 and self.step_enabled(args, "music_removal"):
            patches = checkpoint.load("music_patches")
            if patches is None:
                # Loaded only now: the map found something, so the separator
                # has work. A recording with no bed never pays for it.
                self._load("music")
                self.music_svc.demucs = (self.model_loader.get("demucs")
                                         if self.model_loader else None)
                patches = self.music_svc.strip_music_spans(
                    audio_data, music_map, logger=self.logger)
                checkpoint.save("music_patches", patches)
                self._free(args, "demucs")
            else:
                self.music_svc.apply_music_patches(audio_data, patches)
                if self.logger:
                    self.logger.info(f"Re-applied {len(patches)} cached music "
                                     "patch(es) to the waveform")

        # Cut the sung and standalone-music stretches out before anything else
        # sees the audio. Marking them and letting later stages skip does not
        # work: the diarizer still clusters on them and ASR still receives them,
        # so song lyrics reach the transcript scored as dialogue.
        #
        # This shortens the recording, so every timestamp downstream is in the
        # cut timeline and has to be translated back at export. The map is
        # checkpointed for exactly that.
        timeline = TimelineMap.from_json(checkpoint.load("timeline", fmt="json"))
        cuts = (music_map.excised_spans()
                if self.step_enabled(args, "cut_singing") else [])

        # Refuse to cut away the recording. A tagger that calls most of a
        # podcast singing has gone wrong -- thresholds set too low, or a file
        # that is genuinely music and does not belong in this corpus -- and
        # deleting the audio would turn that into a run that produces nothing
        # and says why only in a log line.
        if cuts:
            share = sum(b - a for a, b, _ in cuts) / max(audio_data.duration, 1e-9)
            if share > CUT_SHARE_LIMIT:
                if self.logger:
                    self.logger.warning(
                        f"Music analysis wants to cut {share * 100:.0f}% of "
                        f"{os.path.basename(audio_path)}; keeping the audio "
                        "whole. Check MUSIC_MAP_* thresholds, or whether this "
                        "file is a recording of music.")
                cuts = []

        if cuts and not timeline:
            trimmed, timeline = excise(audio_data.waveform,
                                       audio_data.sample_rate,
                                       [(a, b) for a, b, _ in cuts])
            audio_data.waveform = trimmed
            audio_data.duration = len(trimmed) / float(audio_data.sample_rate)
            checkpoint.save("timeline", timeline.to_json(), fmt="json")
            if self.logger:
                self.logger.info(
                    f"Cut {timeline.removed:.1f}s of singing/music from "
                    f"{len(cuts)} stretch(es); {audio_data.duration / 60:.1f} min remain")
        elif timeline:
            # A later run() entry: the audio was reloaded whole, so re-cut it to
            # match the timeline the earlier stages already worked in.
            trimmed, _ = excise(audio_data.waveform, audio_data.sample_rate,
                                [(a, b) for a, b, _ in cuts])
            audio_data.waveform = trimmed
            audio_data.duration = len(trimmed) / float(audio_data.sample_rate)
        self.timeline = timeline

        # Everything after the cut works in the shortened timeline, so the map
        # separation consults has to move with it.
        self.separation_svc.music_map = (music_map.remap(timeline) if timeline
                                         else music_map)

        if "music" not in computed_stages:
            stage_out.write_music(music_map, timeline,
                                  audio_data.waveform, audio_data.sample_rate)
            computed_stages.add("music")

        if getattr(args, "stop_after", None) == "music":
            if self.logger:
                self.logger.info("Stopping after music analysis as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "music"})
            return None

        # Diarization, ASR and export are load-bearing: nothing downstream has
        # an input without them. Turning one off therefore ends the run here
        # rather than producing a file with the stage quietly missing.
        for _required in ("diarization", "asr", "export"):
            if not self.step_enabled(args, _required):
                if self.logger:
                    self.logger.info(
                        f"Step '{_required}' is off in the profile; stopping after "
                        "music analysis. Nothing downstream of it has an input.")
                stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                          "stopped_before": _required})
                return None

        # 3. Diarization (with VAD & Chunking)
        # Stage artifacts are written only when a stage actually computes.
        # Under stage-major execution run() is re-entered once per stage, so a
        # later stage re-reads every earlier checkpoint; rewriting their output
        # each time re-emitted the same JSON, the same clips and the same
        # warnings four times over for a single file.
        computed = set()

        if checkpoint.exists("diarization"):
            if self.logger: self.logger.info("Loading Diarization from checkpoint")
            diarization_result = checkpoint.load("diarization")
        else:
            self._load("base")
            self._load("diarization")
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
            computed.add("diarization")

        if "diarization" in computed:
            stage_out.write_diarization(diarization_result.segments, audio_data.duration)


        self._free(args, "diarizer", "vad")
        # Unloading the client does not touch the subprocess holding the weights.
        self._release_worker(args, "diarizen")

        if getattr(args, "stop_after", None) == "diarization":
            if self.logger: self.logger.info("Stopping pipeline after diarization as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "diarization"})
            return None
            
        # 4. Speech Separation (Overlap)
        if checkpoint.exists("separation"):
            if self.logger: self.logger.info("Loading Separation from checkpoint")
            enhanced_segments = checkpoint.load("separation")
        else:
            # Both separated and failed clips are dumped next to this file's own
            # outputs so the QC thresholds can be judged by ear rather than from
            # counters. Built from output_dir, not from the audio's parent
            # directory: every file in a batch shares that parent, and the dump
            # filenames carry a timestamp and speaker pair but not the audio
            # name, so a shared directory means later files silently overwrite
            # earlier ones -- the collision _resolve_output_dir exists to avoid.
            if not self.step_enabled(args, "separation"):
                # Skipped, not failed: every segment still becomes an
                # EnhancedSegment carrying its slice of the mixture, so ASR and
                # export downstream see the same shape either way.
                if self.logger:
                    self.logger.info("Separation is off in the profile; segments pass through unseparated")
                enhanced_segments = self.separation_svc.passthrough(
                    diarization_result.segments, audio_data)
            else:
                self._load("separation")
                self.separation_svc.dump_dir = os.path.join(
                    output_dir, "03_separation", "audio", "raw")
                enhanced_segments = self.separation_svc.process_overlaps(diarization_result.segments, audio_data)
            if self.logger: self.logger.info(f"[DEBUG] After Separation: {len(enhanced_segments)} segments")
            checkpoint.save("separation", enhanced_segments)
            # The counters live on the service and reset_stats() clears them at
            # the start of every run(). Under stage-major execution the export
            # happens in a later run() than this one, so the report has to be
            # captured here or it is written empty.
            if hasattr(self.separation_svc, "report_payload"):
                checkpoint.save("separation_report",
                                self.separation_svc.report_payload())
            computed.add("separation")

        if "separation" in computed:
            stage_out.write_separation(
                enhanced_segments, audio_data.duration,
                report=self.separation_svc.report_payload()
                if hasattr(self.separation_svc, "report_payload") else None)
            stage_out.write_separated_audio(enhanced_segments, audio_data.sample_rate)

        self._free(args, "separator", "embedder")
        self._release_worker(args, "sidon")

        if getattr(args, "stop_after", None) == "separation":
            if self.logger: self.logger.info("Stopping pipeline after separation as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "separation"})
            return None
            
        # 5. Background Music Removal -- the fallback for runs with no map.
        #
        # When the sweep ran, the beds were already taken out of the waveform
        # before diarization, and every segment here is a slice of that cleaned
        # audio. Running this pass as well would hand a vocal separator its own
        # output and ask it to find a voice in it a second time -- and would
        # reload the separator to do it.
        # Off by default. It predates the waveform pass and does the same job a
        # stage later and one segment at a time, so with a map it is redundant
        # and without one it is a slower, coarser version of the same thing.
        # Kept behind a switch rather than deleted: a run with music_analysis
        # off has nothing else that would strip a bed.
        _has_map = self.step_enabled(args, "music_analysis") and bool(music_map)
        if _has_map or not self.step_enabled(args, "music_removal_fallback"):
            if self.logger:
                self.logger.info(
                    "Skipping the per-segment music pass "
                    + ("(the waveform was already cleaned)" if _has_map
                       else "(turned off in the profile)"))
        elif checkpoint.exists("music_removal"):
            if self.logger: self.logger.info("Loading Music Removal from checkpoint")
            enhanced_segments = checkpoint.load("music_removal")
        else:
            self._load("music")
            enhanced_segments = self.music_svc.process_segments(enhanced_segments, audio_data)
            if self.logger: self.logger.info(f"[DEBUG] After Music Removal: {len(enhanced_segments)} segments")
            checkpoint.save("music_removal", enhanced_segments)
            computed.add("music_removal")

        if "music_removal" in computed:
            stage_out.write_music_removal(enhanced_segments, audio_data.duration)

        self._free(args, "panns", "demucs")
            
        if getattr(args, "stop_after", None) == "music_removal":
            if self.logger: self.logger.info("Stopping pipeline after music_removal as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "music_removal"})
            return None
            
        # 6. ASR Ensemble (MoE)
        if checkpoint.exists("asr"):
            if self.logger: self.logger.info("Loading ASR from checkpoint")
            transcripts = checkpoint.load("asr")
        else:
            self._load("asr")
            if self.logger: self.logger.info(f"[DEBUG] Sending {len(enhanced_segments)} segments to ASR")
            transcripts = self.asr_svc.process(enhanced_segments, audio_data)
            if self.logger: self.logger.info(f"[DEBUG] ASR returned {len(transcripts)} transcripts")
            checkpoint.save("asr", transcripts)
            computed.add("asr")

        if "asr" in computed:
            stage_out.write_asr(transcripts)

        # The Qwen3 worker is done; the refinement LLM needs its ~4.7GB.
        self._release_worker(args, "qwen3")

        if getattr(args, "stop_after", None) == "asr":
            if self.logger: self.logger.info("Stopping pipeline after asr as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "asr"})
            return transcripts
            
        # 7. Qwen3-Omni Captioning
        if self.step_enabled(args, "captioning"):
            if checkpoint.exists("captioning"):
                if self.logger: self.logger.info("Loading Captioning from checkpoint")
                transcripts = checkpoint.load("captioning")
            else:
                self._load("caption")
                enhanced_audio_dict = {s.index: s.enhanced_audio for s in enhanced_segments if s.enhanced_audio is not None}
                transcripts = self.caption_svc.add_captions(transcripts, audio_data, enhanced_audio_dict)
                checkpoint.save("captioning", transcripts)

        # No release here on purpose. Qwen3OmniCaptioner is an HTTP client to a
        # server the pipeline does not start, not a model on this card, so it
        # holds nothing to free -- the same reason "qwen3" is absent from the
        # ASR stage's release list.
        
        if getattr(args, "stop_after", None) == "captioning":
            if self.logger: self.logger.info("Stopping pipeline after captioning as requested by --stop_after.")
            stage_out.write_manifest({"audio_file": os.path.basename(audio_path),
                                      "stopped_after": "captioning"})
            return transcripts
                
        # 8. LLM Refinement
        if self.step_enabled(args, "refinement"):
            # Checkpointed like every other stage. It is the most expensive one
            # -- 200 segments took 12 minutes on a T4, about half the run -- so
            # without this a retry after a failure anywhere downstream, or a
            # second pass over a finished file, pays for all of it again.
            if checkpoint.exists("refinement"):
                if self.logger: self.logger.info("Loading Refinement from checkpoint")
                transcripts = checkpoint.load("refinement")
            else:
                import copy
                # Keep the pre-refinement text so the stage can report exactly
                # what the model rewrote. A refinement pass that quietly
                # rewrites most of the corpus is a regression, and without this
                # it looks like success.
                before = copy.deepcopy(transcripts)
                # None keeps the service's built-in fusion prompt; only an
                # explicit override replaces it.
                transcripts = self.refinement_svc.refine(
                    transcripts, getattr(args, "llm_prompt", None) or None
                )
                checkpoint.save("refinement", transcripts)
                computed.add("refinement")
                stage_out.write_refinement(transcripts, before=before)

            # Export needs no GPU, and a following file needs this memory.
            if not getattr(args, "keep_models", False):
                self.refinement_svc.unload()

        
        # 9. Export Results
        save_path = output_dir
        os.makedirs(save_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        if hasattr(self.separation_svc, "write_report"):
            # Prefer the counters the separation stage checkpointed: this export
            # may be running in a later run() whose reset_stats() already
            # cleared the live ones. Falling back to None keeps the single-file
            # path, where separation ran moments ago, exactly as it was.
            self.separation_svc.write_report(
                save_path, base_name,
                payload=checkpoint.load("separation_report"))
        
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
        
        # A page for listening through the result and correcting it: the two
        # audio versions, all three ASR outputs, the fused text and an editable
        # copy, side by side. Built here because it needs the exported clips,
        # which only exist once the steps above have run.
        review_page = None
        if getattr(args, "review_page", True):
            try:
                from tools.make_review_page import build_review_page
                review_page = build_review_page(
                    save_path,
                    max_mb=getattr(args, "review_max_mb", None),
                    logger=self.logger)
            except Exception as e:
                # A missing review page must not fail a run whose transcripts
                # are already on disk.
                if self.logger:
                    self.logger.warning(f"Could not build the review page: {e}")

        # One index over every stage, written last so it can compare them.
        final = {"json": f"{base_name}.json", "srt": f"{base_name}.srt"}
        if review_page:
            final["review"] = os.path.basename(review_page)
        stage_out.write_manifest(metadata, extra={"final": final})

        if self.logger: self.logger.info(f"Pipeline completed successfully. Results saved to {save_path}")
        return transcripts
