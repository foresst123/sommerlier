"""Time the pipeline's steps without touching the code that runs them.

`install(monitor)` wraps the methods listed in TARGETS on their *classes* (not on
instances -- SeparationService is copied per file, and a wrapper stored on an
instance would keep calling the original object). Each call then records one
"span" (name, seconds, stage, file, thread, depth) with the PerformanceMonitor,
which writes it to a file. With no monitor installed the wrappers call straight
through, so installing is the only switch.

The batch loop names the stage and file a thread is working on with
`file_stage(stage, path)`; spans recorded inside inherit them.
"""

import contextlib
import functools
import importlib
import inspect
import os
import threading
import time

_monitor = None
_local = threading.local()
_installed = []      # (owner, attribute, original descriptor)

# (module, class or None for a module-level function, attribute, span name)
TARGETS = [
    ("services.audio_service", "AudioService", "load_audio", "audio.load"),
    ("services.music_service", "MusicService", "strip_music_spans", "music.strip_spans"),
    ("services.music_service", "MusicService", "strip_full_recording", "music.strip_full"),
    ("services.music_service", "MusicService", "apply_music_patches", "music.apply_patches"),
    ("services.diarization_service", "DiarizationService", "prepare_chunks", "diarization.prepare_chunks"),
    ("services.diarization_service", "DiarizationService", "diarize_raw", "diarization.diarize_raw"),
    ("services.diarization_service", "DiarizationService", "diarize_postprocess", "diarization.postprocess"),
    ("services.separation_service", "SeparationService", "prefetch_overlap_plan", "separation.build_windows"),
    ("services.separation_service", "SeparationService", "process_overlaps", "separation.process_overlaps"),
    ("services.separation_service", "SeparationService", "passthrough", "separation.passthrough"),
    ("services.asr_service", "ASRService", "process", "asr.process"),
    ("services.asr_service", "ASRService", "_prepare", "asr.prepare"),
    ("services.asr_service", "ASRService", "_transcribe_cross_file", "asr.inference_cross_file"),
    ("services.asr_service", "ASRService", "_transcribe_per_file", "asr.inference_per_file"),
    ("services.asr_service", "ASRService", "_vote", "asr.vote"),
    ("services.caption_service", "CaptionService", "add_captions", "captions.add"),
    ("services.diarization_refinement_service", "DiarizationRefinementService", "refine", "refinement.refine"),
    ("services.diarization_refinement_service", "DiarizationRefinementService", "_refine_queued", "refinement.queue"),
    ("services.diarization_refinement_service", "DiarizationRefinementService", "_refine_sequential", "refinement.sequential"),
    ("services.diarization_refinement_service", "DiarizationRefinementService", "_refine_batch", "refinement.inference_batch"),
    ("services.conversation_export_service", "ConversationExportService", "run", "conversation_exports.run"),
    ("services.conversation_export_service", "ConversationExportService", "_find_with_llm", "conversation_exports.find"),
    ("services.conversation_export_service", "ConversationExportService", "_judge", "conversation_exports.judge"),
    ("services.conversation_export_service", "ConversationExportService", "_plan_and_write", "conversation_exports.write_cpu"),
    ("services.clean_two_channel_dataset_service", "CleanTwoChannelDatasetService", "export", "clean_data.export"),
    ("services.export_service", "ExportService", "export_json", "export.json"),
    ("services.export_service", "ExportService", "export_srt", "export.srt"),
    ("services.export_service", "ExportService", "export_mp3_segments", "export.mp3_segments"),
    ("services.export_service", "ExportService", "export_separated_audio", "export.separated_audio"),
    ("services.model_loader", "ModelLoader", "load_base_models", "load_models.base"),
    ("services.model_loader", "ModelLoader", "load_diarization_models", "load_models.diarization"),
    ("services.model_loader", "ModelLoader", "load_separation_models", "load_models.separation"),
    ("services.model_loader", "ModelLoader", "load_tagger", "load_models.tagger"),
    ("services.model_loader", "ModelLoader", "load_music_models", "load_models.music"),
    ("services.model_loader", "ModelLoader", "load_asr_models", "load_models.asr"),
    ("services.model_loader", "ModelLoader", "load_caption_model", "load_models.caption"),
    ("services.model_loader", "ModelLoader", "unload", "unload_model"),
    ("services.pipeline_service", "PipelineService", "_ensure_worker", "ensure_worker"),
    ("services.pipeline_service", "PipelineService", "_release_worker", "release_worker"),
    ("services.pipeline_service", "PipelineService", "_free", "free_models"),
    ("services.pipeline_service", "PipelineService", "_reclaim_vram", "reclaim_vram"),
    ("services.pipeline_service", "PipelineService", "_strip_music", "music.strip"),
    ("services.pipeline_service", "PipelineService", "_measure_processed_noise", "music.measure_noise"),
    ("services.pipeline_service", "PipelineService", "_relabel_speakers", "relabel.run"),
    ("services.pipeline_service", "PipelineService", "_align_words", "word_alignment.run"),
    ("services.pipeline_service", "PipelineService", "_export_conversation_exports", "conversation_exports.write"),
    ("services.pipeline_service", "PipelineService", "_export_clean_two_channel_dataset", "clean_data.write"),
    ("services.pipeline_service", None, "build_maps", "music.analyse_maps"),
    ("services.pipeline_service", None, "excise", "music.excise"),
]

# Spans that name a worker: which positional argument is its name.
_WORKER_ARG = {"ensure_worker": 1, "release_worker": 2}


def _context():
    if not hasattr(_local, "stage"):
        _local.stage, _local.file, _local.depth = None, None, 0
    return _local


def _describe(name, args):
    index = _WORKER_ARG.get(name)
    if index is not None and len(args) > index and isinstance(args[index], str):
        return {"worker": args[index]}
    return {}


def _wrap(function, name):
    @functools.wraps(function)
    def timed(*args, **kwargs):
        monitor = _monitor
        if monitor is None:
            return function(*args, **kwargs)
        ctx = _context()
        depth, ctx.depth = ctx.depth, ctx.depth + 1
        started = time.perf_counter()
        error = False
        try:
            return function(*args, **kwargs)
        except BaseException:
            error = True
            raise
        finally:
            ctx.depth = depth
            fields = _describe(name, args)
            if error:
                fields["error"] = True
            monitor.record_span(name, time.perf_counter() - started, stage=ctx.stage,
                                file=ctx.file, thread=threading.current_thread().name,
                                depth=depth, **fields)
    timed._profiled = True
    return timed


def current_file():
    """Basename of the file this thread is working on, or None."""
    return _context().file


@contextlib.contextmanager
def file_stage(stage, path):
    """Mark this thread as running `stage` on `path`; records one span for it.

    The marker is set even when no monitor is installed, so `current_file()`
    (used to label progress lines) works without profiling.
    """
    monitor = _monitor
    ctx = _context()
    previous = (ctx.stage, ctx.file, ctx.depth)
    ctx.stage, ctx.file, ctx.depth = stage, os.path.basename(str(path)), 0
    started = time.perf_counter()
    error = False
    try:
        yield
    except BaseException:
        error = True
        raise
    finally:
        if monitor is not None:
            fields = {"error": True} if error else {}
            monitor.record_span("file_stage", time.perf_counter() - started, stage=stage,
                                file=ctx.file, thread=threading.current_thread().name,
                                **fields)
        ctx.stage, ctx.file, ctx.depth = previous


def install(monitor):
    """Start timing. Safe to call twice; modules that cannot be imported are skipped."""
    global _monitor
    if _installed:
        _monitor = monitor
        return
    for module_name, class_name, attribute, name in TARGETS:
        try:
            module = importlib.import_module(module_name)
            owner = getattr(module, class_name) if class_name else module
            original = inspect.getattr_static(owner, attribute)
        except (ImportError, AttributeError):
            continue
        if isinstance(original, staticmethod):
            wrapped = staticmethod(_wrap(original.__func__, name))
        elif inspect.isfunction(original):
            wrapped = _wrap(original, name)
        else:
            continue
        setattr(owner, attribute, wrapped)
        _installed.append((owner, attribute, original))
    _monitor = monitor


def uninstall():
    """Undo install(): the original methods come back. Used by tests."""
    global _monitor
    _monitor = None
    while _installed:
        owner, attribute, original = _installed.pop()
        setattr(owner, attribute, original)
