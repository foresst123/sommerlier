"""Splitting a two-speaker mixture, and working out which voice is whose.

Two steps that are worth keeping distinct, because only one of them is
model-specific.

The separator produces two tracks. The one the profiles name is DialogueSidon,
which is *blind*: it is never told who is in the mixture, so the tracks come
back in whatever order it chose. Everything else in this module exists because
of that -- WeSpeaker embeds each track, scores it against the enrollments mined for
each speaker, and assigns them; `_repair_chunk_swaps` catches the separator
changing its mind about channel order mid-file. Similarity now reports
confidence rather than rejecting audio that is otherwise usable.

USEF-TFGridNet used to sit behind the same interface and is gone. It was
target-conditioned, returned its tracks already ordered, and skipped the
assignment entirely -- and while it was there the shared constants drifted to
suit it, which left Sidon running on a 2s window with no solo audio to score
against. The `ordered` flag survives on the base backend for a future
conditioned model, but nothing sets it now.

Sidon is also generative, which is a property of the corpus and not of this
code: what it returns is audio the model produced, not audio the microphone
recorded. See models/separation_backends.py and doc/audio-cleanliness.md.
"""
import os
import sys
import collections
import copy
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import librosa
from typing import Dict, List, Tuple, Union, Optional
import torch.nn.functional as F

# Sidon advances by CHUNK_SECONDS - OVERLAP_SECONDS (20 - 5) between chunks, so
# a seam falls every 15s, not every 20s. Walking the repair on 20s blocks put
# the boundaries out of phase with the seams and left a block holding both a
# correct and an inverted stretch, which one flip cannot fix.
STITCH_CHUNK_SEC = 15.0
# Correlation advantage the swapped ordering must show before a block is
# flipped, so near-ties on quiet audio are left alone.
STITCH_SWAP_MARGIN = 0.05


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Zero-mean correlation of two equal-length signals, 0.0 when either is flat."""
    if x.size == 0 or y.size == 0:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


# Below this RMS a probe region carries no speech, only the separator's noise
# floor. Measured: a correctly-empty track sat at 3e-5 while a track holding the
# speaker sat at 1.2e-1, four orders of magnitude apart.
ABS_SILENCE_RMS = 1e-3

# Voiced audio WeSpeaker needs before its embedding is worth comparing. It pools
# statistics over time, so a shorter probe gives a noisier vector -- and a noisy
# vector competes in the A/B assignment, where being wrong flips which speaker a
# track is labelled as.
#
# This applies to the solo regions the assignment is scored on, not to the
# overlap being spliced: a 0.24s backchannel is never measured here. The
# stitched window contributes BSS_STITCH_SOLO (3s) per speaker, so 1s leaves
# room for the silence filter to drop pauses without failing the whole probe,
# while still asking for enough audio to trust the score.
BSS_MIN_VOICED_SEC = float(os.environ.get("BSS_MIN_VOICED_SEC", "1.0"))


class _AssignmentBatchPolicy:
    """One-way batch/fan-out decision shared by every file in the stage.

    ``warmup_requests`` counts embeddable similarity probes, not windows.  A
    probe only counts as batched when the worker reports that it actually ran
    in a multi-item ONNX call.  Once the observed hit rate drops below the
    threshold the stage stays in fan-out mode; this avoids oscillating between
    two scheduling strategies while files are already in flight.
    """

    def __init__(self, warmup_requests=20, min_hit_rate=0.5):
        self.warmup_requests = max(1, int(warmup_requests or 20))
        self.min_hit_rate = min(1.0, max(0.0, float(min_hit_rate)))
        self.total_items = 0
        self.batched_items = 0
        self.fanout = False
        self._lock = threading.Lock()

    def should_batch(self):
        with self._lock:
            return not self.fanout

    def observe(self, batched_items, total_items):
        total_items = max(0, int(total_items or 0))
        batched_items = min(total_items, max(0, int(batched_items or 0)))
        with self._lock:
            switched = False
            if total_items and not self.fanout:
                self.total_items += total_items
                self.batched_items += batched_items
                if self.total_items >= self.warmup_requests:
                    hit_rate = self.batched_items / self.total_items
                    if hit_rate < self.min_hit_rate:
                        self.fanout = True
                        switched = True
            hit_rate = (self.batched_items / self.total_items
                        if self.total_items else 0.0)
            return {
                "total_items": self.total_items,
                "batched_items": self.batched_items,
                "hit_rate": hit_rate,
                "fanout": self.fanout,
                "switched": switched,
            }


class BssSeparator:
    """Blind source separation, then WeSpeaker speaker assignment.

    The backend named in the profile splits the mixture; WeSpeaker decides
    which of the two tracks belongs to which speaker. If only one identity can
    be scored, its track is selected and the remaining label goes to the other
    track. A target-conditioned backend declares `ordered = True`.
    """
    
    def __init__(self, device: torch.device, process=None, checkpoint_path: str = None,
                 separator: str = None, embedding_repository: str = None,
                 embedding_filename: str = None, embedding_revision: str = None,
                 logger=None, embedding_threads: int = None, score_workers: int = 1,
                 assignment_process=None, assignment_batching: bool = False,
                 assignment_batch_warmup_requests: int = 20,
                 assignment_batch_min_hit_rate: float = 0.5):
        import tempfile

        from models.separation_backends import make_backend

        self.device = device
        self._process = process
        self.speaker_embedder = None
        self.target_embed_cache: Dict[str, torch.Tensor] = {}
        self._target_locks = {}
        self._target_locks_guard = threading.Lock()
        self.assignment_batching = bool(assignment_batching)
        self._assignment_batch_policy = _AssignmentBatchPolicy(
            assignment_batch_warmup_requests, assignment_batch_min_hit_rate)
        self._temp_dir = tempfile.mkdtemp(prefix="bss_exchange_")
        self._req_counter = 0
        self._logger = logger
        self._embedding_threads = embedding_threads or None
        # When set, Silero VAD and WeSpeaker run in separate worker processes
        # (assignment_worker.py) reached through this pool, and nothing is loaded
        # here. See _remote().
        self._assignment = assignment_process
        self._remote_counter = itertools.count()
        # Seconds spent per assignment phase, summed over every window (and over
        # the parallel scoring tasks, so it can exceed wall time). Shared by the
        # per-file clones made in fork(). Read by SeparationService._report_stats.
        self.timing = collections.Counter()
        self._timing_lock = threading.Lock()
        # The four probe embeddings that score a window are independent of each
        # other; with score_workers > 1 they run at the same time.
        self.score_workers = max(1, int(score_workers or 1))
        self._score_pool = (ThreadPoolExecutor(
            max_workers=self.score_workers, thread_name_prefix="bss-score")
            if self.score_workers > 1 else None)

        # Which separator produces the two tracks. Everything else in this
        # class -- enrollment embeddings, QC scoring, the not-A test -- is the
        # same whichever one runs, which is what makes them comparable.
        name = separator or os.environ.get("BSS_SEPARATOR", "sidon")
        self._separator_name = name
        self.backend = make_backend(name, process=process, temp_dir=self._temp_dir,
                                    device=device, logger=logger)
        self._share_timing_with_backend(self.backend)

        self._vad = None
        self._vad_lock = threading.Lock()
        if self._assignment is not None:
            if logger:
                logger.info("[BSS] speaker assignment (Silero + WeSpeaker) runs in "
                            "worker processes")
            return

        self._load_model(embedding_repository, embedding_filename, embedding_revision)

        # Silero VAD to keep only real speech in each probe before scoring.
        # Energy alone cannot tell a separator's residual noise from voice, so
        # a track that should be silent here can still yield an embedding built
        # from noise and invert the A/B assignment. Silero judges voice, not
        # loudness. Falls back to the energy gate if it cannot load.
        try:
            from models.silero_vad import SileroVAD
            self._vad = SileroVAD(device=self.device)
            if logger:
                logger.info("[BSS] probe filtering: Silero VAD")
        except Exception as exc:
            print(f"[BSS] Silero VAD unavailable ({exc}); using energy gate",
                  file=sys.stderr)

    def fork(self):
        """Create file-local assignment state while sharing heavy embedders.

        DialogueSidon itself lives in the external worker pool. The clone owns
        only a request backend, scratch directory and speaker cache; WeSpeaker
        and Silero weights stay shared in the main process.
        """
        import tempfile
        from models.separation_backends import make_backend

        clone = copy.copy(self)
        clone.target_embed_cache = {}
        clone._target_locks = {}
        clone._target_locks_guard = threading.Lock()
        clone._temp_dir = tempfile.mkdtemp(prefix="bss_exchange_")
        clone._req_counter = 0
        clone.backend = make_backend(
            self._separator_name, process=self._process,
            temp_dir=clone._temp_dir, device=self.device, logger=self._logger)
        self._share_timing_with_backend(clone.backend)
        return clone

    def _share_timing_with_backend(self, backend):
        """Let the backend add its timings to this separator's counter."""
        if hasattr(backend, "_note_timing"):
            backend.timing, backend.timing_lock = self.timing, self._timing_lock

    @property
    def process(self):
        return self._process

    @process.setter
    def process(self, proc):
        """Re-point at a worker that was restarted between files.

        The pipeline releases the worker at the end of the stage and starts a
        fresh one for the next file, so the process this was built with is dead
        by then. Setting it here without telling the backend left the backend
        holding the corpse: every job after the first file failed on a pipe
        nobody was reading.
        """
        self._process = proc
        setter = getattr(self.backend, "set_process", None)
        if setter:
            setter(proc)

    def reset_speakers(self):
        """Forget the cached enrollment embeddings.

        The cache is keyed by the diarizer's speaker label -- "1", "2" -- and
        those labels restart at 1 for every file. Carried across a batch, the
        second file's speaker "1" hits the first file's entry and every track in
        it gets scored against a stranger's voice: similarities collapse, the
        A/B assignment inverts, and QC rejects work that was fine. It only shows
        up from the second file onwards, which is why a single-file run looks
        healthy.
        """
        self.target_embed_cache.clear()

    def close(self):
        """Remove the scratch directory used to exchange arrays with the worker."""
        import shutil
        shutil.rmtree(self._temp_dir, ignore_errors=True)
        
    def _load_model(self, repository=None, filename=None, revision=None):
        print(f"[TSE Model] Initializing WeSpeaker on {self.device}...")

        try:
            from models.wespeaker_embedding import WeSpeakerONNXEmbedder
            kwargs = {"device": self.device}
            if getattr(self, "_embedding_threads", None):
                kwargs["threads"] = self._embedding_threads
            if repository:
                kwargs["repository"] = repository
            if filename:
                kwargs["filename"] = filename
            if revision:
                kwargs["revision"] = revision
            self.speaker_embedder = WeSpeakerONNXEmbedder(**kwargs)
            print("[TSE Model] WeSpeaker ResNet293-LM is ready.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize WeSpeaker: {e}")
            
    def _tick(self, key, started):
        with self._timing_lock:
            self.timing[key] += time.perf_counter() - started
            self.timing[key + "_calls"] += 1

    def _run_scores(self, tasks):
        """Call every task and return the results in task order.

        Concurrent when score_workers > 1; the outcome never depends on it.
        """
        if getattr(self, "_score_pool", None) is None or len(tasks) < 2:
            return [task() for task in tasks]
        futures = [self._score_pool.submit(task) for task in tasks]
        return [future.result() for future in futures]

    def _remote(self, cmd, audio, sample_rate, **options):
        """Ask an assignment worker to do `cmd` on `audio`; returns its reply.

        Audio travels as a .npy in this separator's scratch directory (one
        per request, so parallel calls do not collide). The pool hands the
        request to whichever worker is idle.
        """
        tag = f"{os.getpid()}_{next(self._remote_counter)}_{threading.get_ident()}"
        audio_path = os.path.join(self._temp_dir, f"assign_{tag}.npy")
        out_path = os.path.join(self._temp_dir, f"assign_{tag}_out.npy")
        np.save(audio_path, np.asarray(audio, dtype=np.float32))
        try:
            reply = self._assignment.request(
                {"cmd": cmd, "id": tag, "audio_path": audio_path,
                 "out_path": out_path, "sr": int(sample_rate), **options},
                response_id=tag)
            if reply.get("error"):
                raise RuntimeError(f"assignment worker: {reply['error']}")
            if reply.get("probe_path"):
                reply["probe"] = np.load(reply["probe_path"])
            return reply
        finally:
            for path in (audio_path, out_path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _embedding_from_reply(self, reply):
        values = reply.get("embedding")
        if values is None:
            return None
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _get_embedding(self, audio_array: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """Helper to get speaker embedding from 1D numpy array."""
        if self._assignment is not None:
            _t = time.perf_counter()
            embedding = self._embedding_from_reply(
                self._remote("embed", audio_array, sample_rate))
            self._tick("remote", _t)
            return embedding
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
        if self.speaker_embedder is None:
            raise RuntimeError("WeSpeaker is not loaded.")
        return self.speaker_embedder.embed(audio_array, sample_rate)

    def _get_embeddings(self, audios, sample_rate):
        """Batch embeddings with the exact preprocessing of _get_embedding."""
        if self._assignment is not None:
            return self._remote_embeddings("embed_batch", audios, sample_rate)
        results = [None] * len(audios)
        prepared, indices = [], []
        for index, audio in enumerate(audios):
            try:
                if sample_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                prepared.append(audio)
                indices.append(index)
            except Exception as exc:
                results[index] = exc
        if prepared:
            values = self.speaker_embedder.embed_batch(prepared, sample_rate)
            if len(values) != len(indices):
                raise RuntimeError("WeSpeaker batch result count mismatch")
            for index, value in zip(indices, values):
                results[index] = value
        return results

    def _remote_embeddings(self, command, audios, sample_rate, **options):
        values, _ = self._remote_embeddings_with_stats(
            command, audios, sample_rate, **options)
        return values

    def _remote_embeddings_with_stats(self, command, audios, sample_rate, **options):
        if not audios:
            return [], {}
        started = time.perf_counter()
        try:
            reply = self._remote(
                command, np.concatenate(audios), sample_rate,
                lengths=[len(audio) for audio in audios], **options)
            rows = reply.get("results", [])
            if len(rows) != len(audios):
                raise RuntimeError("assignment worker batch result count mismatch")
            stats = reply.get("batch_stats", {})
            with self._timing_lock:
                for key, value in stats.items():
                    self.timing["embedding_" + key] += value
            return ([RuntimeError(row["error"]) if row.get("error") else
                     self._embedding_from_reply(row) for row in rows], stats)
        finally:
            self._tick("remote", started)

    def _get_target_embedding(self, enrollment_audios: List[np.ndarray], target_id: str, sample_rate: int) -> Optional[torch.Tensor]:
        """Calculate and cache the target embedding. Returns None if no audios provided."""
        # Assignment-ahead may ask for the same speaker from several windows.
        # Only one computes its centroid; other speakers remain independent.
        if target_id and hasattr(self, "_target_locks_guard"):
            with self._target_locks_guard:
                lock = self._target_locks.setdefault(target_id, threading.Lock())
            with lock:
                return self._compute_target_embedding(enrollment_audios, target_id, sample_rate)
        return self._compute_target_embedding(enrollment_audios, target_id, sample_rate)

    def _compute_target_embedding(self, enrollment_audios, target_id, sample_rate):
        if target_id and target_id in self.target_embed_cache:
            return self.target_embed_cache[target_id]
            
        enroll_embeddings = []
        audios = [e for e in enrollment_audios if len(e) > 0]
        batched = None
        if getattr(self, "assignment_batching", False):
            try:
                batched = self._get_embeddings(audios, sample_rate)
            except Exception as exc:
                # Preserve the individual retry path for a transport/session
                # failure, just as when a single enrollment request fails.
                print(f"[TSE] enrollment batch failed; retrying individually: {exc}",
                      file=sys.stderr)
        for index, e in enumerate(audios):
            if len(e) > 0:
                # Normalize each clip before averaging: raw speaker embeddings have
                # length-dependent norms, so the longest clip would otherwise
                # dominate the centroid.
                try:
                    value = batched[index] if batched is not None else self._get_embedding(e, sample_rate)
                    if isinstance(value, Exception):
                        raise value
                    embedding = F.normalize(value, p=2, dim=0)
                    if torch.isfinite(embedding).all() and torch.linalg.vector_norm(embedding) > 0:
                        enroll_embeddings.append(embedding)
                except Exception as exc:
                    print(
                        f"[TSE] Warning: enrollment for {target_id} is not "
                        f"embeddable ({type(exc).__name__}: {exc})",
                        file=sys.stderr,
                    )
                
        if not enroll_embeddings:
            print(f"[TSE] Warning: No valid enrollment audios provided for target {target_id}")
            return None
            
        target_embed = torch.stack(enroll_embeddings).mean(dim=0)
        if not torch.isfinite(target_embed).all() or torch.linalg.vector_norm(target_embed) == 0:
            return None
        target_embed = F.normalize(target_embed, p=2, dim=0)
        
        if target_id:
            self.target_embed_cache[target_id] = target_embed
            
        return target_embed

    def _repair_chunk_swaps(self, track_1: np.ndarray, track_2: np.ndarray, sr: int,
                            embed_A, embed_B, chunk_sec: float = STITCH_CHUNK_SEC):
        """Undo channel inversions the separator introduced between its chunks.

        Sidon separates in fixed-length chunks and keeps channel order across
        them by correlating their overlap. When that overlap holds only one
        voice, the correlation cannot tell the orderings apart and every chunk
        after the bad seam comes back inverted -- a track that is clean
        everywhere yet carries the wrong speaker in part of its span, which the
        A/B assignment cannot express because it picks one orientation for the
        whole window.

        Correlating the stitched output against itself does not detect this:
        different blocks hold different words, so their correlation is ~0
        regardless of ordering (measured: |r| < 0.03 either way). Score each
        block against the two enrollment embeddings instead and keep the
        orientation that matches them, which is what the speakers' identity --
        not their waveform -- actually distinguishes.

        Returns the repaired pair and the number of blocks flipped.
        """
        n = min(len(track_1), len(track_2))
        block = max(1, int(chunk_sec * sr))
        if n <= block or embed_A is None or embed_B is None:
            return track_1, track_2, 0

        out_1 = track_1[:n].copy()
        out_2 = track_2[:n].copy()
        flips = 0

        for start in range(0, n, block):
            end = min(start + block, n)
            seg_1, seg_2 = out_1[start:end], out_2[start:end]

            try:
                e1 = self._block_embedding(seg_1, sr)
                e2 = self._block_embedding(seg_2, sr)
            except Exception as exc:
                print(f"[TSE] chunk {start}:{end} unscorable: {exc}", file=sys.stderr)
                continue
            if e1 is None or e2 is None:
                # One side is silent here, so this block says nothing about
                # ordering. Leaving it alone is right: flipping on a missing
                # score is how the noise-probe bug inverted assignments.
                continue

            direct = float(torch.dot(embed_A, e1)) + float(torch.dot(embed_B, e2))
            swapped = float(torch.dot(embed_A, e2)) + float(torch.dot(embed_B, e1))
            if swapped > direct + STITCH_SWAP_MARGIN:
                out_1[start:end] = track_2[start:end]
                out_2[start:end] = track_1[start:end]
                flips += 1

        return out_1, out_2, flips

    def _block_embedding(self, block: np.ndarray, sr: int):
        """Normalized WeSpeaker embedding for one block, or None if it holds no speech."""
        probe = self._gather_probe(block, [(0, len(block))], sr)
        if probe is None:
            return None
        return F.normalize(self._get_embedding(probe, sr), p=2, dim=0)

    @staticmethod
    def _xfade_join(pieces, sr, overlap_sec: float = 0.02):
        """Concatenate voiced pieces with a short cross-fade at each join.

        A butt-join between two separately-cut speech pieces leaves a step
        discontinuity -- an edge speaker embedding reads as a transient. Overlap-adding a
        few ms with a raised-cosine ramp smooths it. The overlap is only at the
        seam, so at most a few ms of one piece's tail blends into the next
        piece's head; it never swallows a whole word. Pieces shorter than the
        overlap are appended whole.
        """
        pieces = [p for p in pieces if p is not None and p.size]
        if not pieces:
            return None
        ov = max(1, int(overlap_sec * sr))
        out = pieces[0].astype(np.float32, copy=True)
        for nxt in pieces[1:]:
            nxt = nxt.astype(np.float32)
            if out.size >= ov and nxt.size >= ov:
                ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ov, dtype=np.float32)))
                out[-ov:] = out[-ov:] * (1.0 - ramp) + nxt[:ov] * ramp
                out = np.concatenate([out, nxt[ov:]])
            else:
                out = np.concatenate([out, nxt])
        return out if out.size else None

    def _gather_probe(self, track: np.ndarray, spans, sr: int, floor_db: float = -40.0,
                      min_voiced_sec: float = None, abs_floor_rms: float = ABS_SILENCE_RMS):
        """Keep only real speech from `spans` of `track`, joined with a seam-only
        cross-fade, and return it for scoring.

        Silero VAD decides what is voice; an energy gate cannot tell a
        separator's residual noise from speech, so a track that should be silent
        here would otherwise pass and its noise embedding could invert the A/B
        assignment. Voiced runs are cross-faded at the join, not butted, so no
        step discontinuity reaches WeSpeaker. Falls back to the energy gate when
        Silero is unavailable or errors.

        Returns None when there is too little voiced audio to trust, which means
        "this speaker is not present here" -- distinct from "extraction failed".
        """
        seg = self._cut_spans(track, spans)
        if seg is None:
            return None
        if self._assignment is not None:
            _t = time.perf_counter()
            reply = self._remote("probe", seg, sr, floor_db=floor_db,
                                 min_voiced_sec=min_voiced_sec,
                                 abs_floor_rms=abs_floor_rms)
            self._tick("remote", _t)
            return reply.get("probe")
        return self._probe_from_segment(seg, sr, floor_db, min_voiced_sec, abs_floor_rms)

    @staticmethod
    def _cut_spans(track: np.ndarray, spans):
        """The audio of `spans` joined end to end, or None when there is none."""
        if not spans:
            return None
        pieces = [track[max(0, a):min(len(track), b)] for a, b in spans]
        pieces = [pc for pc in pieces if pc.size]
        if not pieces:
            return None
        return np.concatenate(pieces).astype(np.float32)

    def _probe_embedding(self, track: np.ndarray, spans, sr: int, floor_db: float = -40.0,
                         min_voiced_sec: float = None,
                         abs_floor_rms: float = ABS_SILENCE_RMS):
        """Voiced probe of `spans`, embedded; None when there is no probe.

        Locally this is _gather_probe then _get_embedding. With assignment
        workers it is one round trip, so the probe never travels back.
        """
        if self._assignment is None:
            _t = time.perf_counter()
            probe = self._gather_probe(track, spans, sr, floor_db, min_voiced_sec,
                                       abs_floor_rms)
            self._tick("probe_vad", _t)
            if probe is None:
                return None
            _t = time.perf_counter()
            embedding = F.normalize(self._get_embedding(probe, sr), p=2, dim=0)
            self._tick("wespeaker", _t)
            return embedding
        seg = self._cut_spans(track, spans)
        if seg is None:
            return None
        if min_voiced_sec is None:
            min_voiced_sec = BSS_MIN_VOICED_SEC
        _t = time.perf_counter()
        reply = self._remote("probe_embed", seg, sr, floor_db=floor_db,
                             min_voiced_sec=min_voiced_sec, abs_floor_rms=abs_floor_rms)
        self._tick("remote", _t)
        embedding = self._embedding_from_reply(reply)
        return None if embedding is None else F.normalize(embedding, p=2, dim=0)

    def _batch_policy(self):
        policy = getattr(self, "_assignment_batch_policy", None)
        if policy is None:
            # Protocol tests construct BssSeparator without __init__. Production
            # always takes the configured values above.
            policy = _AssignmentBatchPolicy()
            self._assignment_batch_policy = policy
        return policy

    def _note_batch_policy(self, snapshot, batched_items, observed_items):
        with self._timing_lock:
            self.timing["adaptive_batch_checks"] += 1
            self.timing["adaptive_batch_observed_items"] += observed_items
            self.timing["adaptive_batch_batched_items"] += min(observed_items,
                                                                 batched_items)
            if snapshot["switched"]:
                self.timing["adaptive_batch_switches"] += 1
        if snapshot["switched"] and getattr(self, "_logger", None):
            self._logger.info(
                "[BSS] similarity batching hit-rate %.1f%% after %d probes; "
                "using assignment fan-out for the rest of the stage",
                100.0 * snapshot["hit_rate"], snapshot["total_items"])

    @staticmethod
    def _normalized_embedding(value):
        if value is None or isinstance(value, Exception):
            return value
        return F.normalize(value, p=2, dim=0)

    def _probe_embeddings_fanout(self, probes, sr):
        """Run complete VAD+embedding requests independently across workers."""
        def task(track, spans):
            def run():
                try:
                    return self._probe_embedding(track, spans, sr)
                except Exception as exc:
                    return exc
            return run

        with self._timing_lock:
            self.timing["adaptive_fanout_items"] += len(probes)
        return self._run_scores([task(track, spans) for track, spans in probes])

    def _probe_embeddings_adaptive(self, probes, sr):
        """VAD first, batch exact-length probes and fan out every singleton.

        No waveform is padded or cropped. Different post-VAD lengths are sent
        as independent ``embed`` requests, which lets the worker pool schedule
        them on different assignment processes immediately.
        """
        results = [None] * len(probes)

        def prepare(track, spans):
            def run():
                try:
                    return self._gather_probe(track, spans, sr)
                except Exception as exc:
                    return exc
            return run

        prepared = self._run_scores([
            prepare(track, spans) for track, spans in probes])
        groups = collections.defaultdict(list)
        for index, audio in enumerate(prepared):
            if isinstance(audio, Exception):
                results[index] = audio
            elif audio is not None:
                # At a fixed sample rate, equal sample counts produce equal
                # WeSpeaker feature shapes. Exact grouping avoids padding and
                # therefore preserves similarity quality bit-for-bit.
                groups[len(audio)].append((index, audio))

        jobs = []
        for items in groups.values():
            if len(items) > 1:
                def batch_job(items=items):
                    try:
                        values, stats = self._remote_embeddings_with_stats(
                            "embed_batch", [audio for _, audio in items], sr)
                    except Exception as exc:
                        values, stats = [exc] * len(items), {}
                    return items, values, stats
                jobs.append(batch_job)
            else:
                def single_job(items=items):
                    try:
                        value = self._get_embedding(items[0][1], sr)
                    except Exception as exc:
                        value = exc
                    return items, [value], {}
                jobs.append(single_job)

        batched_items = 0
        for items, values, stats in self._run_scores(jobs):
            batched_items += int(stats.get("batched_items", 0))
            for (index, _), value in zip(items, values):
                results[index] = self._normalized_embedding(value)

        observed_items = sum(len(items) for items in groups.values())
        snapshot = self._batch_policy().observe(batched_items, observed_items)
        self._note_batch_policy(snapshot, batched_items, observed_items)
        return results

    def _probe_embeddings(self, probes, sr):
        """Gather independent probes and return normalized tensors/errors/None."""
        if self._assignment is not None:
            if not self._batch_policy().should_batch():
                return self._probe_embeddings_fanout(probes, sr)
            return self._probe_embeddings_adaptive(probes, sr)

        results = [None] * len(probes)
        audios, indices = [], []
        for index, (track, spans) in enumerate(probes):
            try:
                audio = self._gather_probe(track, spans, sr)
                if audio is not None:
                    audios.append(audio)
                    indices.append(index)
            except Exception as exc:
                results[index] = exc
        if not audios:
            return results
        try:
            values = self._get_embeddings(audios, sr)
        except Exception as exc:
            values = [exc] * len(audios)
        for index, value in zip(indices, values):
            results[index] = self._normalized_embedding(value)
        return results

    def _score_probe_batch(self, requests, sr, full_span, sources, errors):
        """Same clean-probe/full-context fallback, with one embedding per input.

        The normalized probe is independent of the speaker it is compared to.
        Reuse it when A and B ask for the same track and sample ranges.
        """
        scores = [None] * len(requests)
        cache = {}
        for fallback in (False, True):
            pending, unique = [], {}
            failed_inputs = set()
            for index, (track, spans, target, key) in enumerate(requests):
                if target is None or scores[index] is not None or (fallback and not spans):
                    continue
                candidate = full_span if fallback or not spans else spans
                source = "full_context" if fallback or not spans else "clean_probe"
                identity = (id(track), tuple(tuple(span) for span in candidate))
                pending.append((index, target, key, source, identity))
                if identity not in cache:
                    unique[identity] = (track, candidate)
            if unique:
                values = self._probe_embeddings(list(unique.values()), sr)
                cache.update(zip(unique, values))
            for index, target, key, source, identity in pending:
                try:
                    embedding = cache[identity]
                    if isinstance(embedding, Exception):
                        raise embedding
                    if embedding is None:
                        continue
                    score = float(torch.dot(target, embedding))
                    if not np.isfinite(score):
                        raise ValueError("nonfinite similarity")
                except Exception as exc:
                    errors[f"{key}:{source}"] = f"{type(exc).__name__}: {exc}"
                    # A failed attempt must not suppress the ordinary retry
                    # when clean_probe and full_context happen to be identical.
                    failed_inputs.add(identity)
                    continue
                scores[index] = score
                sources[key] = source
            for identity in failed_inputs:
                cache.pop(identity, None)
        for score, (_, _, target, key) in zip(scores, requests):
            if score is None and target is not None:
                sources[key] = "unscorable"
        return scores

    def _probe_from_segment(self, seg: np.ndarray, sr: int, floor_db: float = -40.0,
                            min_voiced_sec: float = None,
                            abs_floor_rms: float = ABS_SILENCE_RMS):
        """The voiced part of `seg` (already cut from the track), or None."""
        if min_voiced_sec is None:
            min_voiced_sec = BSS_MIN_VOICED_SEC

        # --- Silero path: keep exactly the voiced runs, cross-fade the joins ---
        vad = getattr(self, "_vad", None)
        if vad is not None and seg.size >= int(0.10 * sr):
            try:
                with self._vad_lock:
                    ts = vad.get_speech_timestamps(seg, sampling_rate=sr)
            except Exception:
                ts = None
            if ts:
                voiced = [seg[t["start"]:t["end"]] for t in ts]
                joined = self._xfade_join(voiced, sr)
                if joined is not None and joined.size >= int(min_voiced_sec * sr):
                    return joined
                return None          # some speech, but not enough to trust
            elif ts == []:
                return None          # Silero is confident there is no speech here
            # ts is None: Silero errored -> fall through to the energy gate

        # --- energy gate (fallback) ---
        frame = max(1, int(0.02 * sr))
        if seg.size < frame * 2:
            min_samples = int(min_voiced_sec * sr)

            if seg.size < min_samples:
                return None

            rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))
            if rms < abs_floor_rms:
                return None

            return seg
        n = seg.size // frame
        frames = seg[: n * frame].reshape(n, frame)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)

        # An absolute floor first. The relative test below rescales by the
        # track's own 95th percentile, so a track that is silent here -- because
        # the separator correctly put this speaker on the *other* track -- still
        # passes every frame and yields an embedding built from noise. That
        # score then competes in the A/B assignment and can invert it.
        ref_abs = float(np.percentile(rms, 95))
        if ref_abs < abs_floor_rms:
            return None

        ref = ref_abs + 1e-12
        keep = 20.0 * np.log10(rms / ref) > floor_db
        if keep.sum() * frame < min_voiced_sec * sr:
            return None
        return frames[keep].reshape(-1)

    def separate_raw(self, mixture_audio: np.ndarray, sample_rate: int = 16000,
                     enroll_A=None, enroll_B=None):
        """Run only the blind separator so GPU work can be prefetched."""
        if len(mixture_audio) == 0:
            raise ValueError("Input mixture_audio is empty.")
        return self.backend.separate(
            mixture_audio, sample_rate, enroll_A=enroll_A, enroll_B=enroll_B)

    def postprocess_separated(self, mixture_audio: np.ndarray, raw_tracks,
                              enroll_A: List[np.ndarray], enroll_B: List[np.ndarray],
                              sample_rate: int = 16000, id_A: Optional[str] = None,
                              id_B: Optional[str] = None,
                              probe_A: Optional[List[Tuple[int, int]]] = None,
                              probe_B: Optional[List[Tuple[int, int]]] = None,
                              core_range: Optional[Tuple[int, int]] = None):
        """Map a raw separator result onto speakers A and B.

        probe_A / probe_B: (start, end) sample ranges within mixture_audio where
        that speaker is known to speak ALONE. Assignment and the returned
        similarities are measured there.

        core_range: overlap core in mixture samples, used only by the "not-A"
        relative test for a speaker that has no solo region.

        Returns (track_A, track_B, sim_A, sim_B, diag).
        """
        if not self.speaker_embedder and self._assignment is None:
            raise RuntimeError("WeSpeaker is not loaded.")
        import torchaudio.functional as F_audio

        track_1_np, track_2_np, target_sr = raw_tracks
        track_1_np = np.asarray(track_1_np, dtype=np.float32)
        track_2_np = np.asarray(track_2_np, dtype=np.float32)

        track_1_tensor = torch.from_numpy(track_1_np).to(self.device)
        track_2_tensor = torch.from_numpy(track_2_np).to(self.device)

        # --- WeSpeaker matching: Tính embedding cho các speaker có mẫu ---
        _t = time.perf_counter()
        embed_A = self._get_target_embedding(enroll_A, id_A, sample_rate)
        embed_B = self._get_target_embedding(enroll_B, id_B, sample_rate)
        self._tick("enrollment", _t)

        # Chỉ sửa chunk swap nếu CẢ HAI đều có embedding chuẩn
        n_flips = 0
        if not getattr(self.backend, "ordered", False):
            if embed_A is not None and embed_B is not None:
                track_1_np, track_2_np, n_flips = self._repair_chunk_swaps(
                    track_1_np, track_2_np, target_sr, embed_A, embed_B)
        if n_flips:
            track_1_tensor = torch.from_numpy(track_1_np).to(self.device)
            track_2_tensor = torch.from_numpy(track_2_np).to(self.device)
            print(f"[TSE Model] repaired {n_flips} chunk-seam channel swap(s)", file=sys.stderr)

        scale = target_sr / float(sample_rate)
        def _rescale(spans):
            if not spans:
                return None
            return [(int(a * scale), int(b * scale)) for a, b in spans]

        span_A = _rescale(probe_A)
        span_B = _rescale(probe_B)
        full_span = [(0, len(track_1_np))]
        score_sources = {}
        scoring_errors = {}

        def _score(track_np, spans, target_embed, key):
            attempts = []
            if spans:
                attempts.append((spans, "clean_probe"))
            attempts.append((full_span, "full_context"))
            for candidate_spans, source in attempts:
                try:
                    emb = self._probe_embedding(track_np, candidate_spans, target_sr)
                    if emb is None:
                        continue
                    score = float(torch.dot(target_embed, emb))
                    if not np.isfinite(score):
                        raise ValueError("nonfinite similarity")
                except Exception as exc:
                    scoring_errors[f"{key}:{source}"] = f"{type(exc).__name__}: {exc}"
                    continue
                score_sources[key] = source
                return score
            score_sources[key] = "unscorable"
            return None

        # Chỉ chấm điểm nếu có embedding
        def _later(track_np, spans, embed, key):
            return (lambda: _score(track_np, spans, embed, key)
                    if embed is not None else None)

        if getattr(self, "assignment_batching", False):
            s_1A, s_2A, s_1B, s_2B = self._score_probe_batch([
                (track_1_np, span_A, embed_A, "track1_A"),
                (track_2_np, span_A, embed_A, "track2_A"),
                (track_1_np, span_B, embed_B, "track1_B"),
                (track_2_np, span_B, embed_B, "track2_B"),
            ], target_sr, full_span, score_sources, scoring_errors)
        else:
            s_1A, s_2A, s_1B, s_2B = self._run_scores([
                _later(track_1_np, span_A, embed_A, "track1_A"),
                _later(track_2_np, span_A, embed_A, "track2_A"),
                _later(track_1_np, span_B, embed_B, "track1_B"),
                _later(track_2_np, span_B, embed_B, "track2_B"),
            ])

        def _probe_rms(track, spans):
            pieces = [
                np.asarray(track[max(0, int(a)):min(len(track), int(b))],
                           dtype=np.float64)
                for a, b in (spans or [])
                if int(b) > int(a)
            ]
            pieces = [piece for piece in pieces if len(piece)]
            if not pieces:
                return None
            probe = np.concatenate(pieces)
            return float(np.sqrt(np.mean(np.square(
                probe
            )) + 1e-12))

        # A clean diarization probe is weaker evidence than speaker identity, but
        # much stronger than trusting an unordered backend. Positive evidence
        # means track1=A, track2=B; negative evidence means swap.
        probe_energy = {
            "track1_A": _probe_rms(track_1_np, span_A),
            "track2_A": _probe_rms(track_2_np, span_A),
            "track1_B": _probe_rms(track_1_np, span_B),
            "track2_B": _probe_rms(track_2_np, span_B),
        }
        probe_evidence = 0.0
        probe_terms = 0
        eps = 1e-6
        if probe_energy["track1_A"] is not None and probe_energy["track2_A"] is not None:
            probe_evidence += np.log(
                (probe_energy["track1_A"] + eps) /
                (probe_energy["track2_A"] + eps)
            )
            probe_terms += 1
        if probe_energy["track1_B"] is not None and probe_energy["track2_B"] is not None:
            probe_evidence += np.log(
                (probe_energy["track2_B"] + eps) /
                (probe_energy["track1_B"] + eps)
            )
            probe_terms += 1
        probe_margin = float(np.log(1.20))
        probe_direct = (
            None
            if probe_terms == 0 or abs(probe_evidence) < probe_margin
            else probe_evidence >= 0.0
        )

        def _assign_by_probe(default_mode):
            if probe_direct is None:
                return (track_1_tensor, track_2_tensor, track_1_np, track_2_np,
                        default_mode)
            if probe_direct:
                return (track_1_tensor, track_2_tensor, track_1_np, track_2_np,
                        "clean_probe_energy")
            return (track_2_tensor, track_1_tensor, track_2_np, track_1_np,
                    "clean_probe_energy_swapped")

        if getattr(self.backend, "ordered", False):
            assignment_mode = "backend_ordered"
            out_A_tensor, out_B_tensor = track_1_tensor, track_2_tensor
            out_A_np, out_B_np = track_1_np, track_2_np
            sim_A, sim_B = s_1A, s_2B
        else:
            # ─── LOGIC GÁN LOẠI TRỪ ───
            # Trường hợp 1: Có cả 2 mẫu -> So sánh điểm bình thường
            if embed_A is not None and embed_B is not None:
                assignment_mode = "dual_wespeaker"
                def _n(x): return -1.0 if x is None else x
                if all(value is None for value in (s_1A, s_2A, s_1B, s_2B)):
                    (out_A_tensor, out_B_tensor, out_A_np, out_B_np,
                     assignment_mode) = _assign_by_probe("deterministic_unscorable_wespeaker")
                    sim_A = sim_B = None
                elif (_n(s_1A) + _n(s_2B)) >= (_n(s_2A) + _n(s_1B)):
                    out_A_tensor, out_B_tensor = track_1_tensor, track_2_tensor
                    out_A_np, out_B_np = track_1_np, track_2_np
                    sim_A, sim_B = s_1A, s_2B
                else:
                    out_A_tensor, out_B_tensor = track_2_tensor, track_1_tensor
                    out_A_np, out_B_np = track_2_np, track_1_np
                    sim_A, sim_B = s_2A, s_1B

            # Trường hợp 2: Chỉ có mẫu A (Không có B) -> Dùng A chọn track, track còn lại nhường B
            elif embed_A is not None:
                assignment_mode = "speaker_A_wespeaker_complement_B"
                if s_1A is None and s_2A is None:
                    (out_A_tensor, out_B_tensor, out_A_np, out_B_np,
                     assignment_mode) = _assign_by_probe("deterministic_unscorable_A")
                    sim_A = sim_B = None
                elif (s_1A if s_1A is not None else -1.0) >= (
                    s_2A if s_2A is not None else -1.0
                ):
                    out_A_tensor, out_B_tensor = track_1_tensor, track_2_tensor
                    out_A_np, out_B_np = track_1_np, track_2_np
                    sim_A, sim_B = s_1A, None
                else:
                    out_A_tensor, out_B_tensor = track_2_tensor, track_1_tensor
                    out_A_np, out_B_np = track_2_np, track_1_np
                    sim_A, sim_B = s_2A, None

            # Trường hợp 3: Chỉ có mẫu B (Không có A) -> Dùng B chọn track, track còn lại nhường A
            elif embed_B is not None:
                assignment_mode = "speaker_B_wespeaker_complement_A"
                if s_1B is None and s_2B is None:
                    (out_A_tensor, out_B_tensor, out_A_np, out_B_np,
                     assignment_mode) = _assign_by_probe("deterministic_unscorable_B")
                    sim_A = sim_B = None
                elif (s_1B if s_1B is not None else -1.0) >= (
                    s_2B if s_2B is not None else -1.0
                ):
                    out_A_tensor, out_B_tensor = track_2_tensor, track_1_tensor
                    out_A_np, out_B_np = track_2_np, track_1_np
                    sim_A, sim_B = None, s_1B
                else:
                    out_A_tensor, out_B_tensor = track_1_tensor, track_2_tensor
                    out_A_np, out_B_np = track_1_np, track_2_np
                    sim_A, sim_B = None, s_2B

            # Trường hợp 4: Cả 2 đều không có mẫu -> Gán mặc định
            else:
                (out_A_tensor, out_B_tensor, out_A_np, out_B_np,
                 assignment_mode) = _assign_by_probe("deterministic_no_enrollment")
                sim_A, sim_B = None, None

        # ─── ĐÁNH GIÁ NOT-A AN TOÀN ───
        diag = {
            "anchor_self": None,
            "anchor_other": None,
            "other_rms": None,
            "assignment_mode": assignment_mode,
            "score_sources": score_sources,
            "scoring_errors": scoring_errors,
            "output_scores": ([[s_1A, s_1B], [s_2A, s_2B]]
                              if out_A_np is track_1_np else
                              [[s_2A, s_2B], [s_1A, s_1B]]),
            "assignment_margin": abs(sum(
                left - right for left, right in ((s_1A, s_2A), (s_2B, s_1B))
                if left is not None and right is not None)),
            "probe_energy": probe_energy,
            "probe_log_evidence": (
                None if probe_direct is None else float(probe_evidence)
            ),
        }
        if core_range is not None and (embed_A is not None or embed_B is not None):
            c0, c1 = int(core_range[0] * scale), int(core_range[1] * scale)
            c0, c1 = max(0, c0), min(len(out_A_np), c1)
            if c1 > c0:
                # Ưu tiên lấy anchor từ người THỰC SỰ có embedding
                if embed_A is not None and (sim_A is not None or embed_B is None):
                    anchor_embed = embed_A
                    self_np = out_A_np
                    other_np = out_B_np
                else:
                    anchor_embed = embed_B
                    self_np = out_B_np
                    other_np = out_A_np

                # Chỉ tính dot-product nếu anchor_embed hợp lệ
                if anchor_embed is not None:
                    core_other = other_np[c0:c1]
                    diag["other_rms"] = float(np.sqrt((core_other ** 2).mean() + 1e-12))
                    for key, arr in (("anchor_self", self_np[c0:c1]), ("anchor_other", core_other)):
                        try:
                            pr = self._gather_probe(arr, [(0, len(arr))], target_sr, min_voiced_sec=0.05)
                            if pr is not None:
                                e = F.normalize(self._get_embedding(pr, target_sr), p=2, dim=0)
                                value = float(torch.dot(anchor_embed, e))
                                diag[key] = value if np.isfinite(value) else None
                        except Exception as exc:
                            diag.setdefault("diagnostic_errors", {})[key] = (
                                f"{type(exc).__name__}: {exc}")

        def restore_track(track_tensor_in):
            if sample_rate != target_sr:
                track_tensor_out = F_audio.resample(track_tensor_in.unsqueeze(0).cpu(), target_sr, sample_rate).squeeze(0)
            else:
                track_tensor_out = track_tensor_in.cpu()
                
            track_np = track_tensor_out.numpy()
            orig_len = len(mixture_audio)
            
            if len(track_np) > orig_len:
                track_np = track_np[:orig_len]
            elif len(track_np) < orig_len:
                track_np = np.pad(track_np, (0, orig_len - len(track_np)))
            return track_np
            
        return restore_track(out_A_tensor), restore_track(out_B_tensor), sim_A, sim_B, diag

    def separate_two_speakers(self, mixture_audio: np.ndarray,
                              enroll_A: List[np.ndarray], enroll_B: List[np.ndarray],
                              sample_rate: int = 16000, id_A: Optional[str] = None,
                              id_B: Optional[str] = None,
                              probe_A: Optional[List[Tuple[int, int]]] = None,
                              probe_B: Optional[List[Tuple[int, int]]] = None,
                              core_range: Optional[Tuple[int, int]] = None):
        """Compatibility path: run separation and assignment synchronously."""
        raw_tracks = self.separate_raw(
            mixture_audio, sample_rate, enroll_A=enroll_A, enroll_B=enroll_B)
        return self.postprocess_separated(
            mixture_audio, raw_tracks,
            enroll_A=enroll_A, enroll_B=enroll_B,
            sample_rate=sample_rate, id_A=id_A, id_B=id_B,
            probe_A=probe_A, probe_B=probe_B, core_range=core_range,
        )
