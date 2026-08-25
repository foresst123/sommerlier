"""Enrollment embeddings must not survive from one file to the next.

The separator caches them under the diarizer's speaker label, and those labels
restart at "1" for every file. Left in place, the second file's speaker "1"
picks up the first file's entry and every track is scored against a stranger's
voice -- which is invisible on a single-file run.

Run:  python -m pytest tests/test_speaker_cache.py -q     (from podcast-pipeline/)
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeExtractor:
    """Stands in for TargetSpeakerExtractor: only the cache matters here."""

    def __init__(self):
        self.target_embed_cache = {}

    def reset_speakers(self):
        self.target_embed_cache.clear()

    def enroll(self, speaker, voice):
        """Mimic _get_target_embedding: return the cached entry if present."""
        if speaker in self.target_embed_cache:
            return self.target_embed_cache[speaker]
        self.target_embed_cache[speaker] = voice
        return voice


def _service(extractor):
    from services.separation_service import TargetExtractionService
    svc = TargetExtractionService.__new__(TargetExtractionService)
    svc.tse_model = extractor
    svc.logger = None
    return svc


def test_a_second_file_does_not_inherit_the_first_files_voices():
    tse = _FakeExtractor()
    svc = _service(tse)

    svc.reset_stats()
    assert tse.enroll("1", "voice-of-file-A") == "voice-of-file-A"

    svc.reset_stats()                       # next file
    got = tse.enroll("1", "voice-of-file-B")

    assert got == "voice-of-file-B", (
        "speaker 1 of the second file was scored against the first file's voice")


def test_without_the_reset_the_stale_voice_would_win():
    """Pins what the bug actually was, so a future refactor cannot quietly
    reintroduce it."""
    tse = _FakeExtractor()
    tse.enroll("1", "voice-of-file-A")

    assert tse.enroll("1", "voice-of-file-B") == "voice-of-file-A"


def test_resetting_stats_clears_the_cache():
    tse = _FakeExtractor()
    tse.target_embed_cache["1"] = "stale"
    tse.target_embed_cache["2"] = "stale"

    _service(tse).reset_stats()

    assert tse.target_embed_cache == {}


def test_a_separator_without_the_hook_is_tolerated():
    """--tse off leaves tse_model as None, and older stubs lack the method."""
    _service(None).reset_stats()
    _service(object()).reset_stats()


def test_the_extractor_exposes_reset_speakers():
    with open(os.path.join(ROOT, "models/tse_model.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "TargetSpeakerExtractor")
    names = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
    assert "reset_speakers" in names
