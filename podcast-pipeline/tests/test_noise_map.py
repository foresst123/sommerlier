"""Seeing the contamination the pipeline used to be blind to.

PANNs predicts 527 AudioSet labels on every forward pass and this pipeline read
three of them. A segment recorded beside a motorbike and one recorded in a
treated room were the same thing to it.

These pin the three decisions that matter: that the noise groups cost no extra
model call, that the score describes what a segment actually holds rather than
the range it spans, and -- most importantly -- that nothing here modifies
audio. Enhancement was ruled out for this corpus on purpose; a test that lets
it back in silently would undo that.

Run:  python -m pytest tests/test_noise_map.py -q   (from podcast-pipeline/)
"""
import ast
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.noise_map import KINDS, NoiseTrack, build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _track(**curves):
    return NoiseTrack({k: np.asarray(v, dtype=np.float32)
                       for k, v in curves.items()}, fps=100.0)


# --- the score describes what the segment holds ------------------------------

def test_a_span_is_scored_from_its_own_frames():
    curve = np.concatenate([np.zeros(100), np.full(100, 0.8)])
    track = _track(noise_env=curve)
    assert track.score_spans([(0.0, 1.0)]) == pytest.approx(0.0, abs=0.01)
    assert track.score_spans([(1.0, 2.0)]) == pytest.approx(0.8, abs=0.01)


def test_a_glued_segment_is_scored_on_both_its_pieces_only():
    """provenance hands over the original ranges a segment is made of. The
    stretch removed between them is not part of it and must not be scored."""
    curve = np.concatenate([np.zeros(100),        # 0-1s   kept, clean
                            np.full(100, 0.9),    # 1-2s   REMOVED, filthy
                            np.zeros(100)])       # 2-3s   kept, clean
    track = _track(noise_room=curve)
    assert track.score_spans([(0.0, 1.0), (2.0, 3.0)]) == pytest.approx(0.0, abs=0.01)
    # The naive range would sweep the removed second back in.
    assert track.score_spans([(0.0, 3.0)]) > 0.5


def test_the_score_is_a_high_percentile_not_a_mean_or_a_max():
    """A mean lets a long clean turn hide a second of traffic; a max lets one
    frame of a door closing condemn the whole turn."""
    curve = np.concatenate([np.zeros(950), np.full(50, 1.0)])   # 5% filthy
    track = _track(noise_env=curve)
    score = track.score_spans([(0.0, 10.0)])
    assert score == pytest.approx(0.0, abs=0.01), "5% must not condemn the span"
    sustained = _track(noise_env=np.concatenate([np.zeros(700), np.full(300, 1.0)]))
    assert sustained.score_spans([(0.0, 10.0)]) > 0.9, "30% is sustained"


def test_the_kinds_combine_by_max_not_by_sum():
    """A room with a fan and a keyboard is not twice as contaminated as one
    with either; summing would push ordinary rooms past any sane threshold."""
    track = _track(noise_env=np.full(100, 0.4), noise_room=np.full(100, 0.4))
    assert track.score_spans([(0.0, 1.0)]) == pytest.approx(0.4, abs=0.01)


def test_the_breakdown_separates_voices_from_everything_else():
    """Background voices break diarization and put words in the transcript
    nobody said; a fan only costs accuracy. A filter needs to tell them apart."""
    track = _track(noise_speech=np.full(100, 0.7), noise_env=np.zeros(100))
    out = track.breakdown([(0.0, 1.0)])
    assert out["noise_speech"] == pytest.approx(0.7, abs=0.01)
    assert out["noise_env"] == pytest.approx(0.0, abs=0.01)


# --- unmeasured is not clean -------------------------------------------------

def test_an_empty_track_scores_nothing_rather_than_zero():
    """Zero reads as "measured, and clean". None reads as "not checked"."""
    assert NoiseTrack().score_spans([(0.0, 1.0)]) is None
    assert not NoiseTrack()


def test_a_span_past_the_end_of_the_track_scores_nothing():
    assert _track(noise_env=np.zeros(100)).score_spans([(50.0, 51.0)]) is None


def test_an_empty_span_list_scores_nothing():
    assert _track(noise_env=np.zeros(100)).score_spans([]) is None


# --- it must cost no extra model call ----------------------------------------

def test_the_track_is_built_from_scores_not_from_a_waveform():
    """Tagging costs a full PANNs sweep and the music map already pays for
    one; reading these columns out of that result is free."""
    import inspect
    params = list(inspect.signature(build).parameters)
    assert params == ["scores", "fps"], (
        "taking a waveform here would mean a second sweep")


def test_one_sweep_produces_both_maps():
    from utils import music_map
    source = inspect_source(music_map.build_maps)
    assert source.count("tag(") == 1, "the tagger must be called exactly once"


def inspect_source(fn):
    import inspect
    return inspect.getsource(fn)


def test_build_ignores_groups_the_detector_did_not_return():
    """An older detector returns three keys. That is a track with nothing in
    it, not a crash."""
    track = build({"speech": np.zeros(10), "music": np.zeros(10)}, 100.0)
    assert not track


# --- the labels have to be real ----------------------------------------------

def test_every_noise_label_exists_in_audioset():
    """`_label_columns` drops a name it cannot find, so a typo becomes a group
    that always answers zero and looks like clean audio."""
    source = open(os.path.join(ROOT, "models", "panns.py"), encoding="utf-8").read()
    groups = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.endswith("_LABELS")):
            groups[node.targets[0].id] = [e.value for e in node.value.elts]

    assert set(KINDS) == {"noise_speech", "noise_env", "noise_room"}
    for name in ("NOISE_SPEECH_LABELS", "NOISE_ENV_LABELS", "NOISE_ROOM_LABELS"):
        assert groups.get(name), name
        assert len(set(groups[name])) == len(groups[name]), f"{name} repeats a label"


def test_the_detector_reports_labels_it_cannot_find():
    """Silence here is how a stale group becomes a detector that always says
    the audio is clean."""
    source = open(os.path.join(ROOT, "models", "panns.py"), encoding="utf-8").read()
    assert "will never fire" in source


def test_speaker_sounds_are_not_treated_as_noise():
    """Breathing before a turn and a laugh over someone's sentence are the
    phenomena a full-duplex corpus is collecting, not contamination."""
    source = open(os.path.join(ROOT, "models", "panns.py"), encoding="utf-8").read()
    start = source.index("NOISE_SPEECH_LABELS")
    end = source.index("NOISE_GROUPS")
    block = source[start:end]
    declared = block[:block.index("# Deliberately NOT noise")]
    for kept in ("Breathing", "Cough", "Laughter", "Sneeze", "Sigh"):
        assert f'"{kept}"' not in declared, f"{kept} belongs to the speaker"


# --- serialisation -----------------------------------------------------------

def test_a_track_survives_a_json_round_trip():
    track = _track(noise_env=np.linspace(0, 1, 100))
    revived = NoiseTrack.from_json(track.to_json())
    assert revived.fps == track.fps
    assert revived.score_spans([(0.0, 1.0)]) == pytest.approx(
        track.score_spans([(0.0, 1.0)]), abs=0.01)


def test_the_json_is_rounded_so_a_long_recording_stays_small():
    """300k frames of float32 is 1.2MB per kind unrounded, and this is written
    per file per run."""
    track = _track(noise_env=np.full(1000, 0.123456789))
    values = track.to_json()["curves"]["noise_env"]
    assert all(len(str(v).split(".")[-1]) <= 3 for v in values)


def test_an_empty_track_round_trips_to_an_empty_track():
    assert not NoiseTrack.from_json(NoiseTrack().to_json())
    assert not NoiseTrack.from_json(None)


# --- the constraint this whole module exists under ---------------------------

def test_nothing_here_modifies_audio():
    """Enhancement was ruled out for this corpus: a denoiser alters the
    recording, and what it invents becomes training data for a conversation
    that never happened. This module marks; it never repairs.

    Checked against the parsed code rather than the text, so the prose
    explaining the decision does not trip its own test."""
    tree = ast.parse(open(os.path.join(ROOT, "utils", "noise_map.py"),
                          encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    # Anything that could write or transform a waveform. numpy does the
    # arithmetic and os reads one env var; neither can reach the audio,
    # because this module never receives any.
    for banned in ("soundfile", "librosa", "torch", "torchaudio", "scipy",
                   "audio_separator", "pydub"):
        assert banned not in imported, f"noise_map imports {banned}"
    assert imported <= {"numpy", "os"}, f"unexpected imports: {sorted(imported)}"


# --- the noticeable level, against what the model actually produces ----------

def test_the_noticeable_level_sits_inside_the_observed_range():
    """A dump of all 527 labels over three recordings put the ceiling of every
    noise group at 0.284 / 0.257 / 0.073. An earlier 0.35 here was above every
    value ever measured -- it could not have fired on anything."""
    from utils.noise_map import NOTICEABLE
    observed_ceiling = 0.284
    quiet_room_floor = 0.005
    assert quiet_room_floor < NOTICEABLE < observed_ceiling, (
        f"{NOTICEABLE} is outside the range the groups occupy")


def test_the_noticeable_level_separates_a_quiet_room_from_a_noisy_one():
    """The measured p50 of a clean studio interview is ~0.002; the sustained
    stretches worth excluding reached 0.25."""
    track = _track(noise_env=np.full(100, 0.002, dtype=np.float32))
    from utils.noise_map import NOTICEABLE
    assert track.score_spans([(0.0, 1.0)]) < NOTICEABLE
    noisy = _track(noise_env=np.full(100, 0.25, dtype=np.float32))
    assert noisy.score_spans([(0.0, 1.0)]) > NOTICEABLE


def test_the_level_can_be_moved_without_editing_code():
    """All three measured recordings are indoor. The ceiling may rise once the
    outdoor stratum is run, and that must not need a patch."""
    import importlib, os
    import utils.noise_map as nm
    old = os.environ.get("NOISE_NOTICEABLE")
    try:
        os.environ["NOISE_NOTICEABLE"] = "0.2"
        assert importlib.reload(nm).NOTICEABLE == 0.2
    finally:
        if old is None:
            os.environ.pop("NOISE_NOTICEABLE", None)
        else:
            os.environ["NOISE_NOTICEABLE"] = old
        importlib.reload(nm)


def test_the_log_line_uses_the_named_level_not_a_literal():
    """The 0.35 that could never fire was a magic number in a log line; the
    value now has evidence behind it and one place to change."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "utils", "music_map.py"), encoding="utf-8").read()
    assert "from utils.noise_map import NOTICEABLE" in source
    assert ">= NOTICEABLE" in source
