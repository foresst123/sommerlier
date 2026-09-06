"""Which stages of a run are switched on.

The profile's `steps` block is the answer, one key per stage, false to skip.
It lives here rather than on PipelineService because two different places ask
the question and they have to agree: the pipeline decides whether to run a
stage, and the model loader decides whether to load what that stage needs. When
they disagreed, `steps.music_analysis: true` produced a run that skipped the
tagger and reported an empty music map -- a stage that says it ran and did
nothing, which is the worst of the three possible outcomes.
"""

# A stage's older gate, for a profile written before `steps` existed. Several
# of these flags cover more than one stage, which is the reason `steps` exists:
# `panns` turned on both the sweep and the vocal separator, and there was no
# way to ask for one without the other.
LEGACY_FLAG = {
    "music_analysis": "panns",
    "music_removal": "panns",
    "separation": "tse",
    "captioning": "qwen3omni",
    "refinement": "llm_refinement",
}


def step_enabled(args, name: str) -> bool:
    """Whether `name` runs. Unlisted steps run, which is the old behaviour."""
    explicit = getattr(args, f"step_{name}", None)
    if explicit is not None:
        return bool(explicit)
    legacy = LEGACY_FLAG.get(name)
    return bool(getattr(args, legacy, True)) if legacy else True


# The three stages nothing downstream has an input without. PipelineService
# stops the run at the music stage when any one of them is off, which is what
# makes reachability a different question from whether a step is switched on.
LOAD_BEARING = ("diarization", "asr", "export")

# Stages that run before that check, and so run whatever else is off.
BEFORE_THE_CHECK = ("music_analysis", "music_removal", "cut_singing")


def will_run(args, name: str) -> bool:
    """Whether this run actually reaches `name`.

    `step_enabled` answers whether a stage is switched on; this answers whether
    the run gets there at all. The two differ whenever a load-bearing stage is
    off: with diarization disabled the run stops after the music stage, so ASR
    is still "enabled" and will never execute.

    Worker processes are the reason this matters. They are spawned before the
    first file is opened, each with its own interpreter and weights, so asking
    the wrong question starts a subprocess for a stage that never comes -- or,
    on an install that has no interpreter for it, ends the run there.
    """
    if not step_enabled(args, name):
        return False
    if name in BEFORE_THE_CHECK:
        return True
    return all(step_enabled(args, required) for required in LOAD_BEARING)
