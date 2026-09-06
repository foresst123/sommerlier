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
