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
# `music` turns on both the sweep and the vocal separator, and there is no way
# to ask for one without the other. (It was spelled `panns` until the tagger
# behind it was replaced; main.py still accepts that spelling.)
LEGACY_FLAG = {
    "music_analysis": "music",
    "music_removal": "music",
    "separation": "bss",
    "captioning": "qwen3omni",
    "refinement": "llm_refinement",
}


def review_page_enabled(args) -> bool:
    """Whether the closing HTML page is written.

    One switch: `pipeline.review_page` in the profile, or --review_page /
    --no_review_page on the command line (the command line wins). Not set at all
    means on, so a profile written before the page was a stage still gets one.
    """
    value = getattr(args, "review_page", None)
    return True if value is None else bool(value)


def step_enabled(args, name: str) -> bool:
    """Whether `name` runs. Unlisted steps run, which is the old behaviour."""
    explicit = getattr(args, f"step_{name}", None)
    if explicit is not None:
        return bool(explicit)
    legacy = LEGACY_FLAG.get(name)
    return bool(getattr(args, legacy, True)) if legacy else True


def opt_in_step_enabled(args, name: str) -> bool:
    """Whether an opt-in `name` runs: only when the profile or --steps says so.

    `step_enabled` treats a step nobody listed as ON, which is right for every
    stage that existed before `steps` did and wrong for one that produces new
    output. Under that rule a profile written before the step existed would
    start writing it the next time it was run. Here silence means off, and
    "not checked" is never read as "wanted".
    """
    return getattr(args, f"step_{name}", None) is True


# Stages that come after the CPU-only --stop_after separation_plan point. A run
# that stops there never reaches them, so their GPU models are not loaded.
_AFTER_SEPARATION_PLAN = frozenset({
    "separation", "music_removal", "denoise_pre_asr", "asr", "captioning",
    "refinement", "speaker_relabel", "transcript_review", "persona_profile",
    "word_alignment", "conversation_exports",
})


def will_run(args, name: str) -> bool:
    """Whether this run actually reaches `name`.

    A step runs if and only if it is enabled in the config, and the run was not
    stopped before it: --stop_after separation_plan stops before every stage in
    _AFTER_SEPARATION_PLAN. Each stage still checks its own prerequisites.
    """
    if (getattr(args, "stop_after", None) == "separation_plan"
            and name in _AFTER_SEPARATION_PLAN):
        return False
    return step_enabled(args, name)
