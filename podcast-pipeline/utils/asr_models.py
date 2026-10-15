"""Which ASR models a run uses, from the profile's `pipeline.asr_models`."""

ALL_ASR_MODELS = ("qwen3", "whisper", "phowhisper")


def enabled_asr_models(env_profile) -> tuple:
    """The ASR models to load, in a stable order; all three when the profile says nothing.

    A single model skips ROVER: there is nothing to vote between.
    """
    asked = ((env_profile or {}).get("pipeline") or {}).get("asr_models")
    if asked is None:
        return ALL_ASR_MODELS
    if isinstance(asked, str):
        asked = [asked]
    unknown = sorted(set(asked) - set(ALL_ASR_MODELS))
    if unknown or not asked:
        raise ValueError(
            f"pipeline.asr_models must be a non-empty subset of {list(ALL_ASR_MODELS)}; "
            f"got {list(asked)}")
    return tuple(name for name in ALL_ASR_MODELS if name in asked)
