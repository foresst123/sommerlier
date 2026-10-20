"""Which diarization backend a run uses: DiariZen (default) or Nemotron."""

BACKENDS = ("diarizen", "nemotron")
DEFAULT_BACKEND = "diarizen"


def resolve_diarization_backend(args, env_profile=None) -> str:
    """CLI flag wins, then ``models.diarization_backend`` in the profile, then DiariZen."""
    value = getattr(args, "diarization_backend", None)
    if value is None:
        value = ((env_profile or {}).get("models") or {}).get("diarization_backend")
    value = DEFAULT_BACKEND if value is None else str(value).strip().lower()
    if value not in BACKENDS:
        raise ValueError(
            f"Unknown diarization backend {value!r}; expected one of {', '.join(BACKENDS)}")
    if value != DEFAULT_BACKEND and getattr(args, "dia3", False):
        raise ValueError(
            f"--dia3 (Pyannote) cannot be combined with diarization backend {value!r}")
    return value
