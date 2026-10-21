"""Helpers shared by the standalone worker scripts (no repo imports, so they run
in any worker virtualenv)."""


def describe_exception(exc: BaseException) -> str:
    """`Type: message`, followed by every underlying cause.

    Libraries such as qwen-asr catch the real ImportError and re-raise a generic
    "vLLM is not available"; the message the parent shows must carry the cause.
    """
    parts, seen = [], set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " <- caused by ".join(parts)
