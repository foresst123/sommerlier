"""Generic `--set KEY=VALUE` overrides applied to the loaded profile (main.py)."""

import json

# Sections a profile always may have; any key already in the profile is allowed too.
BASE_SECTIONS = ("steps", "pipeline", "models", "performance", "audio_quality", "outputs", "batch")


def parse_set(item):
    """'a.b.c=VALUE' -> (['a','b','c'], value). VALUE is JSON when it parses, else the raw string."""
    key, sep, raw = str(item).partition("=")
    key = key.strip()
    if not sep or not key:
        raise ValueError(f"--set {item!r}: expected KEY=VALUE (e.g. steps.music_removal=false)")
    path = [p.strip() for p in key.split(".")]
    if any(not p for p in path):
        raise ValueError(f"--set {item!r}: empty segment in key {key!r}")
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    return path, value


def apply_overrides(profile, items):
    """Set each KEY=VALUE in the profile dict, in place, as if config.json had it.

    Returns the applied list as 'key=<json value>' strings (for the log line).
    """
    applied = []
    for item in items or []:
        path, value = parse_set(item)
        allowed = set(BASE_SECTIONS) | set(profile)
        if path[0] not in allowed:
            raise ValueError(f"--set {item!r}: unknown section {path[0]!r}; allowed: {sorted(allowed)}")
        node = profile
        for i, seg in enumerate(path[:-1]):
            nxt = node.get(seg)
            if nxt is None:
                nxt = node[seg] = {}
            elif not isinstance(nxt, dict):
                raise ValueError(f"--set {item!r}: {'.'.join(path[:i + 1])} is not a section")
            node = nxt
        node[path[-1]] = value
        applied.append(f"{'.'.join(path)}={json.dumps(value, ensure_ascii=False)}")
    return applied
