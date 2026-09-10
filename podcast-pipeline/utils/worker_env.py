"""Locating the isolated Python interpreter each worker subprocess runs under."""

import os

# Each worker lives in its own virtualenv because their dependency sets conflict
# with the main environment. Falling back to the main interpreter would fail deep
# inside model loading instead of here, so resolution is strict.
_SEARCH_ROOTS = [
    "/kaggle/temp",
    "/kaggle/working",
]


def resolve_worker_python(worker_name: str, config=None, env_profile=None, logger=None) -> str:
    """Return the interpreter for ``worker_name``'s virtualenv (e.g. "qwen3").

    Resolution order: the ``{WORKER}_PYTHON`` environment variable, then any
    ``worker_envs`` mapping in the active config profile, then the conventional
    ``<root>/<name>_env/bin/python`` locations. Raises when nothing resolves.
    """
    env_var = f"{worker_name.upper()}_PYTHON"
    explicit = os.environ.get(env_var)
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise FileNotFoundError(
            f"{env_var} points at {explicit}, which does not exist."
        )

    if env_profile:
        configured = (env_profile.get("worker_envs") or {}).get(worker_name)
        if configured:
            if os.path.exists(configured):
                return configured
            raise FileNotFoundError(
                f"worker_envs.{worker_name} in the active config profile points at "
                f"{configured}, which does not exist."
            )

    pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    for root in _SEARCH_ROOTS:
        candidates.append(os.path.join(root, f"{worker_name}_env", "bin", "python"))
    for depth in ("..", os.path.join("..", "..")):
        candidates.append(
            os.path.join(pipeline_dir, depth, f"{worker_name}_env", "bin", "python")
        )

    for candidate in candidates:
        if os.path.exists(candidate):
            resolved = os.path.abspath(candidate)
            if logger:
                logger.info(f"Resolved {worker_name} interpreter: {resolved}")
            return resolved

    raise FileNotFoundError(
        f"No interpreter found for the '{worker_name}' worker. Set {env_var}, add "
        f"worker_envs.{worker_name} to the config profile, or create "
        f"{worker_name}_env next to the repository. Searched: "
        + ", ".join(os.path.abspath(c) for c in candidates)
    )


def resolve_checkpoint(name: str, candidates, env_profile=None, config_key=None, logger=None) -> str:
    """Return the first existing path from ``candidates``.

    ``env_profile['checkpoints'][config_key]`` takes precedence when present, so
    a deployment can name its own location instead of relying on search order.
    """
    if env_profile and config_key:
        configured = (env_profile.get("checkpoints") or {}).get(config_key)
        if configured:
            if os.path.exists(configured):
                return configured
            raise FileNotFoundError(
                f"checkpoints.{config_key} in the active config profile points at "
                f"{configured}, which does not exist."
            )

    for path in candidates:
        if path and os.path.exists(path):
            if logger:
                logger.info(f"Resolved {name} checkpoint: {path}")
            return path

    raise FileNotFoundError(
        f"No checkpoint found for {name}. Searched: " + ", ".join(str(c) for c in candidates)
    )
