"""Where the final dataset goes: `outputs.clean_two_channel_dir` when it is set."""
import os
from typing import Optional, Tuple


def final_output_root(config: dict, args) -> str:
    """`outputs.clean_two_channel_dir` of the active env, "" when unset (feature off)."""
    profile = (config or {}).get("environments", {}).get(getattr(args, "env", ""), {})
    return str(((profile.get("outputs") or {}).get("clean_two_channel_dir")) or "").strip()


_LHOTSE_KEYS = ("coverage_check", "coverage_max_uncovered", "coverage_rms_threshold",
                "coverage_min_speech_frames")


def lhotse_params(config: dict, args):
    """`outputs.lhotse` of the active env as LhotseManifestParams; None when it is not set.

    Only the coverage check's settings can be changed here. An unknown key raises, so a
    typo is an error rather than a setting that quietly did not apply.
    """
    profile = (config or {}).get("environments", {}).get(getattr(args, "env", ""), {})
    section = dict((profile.get("outputs") or {}).get("lhotse") or {})
    if not section:
        return None
    unknown = sorted(set(section) - set(_LHOTSE_KEYS))
    if unknown:
        raise ValueError(f"unknown key(s) in outputs.lhotse: {', '.join(unknown)}")
    from services.lhotse_manifest_service import LhotseManifestParams
    return LhotseManifestParams(**section)


def conversation_export_dir(config: dict, args, output_dir: str) -> Tuple[str, bool]:
    """(out_dir, shared). Shared means every recording writes into the final root."""
    root = final_output_root(config, args)
    if root:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(root))), True
    return os.path.join(output_dir, "conversation_exports"), False


def run_lhotse_after_export(root: str, source_stem: str, report: Optional[dict] = None,
                            logger=None, params=None) -> Optional[dict]:
    """Write this recording's Lhotse manifest; never raises. Failures go to `report`."""
    try:
        from services.lhotse_manifest_service import build_recording_manifest
        res = build_recording_manifest(root, source_stem, params)
        info = {"written": res.written, "skipped": len(res.skipped),
                "manifest": res.output_path}
        if logger:
            logger.info(f"[lhotse] {source_stem}: {res.written} cut(s), "
                        f"{len(res.skipped)} skipped -> {res.output_path}")
    except Exception as exc:                                   # converter must not fail the run
        info = {"error": str(exc)}
        if logger:
            logger.warning(f"[lhotse] manifest for {source_stem} failed: {exc}")
    if report is not None:
        report["lhotse"] = info
    return info
