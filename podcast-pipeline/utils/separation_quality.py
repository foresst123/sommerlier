"""Sample-aligned quality and identity checks for separated windows."""
import numpy as np


def track_quality(host, track, sr, floor=0.002, frame_sec=0.02):
    host = np.asarray(host, dtype=np.float64).reshape(-1)
    track = np.asarray(track, dtype=np.float64).reshape(-1)
    n = min(len(host), len(track))
    result = {"samples": n, "floor_rms": float(floor), "coverage": None,
              "frame_samples": max(1, round(frame_sec * sr))}
    if n == 0 or len(track) < len(host):
        return dict(result, accepted=False, status="short_track")
    if not np.isfinite(host).all() or not np.isfinite(track).all():
        return dict(result, accepted=False, status="nonfinite_audio")
    frame = result["frame_samples"]
    starts = np.arange(0, n, frame)
    counts = np.minimum(frame, n - starts)
    h = np.sqrt(np.add.reduceat(host[:n] ** 2, starts) / counts)
    t = np.sqrt(np.add.reduceat(track[:n] ** 2, starts) / counts)
    voiced = h >= max(float(h.max()) * 0.25, floor)
    result.update(host_rms=float(np.sqrt(np.mean(host[:n] ** 2))),
                  track_rms=float(np.sqrt(np.mean(track[:n] ** 2))),
                  host_frame_rms=h.tolist(), track_frame_rms=t.tolist(),
                  frame_lengths=counts.tolist(), voiced_frames=voiced.tolist())
    if not voiced.any():
        return dict(result, accepted=True, status="quiet_mixture")
    alive = t >= max(float(t.max()) * 0.05, floor)
    coverage = float(np.sum(counts[voiced & alive]) / np.sum(counts[voiced]))
    result["coverage"] = coverage
    if coverage >= 0.10:
        return dict(result, accepted=True, status="energy_present")
    status = "insufficient_evidence" if n < round(0.1 * sr) else "low_energy_coverage"
    return dict(result, accepted=False, status=status)


def base_view(window, tracks):
    base = next(piece for piece in window.layout["pieces"] if piece["kind"] == "base")
    index = window.layout["pieces"].index(base)
    fades = window.layout.get("join_crossfade_samples")
    if fades is None:
        fades = [window.layout.get("crossfade_samples", 0)] * (
            len(window.layout["pieces"]) - 1)
    left = fades[index - 1] if index else 0
    right = fades[index] if index < len(fades) else 0
    lo, hi = base["source_samples"]
    offset = base["window_samples"][0]
    return (lo + left, hi - right), tuple(
        np.asarray(track[offset + left:offset + hi - lo - right]).copy()
        for track in tracks)


def continuity_evidence(previous, window, tracks, sr, floor=0.002):
    bounds, current = base_view(window, tracks)
    result = {"status": "no_shared_context", "swap": False, "margin": None}
    if previous is None:
        return result
    lo, hi = max(bounds[0], previous["bounds"][0]), min(bounds[1], previous["bounds"][1])
    if hi - lo < round(0.1 * sr):
        return result
    old = [t[lo - previous["bounds"][0]:hi - previous["bounds"][0]]
           for t in previous["tracks"]]
    new = [t[lo - bounds[0]:hi - bounds[0]] for t in current]

    def correlation(a, b):
        if len(a) != len(b) or not len(a):
            return None
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return None
        if min(np.sqrt(np.mean(a * a)), np.sqrt(np.mean(b * b))) < floor:
            return None
        a, b = a - a.mean(), b - b.mean()
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return abs(float(np.dot(a, b) / denominator)) if denominator > 1e-12 else None

    matrix = [[correlation(a, b) for b in new] for a in old]
    direct = [matrix[0][0], matrix[1][1]]
    swapped = [matrix[0][1], matrix[1][0]]
    direct_score = sum(v or 0.0 for v in direct)
    swap_score = sum(v or 0.0 for v in swapped)
    margin = swap_score - direct_score
    enough = max(direct_score, swap_score) >= 0.3 and abs(margin) >= 0.1
    return dict(result, status="measured" if enough else "ambiguous",
                source_samples=[lo, hi], correlations=matrix,
                margin=margin, swap=bool(enough and margin > 0))
