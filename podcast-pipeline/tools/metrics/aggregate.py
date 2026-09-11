"""Aggregate recordings equally; preserve paired evidence only for polish."""

from collections import defaultdict
from statistics import mean, median

from tools.matrix_axes import AXES

METRICS = {
    "delta_bak": ("dnsmos_delta.BAK", "Delta BAK"),
    "delta_sig": ("dnsmos_delta.SIG", "Delta SIG"),
    "ovrl": ("dnsmos.OVRL", "OVRL"),
    "tilt_db": ("spectral.tilt_db", "Tilt dB"),
    "gated_pct": ("spectral.gated_pct", "Gated %"),
    "f0_cents": ("f0.median_abs_cents", "F0 cents"),
    "over50": ("f0.pct_over_50_cents", "F0 >50c %"),
    "sim_enroll": ("ecapa.sim_enroll", "Enrollment sim"),
}


def metric_values(record):
    values = {}
    for key, (path, _) in METRICS.items():
        value = record
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        values[key] = value if isinstance(value, (int, float)) else None
    return values


def aggregate_records(records):
    groups = defaultdict(list)
    for row in records:
        if row.get("status") == "done" and row.get("track") in ("A", "B"):
            groups[(row["combo"], row["stem"])].append(row)
    by_file = []
    for (combo, stem), rows in groups.items():
        values = [metric_values(r) for r in rows]
        summary = {key: median([v[key] for v in values if v[key] is not None])
                   if any(v[key] is not None for v in values) else None for key in METRICS}
        by_file.append(dict(combo=combo, stem=stem, axes=rows[0].get("axes", {}),
                            n_tracks=len(rows), metrics=summary))
    by_combo = {}
    for combo in {row["combo"] for row in by_file}:
        subset = [row for row in by_file if row["combo"] == combo]
        by_combo[combo] = {key: mean([row["metrics"][key] for row in subset if row["metrics"][key] is not None])
                          if any(row["metrics"][key] is not None for row in subset) else None for key in METRICS}
    return by_file, by_combo


def marginal_effects(by_file, records):
    effects = []
    for axis, levels in AXES.items():
        for treatment in levels[1:]:
            groups = defaultdict(dict)
            for row in by_file:
                axes = row["axes"]
                if not all(k in axes for k in AXES) or axes[axis] not in (levels[0], treatment):
                    continue
                key = (row["stem"], *(axes[k] for k in AXES if k != axis))
                groups[key][axes[axis]] = row["metrics"]
            paired = [g for g in groups.values() if levels[0] in g and treatment in g]
            if axis == "F":
                # Rebuild using common windows/tracks, not medians from unequal
                # subsets when an enhancement or measurement failed.
                windows = defaultdict(dict)
                for row in records:
                    axes = row.get("axes", {})
                    if row.get("status") != "done" or row.get("track") not in ("A", "B") or not all(k in axes for k in AXES):
                        continue
                    if axes[axis] not in (levels[0], treatment):
                        continue
                    key = (row["stem"], *(axes[k] for k in AXES if k != axis), row["tag"], row["track"])
                    windows[key][axes[axis]] = metric_values(row)
                per_file = defaultdict(list)
                for key, group in windows.items():
                    if levels[0] in group and treatment in group:
                        per_file[key[:-2]].append(group)
                paired = []
                for group in per_file.values():
                    paired.append({level: {metric: median([g[level][metric] for g in group
                        if g[level][metric] is not None and all(g[v][metric] is not None for v in (levels[0], treatment))])
                        if any(all(g[v][metric] is not None for v in (levels[0], treatment)) for g in group)
                        else None for metric in METRICS} for level in (levels[0], treatment)})
            measures = {}
            for metric in METRICS:
                values = [(g[levels[0]][metric], g[treatment][metric]) for g in paired
                          if g[levels[0]][metric] is not None and g[treatment][metric] is not None]
                measures[metric] = dict(
                    control=mean([a for a, _ in values]) if values else None,
                    treatment=mean([b for _, b in values]) if values else None,
                    delta=mean([b-a for a, b in values]) if values else None,
                    n=len(values))
            effects.append(dict(axis=axis, control=levels[0], treatment=treatment,
                                paired_windows=axis == "F", metrics=measures))
    return effects
