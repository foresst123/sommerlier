"""Turn the monitor's events and resource samples into one readable text report.

Pure functions: `build_report(events, samples, elapsed)` takes what
PerformanceMonitor recorded and returns text, so it can be tested with made-up
events. Nothing here touches the console log -- the report is a file.
"""

from collections import defaultdict

_STAGE_ORDER = ("music", "diarization", "separation", "music_removal", "asr", "refinement",
                "speaker_relabel", "word_alignment", "conversation_exports",
                "clean-data+export")


def _fmt_s(seconds):
    return f"{seconds:8.1f}s"


def _pct(part, whole):
    return f"{100.0 * part / whole:5.1f}%" if whole > 0 else "   n/a"


def _stage_key(stage):
    return (_STAGE_ORDER.index(stage) if stage in _STAGE_ORDER else len(_STAGE_ORDER), stage)


def _table(rows, header):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [
        len(h) for h in header]
    line = lambda cells: "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                                   for i, (c, w) in enumerate(zip(cells, widths)))
    return [line(header), line(["-" * w for w in widths]), *[line(r) for r in rows]]


def _stages(events, elapsed):
    finished = [e for e in events if e.get("event") == "stage_finished"]
    total = sum(e["seconds"] for e in finished) or elapsed
    rows = [[e["stage"], f"{e['seconds']:.1f}", _pct(e["seconds"], total),
             e.get("files", ""), e.get("failures", "")]
            for e in sorted(finished, key=lambda e: _stage_key(e["stage"]))]
    out = ["1. STAGES (wall clock of each batch stage)"]
    out += _table(rows, ["stage", "seconds", "% of stages", "files", "failed"])
    out.append(f"total in stages: {total:.1f}s   whole run: {elapsed:.1f}s")
    return out


def _file_stages(events):
    spans = [e for e in events if e.get("event") == "span" and e.get("name") == "file_stage"]
    if not spans:
        return []
    rows = [[e.get("stage", ""), e.get("file", ""), f"{e['seconds']:.1f}",
             "FAILED" if e.get("error") else ""]
            for e in sorted(spans, key=lambda e: (_stage_key(e.get("stage", "")),
                                                   e.get("file", "")))]
    return ["", "2. PER FILE, PER STAGE (time inside run() for that stage)",
            *_table(rows, ["stage", "file", "seconds", ""])]


def _steps(events):
    stage_seconds = {e["stage"]: e["seconds"] for e in events
                     if e.get("event") == "stage_finished"}
    grouped = defaultdict(lambda: [0, 0.0, 0.0])
    for e in events:
        if e.get("event") != "span" or e.get("name") == "file_stage":
            continue
        entry = grouped[(e.get("stage") or "-", e["name"])]
        entry[0] += 1
        entry[1] += e["seconds"]
        entry[2] = max(entry[2], e["seconds"])
    if not grouped:
        return []
    rows = []
    for (stage, name), (calls, total, longest) in sorted(
            grouped.items(), key=lambda kv: (_stage_key(kv[0][0]), -kv[1][1])):
        rows.append([stage, name, calls, f"{total:.1f}", f"{total / calls:.2f}",
                     f"{longest:.1f}", _pct(total, stage_seconds.get(stage, 0))])
    return ["", "3. STEPS (calls summed over files; a step running for several files at once "
            "can exceed its stage's wall time)",
            *_table(rows, ["stage", "step", "calls", "total s", "mean s", "longest s",
                           "% of stage"])]


def _workers(events):
    out, loading = [], {}
    rows = []
    for e in events:
        kind, worker = e.get("event"), e.get("worker")
        if kind == "worker_loading":
            loading[worker] = e["time"]
        elif kind == "worker_ready" and worker in loading:
            rows.append([worker, "start -> ready", f"{e['time'] - loading.pop(worker):.1f}"])
    for e in events:
        if e.get("event") == "span" and e.get("name") in ("release_worker",):
            rows.append([e.get("worker", "-"), "stop", f"{e['seconds']:.1f}"])
    profiles = [e for e in events if e.get("event") == "worker_profile"]
    if not rows and not profiles:
        return []
    out.append("")
    out.append("4. WORKER PROCESSES")
    out += _table(rows, ["worker", "what", "seconds"]) if rows else []
    for e in profiles:
        wall = e.get("wall_seconds", 0.0)
        out.append(f"  {e['worker']}: {e.get('calls', 0)} calls over {wall:.1f}s, "
                   f"waited {e.get('lease_wait_seconds', 0.0):.1f}s for an idle worker")
        for i, w in enumerate(e.get("workers", [])):
            out.append(f"    worker {i}: busy {w['busy_seconds']:.1f}s "
                       f"({_pct(w['busy_seconds'], wall).strip()}), {w['calls']} calls")
    return out


def _separation(events):
    profiles = [e for e in events if e.get("event") == "separation_profile"]
    if not profiles:
        return []
    out = ["", "5. SEPARATION INSIDE (per file; parallel parts are summed)"]
    for e in profiles:
        v = e.get("values", {})
        n = max(v.get("windows", 0), 1)
        consumer = v.get("consumer", 0.0)
        out.append(f"  {e.get('file', '?')}: {int(v.get('windows', 0))} windows, "
                   f"ordered consumer {consumer:.1f}s ({consumer / n:.2f}s per window)")
        for label, key in (("waiting for Sidon result", "raw_wait"),
                           ("speaker assignment", "post"),
                           ("everything else in the consumer", "other")):
            out.append(f"    {label:34s} {v.get(key, 0.0):8.1f}s  "
                       f"{_pct(v.get(key, 0.0), consumer)}")
        if v.get("gpu_calls"):
            out.append(f"    Sidon jobs: {int(v['gpu_calls'])}; queued for a thread "
                       f"{v.get('gpu_queue', 0.0):.1f}s; running {v.get('gpu_run', 0.0):.1f}s")
        if v.get("sidon_calls"):
            calls = v["sidon_calls"]
            out.append(f"    Sidon per call: total {v['sidon_total'] / calls:.2f}s = "
                       f"GPU inference {v['sidon_infer'] / calls:.2f}s + worker file/copy "
                       f"{(v['sidon_worker'] - v['sidon_infer']) / calls:.2f}s + pipe "
                       f"{(v['sidon_roundtrip'] - v['sidon_worker']) / calls:.2f}s + parent "
                       f"file {(v['sidon_total'] - v['sidon_roundtrip']) / calls:.2f}s")
        parts = [(label, v.get(k, 0.0), int(v.get(k + "_calls", 0)))
                 for label, k in (("enrollment", "enrollment"), ("probe/VAD", "probe_vad"),
                                  ("WeSpeaker", "wespeaker"),
                                  ("assignment workers", "remote"))]
        if any(sec for _, sec, _ in parts):
            out.append("    assignment work (summed over parallel tasks): " + ", ".join(
                f"{label} {sec:.1f}s ({calls} calls)" for label, sec, calls in parts if sec))
        attempts = int(v.get("embedding_batch_attempts", 0))
        batches = int(v.get("embedding_batches", 0))
        batched_items = int(v.get("embedding_batched_items", 0))
        fallbacks = int(v.get("embedding_batch_fallbacks", 0))
        singles = int(v.get("embedding_single_calls", 0))
        if attempts or singles:
            out.append(
                "    WeSpeaker batching: "
                f"{batches}/{attempts} batches accepted, {batched_items} items batched, "
                f"{fallbacks} batch fallbacks, {singles} single-item calls")
        adaptive_total = int(v.get("adaptive_batch_observed_items", 0))
        adaptive_batched = int(v.get("adaptive_batch_batched_items", 0))
        fanout_items = int(v.get("adaptive_fanout_items", 0))
        if adaptive_total or fanout_items:
            hit_rate = (100.0 * adaptive_batched / adaptive_total
                        if adaptive_total else 0.0)
            mode = ("fan-out" if v.get("adaptive_batch_switches", 0)
                    else "exact-length batching")
            out.append(
                f"    similarity scheduler: {adaptive_batched}/{adaptive_total} "
                f"items batched ({hit_rate:.1f}% hit), mode {mode}, "
                f"{fanout_items} later fan-out items")
    return out


def _resources(events, samples):
    windows = [(e["stage"], e["time"] - e["seconds"], e["time"]) for e in events
               if e.get("event") == "stage_finished"]
    if not samples or not windows:
        return []
    rows = []
    for stage, start, end in sorted(windows, key=lambda w: _stage_key(w[0])):
        inside = [s for s in samples if start <= s["time"] <= end]
        if not inside:
            continue
        cpu = [s["cpu_pct"] for s in inside if s.get("cpu_pct") is not None]
        gpus = sorted({g for s in inside for g in s.get("gpu_util", {})})
        cells = [stage, len(inside),
                 f"{sum(cpu) / len(cpu):.0f}%" if cpu else "n/a"]
        for g in gpus:
            values = [s["gpu_util"][g] for s in inside if g in s.get("gpu_util", {})]
            cells.append(f"{sum(values) / len(values):.0f}% / {max(values):.0f}%")
        rows.append((cells, gpus))
    if not rows:
        return []
    all_gpus = sorted({g for _, gpus in rows for g in gpus})
    header = ["stage", "samples", "CPU avg"] + [f"GPU{g} avg/peak" for g in all_gpus]
    table = []
    for cells, gpus in rows:
        cells = cells[:3] + [cells[3 + gpus.index(g)] if g in gpus else "n/a"
                             for g in all_gpus]
        table.append(cells)
    return ["", "6. RESOURCE USE PER STAGE (sampled every second)",
            *_table(table, header)]


def _top(events):
    grouped = defaultdict(float)
    for e in events:
        if e.get("event") == "span" and e.get("name") != "file_stage":
            grouped[e["name"]] += e["seconds"]
    if not grouped:
        return []
    top = sorted(grouped.items(), key=lambda kv: -kv[1])[:10]
    return ["", "7. LARGEST STEPS OVERALL", *_table(
        [[name, f"{sec:.1f}"] for name, sec in top], ["step", "total s"])]


def _focused_stages(events):
    """Show the decomposition most useful for the A100 bottleneck review."""
    focus = ("asr", "refinement", "conversation_exports")
    stage_seconds = {e["stage"]: float(e["seconds"]) for e in events
                     if e.get("event") == "stage_finished"}
    grouped = defaultdict(lambda: [0, 0.0])
    for e in events:
        if e.get("event") != "span" or e.get("name") == "file_stage":
            continue
        stage = e.get("stage")
        if stage in focus:
            item = grouped[(stage, e.get("name", "?"))]
            item[0] += 1
            item[1] += float(e.get("seconds", 0.0))
    if not grouped:
        return []
    rows = []
    for (stage, name), (calls, seconds) in sorted(
            grouped.items(), key=lambda item: (_stage_key(item[0][0]), -item[1][1])):
        wall = stage_seconds.get(stage, 0.0)
        rows.append([stage, name, calls, f"{seconds:.1f}", _pct(seconds, wall)])
    return ["", "8. FOCUSED DECOMPOSITION: ASR / REFINEMENT / CONVERSATION",
            "   Nested spans are listed separately; their totals must not be summed as wall time.",
            *_table(rows, ["stage", "part", "calls", "seconds", "% of stage"])]


def build_report(events, samples, elapsed):
    lines = ["PIPELINE PERFORMANCE REPORT", "=" * 27, ""]
    for part in (_stages(events, elapsed), _file_stages(events), _steps(events),
                 _workers(events), _separation(events), _resources(events, samples),
                 _top(events), _focused_stages(events)):
        lines += part
    return "\n".join(lines) + "\n"
