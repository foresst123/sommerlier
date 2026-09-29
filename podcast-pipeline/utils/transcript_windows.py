"""Cut a transcript into prompt-sized windows, and print one segment as a line.

A podcast does not fit in one prompt, and `models.refinement.max_batch_tokens`
on the small profile is only 8192. So the transcript is read in overlapping
windows sized by tokens, not by segment count: a run of long turns and a run of
backchannels cost very different amounts.

The overlap exists to give a segment near a window's edge the context on both
sides in some other window. Each position is then *owned* by exactly one window
-- its core -- and only proposals inside that core are read. Owning is what
replaces voting between windows: nothing has to be reconciled, because nothing
is decided twice.

Pure functions over token counts, so the arithmetic is tested without a
tokenizer.
"""

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Window:
    start: int          # first position shown to the model
    stop: int           # one past the last position shown
    core_start: int     # first position this window owns
    core_stop: int      # one past the last position this window owns

    def shows(self, position: int) -> bool:
        return self.start <= position < self.stop

    def owns(self, position: int) -> bool:
        return self.core_start <= position < self.core_stop


def build_windows(token_counts: Sequence[int], budget: int,
                  overlap: int = 0) -> List[Window]:
    """Windows over `len(token_counts)` positions, each within `budget` tokens.

    A single position larger than the budget still gets a window of its own,
    rather than being dropped or looping forever: refusing to show a segment is
    a silent way of never checking it. Consecutive windows share `overlap`
    positions; the cores partition the whole range, each boundary sitting in the
    middle of the shared stretch.
    """
    n = len(token_counts)
    if n == 0:
        return []
    budget = max(1, int(budget))
    overlap = max(0, int(overlap))

    spans = []
    start = 0
    while start < n:
        stop, used = start, 0
        while stop < n and (stop == start or used + token_counts[stop] <= budget):
            used += token_counts[stop]
            stop += 1
        spans.append((start, stop))
        if stop >= n:
            break
        # Always move forward, even when the overlap is as long as the window.
        start = max(start + 1, stop - overlap)

    windows = []
    for i, (start, stop) in enumerate(spans):
        core_start = 0 if i == 0 else (spans[i][0] + spans[i - 1][1]) // 2
        core_stop = n if i == len(spans) - 1 else (spans[i + 1][0] + stop) // 2
        windows.append(Window(start, stop, core_start, core_stop))
    return windows


def line_number(value) -> Optional[int]:
    """The line number a model named, or None when what it wrote names no line.

    Models write 12, 12.0, "12", "#12", "[12]" and "dòng 12" for the same line;
    all of them are 12. Zero, negatives, fractions, booleans and prose are not
    lines -- lines count from 1.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        found = re.search(r"\d+", str(value))
        number = float(found.group()) if found else None
    if number is None or not math.isfinite(number) or number != int(number) or number < 1:
        return None
    return int(number)


def clock(seconds: float) -> str:
    """mm:ss.s, with minutes allowed past 59 -- a long recording has no hours."""
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def format_line(seg, number: int, name: str, locked: bool = False) -> str:
    """One segment as the model reads it: `[number] start-end NAME (gap) text`.

    The model points at a line by its `number` -- 1, 2, 3... within the window it
    was shown -- and never by the segment's own index. A short count is far
    easier to copy back correctly than a zero-padded id, and a wrong one is
    plainly out of range instead of quietly naming some other segment. `name` is
    the speaker as the model should call them (a letter, not the diarizer's
    label, which is often a bare digit that reads like a line number).

    `gap_before` is shown because it is the strongest timing evidence for a
    change of turn: a long pause suggests one, a negative one is an
    interruption. It is None when a cut lies in the silence, and then it is left
    out rather than shown as zero.
    """
    text = " ".join(str(getattr(seg, "text", "") or "").split())
    parts = [f"[{number}]", f"{clock(seg.start)}-{clock(seg.end)}", str(name)]
    if locked:
        parts.append("[cố định]")
    gap = getattr(seg, "gap_before", None)
    if gap is not None:
        parts.append(f"(gap {gap:+.1f}s)")
    parts.append(text)
    return " ".join(parts)
