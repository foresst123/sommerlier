"""Where a separator's window is allowed to reach.

Only the walls live here now. How far to actually grow inside them belongs to
SeparationService._build_window, because the answer depends on where each
speaker's solo audio is -- which is diarization's business, not this module's.

The walls are excise seams. Cutting sung and standalone-music stretches out
leaves points where two parts of the recording that were never adjacent now
touch. Growing a window across one drags in audio from somewhere else entirely,
and worse, audio whose speakers have nothing to do with the overlap being
separated. That constraint is independent of which separator runs.

This module briefly held the whole sizing policy, pinned at USEF-TSE's fixed
2-second window -- a real constraint of its ONNX graph, and the wrong one for
anything else. `widen` and `window_for` are kept for callers that only need a
span of a given length centred on an overlap.
"""

import os

# The window the separator wants around an overlap. This was 2.0s while
# USEF-TFGridNet was the backend, and that was not a preference: its ONNX graph
# bakes TF-GridNet's unfold constants in at [1, 16000] @ 8 kHz, so 2s was the
# only length it accepted.
#
# Sidon has no such constraint and the opposite need. It is blind -- nobody
# tells it who is in the mixture -- so ECAPA has to work out which returned
# track is whose, and it can only do that from stretches where one speaker is
# audible alone. A 2s window centred on an overlap contains no such stretch by
# construction. Sidon also chunks internally at 20s (CHUNK_SECONDS in
# sidon_infer.py), so anything shorter runs it far outside its design point.
#
# Measured on one recording: 2s windows with empty probes gave ECAPA
# similarity p50 0.15, which is what two unrelated speakers score.
WINDOW_TARGET = float(os.environ.get("BSS_WINDOW_TARGET", "15.0"))


def bounds(lo, hi, seams, duration, minimum=None):
    """The nearest wall on each side of [lo, hi], in the same clock.

    Walls are the seams plus the two ends of the recording. An overlap sitting
    between two seams can only ever be widened inside that stretch.
    """
    minimum = WINDOW_TARGET if minimum is None else minimum
    floor_, ceil_ = 0.0, float(duration)
    for seam in seams or ():
        if seam <= lo and seam > floor_:
            floor_ = seam
        if seam >= hi and seam < ceil_:
            ceil_ = seam
    return floor_, ceil_


def widen(lo, hi, floor_, ceil_, minimum=None):
    """Grow [lo, hi] to `minimum` seconds without crossing floor_ or ceil_.

    Even on both sides, because the overlap is what matters and centring it
    gives the model the same amount of run-up and run-out. When one side runs
    into a wall -- a seam, or the start or end of the recording -- the shortfall
    moves to the other side rather than being given up: the model needs its two
    seconds more than it needs the overlap centred.

    Returns (lo, hi) and does not promise `minimum` seconds; a stretch between
    two seams can be shorter than that, and the caller pads what is missing,
    which is what the ONNX contract says to do.
    """
    minimum = WINDOW_TARGET if minimum is None else minimum
    need = minimum - (hi - lo)
    if need <= 0:
        return lo, hi

    room_left, room_right = max(0.0, lo - floor_), max(0.0, ceil_ - hi)
    take_left = min(need / 2.0, room_left)
    take_right = min(need / 2.0, room_right)

    # Whatever one side could not give, ask the other for.
    short = need - take_left - take_right
    if short > 1e-9:
        extra = min(short, room_left - take_left)
        take_left += extra
        short -= extra
    if short > 1e-9:
        take_right += min(short, room_right - take_right)

    return lo - take_left, hi + take_right


def window_for(lo, hi, seams, duration, minimum=None):
    """The span to hand the separator for the overlap [lo, hi]."""
    floor_, ceil_ = bounds(lo, hi, seams, duration, minimum)
    return widen(lo, hi, floor_, ceil_, minimum)
