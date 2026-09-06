"""How much audio to hand the separator around one overlap.

USEF-TSE's ONNX graph takes a fixed 2-second mixture window; anything longer is
cut into 2s pieces and the outputs concatenated, which is what its own README
prescribes. So the only question here is what to do with an overlap *shorter*
than 2 seconds -- a 0.34s backchannel, which on this corpus is the common case.

The answer is not "pad it with silence". The model has to hear the voices it is
separating, and two seconds of real audio around a backchannel is what lets it.
Padding is the fallback for when the recording cannot supply that, not the plan.

What this replaces: a window assembled out of five seconds of each speaker's
solo speech plus the overlap, 6-12s in all. That existed because DialogueSidon
separates blind -- it has to find both voices in the mixture itself, and at a
lopsided balance it collapses to "one source carries everything". USEF-TSE is
told who to extract by an 8-second enrollment, so the mixture does not have to
carry that evidence any more.

The one thing the widening must respect is a seam: excising sung and
standalone-music stretches leaves points where two parts of the recording that
were never adjacent now touch. Widening across one drags in audio from
somewhere else entirely -- and worse, audio whose speaker is unrelated to the
overlap being separated.
"""

import os

# The model's fixed window. Not a preference: the ONNX graph bakes TF-GridNet's
# unfold constants in at [1, 16000] @ 8 kHz.
MODEL_WINDOW = float(os.environ.get("TSE_MODEL_WINDOW", "2.0"))


def bounds(lo, hi, seams, duration, minimum=None):
    """The nearest wall on each side of [lo, hi], in the same clock.

    Walls are the seams plus the two ends of the recording. An overlap sitting
    between two seams can only ever be widened inside that stretch.
    """
    minimum = MODEL_WINDOW if minimum is None else minimum
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
    minimum = MODEL_WINDOW if minimum is None else minimum
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
