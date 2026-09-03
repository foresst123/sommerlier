"""Measuring whether inpainting a spliced span beats what separation produced.

The question this answers is narrow and worth stating plainly. Where two people
overlap, TSE replaces a span of the recording with Sidon's reconstruction of one
speaker. That span is short -- 0.42s at the median across 48 of them -- and its
two edges are the *same speaker continuing to talk*, which measures at r=0.91
spectral correlation. Those are the conditions inpainting is built for, so the
question is whether filling the span from its edges beats resynthesising it.

Two methods are compared against the mixture, per span:

  sidon    what the pipeline produces today
  bridge   the span replaced by a cross-fade between the audio either side,
           with the measured room floor underneath

`bridge` is deliberately not a model. It is the cheapest thing that respects the
two-sided constraint, and it exists to set the bar: a generative inpainter has to
beat this to be worth its weight, its language mismatch and its inference cost.
SIEDD (arXiv 2608.06424) is the obvious candidate to slot in beside it -- gaps of
250-1500ms, no transcript needed -- but its code is not published (the repository
in the paper 404s and nothing matching is on the Hub), and it is trained on
English single-speaker audio, so tone-language behaviour on overlapped speech is
unknown. Hence a placeholder method rather than a dependency.

Nothing here is scored against a reference transcript, because there isn't one
per span. What is measured is agreement with the mixture over the span -- how
much of what was actually recorded survives -- plus continuity at the two seams,
which is what makes a splice audible.

Run:  python tools/try_inpaint.py            (from podcast-pipeline/)
      python tools/try_inpaint.py --dump out/    to write audio to listen to
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 20 ms at 24 kHz: short enough to sit inside a phone, long enough that phase
# does not dominate the frame energy.
FRAME_MS = 20


def _load(path):
    import soundfile as sf
    audio, sr = sf.read(path)
    return (audio if audio.ndim == 1 else audio.mean(axis=1)), sr


def _spectrum(x):
    """Log power spectrum, normalised -- compares shape, not loudness."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    return np.log(spec / (spec.sum() + 1e-20) + 1e-12)


def spectral_agreement(a, b):
    """Correlation between two log spectra, in [-1, 1]."""
    n = min(len(a), len(b))
    if n < 64:
        return float("nan")
    return float(np.corrcoef(_spectrum(a[:n]), _spectrum(b[:n]))[0, 1])


def seam_step(track, index, sr):
    """Level jump in dB across a boundary, measured over one frame each side.

    A splice is heard at its edges before it is heard in the middle, so this is
    the number that predicts whether a listener notices the join at all.
    """
    frame = int(sr * FRAME_MS / 1000)
    lo, hi = index - frame, index + frame
    if lo < 0 or hi > len(track):
        return float("nan")
    before = np.sqrt(np.mean(track[lo:index] ** 2)) + 1e-12
    after = np.sqrt(np.mean(track[index:hi] ** 2)) + 1e-12
    return float(abs(20 * np.log10(after / before)))


def room_floor(mixture, sr, exclude, quantile=10):
    """The recording's own noise floor, in RMS.

    Taken from the quiet frames of the mixture outside the span, so the fill
    sits on the same floor as the audio around it rather than on digital
    silence -- which is what makes a gap sound switched off rather than quiet.
    """
    frame = int(sr * FRAME_MS / 1000)
    usable = np.concatenate([mixture[:exclude[0]], mixture[exclude[1]:]])
    count = len(usable) // frame
    if count < 5:
        return 0.0
    frames = usable[:count * frame].reshape(-1, frame)
    return float(np.percentile(np.sqrt((frames ** 2).mean(axis=1)), quantile))


def bridge(mixture, sr, start, end, context=0.15):
    """Fill [start, end) by cross-fading its two edges over the room floor.

    Not an attempt at speech: it is the two-sided constraint honoured as
    cheaply as possible, so that anything generative has something to beat.
    """
    span = end - start
    if span <= 0:
        return np.zeros(0)

    pad = int(context * sr)
    left = mixture[max(0, start - pad):start]
    right = mixture[end:min(len(mixture), end + pad)]
    if len(left) < 64 or len(right) < 64:
        return np.full(span, room_floor(mixture, sr, (start, end)))

    # Tile each side out to the full span, then cross-fade. Reversing the tail
    # of each avoids the click that repeating a waveform head-to-tail creates.
    def stretch(chunk, length):
        out = np.zeros(length)
        pos = 0
        flip = False
        while pos < length:
            piece = chunk[::-1] if flip else chunk
            take = min(len(piece), length - pos)
            out[pos:pos + take] = piece[:take]
            pos += take
            flip = not flip
        return out

    fade = np.linspace(0.0, 1.0, span)
    filled = stretch(left[::-1], span)[::-1] * (1 - fade) + stretch(right, span) * fade

    # Level the fill to its own edges, not to the room floor. Both edges are
    # the same speaker mid-sentence, so the span is speech continuing -- forcing
    # it down to the noise floor punches a hole where a syllable belongs, which
    # measured as zero agreement with the mixture and a 29 dB seam. The floor is
    # only the right target when the edges are themselves quiet, which the
    # interpolation below handles by construction.
    edges = 0.5 * (np.sqrt(np.mean(left ** 2)) + np.sqrt(np.mean(right ** 2)))
    target = max(edges, room_floor(mixture, sr, (start, end)))
    level = np.sqrt(np.mean(filled ** 2)) + 1e-12
    return filled * (target / level)


def evaluate(mixture, separated, sr, start, end):
    """Both methods over one span, scored the same way."""
    span_mix = mixture[start:end]

    rebuilt = separated.copy()
    rebuilt[start:end] = bridge(mixture, sr, start, end)

    rows = {}
    for name, track in (("sidon", separated), ("bridge", rebuilt)):
        rows[name] = {
            "agreement": spectral_agreement(track[start:end], span_mix),
            "seam_in": seam_step(track, start, sr),
            "seam_out": seam_step(track, end, sr),
            "level_db": float(20 * np.log10(
                (np.sqrt(np.mean(track[start:end] ** 2)) + 1e-12)
                / (np.sqrt(np.mean(span_mix ** 2)) + 1e-12))),
        }
    return rows, rebuilt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="per-file directory under _final/")
    ap.add_argument("--dump", default=None, help="write per-span audio here")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.run_dir:
        run = os.path.abspath(args.run_dir)
    else:
        found = glob.glob(os.path.join(
            root, "..", "kaggle 2", "working", "vi_audio", "_final", "*", "hoahau"))
        if not found:
            sys.exit("no run directory found; pass one explicitly")
        run = found[0]

    name = os.path.basename(run)
    inter = os.path.join(run, f"{name}_intermediate_separation.json")
    if not os.path.exists(inter):
        sys.exit(f"missing {inter}")

    with open(inter, encoding="utf-8") as fh:
        payload = json.load(fh)
    segments = payload if isinstance(payload, list) else payload.get("segments", payload)

    # The recording sits above _final/, whose depth depends on how the run was
    # laid out; walk up rather than assuming a fixed number of levels.
    source = []
    probe = run
    for _ in range(5):
        probe = os.path.dirname(probe)
        source = [p for p in glob.glob(os.path.join(probe, f"{name}.*"))
                  if p.lower().endswith((".wav", ".mp3", ".m4a", ".flac"))]
        if source:
            break
    if not source:
        sys.exit(f"could not find the original recording for {name!r}")
    mixture, sr = _load(source[0])

    # One overlap yields a span on both speakers' segments, and they are not
    # duplicates -- each is that speaker's own track over the same moment. They
    # are, however, opposite cases:
    #
    #   the interrupted speaker  span is a fraction of a longer turn, edges are
    #                            their voice continuing -> inpainting applies
    #   the backchannel speaker  the segment *is* the span, so there are no
    #                            edges to interpolate from, and filling it would
    #                            erase the only thing they said
    #
    # Only the first is a candidate. Requiring real context on both sides is
    # what separates them, and it is also what inpainting needs to work at all.
    context_needed = 0.10
    spans = [(s, a, b) for s in segments
             for a, b, *_ in (s.get("tse_spans") or [])
             if a - float(s["start"]) >= context_needed
             and float(s["end"]) - b >= context_needed]
    if args.limit:
        spans = spans[:args.limit]
    if not spans:
        sys.exit("no spliced spans in this run")

    audio_dir = os.path.join(run, "02_separation", "audio", "separated")
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    results, skipped = [], 0
    for seg, a, b in spans:
        index = str(seg.get("index", ""))
        match = glob.glob(os.path.join(audio_dir, f"{index}_*.wav"))
        if not match:
            skipped += 1
            continue
        separated, sr_sep = _load(match[0])

        # The separated clip covers the segment; the span is absolute time.
        offset = float(seg["start"])
        start = int(round((a - offset) * sr_sep))
        end = int(round((b - offset) * sr_sep))
        if start < 0 or end > len(separated) or end <= start:
            skipped += 1
            continue

        # The mixture is read at its own rate; align by resampling the slice.
        mix_slice = mixture[int(offset * sr):int(float(seg["end"]) * sr)]
        if sr != sr_sep and len(mix_slice):
            import torch
            import torchaudio.functional as AF
            mix_slice = AF.resample(
                torch.from_numpy(mix_slice).float(), sr, sr_sep).numpy()
        if len(mix_slice) < end:
            skipped += 1
            continue

        rows, rebuilt = evaluate(mix_slice, separated, sr_sep, start, end)
        rows["span"] = b - a
        rows["index"] = index
        results.append(rows)

        if args.dump:
            import soundfile as sf
            tag = f"{index}_{b - a:.2f}s"
            sf.write(os.path.join(args.dump, f"{tag}_1_mix.wav"),
                     mix_slice[:len(separated)], sr_sep)
            sf.write(os.path.join(args.dump, f"{tag}_2_sidon.wav"), separated, sr_sep)
            sf.write(os.path.join(args.dump, f"{tag}_3_bridge.wav"), rebuilt, sr_sep)

    if not results:
        sys.exit("no span could be evaluated")

    print(f"spans evaluated: {len(results)}   skipped: {skipped}")
    lengths = np.array([r["span"] for r in results])
    print(f"span length: p50={np.median(lengths):.2f}s  "
          f"min={lengths.min():.2f}  max={lengths.max():.2f}\n")

    print(f"{'':8} {'agreement':>10} {'seam in':>9} {'seam out':>9} {'level':>8}")
    for method in ("sidon", "bridge"):
        rows = [r[method] for r in results]
        def med(key):
            values = [r[key] for r in rows if not np.isnan(r[key])]
            return np.median(values) if values else float("nan")
        print(f"{method:8} {med('agreement'):10.3f} {med('seam_in'):9.1f} "
              f"{med('seam_out'):9.1f} {med('level_db'):8.1f}")

    print("\nagreement: spectral correlation with the mixture over the span "
          "(higher = more of what was recorded survives)")
    print("seam:      level step in dB at the splice edge (lower = less audible)")
    print("level:     span loudness vs the mixture, dB")

    if args.dump:
        print(f"\naudio written to {args.dump}")


if __name__ == "__main__":
    main()
