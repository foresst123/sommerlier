"""Measuring MossFormer2 against Sidon on this corpus's own overlaps.

Sidon is a diffusion model with a VAE decoder: it resynthesises the target
rather than filtering the mixture, and three consequences of that were measured
on the separated spans of this recording --

    noise floor      -75.4 dB against the mixture's own -38.6 dB
    frames gated     46% of the output, against 6% in the mixture
    spectral tilt    -4.2 dB at 100-200 Hz, +3.5 dB at 1-4 kHz

A masking separator predicts a filter over the mixture, so it cannot invent a
tilt or gate a frame to digital silence: whatever it keeps is the recording.
That is the hypothesis. Whether it holds on Vietnamese conversational speech is
a separate question, and the reason for this script rather than a swap.

What it does NOT decide: which separator produces better transcripts. Both are
blind, both need ECAPA to say which output is which speaker, and the pipeline
already does that. This measures signal properties only -- run the ASR
comparison afterwards, on the winner.

Setup (the model is not a pipeline dependency):

    pip install clearvoice

Run:  python tools/compare_separators.py                (from podcast-pipeline/)
      python tools/compare_separators.py --dump out/    to listen
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MossFormer2_SS_16K is a 16 kHz model and the pipeline works at 24 kHz. The
# resample costs everything above 8 kHz, which on this corpus is 0.2% of total
# energy -- but 18% of the energy in fricative frames, where /s/, /x/ and /ch/
# live. That is the trade this script exists to price, so it is measured below
# rather than assumed away.
MOSSFORMER_SR = 16000
FRAME_MS = 20


def _load(path):
    import soundfile as sf
    audio, sr = sf.read(path)
    return (audio if audio.ndim == 1 else audio.mean(axis=1)), sr


def _frames(x, sr, ms=FRAME_MS):
    step = int(sr * ms / 1000)
    count = len(x) // step
    if count < 1:
        return np.zeros((0, step))
    return x[:count * step].reshape(-1, step)


def noise_floor_db(x, sr):
    """Quiet-frame level relative to the loud ones, in dB.

    The number that separates a recording from a gate: real rooms sit around
    -40 dB, and anything far below that is silence the separator inserted.
    """
    frames = _frames(x, sr)
    if not len(frames):
        return float("nan")
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-20)
    peak = np.percentile(rms, 95)
    if peak <= 1e-9:
        return float("nan")
    return float(20 * np.log10(np.percentile(rms, 10) / peak))


def gated_fraction(x, sr, floor_db=-50.0):
    """Share of frames pushed below `floor_db` of the loud level."""
    frames = _frames(x, sr)
    if not len(frames):
        return float("nan")
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-20)
    peak = np.percentile(rms, 95)
    if peak <= 1e-9:
        # Silent throughout. Measured against its own peak every frame counts
        # as loud, which reports a dead track as perfectly ungated; against the
        # recording it is entirely gated, which is the truth being reported.
        return 1.0
    return float((rms < peak * 10 ** (floor_db / 20)).mean())


def band_shape(x, sr, edges=(0, 100, 200, 300, 500, 700, 1000, 1400,
                             2000, 2800, 4000, 5600, 8000)):
    """Per-band share of total energy -- comparable regardless of loudness."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    bands = np.array([spec[(freqs >= lo) & (freqs < hi)].sum()
                      for lo, hi in zip(edges[:-1], edges[1:])])
    return bands / (bands.sum() + 1e-20)


def tilt_db(track, mixture, sr):
    """Mean absolute deviation from the mixture's spectral shape, in dB.

    Zero means the separator kept the recording's timbre; a large number means
    it imposed one of its own.
    """
    n = min(len(track), len(mixture))
    if n < sr // 4:
        return float("nan")
    if np.sqrt(np.mean(track[:n] ** 2)) < 1e-6:
        # A dead track has no spectrum to compare. Returning a number here gave
        # 120 dB, which is not a tilt -- it is the epsilon in the logarithm --
        # and it would dominate any median it landed in.
        return float("nan")
    ratio = band_shape(track[:n], sr) / (band_shape(mixture[:n], sr) + 1e-20)
    return float(np.abs(10 * np.log10(ratio + 1e-12)).mean())


def fricative_frames(mixture, sr):
    """Indices of the frames in `mixture` that carry fricative energy.

    Chosen once, on the recording, and then reused for every track measured
    against it. Picking them per-track looked equivalent and was not: Sidon
    removes the noise floor, which moves every percentile, so "the top 15% of
    ZCR" selected genuine consonants in the mixture and something else
    entirely in its output -- the two numbers were then not comparable at all
    (0.4% against 40%, where the whole-signal difference is 0.12% against
    0.16%).

    Fricatives are quiet, so the level test only discards silence; requiring
    them to be loud selected almost nothing.
    """
    frames = _frames(mixture, sr)
    if len(frames) < 10:
        return None
    zcr = (np.diff(np.sign(frames), axis=1) != 0).mean(axis=1)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-20)
    picked = np.where((zcr > np.percentile(zcr, 85))
                      & (rms > np.percentile(rms, 30)))[0]
    return picked if len(picked) >= 3 else None


def fricative_hf_share(x, sr, index=None, lo=4000.0, hi=8000.0):
    """Energy in [lo, hi) over the fricative frames, as a share of their total.

    The band stops at 8 kHz on purpose: above it a 16 kHz model's output is
    exactly zero by construction, so including that range would score the
    resample rather than the separator, identically for any 16 kHz model.
    """
    frames = _frames(x, sr)
    if not len(frames):
        return float("nan")
    if index is None:
        index = fricative_frames(x, sr)
        if index is None:
            return float("nan")
    index = index[index < len(frames)]
    if len(index) < 3:
        return float("nan")
    picked = frames[index]
    if np.sqrt(np.mean(picked ** 2)) < 1e-9:
        return float("nan")
    spec = np.abs(np.fft.rfft(picked * np.hanning(frames.shape[1]), axis=1)) ** 2
    freqs = np.fft.rfftfreq(frames.shape[1], 1.0 / sr)
    return float(spec[:, (freqs >= lo) & (freqs < hi)].sum() / (spec.sum() + 1e-20))


def measure(track, mixture, sr, fricatives=None):
    return {
        "floor_db": noise_floor_db(track, sr),
        "gated": gated_fraction(track, sr),
        "tilt_db": tilt_db(track, mixture, sr),
        "fricative_hf": fricative_hf_share(track, sr, index=fricatives),
    }


def run_mossformer(mixture, sr, model=None):
    """Separate one mixture, returning two tracks at the input rate.

    The model works at 16 kHz; the audio goes down and comes back up so every
    measurement is made on the same timebase as Sidon's output. That round trip
    is part of what is being priced, not an artefact to correct for.
    """
    import librosa

    if model is None:
        from clearvoice import ClearVoice
        model = ClearVoice(task="speech_separation",
                           model_names=["MossFormer2_SS_16K"])

    audio = mixture if sr == MOSSFORMER_SR else librosa.resample(
        mixture, orig_sr=sr, target_sr=MOSSFORMER_SR)
    batch = np.reshape(audio, [1, audio.shape[0]]).astype(np.float32)

    # [speaker, batch, length]
    separated = model(batch, False)

    out = []
    for speaker in range(separated.shape[0]):
        track = separated[speaker, 0, :]
        if sr != MOSSFORMER_SR:
            track = librosa.resample(track, orig_sr=MOSSFORMER_SR, target_sr=sr)
        out.append(track[:len(mixture)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="per-file directory under _final/")
    ap.add_argument("--dump", default=None, help="write audio here to listen")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = os.path.abspath(args.run_dir) if args.run_dir else None
    if run is None:
        found = glob.glob(os.path.join(root, "..", "kaggle*", "working", "vi_audio",
                                       "_final", "*", "*"))
        found = [p for p in found if os.path.isdir(os.path.join(p, "02_separation"))]
        if not found:
            sys.exit("no run directory found; pass one explicitly")
        run = found[0]

    pairs_dir = os.path.join(run, "02_separation", "audio", "raw", "separated")
    mixes = sorted(glob.glob(os.path.join(pairs_dir, "*_mix.wav")))
    if args.limit:
        mixes = mixes[:args.limit]
    if not mixes:
        sys.exit(f"no mixture/track dumps under {pairs_dir}")

    try:
        from clearvoice import ClearVoice
        model = ClearVoice(task="speech_separation",
                           model_names=["MossFormer2_SS_16K"])
    except Exception as exc:
        sys.exit(f"MossFormer2 unavailable ({exc}).\n"
                 "Install it with:  pip install clearvoice")

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    rows = {"sidon": [], "mossformer": []}
    mixture_rows = []
    used = 0

    for mix_path in mixes:
        try:
            mixture, sr = _load(mix_path)
        except Exception:
            continue

        # Every track Sidon produced, including one that came back silent.
        # Dropping the silent ones would have compared Sidon's better half
        # against all of MossFormer2: emitting one source and silence is a
        # failure mode this is meant to catch, not an entry to exclude.
        sidon = []
        for tag in ("trackA", "trackB"):
            path = mix_path.replace("_mix", f"_{tag}")
            if not os.path.exists(path):
                continue
            try:
                track, _ = _load(path)
            except Exception:
                continue
            sidon.append(track)
        if not sidon:
            continue

        try:
            moss = run_mossformer(mixture, sr, model)
        except Exception as exc:
            print(f"  MossFormer2 failed on {os.path.basename(mix_path)}: {exc}")
            continue

        # One frame selection per mixture, shared by every track measured
        # against it, so the fricative numbers describe the same moments.
        fricatives = fricative_frames(mixture, sr)
        mixture_rows.append(measure(mixture, mixture, sr, fricatives))
        for track in sidon:
            rows["sidon"].append(measure(track, mixture, sr, fricatives))
        for track in moss:
            rows["mossformer"].append(measure(track, mixture, sr, fricatives))

        if args.dump:
            import soundfile as sf
            tag = os.path.basename(mix_path).replace("_mix.wav", "")
            sf.write(os.path.join(args.dump, f"{tag}_0_mix.wav"), mixture, sr)
            for i, track in enumerate(sidon):
                sf.write(os.path.join(args.dump, f"{tag}_1_sidon_{i}.wav"), track, sr)
            for i, track in enumerate(moss):
                sf.write(os.path.join(args.dump, f"{tag}_2_moss_{i}.wav"), track, sr)
        used += 1

    if not used:
        sys.exit("nothing could be measured")

    def med(source, key):
        values = [r[key] for r in source if not np.isnan(r[key])]
        return np.median(values) if values else float("nan")

    def dead(source):
        """Tracks that came back with no signal at all."""
        return sum(1 for r in source if r["gated"] >= 0.999)

    print(f"mixtures: {used}   sidon tracks: {len(rows['sidon'])}   "
          f"mossformer tracks: {len(rows['mossformer'])}")
    print(f"silent tracks: sidon {dead(rows['sidon'])}, "
          f"mossformer {dead(rows['mossformer'])}\n")
    print(f"{'':12} {'floor dB':>9} {'gated':>7} {'tilt dB':>8} {'fric 4-8k':>10}")
    print(f"{'mixture':12} {med(mixture_rows, 'floor_db'):9.1f} "
          f"{med(mixture_rows, 'gated') * 100:6.1f}% {0.0:8.2f} "
          f"{med(mixture_rows, 'fricative_hf') * 100:9.1f}%")
    for name in ("sidon", "mossformer"):
        print(f"{name:12} {med(rows[name], 'floor_db'):9.1f} "
              f"{med(rows[name], 'gated') * 100:6.1f}% "
              f"{med(rows[name], 'tilt_db'):8.2f} "
              f"{med(rows[name], 'fricative_hf') * 100:9.1f}%")

    print("\nfloor dB   quiet-frame level vs loud; the mixture's own value is the target")
    print("gated      frames below -50 dB; high means silence was inserted")
    print("tilt dB    mean |deviation| from the mixture's spectral shape; 0 is faithful")
    print("fric 4-8k  energy in 4-8 kHz within fricative frames -- what 16 kHz costs")
    print("           (above 8 kHz is zero for any 16 kHz model, so it is excluded)")

    if args.dump:
        print(f"\naudio written to {args.dump}")


if __name__ == "__main__":
    main()
