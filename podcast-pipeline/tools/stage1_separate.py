"""Step two: run every separator over every detector's music spans.

Runs under its own interpreter (audio-separator pins numpy against the
pipeline's), reading the spans step one wrote and writing one cleaned wav per
(recording, detector, separator, span) plus a json of what it did.
"""
import json
import os
import sys
import time

OUT = sys.argv[1]
SEPARATORS = {
    "bsroformer_ep368": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "melband_denoise": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    "melband_denoise_aggr": "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
}
# Which produced stem is the one to keep. The denoise checkpoints emit
# (other, noise) rather than (vocals, instrumental): "other" is the cleaned
# signal. Guessing wrong here silently keeps the noise and throws the voice
# away, so the choice is made by name and checked against what came back.
KEEP = ("vocals", "other", "speech", "dry", "no reverb")


def pick(files):
    for want in KEEP:
        for f in files:
            if f"_({want})_".lower() in os.path.basename(f).lower():
                return f
    return files[0] if files else None


def main():
    from audio_separator.separator import Separator

    with open(os.path.join(OUT, "index.json")) as f:
        index = json.load(f)

    # Resume rather than restart. A separator takes over an hour here, and the
    # run has already died once mid-download; losing finished work to a broken
    # connection would make the next failure cost the same again.
    report_path = os.path.join(OUT, "separate.json")
    report = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)

    for sep_name, ckpt in SEPARATORS.items():
        todo = [k for k in index if f"{k}__{sep_name}" not in report]
        if not todo:
            print(f"[{sep_name}] da xong ca {len(index)} to hop, bo qua", flush=True)
            continue
        sep = Separator(output_dir=os.path.join(OUT, "sep", sep_name),
                        output_format="WAV", log_level=40)
        sep.load_model(model_filename=ckpt)

        for key in todo:
            entry = index[key]
            span_dir = os.path.join(OUT, "spans", key)
            # Assigning sep.output_dir here does nothing: audio-separator
            # reads it when the Separator is constructed. Rather than rebuild
            # one per key -- which would reload the checkpoint every time --
            # every key writes into sep/<separator>/ and is told apart by the
            # span filename, which carries index and timestamps.
            dest = os.path.join(OUT, "sep", sep_name)

            done, failed, seconds = [], [], 0.0
            t0 = time.time()
            for span in entry["separate"]:
                src = os.path.join(span_dir, span["file"])
                if not os.path.exists(src):
                    continue
                try:
                    made = sep.separate(src)
                    made = [m if os.path.isabs(m) else os.path.join(dest, m)
                            for m in made]
                    keep = pick(made)
                    done.append({**span, "kept": os.path.basename(keep) if keep
                                 else None,
                                 "stems": [os.path.basename(m) for m in made]})
                    seconds += span["end"] - span["start"]
                except Exception as exc:
                    failed.append({**span, "error": f"{type(exc).__name__}: {exc}"})

            report[f"{key}__{sep_name}"] = {
                "separator": sep_name, "checkpoint": ckpt,
                "spans_done": done, "spans_failed": failed,
                "audio_seconds": seconds, "wall_seconds": time.time() - t0,
            }
            print(f"[{sep_name}] {key}: {len(done)} ok, {len(failed)} fail, "
                  f"{seconds:.0f}s audio in {time.time()-t0:.0f}s", flush=True)

            with open(report_path, "w") as f:
                json.dump(report, f, indent=1)

    print("\nSEPARATE_DONE")


if __name__ == "__main__":
    main()
