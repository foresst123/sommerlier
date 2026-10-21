# compare_data.json (version 1)

Written by `python tools/build_compare_data.py --manifest OUT/variants_manifest.json --out compare_data.json`.
Any metric may be `null` (stage missing, not computable). Extra keys are allowed.

## Top level
- `baseline`: variant id everything is compared against (`"none"`).
- `variants`: `[{id, chain, label, status, note?, group?}]` (`group` is an optional tag from the variants file, shown next to the chip; status `skipped` = backend unavailable, reason in `note`). Variants with status `failed`/`missing` are
  listed (with a `note`) but have no data below.
- `files[]`: one entry per file stem: `name`, `duration` (s), `audio`, `text`, `text_diff`,
  `speakers`, `metrics`; all keyed by variant id (the baseline included).
- `summary[variant][section.metric]`: mean over files of every numeric metric (bools count as 0/1;
  files where it is null are skipped).

## Time base
All times are seconds on `01_music/after_music.wav` (the "cut" timeline), the same one the stage
outputs use. A denoise stage keeps length, so the filtered wav shares it (see `lag_samples`).

## files[].audio
`raw` = abs path of `01_music/after_music.wav`; `<variant id>` = that variant's
`01b_denoise/after_denoise.wav` (baseline: same as raw; null if absent).

## files[].text[variant]
`segments`: `[{start,end,speaker,text}]` from `05_asr/transcripts.json` (`text` is the ensemble
text); `full_text`: segments in time order joined by spaces.

## files[].text_diff[variant]
- `wer_vs_baseline`, `ops {sub,ins,del,equal}`: word edits turning the baseline transcript into
  the variant's (baseline = reference). `wer_vs_reference`: same against the human reference, or null.
- `segments[]`: one per baseline segment: `start,end`, `baseline_text`, `variant_text` (both
  normalised: NFC, lowercase, punctuation removed, diacritics kept), `ops`
  (`["equal","w"] ["sub","base","var"] ["del","w"] ["ins","w"]`), `wer`
  (errors / max(baseline words,1)). Empty list for the baseline itself.
- Alignment is by time: variant words go to the baseline segment they overlap most (word
  timestamps when a segment's `words` match its text token for token, otherwise spread uniformly
  over the words' span / the segment span). A word moved across a baseline segment boundary counts
  as an error on both sides.

## files[].speakers[variant]
From `02_diarization/segments.json` on a 10 ms grid (overlap supported).
`n_speakers`, `speaker_seconds{label:s}`, `label_map{variant_label:baseline_label|null}` (one-to-one,
maximum shared seconds), `disagree_seconds`, `disagree_ranges[{start,end,baseline,variant,kind}]`
(labels joined by `+`; `kind`: `swap` both speak with different mapped labels, `missing` baseline
speech but variant silent, `extra` variant speech but baseline silent, `overlap` only one of the
two has overlapped speech), `der_vs_baseline` = (missed + false alarm + confusion) / baseline
speaker-seconds.

## files[].metrics[variant]
Direction: **up** = higher is better, **down** = lower is better, **neutral** = interpret, no
direction. Differences from the baseline are what matter, not absolute values.

audio (variant wav vs raw):
- `noise_floor_db` (10th percentile of 25 ms frame RMS, dBFS): down is "cleaner" but over-gating also lowers it; neutral.
- `gated_frame_pct` (% frames below -60 dBFS): neutral (rises when a filter gates/zeros pauses).
- `lag_samples` (variant delay vs raw, first 60 s, +-1 s search): down, 0 is ideal (|value|).
- `corr_with_raw` (Pearson after removing lag): neutral; near 1 = filter barely changed signal.
- `hf_energy_ratio` (energy > 4 kHz / total): neutral (low = muffled, high = hiss kept).
- `length_equal` (bool): up.
- `ecapa_cos` (speaker-embedding cosine, first 30 s, raw vs variant): up. Computed only when
  `WESPEAKER_MODEL_PATH` points to a local ONNX file (WeSpeaker, not ECAPA; key kept per contract); else null.
- `learned_mos` (dict or null): `{mos, dnsmos_ovrl, scoreq, utmos, nisqa_mos}`, each about 1-5, of the
  variant wav; `learned_mos_delta` (dict or null): the same keys, variant minus raw (the baseline's
  own audio is the raw wav, so its delta is 0). Both null unless the optional Uni-VERSA-Ext
  predictor is configured (env `UNIVERSA_MODEL`, `UNIVERSA_CONFIG`, `UNIVERSA_REPO`; see
  `learned_mos.py`) and `embed` is on. Direction: **neutral**. In `summary` the keys are flattened:
  `audio.learned_mos.<key>`, `audio.learned_mos_delta.<key>`. Read it like this:
  - It is a distilled mimic of DNSMOS/UTMOS/NISQA/SCOREQ plus a human-MOS head, NOT an independent
    replacement for DNSMOS: it inherits their blind spots. Trained on English/Mandarin, never on Vietnamese.
  - Learned MOS predictors reward hallucinated, fluent-sounding output (URGENT 2025 overview,
    arXiv 2505.23212: the purely generative system topped DNSMOS/NISQA/UTMOS while hallucinating;
    arXiv 2506.12260: an enhancer trained against the predictor produced non-speech it rated highly).
  - It runs at 16 kHz, so it is blind above 8 kHz of a 24 kHz input (HF artefacts, the 8-12 kHz band).
  - The HF weights carry no license and were trained on CC BY-NC-SA corpora: internal analysis only.
  - Rule: delta only, never a gate or a loss; always read it next to `wer_vs_baseline` and `ecapa_cos`.

diarization: `n_speakers` neutral (2 expected for a two-person conversation), `n_segments` neutral,
`overlap_seconds` neutral, `median_segment_s` up (fragmentation lowers it),
`short_segments_lt_0_5s` down.

separation: `overlap_spans` (spliced + failed) neutral, `segments_separated` neutral,
`failed_spans` down, `mean_similarity` (mean of `bss_spans` similarity, >= 0 only) up.

asr: `n_words` neutral, `n_segments` neutral, `empty_segments` down, `words_per_sec` neutral
(words / summed segment seconds; very high or low = hallucination or dropout),
`wer_vs_baseline` neutral (distance, not accuracy), `wer_vs_reference` down,
`words_in_quiet_frames` down (words whose whole span is below -50 dBFS in the variant audio,
likely hallucinated; null when no word timestamps).

Stage layout read: `01_music/after_music.wav`, `01b_denoise/{after_denoise.wav,denoise.json}`,
`02_diarization/segments.json`, `03_separation/{segments,stats}.json`, `05_asr/transcripts.json`.
