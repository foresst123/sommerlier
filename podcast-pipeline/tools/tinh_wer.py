import csv
import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from jiwer import process_words, process_characters


# ============================================================
# CONFIG
# ============================================================

HTML_FILE = (
    "/home/lamkd2/Documents/sommerlier/"
    "podcast-pipeline/tools/hoahau_review(2).html"
)

OUTPUT_DIR = (
    Path(HTML_FILE).parent / "asr_evaluation"
)

MODELS = [
    "Whisper",
    "PhoWhisper",
    "Qwen3",
    "Final",
]

BASE_MODELS = [
    "Whisper",
    "PhoWhisper",
    "Qwen3",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def clean_text(text):
    """
    Chuẩn hóa text trước khi tính WER/CER.

    - lowercase
    - punctuation -> space
    - normalize whitespace
    """

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = " ".join(text.split())

    return text


# ============================================================
# HTML PARSER
# ============================================================

def parse_html(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        html = f.read()

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    rows = soup.find_all(
        "tr",
        attrs={"data-i": True},
    )

    segments = []

    for row in rows:

        segment_id = row.get(
            "data-i",
            "",
        )

        # ----------------------------------------------------
        # Speaker / ID
        # ----------------------------------------------------

        speaker = ""

        id_cell = row.find(
            "td",
            class_="id",
        )

        if id_cell:

            speaker = id_cell.get_text(
                " ",
                strip=True,
            )

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        start = ""
        end = ""
        duration = ""

        time_cell = row.find(
            "td",
            class_="time",
        )

        if time_cell:

            values = list(
                time_cell.stripped_strings
            )

            if len(values) >= 1:
                start = values[0]

            if len(values) >= 2:
                end = values[1]

            if len(values) >= 3:
                duration = values[2]

        # ----------------------------------------------------
        # ASR
        # ----------------------------------------------------

        asr = {
            "Whisper": "",
            "PhoWhisper": "",
            "Qwen3": "",
        }

        asr_cell = row.find(
            "td",
            class_="asr",
        )

        if asr_cell:

            for div in asr_cell.find_all(
                "div"
            ):

                span = div.find("span")

                if span is None:
                    continue

                model_name = span.get_text(
                    strip=True
                )

                # Clone text without model label
                span.extract()

                text = div.get_text(
                    " ",
                    strip=True,
                )

                if model_name in asr:
                    asr[model_name] = text

        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        final = ""

        final_cell = row.find(
            "td",
            class_="final",
        )

        if final_cell:

            final = final_cell.get_text(
                " ",
                strip=True,
            )

        # ----------------------------------------------------
        # EDIT / REFERENCE
        # ----------------------------------------------------

        reference = ""

        edit_box = row.find(
            "textarea",
            class_="edit",
        )

        if edit_box:

            reference = edit_box.get_text()

        # ----------------------------------------------------
        # NOTE
        # ----------------------------------------------------

        note = ""

        note_box = row.find(
            "textarea",
            class_="note",
        )

        if note_box:

            note = note_box.get_text(
                strip=True,
            )

        # ----------------------------------------------------
        # Changed flag
        # ----------------------------------------------------

        changed = "changed" in (
            row.get("class") or []
        )

        segments.append({
            "id": segment_id,

            "speaker": speaker,

            "start": start,
            "end": end,
            "duration": duration,

            "reference": reference,

            "Whisper": asr["Whisper"],
            "PhoWhisper": asr["PhoWhisper"],
            "Qwen3": asr["Qwen3"],

            "Final": final,

            "note": note,

            "changed": changed,
        })

    return segments


# ============================================================
# SAFE JIWER
# ============================================================

def jiwer_words(
    reference,
    hypothesis,
):

    """
    IMPORTANT:

    JiWER expects a list of sentences when using
    process_words(refs, hyps).

    Therefore for one segment we ALWAYS use:

        [reference]
        [hypothesis]

    """

    return process_words(
        [reference],
        [hypothesis],
    )


# ============================================================
# GLOBAL WORD METRICS
# ============================================================

def calculate_word_metrics(
    references,
    hypotheses,
):

    result = process_words(
        references,
        hypotheses,
    )

    hits = result.hits
    substitutions = result.substitutions
    deletions = result.deletions
    insertions = result.insertions

    reference_words = (
        hits
        + substitutions
        + deletions
    )

    hypothesis_words = (
        hits
        + substitutions
        + insertions
    )

    errors = (
        substitutions
        + deletions
        + insertions
    )

    if reference_words > 0:

        wer_value = (
            errors
            / reference_words
        )

        accuracy = (
            hits
            / reference_words
        )

        substitution_rate = (
            substitutions
            / reference_words
        )

        deletion_rate = (
            deletions
            / reference_words
        )

        insertion_rate = (
            insertions
            / reference_words
        )

    else:

        wer_value = 0.0
        accuracy = 1.0
        substitution_rate = 0.0
        deletion_rate = 0.0
        insertion_rate = 0.0

    return {
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,

        "reference_words": reference_words,
        "hypothesis_words": hypothesis_words,

        "wer": wer_value,
        "accuracy": accuracy,

        "substitution_rate":
            substitution_rate,

        "deletion_rate":
            deletion_rate,

        "insertion_rate":
            insertion_rate,
    }


# ============================================================
# GLOBAL CHARACTER METRICS
# ============================================================

def calculate_character_metrics(
    references,
    hypotheses,
):

    result = process_characters(
        references,
        hypotheses,
    )

    hits = result.hits
    substitutions = result.substitutions
    deletions = result.deletions
    insertions = result.insertions

    reference_chars = (
        hits
        + substitutions
        + deletions
    )

    errors = (
        substitutions
        + deletions
        + insertions
    )

    if reference_chars > 0:

        cer_value = (
            errors
            / reference_chars
        )

    else:

        cer_value = 0.0

    return {
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,

        "reference_chars":
            reference_chars,

        "cer": cer_value,
    }


# ============================================================
# DETAILED WORD ERRORS
# ============================================================

def get_word_errors(
    reference,
    hypothesis,
):
    """
    Phân tích chi tiết lỗi WER theo từng word.

    Trả về:
        - substitutions
        - deletions
        - insertions

    Compatible với nhiều version JiWER:
        ref_start / ref_start_idx
        ref_end / ref_end_idx
        hyp_start / hyp_start_idx
        hyp_end / hyp_end_idx
    """

    reference = clean_text(reference)
    hypothesis = clean_text(hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # --------------------------------------------------------
    # IMPORTANT:
    # process_words cần list sentence
    # --------------------------------------------------------

    result = process_words(
        [reference],
        [hypothesis],
    )

    substitutions = []
    deletions = []
    insertions = []

    if not result.alignments:
        return {
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
        }

    # Vì truyền [reference], [hypothesis],
    # nên lấy alignment của sentence đầu tiên
    alignment = result.alignments[0]

    for chunk in alignment:

        # ====================================================
        # Tương thích nhiều version JiWER
        # ====================================================

        if hasattr(chunk, "ref_start_idx"):
            ref_start = chunk.ref_start_idx
            ref_end = chunk.ref_end_idx
            hyp_start = chunk.hyp_start_idx
            hyp_end = chunk.hyp_end_idx

        else:
            ref_start = chunk.ref_start
            ref_end = chunk.ref_end
            hyp_start = chunk.hyp_start
            hyp_end = chunk.hyp_end

        # ----------------------------------------------------
        # Lấy words
        # ----------------------------------------------------

        ref_part = ref_words[
            ref_start:ref_end
        ]

        hyp_part = hyp_words[
            hyp_start:hyp_end
        ]

        # ----------------------------------------------------
        # Loại lỗi
        # ----------------------------------------------------

        error_type = chunk.type

        # ====================================================
        # CORRECT
        # ====================================================

        if error_type == "equal":
            continue

        # ====================================================
        # SUBSTITUTION
        # ====================================================

        elif error_type in (
            "substitute",
            "substitution",
        ):

            substitutions.append({
                "reference":
                    " ".join(ref_part),

                "hypothesis":
                    " ".join(hyp_part),

                "reference_position": [
                    ref_start,
                    ref_end,
                ],

                "hypothesis_position": [
                    hyp_start,
                    hyp_end,
                ],

                "type":
                    "substitution",
            })

        # ====================================================
        # DELETION
        # ====================================================

        elif error_type in (
            "delete",
            "deletion",
        ):

            deletions.append({
                "reference":
                    " ".join(ref_part),

                "hypothesis":
                    "",

                "reference_position": [
                    ref_start,
                    ref_end,
                ],

                "hypothesis_position": [
                    hyp_start,
                    hyp_end,
                ],

                "type":
                    "deletion",
            })

        # ====================================================
        # INSERTION
        # ====================================================

        elif error_type in (
            "insert",
            "insertion",
        ):

            insertions.append({
                "reference":
                    "",

                "hypothesis":
                    " ".join(hyp_part),

                "reference_position": [
                    ref_start,
                    ref_end,
                ],

                "hypothesis_position": [
                    hyp_start,
                    hyp_end,
                ],

                "type":
                    "insertion",
            })

        # ====================================================
        # UNKNOWN
        # ====================================================

        else:

            print(
                f"[WARNING] Unknown alignment type: "
                f"{error_type}"
            )

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }

# ============================================================
# SINGLE SEGMENT ANALYSIS
# ============================================================

def analyze_one_segment(
    segment,
    model,
):

    reference = clean_text(
        segment["reference"]
    )

    hypothesis = clean_text(
        segment[model]
    )

    result = jiwer_words(
        reference,
        hypothesis,
    )

    reference_words = (
        result.hits
        + result.substitutions
        + result.deletions
    )

    hypothesis_words = (
        result.hits
        + result.substitutions
        + result.insertions
    )

    errors = (
        result.substitutions
        + result.deletions
        + result.insertions
    )

    if reference_words > 0:

        segment_wer = (
            errors
            / reference_words
        )

    else:

        segment_wer = (
            0.0
            if hypothesis_words == 0
            else 1.0
        )

    details = get_word_errors(
        reference,
        hypothesis,
    )

    return {

        "id":
            segment["id"],

        "speaker":
            segment["speaker"],

        "start":
            segment["start"],

        "end":
            segment["end"],

        "duration":
            segment["duration"],

        "reference":
            reference,

        "hypothesis":
            hypothesis,

        "reference_words":
            reference_words,

        "hypothesis_words":
            hypothesis_words,

        "hits":
            result.hits,

        "substitutions":
            result.substitutions,

        "deletions":
            result.deletions,

        "insertions":
            result.insertions,

        "errors":
            errors,

        "wer":
            segment_wer,

        "perfect":
            reference == hypothesis,

        "substitution_details":
            details["substitutions"],

        "deletion_details":
            details["deletions"],

        "insertion_details":
            details["insertions"],

        "note":
            segment["note"],

        "changed":
            segment["changed"],
    }


# ============================================================
# PERCENTILE
# ============================================================

def percentile(
    values,
    p,
):

    if not values:
        return 0.0

    values = sorted(values)

    index = (
        (len(values) - 1)
        * p
    )

    low = int(index)

    high = min(
        low + 1,
        len(values) - 1,
    )

    weight = index - low

    return (
        values[low]
        * (1 - weight)
        +
        values[high]
        * weight
    )


# ============================================================
# SEGMENT DISTRIBUTION
# ============================================================

def calculate_segment_statistics(
    details,
):

    if not details:

        return {}

    wers = [
        x["wer"]
        for x in details
    ]

    perfect = sum(
        x["perfect"]
        for x in details
    )

    mismatch = (
        len(details)
        - perfect
    )

    return {

        "total":
            len(details),

        "perfect":
            perfect,

        "mismatch":
            mismatch,

        "perfect_rate":
            perfect / len(details),

        "mismatch_rate":
            mismatch / len(details),

        "mean_wer":
            sum(wers) / len(wers),

        "median_wer":
            percentile(
                wers,
                0.50,
            ),

        "p75_wer":
            percentile(
                wers,
                0.75,
            ),

        "p90_wer":
            percentile(
                wers,
                0.90,
            ),

        "p95_wer":
            percentile(
                wers,
                0.95,
            ),

        "p99_wer":
            percentile(
                wers,
                0.99,
            ),

        "zero_wer":
            sum(
                x == 0
                for x in wers
            ),

        "wer_0_10":
            sum(
                x <= 0.10
                for x in wers
            ),

        "wer_10_25":
            sum(
                0.10 < x <= 0.25
                for x in wers
            ),

        "wer_25_50":
            sum(
                0.25 < x <= 0.50
                for x in wers
            ),

        "wer_50_100":
            sum(
                0.50 < x <= 1.00
                for x in wers
            ),

        "wer_over_100":
            sum(
                x > 1.00
                for x in wers
            ),
    }


# ============================================================
# ERROR FREQUENCY
# ============================================================

def calculate_error_frequency(
    details,
):

    substitution_counter = Counter()
    deletion_counter = Counter()
    insertion_counter = Counter()

    for item in details:

        for error in item[
            "substitution_details"
        ]:

            key = (
                error["reference"],
                error["hypothesis"],
            )

            substitution_counter[
                key
            ] += 1

        for error in item[
            "deletion_details"
        ]:

            deletion_counter[
                error["reference"]
            ] += 1

        for error in item[
            "insertion_details"
        ]:

            insertion_counter[
                error["hypothesis"]
            ] += 1

    return {

        "substitutions": [
            {
                "reference": k[0],
                "hypothesis": k[1],
                "count": v,
            }

            for k, v
            in substitution_counter.most_common()
        ],

        "deletions": [
            {
                "reference": k,
                "count": v,
            }

            for k, v
            in deletion_counter.most_common()
        ],

        "insertions": [
            {
                "hypothesis": k,
                "count": v,
            }

            for k, v
            in insertion_counter.most_common()
        ],
    }


# ============================================================
# TOP BAD SEGMENTS
# ============================================================

def get_worst_segments(
    details,
    n=50,
):

    sorted_details = sorted(
        details,
        key=lambda x: (
            x["wer"],
            x["errors"],
        ),
        reverse=True,
    )

    return sorted_details[:n]


# ============================================================
# CROSS MODEL ANALYSIS
# ============================================================

def cross_model_analysis(
    segments,
    details_by_model,
):

    results = []

    for i, segment in enumerate(
        segments
    ):

        row = {
            "id":
                segment["id"],

            "reference":
                clean_text(
                    segment["reference"]
                ),
        }

        perfect_models = []

        for model in MODELS:

            detail = (
                details_by_model[
                    model
                ][i]
            )

            row[
                f"{model}_wer"
            ] = detail["wer"]

            row[
                f"{model}_perfect"
            ] = detail["perfect"]

            if detail["perfect"]:
                perfect_models.append(
                    model
                )

        row[
            "num_perfect_models"
        ] = len(perfect_models)

        row[
            "perfect_models"
        ] = ", ".join(
            perfect_models
        )

        row[
            "all_correct"
        ] = (
            len(perfect_models)
            == len(MODELS)
        )

        row[
            "all_wrong"
        ] = (
            len(perfect_models)
            == 0
        )

        results.append(row)

    return results


# ============================================================
# FINAL VS BASE MODEL
# ============================================================

def final_vs_model(
    details_by_model,
    base_model,
):

    base_details = (
        details_by_model[
            base_model
        ]
    )

    final_details = (
        details_by_model[
            "Final"
        ]
    )

    results = []

    improved = 0
    worsened = 0
    unchanged = 0

    base_total_errors = 0
    final_total_errors = 0

    base_reference_words = 0
    final_reference_words = 0

    for i in range(
        len(base_details)
    ):

        base = base_details[i]
        final = final_details[i]

        base_wer = base["wer"]
        final_wer = final["wer"]

        improvement = (
            base_wer
            - final_wer
        )

        if improvement > 0:
            status = "improved"
            improved += 1

        elif improvement < 0:
            status = "worsened"
            worsened += 1

        else:
            status = "unchanged"
            unchanged += 1

        # ----------------------------------------------------
        # Exact correctness transition
        # ----------------------------------------------------

        if (
            not base["perfect"]
            and final["perfect"]
        ):

            transition = (
                "wrong_to_correct"
            )

        elif (
            base["perfect"]
            and not final["perfect"]
        ):

            transition = (
                "correct_to_wrong"
            )

        elif (
            base["perfect"]
            and final["perfect"]
        ):

            transition = (
                "correct_to_correct"
            )

        else:

            transition = (
                "wrong_to_wrong"
            )

        results.append({

            "id":
                base["id"],

            "reference":
                base["reference"],

            "base_hypothesis":
                base["hypothesis"],

            "final_hypothesis":
                final["hypothesis"],

            "base_wer":
                base_wer,

            "final_wer":
                final_wer,

            "wer_improvement":
                improvement,

            "status":
                status,

            "transition":
                transition,

            "base_errors":
                base["errors"],

            "final_errors":
                final["errors"],
        })

        base_total_errors += (
            base["errors"]
        )

        final_total_errors += (
            final["errors"]
        )

        base_reference_words += (
            base["reference_words"]
        )

        final_reference_words += (
            final["reference_words"]
        )

    base_global_wer = (
        base_total_errors
        / base_reference_words
        if base_reference_words
        else 0
    )

    final_global_wer = (
        final_total_errors
        / final_reference_words
        if final_reference_words
        else 0
    )

    absolute_improvement = (
        base_global_wer
        - final_global_wer
    )

    relative_improvement = (
        absolute_improvement
        / base_global_wer
        if base_global_wer
        else 0
    )

    summary = {

        "base_model":
            base_model,

        "base_wer":
            base_global_wer,

        "final_wer":
            final_global_wer,

        "absolute_improvement":
            absolute_improvement,

        "relative_improvement":
            relative_improvement,

        "segments_improved":
            improved,

        "segments_worsened":
            worsened,

        "segments_unchanged":
            unchanged,

        "wrong_to_correct":
            sum(
                x["transition"]
                == "wrong_to_correct"
                for x in results
            ),

        "correct_to_wrong":
            sum(
                x["transition"]
                == "correct_to_wrong"
                for x in results
            ),

        "correct_to_correct":
            sum(
                x["transition"]
                == "correct_to_correct"
                for x in results
            ),

        "wrong_to_wrong":
            sum(
                x["transition"]
                == "wrong_to_wrong"
                for x in results
            ),
    }

    return {
        "summary": summary,
        "segments": results,
    }


# ============================================================
# PRINT MODEL REPORT
# ============================================================

def print_model_report(
    model,
    result,
):

    w = result["word"]
    c = result["character"]
    s = result["segments"]

    print()
    print("=" * 90)
    print(f"{model.upper()} EVALUATION")
    print("=" * 90)

    print()
    print("WORD LEVEL")
    print("-" * 90)

    print(
        f"Reference words:      "
        f"{w['reference_words']}"
    )

    print(
        f"Hypothesis words:     "
        f"{w['hypothesis_words']}"
    )

    print(
        f"Correct / Hit:        "
        f"{w['hits']}"
    )

    print(
        f"Substitution:         "
        f"{w['substitutions']}"
    )

    print(
        f"Deletion:             "
        f"{w['deletions']}"
    )

    print(
        f"Insertion:            "
        f"{w['insertions']}"
    )

    print(
        f"Total errors:         "
        f"{w['errors']}"
    )

    print()

    print(
        f"WER:                  "
        f"{w['wer'] * 100:.4f}%"
    )

    print(
        f"Accuracy:             "
        f"{w['accuracy'] * 100:.4f}%"
    )

    print(
        f"Substitution rate:    "
        f"{w['substitution_rate'] * 100:.4f}%"
    )

    print(
        f"Deletion rate:        "
        f"{w['deletion_rate'] * 100:.4f}%"
    )

    print(
        f"Insertion rate:       "
        f"{w['insertion_rate'] * 100:.4f}%"
    )

    print()
    print("CHARACTER LEVEL")
    print("-" * 90)

    print(
        f"Reference chars:      "
        f"{c['reference_chars']}"
    )

    print(
        f"Correct / Hit:        "
        f"{c['hits']}"
    )

    print(
        f"Substitution:         "
        f"{c['substitutions']}"
    )

    print(
        f"Deletion:             "
        f"{c['deletions']}"
    )

    print(
        f"Insertion:            "
        f"{c['insertions']}"
    )

    print(
        f"Total errors:         "
        f"{c['errors']}"
    )

    print(
        f"CER:                  "
        f"{c['cer'] * 100:.4f}%"
    )

    print()
    print("SEGMENT LEVEL")
    print("-" * 90)

    print(
        f"Total segments:       "
        f"{s['total']}"
    )

    print(
        f"Perfect segments:     "
        f"{s['perfect']}"
    )

    print(
        f"Mismatch segments:    "
        f"{s['mismatch']}"
    )

    print(
        f"Perfect rate:         "
        f"{s['perfect_rate'] * 100:.4f}%"
    )

    print(
        f"Mismatch rate:        "
        f"{s['mismatch_rate'] * 100:.4f}%"
    )

    print()

    print(
        f"Mean WER:             "
        f"{s['mean_wer'] * 100:.4f}%"
    )

    print(
        f"Median WER:           "
        f"{s['median_wer'] * 100:.4f}%"
    )

    print(
        f"P75 WER:              "
        f"{s['p75_wer'] * 100:.4f}%"
    )

    print(
        f"P90 WER:              "
        f"{s['p90_wer'] * 100:.4f}%"
    )

    print(
        f"P95 WER:              "
        f"{s['p95_wer'] * 100:.4f}%"
    )

    print(
        f"P99 WER:              "
        f"{s['p99_wer'] * 100:.4f}%"
    )

    print()

    print(
        f"WER = 0%:             "
        f"{s['zero_wer']}"
    )

    print(
        f"WER <= 10%:           "
        f"{s['wer_0_10']}"
    )

    print(
        f"10% < WER <= 25%:     "
        f"{s['wer_10_25']}"
    )

    print(
        f"25% < WER <= 50%:     "
        f"{s['wer_25_50']}"
    )

    print(
        f"50% < WER <= 100%:    "
        f"{s['wer_50_100']}"
    )

    print(
        f"WER > 100%:           "
        f"{s['wer_over_100']}"
    )


# ============================================================
# PRINT GLOBAL SUMMARY
# ============================================================

def print_global_summary(
    results,
):

    print()
    print()
    print("=" * 120)
    print("GLOBAL MODEL COMPARISON")
    print("=" * 120)

    print(
        f"{'Model':<15}"
        f"{'WER':>10}"
        f"{'CER':>10}"
        f"{'Acc':>10}"
        f"{'Sub':>10}"
        f"{'Del':>10}"
        f"{'Ins':>10}"
        f"{'Perfect':>12}"
        f"{'Mismatch':>12}"
    )

    print("-" * 120)

    for model in MODELS:

        result = results[model]

        w = result["word"]
        c = result["character"]
        s = result["segments"]

        print(
            f"{model:<15}"
            f"{w['wer'] * 100:>9.2f}%"
            f"{c['cer'] * 100:>9.2f}%"
            f"{w['accuracy'] * 100:>9.2f}%"
            f"{w['substitution_rate'] * 100:>9.2f}%"
            f"{w['deletion_rate'] * 100:>9.2f}%"
            f"{w['insertion_rate'] * 100:>9.2f}%"
            f"{s['perfect_rate'] * 100:>11.2f}%"
            f"{s['mismatch_rate'] * 100:>11.2f}%"
        )

    print("=" * 120)


# ============================================================
# PRINT FINAL IMPROVEMENT
# ============================================================

def print_final_improvement(
    final_comparisons,
):

    print()
    print("=" * 100)
    print("FINAL PIPELINE IMPROVEMENT")
    print("=" * 100)

    for model in BASE_MODELS:

        summary = (
            final_comparisons[
                model
            ]["summary"]
        )

        print()
        print(
            f"{model} -> FINAL"
        )

        print(
            f"  Base WER:              "
            f"{summary['base_wer'] * 100:.4f}%"
        )

        print(
            f"  Final WER:             "
            f"{summary['final_wer'] * 100:.4f}%"
        )

        print(
            f"  Absolute improvement:  "
            f"{summary['absolute_improvement'] * 100:.4f} pp"
        )

        print(
            f"  Relative improvement:  "
            f"{summary['relative_improvement'] * 100:.2f}%"
        )

        print()

        print(
            f"  Segments improved:     "
            f"{summary['segments_improved']}"
        )

        print(
            f"  Segments worsened:     "
            f"{summary['segments_worsened']}"
        )

        print(
            f"  Segments unchanged:    "
            f"{summary['segments_unchanged']}"
        )

        print()

        print(
            f"  Wrong -> Correct:      "
            f"{summary['wrong_to_correct']}"
        )

        print(
            f"  Correct -> Wrong:      "
            f"{summary['correct_to_wrong']}"
        )

        print(
            f"  Correct -> Correct:    "
            f"{summary['correct_to_correct']}"
        )

        print(
            f"  Wrong -> Wrong:        "
            f"{summary['wrong_to_wrong']}"
        )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data,
    path,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    rows,
    path,
):

    if not rows:
        return

    # Union of all keys
    fieldnames = []

    for row in rows:

        for key in row.keys():

            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            output = {}

            for key in fieldnames:

                value = row.get(
                    key,
                    "",
                )

                if isinstance(
                    value,
                    (list, dict),
                ):

                    value = json.dumps(
                        value,
                        ensure_ascii=False,
                    )

                output[key] = value

            writer.writerow(output)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("DETAILED ASR EVALUATION")
    print("=" * 90)

    print(
        f"Input:    {HTML_FILE}"
    )

    print(
        f"Output:   {OUTPUT_DIR}"
    )

    # ========================================================
    # Parse
    # ========================================================

    segments = parse_html(
        HTML_FILE
    )

    print(
        f"Segments: {len(segments)}"
    )

    if not segments:

        print(
            "ERROR: Không tìm thấy segment."
        )

        return

    # ========================================================
    # Analyze
    # ========================================================

    results = {}

    details_by_model = {}

    error_frequency = {}

    worst_segments = {}

    for model in MODELS:

        print()
        print(
            f"Analyzing {model}..."
        )

        references = [
            clean_text(
                segment["reference"]
            )
            for segment in segments
        ]

        hypotheses = [
            clean_text(
                segment[model]
            )
            for segment in segments
        ]

        word = calculate_word_metrics(
            references,
            hypotheses,
        )

        character = (
            calculate_character_metrics(
                references,
                hypotheses,
            )
        )

        details = []

        for segment in segments:

            details.append(
                analyze_one_segment(
                    segment,
                    model,
                )
            )

        segment_stats = (
            calculate_segment_statistics(
                details
            )
        )

        results[model] = {

            "word": word,

            "character": character,

            "segments":
                segment_stats,
        }

        details_by_model[
            model
        ] = details

        error_frequency[
            model
        ] = calculate_error_frequency(
            details
        )

        worst_segments[
            model
        ] = get_worst_segments(
            details,
            n=100,
        )

    # ========================================================
    # Cross Model
    # ========================================================

    cross_model = cross_model_analysis(
        segments,
        details_by_model,
    )

    # ========================================================
    # Final comparison
    # ========================================================

    final_comparisons = {}

    for model in BASE_MODELS:

        final_comparisons[
            model
        ] = final_vs_model(
            details_by_model,
            model,
        )

    # ========================================================
    # Print
    # ========================================================

    for model in MODELS:

        print_model_report(
            model,
            results[model],
        )

    print_global_summary(
        results
    )

    print_final_improvement(
        final_comparisons
    )

    # ========================================================
    # Save complete JSON
    # ========================================================

    full_result = {

        "input":
            HTML_FILE,

        "reference":
            "edit",

        "segments":
            len(segments),

        "models":
            results,

        "error_frequency":
            error_frequency,

        "worst_segments":
            worst_segments,

        "cross_model":
            cross_model,

        "final_comparison":
            final_comparisons,
    }

    json_path = (
        OUTPUT_DIR
        / "full_evaluation.json"
    )

    save_json(
        full_result,
        json_path,
    )

    # ========================================================
    # Save model CSV
    # ========================================================

    for model in MODELS:

        path = (
            OUTPUT_DIR
            / f"{model.lower()}_segments.csv"
        )

        save_csv(
            details_by_model[
                model
            ],
            path,
        )

    # ========================================================
    # Save cross model
    # ========================================================

    save_csv(
        cross_model,
        OUTPUT_DIR
        / "cross_model.csv",
    )

    # ========================================================
    # Save final comparison
    # ========================================================

    for model in BASE_MODELS:

        save_csv(
            final_comparisons[
                model
            ]["segments"],
            OUTPUT_DIR
            / (
                f"{model.lower()}"
                "_vs_final.csv"
            ),
        )

    # ========================================================
    # Save top errors
    # ========================================================

    for model in MODELS:

        frequency = (
            error_frequency[
                model
            ]
        )

        save_csv(
            frequency[
                "substitutions"
            ],
            OUTPUT_DIR
            / (
                f"{model.lower()}"
                "_top_substitutions.csv"
            ),
        )

        save_csv(
            frequency[
                "deletions"
            ],
            OUTPUT_DIR
            / (
                f"{model.lower()}"
                "_top_deletions.csv"
            ),
        )

        save_csv(
            frequency[
                "insertions"
            ],
            OUTPUT_DIR
            / (
                f"{model.lower()}"
                "_top_insertions.csv"
            ),
        )

        save_csv(
            worst_segments[
                model
            ],
            OUTPUT_DIR
            / (
                f"{model.lower()}"
                "_worst_segments.csv"
            ),
        )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print()
    print("=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    print(
        f"Directory:"
        f" {OUTPUT_DIR}"
    )

    print()

    print(
        "Main:"
    )

    print(
        "  full_evaluation.json"
    )

    print(
        "  cross_model.csv"
    )

    print()

    print(
        "Per model:"
    )

    for model in MODELS:

        print(
            f"  {model.lower()}_segments.csv"
        )

    print()

    print(
        "Final comparison:"
    )

    for model in BASE_MODELS:

        print(
            f"  {model.lower()}_vs_final.csv"
        )

    print()

    print(
        "Error analysis:"
    )

    for model in MODELS:

        print(
            f"  {model.lower()}_top_substitutions.csv"
        )

        print(
            f"  {model.lower()}_top_deletions.csv"
        )

        print(
            f"  {model.lower()}_top_insertions.csv"
        )

        print(
            f"  {model.lower()}_worst_segments.csv"
        )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()