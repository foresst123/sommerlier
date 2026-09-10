# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT

"""LLM output parsing, cost accounting and Korean transliteration helpers.

Not currently wired into the pipeline: the LLM refinement stage parses its own
plain-text output. Kept as the single home for these helpers so the parsers are
here when a structured-output refinement path needs them.
"""

import re
import json
import ast
from typing import List

# English pattern for Korean transliteration
ENG_PATTERN = re.compile(r"[A-Za-z]+")

# USD per million tokens, covering every model either caller priced.
COST_PER_MILLION_INPUT = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-4-turbo": 10.00,
    "gpt-3.5-turbo": 0.50,
    "gpt-4.1": 2.00,
    "gpt-4.1-mini": 0.40,
    "gpt-4.1-nano": 0.10,
    "openai-o3": 2.00,
    "openai-o4-mini": 1.10,
}

COST_PER_MILLION_OUTPUT = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.60,
    "gpt-4-turbo": 30.00,
    "gpt-3.5-turbo": 1.50,
    "gpt-4.1": 8.00,
    "gpt-4.1-mini": 1.60,
    "gpt-4.1-nano": 0.40,
    "openai-o3": 8.00,
    "openai-o4-mini": 4.40,
}

COST_PER_MILLION_CACHED_INPUT = {
    "gpt-4.1": 0.50,
    "gpt-4.1-mini": 0.10,
    "gpt-4.1-nano": 0.025,
    "openai-o3": 0.50,
    "openai-o4-mini": 0.275,
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int,
                   cached_input_tokens: int = 0, strict: bool = False) -> float:
    """
    Calculate the cost of API usage based on model name and token counts.

    Args:
        model_name: Name of the LLM model
        input_tokens: Number of input tokens (billed at the full input rate)
        output_tokens: Number of output tokens
        cached_input_tokens: Subset of input tokens served from cache, billed at
            the cached rate where the model publishes one
        strict: Raise on an unpriced model instead of returning 0.0

    Returns:
        Total cost in USD
    """
    if model_name not in COST_PER_MILLION_INPUT:
        if strict:
            raise ValueError(f"Model '{model_name}' not found in pricing table.")
        return 0.0

    input_cost = (input_tokens / 1_000_000) * COST_PER_MILLION_INPUT[model_name]
    output_cost = (output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT.get(model_name, 0.0)

    cached_rate = COST_PER_MILLION_CACHED_INPUT.get(
        model_name, COST_PER_MILLION_INPUT[model_name]
    )
    cached_cost = (cached_input_tokens / 1_000_000) * cached_rate

    return input_cost + output_cost + cached_cost


def speaker_tagged_text(data):
    """
    Generate speaker-tagged text from segment data.

    Args:
        data: List of segments with 'speaker' and 'text' fields

    Returns:
        String with speaker tags and text
    """
    result = []
    for item in data:
        speaker = item.get("speaker", "Unknown")
        text = item.get("text", "")
        result.append(f"[{speaker}]: {text}")
    return "\n".join(result)


def parse_speaker_summary(llm_output: str) -> list | None:
    """
    Extract and parse a JSON array from the LLM output string.
    Handles 'json' prefix, code blocks (```), leading/trailing whitespace, etc.
    """
    if not llm_output:
        return None

    try:
        # Remove code blocks such as ```json ... ``` or ``` ... ```
        # Use a regular expression to find content between square brackets '[' and ']'
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Convert the JSON string to a Python object (list of dicts)
            return json.loads(json_str)
        else:
            print("Parsing Error: Could not find a valid JSON array format ([]).")
            return None

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"Unknown parsing error: {e}")
        return None


def process_llm_diarization_output(llm_output: str) -> list[dict]:
    """
    Process LLM output for diarization results.

    Args:
        llm_output: Raw LLM output string

    Returns:
        List of dictionaries containing diarization data
    """
    # 1. Find ```json ... ``` code block in LLM output
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_output)
    if not json_match:
        # If no ```json block is found, attempt to parse the entire string
        json_string = llm_output
    else:
        json_string = json_match.group(1)

    # 2. Parse the JSON string into a Python object
    try:
        llm_data = json.loads(json_string)
    except json.JSONDecodeError:
        # In case the LLM output is in Python list format ('[{"text":...}]')
        try:
            # ast.literal_eval is a more secure version of eval.
            llm_data = ast.literal_eval(json_string)
        except (ValueError, SyntaxError) as e:
            print(f"Error: Failed to parse as both JSON and Python literal. {e!r}")
            return []

    return llm_data


def _default_g2p():
    """Build a g2pk converter on demand.

    g2pk is a Korean-only dependency, so importing it at module scope would make
    this module unimportable for the Vietnamese pipeline that never calls it.
    """
    from g2pk import G2p
    return G2p()


def ko_transliterate_english(text: str, g2p_func=None) -> str:
    """
    Find English segments in the input string and convert them to Korean pronunciation.

    Args:
        text: Input text with English words
        g2p_func: Converter to use; defaults to g2pk.G2p

    Returns:
        Text with English transliterated to Korean pronunciation
    """
    if g2p_func is None:
        g2p_func = _default_g2p()

    def _repl(m: re.Match) -> str:
        return g2p_func(m.group(0))
    return ENG_PATTERN.sub(_repl, text)


def ko_process_json(input_list: List[dict], g2p_func=None) -> None:
    """
    Process JSON list to transliterate English to Korean.

    Args:
        input_list: List of dictionaries with 'text' field
        g2p_func: Converter to use; defaults to g2pk.G2p
    """
    if g2p_func is None:
        g2p_func = _default_g2p()

    for entry in input_list:
        text = entry.get("text", "")
        # Convert if text contains English
        if re.search(r"[A-Za-z]", text):
            entry["text"] = ko_transliterate_english(text, g2p_func)
