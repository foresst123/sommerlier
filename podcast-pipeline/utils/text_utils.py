import re

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = {
        "gpt-4.1": {
            "input": 2.00 / 1_000_000,
            "cached_input": 0.50 / 1_000_000,
            "output": 8.00 / 1_000_000,
        },
        "gpt-4.1-mini": {
            "input": 0.40 / 1_000_000,
            "cached_input": 0.10 / 1_000_000,
            "output": 1.60 / 1_000_000,
        },
        "gpt-4.1-nano": {
            "input": 0.10 / 1_000_000,
            "cached_input": 0.025 / 1_000_000,
            "output": 0.40 / 1_000_000,
        },
        "openai-o3": {
            "input": 2.00 / 1_000_000,
            "cached_input": 0.50 / 1_000_000,
            "output": 8.00 / 1_000_000,
        },
        "openai-o4-mini": {
            "input": 1.10 / 1_000_000,
            "cached_input": 0.275 / 1_000_000,
            "output": 4.40 / 1_000_000,
        },
    }

    if model_name not in pricing:
        raise ValueError(f"Model '{model_name}' not found in pricing table.")

    rates = pricing[model_name]
    input_cost = input_tokens * rates["input"]
    output_cost = output_tokens * rates["output"]
    total_cost = input_cost + output_cost

    return total_cost

ENG_PATTERN = re.compile(r"[A-Za-z]+")

def ko_transliterate_english(text: str, g2p_func=None) -> str:
    """
    Find English segments in the input string and convert them to Korean pronunciation.
    Requires passing a g2p_func (e.g. g2pk.G2p)
    """
    if g2p_func is None:
        return text
    def _repl(m: re.Match) -> str:
        segment = m.group(0)
        return g2p_func(segment)
    return ENG_PATTERN.sub(_repl, text)

def ko_process_json(input_list: list, g2p_func=None) -> None:
    for entry in input_list:
        text = entry.get("text", "")
        if re.search(r"[A-Za-z]", text):
            entry["text"] = ko_transliterate_english(text, g2p_func)
