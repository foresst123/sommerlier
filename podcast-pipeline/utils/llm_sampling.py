"""How the resident LLM samples its tokens.

Greedy decoding suits the passes that must be repeatable and answer in a fixed
shape (refinement fusion, and any call made without thinking). A model asked to
think at temperature 0 can fall into an endless loop of the same reasoning until
the token limit, which no larger limit fixes. For the thinking calls the Qwen3.5
model card recommends sampling with a presence penalty, so those use it.

Plain data and no imports: the refinement worker runs in another environment.
"""

# Qwen/Qwen3.5-9B model card, thinking mode; the presence penalty is the one it
# says to raise (0-2) against endless repetition. The seed keeps a rerun of the
# same window from giving a different answer.
THINKING_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.0,
    "seed": 0,
}

# vLLM ends a reply early when the same run of tokens repeats back to back (see
# vllm.sampling_params.RepetitionDetectionParams; the worker builds the object).
# A thinking reply that loops was seen alternating two 60-100 token blocks
# dozens of times, so a pattern up to 256 tokens repeated 3 times catches it
# after ~600 tokens instead of at the token limit. The 16-token floor keeps
# short legitimate repeats (lists of labels, JSON keys) from counting.
REPETITION_DETECTION = {"max_pattern_size": 256, "min_pattern_size": 16, "min_count": 3}


def vllm_sampling_kwargs(thinking: bool, max_tokens: int) -> dict:
    """Arguments for vllm.SamplingParams (repetition_detection is plain data here)."""
    if thinking:
        return {**THINKING_SAMPLING, "max_tokens": max_tokens, "repetition_penalty": 1.0,
                "repetition_detection": dict(REPETITION_DETECTION)}
    return {"temperature": 0.0, "max_tokens": max_tokens, "repetition_penalty": 1.0}


def hf_generate_kwargs(thinking: bool) -> dict:
    """Sampling arguments for transformers' generate() (it has no presence penalty)."""
    if thinking:
        return {"do_sample": True, "temperature": THINKING_SAMPLING["temperature"],
                "top_p": THINKING_SAMPLING["top_p"], "top_k": THINKING_SAMPLING["top_k"],
                "repetition_penalty": 1.0}
    return {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
            "repetition_penalty": 1.0}
