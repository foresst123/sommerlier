"""Ask the resident LLM many independent questions, a batch at a time.

Shared by the passes that reuse the refinement model (speaker relabel,
conversation-export judging). A batch that does not fit -- the model's `max_batch_tokens` guard
or an out-of-memory -- is halved and retried, as refinement does; a question
that still fails on its own is recorded and skipped, never raised. The callers
treat an unanswered question as "no change", which is the safe default for
passes that only ever narrow what the pipeline already produced.
"""

from typing import List, Optional, Sequence, Tuple

# With thinking on, the reply starts with the model's reasoning and only then the
# answer, all inside `max_new_tokens`. Under this a thinking model is cut off
# before it answers, so a smaller configured limit is raised to it.
THINKING_MIN_NEW_TOKENS = 2048


def reply_budget(max_new_tokens: int, thinking: bool) -> int:
    """Tokens one reply may use: the configured limit, never below the floor when thinking."""
    budget = max(1, int(max_new_tokens))
    return max(budget, THINKING_MIN_NEW_TOKENS) if thinking else budget


def ask_in_batches(llm, system_prompt: str, messages: Sequence[str], *,
                   per_call: int, max_new_tokens: int, label: str,
                   logger=None, thinking: bool = False) -> Tuple[List[Optional[str]], int]:
    """(replies, unanswered): one reply per message, None where none came.

    `per_call` is how many messages go in one `generate_texts` call; the caller
    sizes it from the model's limits and the messages' length. `thinking` is only
    passed on when it is on, so a model surface that knows nothing of it is
    called exactly as before.
    """
    extra = {"thinking": True} if thinking else {}
    replies: List[Optional[str]] = [None] * len(messages)
    unanswered = 0
    per_call = max(1, int(per_call))
    start = 0
    while start < len(messages):
        size = min(per_call, len(messages) - start)
        while True:
            chunk = list(messages[start:start + size])
            ok, outputs = llm.generate_texts(
                system_prompt, chunk, max_new_tokens, use_prefix=False,
                labels=[f"{label} {start + k}" for k in range(len(chunk))], **extra)
            if ok and len(outputs) == len(chunk):
                replies[start:start + len(chunk)] = outputs
                break
            if size > 1:
                size = max(1, size // 2)
                if logger:
                    logger.info(f"[{label}] retrying with {size} per call")
                continue
            unanswered += 1
            if logger:
                logger.warning(f"[{label}] item {start} could not be answered")
            break
        start += len(chunk)
    return replies, unanswered
