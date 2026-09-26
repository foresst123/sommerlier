"""Reading a chat model's reply, and asking it in batches.

Run:  python -m pytest tests/test_llm_replies.py -q     (from podcast-pipeline/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_batches import THINKING_MIN_NEW_TOKENS, ask_in_batches, reply_budget
from utils.llm_json import clean_reply, is_cut_in_thought, is_readable, objects_in


# --- reasoning around the answer ---------------------------------------------------

def test_a_reply_that_stops_inside_its_reasoning_is_cut_in_thought():
    assert is_cut_in_thought("<think>đang nghĩ dở")
    assert is_cut_in_thought("<think>xong</think>[1] <think>lại nghĩ tiếp")


def test_a_reply_with_its_reasoning_closed_or_none_is_not():
    assert not is_cut_in_thought("<think>xong</think>[]")
    assert not is_cut_in_thought("[]")
    assert not is_cut_in_thought("")
    assert not is_cut_in_thought(None)


def test_the_answer_after_the_reasoning_is_what_is_kept():
    assert clean_reply('<think>{"a": 1}</think>\n{"b": 2}') == '{"b": 2}'


def test_a_reply_cut_in_thought_is_not_readable_and_prose_and_empty_list_are_told_apart():
    assert not is_readable("<think>{\"a\": 1} rồi thì")
    assert not is_readable("Không có gì sai.")
    assert is_readable("[]") and is_readable('<think>x</think>{"a": 1}')


# --- the budget ------------------------------------------------------------------------

def test_without_thinking_the_budget_is_what_was_configured():
    assert reply_budget(256, False) == 256 and reply_budget(0, False) == 1


def test_with_thinking_the_budget_never_falls_below_the_floor():
    assert reply_budget(256, True) == THINKING_MIN_NEW_TOKENS
    assert reply_budget(4096, True) == 4096


# --- asking -----------------------------------------------------------------------------

class _Plain:
    """A model surface that knows nothing about thinking."""

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None):
        return True, ["ok" for _ in user_messages]


class _Aware(_Plain):
    def __init__(self):
        self.seen = []

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None, thinking=False):
        self.seen.append(thinking)
        return True, ["ok" for _ in user_messages]


def test_thinking_off_calls_the_model_exactly_as_before():
    replies, unanswered = ask_in_batches(_Plain(), "s", ["a", "b"], per_call=2,
                                         max_new_tokens=10, label="x")
    assert replies == ["ok", "ok"] and unanswered == 0


def test_thinking_on_is_passed_on_to_every_call():
    llm = _Aware()
    ask_in_batches(llm, "s", ["a", "b", "c"], per_call=2, max_new_tokens=10,
                   label="x", thinking=True)
    assert llm.seen == [True, True]


def test_a_reply_whose_reasoning_was_opened_by_the_prompt_is_cut_at_the_closing_tag():
    """The chat template can open <think> itself, leaving only "reasoning</think>answer"."""
    raw = 'Có thể là {"i": 3, "speaker": "B"} hoặc [5].</think>\n[{"i": 6, "speaker": "A"}]'
    assert clean_reply(raw) == '[{"i": 6, "speaker": "A"}]'
    assert objects_in(raw) == [{"i": 6, "speaker": "A"}]


def test_a_matched_pair_and_a_closing_only_tag_are_both_handled():
    assert clean_reply("<think>a</think>b</think>[1]") == "[1]"
    assert clean_reply("[2]") == "[2]"
