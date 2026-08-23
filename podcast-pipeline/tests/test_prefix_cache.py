"""Prefix caching for the refinement LLM.

The system prompt is ~1200 tokens and identical on every request, so its keys
and values are computed once and reused. The correctness bar is that output is
byte-identical to the full-prompt path -- a mis-shaped cache produces wrong
Vietnamese rather than an exception, so it must decline instead of guessing.

Run:  python -m pytest tests/test_prefix_cache.py -q     (from podcast-pipeline/)
"""
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "services/diarization_refinement_service.py")


def _method(name):
    """Load one method in isolation: the module pulls in librosa/soundfile."""
    with open(SRC, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 "<t>", "exec"), ns)
    return ns[name]


class _ChatMLTok:
    """The layout Qwen uses: one <|im_start|> block per turn."""
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


class _FoldedTok:
    """A template that folds the system turn into the user turn."""
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        return "ONE_BLOCK" if len(msgs) == 1 else "A COMPLETELY DIFFERENT LAYOUT"


def _svc(tok):
    obj = type("S", (), {})()
    obj.tokenizer = tok
    obj._split_prompt = _method("_split_prompt").__get__(obj)
    return obj


SYS = "Bạn là trợ lý hợp nhất transcript. " * 40


def test_prefix_is_shared_and_tails_differ():
    s = _svc(_ChatMLTok())
    a = s._split_prompt(SYS, "Bản 1: xin chào\n")
    b = s._split_prompt(SYS, "Bản 1: cảm ơn\n")

    assert a is not None and b is not None
    assert a[0] == b[0], "the cached part must be identical across requests"
    assert a[1] != b[1], "the per-request tail must not be"


def test_prefix_plus_tail_reconstructs_the_prompt_exactly():
    """This is what makes reusing the cache safe: the model sees the same
    token sequence, only the prefix's attention is not recomputed."""
    tok = _ChatMLTok()
    s = _svc(tok)
    user = "Bản 1: xin chào\nBản 2: chào bạn\n"
    prefix, tail = s._split_prompt(SYS, user)
    full = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        add_generation_prompt=True)
    assert prefix + tail == full


def test_a_template_without_a_standalone_system_turn_is_declined():
    assert _svc(_FoldedTok())._split_prompt(SYS, "x") is None


def test_an_empty_prefix_is_declined():
    """An empty shared part later reshapes into an error deep inside the model;
    catching it here keeps the failure legible."""
    class _Empty:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return "" if len(msgs) == 1 else "user turn only"
    assert _svc(_Empty())._split_prompt(SYS, "x") is None


# --- configuration ----------------------------------------------------------

def _config():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def test_kaggle_keeps_prefix_cache_off():
    """A T4 running batch 2 has neither the headroom nor much to gain."""
    assert _config()["environments"]["kaggle"]["models"]["refinement"]["prefix_cache"] is False


def test_a100_profile_exists_and_turns_it_on():
    a100 = _config()["environments"]["a100"]
    ref = a100["models"]["refinement"]
    assert ref["prefix_cache"] is True
    assert ref["batch_size"] >= 16, "the saving scales with batch size"
    assert a100["allow_tf32"] is True


def test_refinement_config_keys_match_the_constructor():
    with open(SRC, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    params = [a.arg for a in init.args.args if a.arg != "self"]
    for env, profile in _config()["environments"].items():
        for key in profile.get("models", {}).get("refinement", {}):
            assert key in params, f"{env}: models.refinement.{key} is not a constructor arg"
