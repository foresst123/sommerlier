"""A worker start-up error names its root cause, not only the wrapper."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker_errors import describe_exception


def test_the_whole_cause_chain_is_reported():
    try:
        try:
            raise OSError("libcudnn.so.9: cannot open shared object file")
        except OSError as inner:
            raise ImportError("vLLM is not available") from inner
    except ImportError as exc:
        text = describe_exception(exc)
    assert text.startswith("ImportError: vLLM is not available")
    assert "OSError: libcudnn.so.9" in text


def test_an_exception_without_a_cause_is_just_itself():
    assert describe_exception(ValueError("bad")) == "ValueError: bad"


def test_a_cycle_in_the_chain_does_not_loop_forever():
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert describe_exception(a).count("ValueError") == 2
