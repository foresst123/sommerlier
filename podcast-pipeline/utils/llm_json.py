"""Reading JSON out of a chat model's reply, which is rarely just JSON.

Replies arrive wrapped in a code fence, preceded by reasoning, or cut off by the
token limit halfway through the last object. The passes that read them (speaker
relabel, conversation-export judging) all want the same thing: whatever complete
objects the reply holds, and an empty result -- never an exception -- when it
holds none.
"""

import json
import re
from typing import Iterator, List, Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def clean_reply(raw: Optional[str]) -> str:
    """The reply without reasoning or a code fence around the answer."""
    text = _THINK_RE.sub("", raw or "")
    # A reply cut off inside its reasoning holds drafts of an answer, not one.
    text = _OPEN_THINK_RE.sub("", text)
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    return text.strip()


def iter_objects(text: str) -> Iterator[str]:
    """Yield each balanced {...} in `text`, tolerating braces inside strings.

    For a reply that is not valid JSON as a whole -- usually cut off by the token
    limit -- so the objects that did finish still count.
    """
    depth, start, in_string, escaped = 0, None, False, False
    for pos, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = pos
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:pos + 1]
                start = None


def loads_loose(text: str):
    """The JSON value in `text` (or in its outermost [...]), else None."""
    for candidate in (text, text[text.find("["):text.rfind("]") + 1] if "[" in text else ""):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def is_readable(raw: Optional[str]) -> bool:
    """Whether a reply held any JSON at all.

    `objects_in` returns an empty list both for a model that answered `[]` and
    for one that answered in prose, and a pass that treats "nothing to change"
    as the safe default cannot tell them apart. An empty list is an answer;
    prose is not.
    """
    text = clean_reply(raw)
    if not text:
        return False
    if isinstance(loads_loose(text), (list, dict)):
        return True
    for chunk in iter_objects(text):
        try:
            json.loads(chunk)
            return True
        except ValueError:
            continue
    return False


def objects_in(raw: Optional[str], wrapper_keys=()) -> List[dict]:
    """Every dict a reply holds, whole-reply parse first, salvage second.

    A top-level list gives its items; a top-level object gives the list under
    one of `wrapper_keys` if it has one, otherwise itself. Non-dict items are
    dropped.
    """
    text = clean_reply(raw)
    if not text:
        return []
    parsed = loads_loose(text)
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        wrapped = next((parsed[k] for k in wrapper_keys
                        if isinstance(parsed.get(k), list)), None)
        items = wrapped if wrapped is not None else [parsed]
    else:
        items = []
        for chunk in iter_objects(text):
            try:
                items.append(json.loads(chunk))
            except ValueError:
                continue
    return [item for item in items if isinstance(item, dict)]
