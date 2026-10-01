"""Integer matching in free text.

A number must not be glued to other digits or be part of a decimal, but may be followed by
sentence punctuation: "is 686." matches 686, "686.5" and "3.686" do not. (An earlier version
used (?![\\d.]) and silently skipped every number that ended a sentence.) Thousands-separated
forms ("142,857") are handled too.
"""
import re

INT = re.compile(r"(?<![\d.])-?\d+(?!\d|\.\d)")


def _forms(v):
    forms = [str(v)]
    if abs(v) >= 1000:
        forms.append(f"{v:,}")
    return forms


def num_re(v):
    alts = "|".join(re.escape(f) for f in sorted(_forms(v), key=len, reverse=True))
    return re.compile(rf"(?<![\d.,])(?:{alts})(?!\d|[.,]\d)")


def contains(text, v):
    return num_re(v).search(text) is not None


def swap(text, old, new):
    """Replace every occurrence of integer `old` by `new`."""
    return num_re(old).sub(str(new), text)
