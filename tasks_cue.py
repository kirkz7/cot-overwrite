"""Exp 18: explicit order cues vs presentation position.

Exp 1 cannot tell a *rational* recency default ("text written later supersedes") from a
*positional* one ("take whatever is presented last / closest to the question"), because in every
condition the latest value is also the last one presented. Here the trace carries explicit order
information that disagrees with presentation order:
  full        ordered, no cue                                   (latest = last presented)
  shuf        shuffled, no cue                                  (no way to know; baseline)
  shuf_step   shuffled, every line tagged "Step i:"             (latest = highest step)
  rev         newest-first, no cue                              (latest = FIRST presented)
  rev_header  newest-first, header says so                      (latest = first presented)
  rev_step    newest-first, every line tagged "Step i:"         (latest = first presented)
  shuf_neutral  as shuf but the answer stem avoids the word "final"
Natural-language style uses clock times ("[10:07] Box E has 37 marbles.") instead of step numbers.
"""
from tasks import _box, make_head, presented_lines

HEADER = {"sym": "Let me trace the program step by step.", "nl": "Let me track the boxes step by step."}
REV_HEADER = {"sym": "Let me trace the program. The steps below are listed from the most recent to the earliest.",
              "nl": "Let me track the boxes. The log below is listed from the most recent entry to the earliest."}
CONDS = ["full", "shuf", "shuf_step", "rev", "rev_header", "rev_step", "shuf_neutral"]


def ordered_lines(ex, cond):
    """(step_number, line) pairs in presented order; step numbers are 1-based program order."""
    idx = {id(l): i + 1 for i, l in enumerate(ex.lines)}
    if cond == "full":
        lines = ex.lines
    elif cond.startswith("shuf"):
        lines = presented_lines(ex, "shuf")
    else:
        lines = ex.lines[::-1]
    return [(idx[id(l)], l) for l in lines]


def tag(step, style):
    return f"Step {step}: " if style == "sym" else f"[10:{step:02d}] "


def build_cue_prompt(ex, query_var, cond, tok, style="sym", plain=False):
    sfx = "_nl" if style == "nl" else ""
    head = make_head("\n".join(getattr(l, "prog" + sfx) for l in ex.lines), query_var, tok, plain, style)
    pairs = ordered_lines(ex, cond)
    tagged = cond in ("shuf_step", "rev_step")
    body = [(tag(s, style) if tagged else "") + getattr(l, "bare" + sfx) for s, l in pairs]
    header = REV_HEADER[style] if cond == "rev_header" else HEADER[style]
    text = head + header + "\n" + "\n".join(body) + "\n"
    if cond == "shuf_neutral":
        stem = f"So {query_var} = " if style == "sym" else f"So the number of marbles in {_box(query_var)} is "
    else:
        stem = (f"Therefore, the final value of {query_var} is " if style == "sym"
                else f"Therefore, the number of marbles in {_box(query_var)} is ")
    return text + stem, [l.value for _, l in pairs if l.var == query_var]
