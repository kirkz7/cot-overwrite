"""Synthetic variable-tracking tasks with controlled number of overwrites.

A program of N_LINES assignment lines. The target variable is assigned k+1 times
(k overwrites); every other line assigns a distinct distractor variable exactly once.
All values in a trace are distinct two-digit ints, so a wrong answer can be
attributed to "retracted old value of the queried var" vs "some other trace value".
"""
import random
from dataclasses import dataclass, field

N_LINES = 16
VAR_POOL = list("abcdefghijkmnpqrstuvwyz")  # no l/o/x to avoid confusion
LO, HI = 10, 99


@dataclass
class Line:
    var: str
    prog: str      # how it appears in the question's program
    bare: str      # CoT line, value only
    chained: str   # CoT line, with the computation
    value: int
    # natural-language renderings (boxes of marbles); same numbers as the symbolic ones
    prog_nl: str = ""
    bare_nl: str = ""
    chained_nl: str = ""


def _box(v):
    return f"box {v.upper()}"


def _cap(s):
    return s[:1].upper() + s[1:]


def _pm(op):
    return "plus" if op == "+" else "minus"


@dataclass
class Example:
    k: int
    target: str
    control: str               # a once-assigned var in the same trace
    lines: list = field(default_factory=list)
    seed: int = 0

    def history(self, var):
        return [l.value for l in self.lines if l.var == var]


def _rand_op(rng, cur, used):
    for _ in range(100):
        a = rng.randint(2, 9)
        op = rng.choice("+-")
        new = cur + a if op == "+" else cur - a
        if LO <= new <= HI and new not in used:
            return op, a, new
    return None


def make_example(k, seed):
    rng = random.Random(seed)
    while True:
        vars_ = rng.sample(VAR_POOL, N_LINES - k)
        target, distractors = vars_[0], vars_[1:]
        used = set()
        # target chain
        v0 = rng.randint(LO + 10, HI - 10)
        used.add(v0)
        B = _box(target)
        t_lines = [Line(target, f"{target} = {v0}", f"{target} = {v0}", f"{target} = {v0}", v0,
                        f"{_cap(B)} starts with {v0} marbles.", f"{_cap(B)} has {v0} marbles.",
                        f"{_cap(B)} has {v0} marbles.")]
        cur, ok = v0, True
        for _ in range(k):
            r = _rand_op(rng, cur, used)
            if r is None:
                ok = False
                break
            op, a, new = r
            used.add(new)
            t_lines.append(Line(target, f"{target} = {target} {op} {a}", f"{target} = {new}",
                                f"{target} = {cur} {op} {a} = {new}", new,
                                f"Add {a} marbles to {B}." if op == "+" else f"Remove {a} marbles from {B}.",
                                f"{_cap(B)} has {new} marbles.",
                                f"{_cap(B)} had {cur}, {_pm(op)} {a} makes {new} marbles."))
            cur = new
        if not ok:
            continue
        # distractors: d = p op a, distinct values
        d_lines = []
        for d in distractors:
            for _ in range(100):
                p, a = rng.randint(LO, HI), rng.randint(2, 9)
                op = rng.choice("+-")
                val = p + a if op == "+" else p - a
                if LO <= val <= HI and val not in used:
                    break
            else:
                ok = False
                break
            used.add(val)
            D = _cap(_box(d))
            d_lines.append(Line(d, f"{d} = {p} {op} {a}", f"{d} = {val}", f"{d} = {p} {op} {a} = {val}", val,
                                f"{D} starts with {p} marbles, then {a} " + ("more are added." if op == "+" else "are removed."),
                                f"{D} has {val} marbles.", f"{D} had {p}, {_pm(op)} {a} makes {val} marbles."))
        if not ok:
            continue
        # interleave: target lines keep relative order, placed at random positions
        slots = sorted(rng.sample(range(N_LINES), k + 1))
        lines, ti, di = [], 0, 0
        for i in range(N_LINES):
            if i in slots:
                lines.append(t_lines[ti]); ti += 1
            else:
                lines.append(d_lines[di]); di += 1
        return Example(k=k, target=target, control=rng.choice(distractors), lines=lines, seed=seed)


def presented_lines(ex, cond, shuffle_seed=0):
    """The CoT lines in the order the probe sees them."""
    if cond == "io":
        return []
    lines = ex.lines[:]
    if cond == "shuf":
        rng = random.Random(10_000 + ex.seed * 7 + shuffle_seed)
        while True:
            rng.shuffle(lines)
            if lines != ex.lines:
                break
    return lines


def make_head(program, query_var, tok, plain=False, style="sym"):
    if style == "nl":
        user = (f"Consider the following sequence of events. They happen in order, from top to bottom.\n\n"
                f"{program}\n\nHow many marbles are in {_box(query_var)} at the end?")
    else:
        user = (f"Consider the following program. Each line is executed in order, from top to bottom.\n\n"
                f"{program}\n\nWhat is the final value of {query_var} after the program finishes?")
    if plain:  # base models: no chat template
        return f"Question: {user}\n\nAnswer: "
    msgs = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def finish(head, cot_lines, query_var, style="sym"):
    if style == "nl":
        body = ("Let me track the boxes step by step.\n" + "\n".join(cot_lines) + "\n") if cot_lines else ""
        return head + body + f"Therefore, the number of marbles in {_box(query_var)} is "
    body = ("Let me trace the program step by step.\n" + "\n".join(cot_lines) + "\n") if cot_lines else ""
    return head + body + f"Therefore, the final value of {query_var} is "


def build_prompt(ex, query_var, cond, fmt, tok, shuffle_seed=0, plain=False, style="sym"):
    """cond in {io, full, shuf}; style in {sym, nl}. Returns the prefix ending right before the answer digits."""
    sfx = "_nl" if style == "nl" else ""
    head = make_head("\n".join(getattr(l, "prog" + sfx) for l in ex.lines), query_var, tok, plain, style)
    cot = [getattr(l, fmt + sfx) for l in presented_lines(ex, cond, shuffle_seed)]
    return finish(head, cot, query_var, style)

