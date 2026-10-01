"""Exp 2: self-correction traces. The program is monotonic (every var assigned once);
the CoT computes the target wrongly k times before the correct value, with different
correction-marker styles. CoT length is fixed at N_LINES (program has N_LINES - k lines).
"""
import random
from dataclasses import dataclass, field

from tasks import HI, LO, N_LINES, VAR_POOL, finish, make_head

CHAIN = 3  # target = c2 op a, c2 = c1 op a, c1 = literal op a
MARKERS = ["none", "back", "fwd", "both"]


@dataclass
class CLine:
    var: str
    value: int
    prog: str = ""        # program text ("" for wrong attempts, which are CoT-only)
    attempt: int = -1     # -1: not a target attempt; else index among target attempts
    wrong: bool = False


@dataclass
class RevExample:
    k: int
    target: str
    gold: int
    wrong_vals: list
    program: list = field(default_factory=list)   # CLine, program order
    cot: list = field(default_factory=list)       # CLine, ordered CoT
    seed: int = 0


def _op(rng, cur, used):
    for _ in range(200):
        a, op = rng.randint(2, 9), rng.choice("+-")
        new = cur + a if op == "+" else cur - a
        if LO <= new <= HI and new not in used:
            return op, a, new
    return None


def make_rev_example(k, seed):
    rng = random.Random(seed)
    while True:
        n_prog = N_LINES - k
        vars_ = rng.sample(VAR_POOL, n_prog)
        chain_vars, distractors = vars_[:CHAIN], vars_[CHAIN:]
        target = chain_vars[-1]
        used, prog_chain = set(), []
        cur = rng.randint(LO + 15, HI - 15)
        ok = True
        for i, v in enumerate(chain_vars):
            r = _op(rng, cur, used)
            if r is None:
                ok = False; break
            op, a, new = r
            src = str(cur) if i == 0 else chain_vars[i - 1]
            prog_chain.append(CLine(v, new, f"{v} = {src} {op} {a}"))
            used.add(new); cur = new
        if not ok:
            continue
        gold = cur
        wrong = []
        while len(wrong) < k:
            w = gold + rng.choice([-1, 1]) * rng.randint(1, 9)
            if LO <= w <= HI and w not in used:
                wrong.append(w); used.add(w)
        prog_d = []
        for d in distractors:
            for _ in range(200):
                p, a, op = rng.randint(LO, HI), rng.randint(2, 9), rng.choice("+-")
                val = p + a if op == "+" else p - a
                if LO <= val <= HI and val not in used:
                    break
            else:
                ok = False; break
            used.add(val)
            prog_d.append(CLine(d, val, f"{d} = {p} {op} {a}"))
        if not ok:
            continue
        # interleave chain (in order) with distractors
        slots = sorted(rng.sample(range(n_prog), CHAIN))
        program, ci, di = [], 0, 0
        for i in range(n_prog):
            if i in slots:
                program.append(prog_chain[ci]); ci += 1
            else:
                program.append(prog_d[di]); di += 1
        # ordered CoT: wrong attempts immediately precede the correct target line
        cot = []
        for l in program:
            if l.var == target:
                for j, w in enumerate(wrong):
                    cot.append(CLine(target, w, attempt=j, wrong=True))
                cot.append(CLine(target, gold, prog=l.prog, attempt=k))
            else:
                cot.append(l)
        return RevExample(k, target, gold, wrong, program, cot, seed)


def render(l, marker):
    s = f"{l.var} = {l.value}"
    if l.attempt < 0 or marker == "none":
        return s
    if marker in ("fwd", "both") and l.attempt >= 1:
        s = f"Wait, actually {s}"
    if marker in ("back", "both") and l.wrong:
        s = f"{s}. Hmm, that's wrong."
    elif marker in ("fwd", "both") and l.attempt >= 1:
        s = s + "."
    return s


def presented(ex, cond, shuffle_seed=0):
    if cond == "io":
        return []
    lines = ex.cot[:]
    if cond == "shuf":
        rng = random.Random(20_000 + ex.seed * 7 + shuffle_seed)
        while True:
            rng.shuffle(lines)
            if lines != ex.cot:
                break
    return lines


def build_rev_prompt(ex, cond, marker, tok, plain=False):
    head = make_head("\n".join(l.prog for l in ex.program), ex.target, tok, plain)
    return finish(head, [render(l, marker) for l in presented(ex, cond)], ex.target)
