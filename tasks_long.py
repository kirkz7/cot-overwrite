"""Exp 8: long overwrite chains (k up to 64) at fixed length N_LONG, to locate where the
reader's recency heuristic (small k) meets the primacy bias reported for many in-context
updates (Qiao et al. 2026; Chattaraj & Raj 2026).

Same Line/Example types as tasks.py; all values are distinct ints in [10, 99].
"""
import random

from tasks import HI, LO, VAR_POOL, Example, Line

N_LONG = 72


def _names(rng, n):
    pool = VAR_POOL + [a + b for a in VAR_POOL for b in "123456789"]
    return rng.sample(pool, n)


def make_long_example(k, seed, n_lines=N_LONG, max_step=30):
    rng = random.Random(seed)
    while True:
        vars_ = _names(rng, n_lines - k)
        target, distractors = vars_[0], vars_[1:]
        v0 = rng.randint(LO + 10, HI - 10)
        used = {v0}
        t_lines = [Line(target, f"{target} = {v0}", f"{target} = {v0}", f"{target} = {v0}", v0)]
        cur, ok = v0, True
        for _ in range(k):
            opts = [v for v in range(max(LO, cur - max_step), min(HI, cur + max_step) + 1) if v not in used]
            if not opts:
                ok = False
                break
            new = rng.choice(opts)
            op, a = ("+", new - cur) if new > cur else ("-", cur - new)
            used.add(new)
            t_lines.append(Line(target, f"{target} = {target} {op} {a}", f"{target} = {new}",
                                f"{target} = {cur} {op} {a} = {new}", new))
            cur = new
        if not ok:
            continue
        d_lines = []
        free = [v for v in range(LO, HI + 1) if v not in used]
        rng.shuffle(free)
        if len(free) < len(distractors):
            continue
        for d, val in zip(distractors, free):
            a, op = rng.randint(2, 9), rng.choice("+-")
            p = val - a if op == "+" else val + a
            d_lines.append(Line(d, f"{d} = {p} {op} {a}", f"{d} = {val}", f"{d} = {p} {op} {a} = {val}", val))
        slots = sorted(rng.sample(range(n_lines), k + 1))
        lines, ti, di = [], 0, 0
        for i in range(n_lines):
            if i in slots:
                lines.append(t_lines[ti]); ti += 1
            else:
                lines.append(d_lines[di]); di += 1
        return Example(k=k, target=target, control=rng.choice(distractors), lines=lines, seed=seed)
