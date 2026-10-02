"""Exploration E5 (EXPLORE_PLAN.md): LoRA training data made only of CoT state-tracking traces.

Same task as Exp 1 / 18 (tasks.py, 16-line programs; the trace sits in the assistant turn), items disjoint from every
test (seeds 6,000,000+ for training and 7,000,000+ for validation; tests use k * 100,000 + i). Half the items use the
symbolic style with "Step i:" tags, half the marble-box style with clock-time tags "[10:07]" (Exp 18's two styles).
  decoupled: 25% chronological untagged, 25% chronological tagged, 25% newest-first tagged, 25% shuffled tagged
  control:   the same items, styles and tags, always chronological
No order headers ("listed from the most recent ...") are used, so the CoT rev_header test stays an untrained cue.
The answer is the queried variable's final value; the loss is on the answer only (train_lora.py).
usage: python gen_cot_train.py
"""
import json
import random

from tasks import _box, make_example
from tasks_cue import HEADER, ordered_lines, tag

CONDS = ["full", "full_step", "rev_step", "shuf_step"]


def render(ex, cond, style):
    sfx = "_nl" if style == "nl" else ""
    program = "\n".join(getattr(l, "prog" + sfx) for l in ex.lines)
    q = ex.target
    if style == "nl":
        user = (f"Consider the following sequence of events. They happen in order, from top to bottom.\n\n"
                f"{program}\n\nHow many marbles are in {_box(q)} at the end?")
        stem = f"Therefore, the number of marbles in {_box(q)} is "
    else:
        user = (f"Consider the following program. Each line is executed in order, from top to bottom.\n\n"
                f"{program}\n\nWhat is the final value of {q} after the program finishes?")
        stem = f"Therefore, the final value of {q} is "
    pairs = ordered_lines(ex, "full" if cond in ("full", "full_step") else cond)
    tagged = cond != "full"
    body = [(tag(s, style) if tagged else "") + getattr(l, "bare" + sfx) for s, l in pairs]
    prefix = HEADER[style] + "\n" + "\n".join(body) + "\n" + stem
    seen = [l.value for _, l in pairs if l.var == q]
    return user, prefix, seen


def make_rows(n, seed0, mode):
    rows = []
    rng = random.Random(seed0)
    for i in range(n):
        k = rng.choice([1, 2, 3, 4, 6, 8])
        style = rng.choice(["sym", "nl"])
        cond = rng.choice(CONDS)
        ex = make_example(k, seed=seed0 + i)
        c = cond if mode == "decoupled" else ("full" if cond == "full" else "full_step")
        user, prefix, seen = render(ex, c, style)
        final = ex.history(ex.target)[-1]
        order = {"full": "chrono", "full_step": "chrono", "rev_step": "reverse", "shuf_step": "shuffle"}[c]
        rows.append(dict(id=i, style=style, k=k, cond=c, dated=c != "full", order=order, qtype="current",
                         prompt=user, prefix=prefix, answer=str(final),
                         answer_pos="last" if seen[-1] == final else "first" if seen[0] == final else "middle"))
    return rows


def main():
    for mode in ("decoupled", "chrono"):
        for split, n, s0 in (("train", 3000, 6_000_000), ("val", 300, 7_000_000)):
            rows = make_rows(n, s0, mode)
            with open(f"data_train/cot_{mode}_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    # checks: the two modes differ only in the order of the trace lines
    d = [json.loads(l) for l in open("data_train/cot_decoupled_train.jsonl", encoding="utf-8")]
    c = [json.loads(l) for l in open("data_train/cot_chrono_train.jsonl", encoding="utf-8")]
    same = all(a["prompt"] == b["prompt"] and a["answer"] == b["answer"] and
               sorted(a["prefix"].splitlines()) == sorted(b["prefix"].splitlines()) for a, b in zip(d, c))
    print("decoupled vs control differ only in line order:", same)
    from collections import Counter
    print("decoupled conds", Counter(r["cond"] for r in d), "| answer position", Counter(r["answer_pos"] for r in d))
    print("control conds", Counter(r["cond"] for r in c), "| answer position", Counter(r["answer_pos"] for r in c))


if __name__ == "__main__":
    main()
