"""E18 (EXPLORE_PLAN.md, 10-06): E17's binding data with the dated list moved into Qwen3's thinking block.

Why (NOTEBOOK 10-05/06): E17 always wrote the dated list in the visible reply, so it ignored "answer with the letter
only" (PersonaMem -14). The E18 diagnostic showed the list IS needed when dates are far from facts (LongMemEval rev:
answer-only E13b 46.2 vs list E13 62.8, -16.7 [-26.9, -6.4]), but not when dates sit next to facts (emails, ConvoMem).
So: thinking on  -> "<think>\\n<dated list + one line>\\n</think>\\n\\n<answer in the requested format>"
    thinking off -> "<answer in the requested format>" only (Qwen3's template already puts an empty think block in the prompt)
Every sample is drawn in one of the two modes (1/2 each). Same documents, block orders and seeds as E13 / E17.
Question kinds (separate rng per sample, so documents stay identical to E13 / E17):
  temporal   0.65  E17's six types and three answer styles (value / sentence / dated)
  choice     0.15  multiple choice over the target's values + an unseen distractor; "letter only"
  single     0.10  an attribute recorded once (no conflict: nothing to order)
  absent     0.10  an attribute the pages never mention (the thought stops at "No record mentions ...")
The thought lists ONLY the queried attribute's records. The E12 rows mixed in (1000, short local-date records) are
converted the same way: their reasoning goes into the thought (thinking on) or is dropped (thinking off).
usage: python gen_bind_data_v3.py
"""
import collections
import copy
import hashlib
import json
import random

import gen_train_data as g
from gen_bind_data import make_sample, TRAIN_GENRES, DEV_GENRES
from gen_bind_data_v2 import INSTR, rewrite, cap

LETTER = {"train": ["Answer with the letter of the correct option only, e.g. (b).", "Reply with the option letter only."],
          "dev": ["Give only the letter of the right option."]}
KINDS, KW = ["temporal", "choice", "single", "absent"], [0.65, 0.15, 0.10, 0.10]


def think_answer(thought, final, think):
    return (f"<think>\n{thought}\n</think>\n\n{final}" if think else final)


def styled(rng, split, value_final, sentence, dated):
    style = rng.choices(["value", "sentence", "dated"], [0.4, 0.4, 0.2])[0] if dated else rng.choice(["value", "sentence"])
    final = {"value": value_final, "sentence": sentence, "dated": dated}[style]
    return style, final, rng.choice(INSTR[split][style])


def build(s, split, rng):
    v = copy.deepcopy(s["_v2"])
    kind = rng.choices(KINDS, KW)[0]
    a, vals, chain = v["a"], v["vals"], v["chain"]
    lst = [f"Records about {a}, from oldest to newest:"] + [f"- {d}: {x}" for d, x in chain]
    head = f"{v['intro']}\n\n" + "\n\n".join(v["docs"]) + "\n\n"
    once = [o for o in v["others"] if sum(1 for p in v["others"] if p[0] == o[0]) == 1]
    if kind == "single" and not once:
        kind = "temporal"
    if kind == "temporal":
        r = rewrite(s, split, rng)                               # E17 question, instruction and final answer
        thought = r["answer"].rsplit("\nAnswer:", 1)[0]
        return dict(r, kind=kind), thought
    s.pop("_v2")
    if kind == "choice":
        qt = rng.choice(["current", "first"])
        gold = vals[-1] if qt == "current" else vals[0]
        opts = list(dict.fromkeys([gold] + [x for x in vals if x != gold][:2]))
        while len(opts) < 4:
            x = g.ATTRS[next(k for k in g.ATTRS if g.ATTRS[k][0] == a)][1](rng)
            if x not in opts:
                opts.append(x)
        rng.shuffle(opts)
        letter = "abcd"[opts.index(gold)]
        q = (f"According to these pages, what is {a} at the end of the period they cover?" if qt == "current"
             else f"According to these pages, what was {a} when it was first recorded?")
        q += " Options:\n" + "\n".join(f"({l}) {o}" for l, o in zip("abcd", opts))
        instr = rng.choice(LETTER[split])
        line = f"The {'newest' if qt == 'current' else 'oldest'} record says {gold}, which is option ({letter})."
        s.update(prompt=head + f"{q}\n{instr}", qtype=f"choice_{qt}", style="letter", instr=instr, final=f"({letter})",
                 need=[f"({letter})"], value=f"({letter})")
        return dict(s, kind=kind), "\n".join(lst + [line])
    if kind == "single":
        a2, x, d = rng.choice(once)
        a2n = g.ATTRS[a2][0]
        style, final, instr = styled(rng, split, x, f"According to these pages, {a2n} is {x}.",
                                     f"According to the record dated {d}, {a2n} is {x}.")
        s.update(prompt=head + f"According to these pages, what is {a2n}? {instr}", qtype="single", style=style,
                 instr=instr, final=final, need=[x] + ([d] if style == "dated" else []), value=x)
        return dict(s, kind=kind), f"Records about {a2n}, from oldest to newest:\n- {d}: {x}\nOnly one record mentions {a2n}."
    a2 = rng.choice(sorted(set(g.ATTRS) - set(v["attrs"])))
    a2n = g.ATTRS[a2][0]
    style, final, instr = styled(rng, split, "Not mentioned", f"These pages do not mention {a2n}.", None)
    s.update(prompt=head + f"According to these pages, what is {a2n}? {instr}", qtype="absent", style=style, instr=instr,
             final=final, need=["not mention"] if style == "sentence" else ["Not mentioned"], value="Not mentioned")
    return dict(s, kind=kind), f"No record mentions {a2n}."


def convert_e12(r, rng):
    think = rng.random() < 0.5
    thought = r["answer"].rsplit("\nAnswer:", 1)[0]
    return dict(r, answer=think_answer(thought, r["final"], think), think=think, kind="e12", style="value",
                need=[r["final"]], value=r["final"])


def main():
    wiki = g.load_wiki()
    mix = [json.loads(l) for l in open("data_train/reason_decoupled_train.jsonl", encoding="utf-8")][:1000]
    mrng = random.Random(1818)
    mix = [convert_e12(r, mrng) for r in mix]
    for split, n, seed, genres in (("train", 2500, 41, TRAIN_GENRES), ("val", 300, 42, TRAIN_GENRES), ("dev", 300, 43, DEV_GENRES)):
        rng = random.Random(seed)
        rows = []
        for i in range(n):
            s = make_sample(rng, wiki, "decoupled", i, genres)
            r2 = random.Random(f"e18-{split}-{i}")
            row, thought = build(s, "dev" if split == "dev" else "train", r2)
            row["think"] = r2.random() < 0.5
            row["answer"] = think_answer(thought, row["final"], row["think"])
            rows.append(row)
        # documents identical to E13 / E17 (only the question, instruction and reply differ)
        old = {r["id"]: r for r in map(json.loads, open(f"data_train/bind_decoupled_{split}.jsonl", encoding="utf-8")) if "n_blocks" in r}
        same = all(r["prompt"].rsplit("\n\n", 1)[0] == old[r["id"]]["prompt"].rsplit("\n\n", 1)[0] for r in rows)
        print(split, "documents identical to E13 / E17:", same)
        assert same
        out = rows + (mix if split == "train" else [])
        if split == "train":
            random.Random(7).shuffle(out)
        path = f"data_train/bind3_decoupled_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(path, len(out), hashlib.sha256(open(path, "rb").read()).hexdigest()[:16].upper())
        print("  kind", dict(collections.Counter(r["kind"] for r in rows)), "| style", dict(collections.Counter(r["style"] for r in rows)),
              "| think", dict(collections.Counter(r["think"] for r in rows)))
        ok = all(all(x in r["final"] for x in r["need"]) for r in rows)
        vis_list = sum("from oldest to newest" in r["answer"].split("</think>")[-1] for r in out)
        print("  every final contains its needed strings:", ok, "| visible replies containing a list:", vis_list)
        assert ok and vis_list == 0


if __name__ == "__main__":
    main()
