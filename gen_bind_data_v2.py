"""E17 (EXPLORE_PLAN.md, 10-05): E13's binding data with varied question types and answer-format instructions.

Why: E13 learned "Answer: <value>" for every question (all E13 prompts said "Answer with the value only." and asked for a
single value), and kept that format on MemConflict, whose prompt asks for one sentence and whose questions ask what changed.
Same documents, block orders and seeds as E13 (gen_bind_data.make_sample, same rng stream); only the question, the format
instruction and the final answer line change. Extra randomness comes from a separate rng per sample, so the documents stay
identical to E13's.
  question types: current / first / asof / when (as E13) + change_last ("most recent change") + change_all ("how did it change")
  instructions  : value only / one sentence / one sentence with the record's date; train and dev use DIFFERENT phrasings
Target: the same dated list (oldest -> newest) and reasoning line, then "Answer: " + the answer in the requested format.
Decoupled mode only (the chronological control follows only if E17 passes). Mixed in: the same 1000 E12 rows as E13.
usage: python gen_bind_data_v2.py
"""
import collections
import hashlib
import json
import random

import gen_train_data as g
from gen_bind_data import make_sample, TRAIN_GENRES, DEV_GENRES

INSTR = {
    "train": {"value": ["Answer with the value only.", "Give only the value."],
              "sentence": ["Answer in one short sentence.", "Answer in one complete sentence."],
              "dated": ["Answer in one sentence and give the date of the record you rely on."]},
    "dev": {"value": ["Reply with just the value."],
            "sentence": ["Respond in a single full sentence."],
            "dated": ["Write one sentence that also mentions the date of the relevant record."]},
}
QTYPES = ["current", "first", "asof", "when", "change_last", "change_all"]
QW = [0.30, 0.10, 0.15, 0.15, 0.15, 0.15]


def cap(s):
    return s[0].upper() + s[1:]


def rewrite(s, split, rng):
    v = s.pop("_v2")
    a, vals, chain, j, wv, w = v["a"], v["vals"], v["chain"], v["j"], v["wv"], v["when"]
    dates = [d for d, _ in chain]
    qt = rng.choices(QTYPES, QW)[0]
    if qt.startswith("change"):                      # a change has no single value to give
        style = rng.choices(["sentence", "dated"], [0.7, 0.3])[0]
    elif qt == "when":                               # the answer already is a date
        style = rng.choice(["value", "sentence"])
    else:
        style = rng.choices(["value", "sentence", "dated"], [0.4, 0.4, 0.2])[0]
    if qt == "current":
        q, val, line = f"According to these pages, what is {a} at the end of the period they cover?", vals[-1], f"The newest record says {vals[-1]}."
        sent, dated, need = f"At the end of the period these pages cover, {a} is {vals[-1]}.", f"According to the record dated {dates[-1]}, {a} is {vals[-1]}.", [vals[-1]]
    elif qt == "first":
        q, val, line = f"According to these pages, what was {a} when it was first recorded?", vals[0], f"The oldest record says {vals[0]}."
        sent, dated, need = f"When it was first recorded, {a} was {vals[0]}.", f"{cap(a)} was first recorded as {vals[0]}, on {dates[0]}.", [vals[0]]
    elif qt == "asof":
        q, val = f"According to these pages, what was {a} on {w}?", vals[j]
        line = f"The question asks about {w}; the last record before then says {vals[j]}."
        sent, dated, need = f"On {w}, {a} was {vals[j]}.", f"On {w}, {a} was {vals[j]}, as recorded on {dates[j]}.", [vals[j]]
    elif qt == "when":
        q, val = f"According to these pages, on which date was {a} recorded as {vals[wv]}?", dates[wv]
        line = f"The record giving {vals[wv]} is dated {dates[wv]}."
        sent, dated, need = f"{cap(a)} was recorded as {vals[wv]} on {dates[wv]}.", None, [dates[wv]]
    elif qt == "change_last":
        x, y = vals[-2], vals[-1]
        q, val, line = f"According to these pages, what was the most recent change to {a}?", None, f"The last two records say {x} and then {y}."
        sent, dated, need = f"{cap(a)} changed from {x} to {y}.", f"On {dates[-1]}, {a} changed from {x} to {y}.", [x, y]
    else:
        q, val, line = f"According to these pages, how did {a} change over the period they cover?", None, \
            f"The records, in time order, say {', '.join(vals)}."
        sent = f"{cap(a)} changed from {vals[0]} to " + ", then to ".join(vals[1:]) + "."
        dated = f"{cap(a)} changed from {vals[0]} ({dates[0]}) to " + ", then to ".join(f"{x} ({d})" for d, x in chain[1:]) + "."
        need = list(vals)
    final = {"value": val, "sentence": sent, "dated": dated}[style]
    if style == "dated":
        need = need + [d for d in dates if d in final]
    instr = rng.choice(INSTR[split]["value" if style == "value" else style])
    s["prompt"] = f"{v['intro']}\n\n" + "\n\n".join(v["docs"]) + f"\n\n{q} {instr}"
    s["answer"] = "\n".join([f"Records about {a}, from oldest to newest:"] + [f"- {d}: {x}" for d, x in chain] + [line, f"Answer: {final}"])
    s.update(qtype=qt, style=style, instr=instr, final=final, need=need, value=val)
    return s


def main():
    wiki = g.load_wiki()
    mix = [json.loads(l) for l in open("data_train/reason_decoupled_train.jsonl", encoding="utf-8")][:1000]
    for split, n, seed, genres in (("train", 2500, 41, TRAIN_GENRES), ("val", 300, 42, TRAIN_GENRES), ("dev", 300, 43, DEV_GENRES)):
        rng = random.Random(seed)
        rows = []
        for i in range(n):
            s = make_sample(rng, wiki, "decoupled", i, genres)
            rows.append(rewrite(s, "dev" if split == "dev" else "train", random.Random(f"e17-{split}-{i}")))
        # same documents and order as E13 (only the question / instruction / final line differ)
        old = [json.loads(l) for l in open(f"data_train/bind_decoupled_{split}.jsonl", encoding="utf-8")]
        old = {r["id"]: r for r in old if "n_blocks" in r}            # bind rows only (the E12 mix rows reuse ids)
        same = all(r["prompt"].rsplit("\n\n", 1)[0] == old[r["id"]]["prompt"].rsplit("\n\n", 1)[0] for r in rows if r["id"] in old)
        print(split, "documents identical to E13:", same, f"({sum(r['id'] in old for r in rows)} compared)")
        assert same
        out = rows + (mix if split == "train" else [])
        if split == "train":
            random.Random(7).shuffle(out)
        path = f"data_train/bind2_decoupled_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(path, len(out), hashlib.sha256(open(path, "rb").read()).hexdigest()[:16].upper())
        print("  qtype", dict(collections.Counter(r["qtype"] for r in rows)), "| style", dict(collections.Counter(r["style"] for r in rows)),
              "| order", dict(collections.Counter(r["order"] for r in rows)))
        print("  every target contains its needed strings:", all(all(x in r["answer"].rsplit("Answer:", 1)[1] for x in r["need"]) for r in rows))


if __name__ == "__main__":
    main()
