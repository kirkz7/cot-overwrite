"""E18.1 (EXPLORE_PLAN.md, 10-06): E18's data + questions that "list the records, take the newest" cannot answer,
+ the base model's own general replies (self-distillation), so that thinking is not hijacked by one template.

Why (NOTEBOOK 10-06 "E18 思考开伤 PersonaMem 的诊断"): with thinking on, 98% of E18's thoughts on PersonaMem started
with the trained record list and 55% ended with "the newest record says ..."; questions about the REASON for an
update or the whole evolution were answered as "latest value" (reasons 83.3 -> 68.2). Long lists were not a general
problem (MemConflict thoughts: median 2 lines, 0% unfinished), so record-compression is NOT added.
Changes vs E18 (same documents, block orders, seeds, thinking on/off 1/2 each, same 1000 E12 rows):
  kinds   temporal 0.55, choice (value options) 0.10, choice_sent (sentence options) 0.08,
          reason 0.12 (NEW: "why was X changed to Y?" - the update sentence gets a "because ..." clause; the thought
          says the question asks for a reason, not a value, and quotes it; no record list), single 0.08, absent 0.07
  general data/selfdistill_train.jsonl (gen_selfdistill.py): Qwen3-4B's own replies to general prompts, thinking on
          and off, mixed in at about 11% (as FILM-7B mixed general data to avoid forgetting)
usage: python gen_bind_data_v4.py      (needs data_train/selfdistill_train.jsonl for the train split)
"""
import collections
import copy
import hashlib
import json
import random
import re

import gen_train_data as g
from gen_bind_data import make_sample, TRAIN_GENRES, DEV_GENRES
from gen_bind_data_v2 import INSTR, cap
from gen_bind_data_v3 import LETTER, build as build_v3, convert_e12, think_answer

KINDS = ["temporal", "choice", "choice_sent", "reason", "single", "absent"]
KW = [0.55, 0.10, 0.08, 0.12, 0.08, 0.07]
REASONS = {
    "train": ["the previous arrangement had become too expensive", "the team had grown since the last review",
              "the old contract ended", "the earlier choice kept causing delays", "management asked for it after the audit",
              "the old option no longer met the safety rules", "a customer complained about the previous setup",
              "the supplier changed its terms", "the budget for the quarter was cut", "the old setting failed an inspection"],
    "dev": ["the earlier option was discontinued by its maker", "the new site manager preferred it",
            "the previous arrangement could not handle the winter workload"],
}
REASON_INSTR = {"train": ["Answer in one short sentence.", "Answer in one complete sentence."],
                "dev": ["Respond in a single full sentence."]}


def add_reason(v, k, reason):
    """Append ', because <reason>' to the sentence recording vals[k] (the block dated chain[k][0]). None if not unique."""
    d, val = v["chain"][k]
    a = v["a"]
    bi = [i for i, b in enumerate(v["docs"]) if b.split("\n", 1)[0].endswith(d) or f" {d}" in b.split("\n", 1)[0]]
    if len(bi) != 1:
        return None
    block = v["docs"][bi[0]]
    hits = [m.start() for m in re.finditer(re.escape(val), block)]
    if len(hits) != 1:
        return None
    p = hits[0]
    near = block[max(0, p - 120): p + len(val) + 120].lower()
    if a.lower() not in near:
        return None
    end = block.find(".", p + len(val))
    while end != -1 and end + 1 < len(block) and block[end + 1] not in " \n":   # skip decimal points ($12.50)
        end = block.find(".", end + 1)
    if end == -1:
        return None
    docs = list(v["docs"])
    docs[bi[0]] = block[:end] + f", because {reason}" + block[end:]
    return docs


def build(s, split, rng):
    sp = "dev" if split == "dev" else "train"
    kind = rng.choices(KINDS, KW)[0]
    v = copy.deepcopy(s["_v2"])
    a, vals, chain = v["a"], v["vals"], v["chain"]
    head = lambda docs: f"{v['intro']}\n\n" + "\n\n".join(docs) + "\n\n"
    if kind == "reason":
        k = rng.randrange(1, len(vals))
        reason = rng.choice(REASONS[sp])
        docs = add_reason(v, k, reason)
        if docs is None:
            kind = "temporal"
        else:
            s.pop("_v2")
            instr = rng.choice(REASON_INSTR[sp])
            final = f"It was changed because {reason}."
            s.update(prompt=head(docs) + f"According to these pages, why was {a} changed to {vals[k]}? {instr}",
                     qtype="reason", style="sentence", instr=instr, final=final, need=[reason], value=reason, kind=kind)
            thought = (f"The question asks for the reason behind a change, not for the current value.\n"
                       f"The record from {chain[k][0]} that changed {a} to {vals[k]} says it was because {reason}.")
            return s, thought
    if kind == "choice_sent":
        s.pop("_v2")
        gold = vals[-1]
        opts = list(dict.fromkeys([gold] + vals[:-1][-2:]))
        key = next(x for x in g.ATTRS if g.ATTRS[x][0] == a)
        while len(opts) < 4:
            x = g.ATTRS[key][1](rng)
            if x not in opts:
                opts.append(x)
        rng.shuffle(opts)
        letter = "abcd"[opts.index(gold)]
        stmts = [f"({l}) At the end of the period these pages cover, {a} is {o}." for l, o in zip("abcd", opts)]
        instr = rng.choice(LETTER[sp])
        s.update(prompt=head(v["docs"]) + "Which statement is correct according to these pages?\n" + "\n".join(stmts) + f"\n{instr}",
                 qtype="choice_sent", style="letter", instr=instr, final=f"({letter})", need=[f"({letter})"], value=f"({letter})",
                 kind=kind)
        lst = [f"Records about {a}, from oldest to newest:"] + [f"- {d}: {x}" for d, x in chain]
        return s, "\n".join(lst + [f"The newest record says {gold}, which is statement ({letter})."])
    # temporal / choice / single / absent (and a reason item whose sentence could not be found): E18's construction
    return build_v3(s, sp, rng, kind=kind)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--general", default="data_train/selfdistill_train.jsonl")
    ap.add_argument("--prefix", default="data_train/bind4_decoupled")
    args = ap.parse_args()
    wiki = g.load_wiki()
    mix = [json.loads(l) for l in open("data_train/reason_decoupled_train.jsonl", encoding="utf-8")][:1000]
    mrng = random.Random(1818)
    mix = [convert_e12(r, mrng) for r in mix]                          # identical to E18's conversion
    general = [json.loads(l) for l in open(args.general, encoding="utf-8")]
    for split, n, seed, genres in (("train", 2500, 41, TRAIN_GENRES), ("val", 300, 42, TRAIN_GENRES), ("dev", 300, 43, DEV_GENRES)):
        rng = random.Random(seed)
        rows = []
        for i in range(n):
            s = make_sample(rng, wiki, "decoupled", i, genres)
            r2 = random.Random(f"e181-{split}-{i}")
            row, thought = build(s, split, r2)
            row.pop("_v2", None)
            row["think"] = r2.random() < 0.5
            row["answer"] = think_answer(thought, row["final"], row["think"])
            rows.append(row)
        out = rows + (mix + general if split == "train" else [])
        if split == "train":
            random.Random(7).shuffle(out)
        path = f"{args.prefix}_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(path, len(out), hashlib.sha256(open(path, "rb").read()).hexdigest()[:16].upper())
        print("  kind", dict(collections.Counter(r["kind"] for r in rows)), "| think", dict(collections.Counter(r["think"] for r in rows)),
              "| general rows", len(general) if split == "train" else 0, f"({len(general) / len(out) * 100:.1f}% of train)" if split == "train" else "")
        ok = all(all(x in r["final"] for x in r["need"]) for r in rows)
        vis = sum("from oldest to newest" in r["answer"].split("</think>")[-1] for r in rows + mix)
        print("  every final contains its needed strings:", ok, "| visible replies with a list (bind + E12 rows):", vis)
        assert ok and vis == 0


if __name__ == "__main__":
    main()
