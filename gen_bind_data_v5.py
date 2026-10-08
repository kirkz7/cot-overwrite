"""E18.2 (CLOUD_NOTEBOOK.md "E18.2", user 10-08): E18.1's data + questions about an attribute recorded MANY times,
whose thought stays short (no full record list), + a light unlikelihood penalty on the model's own looping thoughts.

Why (CLOUD_NOTEBOOK 10-08 "8B 循环诊断"): E18.1-8B's unfinished PersonaMem thoughts are record lists (61% list lines,
runs of 7 / 39 lines median / p90) that turn into repeated lines; every E18.1 target lists 2-5 records, so a question
with many relevant records sends the 8B model outside what it was trained on. E18.1-4B and base 8B do not loop.
New kind "many" (this file): the target attribute is recorded 7-14 times (values may come back, never twice in a row),
short blocks. The thought never lists all records:
  current / previous / choice_sent : the last three records (oldest -> newest) + one line
  first                            : the first three records + one line
  asof                             : the records just before and after the date + one line
  history                          : one line with the values in time order (no dates, no list)
stages:
  many      -> data_train/many_{train,val,dev,harvest}.jsonl (separate seeds; dev uses the held-out genres)
  assemble  -> <prefix>_{train,val,dev}.jsonl = E18.1's files (unchanged rows) + many rows + unlikelihood rows
usage: python gen_bind_data_v5.py many
       python gen_bind_data_v5.py assemble --base data_train/bind4q8_decoupled --ul data_train/ul_loops8b.jsonl --prefix data_train/bind5q8_decoupled
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import random

import gen_train_data as g
from gen_bind_data import GENRES, INTRO, TRAIN_GENRES, DEV_GENRES, DATE_FMTS
from gen_bind_data_v3 import LETTER, styled, think_answer

QTYPES = ["current", "previous", "first", "asof", "history", "choice_sent"]
QW = [0.30, 0.15, 0.10, 0.15, 0.15, 0.15]
N_MANY = {"train": 600, "val": 100, "dev": 200, "harvest": 1000}
SEEDS = {"train": 51, "val": 52, "dev": 53, "harvest": 54}


def make_many(rng, wiki, idx, genres):
    genre = rng.choice(genres)
    attrs = rng.sample(sorted(g.ATTRS), rng.randint(2, 4))
    target, others = attrs[0], attrs[1:]
    n = rng.randint(7, 14)
    vals = []
    while len(vals) < n:
        v = g.ATTRS[target][1](rng)
        if not vals or v != vals[-1]:
            vals.append(v)
    n_blocks = n + rng.randint(0, 3)
    t_blocks = sorted(rng.sample(range(n_blocks), n))
    stmts = [[] for _ in range(n_blocks)]
    for v, b in zip(vals, t_blocks):
        stmts[b].append(dict(attr=target, value=v, tpl=rng.choice(g.TEMPLATES), target=True))
    for o in others:
        for _ in range(rng.randint(1, 3)):
            stmts[rng.randrange(n_blocks)].append(dict(attr=o, value=g.ATTRS[o][1](rng), tpl=rng.choice(g.TEMPLATES)))
    long_doc = rng.random() < 0.25
    bodies = []
    for b in range(n_blocks):
        words = rng.randint(150, 400) if long_doc else rng.randint(60, 200)
        prose = []
        while sum(len(p.split()) for p in prose) < words:
            prose.append(g.wiki_sentences(rng, wiki, words - sum(len(p.split()) for p in prose)))
        sents = [s for s in " ".join(prose).split(". ") if s]
        for st in stmts[b]:
            sents.insert(rng.randint(0, len(sents)), g.sentence(st).rstrip("."))
        bodies.append(". ".join(sents).rstrip(".") + ".")
    t0 = dt.date(rng.randint(2012, 2025), rng.randint(1, 12), rng.randint(1, 28))
    dates = []
    for b in range(n_blocks):
        dates.append(t0)
        t0 = t0 + dt.timedelta(days=rng.randint(3, 60))
    fmt = rng.choice(DATE_FMTS)
    order = rng.choice(["chrono", "reverse", "shuffle"])
    perm = rng.sample(range(n_blocks), n_blocks)
    shown = list(range(n_blocks)) if order == "chrono" else list(range(n_blocks))[::-1] if order == "reverse" else perm
    blocks = [GENRES[genre].format(d=dates[b].strftime(fmt)) + "\n" + bodies[b] for b in shown]
    chain = [(dates[b].strftime(fmt), v) for v, b in zip(vals, t_blocks)]
    j = rng.randrange(n - 1)
    lo, hi = dates[t_blocks[j]], dates[t_blocks[j + 1]]
    when = (lo + dt.timedelta(days=rng.randint(1, max(1, (hi - lo).days - 1)))).strftime(fmt)
    return dict(id=f"many-{idx}", fmt=genre, dated=True, order=order, header=False, k=n - 1, n_blocks=n_blocks, long=long_doc,
                _m=dict(a=g.ATTRS[target][0], key=target, vals=vals, chain=chain, j=j, when=when,
                        head=f"{INTRO[genre]}\n\n" + "\n\n".join(blocks) + "\n\n"))


def build_many(s, split, rng):
    sp = "dev" if split == "dev" else "train"
    m = s.pop("_m")
    a, vals, chain, j, w = m["a"], m["vals"], m["chain"], m["j"], m["when"]
    qt = rng.choices(QTYPES, QW)[0]
    span = f"{g.cap(a)} is recorded many times, from {chain[0][0]} to {chain[-1][0]}."
    last3 = [f"The last three records, from oldest to newest:"] + [f"- {d}: {x}" for d, x in chain[-3:]]
    if qt in ("current", "previous"):
        gold = vals[-1] if qt == "current" else vals[-2]
        q = (f"According to these pages, what is {a} at the end of the period they cover?" if qt == "current"
             else f"According to these pages, what was {a} just before its most recent change?")
        line = f"The newest record says {gold}." if qt == "current" else f"The record before the newest one says {gold}."
        d_gold = chain[-1][0] if qt == "current" else chain[-2][0]
        sent = (f"At the end of the period these pages cover, {a} is {gold}." if qt == "current"
                else f"Just before its most recent change, {a} was {gold}.")
        style, final, instr = styled(rng, sp, gold, sent, f"According to the record dated {d_gold}, {a} was {gold}.")
        thought = [span, "Only the newest records matter here."] + last3 + [line]
        need = [gold] + ([d_gold] if style == "dated" else [])
    elif qt == "first":
        gold = vals[0]
        q = f"According to these pages, what was {a} when it was first recorded?"
        style, final, instr = styled(rng, sp, gold, f"When it was first recorded, {a} was {gold}.",
                                     f"{g.cap(a)} was first recorded as {gold}, on {chain[0][0]}.")
        thought = [span, "Only the oldest records matter here.", "The first three records, from oldest to newest:"] + \
                  [f"- {d}: {x}" for d, x in chain[:3]] + [f"The oldest record says {gold}."]
        need = [gold] + ([chain[0][0]] if style == "dated" else [])
    elif qt == "asof":
        gold = vals[j]
        q = f"According to these pages, what was {a} on {w}?"
        style, final, instr = styled(rng, sp, gold, f"On {w}, {a} was {gold}.", f"On {w}, {a} was {gold}, as recorded on {chain[j][0]}.")
        thought = [span, f"Only the records around {w} matter here.", f"The records just before and after {w}:",
                   f"- {chain[j][0]}: {vals[j]}", f"- {chain[j + 1][0]}: {vals[j + 1]}",
                   f"The last record before {w} says {gold}."]
        need = [gold] + ([chain[j][0]] if style == "dated" else [])
    elif qt == "history":
        q = f"According to these pages, how did {a} change over time?"
        style = "sentence"
        instr = rng.choice(["Answer in one sentence, in time order.", "List the values in one sentence, oldest first."]
                           if sp == "train" else ["Give the values in a single sentence, from oldest to newest."])
        final = f"{g.cap(a)} went from {vals[0]} to " + ", then to ".join(vals[1:]) + "."
        thought = [span, "The question asks for the whole history, so I give the values in time order in one line.",
                   "In time order: " + ", ".join(vals) + "."]
        gold, need = vals[-1], list(vals)
    else:   # choice_sent: statements about the current value (as E18.1)
        gold = vals[-1]
        opts = list(dict.fromkeys([gold] + [x for x in vals[::-1] if x != gold][:2]))
        while len(opts) < 4:
            x = g.ATTRS[m["key"]][1](rng)
            if x not in opts:
                opts.append(x)
        rng.shuffle(opts)
        letter = "abcd"[opts.index(gold)]
        q = "Which statement is correct according to these pages?\n" + "\n".join(
            f"({l}) At the end of the period these pages cover, {a} is {o}." for l, o in zip("abcd", opts))
        instr = rng.choice(LETTER[sp])
        style, final, need = "letter", f"({letter})", [f"({letter})"]
        thought = [span, "Only the newest records matter here."] + last3 + [f"The newest record says {gold}, which is statement ({letter})."]
    sep = "\n" if qt == "choice_sent" else " "
    s.update(prompt=m["head"] + f"{q}{sep}{instr}", qtype=f"many_{qt}", style=style, instr=instr, final=final, need=need,
             value=gold, kind="many")
    return s, "\n".join(thought)


def stage_many(args):
    wiki = g.load_wiki()
    for split in ("train", "val", "dev", "harvest"):
        rng = random.Random(SEEDS[split])
        rows = []
        for i in range(N_MANY[split]):
            s = make_many(rng, wiki, f"{split}-{i}", DEV_GENRES if split == "dev" else TRAIN_GENRES)
            r2 = random.Random(f"e182-{split}-{i}")
            row, thought = build_many(s, split, r2)
            # harvest prompts: thinking on (they are run through the E18.1 model to collect its own loops)
            row["think"] = True if split in ("harvest", "dev") else r2.random() < 0.5
            row["answer"] = think_answer(thought, row["final"], row["think"])
            rows.append(row)
        ok = all(all(x in r["final"] for x in r["need"]) for r in rows)
        maxl = max(len(r["answer"].split("</think>")[0].splitlines()) for r in rows)
        path = f"data_train/many_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(path, len(rows), hashlib.sha256(open(path, "rb").read()).hexdigest()[:16].upper(),
              "| qtype", dict(collections.Counter(r["qtype"] for r in rows)), "| think", sum(r["think"] for r in rows),
              "| records", dict(sorted(collections.Counter(r["k"] + 1 for r in rows).items())),
              "| finals ok", ok, "| max thought lines", maxl)
        assert ok


def stage_assemble(args):
    ul = [json.loads(l) for l in open(args.ul, encoding="utf-8")] if args.ul else []
    ul = ul[:args.ul_max]
    if args.ul and len(ul) < args.ul_min:   # stop rule (CLOUD_NOTEBOOK "E18.2"): no training data is written
        raise SystemExit(f"only {len(ul)} looping thoughts (< {args.ul_min}): not assembling E18.2, report first")
    for split in ("train", "val", "dev"):
        base = [json.loads(l) for l in open(f"{args.base}_{split}.jsonl", encoding="utf-8")]
        many = [json.loads(l) for l in open(f"data_train/many_{split}.jsonl", encoding="utf-8")]
        out = base + many + (ul if split == "train" else [])
        if split == "train":
            random.Random(7).shuffle(out)
        path = f"{args.prefix}_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(path, len(out), hashlib.sha256(open(path, "rb").read()).hexdigest()[:16].upper(),
              f"| E18.1 rows {len(base)} | many {len(many)} | unlikelihood {len(ul) if split == 'train' else 0}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["many", "assemble"])
    ap.add_argument("--base", default="data_train/bind4q8_decoupled")
    ap.add_argument("--ul", default="")
    ap.add_argument("--ul_max", type=int, default=300)
    ap.add_argument("--ul_min", type=int, default=50)
    ap.add_argument("--prefix", default="data_train/bind5q8_decoupled")
    args = ap.parse_args()
    {"many": stage_many, "assemble": stage_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
