"""Exploration E13 (EXPLORE_PLAN.md): train the binding of facts to time when the date is far from the fact.

Long block-dated documents: each block starts with ONE date line and holds 150-1500 tokens of unrelated prose with the
statements buried at random depths, so a statement's date is the header of the block it sits in. Genres (no chat, no
email, no git / changelog: those are the test formats):
  train / val : diary, minutes (long), lab notebook, status report, field log
  dev         : letters, ship's log (held-out genres, only to sanity-check training; not in any test)
Question types: the value at the end (current), the first value, the value as of a date between two records (asof), and
"on which date was <attribute> recorded as <value>?" (when) - the last one trains fact -> date retrieval directly.
Target: the E12 time-organized reasoning (dated list oldest -> newest, one line, Answer).
--mode decoupled: blocks shown chronological / reverse / shuffled (1/3 each); --mode chrono: same draws, always
chronological (control). Every random draw happens in both modes, so the two differ ONLY in block order.
Mixed in (both modes): 1000 rows of the E12 reasoning data (local per-record dates), to keep that skill.
usage: python gen_bind_data.py
"""
import collections
import datetime as dt
import hashlib
import json
import random

import gen_train_data as g

GENRES = {"diary": "Diary entry - {d}", "minutes": "Minutes of the meeting held on {d}", "lab": "Lab notebook - {d}",
          "report": "Status report - {d}", "field": "Field log, {d}",
          "letters": "Letter dated {d}\nDear team,", "ship": "Ship's log - {d}"}
INTRO = {"diary": "Here are pages from a project diary.", "minutes": "Here are the minutes of a series of meetings.",
         "lab": "Here are pages from a lab notebook.", "report": "Here are several status reports.",
         "field": "Here are entries from a field log.", "letters": "Here are letters sent to the team.",
         "ship": "Here are entries from a ship's log."}
TRAIN_GENRES = ["diary", "minutes", "lab", "report", "field"]
DEV_GENRES = ["letters", "ship"]
DATE_FMTS = ["%Y-%m-%d", "%b %d, %Y", "%d %b %Y", "%d %B %Y", "%A, %d %B %Y"]   # date only; blocks are >= 2 days apart


def make_sample(rng, wiki, mode, idx, genres):
    genre = rng.choice(genres)
    attrs = rng.sample(sorted(g.ATTRS), rng.randint(2, 4))
    target, others = attrs[0], attrs[1:]
    k = rng.randint(1, 4)
    vals = []
    while len(vals) < k + 1:
        v = g.ATTRS[target][1](rng)
        if v not in vals:
            vals.append(v)
    n_blocks = k + 1 + rng.randint(0, 3)
    t_blocks = sorted(rng.sample(range(n_blocks), k + 1))           # each target record in its own block, in time order
    tpl = rng.sample(g.TEMPLATES, k + 1)
    stmts = [[] for _ in range(n_blocks)]
    for v, b, tp in zip(vals, t_blocks, tpl):
        stmts[b].append(dict(attr=target, value=v, tpl=tp, target=True))
    for o in others:
        for _ in range(rng.randint(1, 2)):
            stmts[rng.randrange(n_blocks)].append(dict(attr=o, value=g.ATTRS[o][1](rng), tpl=rng.choice(g.TEMPLATES)))
    # block bodies: unrelated prose with the statements at random depths
    long_doc = rng.random() < 0.5
    bodies = []
    for b in range(n_blocks):
        words = rng.randint(400, 1100) if long_doc else rng.randint(110, 400)   # ~150-1500 tokens per block
        prose = []
        while sum(len(p.split()) for p in prose) < words:            # one WikiText paragraph is often shorter than this
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
    if mode == "chrono":
        order = "chrono"
    idx_shown = list(range(n_blocks)) if order == "chrono" else list(range(n_blocks))[::-1] if order == "reverse" else perm
    blocks = [GENRES[genre].format(d=dates[b].strftime(fmt)) + "\n" + bodies[b] for b in idx_shown]
    # question
    a = g.ATTRS[target][0]
    chain = [(dates[b].strftime(fmt), v) for v, b in zip(vals, t_blocks)]
    qtype = rng.choices(["current", "first", "asof", "when"], [0.45, 0.15, 0.2, 0.2])[0]
    j = rng.randrange(k)
    lo, hi = dates[t_blocks[j]], dates[t_blocks[j + 1]]
    when = lo + dt.timedelta(days=rng.randint(1, max(1, (hi - lo).days - 1)))
    wv = rng.randrange(k + 1)
    if qtype == "current":
        q, ans, line = f"According to these pages, what is {a} at the end of the period they cover?", vals[-1], f"The newest record says {vals[-1]}."
    elif qtype == "first":
        q, ans, line = f"According to these pages, what was {a} when it was first recorded?", vals[0], f"The oldest record says {vals[0]}."
    elif qtype == "asof":
        w = when.strftime(fmt)
        q, ans = f"According to these pages, what was {a} on {w}?", vals[j]
        line = f"The question asks about {w}; the last record before then says {vals[j]}."
    else:
        q, ans = f"According to these pages, on which date was {a} recorded as {vals[wv]}?", chain[wv][0]
        line = f"The record giving {vals[wv]} is dated {chain[wv][0]}."
    prompt = f"{INTRO[genre]}\n\n" + "\n\n".join(blocks) + f"\n\n{q} Answer with the value only."
    reasoning = "\n".join([f"Records about {a}, from oldest to newest:"] + [f"- {d}: {v}" for d, v in chain] + [line, f"Answer: {ans}"])
    return dict(id=idx, fmt=genre, dated=True, order=order, header=False, qtype=qtype, k=k, n_blocks=n_blocks,
                long=long_doc, prompt=prompt, answer=reasoning, final=ans,
                _v2=dict(a=a, vals=vals, chain=chain, j=j, wv=wv, when=when.strftime(fmt), docs=blocks, intro=INTRO[genre]))


def main():
    wiki = g.load_wiki()
    mix = {m: [json.loads(l) for l in open(f"data_train/reason_{m}_train.jsonl", encoding="utf-8")][:1000]
           for m in ("decoupled", "chrono")}
    for split, n, seed, genres in (("train", 2500, 41, TRAIN_GENRES), ("val", 300, 42, TRAIN_GENRES), ("dev", 300, 43, DEV_GENRES)):
        rows = {}
        for mode in ("decoupled", "chrono"):
            rng = random.Random(seed)
            rows[mode] = [make_sample(rng, wiki, mode, i, genres) for i in range(n)]
        same = all(sorted(a["prompt"].split("\n\n")) == sorted(b["prompt"].split("\n\n")) and a["answer"] == b["answer"]
                   for a, b in zip(rows["decoupled"], rows["chrono"]))
        print(split, "decoupled vs control: same blocks and targets, only order differs:", same)
        assert same
        for mode in ("decoupled", "chrono"):
            out = rows[mode] + (mix[mode] if split == "train" else [])
            if split == "train":
                random.Random(7).shuffle(out)
            path = f"data_train/bind_{mode}_{split}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in out:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "_v2"}) + "\n")   # _v2: for gen_bind_data_v2.py
            print(path, len(out), hashlib.sha256(open(path, "rb").read()).hexdigest()[:16].upper())
        r = rows["decoupled"]
        print("  genre", dict(collections.Counter(x["fmt"] for x in r)), "| order", dict(collections.Counter(x["order"] for x in r)),
              "| qtype", dict(collections.Counter(x["qtype"] for x in r)), "| median chars", sorted(len(x["prompt"]) for x in r)[len(r) // 2])
        bad = [x for x in r if g.CUE.search(x["prompt"].split("\n\n", 1)[1].rsplit("\n\n", 1)[0])]
        print("  prompts containing a recency cue word:", len(bad))


if __name__ == "__main__":
    main()
