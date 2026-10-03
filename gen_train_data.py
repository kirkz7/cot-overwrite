"""Synthetic training / validation / dev data for the order-vs-time fine-tune (PREREG_TRAINING.md, v2).

Each sample: dated records; one attribute ("target") is recorded k+1 times with different values, other
attributes act as distractors, unrelated filler entries pad the length (25% of samples are 3-6k tokens long).
The question asks for the value at the end of the period (latest date), the first recorded value, or the
value as of a given date.

Formats
  train / val : notes, ticket, csv, ledger, sensor, minutes (block-dated: one date header over many items)
  dev         : calendar, audit (held-out formats, used only to choose hyperparameters; not in train, not in any test)
Never used anywhere here (they belong to tests): chat / email / message boards / git log / changelog / tool-call
traces / CoT / numbered fact lists / time-interval facts / YAML or key: value snapshots.
Wording is symmetric: every record of an attribute draws its sentence from one template pool (assigned in
chronological order, so the chrono control renders identically), and no record or filler uses recency cue words.

--mode decoupled : dated samples are shown in chronological, reverse or shuffled order (1/3 each)
--mode chrono    : same generator and seeds, dated samples always chronological (control); identical except order
usage: python gen_train_data.py --mode decoupled --split train --n 6000
"""
import argparse
import collections
import datetime as dt
import json
import math
import os
import random
import re

TRAIN_FORMATS = ["notes", "ticket", "csv", "ledger", "sensor", "minutes"]
DEV_FORMATS = ["calendar", "audit"]
TABULAR = {"csv", "ledger", "sensor", "audit"}
CUE = re.compile(r"\b(now|updated?|changed?|latest|new|newer|current(ly)?|no longer|switch(ed)?|moved?|recent(ly)?|"
                 r"replaced?|anymore|instead)\b", re.I)

CITIES = ["Lisbon", "Osaka", "Denver", "Tallinn", "Quito", "Perth", "Leeds", "Austin", "Lyon", "Busan", "Cork", "Nagpur"]
NAMES = ["Priya", "Tomas", "Ines", "Kwame", "Mei", "Oskar", "Lena", "Rafael", "Yusuf", "Hana", "Marta", "Dmitri"]
COLORS = ["teal", "amber", "slate", "crimson", "olive", "ivory", "navy", "coral"]
VENDORS = ["Northwind", "Bluepeak", "Corvex", "Halden", "Mirabel", "Quillon", "Sandria", "Tovar"]
ROOMS = ["Room 4B", "Room 12", "Lab C", "Hall 2", "Suite 310", "Annex 7", "Room 21A", "Studio 5"]
PLANS = ["Basic", "Plus", "Team", "Starter", "Flex", "Pro Lite", "Family", "Solo"]
FONTS = ["Inter", "Roboto", "Lato", "Merriweather", "Fira Sans", "Source Serif", "Nunito", "Karla"]
COURIERS = ["SwiftPost", "Parcelo", "RedLine", "Courant", "BoxHop", "Zipline Freight", "Medley", "Orbit Express"]
CADENCE = ["weekly", "biweekly", "monthly", "every three weeks", "quarterly", "twice a month"]
# attribute -> (noun phrase, value sampler); values never encode time order (no versions, no monotone counters)
ATTRS = {
    "office": ("the office city", lambda r: r.choice(CITIES)),
    "owner": ("the project owner", lambda r: r.choice(NAMES)),
    "budget": ("the monthly budget", lambda r: f"${r.randrange(12, 980) * 10}"),
    "team": ("the team size", lambda r: str(r.randint(3, 85))),
    "vendor": ("the packaging vendor", lambda r: r.choice(VENDORS)),
    "room": ("the meeting room", lambda r: r.choice(ROOMS)),
    "color": ("the brand color", lambda r: r.choice(COLORS)),
    "plan": ("the subscription plan", lambda r: r.choice(PLANS)),
    "stock": ("the stock of part X-17", lambda r: str(r.randint(20, 900))),
    "temp": ("the freezer setpoint", lambda r: f"{r.randint(-30, -12)} C"),
    "contact": ("the on-site contact", lambda r: r.choice(NAMES)),
    "price": ("the unit price", lambda r: f"${r.randint(3, 240)}.{r.randint(0, 99):02d}"),
    "limit": ("the daily upload limit", lambda r: f"{r.randint(2, 90)} GB"),
    "route": ("the delivery route", lambda r: f"Route {r.randint(1, 60)}"),
    "shift": ("the night-shift lead", lambda r: r.choice(NAMES)),
    "fee": ("the late fee", lambda r: f"${r.randint(5, 95)}"),
    "servers": ("the number of build servers", lambda r: str(r.randint(2, 64))),
    "font": ("the default font", lambda r: r.choice(FONTS)),
    "courier": ("the courier service", lambda r: r.choice(COURIERS)),
    "aisle": ("the storage aisle for part X-17", lambda r: f"Aisle {r.randint(1, 40)}{r.choice('ABCD')}"),
    "cadence": ("the review cadence", lambda r: r.choice(CADENCE)),
    "ratecap": ("the API rate cap", lambda r: f"{r.randint(1, 50) * 100} requests per minute"),
    "pager": ("the pager rotation lead", lambda r: r.choice(NAMES)),
    "bikes": ("the number of bikes in the shared garage", lambda r: str(r.randint(1, 12))),
    "plants": ("the number of plants in the lobby", lambda r: str(r.randint(2, 40))),
    "desk": ("the hot-desk count on floor 3", lambda r: str(r.randint(4, 60))),
}
TEMPLATES = ["{a} is {v}.", "{a}: {v}.", "Recorded {a} as {v}.", "Confirmed that {a} is {v}.", "Per the team, {a} is {v}.",
             "Checked {a}; it is {v}.", "Noting {a}: {v}.", "{a} = {v}.", "After the review, {a} was set to {v}.",
             "For the record, {a} is {v}.", "We went with {v} for {a}.", "Agreed in the meeting: {a} is {v}.",
             "{v} is what we have for {a}.", "Quick note on {a}: it is {v}.", "The figure for {a} came in at {v}.",
             "Settled on {v} as {a}."]
FILLER_ATTRS = [("the humidity in bay 3", lambda r: f"{r.randint(20, 80)}%"), ("the visitor count", lambda r: str(r.randint(0, 400))),
                ("the printer toner level", lambda r: f"{r.randint(5, 100)}%"), ("the dock 2 queue length", lambda r: str(r.randint(0, 40))),
                ("the badge reader status", lambda r: r.choice(["ok", "offline", "degraded"])),
                ("the parking occupancy", lambda r: f"{r.randint(10, 100)}%"), ("the generator fuel level", lambda r: f"{r.randint(10, 100)}%")]
DATE_FMTS = ["%Y-%m-%d", "%b %d, %Y", "%d %b %Y", "%d %B %Y", "%Y-%m-%d %H:%M", "%a %d %b %Y"]
TIME_FMT = "%Y-%m-%d %H:%M"
NO_DATE_HEADER = "The entries below are listed from newest to oldest."
INTRO = {"notes": "Here are entries from a shared notes app.", "ticket": "Here are entries from an internal ticket tracker.",
         "csv": "Here is an exported records table.", "ledger": "Here are lines from an operations ledger.",
         "sensor": "Here are entries from a facility readings log.", "minutes": "Here are minutes from a series of meetings.",
         "calendar": "Here are entries from a team calendar.", "audit": "Here is an audit table."}


def cap(s):
    return s[0].upper() + s[1:]


def sentence(r):
    s = cap(r["tpl"].format(a=ATTRS[r["attr"]][0], v=r["value"]))
    assert not CUE.search(s), s
    return (r["pre"] + " " + s) if r.get("pre") else s


def clean_wiki(t):
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    t = re.sub(r" ([,.;:!?)'])", r"\1", t)
    return re.sub(r"([(]) ", r"\1", t).replace(" 's", "'s").replace(" n't", "n't")


def wiki_sentences(rng, wiki, words):
    """Consecutive whole sentences from one WikiText paragraph, about `words` words."""
    sents = re.split(r"(?<=[.!?])\s+", rng.choice(wiki))
    i, out = rng.randrange(len(sents)), []
    while i < len(sents) and sum(len(s.split()) for s in out) < words:
        out.append(sents[i]); i += 1
    return " ".join(out)


def filler_record(rng, fmt, wiki):
    if fmt in TABULAR:
        a, f = rng.choice(FILLER_ATTRS)
        return dict(filler_attr=a, filler_value=f(rng))
    return dict(filler=wiki_sentences(rng, wiki, rng.randint(20, 90)))


def body_of(r):
    if "filler_attr" in r:
        return cap(f"{r['filler_attr']}: {r['filler_value']}.")
    return r.get("filler") or sentence(r)


def render(fmt, units, dated, datefmt, header):
    """units: records (or, for minutes, blocks = lists of records) in presentation order. No randomness."""
    d = (lambda r: r["date"].strftime(datefmt)) if dated else (lambda r: None)
    lines = []
    if fmt == "csv":
        lines.append("date,field,value" if dated else "field,value")
    if fmt == "audit":
        lines += (["| date | item | value |", "|---|---|---|"] if dated else ["| item | value |", "|---|---|"])
    for u in units:
        if fmt == "minutes":
            head = f"Minutes - {d(u[0])}" if dated else "Minutes"
            lines.append(head + "\n" + "\n".join("- " + body_of(r) for r in u))
            continue
        r = u
        if fmt in ("csv", "audit"):
            field, val = (r["filler_attr"], r["filler_value"]) if "filler_attr" in r else (ATTRS[r["attr"]][0], r["value"])
            if fmt == "csv":
                lines.append(",".join(([d(r)] if dated else []) + [field, '"' + val + '"']))
            else:
                lines.append("| " + " | ".join(([d(r)] if dated else []) + [field, val]) + " |")
        elif fmt == "notes":
            lines.append((f"[Note | {d(r)}]\n" if dated else "[Note]\n") + body_of(r))
        elif fmt == "ticket":
            lines.append(f"Ticket #{r['tid']}" + (f" | {d(r)}" if dated else "") + f"\n{body_of(r)}")
        elif fmt == "ledger":
            lines.append((f"{d(r)} | " if dated else "") + f"entry {r['tid']} | {body_of(r)}")
        elif fmt == "sensor":
            lines.append(("Reading taken " + d(r) + ": " if dated else "Reading: ") + body_of(r))
        else:   # calendar
            lines.append((f"Calendar entry ({d(r)}): " if dated else "Calendar entry: ") + body_of(r))
    sep = "\n" if fmt in TABULAR else "\n\n"
    return ((NO_DATE_HEADER + "\n\n") if header else "") + sep.join(lines)


def est_tokens(fmt, recs):
    """Rough token count (tables of short lines are denser in tokens per character than prose)."""
    over = {"csv": 30, "ledger": 30, "sensor": 40, "audit": 30, "notes": 30, "ticket": 30, "minutes": 6, "calendar": 35}[fmt]
    return sum(len(body_of(r)) + over for r in recs) / (2.7 if fmt in TABULAR else 3.8)


def make_sample(rng, wiki, mode, idx, formats):
    fmt = rng.choice(formats)
    dated = rng.random() < 0.8
    attrs = rng.sample(sorted(ATTRS), rng.randint(2, 5))
    target, others = attrs[0], attrs[1:]
    k = rng.randint(1, 5)                                       # number of updates of the target
    vals = []
    while len(vals) < k + 1:                                    # distinct values
        v = ATTRS[target][1](rng)
        if v not in vals:
            vals.append(v)
    recs = [dict(attr=target, value=v, target=True) for v in vals]
    for o in others:
        recs += [dict(attr=o, value=ATTRS[o][1](rng)) for _ in range(rng.randint(1, 3))]
    for r in recs:                                              # 30%: the statement sits inside unrelated prose
        r["tpl"] = rng.choice(TEMPLATES)
        r["pre"] = wiki_sentences(rng, wiki, rng.randint(8, 30)) if (fmt not in TABULAR and rng.random() < 0.3) else ""
    # target length in tokens: 25% long (3-6k), else log-uniform 150-2500
    target_tok = rng.uniform(3000, 6000) if rng.random() < 0.25 else math.exp(rng.uniform(math.log(150), math.log(2500)))
    while est_tokens(fmt, recs) < target_tok:
        recs.append(filler_record(rng, fmt, wiki))
    n_fill = sum(1 for r in recs if "filler" in r or "filler_attr" in r)
    # chronological order: the target's values keep their order; other records are interleaved at random
    t_idx = set(rng.sample(range(len(recs)), k + 1))
    rest = [r for r in recs if not r.get("target")]
    rng.shuffle(rest)
    chrono, ti, ri = [], 0, 0
    for i in range(len(recs)):
        if i in t_idx:
            chrono.append(recs[ti]); ti += 1
        else:
            chrono.append(rest[ri]); ri += 1
    t0 = dt.datetime(rng.randint(2015, 2025), rng.randint(1, 12), rng.randint(1, 28), rng.randint(7, 19), rng.randint(0, 59))
    step = rng.choice([dt.timedelta(hours=6), dt.timedelta(days=2), dt.timedelta(days=20), dt.timedelta(days=90)])
    step = step * (12 / max(12, len(chrono)))                    # long samples: denser records, similar total span
    for r in chrono:
        r["date"] = t0
        r["tid"] = rng.randint(1000, 9999)
        t0 = t0 + step * rng.uniform(0.3, 1.7) + dt.timedelta(minutes=rng.randint(1, 59))
    # minutes: consecutive records form blocks (one date per block); no block holds two target records
    blocks = []
    if fmt == "minutes":
        cur, cap_n = [], rng.randint(1, 8)
        for r in chrono:
            if cur and (len(cur) >= cap_n or (r.get("target") and any(x.get("target") for x in cur))):
                blocks.append(cur); cur, cap_n = [], rng.randint(1, 8)
            cur.append(r)
        blocks.append(cur)
        for b in blocks:                                        # the block's date is every record's effective date
            for r in b:
                r["date"] = b[0]["date"]
    datefmt = rng.choice(DATE_FMTS)
    tdates = [r["date"] for r in chrono if r.get("target")]
    shown_t = [x.strftime(datefmt) for x in tdates]
    if (step < dt.timedelta(days=1) or len(set(shown_t)) < len(shown_t)) and "%H" not in datefmt:
        datefmt = TIME_FMT                                      # target records never share a displayed date
    # every random draw happens in both modes, so the chrono control differs from decoupled ONLY in the order
    o_dated = rng.choice(["chrono", "reverse", "shuffle"])
    o_undated = rng.choice(["chrono", "reverse"])                # undated: chronological, or reversed WITH a header
    units = blocks if fmt == "minutes" else chrono
    perm = rng.sample(range(len(units)), len(units))
    header = False
    if dated:
        order = o_dated if mode == "decoupled" else "chrono"
    else:
        order = o_undated
        header = order == "reverse"
    shown = units[:] if order == "chrono" else units[::-1] if order == "reverse" else [units[i] for i in perm]
    tvals = [r for r in chrono if r.get("target")]               # chronological target records
    a = ATTRS[target][0]
    qtype = rng.choices(["current", "first", "asof"], [0.6, 0.2, 0.2])[0] if dated else rng.choices(["current", "first"], [0.75, 0.25])[0]
    j = rng.randrange(len(tvals) - 1) if len(tvals) > 1 else 0
    frac = rng.uniform(0.2, 0.8)
    if qtype == "current":
        q, ans = f"According to these records, what is {a} at the end of the period they cover?", tvals[-1]["value"]
    elif qtype == "first":
        q, ans = f"According to these records, what was {a} when it was first recorded?", tvals[0]["value"]
    else:
        lo, hi = tvals[j]["date"], tvals[j + 1]["date"]
        when = lo + (hi - lo) * frac
        if when.strftime(datefmt) in (lo.strftime(datefmt), hi.strftime(datefmt)):
            datefmt = TIME_FMT                                   # query falls strictly between the two records
        q = f"According to these records, what was {a} on {when.strftime(datefmt)}?"
        ans = tvals[j]["value"]
    text = render(fmt, shown, dated, datefmt, header)
    prompt = f"{INTRO[fmt]}\n\n{text}\n\n{q} Answer with the value only."
    # chain: the target's records in chronological order, as shown (used only by gen_reason_data.py; no random draws)
    chain = [dict(date=r["date"].strftime(datefmt) if dated else None, value=r["value"]) for r in tvals]
    return dict(id=idx, fmt=fmt, dated=dated, order=order, header=header, qtype=qtype, k=k, n_fill=n_fill,
                prompt=prompt, answer=ans, other_values=[r["value"] for r in tvals if r["value"] != ans],
                attr_name=a, chain=chain, query_when=when.strftime(datefmt) if qtype == "asof" else None)


def load_wiki():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train/0000.parquet", repo_type="dataset",
                        revision="refs/convert/parquet")
    paras = [t.strip() for t in pd.read_parquet(p).text if len(t.split()) > 60 and not t.strip().startswith("=")]
    # filler never contains the cue words, nor words that name a test format
    banned = re.compile(r"\b(commits?|assistants?|sessions?|e-?mails?|changelog|log|posts?|messages?|chat)\b", re.I)
    return [clean_wiki(p) for p in paras if not CUE.search(p) and not banned.search(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["decoupled", "chrono"], default="decoupled")
    ap.add_argument("--split", choices=["train", "val", "dev"], default="train")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    wiki = load_wiki()
    seed = {"train": 11, "val": 22, "dev": 33}[args.split]      # same seeds for both modes: identical except order
    formats = DEV_FORMATS if args.split == "dev" else TRAIN_FORMATS
    rng = random.Random(seed)
    rows = [make_sample(rng, wiki, args.mode, i, formats) for i in range(args.n)]
    if args.show:
        for r in rows[:args.show]:
            print("=" * 40, {k: r[k] for k in ("fmt", "dated", "order", "header", "qtype", "k")}, "->", r["answer"])
            print(r["prompt"][:1800])
        return
    os.makedirs("data_train", exist_ok=True)
    out = f"data_train/{args.mode}_{args.split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(out, len(rows), "| fmt", dict(collections.Counter(r["fmt"] for r in rows)), "| order",
          dict(collections.Counter(r["order"] for r in rows)), "| qtype", dict(collections.Counter(r["qtype"] for r in rows)),
          "| median chars", sorted(len(r["prompt"]) for r in rows)[len(rows) // 2])


if __name__ == "__main__":
    main()
